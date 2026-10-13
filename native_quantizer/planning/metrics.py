"""Weight and activation error measurements used by the planner."""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from native_quantizer.calibration import CalibrationStats
from native_quantizer.codecs.packing import unpack_int4_signed
from native_quantizer.codecs.rotation import build_hadamard, rotate_activation
from native_quantizer.codecs.w4a8 import decode_w4a8_runtime_weight
from native_quantizer.codecs.w6a8 import decode_w6a8_runtime_weight
from native_quantizer.errors import PolicyError
from native_quantizer.identity import (
    FORMAT_INT8,
    FORMAT_W4A4,
    FORMAT_W4A8,
    FORMAT_W6A8,
    INT8_SCALE_MAX,
    W4A4_EMISSION_MAX,
)


@dataclass
class TensorMetrics:
    name: str
    rel_l2: float
    snr_db: float
    cosine: float
    act_rel_l2: Optional[float] = None      # activation-aware (when calibration exists)
    kept: bool = False
    reason: str = ""


def compute_weight_metrics(original: torch.Tensor, dequant: torch.Tensor) -> TensorMetrics:
    o = original.float()
    d = dequant.float()
    signal, error, reconstructed, dot = torch.stack((
        o.square().sum(), (d - o).square().sum(),
        d.square().sum(), (o * d).sum())).tolist()
    if not all(math.isfinite(value)
               for value in (signal, error, reconstructed, dot)):
        return TensorMetrics(name="", rel_l2=1e30, snr_db=-300.0, cosine=-1.0)
    rel_l2 = math.sqrt(error / max(signal, 1e-12))
    snr = 300.0 if error <= 1e-30 else 10.0 * math.log10(
        max(signal, 1e-30) / error)
    if signal <= 1e-30 and reconstructed <= 1e-30:
        cos = 1.0
    else:
        cosine_denom = math.sqrt(max(signal * reconstructed, 1e-30))
        cos = max(-1.0, min(1.0, dot / cosine_denom))
    return TensorMetrics(name="", rel_l2=rel_l2, snr_db=snr, cosine=cos)


def activation_aware_error(original: torch.Tensor, dequant: torch.Tensor,
                           activations: Optional[torch.Tensor]) -> Optional[float]:
    """mean over samples of ||(Q(W)-W) x|| / ||W x|| using recorded activations."""
    if activations is None:
        return None
    o = original.float()
    d = dequant.float()
    delta = d - o
    x = activations.float()
    num = (delta @ x.t()).norm(dim=0)
    den = (o @ x.t()).norm(dim=0).clamp(min=1e-8)
    value = float((num / den).mean())
    return value if math.isfinite(value) else 1e30


def _act_quant_int8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dynamic rowwise int8 activation quantization (runtime behavior)."""
    abs_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    scale = abs_max / float(INT8_SCALE_MAX)
    q = (x / scale).round().clamp_(-128.0, 127.0).to(torch.int8)
    return q, scale


def _act_quant_int4(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dynamic rowwise signed int4 activation quantization (eager W4A4 path)."""
    abs_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scale = abs_max / float(W4A4_EMISSION_MAX)
    q = (x / scale).round().clamp_(-float(W4A4_EMISSION_MAX),
                                   float(W4A4_EMISSION_MAX)).to(torch.int8)
    return q, scale


def _simulate_quantized_chunk(original_chunk: torch.Tensor,
                              tensors: Dict[str, torch.Tensor], fmt: str,
                              group_size: int, convrot_groupsize: int,
                              act_q: torch.Tensor, act_scale: torch.Tensor,
                              linear_dtype: str) -> torch.Tensor:
    """Exact eager-path emulation of the quantized linear for one weight chunk.

    Returns y_q [S, rows] for the chunk. The activation was already rotated
    and quantized once (act_q, act_scale). The emulations mirror the
    comfy-kitchen eager kernels:
      int8_tensorwise: (x_int8 @ w_int8.T) * x_scale * w_scale
      convrot_w4a4:    (x_int4/8 @ w_int4.T) * x_scale * w_scale
      asym_w4a8_int8:  (x_int8 * x_scale) @ dequant(W).T
    """
    if fmt == FORMAT_INT8:
        q, wscale = tensors[""], tensors["_scale"]
        return ((act_q.float() @ q.float().T)
                * act_scale * wscale.float().reshape(1, -1))
    if fmt == FORMAT_W4A4:
        w_int = unpack_int4_signed(tensors[""]).float()
        wscale = tensors["_scale"].float().reshape(1, -1)
        return ((act_q.float() @ w_int.T)
                * act_scale * wscale)
    if fmt == FORMAT_W6A8:
        w_int8 = decode_w6a8_runtime_weight(tensors[""], tensors["_s_rel"], group_size)
        return (act_q.float() @ w_int8.float().T) * act_scale * tensors["_s_channel"].float()[None, :]
    if fmt == FORMAT_W4A8:
        # the runtime GEMM operates in the ConvRot basis: int8 activation
        # rows against the ROTATED int8 weight, scaled by the activation and
        # channel scales. Do NOT use the inverse-rotated physical weight here.
        w_int8 = decode_w4a8_runtime_weight(
            tensors[""], tensors["_s_rel"], tensors.get("_codebook"),
            group_size)
        s_channel = tensors["_s_channel"].float().reshape(1, -1)
        return (act_q.float() @ w_int8.float().T) * act_scale * s_channel
    raise PolicyError(f"unknown format {fmt!r} for runtime simulation")


def runtime_output_rel_l2(original: torch.Tensor,
                          tensors: Dict[str, torch.Tensor], fmt: str,
                          group_size: int, convrot_groupsize: int,
                          activations: torch.Tensor,
                          w4a4_activation_bits: int = 8) -> Optional[float]:
    """mean over samples of ||Y_quant - Y_bf16|| / ||Y_bf16|| for the actual
    runtime operation (activation rotation, activation quantization, quantized
    GEMM with scales), not a reconstructed-weight approximation.

    w4a4_activation_bits is the EFFECTIVE W4A4 activation precision on the
    target runtime (4 on eager regardless of linear_dtype, 4 or 8 on CUDA)."""
    x = activations.float()
    y_ref = x @ original.float().T
    if fmt in (FORMAT_W4A4, FORMAT_W4A8, FORMAT_W6A8):
        h = build_hadamard(convrot_groupsize, device=x.device, dtype=torch.float32)
        x_rot = rotate_activation(x, h, convrot_groupsize)
    else:
        x_rot = x
    if fmt == FORMAT_W4A4 and w4a4_activation_bits == 4:
        act_q, act_scale = _act_quant_int4(x_rot)
    else:
        act_q, act_scale = _act_quant_int8(x_rot)
    y_q = _simulate_quantized_chunk(original, tensors, fmt, group_size,
                                    convrot_groupsize, act_q, act_scale,
                                    "int4" if w4a4_activation_bits == 4
                                    else "int8")
    num = (y_q - y_ref).norm(dim=1)
    den = y_ref.norm(dim=1).clamp(min=1e-8)
    value = float((num / den).mean())
    return value if math.isfinite(value) else 1e30


class _OutputErrorAccumulator:
    """Chunked accumulation of the per-sample runtime output error."""

    def __init__(self, n_samples: int):
        self.num_sq = torch.zeros(n_samples, dtype=torch.float64)
        self.den_sq = torch.zeros(n_samples, dtype=torch.float64)

    def update(self, y_q: torch.Tensor, y_ref: torch.Tensor) -> None:
        d = (y_q.double() - y_ref.double())
        self.num_sq += d.square().sum(dim=1)
        self.den_sq += y_ref.double().square().sum(dim=1)

    def finish(self) -> float:
        ratio = (self.num_sq.sqrt() / self.den_sq.clamp(min=1e-16).sqrt())
        value = float(ratio.mean())
        return value if math.isfinite(value) else 1e30


class SensitivityAnalyzer:
    """Decides keep-precision for candidate layers based on metrics + thresholds."""

    def __init__(self, threshold: Optional[float], error_threshold: float,
                 calibration: Optional[CalibrationStats]):
        self.threshold = threshold
        self.error_threshold = error_threshold
        self.calibration = calibration
        self.results: Dict[str, TensorMetrics] = {}

    def evaluate(self, name: str, original: torch.Tensor, dequant: torch.Tensor,
                 ) -> TensorMetrics:
        m = compute_weight_metrics(original, dequant)
        m.name = name
        if self.calibration is not None:
            acts = None
            for key in (name, name[:-len(".weight")]):
                if key in self.calibration.layers:
                    acts = self.calibration.layers[key]["samples"]
                    break
            m.act_rel_l2 = activation_aware_error(original, dequant, acts)
        self.results[name] = m
        return m

    def decide_keep(self, m: TensorMetrics) -> Tuple[bool, str]:
        reasons = []
        if m.rel_l2 > self.error_threshold:
            reasons.append(f"relL2 {m.rel_l2:.4f} > error-threshold {self.error_threshold}")
        if self.threshold is not None:
            score = m.act_rel_l2 if (m.act_rel_l2 is not None and self.calibration is not None) else m.rel_l2
            if score > self.threshold:
                reasons.append(f"sensitivity {score:.4f} > threshold {self.threshold}")
        return (bool(reasons), "; ".join(reasons))


@dataclass
class _MetricAccumulator:
    name: str
    signal: float = 0.0
    error: float = 0.0
    dot: float = 0.0
    reconstructed: float = 0.0
    act_num_sq: Optional[torch.Tensor] = None
    act_den_sq: Optional[torch.Tensor] = None

    def update(self, original: torch.Tensor, dequant: torch.Tensor,
               activations: Optional[torch.Tensor]) -> None:
        original = original.float()
        dequant = dequant.float()
        delta = dequant - original
        self.signal += float(original.square().sum())
        self.error += float(delta.square().sum())
        self.dot += float((original * dequant).sum())
        self.reconstructed += float(dequant.square().sum())
        if activations is not None:
            x = activations.float()
            num = delta @ x.t()
            den = original @ x.t()
            part_num = num.square().sum(dim=0).cpu()
            part_den = den.square().sum(dim=0).cpu()
            self.act_num_sq = part_num if self.act_num_sq is None else self.act_num_sq + part_num
            self.act_den_sq = part_den if self.act_den_sq is None else self.act_den_sq + part_den

    def finish(self) -> TensorMetrics:
        if not all(math.isfinite(value) for value in (
                self.signal, self.error, self.dot, self.reconstructed)):
            return TensorMetrics(
                self.name, 1e30, -300.0, -1.0,
                act_rel_l2=1e30 if self.act_num_sq is not None else None)
        rel_l2 = math.sqrt(self.error / max(self.signal, 1e-12))
        snr_db = 300.0 if self.error <= 1e-30 else 10.0 * math.log10(
            max(self.signal, 1e-30) / self.error)
        if self.signal <= 1e-30 and self.reconstructed <= 1e-30:
            cosine = 1.0
        else:
            denom = math.sqrt(max(self.signal * self.reconstructed, 1e-30))
            cosine = max(-1.0, min(1.0, self.dot / denom))
        act_rel_l2 = None
        if self.act_num_sq is not None and self.act_den_sq is not None:
            act_rel_l2 = float(
                (self.act_num_sq.sqrt() / self.act_den_sq.clamp(min=1e-16).sqrt()).mean())
            if not math.isfinite(act_rel_l2):
                act_rel_l2 = 1e30
        return TensorMetrics(self.name, rel_l2, snr_db, cosine,
                             act_rel_l2=act_rel_l2)
