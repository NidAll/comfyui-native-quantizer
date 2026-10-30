"""Mixed-precision candidate evaluation and quality/storage/runtime selection."""

from __future__ import annotations

import dataclasses
import math
import torch
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from native_quantizer.architectures.policies import TEXT_TOKEN_EMBEDDING
from native_quantizer.calibration import CalibrationStats
from native_quantizer.checkpoint_io.reader import CheckpointReader
from native_quantizer.checkpoint_io.types import CheckpointInfo, TensorMeta
from native_quantizer.codecs.bounded import (
    _chunk_rows_for_budget,
    _codebook_sample_size,
    _gather_codebook_samples,
    _quant_work_bytes,
    _quantize_row_chunk,
)
from native_quantizer.codecs.dispatch import dequantize_weight_by_format, quantize_weight_by_format
from native_quantizer.codecs.rotation import build_hadamard, rotate_activation
from native_quantizer.codecs.w4a4 import _pick_w4a4_convrot_group
from native_quantizer.dtypes import FLOAT_DTYPES, FP8_DTYPES
from native_quantizer.errors import CompressionGateError, InputError, PolicyError, QualityGateError
from native_quantizer.formats import quantized_format_bytes
from native_quantizer.identity import (
    CODEBOOK_MODES,
    COMPONENT_TEXT_ENCODER,
    DEFAULT_W4A4_LINEAR_DTYPE,
    FORMAT_INT8,
    FORMAT_ORIGINAL,
    FORMAT_W4A4,
    FORMAT_W4A8,
    FORMAT_W6A8,
    MIXED_FORMATS,
    W4A4_QUANT_GROUP_SIZE,
    W4A8_ALS_ITERS,
    W4A8_CONVROT_GROUPSIZE,
)
from native_quantizer.planning.metrics import (
    TensorMetrics,
    _MetricAccumulator,
    _OutputErrorAccumulator,
    _act_quant_int4,
    _act_quant_int8,
    _simulate_quantized_chunk,
    compute_weight_metrics,
    runtime_output_rel_l2,
)
from native_quantizer.planning.types import DecisionKind, TensorDecision
from native_quantizer.runtime.capabilities import (
    RuntimeCapabilities,
    W4A4ExecutionMode,
    resolve_w4a4_execution_mode,
    runtime_capabilities_for,
)
from native_quantizer.runtime.costs import kernel_contract
from native_quantizer.utils import human_bytes
from native_quantizer.optimization import OptimizationConfig


@dataclass(frozen=True)
class MixedProfile:
    name: str
    layer_gate: float            # per-layer error gate (relL2 fraction)
    global_gate: float           # mean error gate over selected layers
    compression_target_bpp: float  # bytes/parameter target for quantized linear bytes
    max_bf16_fraction: float     # max bf16 share of targeted linear bytes


MIXED_PROFILE_DEFAULTS = {
    # Gate defaults are set against the measured W4A8 reference: weight-only
    # relL2 of the codebook path is ~0.073 on real dims, so a balanced profile
    # keeps W4A8 as the workhorse (global gate just above 0.073), conservative
    # forces promotion toward INT8 (gate below it), and size-first admits the
    # ~0.14 W4A4 error for layers that can take it (per-layer gate 0.15).
    "throughput": MixedProfile("throughput", 0.10, 0.080, 1.10, 0.05),
    "balanced": MixedProfile("balanced", 0.10, 0.080, 0.90, 0.05),
    "conservative": MixedProfile("conservative", 0.05, 0.040, 1.05, 0.02),
    "size-first": MixedProfile("size-first", 0.15, 0.100, 0.75, 0.10),
    "adaptive": MixedProfile("adaptive", 0.08, 0.040, 0.90, 0.15),
}


@dataclass
class CandidateResult:
    format: str
    eligible: bool
    reason: str = ""
    estimated_bytes: int = 0
    weight_rel_l2: Optional[float] = None
    act_rel_l2: Optional[float] = None
    # reproducibility record for the W4A4 execution variant
    requested_linear_dtype: Optional[str] = None
    effective_activation_bits: Optional[int] = None
    effective_runtime_path: Optional[str] = None
    runtime_certified: bool = False
    runtime_certain: bool = False
    backend: Optional[str] = None


class MixedPlanner:
    """Evaluate per-format candidates, select cheapest acceptable per layer,
    then promote greedily (best error reduction per extra byte, with original
    precision as the final rescue) until the weighted global error gate
    passes. Quality and compression gates are HARD: a plan that cannot meet
    them raises QualityGateError / CompressionGateError instead of silently
    publishing a checkpoint that misses its targets."""

    def __init__(self, profile_name: str,
                 calibration: Optional[CalibrationStats],
                 max_mem: int, device: torch.device,
                 compute_dtype: Optional[torch.dtype],
                 disabled_formats: Sequence[str] = (),
                 layer_gate: Optional[float] = None,
                 global_gate: Optional[float] = None,
                 runtime: Optional[RuntimeCapabilities] = None,
                 compression_target_bpp: Optional[float] = None,
                 max_bf16_fraction: Optional[float] = None,
                 linear_dtype: str = DEFAULT_W4A4_LINEAR_DTYPE,
                 runtime_certificate: Optional["RuntimeCertificate"] = None,
                 seed: int = 0,
                 codebook_mode: str = "auto",
                 objective: str = "kitchen",
                 kernel_costs: Optional[Dict] = None,
                 enable_w6a8: bool = False, w6a8_group_size: int = 16,
                 optimization=None, workloads=(), target_bytes=None, optimization_calibration=None,
                 enable_int8_convrot=False, quality_reference='none'):
        self.optimization = optimization or OptimizationConfig()
        if objective not in ("kitchen", "size"):
            raise PolicyError(f"unknown mixed objective {objective!r}")
        self.enable_w6a8 = enable_w6a8
        self.enable_int8_convrot = enable_int8_convrot or profile_name == 'adaptive'
        self.w6a8_group_size = w6a8_group_size
        if w6a8_group_size <= 0 or w6a8_group_size % 16:
            raise PolicyError("W6A8 group size must be a positive multiple of 16")
        self.objective = objective
        self.kernel_costs = kernel_costs or {}
        self.workloads = workloads
        self.target_bytes = target_bytes
        self.quality_reference = quality_reference
        self.reference_error = None
        self.reference_fallbacks = {}
        self._fixed_bytes = 0
        if profile_name not in MIXED_PROFILE_DEFAULTS:
            raise PolicyError(f"unknown mixed profile {profile_name!r}")
        self.profile = MIXED_PROFILE_DEFAULTS[profile_name]
        self.calibration = calibration
        self.optimization_calibration = optimization_calibration or calibration
        self.max_mem = max_mem
        self.device = device
        self.compute_dtype = compute_dtype
        self.disabled = set(disabled_formats)
        self.runtime = runtime if runtime is not None \
            else runtime_capabilities_for("cpu")
        self.linear_dtype = linear_dtype
        self.runtime_certificate = runtime_certificate
        self.seed = seed
        if codebook_mode not in CODEBOOK_MODES:
            raise PolicyError(
                f"unknown W4A8 codebook mode {codebook_mode!r}; "
                f"expected one of {CODEBOOK_MODES}")
        self.codebook_mode = codebook_mode
        self.layer_gate = (layer_gate if layer_gate is not None
                           else self.profile.layer_gate)
        self.global_gate = (global_gate if global_gate is not None
                            else self.profile.global_gate)
        self._configured_global_gate = self.global_gate
        self.compression_target_bpp = (
            compression_target_bpp if compression_target_bpp is not None
            else self.profile.compression_target_bpp)
        self.max_bf16_fraction = (
            max_bf16_fraction if max_bf16_fraction is not None
            else self.profile.max_bf16_fraction)
        self.candidates: Dict[str, Dict[str, CandidateResult]] = {}
        self.promotions: List[str] = []
        self.summary: Dict[str, Any] = {}
        self._targeted: List[TensorDecision] = []
        self.encoded_cache = {}
        self.codebooks = {}
        self._cache_bytes = 0

    # -- eligibility -------------------------------------------------------
    def eligible_formats(self, meta: TensorMeta, token_embedding: bool = False) -> List[str]:
        if len(meta.shape) != 2 or meta.dtype not in FLOAT_DTYPES:
            return []
        k = int(meta.shape[1])
        out: List[str] = []
        if token_embedding:
            return ([FORMAT_INT8] if self.runtime.supports(FORMAT_INT8)
                    and FORMAT_INT8 not in self.disabled else [])
        if self.enable_w6a8 and k % 256 == 0 and k % self.w6a8_group_size == 0 and self.runtime.supports(FORMAT_W6A8):
            out.append(FORMAT_W6A8)
        if k % 256 == 0 and self.runtime.supports(FORMAT_W4A8):
            out.append(FORMAT_W4A8)
        if k % W4A4_QUANT_GROUP_SIZE == 0 and self.runtime.supports(FORMAT_W4A4):
            try:
                _pick_w4a4_convrot_group(k)
                out.append(FORMAT_W4A4)
            except PolicyError:
                pass
        if self.runtime.supports(FORMAT_INT8):
            out.append(FORMAT_INT8)
            if self.enable_int8_convrot and k % 256 == 0 and FORMAT_INT8 not in self.disabled:
                out.append('int8-convrot')
        return [f for f in out if f not in self.disabled]

    def runtime_reason(self, fmt: str) -> str:
        fmt = FORMAT_INT8 if fmt == 'int8-convrot' else fmt
        if not self.runtime.supports(fmt):
            return f"not supported on target runtime {self.runtime.target}"
        return f"{self.runtime.describe(fmt)} on {self.runtime.target}"

    def cgs_for(self, k: int, fmt: str) -> int:
        if fmt == FORMAT_W4A4:
            return _pick_w4a4_convrot_group(k)
        return W4A8_CONVROT_GROUPSIZE

    # -- evaluation --------------------------------------------------------
    def _w4a4_scoring_modes(self) -> Tuple[List[int], W4A4ExecutionMode]:
        """Activation-bit modes to score a W4A4 candidate with. Certain
        modes evaluate one value; uncertain modes evaluate BOTH and the
        caller takes the worst (max) error, so the planner never
        optimistically assumes a dispatch it cannot prove."""
        mode = self.w4a4_execution_mode_for_candidate()
        if mode.certain:
            return [mode.activation_bits], mode
        return [4, 8], mode

    def w4a4_execution_mode_for_candidate(self) -> W4A4ExecutionMode:
        """Effective W4A4 execution mode for candidate evaluation: a runtime
        certificate (observed on the target machine) wins over the static
        dispatch model. Uncertain static modes are handled by the caller
        through worst-case evaluation of both A4 and A8."""
        if (self.runtime_certificate is not None
                and self.runtime_certificate.formats.get(FORMAT_W4A4)):
            conf = self.runtime_certificate.formats[FORMAT_W4A4]
            bits = (conf.get("effective_activation_bits")
                    or conf.get("activation_bits"))
            if bits in (4, 8):
                return W4A4ExecutionMode(
                    int(bits), "certified", True,
                    "observed on the target machine "
                    f"({self.runtime_certificate.gpu or 'gpu'})")
        return resolve_w4a4_execution_mode(self.runtime, self.linear_dtype)

    def _activations(self, name: str) -> Optional[torch.Tensor]:
        if self.calibration is None:
            return None
        for key in (name, name[:-len(".weight")]):
            stats = self.calibration.layers.get(key)
            if stats is not None:
                return stats["samples"]
        return None

    def _encoding_activations(self, name):
        if self.optimization_calibration is not None:
            for key in (name, name.removesuffix('.weight')):
                if key in self.optimization_calibration.layers:
                    return self.optimization_calibration.layers[key]['samples']
        return None

    def _evaluate_format(self, reader: CheckpointReader, d: TensorDecision,
                         fmt: str) -> TensorMetrics:
        if fmt == FORMAT_W6A8:
            d = dataclasses.replace(d, group_size=self.w6a8_group_size)
        meta = reader.info.by_name(d.name)
        if meta is None:
            raise PolicyError(f"planner references missing tensor {d.name!r}")
        cgs = self.cgs_for(int(meta.shape[1]), fmt)
        activations = self._activations(d.name)
        encoding_activations = self._encoding_activations(d.name)
        reduction = 'p95' if self.profile.name == 'adaptive' else 'mean'
        work_bytes = _quant_work_bytes(meta, fmt, self.optimization)
        if work_bytes + self._cache_bytes > self.max_mem:
            self.encoded_cache.clear()
            self._cache_bytes = 0
        if _quant_work_bytes(meta, fmt, self.optimization) <= self.max_mem:
            w = reader.read_tensor(d.name)
            w = w.to(device=self.device, dtype=(self.compute_dtype
                     if self.compute_dtype is not None and w.dtype not in FP8_DTYPES else w.dtype))
            try:
                q = quantize_weight_by_format(
                    w, fmt, d.group_size, cgs, seed=self.seed,
                    codebook_mode=self.codebook_mode, optimization=self.optimization,
                    activations=encoding_activations)
                dq = dequantize_weight_by_format(q, fmt, d.group_size, cgs,
                                                 torch.float32)
                metrics = compute_weight_metrics(w, dq)
                if activations is not None:
                    activations = activations.to(device=w.device, dtype=torch.float32)
                    if fmt == FORMAT_W4A4:
                        modes, _mode = self._w4a4_scoring_modes()
                        errs = [
                            runtime_output_rel_l2(
                                w, q, fmt,
                                d.group_size, cgs, activations,
                                w4a4_activation_bits=bits, reduction=reduction)
                            for bits in modes
                        ]
                        metrics.act_rel_l2 = max(errs)  # worst case
                    else:
                        metrics.act_rel_l2 = runtime_output_rel_l2(
                            w, q, fmt,
                            d.group_size, cgs, activations,
                            w4a4_activation_bits=8, reduction=reduction)
                self._cache_encoding(d.name, fmt, q, max(0, self.max_mem - work_bytes))
            finally:
                del w
            return metrics
        # bounded-memory chunked path: rowwise formats chunk exactly; w4a8
        # uses the pre-fit codebook row chunks, identical to the reference
        # path. Runtime output error accumulates per sample across chunks.
        n, k = int(meta.shape[0]), int(meta.shape[1])
        chunk_rows = _chunk_rows_for_budget(k, n, self.max_mem, fmt, self.optimization)
        acc = _MetricAccumulator(d.name)
        out_acc = None
        out_accs = None
        act_q = act_scale = None
        act_qs = act_scales = None
        if activations is not None:
            x = activations.float()
            if fmt in (FORMAT_W4A4, FORMAT_W4A8, FORMAT_W6A8, 'int8-convrot'):
                h = build_hadamard(cgs, device=x.device, dtype=torch.float32)
                x_rot = rotate_activation(x, h, cgs)
            else:
                x_rot = x
            n_samples = int(activations.shape[0])
            if fmt == FORMAT_W4A4:
                modes, _mode = self._w4a4_scoring_modes()
                out_accs = {bits: _OutputErrorAccumulator(n_samples)
                            for bits in modes}
                act_qs = {}
                act_scales = {}
                for bits in modes:
                    if bits == 4:
                        act_qs[bits], act_scales[bits] = _act_quant_int4(x_rot)
                    else:
                        act_qs[bits], act_scales[bits] = _act_quant_int8(x_rot)
            else:
                act_q, act_scale = _act_quant_int8(x_rot)
                out_acc = _OutputErrorAccumulator(n_samples)
        if fmt == FORMAT_W4A8:
            sample_size = _codebook_sample_size(n * k)
            codebook = _gather_codebook_samples(
                reader, d.name, k, d.group_size, cgs, sample_size, chunk_rows,
                compute_dtype=self.compute_dtype, seed=self.seed,
                codebook_mode=self.codebook_mode)
            self._remember_codebook(d.name, codebook)
            for r0 in range(0, n, chunk_rows):
                r1 = min(n, r0 + chunk_rows)
                q = _quantize_row_chunk(reader, d.name, r0, r1, d.group_size,
                                        cgs, codebook, self.device,
                                        self.compute_dtype,
                                        als_iterations=(
                                            3 if self.codebook_mode == "fit"
                                            else W4A8_ALS_ITERS), optimization=self.optimization,
                                        activations=encoding_activations)
                dq = dequantize_weight_by_format(q, FORMAT_W4A8, d.group_size,
                                                 cgs, torch.float32)
                orig_chunk = reader.read_tensor(d.name)[r0:r1]
                # Runtime output error is accumulated separately below; avoid
                # duplicating its two large matrix multiplications here.
                acc.update(orig_chunk, dq, None)
                if out_acc is not None:
                    y_q = _simulate_quantized_chunk(
                        orig_chunk, q, fmt, d.group_size, cgs, act_q,
                        act_scale, "int8")
                    y_ref = activations.float() @ orig_chunk.float().T
                    out_acc.update(y_q.cpu(), y_ref.cpu())
                del q, dq
        else:
            for r0 in range(0, n, chunk_rows):
                r1 = min(n, r0 + chunk_rows)
                chunk = reader.read_tensor(d.name)[r0:r1]
                if self.compute_dtype is not None and chunk.dtype not in FP8_DTYPES:
                    chunk = chunk.to(self.compute_dtype)
                if self.device.type == "cuda":
                    chunk = chunk.to(self.device)
                q = quantize_weight_by_format(
                    chunk, fmt, d.group_size, cgs, seed=self.seed,
                    codebook_mode=self.codebook_mode, optimization=self.optimization,
                    activations=encoding_activations)
                dq = dequantize_weight_by_format(q, fmt, d.group_size, cgs,
                                                 torch.float32)
                chunk_cpu = chunk.cpu()
                q_cpu = {key: value.cpu() for key, value in q.items()} if activations is not None else None
                acc.update(chunk_cpu.float(), dq.cpu(), None)
                if out_acc is not None:
                    y_q = _simulate_quantized_chunk(
                        chunk_cpu, q_cpu, fmt,
                        d.group_size, cgs, act_q, act_scale, "int8")
                    y_ref = activations.float() @ chunk_cpu.float().T
                    out_acc.update(y_q, y_ref)
                if out_accs is not None:
                    y_ref = activations.float() @ chunk_cpu.float().T
                    for bits, oacc in out_accs.items():
                        y_q = _simulate_quantized_chunk(
                            chunk_cpu, q_cpu,
                            fmt, d.group_size, cgs, act_qs[bits],
                            act_scales[bits],
                            "int4" if bits == 4 else "int8")
                        oacc.update(y_q, y_ref)
                del chunk, chunk_cpu, q, q_cpu, dq
        metrics = acc.finish()
        if out_acc is not None:
            metrics.act_rel_l2 = out_acc.finish(reduction)
        if out_accs is not None:
            metrics.act_rel_l2 = max(oacc.finish(reduction) for oacc in out_accs.values())
        return metrics

    def _cache_encoding(self, name, fmt, tensors, available):
        if self.profile.name != 'adaptive':
            return
        if fmt == FORMAT_W4A8:
            self._remember_codebook(name, tensors['_codebook'])
        size = sum(t.numel() * t.element_size() for t in tensors.values())
        codebook_bytes = sum(t.numel() * t.element_size() for t in self.codebooks.values())
        limit = min(self.max_mem // 8, available) - codebook_bytes
        if size > limit:
            return
        while self.encoded_cache and self._cache_bytes + size > limit:
            oldest = next(iter(self.encoded_cache))
            record = self.encoded_cache.pop(oldest)
            self._cache_bytes -= record[1]
        self.encoded_cache[(name, fmt)] = ({k: v.detach().cpu().clone() for k, v in tensors.items()}, size)
        self._cache_bytes += size

    def _remember_codebook(self, name, codebook):
        if self.profile.name == 'adaptive':
            if len(self.codebooks) >= max(1, self.max_mem // 4096):
                self.codebooks.pop(next(iter(self.codebooks)))
            self.codebooks[name] = codebook.detach().cpu().clone()

    def selected_encodings(self):
        return {d.name: self.encoded_cache[(d.name, d.encoding_format)][0]
                for d in self._targeted if d.kind == DecisionKind.QUANTIZE
                and (d.name, d.encoding_format) in self.encoded_cache}

    # -- planning ----------------------------------------------------------
    def plan(self, info: CheckpointInfo,
             decisions: List[TensorDecision]) -> Dict[str, Any]:
        self.candidates.clear()
        self.promotions.clear()
        self.encoded_cache.clear()
        self.codebooks.clear()
        self._cache_bytes = 0
        self.exchanges = []
        self._exchange_error_before = self._exchange_error_after = None
        self.global_gate = self._configured_global_gate
        self.reference_error = None
        self.reference_fallbacks = {}
        self._targeted = [d for d in decisions
                          if d.kind == DecisionKind.QUANTIZE]
        targeted = {d.name for d in self._targeted}
        self._fixed_bytes = sum(t.nbytes for t in info.tensors if t.name not in targeted)
        with CheckpointReader(info) as reader:
            for d in self._targeted:
                meta = info.by_name(d.name)
                if meta is None:
                    raise PolicyError(
                        f"mixed planner references missing tensor {d.name!r}")
                n, k = int(meta.shape[0]), int(meta.shape[1])
                cands: Dict[str, CandidateResult] = {}
                w4a4_mode = self.w4a4_execution_mode_for_candidate()
                for fmt in self.eligible_formats(
                        meta, d.component == COMPONENT_TEXT_ENCODER
                        and bool(TEXT_TOKEN_EMBEDDING.search(d.name))):
                    mode = (w4a4_mode if fmt == FORMAT_W4A4 else None)
                    cand = CandidateResult(
                        format=fmt, eligible=True,
                        estimated_bytes=quantized_format_bytes(n, k, fmt, self.w6a8_group_size if fmt == FORMAT_W6A8 else 16),
                        reason=self.runtime_reason(fmt),
                        requested_linear_dtype=(
                            self.linear_dtype if fmt == FORMAT_W4A4 else None),
                        effective_activation_bits=(
                            mode.activation_bits if mode is not None and
                            mode.certain else None),
                        effective_runtime_path=(
                            mode.path if mode is not None else None),
                        runtime_certified=(
                            self.runtime.capability(fmt).certified
                            if fmt in MIXED_FORMATS else False),
                        runtime_certain=(
                            mode.certain if mode is not None else None),
                        backend=(self.runtime.target if fmt == FORMAT_W4A4
                                 else None))
                    try:
                        metrics = self._evaluate_format(reader, d, fmt)
                    except PolicyError as exc:
                        if "NaN or Inf" in str(exc) or "non-finite" in str(exc):
                            raise InputError(
                                f"non-finite values in quantizable layer "
                                f"{d.name!r} (NaN/Inf); pass "
                                "--nonfinite-policy keep to preserve the layer "
                                "at original precision") from exc
                        raise
                    cand.weight_rel_l2 = metrics.rel_l2
                    cand.act_rel_l2 = metrics.act_rel_l2
                    cands[fmt] = cand
                # record every eligible format even when the runtime cannot
                # run it, so the report explains why it was skipped
                for fmt in (FORMAT_W4A4, FORMAT_W4A8, FORMAT_INT8) + ((FORMAT_W6A8,) if self.enable_w6a8 else ()):
                    if fmt in cands:
                        continue
                    if fmt in self.disabled:
                        cands[fmt] = CandidateResult(
                            format=fmt, eligible=False,
                            reason="disabled by CLI flag",
                            estimated_bytes=quantized_format_bytes(n, k, fmt, self.w6a8_group_size if fmt == FORMAT_W6A8 else 16))
                    elif not self.runtime.supports(fmt):
                        cands[fmt] = CandidateResult(
                            format=fmt, eligible=False,
                            reason=self.runtime_reason(fmt),
                            estimated_bytes=quantized_format_bytes(n, k, fmt, self.w6a8_group_size if fmt == FORMAT_W6A8 else 16))
                    else:
                        cands[fmt] = CandidateResult(
                            format=fmt, eligible=False,
                            reason="shape ineligible",
                            estimated_bytes=quantized_format_bytes(n, k, fmt, self.w6a8_group_size if fmt == FORMAT_W6A8 else 16))
                # original precision is a real candidate: zero error, source
                # bytes. Selection only picks it when every quantized format
                # fails the layer gate; promotion may also upgrade to it.
                cands[FORMAT_ORIGINAL] = CandidateResult(
                    format=FORMAT_ORIGINAL, eligible=True,
                    reason="original precision (source bytes, zero error)",
                    estimated_bytes=meta.nbytes,
                    weight_rel_l2=0.0, act_rel_l2=0.0)
                self.candidates[d.name] = cands

        if self.quality_reference != 'none':
            reference = FORMAT_W6A8 if self.quality_reference == 'w6a8' else FORMAT_INT8
            numerator, denominator = 0., 0
            for d in self._targeted:
                candidates = self.candidates[d.name]
                choice = next((candidates[f] for f in (reference, FORMAT_INT8, FORMAT_ORIGINAL)
                               if f in candidates and candidates[f].eligible), candidates[FORMAT_ORIGINAL])
                if choice.format != reference:
                    self.reference_fallbacks[d.name] = choice.format
                params = self._layer_params(info, d)
                numerator += self._error_of(choice) * params
                denominator += params
            self.reference_error = numerator / max(denominator, 1)
            self.global_gate = min(self.global_gate, self.reference_error)
        self.select(info, decisions)
        promoted = self.promote(info, decisions) if self.global_gate is not None else []
        try:
            self._enforce_gates(info, decisions)
        except (QualityGateError, CompressionGateError):
            if not self.repair_feasibility(info):
                raise
            self._enforce_gates(info, decisions)
        self.optimize_execution(info)
        if self.profile.name == 'adaptive' and self.objective != 'size':
            self.optimize_exchanges(info)
        self._enforce_gates(info, decisions)
        self.summary = self._summarize(info, decisions)
        self.summary.update({
            "selected": sum(d.kind == DecisionKind.QUANTIZE for d in self._targeted),
            "kept": sum(d.kind != DecisionKind.QUANTIZE for d in self._targeted), "promotions": promoted,
            "runtime_backend": self.runtime.target,
            "objective": self.objective,
            "execution_optimizations": self.execution_optimizations,
            "w4a4_linear_dtype": self.linear_dtype,
            "w4a8_codebook_mode": self.codebook_mode,
        })
        return self.summary

    def execution_contract(self, d: TensorDecision, fmt: str):
        mode = self.w4a4_execution_mode_for_candidate()
        native = FORMAT_INT8 if fmt == 'int8-convrot' else fmt
        contract = kernel_contract(
            native, backend=self.runtime.target,
            accelerated=self.runtime.capability(native).accelerated,
            activation_bits=mode.activation_bits if fmt == FORMAT_W4A4 else 8,
            certain=mode.certain,
            embedding=d.component == COMPONENT_TEXT_ENCODER
            and bool(TEXT_TOKEN_EMBEDDING.search(d.name)))
        return dataclasses.replace(contract, activation_transform=True) if fmt == 'int8-convrot' else contract

    def measured_cost(self, shape, fmt, activation_bits, layer=None):
        group = self.w6a8_group_size if fmt == FORMAT_W6A8 else 16 if fmt == FORMAT_W4A8 else None
        samples = [w for w in self.workloads if (w['n'], w['k']) == tuple(shape)]
        if any(len(key) == 7 for key in self.kernel_costs):
            specific = [w for w in samples if w.get('layer') == layer and layer is not None]
            samples = specific or [w for w in samples if w.get('layer') is None]
            values = [(self.kernel_costs.get((*shape, fmt, activation_bits, group, w['m'], w.get('layer'))),
                       w.get('weight', 1.)) for w in samples]
            if not values or any(value is None for value, _ in values):
                return None
            return sum(value * weight for value, weight in values)
        value = self.kernel_costs.get((*shape, fmt, activation_bits, group))
        if value is None and fmt != FORMAT_W6A8:
            value = self.kernel_costs.get((*shape, fmt, activation_bits))
        return value

    def byte_limit(self, info):
        params = sum(self._layer_params(info, d) for d in self._targeted)
        limit = self.compression_target_bpp * params if self.compression_target_bpp is not None else math.inf
        if self.target_bytes is not None:
            limit = min(limit, self.target_bytes - self._fixed_bytes)
        return limit

    def selected_bytes(self, info):
        return sum(self.candidates[d.name][d.encoding_format].estimated_bytes
                   if d.kind == DecisionKind.QUANTIZE else info.by_name(d.name).nbytes
                   for d in self._targeted)

    def repair_feasibility(self, info):
        """Bounded frontier search when greedy upgrades exhaust the byte budget.

        Buckets retain the cheapest prefix at similar error/original storage.
        Exact hard gates are checked after selection, never on bucket estimates.
        This is a feasibility repair, not a claim of globally optimal allocation.
        """
        limit = self.byte_limit(info)
        params = sum(self._layer_params(info, d) for d in self._targeted)
        error_limit = (self.global_gate if self.global_gate is not None else self.layer_gate) * params
        choices = []
        for d in self._targeted:
            choices.append([c for c in self.candidates[d.name].values() if c.eligible
                            and self._error_of(c) is not None and self._error_of(c) <= self.layer_gate])
        remaining = sum(min(c.estimated_bytes for c in cs) for cs in choices)
        states = [(0, 0., 0, ())]
        for d, candidates in zip(self._targeted, choices):
            remaining -= min(c.estimated_bytes for c in candidates)
            buckets = {}
            for used, error, original, path in states:
                for c in candidates:
                    size = used + c.estimated_bytes
                    next_error = error + self._error_of(c) * self._layer_params(info, d)
                    next_original = original + (c.estimated_bytes if c.format == FORMAT_ORIGINAL else 0)
                    if (size + remaining > limit or next_error > error_limit + 1e-12
                            or next_original > self.max_bf16_fraction * limit):
                        continue
                    key = (int(256 * next_error / max(error_limit, 1e-12)),
                           int(32 * next_original / max(self.max_bf16_fraction * limit, 1)))
                    state = (size, next_error, next_original, path + (c.format,))
                    if key not in buckets or state < buckets[key]:
                        buckets[key] = state
            states = sorted(buckets.values(), key=lambda s: (
                s[0] / max(limit, 1) + s[1] / max(error_limit, 1e-12), s))[:512]
            if not states:
                return False
        feasible = [s for s in states if s[2] <= self.max_bf16_fraction * s[0]
                    and any(fmt != FORMAT_ORIGINAL for fmt in s[3])]
        if not feasible:
            return False
        chosen = min(feasible)
        for d, fmt in zip(self._targeted, chosen[3]):
            if d.encoding_format != fmt:
                self.promotions.append(f'{d.name}:{fmt} (budget feasibility repair)')
            self._apply_candidate(info, d, self.candidates[d.name][fmt])
            d.reason = 'mixed planner: budget feasibility repair'
        return True

    def _apply_candidate(self, info, d, candidate):
        d.kind = DecisionKind.KEEP_PRECISION if candidate.format == FORMAT_ORIGINAL else DecisionKind.QUANTIZE
        d.convrot = candidate.format == 'int8-convrot'
        d.format = FORMAT_INT8 if d.convrot else candidate.format
        d.group_size = self.w6a8_group_size if d.format == FORMAT_W6A8 else 16
        d.convrot_groupsize = self.cgs_for(int(info.by_name(d.name).shape[1]), d.format)
        d.linear_dtype = self.linear_dtype if d.format == FORMAT_W4A4 else None

    def candidate_cost(self, info, d, candidate):
        bits = 16 if candidate.format == FORMAT_ORIGINAL else self.execution_contract(d, candidate.format).activation_bits
        return self.measured_cost(tuple(info.by_name(d.name).shape), candidate.format, bits, d.name)

    def optimize_exchanges(self, info):
        """Trade precision between two layers without spending held-out error.

        Unlike a local upgrade rule, a faster W6->W4 choice may pay for a
        compensating W4->W6/INT8 choice. Both native workload costs must be
        measured; the aggregate error, byte and original-precision gates hold.
        """
        self.exchanges = []
        before = self.global_mean_error(info, self._targeted)
        for _ in range(min(32, len(self._targeted))):
            used = self.selected_bytes(info)
            original = sum(info.by_name(d.name).nbytes for d in self._targeted
                           if d.kind != DecisionKind.QUANTIZE)
            moves = []
            for d in self._targeted:
                current = self.candidates[d.name][d.encoding_format]
                current_ms = self.candidate_cost(info, d, current)
                if current_ms is None:
                    continue
                for candidate in self.candidates[d.name].values():
                    error = self._error_of(candidate)
                    if not candidate.eligible or candidate.format == d.encoding_format or error is None or error > self.layer_gate:
                        continue
                    target_ms = self.candidate_cost(info, d, candidate)
                    if target_ms is None:
                        continue
                    delta_error = (error - self._error_of(current)) * self._layer_params(info, d)
                    delta_bytes = candidate.estimated_bytes - current.estimated_bytes
                    delta_original = (candidate.estimated_bytes if candidate.format == FORMAT_ORIGINAL else 0) - (
                        current.estimated_bytes if current.format == FORMAT_ORIGINAL else 0)
                    moves.append((target_ms - current_ms, delta_error, delta_bytes,
                                  delta_original, current_ms, d, candidate))
            # Bound pair search while retaining both fast trades and quality rescues.
            fast = sorted((m for m in moves if m[0] < 0), key=lambda m: (m[0], m[5].name, m[6].format))[:64]
            rescue = sorted(moves, key=lambda m: (m[1], m[0], m[5].name, m[6].format))[:64]
            best = None
            for first in fast:
                for second in [None, *rescue]:
                    if second is not None and first[5] is second[5]:
                        continue
                    changes = (first,) if second is None else (first, second)
                    delta_ms, delta_error, delta_bytes, delta_original, current_ms = (
                        sum(m[i] for m in changes) for i in range(5))
                    total = used + delta_bytes
                    if (delta_ms >= -.02 * current_ms or delta_error > 1e-12
                            or total > self.byte_limit(info)
                            or original + delta_original > self.max_bf16_fraction * total
                            or all((next((m[6].format for m in changes if m[5] is d), d.format)
                                    == FORMAT_ORIGINAL) for d in self._targeted)):
                        continue
                    rank = (delta_ms, delta_error, delta_bytes, tuple(m[5].name for m in changes))
                    if best is None or rank < best[0]:
                        best = (rank, changes)
            if best is None:
                break
            changes = []
            for delta_ms, delta_error, delta_bytes, _, _, d, candidate in best[1]:
                changes.append(dict(layer=d.name, previous=d.encoding_format, selected=candidate.format,
                                    delta_ms=delta_ms, delta_weighted_error=delta_error, delta_bytes=delta_bytes))
                self._apply_candidate(info, d, candidate)
                d.reason = 'adaptive measured exchange; held-out error budget preserved'
            self.exchanges.append(changes)
        self._exchange_error_before = before
        self._exchange_error_after = self.global_mean_error(info, self._targeted)

    def optimize_execution(self, info: CheckpointInfo) -> None:
        """Spend remaining storage budget on removing runtime weight expansion.

        Start from the quality-gated compact plan. Only accept changes that do
        not increase error or original-precision storage. This keeps both
        quality gates monotonic and the hard byte budget intact.
        """
        self.execution_optimizations = []
        if self.objective == "size" or self.profile.name == "size-first":
            return
        params = sum(self._layer_params(info, d) for d in self._targeted)
        used = sum(self.candidates[d.name][d.encoding_format].estimated_bytes
                   if d.kind == DecisionKind.QUANTIZE else info.by_name(d.name).nbytes
                   for d in self._targeted)
        limit = self.byte_limit(info)
        upgrades = []
        for d in self._targeted:
            if d.kind != DecisionKind.QUANTIZE:
                continue
            current = self.candidates[d.name][d.encoding_format]
            contract = self.execution_contract(d, d.encoding_format)
            for candidate in self.candidates[d.name].values():
                if not candidate.eligible or candidate.format == FORMAT_ORIGINAL:
                    continue
                target = self.execution_contract(d, candidate.format)
                shape = tuple(info.by_name(d.name).shape)
                current_ms = self.measured_cost(shape, d.encoding_format, contract.activation_bits, d.name)
                target_ms = self.measured_cost(shape, candidate.format, target.activation_bits, d.name)
                measured = current_ms is not None and target_ms is not None
                benefit = ((current_ms - target_ms) if measured else
                           self._layer_params(info, d) * (contract.preference - target.preference))
                if benefit <= 0:
                    continue
                error = self._error_of(candidate)
                if error is None or error > self._error_of(current):
                    continue
                extra = candidate.estimated_bytes - current.estimated_bytes
                # Prefer measured improvements; units never mix with heuristic scores.
                upgrades.append((not measured, -benefit / max(extra, 1),
                                 d.name, candidate.format, extra))
        changed = set()
        by_name = {d.name: d for d in self._targeted}
        for heuristic, _, name, fmt, extra in sorted(upgrades):
            if name in changed or used + extra > limit:
                continue
            d = by_name[name]
            previous = d.format
            self._apply_candidate(info, d, self.candidates[name][fmt])
            reason = "avoid weight expansion" if heuristic else "measured lower latency"
            d.reason = f"Kitchen policy: {previous} -> {fmt}; {reason}"
            used += extra
            changed.add(name)
            self.execution_optimizations.append(name)

    def _error_of(self, cand: CandidateResult) -> Optional[float]:
        if cand is None:
            return None
        return (cand.act_rel_l2 if cand.act_rel_l2 is not None
                else cand.weight_rel_l2)

    def select(self, info: CheckpointInfo,
               decisions: List[TensorDecision]) -> Tuple[int, int]:
        selected = 0
        kept = 0
        for d in self._targeted:
            cands = self.candidates.get(d.name, {})
            ordered = sorted(
                (c for c in cands.values() if c.eligible),
                key=lambda c: (c.estimated_bytes,
                                c.format != FORMAT_INT8))
            chosen = None
            for cand in ordered:
                err = self._error_of(cand)
                if err is not None and err <= self.layer_gate:
                    chosen = cand
                    break
            if chosen is None:
                # unreachable: ORIGINAL (error 0) always satisfies the gate
                d.kind = DecisionKind.KEEP_PRECISION
                d.reason = ("mixed planner: no candidate within quality gate "
                            f"(gate {self.layer_gate})")
                kept += 1
                continue
            meta = info.by_name(d.name)
            if chosen.format == FORMAT_ORIGINAL:
                self._apply_candidate(info, d, chosen)
                d.reason = ("mixed planner: original precision (no quantized "
                            "format within quality gate "
                            f"{self.layer_gate})")
                kept += 1
                continue
            self._apply_candidate(info, d, chosen)
            d.reason = f"mixed planner: {chosen.format}"
            selected += 1
        return selected, kept

    def _layer_params(self, info: CheckpointInfo, d: TensorDecision) -> int:
        meta = info.by_name(d.name)
        if meta is None or len(meta.shape) != 2:
            return 0
        return int(meta.shape[0]) * int(meta.shape[1])

    def targeted_weighted_error(self, info: CheckpointInfo,
                                decisions: List[TensorDecision]) -> Optional[float]:
        """Param-weighted mean error over the ENTIRE targeted set.

        Layers kept at original precision contribute error 0 but their
        parameters stay in the denominator, so the metric represents the
        whole targeted linear payload, not only the quantized subset."""
        num = 0.0
        den = 0
        for d in self._targeted:
            w = self._layer_params(info, d)
            if w == 0:
                continue
            if d.kind == DecisionKind.QUANTIZE:
                cand = self.candidates.get(d.name, {}).get(d.encoding_format)
                err = self._error_of(cand)
                if err is None:
                    continue
                num += err * w
            # KEEP_PRECISION layers contribute error 0 but stay in the mean
            den += w
        if den == 0:
            return None
        return num / den

    def quantized_weighted_error(self, info: CheckpointInfo,
                                 decisions: List[TensorDecision]) -> Optional[float]:
        """Diagnostic: param-weighted mean error over the quantized subset
        only (excludes original-precision layers from the denominator)."""
        num = 0.0
        den = 0
        for d in decisions:
            if d.kind != DecisionKind.QUANTIZE:
                continue
            cand = self.candidates.get(d.name, {}).get(d.encoding_format)
            err = self._error_of(cand)
            if err is None:
                continue
            w = self._layer_params(info, d)
            num += err * w
            den += w
        if den == 0:
            return None
        return num / den

    def global_mean_error(self, info: CheckpointInfo,
                          decisions: List[TensorDecision]) -> Optional[float]:
        """The optimizer/report metric: targeted_weighted_error (BF16 layers
        included with error 0)."""
        return self.targeted_weighted_error(info, decisions)

    def promote(self, info: CheckpointInfo,
                decisions: List[TensorDecision]) -> List[str]:
        """Greedy: repeatedly promote the layer with the best error reduction
        per extra byte until the weighted global gate passes or no promotion
        helps. Original precision (BF16/FP16) is a real candidate, so a
        sensitive layer can be rescued all the way to its source format."""
        while True:
            mean = self.global_mean_error(info, decisions)
            if mean is None or mean <= self.global_gate:
                break
            best: Optional[Tuple[float, TensorDecision, CandidateResult]] = None
            used, limit = self.selected_bytes(info), self.byte_limit(info)
            for d in self._targeted:
                if d.kind != DecisionKind.QUANTIZE:
                    continue
                meta = info.by_name(d.name)
                n, k = int(meta.shape[0]), int(meta.shape[1])
                cur = self.candidates.get(d.name, {}).get(d.encoding_format)
                cur_err = self._error_of(cur)
                if cur_err is None:
                    continue
                cur_bytes = cur.estimated_bytes
                # all upgrades, original precision included as a candidate:
                # the optimizer picks whichever gives the best marginal
                # error reduction per byte (W4A4->W4A8, W4A4->INT8,
                # W4A8->INT8, any -> original)
                for fmt, cand in self.candidates.get(d.name, {}).items():
                    if fmt == d.encoding_format or not cand.eligible:
                        continue
                    cand_err = self._error_of(cand)
                    if cand_err is None or cand_err >= cur_err:
                        continue
                    extra = max(cand.estimated_bytes - cur_bytes, 1)
                    if used + cand.estimated_bytes - cur_bytes > limit:
                        continue
                    ratio = self._layer_params(info, d) * (cur_err - cand_err) / float(extra)
                    if best is None or ratio > best[0]:
                        best = (ratio, d, cand)
            if best is None:
                break
            _, d, cand = best
            if cand.format == FORMAT_ORIGINAL:
                self._apply_candidate(info, d, cand)
                d.reason = "mixed planner (promoted): original precision"
            else:
                meta = info.by_name(d.name)
                self._apply_candidate(info, d, cand)
                d.reason = f"mixed planner (promoted): {cand.format}"
            # measure AFTER applying, so the log shows the new mean
            mean_after = self.global_mean_error(info, decisions)
            mean_txt = ("n/a" if mean_after is None
                        else f"{mean_after:.4f}")
            self.promotions.append(
                f"{d.name}:{cand.format} (weighted error -> {mean_txt})")
        return self.promotions

    # -- hard gates --------------------------------------------------------
    def _top_contributors(self, info: CheckpointInfo, key: Callable[
            [TensorDecision], float], limit: int = 3) -> List[str]:
        ranked = sorted(self._targeted, key=key, reverse=True)[:limit]
        out = []
        for d in ranked:
            meta = info.by_name(d.name)
            out.append(f"{d.name} ({human_bytes(meta.nbytes)})")
        return out

    def _enforce_gates(self, info: CheckpointInfo,
                       decisions: List[TensorDecision]) -> None:
        if not any(d.kind == DecisionKind.QUANTIZE
                   for d in self._targeted):
            raise QualityGateError(
                "quality gate failed: no candidate layer could be quantized "
                "within the configured gates (layer gate "
                f"{self.layer_gate}); refusing to publish a passthrough-only "
                "checkpoint. Loosen --quality-gate or the profile.")
        final_error = self.global_mean_error(info, decisions)
        if (final_error is not None and self.global_gate is not None
                and final_error > self.global_gate):
            worst = self._top_contributors(
                info, key=lambda d: (self._error_of(
                    self.candidates.get(d.name, {}).get(d.encoding_format)) or 0.0)
                    * self._layer_params(info, d))
            quant_bits = sorted({
                self.candidates.get(d.name, {}).get(d.encoding_format).format
                for d in self._targeted
                if d.kind == DecisionKind.QUANTIZE
                and self.candidates.get(d.name, {}).get(d.encoding_format) is not None})
            raise QualityGateError(
                f"quality gate failed: targeted weighted mean error "
                f"{final_error:.6f} > global gate {self.global_gate:.6f} "
                f"(profile {self.profile.name}). All remaining quantized "
                f"formats: {', '.join(quant_bits) or 'none'}. Largest error "
                f"contributors: {', '.join(worst)}. Raise "
                "--global-error-gate, provide calibration for "
                "activation-aware gates, or use a larger profile.")
        quant_bytes = 0
        kept_bytes = 0
        params = 0
        kept_params = 0
        kept_by_name: List[Tuple[str, int]] = []
        for d in self._targeted:
            meta = info.by_name(d.name)
            n, k = int(meta.shape[0]), int(meta.shape[1])
            params += n * k
            if d.kind == DecisionKind.QUANTIZE:
                quant_bytes += self.candidates[d.name][d.encoding_format].estimated_bytes
            else:
                kept_bytes += meta.nbytes
                kept_params += n * k
                kept_by_name.append((d.name, meta.nbytes))
        effective_bpp = (quant_bytes + kept_bytes) / max(params, 1)
        if self.target_bytes is not None and quant_bytes + kept_bytes + self._fixed_bytes > self.target_bytes:
            raise CompressionGateError('whole-file tensor payload exceeds --target-compression budget; protected tensors are included')
        original_byte_fraction = kept_bytes / max(quant_bytes + kept_bytes, 1)
        original_param_fraction = kept_params / max(params, 1)
        if self.compression_target_bpp is not None \
                and effective_bpp > self.compression_target_bpp:
            kept_share = 100 * original_byte_fraction
            top_kept = ", ".join(name for name, _ in
                                 sorted(kept_by_name, key=lambda kv: -kv[1])[:3])
            raise CompressionGateError(
                f"compression gate failed: effective targeted payload "
                f"{effective_bpp:.4f} bytes/parameter > target "
                f"{self.compression_target_bpp:.4f} (profile "
                f"{self.profile.name}). {kept_share:.1f}% of the targeted "
                f"output bytes are at original precision "
                f"({100*original_param_fraction:.1f}% of parameters); "
                f"largest kept layers: {top_kept or 'none'}. Raise "
                "--max-linear-bytes-per-param or "
                "--max-original-byte-fraction, or loosen the quality gates.")
        if original_byte_fraction > self.max_bf16_fraction:
            top_kept = ", ".join(name for name, _ in
                                 sorted(kept_by_name, key=lambda kv: -kv[1])[:3])
            raise CompressionGateError(
                f"compression gate failed: "
                f"{100*original_byte_fraction:.1f}% of the targeted output "
                f"bytes kept at original precision > "
                f"{100*self.max_bf16_fraction:.1f}% (profile "
                f"{self.profile.name}); {100*original_param_fraction:.1f}% "
                f"of parameters. Largest kept layers: {top_kept or 'none'}. "
                "Raise --max-original-byte-fraction or loosen the quality "
                "gates.")
        self._effective_bpp = effective_bpp
        self._original_byte_fraction = original_byte_fraction
        self._original_param_fraction = original_param_fraction

    def _summarize(self, info: CheckpointInfo,
                   decisions: List[TensorDecision]) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        layer_params: Dict[str, int] = {}
        layer_bytes: Dict[str, int] = {}
        kept_params = 0
        kept_bytes = 0
        for d in self._targeted:
            meta = info.by_name(d.name)
            n, k = int(meta.shape[0]), int(meta.shape[1])
            if d.kind == DecisionKind.QUANTIZE:
                counts[d.format] = counts.get(d.format, 0) + 1
                layer_params[d.format] = layer_params.get(d.format, 0) + n * k
                layer_bytes[d.format] = layer_bytes.get(
                    d.format, 0) + self.candidates[d.name][d.encoding_format].estimated_bytes
            else:
                kept_params += n * k
                kept_bytes += meta.nbytes
        mean = self.targeted_weighted_error(info, decisions)
        q_mean = self.quantized_weighted_error(info, decisions)
        return {
            "profile": self.profile.name,
            "quality_reference": self.quality_reference,
            "reference_error": self.reference_error,
            "reference_fallbacks": self.reference_fallbacks,
            "int8_convrot_layers": [d.name for d in self._targeted if d.convrot],
            "error_metric": 'heldout-p95-runtime-output-relL2' if self.profile.name == 'adaptive' else 'activation-mean-or-weight-relL2',
            "precision_exchanges": getattr(self, 'exchanges', []),
            "exchange_error_before": getattr(self, '_exchange_error_before', None),
            "exchange_error_after": getattr(self, '_exchange_error_after', None),
            "whole_payload_budget": self.target_bytes,
            "layer_gate": self.layer_gate,
            "global_gate": self.global_gate,
            "global_mean_error": mean,
            "targeted_weighted_error": mean,
            "quantized_weighted_error": q_mean,
            "counts": counts,
            "layer_params": layer_params,
            "layer_bytes": layer_bytes,
            "kept_params": kept_params,
            "kept_bytes": kept_bytes,
            "effective_bpp": getattr(self, "_effective_bpp", None),
            "original_precision_parameter_fraction": getattr(
                self, "_original_param_fraction", None),
            "original_precision_output_byte_fraction": getattr(
                self, "_original_byte_fraction", None),
            "compression_target_bpp": self.compression_target_bpp,
            "max_bf16_fraction": self.max_bf16_fraction,
            "execution": {
                d.name: dataclasses.asdict(self.execution_contract(d, d.encoding_format))
                for d in self._targeted if d.kind == DecisionKind.QUANTIZE
            },
            "calibration_coverage": {
                "targeted_layers": len(self._targeted),
                "sampled_layers": sum(self._activations(d.name) is not None
                                      for d in self._targeted),
            },
            "candidates": {
                name: {
                    fmt: {
                        "eligible": cand.eligible,
                        "reason": cand.reason,
                        "bytes": cand.estimated_bytes,
                        "group_size": self.w6a8_group_size if fmt == FORMAT_W6A8 else 16,
                        "weight_rel_l2": cand.weight_rel_l2,
                        "act_rel_l2": cand.act_rel_l2,
                        "requested_linear_dtype": cand.requested_linear_dtype,
                        "effective_activation_bits": cand.effective_activation_bits,
                        "effective_runtime_path": cand.effective_runtime_path,
                        "runtime_certified": cand.runtime_certified,
                        "runtime_certain": cand.runtime_certain,
                        "backend": cand.backend,
                    } for fmt, cand in cands.items()
                } for name, cands in self.candidates.items()
            },
            "runtime": {
                "backend": self.runtime.target,
                "gpu_name": self.runtime.gpu_name,
                "cuda_capability": list(self.runtime.cuda_capability)
                if self.runtime.cuda_capability else None,
                "rocm_arch": self.runtime.rocm_arch,
                "runtime_certified": self.runtime.runtime_certified,
                "formats": {
                    fmt: {
                        "loadable": self.runtime.capability(fmt).loadable,
                        "executable": self.runtime.capability(fmt).executable,
                        "accelerated": self.runtime.capability(fmt).accelerated,
                        "certified": self.runtime.capability(fmt).certified,
                        "backend": self.runtime.capability(fmt).backend,
                        "reason": self.runtime.capability(fmt).reason,
                        "status": self.runtime.describe(fmt),
                    } for fmt in MIXED_FORMATS if fmt != FORMAT_W6A8 or self.enable_w6a8
                },
            },
        }
