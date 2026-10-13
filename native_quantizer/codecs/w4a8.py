"""Codebook W4A8 encoding, scale fitting and reconstruction."""

from __future__ import annotations

import torch
from typing import Any, Optional, Sequence, Tuple

from native_quantizer.codecs.packing import unpack_w4
from native_quantizer.codecs.rotation import _is_power_of_four, rotate_int8_convrot_weight
from native_quantizer.dtypes import FLOAT_DTYPES
from native_quantizer.errors import PolicyError
from native_quantizer.identity import (
    CODEBOOK_DECISION_SAMPLE_COUNT,
    CODEBOOK_KURTOSIS_SAMPLE_COUNT,
    CODEBOOK_MODES,
    CODEBOOK_SAMPLE_COUNT,
    FIXED_W4A8_CODEBOOK,
    W4A8_ALS_ITERS,
    W4A8_GATE_KURTOSIS,
)


def validate_w4_shape(k: int, group_size: int, convrot_groupsize: int) -> None:
    if k <= 0:
        raise PolicyError(f"K must be positive, got {k}")
    if group_size < 4:
        raise PolicyError(f"group_size must be >= 4, got {group_size}")
    if convrot_groupsize < 4 or not _is_power_of_four(convrot_groupsize):
        raise PolicyError(
            f"convrot_groupsize must be a power of 4, got {convrot_groupsize}")
    if k % 16 != 0:
        raise PolicyError(f"K={k} must be divisible by 16 for 4-bit packing")
    if k % group_size != 0:
        raise PolicyError(f"K={k} must be divisible by group_size={group_size}")
    if k % convrot_groupsize != 0:
        raise PolicyError(f"K={k} must be divisible by convrot_groupsize={convrot_groupsize}")
    if (16 % group_size != 0) and (group_size % 16 != 0):
        raise PolicyError(
            f"group_size must divide 16 or be a multiple of 16, got {group_size}")


def w4_weight_is_quantizable(shape: Sequence[int], dtype: torch.dtype,
                             group_size: int, convrot_groupsize: int) -> Tuple[bool, str]:
    if len(shape) != 2:
        return False, f"not 2D (shape {tuple(shape)})"
    if dtype not in FLOAT_DTYPES:
        return False, f"not a float dtype ({dtype})"
    if int(shape[0]) <= 0:
        return False, f"dimensions must be positive, got {tuple(shape)}"
    k = int(shape[1])
    try:
        validate_w4_shape(k, group_size, convrot_groupsize)
    except PolicyError as e:
        return False, str(e)
    return True, "ok"


def fit_codebook(normalized: torch.Tensor, levels: int = 16, iterations: int = 25,
                 sample_size: int = CODEBOOK_SAMPLE_COUNT,
                 seed: int = 0) -> torch.Tensor:
    """Data-free Lloyd-Max codebook on normalized rotated weights.

    Deterministic: subsampling (when needed) uses a fixed-seed generator, exactly
    like the reference implementation. The sample count is canonical (never
    derived from the memory budget), so chunked and in-memory paths agree.
    """
    if levels < 2:
        raise PolicyError(f"codebook levels must be >= 2, got {levels}")
    if iterations < 0:
        raise PolicyError(f"codebook iterations cannot be negative, got {iterations}")
    if sample_size <= 0:
        raise PolicyError(f"codebook sample_size must be positive, got {sample_size}")
    samples = normalized.flatten()
    if samples.numel() > sample_size:
        generator = torch.Generator(device=samples.device).manual_seed(seed)
        indices = torch.randint(0, samples.numel(), (sample_size,),
                                device=samples.device, generator=generator)
        samples = samples[indices]
    samples = samples.float()
    if samples.numel() == 0:
        raise PolicyError("cannot fit a W4A8 codebook from an empty tensor")
    if not bool(torch.isfinite(samples).all()):
        raise PolicyError("cannot quantize a weight containing NaN or Inf")
    codebook = torch.quantile(samples, torch.linspace(0, 1, levels, device=samples.device))
    for _ in range(iterations):
        assignments = assign_codes(samples, codebook).to(torch.long)
        updated = codebook.clone()
        for index in range(levels):
            selected = assignments == index
            if selected.any():
                updated[index] = samples[selected].mean()
        codebook = updated
    return codebook.contiguous()


def _fixed_w4a8_codebook(device: Any = "cpu") -> torch.Tensor:
    return torch.tensor(FIXED_W4A8_CODEBOOK, device=device,
                        dtype=torch.float32)


def _codebook_decision_sample(normalized: torch.Tensor) -> torch.Tensor:
    """Return deterministic, evenly spaced rows for adaptive codebook choice.

    Sampling complete rows makes the choice invariant to the conversion memory
    budget: the in-memory and bounded readers can select exactly the same rows
    without scanning and rotating the entire tensor a second time.
    """
    if normalized.ndim != 3:
        raise PolicyError(
            f"normalized W4A8 weights must be 3D, got {tuple(normalized.shape)}")
    n = int(normalized.shape[0])
    k = int(normalized.shape[1]) * int(normalized.shape[2])
    if n == 0 or k == 0:
        raise PolicyError("cannot choose a W4A8 codebook from an empty tensor")
    max_rows = max(1, CODEBOOK_DECISION_SAMPLE_COUNT // max(k, 1))
    if n <= max_rows:
        return normalized
    indices = torch.linspace(0, n - 1, max_rows,
                             device=normalized.device).long()
    return normalized.index_select(0, indices)


def choose_w4a8_codebook(normalized: torch.Tensor, mode: str = "auto",
                          seed: int = 0) -> torch.Tensor:
    """Choose the fixed fast table or a fitted table for one rotated weight."""
    if mode not in CODEBOOK_MODES:
        raise PolicyError(
            f"unknown W4A8 codebook mode {mode!r}; expected one of {CODEBOOK_MODES}")
    if mode == "fixed":
        return _fixed_w4a8_codebook(normalized.device)
    if mode == "fit":
        return fit_codebook(normalized, levels=16, iterations=25, seed=seed)

    decision_rows = _codebook_decision_sample(normalized)
    sample = decision_rows.detach().flatten()
    if sample.numel() > CODEBOOK_KURTOSIS_SAMPLE_COUNT:
        generator = torch.Generator(device=sample.device).manual_seed(seed)
        indices = torch.randint(
            0, sample.numel(), (CODEBOOK_KURTOSIS_SAMPLE_COUNT,),
            device=sample.device, generator=generator)
        sample = sample[indices]
    sample = sample.float()
    if not bool(torch.isfinite(sample).all()):
        raise PolicyError("cannot quantize a weight containing NaN or Inf")
    mean = sample.mean()
    std = sample.std()
    if not bool(torch.isfinite(std)) or float(std) <= 1e-9:
        return _fixed_w4a8_codebook(normalized.device)
    excess_kurtosis = (((sample - mean) / (std + 1e-9)).pow(4).mean()
                       - 3.0)
    if float(excess_kurtosis) <= W4A8_GATE_KURTOSIS:
        return _fixed_w4a8_codebook(normalized.device)
    return fit_codebook(decision_rows, levels=16, iterations=25, seed=seed)


def assign_codes(normalized: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Nearest codebook index via binary search (ties choose the lower code).

    Lloyd-Max codebooks are monotonic, so only the two values neighboring the
    insertion point can be closest.  This is bit-equivalent to the old
    sixteen-pass scan and substantially reduces allocations and comparisons.
    """
    if codebook.ndim != 1 or codebook.numel() < 2:
        raise PolicyError("W4A8 codebook must be a 1D tensor with at least 2 levels")
    if not bool((codebook[1:] >= codebook[:-1]).all()):
        raise PolicyError("W4A8 codebook levels must be monotonic")
    last = codebook.numel() - 1
    values = normalized.contiguous()
    pos = torch.searchsorted(codebook, values)
    lo = (pos - 1).clamp(0, last)
    hi = pos.clamp(0, last)
    distance_lo = (values - codebook[lo]).abs_()
    distance_hi = (values - codebook[hi]).abs_()
    return torch.where(distance_hi < distance_lo, hi, lo).to(torch.int32)


def assign_grid(weight: torch.Tensor, levels: torch.Tensor, s_channel: torch.Tensor) -> torch.Tensor:
    """Nearest decoded int8 level per group using batched binary search."""
    n, groups, group_size = weight.shape
    if levels.shape != (n, groups, 16):
        raise PolicyError(
            f"W4A8 level grid shape {tuple(levels.shape)} != {(n, groups, 16)}")
    level_rows = levels.reshape(n * groups, 16).contiguous()
    targets = (weight / s_channel.view(-1, 1, 1)).reshape(
        n * groups, group_size).contiguous()
    last = 15
    pos = torch.searchsorted(level_rows, targets)
    lo = (pos - 1).clamp(0, last)
    hi = pos.clamp(0, last)
    distance_lo = targets.sub(torch.gather(level_rows, 1, lo)).abs_()
    distance_hi = targets.sub(torch.gather(level_rows, 1, hi)).abs_()
    return torch.where(distance_hi < distance_lo, hi, lo).to(
        torch.int32).reshape(n, groups, group_size)


def quantize_w4a8_weight(weight: torch.Tensor, group_size: int = 16,
                         convrot_groupsize: int = 256, symmetric: bool = True,
                         scale_dtype: torch.dtype = torch.float8_e4m3fn,
                         codebook: bool = True, seed: int = 0,
                         codebook_mode: str = "auto",
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                    Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Quantize a floating 2D weight into W4A8 storage.

    Returns (packed, s_rel, s_channel, correction, codebook_tensor).
    Bit-exact with comfy-kitchen `quantize_w4a8_int8_weight` when run on the
    same device (see module docstring for the algorithm).
    """
    if scale_dtype not in (torch.float32, torch.float8_e4m3fn):
        raise PolicyError(f"scale_dtype must be float32 or float8_e4m3fn, got {scale_dtype}")
    if weight.ndim != 2:
        raise PolicyError(f"W4A8 weight must be 2D, got {tuple(weight.shape)}")
    if weight.shape[0] <= 0 or weight.shape[1] <= 0:
        raise PolicyError(
            f"W4A8 weight dimensions must be positive, got {tuple(weight.shape)}")
    validate_w4_shape(int(weight.shape[1]), group_size, convrot_groupsize)
    rotated = rotate_int8_convrot_weight(weight, convrot_groupsize)
    return _quantize_rotated_w4a8(rotated, group_size, symmetric, scale_dtype,
                                  codebook, seed=seed,
                                  codebook_mode=codebook_mode)


def _quantize_rotated_w4a8(weight: torch.Tensor, group_size: int, symmetric: bool,
                           scale_dtype: torch.dtype, codebook: bool,
                           seed: int = 0, codebook_mode: str = "auto",
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                      Optional[torch.Tensor], Optional[torch.Tensor]]:
    original_dtype = weight.dtype
    n, k = weight.shape
    groups = k // group_size
    grouped_weight = weight.float().view(n, groups, group_size)

    codebook_tensor: Optional[torch.Tensor] = None
    correction: Optional[torch.Tensor] = None
    if symmetric and codebook:
        group_scale = grouped_weight.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        if not bool(torch.isfinite(group_scale).all()):
            raise PolicyError("cannot quantize a weight containing NaN or Inf")
        normalized = grouped_weight / group_scale
        codebook_tensor = choose_w4a8_codebook(
            normalized, mode=codebook_mode, seed=seed)
        quantized = assign_codes(normalized, codebook_tensor)
        als_iterations = 3 if codebook_mode == "fit" else W4A8_ALS_ITERS
        for _ in range(als_iterations):
            qc = codebook_tensor[quantized]
            group_scale = (
                (grouped_weight * qc).sum(-1, keepdim=True)
                / (qc * qc).sum(-1, keepdim=True).clamp(min=1e-8)
            ).clamp(min=1e-8)
            quantized = assign_codes(grouped_weight / group_scale, codebook_tensor)
        unsigned = quantized.to(torch.int32).view(n, k)
        shifted_weight = codebook_tensor[quantized] * group_scale
    elif symmetric:
        group_scale = (grouped_weight.abs().amax(dim=-1, keepdim=True) / 7.0).clamp(min=1e-8)
        if not bool(torch.isfinite(group_scale).all()):
            raise PolicyError("cannot quantize a weight containing NaN or Inf")
        signed = (grouped_weight / group_scale).round().clamp(-8, 7).to(torch.int32)
        unsigned = (signed + 8).view(n, k)
        shifted_weight = signed * group_scale
    else:
        minimum = grouped_weight.amin(dim=-1, keepdim=True)
        group_scale = ((grouped_weight.amax(dim=-1, keepdim=True) - minimum) / 15.0).clamp(min=1e-8)
        if not bool(torch.isfinite(group_scale).all()):
            raise PolicyError("cannot quantize a weight containing NaN or Inf")
        unsigned = (
            ((grouped_weight - minimum) / group_scale).round().clamp(0, 15)
            .to(torch.int32).view(n, k)
        )
        shifted_weight = (unsigned.view(n, groups, group_size) - 8) * group_scale
        correction = (8.0 * group_scale + minimum).squeeze(-1).t().contiguous().to(original_dtype)

    s_channel = (shifted_weight.abs().amax(dim=(1, 2)) / 127.0).clamp(min=1e-8)
    s_rel = (group_scale.squeeze(-1) / s_channel.unsqueeze(1)).float().contiguous()
    if scale_dtype != torch.float32:
        s_rel = s_rel.to(scale_dtype).contiguous()
    if codebook_tensor is not None:
        levels = (
            (codebook_tensor.view(1, 1, 16) * s_rel.float().unsqueeze(-1))
            .round_().clamp_(-127, 127)
        )
        unsigned = assign_grid(grouped_weight, levels, s_channel).view(n, k)

    packed = (
        ((unsigned[:, 0::2] & 0xF) | ((unsigned[:, 1::2] & 0xF) << 4))
        .to(torch.int8).contiguous()
    )
    return packed, s_rel, s_channel.float().contiguous(), correction, codebook_tensor


def dequantize_w4a8_weight(packed: torch.Tensor, s_rel: torch.Tensor,
                           s_channel: torch.Tensor,
                           codebook: Optional[torch.Tensor] = None,
                           correction: Optional[torch.Tensor] = None,
                           group_size: int = 16, convrot_groupsize: int = 256,
                           output_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Decode W4A8 storage back to the physical [N, K] weight (reference decode).

    Bit-exact with comfy-kitchen `dequantize_w4a8_int8_weight` (eager/triton/cuda).
    """
    if packed.dim() != 2 or packed.dtype != torch.int8:
        raise PolicyError("packed weight must be a 2D int8 tensor")
    n, k_half = packed.shape
    k = k_half * 2
    validate_w4_shape(k, group_size, convrot_groupsize)
    groups = k // group_size
    if tuple(s_rel.shape) != (n, groups):
        raise PolicyError(f"s_rel must have shape {(n, groups)}, got {tuple(s_rel.shape)}")
    if tuple(s_channel.shape) != (n,):
        raise PolicyError(f"s_channel must have shape {(n,)}, got {tuple(s_channel.shape)}")
    if correction is not None and tuple(correction.shape) != (groups, n):
        raise PolicyError(f"correction must have shape {(groups, n)}, got {tuple(correction.shape)}")
    if codebook is not None and tuple(codebook.shape) != (16,):
        raise PolicyError(f"codebook must have shape (16,), got {tuple(codebook.shape)}")

    codes = unpack_w4(packed)
    if codebook is not None:
        values = codebook.to(device=packed.device, dtype=torch.float32)[codes]
    else:
        values = codes.float() - 8.0
    values = values.view(n, groups, group_size) * s_rel.float().unsqueeze(-1)
    int8_weight = values.view(n, k).round().clamp_(-127, 127).to(torch.int8)

    weight_rotated = int8_weight.float().view(n, groups, group_size)
    weight_rotated = weight_rotated * s_channel.float().view(n, 1, 1)
    if correction is not None:
        weight_rotated = weight_rotated + correction.t().unsqueeze(-1).float()
    weight_rotated = weight_rotated.view(n, k)
    return rotate_int8_convrot_weight(weight_rotated, convrot_groupsize).to(output_dtype)


def decode_w4a8_runtime_weight(packed: torch.Tensor, s_rel: torch.Tensor,
                                codebook: Optional[torch.Tensor],
                                group_size: int) -> torch.Tensor:
    """Decode W4A8 storage into the ROTATED runtime weight representation the
    W4A8 GEMM consumes: packed int4 codes -> codebook values -> per-group
    fp8 scales -> rounded int8 levels, WITHOUT the inverse ConvRot rotation.

    This is the weight side of the runtime operation
    (comfy-kitchen `_dequant_int4_grouped_to_int8`). The physical weight
    (`dequantize_w4a8_weight`) additionally rotates back to the ordinary
    basis; mixing the two coordinate systems in one simulation is wrong."""
    codes = unpack_w4(packed)
    if codebook is not None:
        values = codebook.to(device=packed.device, dtype=torch.float32)[codes]
    else:
        values = codes.float() - 8.0
    n, k = values.shape
    groups = k // group_size
    values = values.view(n, groups, group_size) * s_rel.float().unsqueeze(-1)
    return values.view(n, k).round().clamp_(-127, 127).to(torch.int8)
