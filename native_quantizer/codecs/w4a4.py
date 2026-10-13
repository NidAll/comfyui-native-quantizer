"""Rotated signed INT4 encoding and shape eligibility."""

from __future__ import annotations

import torch
from typing import Optional, Sequence, Tuple

from native_quantizer.codecs.packing import unpack_int4_signed
from native_quantizer.codecs.rotation import build_hadamard, rotate_weight
from native_quantizer.dtypes import FLOAT_DTYPES
from native_quantizer.errors import PolicyError
from native_quantizer.identity import W4A4_EMISSION_MAX, W4A4_QUANT_GROUP_SIZE


def quantize_w4a4_weight(weight: torch.Tensor,
                         convrot_groupsize: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """ConvRot W4A4 quantization, byte-identical to comfy-kitchen's eager
    `quantize_convrot_w4a4_weight`: regular-Hadamard weight rotation, rowwise
    symmetric signed int4 with absmax scale = max/7, emission range [-7, 7],
    packed int4 low=even nibble. Returns (packed int8 [N, K/2], scale fp32 [N])."""
    if weight.ndim != 2:
        raise PolicyError(f"W4A4 weight must be 2D, got {tuple(weight.shape)}")
    if weight.shape[0] <= 0 or weight.shape[1] <= 0:
        raise PolicyError(f"W4A4 weight dimensions must be positive, got {tuple(weight.shape)}")
    ok, why = w4a4_weight_is_quantizable(
        weight.shape, weight.dtype, convrot_groupsize)
    if not ok:
        raise PolicyError(f"invalid W4A4 weight: {why}")
    h = build_hadamard(convrot_groupsize, device=weight.device, dtype=weight.dtype)
    rot = rotate_weight(weight, h, convrot_groupsize)
    return quantize_rotated_int4(rot)


def quantize_rotated_int4(rot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Kitchen signed INT4: row scale, even low nibble, row-major int8 storage."""
    abs_max = rot.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    if not bool(torch.isfinite(abs_max).all()):
        raise PolicyError("cannot quantize a weight containing NaN or Inf")
    scale = abs_max / float(W4A4_EMISSION_MAX)
    q = (rot / scale).round().clamp_(-float(W4A4_EMISSION_MAX),
                                    float(W4A4_EMISSION_MAX)).to(torch.int8)
    packed = (q[:, 0::2] & 0xF) | ((q[:, 1::2] & 0xF) << 4)
    return packed, scale.reshape(-1).to(torch.float32)


def dequantize_w4a4_weight(packed: torch.Tensor, scale: torch.Tensor,
                           convrot_groupsize: int,
                           output_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Inverse of quantize_w4a4_weight (matches comfy-kitchen eager dequant)."""
    w_int = unpack_int4_signed(packed).float()
    w_rot = w_int * scale.to(device=packed.device, dtype=torch.float32).reshape(-1, 1)
    h = build_hadamard(convrot_groupsize, device=packed.device, dtype=torch.float32)
    return rotate_weight(w_rot.float(), h, convrot_groupsize).to(output_dtype)


def _pick_w4a4_convrot_group(k: int) -> int:
    """Largest power of 4 in {16, 64, 256} dividing K (kernel-supported set)."""
    for cgs in (256, 64, 16):
        if k % cgs == 0:
            return cgs
    raise PolicyError(f"K={k} has no ConvRot-W4A4 group in {{16, 64, 256}}")


def w4a4_weight_is_quantizable(shape: Sequence[int], dtype: torch.dtype,
                               convrot_groupsize: Optional[int] = None
                               ) -> Tuple[bool, str]:
    """W4A4 eligibility: 2D float, K % 64 == 0 (kernel contract), K % cgs == 0."""
    if len(shape) != 2:
        return False, "not 2D"
    if dtype not in FLOAT_DTYPES:
        return False, f"dtype {dtype} not float"
    k = int(shape[1])
    if int(shape[0]) <= 0 or k <= 0:
        return False, f"dimensions must be positive, got {tuple(shape)}"
    if k % W4A4_QUANT_GROUP_SIZE != 0:
        return False, f"K={k} not divisible by quant_group_size {W4A4_QUANT_GROUP_SIZE}"
    if convrot_groupsize is None:
        try:
            convrot_groupsize = _pick_w4a4_convrot_group(k)
        except PolicyError:
            return False, f"K={k} has no supported ConvRot group"
    if convrot_groupsize <= 0:
        return False, f"convrot_groupsize must be positive, got {convrot_groupsize}"
    if k % convrot_groupsize != 0:
        return False, f"K={k} not divisible by convrot_groupsize {convrot_groupsize}"
    return True, ""
