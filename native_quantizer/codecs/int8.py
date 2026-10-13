"""Row-wise INT8 encoding and shape eligibility."""

from __future__ import annotations

import torch
from typing import Sequence, Tuple

from native_quantizer.dtypes import FLOAT_DTYPES
from native_quantizer.errors import PolicyError
from native_quantizer.identity import INT8_SCALE_MAX


def quantize_int8_tensorwise_weight(
        weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rowwise symmetric INT8 (int8_tensorwise, no ConvRot).

    Serialized contract (matches ComfyUI TensorWiseINT8Layout + the Comfy-Org
    int8_convrot checkpoint family): `weight` int8 [N, K], `weight_scale`
    fp32 [N]. Runtime applies dynamic rowwise int8 activation quantization.
    Works for any K (no divisibility requirement), which is why it is the
    universal fallback tier in mixed mode.
    """
    if weight.ndim != 2:
        raise PolicyError(f"INT8 weight must be 2D, got {tuple(weight.shape)}")
    if weight.shape[0] <= 0 or weight.shape[1] <= 0:
        raise PolicyError(f"INT8 weight dimensions must be positive, got {tuple(weight.shape)}")
    abs_max = weight.abs().amax(dim=-1, keepdim=True)
    if not bool(torch.isfinite(abs_max).all()):
        raise PolicyError("cannot quantize a weight containing NaN or Inf")
    scale = (abs_max.float() / float(INT8_SCALE_MAX)).clamp(min=1e-30)
    q = (weight / scale).round().clamp_(-128.0, 127.0).to(torch.int8)
    # scale is stored [N, 1] so both the eager dequant (q.float() * scale) and
    # the CUDA int8_linear paths broadcast correctly on every backend
    return q.contiguous(), scale.contiguous().to(torch.float32)


def dequantize_int8_tensorwise_weight(
        q: torch.Tensor, scale: torch.Tensor,
        output_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    s = scale.to(device=q.device, dtype=torch.float32)
    w = q.float() * s.reshape(-1, 1)
    return w.to(output_dtype)


def int8_weight_is_quantizable(shape: Sequence[int], dtype: torch.dtype
                               ) -> Tuple[bool, str]:
    if len(shape) != 2:
        return False, "not 2D"
    if dtype not in FLOAT_DTYPES:
        return False, f"dtype {dtype} not float"
    if int(shape[0]) <= 0 or int(shape[1]) <= 0:
        return False, f"dimensions must be positive, got {tuple(shape)}"
    return True, ""
