"""Format routing for tensor encoding and reconstruction."""

from __future__ import annotations

import torch
from typing import Dict

from native_quantizer.codecs.int8 import dequantize_int8_tensorwise_weight, quantize_int8_tensorwise_weight
from native_quantizer.codecs.w4a4 import dequantize_w4a4_weight, quantize_w4a4_weight
from native_quantizer.codecs.w4a8 import dequantize_w4a8_weight, quantize_w4a8_weight
from native_quantizer.codecs.w6a8 import dequantize_w6a8_weight, validate_w6_shape, validate_w6_tensors
from native_quantizer.errors import PolicyError
from native_quantizer.identity import FORMAT_INT8, FORMAT_W4A4, FORMAT_W4A8, FORMAT_W6A8
from native_quantizer.runtime.kitchen import require_w6a8_api


def quantize_weight_by_format(weight: torch.Tensor, fmt: str, group_size: int,
                              convrot_groupsize: int,
                              seed: int = 0,
                              codebook_mode: str = "auto",
                              ) -> Dict[str, torch.Tensor]:
    """Quantize with the W4A8 layout; returns the per-layer output tensors
    keyed by suffix ('' for the packed weight, '_s_rel', '_s_channel',
    '_codebook', optional '_correction')."""
    if fmt == FORMAT_W6A8:
        if weight.ndim != 2 or weight.shape[0] <= 0:
            raise PolicyError("W6A8 requires a nonempty 2D weight")
        validate_w6_shape(int(weight.shape[1]), group_size, convrot_groupsize)
        if not bool(torch.isfinite(weight.float()).all()):
            raise PolicyError("cannot quantize a weight containing NaN or Inf")
        layout = require_w6a8_api()
        packed, params = layout.quantize(weight.float(), group_size=group_size,
            convrot_groupsize=convrot_groupsize, bits=6, symmetric=True,
            scale_dtype=torch.float8_e4m3fn, codebook=False, scale_search=True)
        out = layout.state_dict_tensors(packed, params)
        validate_w6_tensors(out, group_size, convrot_groupsize, weight.shape)
        return out
    if fmt == FORMAT_W4A8:
        packed, s_rel, s_ch, corr, cb = quantize_w4a8_weight(
            weight, group_size=group_size, convrot_groupsize=convrot_groupsize,
            symmetric=True, scale_dtype=torch.float8_e4m3fn, codebook=True,
            seed=seed, codebook_mode=codebook_mode)
        out = {"": packed, "_s_rel": s_rel, "_s_channel": s_ch, "_codebook": cb}
        if corr is not None:
            out["_correction"] = corr
        return out
    if fmt == FORMAT_INT8:
        q, scale = quantize_int8_tensorwise_weight(weight)
        return {"": q, "_scale": scale}
    if fmt == FORMAT_W4A4:
        packed, scale = quantize_w4a4_weight(weight, convrot_groupsize)
        return {"": packed, "_scale": scale}
    raise PolicyError(f"unknown quantization format {fmt!r}")


def dequantize_weight_by_format(tensors: Dict[str, torch.Tensor], fmt: str,
                                group_size: int, convrot_groupsize: int,
                                output_dtype: torch.dtype) -> torch.Tensor:
    if fmt == FORMAT_W6A8:
        validate_w6_tensors(tensors, group_size, convrot_groupsize)
        return dequantize_w6a8_weight(tensors[""], tensors["_s_rel"], tensors["_s_channel"],
                                    group_size, convrot_groupsize).to(output_dtype)
    if fmt == FORMAT_W4A8:
        return dequantize_w4a8_weight(
            tensors[""], tensors["_s_rel"], tensors["_s_channel"],
            codebook=tensors.get("_codebook"), correction=tensors.get("_correction"),
            group_size=group_size, convrot_groupsize=convrot_groupsize,
            output_dtype=output_dtype)
    if fmt == FORMAT_INT8:
        return dequantize_int8_tensorwise_weight(tensors[""], tensors["_scale"],
                                                 output_dtype)
    if fmt == FORMAT_W4A4:
        return dequantize_w4a4_weight(tensors[""], tensors["_scale"],
                                      convrot_groupsize, output_dtype)
    raise PolicyError(f"unknown quantization format {fmt!r}")
