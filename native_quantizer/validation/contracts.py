"""Serialized W6A8 layout and provenance contracts."""

from __future__ import annotations

import torch

from native_quantizer.codecs.w6a8 import validate_w6_shape
from native_quantizer.dtypes import TORCH_TO_SAFE
from native_quantizer.errors import PolicyError
from native_quantizer.identity import FORMAT_MIXED, FORMAT_W6A8, format_revision


def w6a8_checkpoint_contract(st, layer, conf, source_shape=None):
    if conf.get("convrot") is not True:
        raise PolicyError("W6A8 metadata requires convrot=true")
    gs, cgs = conf.get("group_size"), conf.get("convrot_groupsize")
    names = set(st.keys())
    base = layer + ".weight"
    required = {base, base + "_s_rel", base + "_s_channel"}
    if not required <= names or {base + "_codebook", base + "_correction"} & names:
        raise PolicyError("W6A8 requires scales and forbids codebook/correction tensors")
    weight = st.get_slice(base)
    shape = tuple(weight.get_shape())
    if len(shape) != 2 or weight.get_dtype() != "I8" or shape[1] % 3:
        raise PolicyError("malformed W6A8 packed width/dtype")
    n, width = shape
    k = width * 4 // 3
    validate_w6_shape(k, gs, cgs)
    if n <= 0 or source_shape is not None and tuple(source_shape) != (n, k):
        raise PolicyError("W6A8 source and packed dimensions differ")
    rel, channel = st.get_slice(base + "_s_rel"), st.get_slice(base + "_s_channel")
    if tuple(rel.get_shape()) != (n, k // gs) or rel.get_dtype() != TORCH_TO_SAFE[torch.float8_e4m3fn]:
        raise PolicyError("malformed W6A8 FP8 group scales")
    if tuple(channel.get_shape()) != (n,) or channel.get_dtype() != "F32":
        raise PolicyError("malformed W6A8 FP32 channel scales")
    rows = max(1, min(1024, (8 * 1024**2) // (8 * (k // gs) + 16)))
    for r0 in range(0, n, rows):
        sr, sc = rel[r0:r0+rows].float(), channel[r0:r0+rows]
        if not bool(torch.isfinite(sr).all() and torch.isfinite(sc).all()) or bool((sr < 0).any() or (sc <= 0).any()):
            raise PolicyError("non-finite or invalid W6A8 scales")
    return n, k


def w6a8_extension_contract(extension, groups):
    """Validate namespaced provenance without importing a runtime decoder."""
    if not isinstance(extension, dict):
        return False
    mode = extension.get("format")
    quant = extension.get("quantization")
    if not isinstance(quant, dict):
        return False
    formats = quant.get("formats", {})
    contract = formats.get(FORMAT_W6A8, {}) if mode == FORMAT_MIXED and isinstance(formats, dict) else quant
    if not isinstance(contract, dict):
        return False
    return (mode in (FORMAT_W6A8, FORMAT_MIXED)
        and extension.get("format_revision") == format_revision(mode)
        and extension.get("schema") == ("comfy_wxa8/v2" if mode == FORMAT_MIXED else "comfy_wxa8/v1")
        and contract.get("weight_bits") == 6 and contract.get("activation_bits") == 8
        and contract.get("packing") == "int6-nibble-high2-plane"
        and contract.get("scale_dtype") == "fp8_e4m3fn"
        and contract.get("convrot") is True and contract.get("convrot_groupsize") == 256
        and groups == {contract.get("group_size")})
