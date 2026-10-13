"""Safetensors dtype mappings and tensor byte accounting."""

from __future__ import annotations

import math
import safetensors
import safetensors.torch
import torch
from typing import Dict, Sequence

from native_quantizer.errors import InputError


def torch_dtype_name(dt: torch.dtype) -> str:
    return str(dt).replace("torch.", "")


SAFE_TO_TORCH: Dict[str, torch.dtype] = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
    "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
}


_OPTIONAL_SAFE_DTYPES = {
    "F8_E4M3FNUZ": "float8_e4m3fnuz",
    "F8_E5M2FNUZ": "float8_e5m2fnuz",
    "C64": "complex64",
    "U16": "uint16",
    "U32": "uint32",
    "U64": "uint64",
}


_installed_safe_types = getattr(safetensors.torch, "_TYPES", {})


for _safe_name, _torch_name in _OPTIONAL_SAFE_DTYPES.items():
    _dtype = getattr(torch, _torch_name, None)
    if _dtype is not None and _safe_name in _installed_safe_types:
        SAFE_TO_TORCH[_safe_name] = _dtype


TORCH_TO_SAFE: Dict[torch.dtype, str] = {v: k for k, v in SAFE_TO_TORCH.items()}


FP8_DTYPES = {dtype for name, dtype in SAFE_TO_TORCH.items()
              if name.startswith("F8_")}


FLOAT_DTYPES = {torch.float32, torch.float16, torch.bfloat16, torch.float64}


def tensor_nbytes(dtype: torch.dtype, shape: Sequence[int]) -> int:
    return math.prod(int(s) for s in shape) * dtype.itemsize


def torch_dtype_from_safe(name: str) -> torch.dtype:
    try:
        return SAFE_TO_TORCH[name]
    except KeyError as e:
        raise InputError(f"unsupported safetensors dtype {name!r}") from e


def tensor_to_bytes(t: torch.Tensor) -> bytes:
    # Viewing storage as uint8 works for BF16/FP8 as well as NumPy-supported
    # dtypes, and preserves the exact little-endian bytes used by safetensors.
    flat = t.detach().cpu().contiguous().reshape(-1)
    return flat.view(torch.uint8).numpy().tobytes()
