"""Native INT8, INT4, FP8 and Kitchen layout encoding."""

from __future__ import annotations

import torch
from typing import Any, Optional

from native_quantizer.codecs.int8 import quantize_int8_tensorwise_weight
from native_quantizer.codecs.packing import unpack_int4_signed
from native_quantizer.codecs.rotation import _native_convrot, build_hadamard, rotate_weight
from native_quantizer.codecs.w4a4 import quantize_rotated_int4
from native_quantizer.formats import NativeFormat
from native_quantizer.identity import FORMAT_W4A4


def _native_encode(w: torch.Tensor, spec: NativeFormat, scale: Optional[torch.Tensor],
                   convrot: int, kitchen: Any = None, rotation_backend: str = "matrix"):
    if spec.cli == 'int8':
        q, s = quantize_int8_tensorwise_weight(w)
        return {'': q, '_scale': s}, q.float() * s
    if spec.algorithm == FORMAT_W4A4 or spec.cli == 'int8-convrot':
        rotate = (lambda x: rotate_weight(x, build_hadamard(convrot, device=x.device), convrot)) if rotation_backend == 'matrix' else (lambda x: _native_convrot(x, convrot))
        rotated = rotate(w)
        if spec.cli == 'int8-convrot':
            q, s = quantize_int8_tensorwise_weight(rotated)
            return {'': q, '_scale': s}, rotate(q.float() * s)
        q, s = quantize_rotated_int4(rotated)
        return {'': q, '_scale': s}, rotate(unpack_int4_signed(q).float() * s[:, None])
    if spec.cli.startswith('fp8-'):
        dtype = torch.float8_e4m3fn if spec.cli == 'fp8-e4m3' else torch.float8_e5m2
        q = (w / scale).clamp(-torch.finfo(dtype).max, torch.finfo(dtype).max).to(dtype)
        return {'': q}, q.float() * scale
    q, params = kitchen.quantize(w, scale=scale) if spec.cli == 'nvfp4' else kitchen.quantize(w)
    return kitchen.state_dict_tensors(q, params), kitchen.dequantize(q, params).float()
