"""Native layout names, output sizes, aliases and supported options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from native_quantizer.errors import PolicyError
from native_quantizer.identity import FORMAT_INT8, FORMAT_W4A4, FORMAT_W4A8, FORMAT_W6A8


def quantized_format_bytes(n: int, k: int, fmt: str, group_size: int = 16) -> int:
    """Exact serialized byte count of the per-layer format payloads."""
    if fmt == FORMAT_W6A8:
        return n * (3 * k // 4) + n * (k // group_size) + n * 4
    if fmt == FORMAT_W4A8:
        return n * (k // 2) + n * (k // 16) + n * 4 + 16 * 4
    if fmt == FORMAT_W4A4:
        return n * (k // 2) + n * 4
    if fmt == FORMAT_INT8:
        return n * k + n * 4
    raise PolicyError(f"unknown format {fmt!r}")


def format_scale_suffixes(fmt: str) -> Tuple[str, ...]:
    """Output suffixes per format ('' = the packed weight slot)."""
    if fmt == FORMAT_W6A8:
        return ("", "_s_rel", "_s_channel")
    if fmt == FORMAT_W4A8:
        return ("", "_s_rel", "_s_channel", "_codebook")
    if fmt in (FORMAT_W4A4, FORMAT_INT8):
        return ("", "_scale")
    raise PolicyError(f"unknown format {fmt!r}")


FORMAT_TO_KITCHEN_LAYOUT = {
    FORMAT_W4A8: "AsymW4A8Int8Layout",
    FORMAT_W6A8: "AsymW4A8Int8Layout",
    FORMAT_W4A4: "TensorCoreConvRotW4A4Layout",
    FORMAT_INT8: "TensorWiseINT8Layout",
}


@dataclass(frozen=True)
class NativeFormat:
    cli: str
    algorithm: Optional[str]
    description: str
    alignment: int = 1
    kitchen_class: Optional[str] = None


NATIVE_FORMATS = {
    f.cli: f for f in (
        NativeFormat('int8', FORMAT_INT8, 'Rowwise INT8, dynamic activation quantization'),
        NativeFormat('int8-convrot', FORMAT_INT8, 'Rowwise INT8 with 256-wide ConvRot', 256),
        NativeFormat('int4-convrot', FORMAT_W4A4, 'ConvRot packed INT4 with INT4 activation request', 64),
        NativeFormat('w4a4', FORMAT_W4A4, 'ConvRot packed 4-bit weights', 64),
        NativeFormat('fp8-e4m3', 'float8_e4m3fn', 'Tensor-scaled FP8 E4M3'),
        NativeFormat('fp8-e5m2', 'float8_e5m2', 'Tensor-scaled FP8 E5M2'),
        NativeFormat('nvfp4', 'nvfp4', 'NVFP4 with native swizzled block scales', 16,
                     'TensorCoreNVFP4Layout'),
        NativeFormat('mxfp8', 'mxfp8', 'MXFP8 with native E8M0 block scales', 32,
                     'TensorCoreMXFP8Layout'),
        NativeFormat('fp16', None, 'FP16 precision conversion'),
        NativeFormat('bf16', None, 'BF16 precision conversion'),
    )
}


NATIVE_UNSUPPORTED_OPTIONS = (
    'kernel_costs', 'resume', 'validation_only', 'metadata_only',
    'calibration_source', 'calibration_cache', 'calibration_samples',
    'require_calibration', 'runtime_certificate', 'require_runtime_certificate',
    'quality_gate', 'global_error_gate', 'max_linear_bytes_per_param',
    'max_bf16_fraction', 'disable_w4a4', 'disable_w4a8', 'disable_int8',
    'sensitivity_threshold',
)


FORMAT_ALIASES = {'int8_convrot': 'int8-convrot', 'int4_convrot': 'int4-convrot',
                  'int8_tensorwise': 'int8', 'convrot_w4a4': 'w4a4',
                  'asym_w4a8_int8': 'w4a8', 'w6a8_int8': 'w6a8', 'float8_e4m3fn': 'fp8-e4m3',
                  'float8_e5m2': 'fp8-e5m2'}


NATIVE_ENGINE_REVISION = 'native-stream-v2'
