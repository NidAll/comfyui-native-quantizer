"""Optional Comfy Kitchen version and layout API probes."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from native_quantizer.errors import RuntimeCompatibilityError
from native_quantizer.formats import NativeFormat


def kitchen_version() -> str:
    try:
        return importlib.metadata.version("comfy-kitchen")
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def w6a8_api_available() -> bool:
    """Static probe: the older W4A8 layout accepts and ignores extra kwargs."""
    try:
        dist = importlib.metadata.distribution("comfy-kitchen")
        version = tuple(int(v) for v in dist.version.split(".")[:3])
        source = Path(dist.locate_file("comfy_kitchen/tensor/w4a8_int8.py")).read_text()
        return version >= (0, 2, 36) and bool(re.search(r"bits\s*:\s*int\s*=", source)) and "scale_search" in source
    except (OSError, ValueError, importlib.metadata.PackageNotFoundError):
        return False


def require_w6a8_api():
    if not w6a8_api_available():
        raise RuntimeCompatibilityError("W6A8 requires comfy-kitchen >= 0.2.36 with the explicit bits and scale_search API")
    from comfy_kitchen.tensor.w4a8_int8 import AsymW4A8Int8Layout
    import inspect
    parameters = inspect.signature(AsymW4A8Int8Layout.quantize).parameters
    if not {"bits", "scale_search"} <= parameters.keys():
        raise RuntimeCompatibilityError("Kitchen W6A8 API missing; a W4A8 layout alone is insufficient")
    return AsymW4A8Int8Layout


def _native_kitchen(spec: NativeFormat):
    try:
        import comfy_kitchen.tensor as tensor_api
        cls = getattr(tensor_api, spec.kitchen_class)
    except (ImportError, AttributeError) as exc:
        raise RuntimeCompatibilityError(f'{spec.cli} requires a comfy-kitchen release exporting '
                                        f'{spec.kitchen_class}; install/update comfy-kitchen') from exc
    if not all(hasattr(cls, name) for name in ('quantize', 'dequantize', 'state_dict_tensors')):
        raise RuntimeCompatibilityError('unsupported comfy-kitchen layout API')
    return cls
