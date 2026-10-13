"""Local runtime diagnostics shared by CLI and optional frontend callers."""
from native_quantizer.capabilities import frontend_capabilities
from native_quantizer.formats import NATIVE_FORMATS
from native_quantizer.runtime.capabilities import inspect_environment, native_gpu_warning


def probe_runtime() -> dict:
    import torch

    env = inspect_environment()
    free = None
    if env.cuda_available:
        try:
            free = torch.cuda.mem_get_info()[0]
        except (RuntimeError, OSError):
            pass
    return {'capabilities': frontend_capabilities(), 'cuda_available': env.cuda_available,
            'gpu_backend': ('rocm' if getattr(torch.version, 'hip', None) else 'cuda') if env.cuda_available else None,
            'cuda_free_bytes': free, 'kitchen_available': env.has_comfy_kitchen,
            'kitchen_version': env.comfy_kitchen_rev,
            'format_warnings': {name: warning for name in NATIVE_FORMATS
                                if (warning := native_gpu_warning(name, 'nvidia', env.cuda_capability))}}
