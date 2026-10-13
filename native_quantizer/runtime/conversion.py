"""Resolve requested conversion backends without silent device substitution."""
from __future__ import annotations

import torch

from native_quantizer.errors import RuntimeCompatibilityError


def conversion_device(requested: str) -> tuple[torch.device, str]:
    if requested == 'cpu':
        return torch.device('cpu'), 'cpu'
    available = torch.cuda.is_available()
    gpu_backend = 'rocm' if getattr(torch.version, 'hip', None) else 'cuda'
    if requested == 'auto':
        return (torch.device('cuda'), gpu_backend) if available else (torch.device('cpu'), 'cpu')
    if requested not in ('cuda', 'rocm'):
        raise RuntimeCompatibilityError(f'unknown conversion device {requested!r}')
    if not available or requested != gpu_backend:
        raise RuntimeCompatibilityError(f'--device {requested} requested but unavailable; use --device cpu or auto')
    return torch.device('cuda'), gpu_backend
