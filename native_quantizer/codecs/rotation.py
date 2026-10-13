"""Regular Hadamard construction and weight/activation transforms."""

from __future__ import annotations

import torch
from typing import Any, Dict, Tuple

from native_quantizer.errors import PolicyError


_HADAMARD_CACHE: Dict[Tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def _is_power_of_four(value: int) -> bool:
    if value < 1:
        return False
    while value % 4 == 0:
        value //= 4
    return value == 1


def build_hadamard(size: int, device: Any = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Normalized REGULAR orthogonal Hadamard matrix (ConvRot), size = power of 4."""
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    key = (size, dev, dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached
    if size < 4 or not _is_power_of_four(size):
        raise PolicyError(f"Regular Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype, device=dev)
    h = h4
    current = 4
    while current < size:
        h = torch.kron(h, h4)
        current *= 4
    h_norm = h / (size ** 0.5)
    _HADAMARD_CACHE[key] = h_norm
    return h_norm


def rotate_weight(weight: torch.Tensor, h: torch.Tensor, group_size: int) -> torch.Tensor:
    """W_rot = W @ H_block^T (offline weight rotation)."""
    out_f, in_f = weight.shape
    if in_f % group_size != 0:
        raise PolicyError(f"in_features {in_f} not divisible by group_size {group_size}")
    n_groups = in_f // group_size
    grouped = weight.reshape(out_f, n_groups, group_size)
    h_t = h.T.to(dtype=weight.dtype, device=weight.device)
    return torch.matmul(grouped, h_t).reshape(out_f, in_f)


def rotate_activation(x: torch.Tensor, h: torch.Tensor, group_size: int) -> torch.Tensor:
    """x_rot = x @ H_block (online activation rotation)."""
    orig_shape = x.shape
    features = orig_shape[-1]
    if features % group_size != 0:
        raise PolicyError(f"features {features} not divisible by group_size {group_size}")
    n_groups = features // group_size
    grouped = x.reshape(-1, n_groups, group_size)
    hh = h.to(dtype=x.dtype, device=x.device)
    return torch.matmul(grouped, hh).reshape(orig_shape)


def rotate_int8_convrot_weight(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """Portable ConvRot weight rotation (identical to comfy-kitchen)."""
    h = build_hadamard(group_size, device=weight.device, dtype=weight.dtype)
    return rotate_weight(weight, h, group_size)


def _native_convrot(weight: torch.Tensor, size: int) -> torch.Tensor:
    """Regular H4 Kronecker transform, O(K log K), with fixed elementwise reductions.

    This implements the same orthogonal transform as the runtime matrix, with
    FP32 rounding independent of the number of rows in each conversion chunk.
    """
    x = weight.reshape(-1, size)
    stride = 1
    while stride < size:
        v = x.reshape(-1, size // (4 * stride), 4, stride)
        a, b, c, d = v.unbind(2)
        x = torch.stack((a+b+c-d, a+b-c+d, a-b+c+d, -a+b+c+d), dim=2).reshape(-1, size) * 0.5
        stride *= 4
    return x.reshape_as(weight)
