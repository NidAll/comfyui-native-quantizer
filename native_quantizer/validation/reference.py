"""Independent reference reconstruction used by validation."""

from __future__ import annotations

import math
import torch

from native_quantizer.codecs.w6a8 import decode_w6a8_runtime_weight, validate_w6_tensors


def reference_hadamard(size: int) -> torch.Tensor:
    """INDEPENDENT regular-Hadamard construction for the validation-only
    reference decoders. Deliberately shares no code with build_hadamard:
    the point is that a bug in the production path cannot hide inside the
    validator. Slow, unnormalized-cached, fine for validation."""
    h4 = torch.tensor([[1, 1, 1, -1],
                       [1, 1, -1, 1],
                       [1, -1, 1, 1],
                       [-1, 1, 1, 1]], dtype=torch.float32)
    h = h4
    current = 4
    while current < size:
        h = torch.kron(h, h4)
        current *= 4
    return h / math.sqrt(size)


def _reference_rotate(weight: torch.Tensor, convrot_groupsize: int) -> torch.Tensor:
    """W @ H^T per ConvRot group (plain matmul; no fast paths)."""
    h = reference_hadamard(convrot_groupsize).to(weight.device)
    n, k = weight.shape
    groups = k // convrot_groupsize
    return torch.matmul(weight.float().reshape(n, groups, convrot_groupsize),
                        h.T).reshape(n, k)


def reference_decode_w4a8(packed: torch.Tensor, s_rel: torch.Tensor,
                          s_channel: torch.Tensor, codebook: torch.Tensor,
                          group_size: int, convrot_groupsize: int,
                          output_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Independent W4A8 decode from the format spec: even column = low nibble
    (unsigned 4-bit code), codebook lookup, per-group fp8 scale, rowwise
    fp32 channel scale, round to int8, inverse ConvRot. Validation only."""
    n, k_half = packed.shape
    k = k_half * 2
    groups = k // group_size
    p = packed.to(torch.int32) & 0xFF
    codes = torch.empty(n, k, dtype=torch.int32, device=packed.device)
    codes[:, 0::2] = p & 0xF
    codes[:, 1::2] = (p >> 4) & 0xF
    values = codebook.to(device=packed.device, dtype=torch.float32)[codes]
    values = values.view(n, groups, group_size) * s_rel.float().unsqueeze(-1)
    int8_weight = values.view(n, k).round().clamp_(-127, 127).to(torch.int8)
    weight_rotated = (int8_weight.float().view(n, groups, group_size)
                      * s_channel.float().view(n, 1, 1)).view(n, k)
    return _reference_rotate(weight_rotated, convrot_groupsize).to(output_dtype)


def reference_decode_w4a4(packed: torch.Tensor, scale: torch.Tensor,
                          convrot_groupsize: int,
                          output_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Independent W4A4 decode: SIGNED nibbles ([-8, 7], low = even column),
    rowwise absmax/7 scale, inverse ConvRot. Validation only."""
    n, k_half = packed.shape
    k = k_half * 2
    p = packed.to(torch.int32) & 0xFF
    lo = p & 0xF
    hi = (p >> 4) & 0xF
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    codes = torch.empty(n, k, dtype=torch.float32, device=packed.device)
    codes[:, 0::2] = lo
    codes[:, 1::2] = hi
    w_rot = codes * scale.to(device=packed.device, dtype=torch.float32).reshape(-1, 1)
    return _reference_rotate(w_rot, convrot_groupsize).to(output_dtype)


def reference_decode_int8(q: torch.Tensor, scale: torch.Tensor,
                          output_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Independent rowwise INT8 decode: q.float() * scale [N, 1]. Validation
    only (matches the eager backend broadcast exactly)."""
    return (q.float() * scale.to(device=q.device, dtype=torch.float32).reshape(-1, 1)
            ).to(output_dtype)


def reference_decode_w6a8(packed, s_rel, s_channel, group_size=16, convrot_groupsize=256):
    validate_w6_tensors({"": packed, "_s_rel": s_rel, "_s_channel": s_channel}, group_size, convrot_groupsize)
    rotated = decode_w6a8_runtime_weight(packed, s_rel, group_size).float() * s_channel.float()[:, None]
    return _reference_rotate(rotated, convrot_groupsize)
