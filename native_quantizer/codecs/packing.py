"""Packed four-bit tensor decoding."""

from __future__ import annotations

import torch


def unpack_w4(packed: torch.Tensor) -> torch.Tensor:
    """int8 [N, K/2] packed int4 -> int32 codes [N, K] (even col = low nibble)."""
    n, k_half = packed.shape
    k = k_half * 2
    p = packed.to(torch.int32) & 0xFF
    out = torch.empty(n, k, dtype=torch.int32, device=packed.device)
    out[:, 0::2] = p & 0xF
    out[:, 1::2] = (p >> 4) & 0xF
    return out


def unpack_int4_signed(packed: torch.Tensor) -> torch.Tensor:
    """int8 [N, K/2] packed int4 -> int8 codes [N, K] with SIGNED nibble
    interpretation (even column = low nibble). Mirrors comfy-kitchen
    `_unpack_int4_row_major`; used by the W4A4 dequantization path."""
    n, k_half = packed.shape
    p = packed.to(torch.int32) & 0xFF
    lo = p & 0xF
    hi = (p >> 4) & 0xF
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    out = torch.empty(n, k_half * 2, dtype=torch.int8, device=packed.device)
    out[:, 0::2] = lo.to(torch.int8)
    out[:, 1::2] = hi.to(torch.int8)
    return out
