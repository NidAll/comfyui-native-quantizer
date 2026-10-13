"""Uniform six-bit packing, shape checks and reconstruction."""

from __future__ import annotations

import torch

from native_quantizer.codecs.rotation import build_hadamard, rotate_weight
from native_quantizer.dtypes import FLOAT_DTYPES
from native_quantizer.errors import PolicyError


def validate_w6_shape(k: int, group_size: int, convrot_groupsize: int = 256):
    if type(group_size) is not int or group_size <= 0 or group_size % 16:
        raise PolicyError("W6A8 group size must be a positive multiple of 16")
    if k <= 0 or k % group_size or k % 256 or convrot_groupsize != 256:
        raise PolicyError(f"W6A8 requires K divisible by 256 and group_size={group_size}, with ConvRot 256 (K={k})")


def w6_weight_is_quantizable(shape, dtype, group_size, convrot_groupsize):
    if len(shape) != 2 or dtype not in FLOAT_DTYPES or shape[0] <= 0:
        return False, "requires a nonempty 2D float weight"
    try:
        validate_w6_shape(int(shape[1]), group_size, convrot_groupsize)
    except PolicyError as exc:
        return False, str(exc)
    return True, ""


def unpack_w6(packed: torch.Tensor) -> torch.Tensor:
    """Independent low-nibble plane followed by high-two-bit plane decoder."""
    if packed.ndim != 2 or packed.dtype != torch.int8 or packed.shape[1] <= 0 or packed.shape[1] % 3:
        raise PolicyError("W6A8 packed weight must be int8 [N, 3K/4] with integral K")
    n, width = packed.shape
    k = width * 4 // 3
    raw = packed.to(torch.int32) & 255
    low = raw[:, :k // 2]
    high = raw[:, k // 2:]
    codes = torch.empty((n, k), dtype=torch.int32, device=packed.device)
    codes[:, 0::2] = low & 15
    codes[:, 1::2] = (low >> 4) & 15
    for offset in range(4):
        codes[:, offset::4] |= ((high >> (2 * offset)) & 3) << 4
    return codes


def pack_w6(codes: torch.Tensor) -> torch.Tensor:
    if codes.ndim != 2 or codes.shape[1] % 4 or codes.dtype.is_floating_point or bool(((codes < 0) | (codes > 63)).any()):
        raise PolicyError("W6A8 codes must be integral [N,K], K divisible by 4, in [0,63]")
    codes = codes.to(torch.int32)
    low = (codes[:, 0::2] & 15) | ((codes[:, 1::2] & 15) << 4)
    high = torch.zeros_like(codes[:, 0::4])
    for offset in range(4):
        high |= ((codes[:, offset::4] >> 4) & 3) << (2 * offset)
    return torch.cat((low, high), dim=1).to(torch.int8)


def validate_w6_tensors(tensors, group_size, convrot_groupsize=256, source_shape=None):
    if "_codebook" in tensors or "_correction" in tensors:
        raise PolicyError("W6A8 forbids codebook and correction tensors")
    if not {"", "_s_rel", "_s_channel"} <= tensors.keys():
        raise PolicyError("W6A8 requires packed weight, FP8 group scales and FP32 channel scales")
    packed, rel, channel = (tensors[key] for key in ("", "_s_rel", "_s_channel"))
    if packed.ndim != 2 or packed.dtype != torch.int8 or packed.shape[1] % 3:
        raise PolicyError("malformed W6A8 packed width/dtype")
    n, width = packed.shape
    k = width * 4 // 3
    validate_w6_shape(k, group_size, convrot_groupsize)
    if n <= 0 or (source_shape is not None and tuple(source_shape) != (n, k)):
        raise PolicyError("W6A8 source and packed dimensions differ")
    if tuple(rel.shape) != (n, k // group_size) or rel.dtype != torch.float8_e4m3fn:
        raise PolicyError("malformed W6A8 FP8 group scales")
    if tuple(channel.shape) != (n,) or channel.dtype != torch.float32:
        raise PolicyError("malformed W6A8 FP32 channel scales")
    if not bool(torch.isfinite(rel.float()).all() and torch.isfinite(channel).all()) or bool((rel.float() < 0).any() or (channel <= 0).any()):
        raise PolicyError("non-finite or invalid W6A8 scales")
    return n, k


def decode_w6a8_runtime_weight(packed, s_rel, group_size):
    codes = unpack_w6(packed).float() - 32.0
    n, k = codes.shape
    values = codes.reshape(n, k // group_size, group_size) * s_rel.float().unsqueeze(-1)
    return values.reshape(n, k).round().clamp(-127, 127).to(torch.int8)


def dequantize_w6a8_weight(packed, s_rel, s_channel, group_size=16, convrot_groupsize=256):
    validate_w6_tensors({"": packed, "_s_rel": s_rel, "_s_channel": s_channel}, group_size, convrot_groupsize)
    rotated = decode_w6a8_runtime_weight(packed, s_rel, group_size).float() * s_channel.float()[:, None]
    h = build_hadamard(convrot_groupsize, device=rotated.device)
    return rotate_weight(rotated, h, convrot_groupsize)
