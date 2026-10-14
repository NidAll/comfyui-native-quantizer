"""Encoding and sampling within a per-tensor working-memory budget."""

from __future__ import annotations

import math
import torch
from typing import Any, Dict, List, Optional, Tuple

from native_quantizer.checkpoint_io.reader import CheckpointReader
from native_quantizer.checkpoint_io.types import TensorMeta
from native_quantizer.codecs.dispatch import quantize_weight_by_format
from native_quantizer.codecs.rotation import build_hadamard, rotate_int8_convrot_weight
from native_quantizer.codecs.w4a8 import (
    _fixed_w4a8_codebook,
    assign_codes,
    assign_grid,
    choose_w4a8_codebook,
    fit_codebook,
)
from native_quantizer.dtypes import FP8_DTYPES
from native_quantizer.errors import InputError, PolicyError, UsageError
from native_quantizer.formats import format_scale_suffixes
from native_quantizer.identity import (
    CODEBOOK_DECISION_SAMPLE_COUNT,
    CODEBOOK_MODES,
    CODEBOOK_SAMPLE_COUNT,
    FORMAT_W4A8,
    FORMAT_W6A8,
    W4A8_ALS_ITERS,
)
from native_quantizer.logging import log
from native_quantizer.utils import human_bytes


QUANT_WORK_BYTES_PER_ELEMENT = 48


W6A8_WORK_BYTES_PER_ELEMENT = 160


MIN_CHUNK_MEMORY = 32 * 1024 * 1024


def _work_element_bytes(fmt, optimization=None):
    if optimization is not None and optimization.enabled:
        return 256
    return W6A8_WORK_BYTES_PER_ELEMENT if fmt == FORMAT_W6A8 else QUANT_WORK_BYTES_PER_ELEMENT


def _quant_work_bytes(meta: TensorMeta, fmt: str = FORMAT_W4A8, optimization=None) -> int:
    return math.prod(meta.shape) * _work_element_bytes(fmt, optimization)


def _chunk_rows_for_budget(k: int, n: int, max_mem: int, fmt: str = FORMAT_W4A8, optimization=None) -> int:
    if max_mem < MIN_CHUNK_MEMORY:
        raise PolicyError(
            f"--max-memory must be at least {human_bytes(MIN_CHUNK_MEMORY)} for "
            "bounded grouped quantization")
    row_work = k * _work_element_bytes(fmt, optimization)
    if row_work > max_mem:
        raise PolicyError(
            f"--max-memory {human_bytes(max_mem)} cannot hold one {k}-element "
            f"working row (estimated {human_bytes(row_work)})")
    return max(1, min(n, max_mem // row_work))


def _codebook_sample_size(total_elements: int) -> int:
    """Canonical Lloyd-Max sample count. Independent of the memory budget so
    that chunking never changes the quantization algorithm."""
    return min(CODEBOOK_SAMPLE_COUNT, total_elements)


def _quantize_rotated_w4a8_with_codebook(weight: torch.Tensor, group_size: int,
                                         codebook: torch.Tensor,
                                         scale_dtype: torch.dtype = torch.float8_e4m3fn,
                                         als_iterations: int = W4A8_ALS_ITERS,
                                         optimization=None, activations=None, convrot_groupsize=256,
                                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """W4A8 quantization of a rotated weight chunk with a PRE-FIT codebook."""
    if weight.ndim != 2:
        raise PolicyError(
            f"rotated W4A8 weight must be 2D, got {tuple(weight.shape)}")
    if als_iterations < 0:
        raise PolicyError("W4A8 ALS iteration count cannot be negative")
    n, k = weight.shape
    groups = k // group_size
    grouped = weight.float().view(n, groups, group_size)
    group_scale = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    if not bool(torch.isfinite(group_scale).all()):
        raise PolicyError("cannot quantize a weight containing NaN or Inf")
    quantized = assign_codes(grouped / group_scale, codebook)
    for _ in range(als_iterations):
        qc = codebook[quantized]
        group_scale = ((grouped * qc).sum(-1, keepdim=True)
                       / (qc * qc).sum(-1, keepdim=True).clamp(min=1e-8)).clamp(min=1e-8)
        quantized = assign_codes(grouped / group_scale, codebook)
    shifted = codebook[quantized] * group_scale
    s_channel = (shifted.abs().amax(dim=(1, 2)) / 127.0).clamp(min=1e-8)
    s_rel = (group_scale.squeeze(-1) / s_channel.unsqueeze(1)).float().contiguous()
    if scale_dtype != torch.float32:
        s_rel = s_rel.to(scale_dtype).contiguous()
    levels = (codebook.view(1, 1, 16) * s_rel.float().unsqueeze(-1)).round_().clamp_(-127, 127)
    unsigned = assign_grid(grouped, levels, s_channel).view(n, k)
    packed = ((unsigned[:, 0::2] & 0xF) | ((unsigned[:, 1::2] & 0xF) << 4)).to(torch.int8).contiguous()
    if optimization is not None and optimization.enabled:
        from native_quantizer.codecs.optimization import optimize_grouped
        rotate = lambda x: rotate_int8_convrot_weight(x, convrot_groupsize)
        out = optimize_grouped(weight, {'': packed, '_s_rel': s_rel, '_s_channel': s_channel,
                                       '_codebook': codebook}, FORMAT_W4A8, group_size,
                               optimization, activations, rotate)
        return out[''], out['_s_rel'], out['_s_channel'], codebook
    return packed, s_rel, s_channel.float().contiguous(), codebook


def _gather_codebook_samples(reader: CheckpointReader, name: str, k: int,
                             group_size: int, convrot_groupsize: int,
                             sample_size: int, chunk_rows: int,
                             device: Any = "cpu",
                             compute_dtype: Optional[torch.dtype] = None,
                             seed: int = 0,
                             codebook_mode: str = "auto") -> torch.Tensor:
    """Choose a codebook without exceeding the row-chunk memory budget.

    ``fixed`` performs no source read. ``auto`` rotates only the same evenly
    spaced rows inspected by the in-memory adaptive gate. ``fit`` preserves
    the legacy seeded flattened sample, but uses sorted slice boundaries rather
    than rescanning the entire index vector for every row chunk.
    """
    if codebook_mode not in CODEBOOK_MODES:
        raise PolicyError(
            f"unknown W4A8 codebook mode {codebook_mode!r}; "
            f"expected one of {CODEBOOK_MODES}")
    if codebook_mode == "fixed":
        return _fixed_w4a8_codebook("cpu")
    meta = reader.info.by_name(name)
    if meta is None:
        raise PolicyError(f"codebook sampling references missing tensor {name!r}")
    n = int(meta.shape[0])
    total = n * k
    if codebook_mode == "auto":
        max_rows = max(1, CODEBOOK_DECISION_SAMPLE_COUNT // max(k, 1))
        if n <= max_rows:
            idx = torch.arange(n, device="cpu")
        else:
            idx = torch.linspace(0, n - 1, max_rows, device="cpu").long()
        order = None
        flat_indices = False
    elif total > sample_size:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randint(0, total, (sample_size,), device="cpu", generator=gen)
        order = torch.argsort(idx, stable=True)
        idx = idx[order]
        flat_indices = True
    else:
        idx = torch.arange(total, device="cpu")
        order = None
        flat_indices = True
    work_dtype = compute_dtype or torch.float32
    h = build_hadamard(convrot_groupsize, device="cpu", dtype=work_dtype)
    h_t = h.T
    sample_parts: List[torch.Tensor] = []
    sorted_samples = (torch.empty(idx.numel(), dtype=torch.float32)
                      if flat_indices else None)
    n_conv_groups = k // convrot_groupsize
    n_quant_groups = k // group_size
    for r0 in range(0, n, chunk_rows):
        r1 = min(n, r0 + chunk_rows)
        start = r0 * k if flat_indices else r0
        end = r1 * k if flat_indices else r1
        lo = int(torch.searchsorted(
            idx, torch.tensor(start, dtype=idx.dtype), right=False).item())
        hi = int(torch.searchsorted(
            idx, torch.tensor(end, dtype=idx.dtype), right=False).item())
        if lo == hi:
            continue
        if flat_indices:
            chunk = reader.read_tensor(name)[r0:r1].to(work_dtype)
        else:
            local_rows = (idx[lo:hi] - r0).to(torch.long)
            chunk = reader.read_tensor(name)[r0:r1].index_select(
                0, local_rows).to(work_dtype)
        rot = torch.matmul(
            chunk.view(chunk.shape[0], n_conv_groups, convrot_groupsize), h_t
        ).reshape(chunk.shape[0], k)
        # per-group normalization (identical to the in-memory reference path)
        grouped = rot.float().view(chunk.shape[0], n_quant_groups, group_size)
        gs = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        if not bool(torch.isfinite(gs).all()):
            raise PolicyError(
                f"cannot quantize {name!r}: weight contains NaN or Inf")
        normalized = grouped / gs
        if flat_indices:
            local = idx[lo:hi] - start
            sorted_samples[lo:hi] = normalized.reshape(-1)[local]
        else:
            sample_parts.append(normalized)
    if flat_indices:
        if sorted_samples is None or sorted_samples.numel() == 0:
            raise PolicyError("codebook sampling produced no elements")
        if order is not None:
            samples = torch.empty_like(sorted_samples)
            samples[order] = sorted_samples
        else:
            samples = sorted_samples
        return fit_codebook(samples, levels=16, iterations=25,
                            sample_size=sample_size, seed=seed).contiguous()
    if not sample_parts:
        raise PolicyError("codebook sampling produced no elements")
    normalized_rows = torch.cat(sample_parts, dim=0)
    return choose_w4a8_codebook(
        normalized_rows, mode="auto", seed=seed).cpu().contiguous()


def quantize_tensor_bounded(reader: CheckpointReader, name: str, fmt: str,
                            group_size: int, convrot_groupsize: int,
                            max_mem: int, device: Any,
                            compute_dtype: Optional[torch.dtype] = None,
                            seed: int = 0,
                            codebook_mode: str = "auto",
                            optimization=None, activations=None,
                            ) -> Dict[str, torch.Tensor]:
    """Quantize one tensor with a bounded working set (chunked when needed).

    compute_dtype: fp16/bf16 lowers the precision of the rotation + fit math
    (deviation from the fp32 reference path; recorded in metadata).
    """
    device = torch.device(device)
    meta = reader.info.by_name(name)
    n, k = int(meta.shape[0]), int(meta.shape[1])
    full_bytes = meta.nbytes
    work_bytes = _quant_work_bytes(meta, fmt, optimization)
    if work_bytes <= max_mem:
        w = reader.read_tensor(name)
        w = w.to(device=device, dtype=(compute_dtype
                 if compute_dtype is not None and w.dtype not in FP8_DTYPES else w.dtype))
        try:
            try:
                out = quantize_weight_by_format(w, fmt, group_size,
                                                convrot_groupsize, seed=seed,
                                                codebook_mode=codebook_mode,
                                                optimization=optimization, activations=activations)
            except PolicyError as exc:
                if "NaN or Inf" in str(exc) or "non-finite" in str(exc):
                    raise InputError(
                        f"non-finite values in quantizable layer {name!r} "
                        "(NaN/Inf); pass --nonfinite-policy keep to preserve "
                        "the layer at original precision") from exc
                raise
        finally:
            del w
        if device.type != "cpu":
            out = {kk: vv.cpu() for kk, vv in out.items()}
        return out

    log().info("chunked quantization for %s (%s > budget %s)",
               name, human_bytes(full_bytes), human_bytes(max_mem))
    chunk_rows = _chunk_rows_for_budget(k, n, max_mem, fmt, optimization)
    if fmt != FORMAT_W4A8:
        # rowwise formats chunk exactly: no codebook, no cross-row state
        parts: Dict[str, List[torch.Tensor]] = {
            s: [] for s in format_scale_suffixes(fmt)}
        for r0 in range(0, n, chunk_rows):
            r1 = min(n, r0 + chunk_rows)
            chunk = reader.read_tensor(name)[r0:r1]
            if compute_dtype is not None and chunk.dtype not in FP8_DTYPES:
                chunk = chunk.to(compute_dtype)
            if device.type == "cuda":
                chunk = chunk.to(device)
            try:
                part = quantize_weight_by_format(
                    chunk, fmt, group_size, convrot_groupsize, seed=seed,
                    codebook_mode=codebook_mode, optimization=optimization, activations=activations)
            except PolicyError as exc:
                if "NaN or Inf" in str(exc) or "non-finite" in str(exc):
                    raise InputError(
                        f"non-finite values in quantizable layer {name!r} "
                        "(NaN/Inf); pass --nonfinite-policy keep to preserve "
                        "the layer at original precision") from exc
                raise
            for s in parts:
                t = part[s]
                parts[s].append(t.cpu() if device.type != "cpu" else t)
            del chunk, part
        return {s: torch.cat(v, dim=0).contiguous() for s, v in parts.items()}
    sample_size = _codebook_sample_size(n * k)
    codebook = _gather_codebook_samples(reader, name, k, group_size,
                                        convrot_groupsize, sample_size, chunk_rows,
                                        device="cpu", compute_dtype=compute_dtype,
                                        seed=seed, codebook_mode=codebook_mode)
    # per-chunk processing
    packed_parts: List[torch.Tensor] = []
    s_rel_parts: List[torch.Tensor] = []
    s_ch_parts: List[torch.Tensor] = []
    for r0 in range(0, n, chunk_rows):
        r1 = min(n, r0 + chunk_rows)
        chunk = reader.read_tensor(name)[r0:r1]
        if compute_dtype is not None and chunk.dtype not in FP8_DTYPES:
            chunk = chunk.to(compute_dtype)
        if device.type == "cuda":
            chunk = chunk.to(device)
        rot = rotate_int8_convrot_weight(chunk, convrot_groupsize)
        p, s_rel, s_ch, _ = _quantize_rotated_w4a8_with_codebook(
            rot, group_size, codebook.to(device=rot.device),
            als_iterations=(3 if codebook_mode == "fit" else W4A8_ALS_ITERS),
            optimization=optimization, activations=activations, convrot_groupsize=convrot_groupsize)
        packed_parts.append(p.cpu() if device.type != "cpu" else p)
        s_rel_parts.append(s_rel.cpu() if device.type != "cpu" else s_rel)
        s_ch_parts.append(s_ch.cpu() if device.type != "cpu" else s_ch)
        del chunk, rot, p, s_rel, s_ch
    packed = torch.cat(packed_parts, dim=0).contiguous()
    s_rel = torch.cat(s_rel_parts, dim=0).contiguous()
    s_ch = torch.cat(s_ch_parts, dim=0).contiguous()
    return {"": packed, "_s_rel": s_rel, "_s_channel": s_ch, "_codebook": codebook.contiguous()}


def _quantize_row_chunk(reader: CheckpointReader, name: str, r0: int, r1: int,
                        group_size: int, convrot_groupsize: int,
                        codebook: torch.Tensor, device: torch.device,
                        compute_dtype: Optional[torch.dtype],
                        als_iterations: int = W4A8_ALS_ITERS,
                        optimization=None, activations=None,
                        ) -> Dict[str, torch.Tensor]:
    chunk = reader.read_tensor(name)[r0:r1]
    if compute_dtype is not None and chunk.dtype not in FP8_DTYPES:
        chunk = chunk.to(compute_dtype)
    if device.type == "cuda":
        chunk = chunk.to(device)
    rotated = rotate_int8_convrot_weight(chunk, convrot_groupsize)
    try:
        packed, s_rel, s_ch, cb = _quantize_rotated_w4a8_with_codebook(
            rotated, group_size, codebook.to(device=rotated.device),
            als_iterations=als_iterations, optimization=optimization, activations=activations,
            convrot_groupsize=convrot_groupsize)
    except PolicyError as exc:
        if "NaN or Inf" in str(exc) or "non-finite" in str(exc):
            raise InputError(
                f"non-finite values in quantizable layer {name!r} (NaN/Inf); "
                "pass --nonfinite-policy keep to preserve the layer at "
                "original precision") from exc
        raise
    out = {
        "": packed.cpu(),
        "_s_rel": s_rel.cpu(),
        "_s_channel": s_ch.cpu(),
        "_codebook": cb.cpu(),
    }
    del chunk, rotated, packed, s_rel, s_ch, cb
    return out


def _native_rows(reader: CheckpointReader, meta: TensorMeta, budget: int,
                 device: Any = 'cpu', optimization=None):
    """FP32 arithmetic has an explicit conservative temporary allocation budget."""
    n, k = meta.shape
    per_row = max(1, k * (256 if optimization is not None and optimization.enabled else 96))
    if per_row > budget:
        raise UsageError(f'{meta.name}: one row needs about {human_bytes(per_row)}; increase --max-memory')
    rows = max(1, min(n, budget // per_row))
    for start in range(0, n, rows):
        yield reader.read_tensor(meta.name)[start:min(start + rows, n)].to(
            device=device, dtype=torch.float32, copy=True)
