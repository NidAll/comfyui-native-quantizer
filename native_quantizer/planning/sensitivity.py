"""Sensitivity decisions frozen before output inventory creation."""

from __future__ import annotations

import torch
from typing import List, Optional

from native_quantizer.checkpoint_io.reader import CheckpointReader
from native_quantizer.checkpoint_io.types import CheckpointInfo
from native_quantizer.codecs.bounded import (
    _chunk_rows_for_budget,
    _codebook_sample_size,
    _gather_codebook_samples,
    _quant_work_bytes,
    _quantize_row_chunk,
    quantize_tensor_bounded,
)
from native_quantizer.codecs.dispatch import dequantize_weight_by_format
from native_quantizer.errors import PolicyError
from native_quantizer.identity import FORMAT_W4A8, FORMAT_W6A8, W4A8_ALS_ITERS
from native_quantizer.planning.metrics import SensitivityAnalyzer, _MetricAccumulator
from native_quantizer.planning.mixed import MixedPlanner
from native_quantizer.planning.types import DecisionKind, TensorDecision


def apply_sensitivity_prepass(info: CheckpointInfo,
                              decisions: List[TensorDecision],
                              analyzer: SensitivityAnalyzer,
                              max_mem: int, device: torch.device,
                              compute_dtype: Optional[torch.dtype],
                              seed: int = 0,
                              codebook_mode: str = "auto", optimization=None) -> None:
    """Freeze sensitivity decisions before the safetensors inventory is built."""
    with CheckpointReader(info) as reader:
        for decision in decisions:
            if decision.kind != DecisionKind.QUANTIZE:
                continue
            meta = info.by_name(decision.name)
            if meta is None:
                raise PolicyError(
                    f"sensitivity plan references missing tensor {decision.name!r}")
            activations = None
            if analyzer.calibration is not None:
                layer_stats = (analyzer.calibration.layers.get(decision.name)
                               or analyzer.calibration.layers.get(decision.layer))
                if layer_stats is not None:
                    activations = layer_stats["samples"]
            if decision.format == FORMAT_W6A8:
                scorer = MixedPlanner("conservative", analyzer.calibration, max_mem, device,
                                      compute_dtype, enable_w6a8=True,
                                      w6a8_group_size=decision.group_size, optimization=optimization)
                metrics = scorer._evaluate_format(reader, decision, FORMAT_W6A8)
                metrics.name = decision.name
                analyzer.results[decision.name] = metrics
            elif _quant_work_bytes(meta, optimization=optimization) <= max_mem:
                quantized = quantize_tensor_bounded(
                    reader, decision.name, FORMAT_W4A8, decision.group_size,
                    decision.convrot_groupsize, max_mem, device,
                    compute_dtype=compute_dtype, seed=seed,
                    codebook_mode=codebook_mode, optimization=optimization, activations=activations)
                dequant = dequantize_weight_by_format(
                    quantized, FORMAT_W4A8, decision.group_size,
                    decision.convrot_groupsize, torch.float32)
                metrics = analyzer.evaluate(
                    decision.name, reader.read_tensor(decision.name), dequant)
                del quantized, dequant
            else:
                n, k = int(meta.shape[0]), int(meta.shape[1])
                chunk_rows = _chunk_rows_for_budget(k, n, max_mem, optimization=optimization)
                sample_size = _codebook_sample_size(n * k)
                codebook = _gather_codebook_samples(
                    reader, decision.name, k, decision.group_size,
                    decision.convrot_groupsize, sample_size, chunk_rows,
                    compute_dtype=compute_dtype, seed=seed,
                    codebook_mode=codebook_mode)
                accumulator = _MetricAccumulator(decision.name)
                for r0 in range(0, n, chunk_rows):
                    r1 = min(n, r0 + chunk_rows)
                    quantized = _quantize_row_chunk(
                        reader, decision.name, r0, r1, decision.group_size,
                        decision.convrot_groupsize, codebook, device, compute_dtype,
                        als_iterations=(3 if codebook_mode == "fit"
                                        else W4A8_ALS_ITERS), optimization=optimization,
                        activations=activations)
                    dequant = dequantize_weight_by_format(
                        quantized, FORMAT_W4A8, decision.group_size,
                        decision.convrot_groupsize, torch.float32)
                    accumulator.update(reader.read_tensor(decision.name)[r0:r1],
                                       dequant, activations)
                    del quantized, dequant
                metrics = accumulator.finish()
                analyzer.results[decision.name] = metrics
            keep, reason = analyzer.decide_keep(metrics)
            metrics.kept = keep
            metrics.reason = reason
            if keep:
                decision.kind = DecisionKind.KEEP_PRECISION
                decision.reason = f"sensitivity fallback: {reason}"
