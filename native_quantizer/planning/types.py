"""Per-tensor decisions and conversion output inventories."""

from __future__ import annotations

import enum
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from native_quantizer.architectures.detection import DetectionResult
from native_quantizer.identity import FORMAT_W4A8


class DecisionKind(enum.Enum):
    QUANTIZE = "quantize"
    KEEP = "keep"              # passthrough at original precision
    KEEP_PRECISION = "keep_precision"   # kept for sensitivity / policy reasons


@dataclass
class TensorDecision:
    name: str                      # input tensor name (state-dict key)
    kind: DecisionKind
    reason: str
    layer: Optional[str] = None    # layer name (key minus ".weight") when quantized
    group_size: int = 16
    convrot_groupsize: int = 256
    format: str = FORMAT_W4A8      # per-layer format (mixed mode)
    linear_dtype: Optional[str] = None   # W4A4 execution variant (per decision)
    out_dtype: Optional[torch.dtype] = None   # passthrough cast target (if any)
    quantized: bool = False        # True once actually quantized
    component: str = "other"      # diffusion | text_encoder | other


@dataclass
class ConversionPlan:
    fmt: str
    detection: DetectionResult
    decisions: List[TensorDecision]
    metadata_quant: Dict[str, Any]     # _quantization_metadata payload
    metadata_ext: Dict[str, Any]       # comfy_wxa8 extension payload
    output_entries: List[Dict[str, Any]]  # (name, dtype, shape) in write order
    total_in_bytes: int = 0
    total_out_bytes: int = 0
    n_quantized: int = 0
    n_kept: int = 0
    device: str = "cpu"
    chunked_layers: set = field(default_factory=set)   # layers quantized via the
                                                       # bounded-memory chunked path
    mixed_plan: Optional[Dict[str, Any]] = None        # mixed-planner summary

    def quantized_layers(self) -> List[TensorDecision]:
        return [d for d in self.decisions if d.kind == DecisionKind.QUANTIZE]
