"""Checkpoint inventory and tensor descriptions."""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from native_quantizer.errors import InputError


@dataclass
class TensorMeta:
    """Header-level information about one tensor (no data loaded)."""
    name: str
    dtype: torch.dtype
    shape: Tuple[int, ...]
    nbytes: int
    source: str          # file path
    offset: int          # byte offset within source file
    end: int             # exclusive byte end


@dataclass
class CheckpointInfo:
    """Everything known about an input checkpoint without loading data."""
    kind: str                     # "safetensors" | "sharded-safetensors" | "pickle"
    files: List[str]              # data files in logical order
    metadata: Dict[str, str]      # safetensors __metadata__ (or {})
    tensors: List[TensorMeta]     # logical tensor order
    config: Dict[str, Any] = field(default_factory=dict)   # merged HF-style config.json
    model_index: Dict[str, Any] = field(default_factory=dict)  # model_index.json
    shard_index: Dict[str, Any] = field(default_factory=dict)  # model.safetensors.index.json
    total_bytes: int = 0
    is_quantized_input: bool = False
    source_hashes: Dict[str, str] = field(default_factory=dict)
    input_warnings: List[str] = field(default_factory=list)
    _tensor_by_name: Dict[str, TensorMeta] = field(
        init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        for tensor in self.tensors:
            if tensor.name in self._tensor_by_name:
                raise InputError(f"duplicate tensor {tensor.name!r} in checkpoint inventory")
            self._tensor_by_name[tensor.name] = tensor

    def key_set(self) -> set:
        return set(self._tensor_by_name)

    def by_name(self, name: str) -> Optional[TensorMeta]:
        return self._tensor_by_name.get(name)
