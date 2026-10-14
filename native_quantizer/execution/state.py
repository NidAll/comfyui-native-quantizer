"""Conversion journals and resume-compatible plan identities."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from native_quantizer.checkpoint_io.types import CheckpointInfo
from native_quantizer.dtypes import torch_dtype_name
from native_quantizer.files import sha256_file
from native_quantizer.identity import (
    COMPONENT_AUTO,
    FORMAT_W6A8,
    FORMAT_W6A8_REVISION,
    QUANT_ALGORITHM_REVISION,
    format_revision,
    get_converter_version,
)
from native_quantizer.planning.types import ConversionPlan
from native_quantizer.runtime.kitchen import kitchen_version
from native_quantizer.utils import json_dumps
from native_quantizer.optimization import OptimizationConfig


@dataclass
class ConversionState:
    version: int = 2
    plan_hash: str = ""
    output: str = ""
    tmp: str = ""
    fmt: str = ""
    entries: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # name -> {offset,size,status}
    temp_device: Optional[int] = None
    temp_inode: Optional[int] = None
    done: bool = False


def _plan_hash(info: CheckpointInfo, plan: ConversionPlan, args: Any,
               source_hashes: Dict[str, str]) -> str:
    h = hashlib.sha256()
    options = {
        "converter_version": get_converter_version(),
        "format": plan.fmt,
        "format_revision": format_revision(plan.fmt),
        "algorithm_revision": QUANT_ALGORITHM_REVISION,
        "architecture": plan.detection.architecture,
        "prefix": plan.detection.unet_prefix,
        "input_key_mapping": getattr(info, "_source_names", {}),
        "architecture_policy": plan.detection.policy.to_dict(),
        "components": getattr(args, "components", COMPONENT_AUTO),
        "codebook_mode": getattr(args, "codebook_mode", "auto"),
        "group_size": getattr(args, "group_size", None),
        "compute_dtype": getattr(args, "compute_dtype", "auto"),
        "output_dtype": getattr(args, "output_dtype", "auto"),
        "min_numel": getattr(args, "min_numel_override", None),
        "seed": getattr(args, "seed", 0),
        "device": getattr(args, "device", "auto"),
        "max_memory": getattr(args, "max_memory", None),
        "include": getattr(args, "include", []),
        "exclude": getattr(args, "exclude", []),
        "keep_precision": getattr(args, "keep_precision", []),
        "sensitivity_threshold": getattr(args, "sensitivity_threshold", None),
        "error_threshold": getattr(args, "error_threshold", None),
        "calibration_source": getattr(args, "calibration_source", None),
        "calibration_samples": getattr(args, "calibration_samples", None),
        "calibration_cache": getattr(args, "calibration_cache", None),
        "profile": getattr(args, "profile", "auto"),
        "target_runtime": getattr(args, "target_runtime", "auto"),
        "quality_gate": getattr(args, "quality_gate", None),
        "global_error_gate": getattr(args, "global_error_gate", None),
        "max_linear_bytes_per_param": getattr(args, "max_linear_bytes_per_param", None),
        "max_bf16_fraction": getattr(args, "max_bf16_fraction", None),
        "w4a4_linear_dtype": getattr(args, "w4a4_linear_dtype", "int8"),
        "strip_gpu_identity": getattr(args, "strip_gpu_identity", False),
        "runtime_backend": getattr(args, "_effective_runtime", "auto"),
        "runtime_gpu": (getattr(args, "_runtime_gpu_name", None)
                        or getattr(args, "_effective_runtime", None)),
        "runtime_certificate_sha256": (
            sha256_file(args.runtime_certificate)
            if getattr(args, "runtime_certificate", None) else None),
        "disable_w4a4": getattr(args, "disable_w4a4", False),
        "disable_w4a8": getattr(args, "disable_w4a8", False),
        "disable_int8": getattr(args, "disable_int8", False),
        "source_sha256": source_hashes,
        "decisions": [{
            "name": d.name, "kind": d.kind.value, "layer": d.layer,
            "group_size": d.group_size,
            "convrot_groupsize": d.convrot_groupsize,
            "format": d.format,
            "component": d.component,
            "out_dtype": torch_dtype_name(d.out_dtype) if d.out_dtype else None,
        } for d in plan.decisions],
        "entries": [{
            "name": e["name"], "dtype": torch_dtype_name(e["dtype"]),
            "shape": list(e["shape"]), "nbytes": e["nbytes"],
        } for e in plan.output_entries],
    }
    if plan.fmt == FORMAT_W6A8 or getattr(args, "enable_w6a8", False):
        options.update(w6a8_group_size=getattr(args, "w6a8_group_size", 16),
                       enable_w6a8=getattr(args, "enable_w6a8", False),
                       kitchen_version=kitchen_version(), w6a8_revision=FORMAT_W6A8_REVISION)
    optimization = OptimizationConfig.from_options(args)
    if optimization.enabled:
        options['optimization'] = optimization.to_dict()
        options['optimization_calibration_digest'] = getattr(args, '_optimization_calibration_digest', None)
    h.update(json_dumps(options).encode("utf-8"))
    return h.hexdigest()
