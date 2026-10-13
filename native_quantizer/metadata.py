"""Native layer metadata, output inventory and portable conversion provenance."""

from __future__ import annotations

import json
import torch
from typing import Any, Dict, Iterable, List, Optional, Tuple

from native_quantizer.architectures.detection import DetectionResult
from native_quantizer.calibration import CalibrationStats
from native_quantizer.checkpoint_io.fingerprints import _portable_file_labels, _portable_hash_manifest
from native_quantizer.checkpoint_io.reader import RawSafetensorsFile
from native_quantizer.checkpoint_io.types import CheckpointInfo
from native_quantizer.dtypes import FLOAT_DTYPES, tensor_nbytes, torch_dtype_name
from native_quantizer.errors import PolicyError
from native_quantizer.formats import FORMAT_TO_KITCHEN_LAYOUT
from native_quantizer.identity import (
    COMFYUI_BASE,
    COMFYUI_PR,
    COMFYUI_PR_HEAD,
    COMFY_KITCHEN_REV,
    COMPONENT_AUTO,
    COMPONENT_DIFFUSION,
    COMPONENT_TEXT_ENCODER,
    CONVERTER_NAME,
    DEFAULT_W4A4_LINEAR_DTYPE,
    FORMAT_INT8,
    FORMAT_MIXED,
    FORMAT_W4A4,
    FORMAT_W4A8,
    FORMAT_W6A8,
    FORMAT_W6A8_REVISION,
    GROUPED_INT8_FORMATS,
    LAYER_CONF_KEY,
    METADATA_KEY_QUANT,
    MIXED_FORMATS,
    TRITON_MIN_VERSION,
    W4A4_QUANT_GROUP_SIZE,
    W4A8_KERNEL_MIN_SM,
    format_revision,
    get_algorithm_identity,
    get_converter_version,
)
from native_quantizer.planning.metrics import SensitivityAnalyzer
from native_quantizer.planning.types import ConversionPlan, DecisionKind, TensorDecision
from native_quantizer.runtime.capabilities import EnvironmentInfo
from native_quantizer.runtime.kitchen import kitchen_version
from native_quantizer.utils import json_dumps


def layer_quant_config(d: TensorDecision) -> Dict[str, Any]:
    if d.format in GROUPED_INT8_FORMATS:
        return {"format": d.format, "group_size": d.group_size,
                "convrot": True, "convrot_groupsize": d.convrot_groupsize}
    if d.format == FORMAT_W4A4:
        return {"format": d.format, "convrot_groupsize": d.convrot_groupsize,
                "quant_group_size": W4A4_QUANT_GROUP_SIZE,
                "linear_dtype": d.linear_dtype or DEFAULT_W4A4_LINEAR_DTYPE}
    if d.format == FORMAT_INT8:
        return {"format": d.format}
    raise PolicyError(f"unknown per-layer format {d.format!r}")


def inline_quant_metadata(layers: Iterable[str]) -> bool:
    # ComfyUI expands header metadata twice when stripping a model prefix.
    # Inline layer configurations travel with the weights through that stripping.
    return any(layer.startswith(("model.", "net.")) for layer in layers)


def serialized_quant_metadata(qm: Dict[str, Any]) -> Dict[str, Any]:
    return {"layers": {}} if inline_quant_metadata(qm["layers"]) else qm


def read_quant_metadata(path: str) -> Dict[str, Any]:
    """Read both native ComfyUI metadata transports, rejecting conflicting entries."""
    with RawSafetensorsFile(path) as raw:
        qm = json.loads(raw.metadata.get(METADATA_KEY_QUANT, "{}"))
        if not isinstance(qm, dict) or not isinstance(qm.get("layers"), dict):
            raise ValueError(f"{METADATA_KEY_QUANT} must contain a layers object")
        layers = dict(qm["layers"])
        for name, (dtype, shape, start, end) in raw.entries.items():
            if not name.endswith("." + LAYER_CONF_KEY):
                continue
            if dtype != torch.uint8 or len(shape) != 1 or end - start > 1024 * 1024:
                raise ValueError(f"{name}: invalid layer metadata tensor")
            conf = json.loads(raw.read_bytes(name).tobytes().decode("utf-8"))
            layer = name[:-len("." + LAYER_CONF_KEY)]
            if not layer or not isinstance(conf, dict):
                raise ValueError(f"{name}: malformed layer configuration")
            if layer in layers and layers[layer] != conf:
                raise ValueError(f"{name}: conflicts with header layer metadata")
            layers[layer] = conf
        return {**qm, "layers": layers}


def build_output_entries(info: CheckpointInfo, decisions: List[TensorDecision],
                         fmt: str, output_dtype: Optional[torch.dtype],
                         ) -> Tuple[List[Dict[str, Any]], int]:
    """Compute the exact output tensor inventory (name, dtype, shape, nbytes)
    in deterministic write order: quantized layers first (original weight slot),
    then passthrough tensors in input order."""
    if fmt not in (FORMAT_W4A8, FORMAT_W6A8, FORMAT_MIXED):
        raise PolicyError(f"unknown quantization format {fmt!r}")
    entries: List[Dict[str, Any]] = []
    total = 0
    seen = set()
    inline = inline_quant_metadata(d.layer for d in decisions
                                   if d.kind == DecisionKind.QUANTIZE)
    for d in decisions:
        if d.kind == DecisionKind.QUANTIZE:
            layer = d.layer
            meta = info.by_name(d.name)
            n, k = int(meta.shape[0]), int(meta.shape[1])
            base = f"{layer}.weight"
            if d.format == FORMAT_W6A8:
                payload = [(base, torch.int8, (n, 3 * k // 4)),
                           (f"{layer}.weight_s_rel", torch.float8_e4m3fn, (n, k // d.group_size)),
                           (f"{layer}.weight_s_channel", torch.float32, (n,))]
            elif d.format == FORMAT_W4A8:
                groups = k // d.group_size
                extras = [
                    (f"{layer}.weight_s_rel", torch.float8_e4m3fn, (n, groups)),
                    (f"{layer}.weight_s_channel", torch.float32, (n,)),
                    (f"{layer}.weight_codebook", torch.float32, (16,)),
                ]
                payload = [(base, torch.int8, (n, k // 2))] + extras
            elif d.format == FORMAT_W4A4:
                payload = [(base, torch.int8, (n, k // 2)),
                           (f"{layer}.weight_scale", torch.float32, (n,))]
            elif d.format == FORMAT_INT8:
                payload = [(base, torch.int8, (n, k)),
                           (f"{layer}.weight_scale", torch.float32, (n, 1))]
            else:
                raise PolicyError(
                    f"unknown per-layer format {d.format!r} for {d.name!r}")
            if inline:
                conf_bytes = json_dumps(layer_quant_config(d)).encode("utf-8")
                payload.append((f"{layer}.{LAYER_CONF_KEY}", torch.uint8, (len(conf_bytes),)))
            for ename, edtype, eshape in payload:
                if ename in seen:
                    raise PolicyError(f"duplicate output tensor {ename!r}")
                seen.add(ename)
                nb = tensor_nbytes(edtype, eshape)
                entries.append({"name": ename, "dtype": edtype, "shape": eshape, "nbytes": nb})
                total += nb
        else:
            meta = info.by_name(d.name)
            dt = meta.dtype
            if output_dtype is not None and dt in FLOAT_DTYPES:
                dt = output_dtype
            if d.name in seen:
                raise PolicyError(f"duplicate output tensor {d.name!r}")
            seen.add(d.name)
            nb = tensor_nbytes(dt, meta.shape)
            entries.append({"name": d.name, "dtype": dt, "shape": meta.shape, "nbytes": nb})
            total += nb
    return entries, total


def build_quant_metadata(info: CheckpointInfo, plan: ConversionPlan) -> Dict[str, Any]:
    """Official `_quantization_metadata` payload: {"layers": {layer: conf}}."""
    layers: Dict[str, Any] = {}
    for d in plan.quantized_layers():
        if d.layer is None:
            raise PolicyError(f"quantized tensor {d.name!r} has no layer name")
        layers[d.layer] = layer_quant_config(d)
    return {"layers": layers}


def _architecture_dims_fingerprint(info: CheckpointInfo,
                                    detection: DetectionResult) -> Dict[str, Any]:
    """Compact, bug-report-friendly fingerprint: the K values (column counts)
    of all 2D float weights under the detected prefix, capped histogram. Lets a
    user report reproduce the conversion without sharing a multi-GB model."""
    prefix = detection.unet_prefix
    if not any(k.startswith(prefix) for k in info.key_set()):
        prefix = ""
    counter: Dict[str, int] = {}
    for meta in info.tensors:
        if len(meta.shape) != 2 or meta.dtype not in FLOAT_DTYPES:
            continue
        if prefix and not meta.name.startswith(prefix):
            continue
        key = str(int(meta.shape[1]))
        counter[key] = counter.get(key, 0) + 1
    ordered = dict(sorted(counter.items(), key=lambda kv: -kv[1]))
    return {"dims_2d_k": ordered, "n_2d_float": sum(counter.values())}


def _w4a4_runtime_block(plan: ConversionPlan, args: Any) -> Dict[str, Any]:
    """Namespaced W4A4 runtime record. With --strip-gpu-identity the hardware
    identity fields (GPU name, compute capability, ROCm architecture) are
    omitted so the checkpoint can be published without revealing the target
    or conversion machine."""
    mp = plan.mixed_plan or {}
    strip = bool(getattr(args, "strip_gpu_identity", False))
    block = {
        "requested_linear_dtype": mp.get(
            "w4a4_linear_dtype", DEFAULT_W4A4_LINEAR_DTYPE),
        "target": mp.get("runtime_backend"),
        "effective_activation_bits": "per candidate (see report)",
        "certified": bool(mp.get("runtime", {}).get("runtime_certified")),
        "note": "custom fields live in the namespaced extension "
                "block only; official ComfyUI metadata stays native",
    }
    if not strip:
        runtime = mp.get("runtime") or {}
        block["gpu"] = runtime.get("gpu_name")
        block["cuda_capability"] = runtime.get("cuda_capability")
        block["rocm_arch"] = runtime.get("rocm_arch")
    return block


def build_extension_metadata(info: CheckpointInfo, plan: ConversionPlan,
                             env: EnvironmentInfo, args: Any,
                             calibration: Optional[CalibrationStats],
                             sensitivity: Optional[SensitivityAnalyzer],
                             input_hashes: Dict[str, str],
                             tensor_payload_sha256: str,
                             validation_summary: Dict[str, Any],
                             warnings: List[str]) -> Dict[str, Any]:
    """Namespaced extension metadata (never described as official ComfyUI keys)."""
    d = plan.detection
    quant_layers = plan.quantized_layers()
    is_mixed = plan.fmt == FORMAT_MIXED
    quant_block = None
    if is_mixed:
        # v2 schema: the quantization block describes the heterogeneous
        # checkpoint (mode + per-format contracts + distribution), never
        # W4A8-global properties.
        fmt_details = {
            FORMAT_W6A8: {
                "weight_bits": 6, "activation_bits": 8,
                "weight_quantization": "uniform symmetric signed six-bit codes with FP8 grid scale search",
                "scale_dtype": "fp8_e4m3fn", "packing": "int6-nibble-high2-plane",
                "convrot": True, "convrot_groupsize": 256,
                "group_size": getattr(args, "w6a8_group_size", 16),
            },
            FORMAT_W4A8: {
                "weight_bits": 4, "activation_bits": 8,
                "weight_quantization": "per-group 16-entry symmetric "
                                       "adaptive fixed/fitted codebook",
                "codebook_mode": getattr(args, "codebook_mode", "auto"),
                "scale_dtype": "fp8_e4m3fn",
                "packing": "int4-nibble-lsb",
                "convrot": True, "convrot_groupsize": 256,
                "group_size": 16,
            },
            FORMAT_W4A4: {
                "weight_bits": 4,
                "runtime_activation_bits": "backend-dependent: eager always "
                                           "executes A4; CUDA/HIP honor "
                                           "linear_dtype",
                "linear_dtype": (plan.mixed_plan or {}).get(
                    "w4a4_linear_dtype", DEFAULT_W4A4_LINEAR_DTYPE),
                "weight_quantization": "rowwise symmetric signed int4 "
                                       "(absmax/7, range [-7,7])",
                "scale_dtype": "fp32 rowwise [N]",
                "packing": "int4-nibble-lsb",
                "convrot": True,
                "convrot_groupsize": "per layer, power of 4 in {16,64,256}",
                "quant_group_size": 64,
            },
            FORMAT_INT8: {
                "weight_bits": 8, "activation_bits": 8,
                "weight_quantization": "rowwise symmetric int8 (absmax/127)",
                "scale_dtype": "fp32 rowwise [N,1]",
                "packing": "none",
                "convrot": False,
            },
        }
        mp = plan.mixed_plan or {}
        quant_block = {
            "mode": "mixed",
            "weight_bits": "mixed",
            "activation_precision": "per-format",
            "activation_quantization": "runtime dynamic symmetric int8 per "
                                       "input row (after ConvRot rotation for "
                                       "the convrot formats); W4A4 is "
                                       "backend-dependent (see formats)",
            "formats": {
                fmt: fmt_details[fmt] for fmt in MIXED_FORMATS
                if fmt in {f for f in (mp.get("counts") or {})}
            },
            "distribution": {
                "counts": mp.get("counts") or {},
                "layer_params": mp.get("layer_params") or {},
                "layer_bytes": mp.get("layer_bytes") or {},
                "kept_params": mp.get("kept_params") or 0,
                "kept_bytes": mp.get("kept_bytes") or 0,
                "effective_bytes_per_param": mp.get("effective_bpp"),
                "original_precision_parameter_fraction": mp.get(
                    "original_precision_parameter_fraction"),
                "original_precision_output_byte_fraction": mp.get(
                    "original_precision_output_byte_fraction"),
                "global_mean_error": mp.get("global_mean_error"),
                "promotions": mp.get("promotions") or [],
            },
            "profile": mp.get("profile"),
            "layer_gate": mp.get("layer_gate"),
            "global_gate": mp.get("global_gate"),
            "runtime_backend": mp.get("runtime_backend"),
            "w4a4_runtime": _w4a4_runtime_block(plan, args),
            "quality_validation": {
                "level": ("calibrated" if calibration is not None
                          else "unverified"),
                "reference_precision": "bf16/fp16 source",
                "note": ("calibrated = runtime-output layer calibration; "
                         "model-verified and e2e-verified levels are produced "
                         "by testdata/model_quality.py on the target machine"),
            },
            "compute_dtype": getattr(args, "compute_dtype", "auto"),
            "effective_compute_dtype": torch_dtype_name(
                getattr(args, "_compute_dtype_tensor", torch.float32)),
            "passthrough_output_dtype": getattr(args, "output_dtype", "auto"),
            "chunked_layers": sorted(plan.chunked_layers),
        }
    else:
        quant_block = {
            "weight_bits": 4,
            "activation_bits": 8,
            "weight_quantization": "per-group 16-entry symmetric adaptive "
                                   "fixed/fitted codebook",
            "codebook_mode": getattr(args, "codebook_mode", "auto"),
            "activation_quantization": "runtime dynamic symmetric int8 per input row after ConvRot",
            "activation_scale": "fp32 amax(row)/127, clamped to at least 1e-30",
            "activation_rounding": "nearest integer, clamped to [-128,127]",
            "group_size": quant_layers[0].group_size if quant_layers else None,
            "convrot": True,
            "convrot_groupsize": quant_layers[0].convrot_groupsize if quant_layers else None,
            "scale_dtype": "fp8_e4m3fn",
            "packing": "int4-nibble-lsb",
            "symmetric": True,
            "n_quantized_layers": len(quant_layers),
            "n_kept_tensors": plan.n_kept,
            "compute_dtype": getattr(args, "compute_dtype", "auto"),
            "effective_compute_dtype": torch_dtype_name(
                getattr(args, "_compute_dtype_tensor", torch.float32)),
            "passthrough_output_dtype": getattr(args, "output_dtype", "auto"),
            "chunked_layers": sorted(plan.chunked_layers),
        }
    if plan.fmt == FORMAT_W6A8:
        quant_block.update(weight_bits=6, packing="int6-nibble-high2-plane",
            weight_quantization="uniform symmetric signed six-bit codes with FP8 grid scale search")
        quant_block.pop("codebook_mode", None)
    conf = {
        "schema": "comfy_wxa8/v2" if is_mixed else "comfy_wxa8/v1",
        "converter": CONVERTER_NAME,
        "converter_version": get_converter_version(),
        "algorithm_identity": get_algorithm_identity(),
        "format": plan.fmt,
        "format_revision": format_revision(plan.fmt),
        "architecture": d.architecture,
        "detected_component": d.policy.component,
        "selected_components": getattr(args, "components", COMPONENT_AUTO),
        "detection_confidence": d.confidence,
        "unet_prefix": d.unet_prefix,
        "architecture_dims": _architecture_dims_fingerprint(info, d),
        "source": {
            "kind": info.kind,
            "files": _portable_file_labels(info.files),
            "total_bytes": info.total_bytes,
            "sha256": _portable_hash_manifest(info.files, input_hashes),
        },
        "quantization": quant_block,
        "calibration": calibration.to_dict() if calibration is not None else {
            "source": None, "method": "calibration-free (reference format)",
            "synthetic": False},
        "sensitivity": {
            "enabled": sensitivity is not None,
            "threshold": getattr(args, "sensitivity_threshold", None),
            "error_threshold": getattr(args, "error_threshold", None),
            "layers_kept": sorted(m.name for m in (sensitivity.results.values() if sensitivity else []) if m.kept),
        },
        "reproducibility": {
            "seed": getattr(args, "seed", 0),
            "device": getattr(args, "device", "auto"),
            "effective_device": plan.device,
            "torch_version": env.torch_version,
            "deterministic_on_same_backend": True,
            "codebook_subsample_seed": 0,
            "codebook_mode": getattr(args, "codebook_mode", "auto"),
        },
        "compatibility": {
            "comfy_kitchen": {
                "required_revision": COMFY_KITCHEN_REV,
                "pr": 90,
                "merged": True,
                "layouts": sorted({
                    FORMAT_TO_KITCHEN_LAYOUT[d.format]
                    for d in quant_layers if d.format in MIXED_FORMATS
                }) or ["AsymW4A8Int8Layout"],
            },
            "comfyui": {
                "required_pr": COMFYUI_PR,
                "required_head": COMFYUI_PR_HEAD,
                "merged": True,
                "merged_commit": "344b43989e",
                "min_version": "0.31.0",
                "note": "ComfyUI PR #15308 (asym_w4a8_int8 loader) merged 2026-08-07; "
                        "ComfyUI >= v0.31.0 loads W4A8 natively, older builds need "
                        "patches/comfyui_w4a8_loader.patch (base " + COMFYUI_BASE + ")",
            },
            "cuda_backend": {
                "requires": "PyTorch cu130+, SM >= 8.0",
                "min_sm": list(W4A8_KERNEL_MIN_SM),
            },
            "triton_backend": {"requires": f"triton >= {TRITON_MIN_VERSION[0]}.{TRITON_MIN_VERSION[1]} (ROCm)"},
        },
        "output": {
            "tensor_data_sha256": tensor_payload_sha256,
            "tensor_data_bytes": plan.total_out_bytes,
            "entries": len(plan.output_entries),
            "quant_metadata_transport": ("inline" if inline_quant_metadata(
                d.layer for d in quant_layers) else "header"),
            "file_sha256": None,
            "file_sha256_note": "The full-file SHA256 is emitted in the report. "
                                "It cannot be embedded in the file it hashes.",
        },
        "policy_summary": {
            "component_counts": {
                component: sum(1 for item in plan.decisions
                               if item.component == component)
                for component in (COMPONENT_DIFFUSION,
                                  COMPONENT_TEXT_ENCODER, "other")
            },
            "quantized_component_counts": {
                component: sum(1 for item in plan.quantized_layers()
                               if item.component == component)
                for component in (COMPONENT_DIFFUSION,
                                  COMPONENT_TEXT_ENCODER)
            },
            "decision_counts": {
                kind.value: sum(1 for item in plan.decisions if item.kind == kind)
                for kind in DecisionKind
            },
            "reason_counts": dict(sorted({
                reason: sum(1 for item in plan.decisions if item.reason == reason)
                for reason in {item.reason for item in plan.decisions}
            }.items())),
            "full_manifest": "conversion report",
        },
        "validation": validation_summary,
        "warnings": warnings,
    }
    if d.classifier_info.get("variant"):
        conf["model_variant"] = d.classifier_info["variant"]
    conf['input_adapter'] = {
        'hf': getattr(args, '_hf_provenance', None),
        'key_map': getattr(info, '_source_names', {}),
        'source_config': info.config,
    }
    if any(item.format == FORMAT_W6A8 for item in quant_layers) or plan.fmt == FORMAT_W6A8:
        conf["algorithm_identity"]["w6a8_format_revision"] = FORMAT_W6A8_REVISION
        conf["compatibility"]["comfy_kitchen"].update(required_revision="0.2.36", pr=191,
            six_bit_api_required=True)
        conf["compatibility"]["comfyui"] = {"min_version": "0.38.0", "merged": True,
            "note": "Native w6a8_int8 registration required; no loader patch"}
        if plan.fmt == FORMAT_W6A8:
            conf["algorithm_identity"]["quant_algorithm_rev"] = FORMAT_W6A8_REVISION
            conf["reproducibility"].pop("codebook_mode", None)
            conf["reproducibility"].pop("codebook_subsample_seed", None)
        if plan.fmt == FORMAT_W6A8:
            conf["algorithm_identity"].pop("w4a8_format_revision", None)
        conf["reproducibility"].update(kitchen_version=kitchen_version(),
            w6a8_group_size=getattr(args, "w6a8_group_size", 16),
            conversion_budget_bytes=getattr(args, "max_memory", None),
            chunk_budget_note="Across budgets, compare numerically; FP32 channel scales may differ by rounding")
        conf["w6a8_provenance"] = {"implementation": "comfy-kitchen public six-bit API",
            "minimum_kitchen_version": "0.2.36", "minimum_comfyui_version": "0.38.0",
            "upstream": "https://github.com/Comfy-Org/comfy-kitchen/pull/191",
            "bits": 6, "symmetric": True, "codebook": False, "scale_search": True}
    return conf
