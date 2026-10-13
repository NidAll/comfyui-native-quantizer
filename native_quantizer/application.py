"""Conversion orchestration over checkpoint, policy, planner, engine and validation."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os
import stat
import time
import torch
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from native_quantizer import events
from native_quantizer.events import prepare_event_stream
from native_quantizer.architectures.custom import apply_key_mapping, load_architecture_policy
from native_quantizer.architectures.detection import (
    DetectionResult,
    detect_architecture,
    unet_prefix_from_keys,
)
from native_quantizer.architectures.policies import get_family
from native_quantizer.calibration import apply_nonfinite_policy, load_calibration
from native_quantizer.checkpoint_io.fingerprints import hash_checkpoint_files
from native_quantizer.checkpoint_io.reader import CheckpointReader, RawSafetensorsFile, discover_checkpoint
from native_quantizer.checkpoint_io.types import CheckpointInfo
from native_quantizer.checkpoint_io.writer import republish_with_metadata
from native_quantizer.codecs.bounded import _quant_work_bytes
from native_quantizer.errors import (
    CalibrationError,
    CompressionGateError,
    InputError,
    OutputError,
    PolicyError,
    RuntimeCompatibilityError,
    UnknownArchitectureError,
    UnsupportedArchitectureError,
    UsageError,
)
from native_quantizer.execution.mixed import ConversionEngine
from native_quantizer.execution.native import convert_native
from native_quantizer.files import (
    _atomic_write_json,
    _fsync_parent,
    _publish_output,
    _remove_temp_path,
    _same_path,
    sha256_file,
    sha256_safetensors_payload,
)
from native_quantizer.formats import FORMAT_TO_KITCHEN_LAYOUT, NATIVE_FORMATS
from native_quantizer.identity import (
    FORMAT_INT8,
    FORMAT_MIXED,
    FORMAT_W4A4,
    FORMAT_W4A8,
    FORMAT_W6A8,
    GROUPED_INT8_FORMATS,
    LAYER_CONF_KEY,
    METADATA_KEY_EXT,
    METADATA_KEY_QUANT,
    MIXED_FORMATS,
)
from native_quantizer.logging import JsonLogHandler, log
from native_quantizer.metadata import (
    build_extension_metadata,
    build_output_entries,
    build_quant_metadata,
    serialized_quant_metadata,
)
from native_quantizer.options import ConversionOptions
from native_quantizer.planning.classification import _emit_plan_layers, classify_tensors
from native_quantizer.planning.metrics import SensitivityAnalyzer
from native_quantizer.planning.mixed import MixedPlanner
from native_quantizer.planning.sensitivity import apply_sensitivity_prepass
from native_quantizer.planning.types import ConversionPlan, DecisionKind, TensorDecision
from native_quantizer.reporting import (
    _write_reports,
    build_report,
    compression_stats,
    low_compression_warning,
    policy_miss_warning,
    print_console_summary,
)
from native_quantizer.runtime.capabilities import (
    EnvironmentInfo,
    _check_runtime_certificate,
    load_runtime_certificate,
    runtime_capabilities_for,
)
from native_quantizer.runtime.costs import load_kernel_costs
from native_quantizer.runtime.conversion import conversion_device
from native_quantizer.runtime.kitchen import require_w6a8_api, w6a8_api_available
from native_quantizer.sources.huggingface import download_hf_component
from native_quantizer.utils import human_bytes, json_dumps, parse_size
from native_quantizer.validation.validator import Validator, _refresh_validation_summary
from native_quantizer.validation.verify import plan_from_output


def _validate_destination_paths(info: CheckpointInfo, args: Any) -> None:
    destinations: List[Tuple[str, str]] = []
    if getattr(args, "output", None):
        output = os.path.abspath(args.output)
        destinations.extend((
            ("output", output),
            ("conversion temp", output + ".tmp"),
            ("staged output", output + ".staged"),
            ("validation output", output + ".validation"),
            ("resume state", output + ".state.json"),
        ))
        if getattr(args, "metadata_only", False):
            destinations.append(("metadata sidecar", output + ".metadata.json"))
    if getattr(args, "report", None):
        destinations.extend((("report", args.report),
                             ("JSON report", args.report + ".json")))
    if getattr(args, "json_log", None):
        destinations.append(("JSON log", args.json_log))
    if getattr(args, "events_jsonl", None):
        destinations.append(("event stream", args.events_jsonl))
    if getattr(args, "calibration_cache", None):
        destinations.append(("calibration cache", args.calibration_cache))

    protected_sources = list(info.files)
    protected_sources.extend(str(path) for path in Path(args.model).parent.glob('*.index.json')
                             if path.is_file())
    if Path(args.model).is_dir():
        protected_sources.extend(str(path) for path in Path(args.model).glob('*.index.json')
                                 if path.is_file())
    for source in info.files:
        for filename in ('config.json', 'model_index.json'):
            candidate = str(Path(source).parent / filename)
            if os.path.exists(candidate):
                protected_sources.append(candidate)
    for option in ('architecture_policy', 'key_map', 'calibration_source',
                   'kernel_costs', 'runtime_certificate'):
        candidate = getattr(args, option, None)
        if candidate and os.path.isfile(candidate):
            protected_sources.append(candidate)
    for label, path in destinations:
        for source in protected_sources:
            if _same_path(path, source):
                raise OutputError(f"{label} path must not alias input file {source}")
    for index, (left_label, left) in enumerate(destinations):
        for right_label, right in destinations[index + 1:]:
            if _same_path(left, right):
                raise OutputError(
                    f"{left_label} and {right_label} paths must be different: {left}")
    # Config/manifest files are inputs too, including explicit shard indexes.
    if args.model and os.path.isfile(args.model):
        for label, path in destinations:
            if _same_path(path, args.model):
                raise OutputError(f"{label} path must not alias input file {args.model}")


def _check_runtime_compatibility(env: EnvironmentInfo,
                                 planner: MixedPlanner,
                                 decisions: List[TensorDecision],
                                 warnings: List[str]) -> None:
    """Fail closed before conversion when the target machine demonstrably
    lacks a runtime path for a selected format.

    The probe is static (installed package source), never a runtime import.
    When neither comfy-kitchen nor ComfyUI is installed the probe is
    unavailable and the per-format runtime-contract validators remain the
    verification layer (with a warning)."""
    required = sorted({
        d.format for d in decisions
        if d.kind == DecisionKind.QUANTIZE and d.format in MIXED_FORMATS})
    if not required:
        return
    if env.has_comfy_quant_ops:
        missing = [f for f in required if f not in env.comfyui_quant_algos]
        if missing:
            raise RuntimeCompatibilityError(
                "mixed checkpoint requires ComfyUI formats "
                f"{', '.join(required)} but the installed ComfyUI "
                f"quant_ops.py registers only "
                f"{', '.join(env.comfyui_quant_algos) or 'none'}; missing: "
                f"{', '.join(missing)}. Update ComfyUI (>= v0.31.0) or avoid "
                "the affected formats with --disable-*.")
    if env.has_comfy_kitchen:
        layout_attr = {
            FORMAT_W4A8: "comfy_kitchen_has_w4a8_layout",
            FORMAT_W6A8: "comfy_kitchen_has_w6a8_api",
            FORMAT_W4A4: "comfy_kitchen_has_w4a4_layout",
            FORMAT_INT8: "comfy_kitchen_has_int8_layout",
        }
        missing = [f for f in required
                   if not (w6a8_api_available() if f == FORMAT_W6A8 else getattr(env, layout_attr[f], False))]
        if missing:
            raise RuntimeCompatibilityError(
                "mixed checkpoint requires comfy-kitchen layouts "
                f"{', '.join(FORMAT_TO_KITCHEN_LAYOUT[f] for f in missing)} "
                "but the installed comfy-kitchen does not contain them; "
                "update comfy-kitchen or avoid the affected formats with "
                "--disable-*.")
    if not env.has_comfy_kitchen and not env.has_comfy_quant_ops:
        warnings.append(
            "runtime compatibility probe unavailable (neither ComfyUI nor "
            "comfy-kitchen installed); mixed output is verified by the "
            "per-format runtime-contract validators only")


def run_conversion(options: ConversionOptions, env: EnvironmentInfo,
                   argument_error: Optional[Callable[[str], None]] = None) -> int:
    args = dataclasses.replace(options)
    if argument_error is None:
        def argument_error(message: str) -> None:
            raise UsageError(message)
    t_start = time.time()
    warnings: List[str] = []
    if isinstance(args.max_memory, str):
        args.max_memory = parse_size(args.max_memory)
    if args.max_memory <= 0:
        raise UsageError("--max-memory must be positive")
    if args.group_size is not None and args.group_size <= 0:
        raise UsageError("--group-size must be positive")
    if args.calibration_samples is not None and args.calibration_samples <= 0:
        raise UsageError("--calibration-samples must be positive")
    for option, value in (("--sensitivity-threshold", args.sensitivity_threshold),
                          ("--error-threshold", args.error_threshold),
                          ("--quality-gate", args.quality_gate),
                          ("--global-error-gate", args.global_error_gate)):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise UsageError(f"{option} must be a finite non-negative number")
    if (args.max_linear_bytes_per_param is not None
            and (not math.isfinite(args.max_linear_bytes_per_param)
                 or args.max_linear_bytes_per_param <= 0)):
        raise UsageError(
            "--max-linear-bytes-per-param must be a finite positive number")
    for option, value in (
            ("--max-original-byte-fraction", args.max_bf16_fraction),
            ("--min-quantized-byte-fraction",
             args.min_quantized_byte_fraction)):
        if value is not None and (not math.isfinite(value)
                                  or not 0.0 <= value <= 1.0):
            raise UsageError(f"{option} must be a finite number in [0, 1]")
    events.emit("phase", phase="inspection")
    info = discover_checkpoint(args.model, trust_pickle=args.trust_pickle,
                               allow_extra_shard_tensors=
                               args.allow_extra_shard_tensors)
    events.emit("source", files=len(info.files), tensors=len(info.tensors),
                       bytes=info.total_bytes, quantized=info.is_quantized_input)
    apply_key_mapping(info, args.key_map)
    warnings.extend(info.input_warnings)
    _validate_destination_paths(info, args)
    prepare_event_stream(args, source_files=info.files)
    events.emit('phase', phase='inspection')
    events.emit("source", files=len(info.files), tensors=len(info.tensors),
                bytes=info.total_bytes, quantized=info.is_quantized_input)
    if args.json_log:
        log().addHandler(JsonLogHandler(args.json_log))
    if info.kind == "pickle":
        warnings.append("pickle input loaded fully into RAM (streaming not possible); "
                        "only convert checkpoints you trust")

    if info.is_quantized_input and not (args.inspect or args.validation_only):
        raise InputError(
            "input checkpoint already contains quantization markers "
            f"('{METADATA_KEY_QUANT}' metadata or '{LAYER_CONF_KEY}' tensors); "
            "refusing to re-quantize. Use --inspect to review it.")

    def _shape_lookup(name: str) -> Optional[Tuple[int, ...]]:
        m = info.by_name(name)
        return tuple(m.shape) if m is not None else None

    events.emit("phase", phase="detection")
    try:
        detection = detect_architecture(
            info,
            override=None if args.architecture in (None, "auto") else args.architecture,
            shape_lookup=_shape_lookup)
        warnings.extend(detection.warnings)
    except UnknownArchitectureError:
        if args.inspect:
            # inspection must still work for unknown checkpoints
            detection = DetectionResult(
                architecture="unknown", confidence="none",
                policy=get_family("sd15"), unet_prefix=unet_prefix_from_keys(info.key_set()),
                warnings=["architecture not identified (inspection mode)"])
        else:
            raise

    events.emit("detection", architecture=detection.architecture,
                       confidence=detection.confidence,
                       variant=detection.classifier_info.get("variant"),
                       component=detection.policy.component,
                       warnings=detection.warnings)

    if args.inspect:
        print("=" * 70)
        print("input checkpoint inspection")
        print("=" * 70)
        print(f"kind            : {info.kind}")
        print(f"files           : {info.files}")
        print(f"tensors         : {len(info.tensors)}  ({human_bytes(info.total_bytes)})")
        print(f"metadata keys   : {sorted(info.metadata.keys())}")
        print(f"config.json     : {'present' if info.config else 'absent'}")
        print(f"quantized input : {info.is_quantized_input}")
        print(f"detected arch   : {detection.architecture} (confidence {detection.confidence})")
        if detection.classifier_info.get("variant"):
            print(f"model variant   : {detection.classifier_info['variant']}")
        print(f"component       : {detection.policy.component}")
        print(f"unet prefix     : {detection.unet_prefix!r}")
        print("evidence        : " + "; ".join(detection.evidence or ["-"]))
        print("hints           : " + "; ".join(detection.hints or ["-"]))
        print("competing       : " + "; ".join(detection.competing or ["-"]))
        print("policy          : " + detection.policy.family)
        print()
        print("first 40 tensors:")
        for t in info.tensors[:40]:
            print(f"  {t.name:70s} {t.dtype} {tuple(t.shape)}")
        if len(info.tensors) > 40:
            print(f"  ... {len(info.tensors) - 40} more")
        return 0

    if detection.policy.runtime_status == "unsupported" and args.architecture in (None, "auto"):
        raise UnsupportedArchitectureError(
            f"architecture {detection.architecture!r} has no ComfyUI quantized-loading "
            "path; conversion would not be consumable. If you still want to convert it "
            "for research, pass --architecture " + detection.architecture + " explicitly.")

    if args.quantize_text_embeddings and args.format not in ("int8", "int8-convrot", "mixed"):
        raise UsageError('--quantize-text-embeddings requires int8, int8-convrot, or mixed; '
                         'the other formats do not support token embedding lookup')
    if args.quantize_text_embeddings and args.format == "mixed" and args.disable_int8:
        raise UsageError('--quantize-text-embeddings in mixed mode requires INT8; '
                         'remove --disable-int8')

    if args.kernel_costs and args.format != "mixed":
        raise UsageError("--kernel-costs requires --format mixed")
    if args.format in NATIVE_FORMATS:
        if detection.classifier_info.get("variant") == "design-layer" and not args.dry_run:
            log().warning(detection.warnings[-1])
        return convert_native(info, detection, args)
    if args.full_precision_matmul:
        raise UsageError('--full-precision-matmul is available for explicit native formats only')
    if args.validation_only:
        if not args.output or not os.path.exists(args.output):
            argument_error("--validation-only requires an existing --output file")
        if args.format == "w4a8":
            fmt = FORMAT_W4A8
        elif args.format == "w6a8":
            fmt = FORMAT_W6A8
        elif args.format == "mixed":
            fmt = FORMAT_MIXED
        else:
            argument_error(f"--format must be w4a8, w6a8 or mixed, got {args.format!r}")
        plan = plan_from_output(args.output, detection, fmt, info)
        validator = Validator(info, plan, args.output, args, env)
        validation_input_hashes = hash_checkpoint_files(info)
        with CheckpointReader(info) as validation_reader:
            summary = validator.run(reader=validation_reader,
                                    input_hashes=validation_input_hashes)
        quant_rows = []
        for decision in plan.quantized_layers():
            tensor = info.by_name(decision.name)
            shape = tuple(tensor.shape) if tensor is not None else "missing"
            quant_rows.append(
                f"{decision.layer}: {shape} gs={decision.group_size} "
                f"cgs={decision.convrot_groupsize} "
                f"fmt={decision.format}")
        report = build_report(info, plan, env, args, detection, None, {},
                              summary, validation_input_hashes,
                              summary.get("output_sha256", ""),
                              time.time() - t_start, warnings, quant_rows)
        _write_reports(args, report)
        return 0 if summary["n_failed"] == 0 else 2

    if args.format == "w4a8":
        fmt = FORMAT_W4A8
    elif args.format == "w6a8":
        fmt = FORMAT_W6A8
    elif args.format == "mixed":
        fmt = FORMAT_MIXED
    else:
        argument_error(f"--format must be w4a8, w6a8 or mixed, got {args.format!r}")

    if fmt == FORMAT_W6A8 or args.enable_w6a8:
        require_w6a8_api()
        if env.has_comfy_quant_ops and FORMAT_W6A8 not in env.comfyui_quant_algos:
            raise RuntimeCompatibilityError("W6A8 requires ComfyUI >= 0.38.0; installed quant_ops does not register w6a8_int8")
        if args.require_calibration and not args.calibration_source:
            raise CalibrationError("--require-calibration needs --calibration-source")
    # ---- mixed-mode runtime + profile resolution (auto) ----
    # Auto detects the GPU (NVIDIA CUDA, AMD ROCm, or CPU) and picks the
    # balanced profile on an accelerator, conservative on CPU.  The detected
    # architecture is already known here; a calibration-free balanced run is
    # allowed but reported as weight-gated rather than activation-gated.
    effective_runtime = "auto"
    if args.target_runtime != "auto":
        effective_runtime = args.target_runtime
    elif torch.cuda.is_available():
        if getattr(torch.version, "hip", None) is not None:
            effective_runtime = "amd"
        else:
            effective_runtime = "nvidia"
    else:
        effective_runtime = "cpu"
    if fmt == FORMAT_MIXED:
        profile_name = args.profile
        if profile_name == "auto":
            profile_name = "balanced" if effective_runtime != "cpu" else "conservative"
            warnings.append(
                f"auto profile: {profile_name} for target runtime "
                f"{effective_runtime} (set --profile to override)")
        if args.require_calibration and not args.calibration_source:
            argument_error("--require-calibration needs --calibration-source")
        disabled_formats = []
        if args.disable_w4a4:
            disabled_formats.append(FORMAT_W4A4)
        if args.disable_w4a8:
            disabled_formats.append(FORMAT_W4A8)
        if args.disable_int8:
            disabled_formats.append(FORMAT_INT8)
        if len(disabled_formats) == 3 and not args.enable_w6a8:
            argument_error("at least one quantized format must stay enabled in mixed mode")
        if effective_runtime == "cpu":
            warnings.append(
                "W4A4 linear_dtype note: the comfy-kitchen eager backend "
                "accepts int4/int8 but always executes the int4 activation "
                "path; linear_dtype=int8 only changes the CUDA kernels")
        args._mixed_profile_name = profile_name
        args._mixed_disabled_formats = disabled_formats
        args._effective_runtime = effective_runtime

    if args.output is None:
        argument_error("--output is required")

    # ---- output path safety ----
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.isdir(out_path):
        raise OutputError(f"output path is a directory: {out_path}")
    for f in info.files:
        same_file = os.path.abspath(f) == out_path
        if os.path.exists(out_path):
            with contextlib.suppress(OSError):
                same_file = same_file or os.path.samefile(f, out_path)
        if same_file:
            raise OutputError("output path must not be the same as an input file")
    if os.path.exists(out_path) and not args.overwrite and not args.resume:
        raise OutputError(f"output already exists: {out_path} (use --overwrite)")
    # Capture the destination identity before any work begins: with
    # --overwrite the final publication re-verifies that the destination is
    # still the same object (TOCTOU protection).
    expected_dest_identity: Optional[Tuple[int, int]] = None
    if args.overwrite and os.path.lexists(out_path):
        lst = os.lstat(out_path)
        if stat.S_ISLNK(lst.st_mode):
            raise OutputError(
                f"output path is a symlink: {out_path}; refusing --overwrite")
        expected_dest_identity = (int(lst.st_dev), int(lst.st_ino))
    tmp_path = out_path + ".tmp"
    staged_path = out_path + ".staged"
    validation_path = out_path + ".validation"
    state_path = out_path + ".state.json"
    if os.path.exists(tmp_path) and not args.overwrite and not args.resume:
        raise OutputError(f"temp output already exists: {tmp_path} (use --overwrite or --resume)")
    for label, internal_path in (("staged output", staged_path),
                                 ("validation output", validation_path)):
        if not os.path.exists(internal_path):
            continue
        if args.overwrite:
            _remove_temp_path(internal_path)
        elif args.resume:
            # A validation copy is never resume authority.  A staged output is
            # trusted only together with its checksummed state file.
            if internal_path == validation_path or not os.path.exists(state_path):
                _remove_temp_path(internal_path)
        else:
            raise OutputError(
                f"{label} already exists: {internal_path} "
                "(use --overwrite or --resume)")

    # ---- planning ----
    compute_dtype = {"auto": torch.float32, "fp32": torch.float32,
                     "fp16": torch.float16, "bf16": torch.bfloat16}[args.compute_dtype]
    args._compute_dtype_tensor = compute_dtype
    out_dtype = {"auto": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.output_dtype]
    events.emit("phase", phase="planning")
    decisions = classify_tensors(info, detection, fmt, args.w6a8_group_size if fmt == FORMAT_W6A8 else args.group_size,
                                 args.include, args.exclude, args.keep_precision,
                                 out_dtype, None, components=args.components,
                                 quantize_text_embeddings=args.quantize_text_embeddings)
    # ``keep`` must scan before inventory planning because it changes tensor
    # layouts. A W4A8 sensitivity prepass also benefits from early rejection.
    # The common error-only conversion path relies on the quantizers' fused
    # amax/finite checks, avoiding a redundant full read of every large layer.
    early_nonfinite_check = (
        args.nonfinite_policy == "keep"
        or (fmt in GROUPED_INT8_FORMATS
            and (args.sensitivity_threshold is not None
                 or args.error_threshold is not None
                 or args.calibration_source is not None))
    )
    nonfinite_kept = (apply_nonfinite_policy(
        info, decisions, args.nonfinite_policy) if early_nonfinite_check else [])
    if nonfinite_kept:
        warnings.append(
            f"nonfinite policy keep: {len(nonfinite_kept)} layer(s) with NaN/Inf "
            "passed through at original precision: " +
            "; ".join(nonfinite_kept[:3]) +
            (" ..." if len(nonfinite_kept) > 3 else ""))

    effective_device, effective_backend = conversion_device(args.device)

    # Sensitivity / mixed planning is a planning operation.  It must be
    # complete before output shapes and offsets are frozen.
    needs_sensitivity_prepass = (
        fmt in GROUPED_INT8_FORMATS
        and (args.sensitivity_threshold is not None
             or args.error_threshold is not None
             or args.calibration_source is not None)
    )
    prepass_source_hashes = (
        hash_checkpoint_files(info, refresh=True)
        if needs_sensitivity_prepass else None
    )
    calibration = None
    if args.calibration_source:
        events.emit("phase", phase="calibration")
        calibration = load_calibration(args.calibration_source, info,
                                       args.calibration_samples, args.calibration_cache,
                                       args.max_memory)
    if fmt == FORMAT_W6A8:
        if args.require_calibration:
            missing = [d.name for d in decisions if d.kind == DecisionKind.QUANTIZE
                       and (calibration is None or not any(key in calibration.layers for key in (d.name, d.layer)))]
            if missing:
                raise CalibrationError("calibration missing for targeted W6A8 layers: " + ", ".join(missing[:5]))
        if args.require_runtime_certificate and not args.runtime_certificate:
            raise RuntimeCompatibilityError("--require-runtime-certificate needs --runtime-certificate")
        if args.runtime_certificate:
            _check_runtime_certificate(load_runtime_certificate(args.runtime_certificate),
                runtime_capabilities_for(effective_runtime, env), [FORMAT_W6A8])
    sensitivity = None
    mixed_planner = None
    if fmt == FORMAT_MIXED:
        events.emit("phase", phase="mixed_analysis")
        profile_name = getattr(args, "_mixed_profile_name", "balanced")
        layer_gate = args.quality_gate
        if layer_gate is None and args.error_threshold is not None:
            layer_gate = args.error_threshold
        runtime_caps = runtime_capabilities_for(effective_runtime, env)
        certificate = None
        if args.runtime_certificate:
            certificate = load_runtime_certificate(args.runtime_certificate)
            warnings.append(
                f"runtime certificate loaded: backend={certificate.backend}, "
                f"gpu={certificate.gpu or 'unknown'}, formats="
                f"{sorted(certificate.formats)}")
            # Apply the certificate to the capabilities BEFORE planning: the
            # planner, its summary and the extension metadata must all observe
            # one final authoritative RuntimeCapabilities (never mutated after
            # planning). Coverage of the actually selected formats is still
            # enforced after planning, when the selection is known.
            _check_runtime_certificate(certificate, runtime_caps, [])
            cap_map = {FORMAT_W4A4: ("w4a4", runtime_caps.w4a4),
                       FORMAT_W4A8: ("w4a8", runtime_caps.w4a8),
                       FORMAT_INT8: ("int8", runtime_caps.int8),
                       FORMAT_W6A8: ("w6a8", runtime_caps.w6a8)}
            certed = {
                field: dataclasses.replace(cap, certified=True)
                for fmt, (field, cap) in cap_map.items()
                if fmt in certificate.formats}
            runtime_caps = dataclasses.replace(
                runtime_caps, runtime_certified=True, **certed)
        costs, workloads = {}, []
        if args.kernel_costs:
            if effective_runtime != "nvidia":
                raise UsageError("--kernel-costs currently requires --target-runtime nvidia")
            try:
                costs, workloads = load_kernel_costs(args.kernel_costs, env.comfy_kitchen_rev)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                raise UsageError(f"invalid --kernel-costs: {exc}") from exc
        mixed_planner = MixedPlanner(
            profile_name, calibration, args.max_memory, effective_device,
            compute_dtype,
            disabled_formats=getattr(args, "_mixed_disabled_formats", ()),
            layer_gate=layer_gate,
            global_gate=args.global_error_gate,
            runtime=runtime_caps,
            compression_target_bpp=args.max_linear_bytes_per_param,
            max_bf16_fraction=args.max_bf16_fraction,
            linear_dtype=args.w4a4_linear_dtype,
            runtime_certificate=certificate,
            seed=args.seed,
            codebook_mode=args.codebook_mode,
            objective=args.mixed_objective, kernel_costs=costs,
            enable_w6a8=args.enable_w6a8, w6a8_group_size=args.w6a8_group_size)
        if args.require_runtime_certificate and certificate is None:
            raise RuntimeCompatibilityError(
                "--require-runtime-certificate needs --runtime-certificate "
                "(supply a certificate from real native execution on the "
                "target inference machine)")
        if args.require_calibration:
            missing = [d.name for d in decisions if d.kind == DecisionKind.QUANTIZE
                       and mixed_planner._activations(d.name) is None]
            if missing:
                raise CalibrationError(
                    f"calibration missing for {len(missing)} targeted layers: " + ", ".join(missing[:5]))
        mixed_plan = mixed_planner.plan(info, decisions)
        mixed_plan["measured_workloads"] = workloads
        if args.kernel_costs:
            mixed_plan["kernel_costs_sha256"] = sha256_file(args.kernel_costs)
        args._runtime_gpu_name = runtime_caps.gpu_name
        _check_runtime_compatibility(env, mixed_planner, decisions, warnings)
        if certificate is not None:
            required = sorted({
                d.format for d in decisions
                if d.kind == DecisionKind.QUANTIZE
                and d.format in MIXED_FORMATS})
            _check_runtime_certificate(certificate, runtime_caps, required)
        if not calibration:
            warnings.append(
                "mixed planner: no calibration activations, quality gates use "
                "weight-only reconstruction error (pass --calibration-source "
                "for activation-aware gates)")
        if mixed_plan["promotions"]:
            warnings.append(
                f"mixed planner: global gate required {len(mixed_plan['promotions'])} "
                "promotion(s): " + "; ".join(mixed_plan["promotions"][:3]) +
                (" ..." if len(mixed_plan["promotions"]) > 3 else ""))
        if mixed_plan["kept"]:
            warnings.append(
                f"mixed planner kept {mixed_plan['kept']} layer(s) at original "
                "precision (no candidate within the quality gate)")
        if profile_name != "balanced" and args.quality_gate is None \
                and args.global_error_gate is None:
            warnings.append(
                f"mixed profile {profile_name}: layer gate "
                f"{mixed_planner.layer_gate}, global gate "
                f"{mixed_planner.global_gate} (override with --quality-gate / "
                "--global-error-gate)")
    elif needs_sensitivity_prepass:
        sensitivity = SensitivityAnalyzer(
            args.sensitivity_threshold,
            args.error_threshold if args.error_threshold is not None
            else detection.policy.max_rel_l2,
            calibration)
        apply_sensitivity_prepass(info, decisions, sensitivity, args.max_memory,
                                  effective_device, compute_dtype,
                                  seed=args.seed,
                                  codebook_mode=args.codebook_mode)
        kept_by_sensitivity = sum(1 for m in sensitivity.results.values() if m.kept)
        if kept_by_sensitivity:
            warnings.append(
                f"sensitivity prepass retained {kept_by_sensitivity} layer(s) at "
                "original precision")
        post_prepass_hashes = hash_checkpoint_files(info, refresh=True)
        if post_prepass_hashes != prepass_source_hashes:
            raise InputError(
                "one or more source files changed during sensitivity planning; "
                "refusing to build an output inventory from mixed data")

    plan = ConversionPlan(fmt=fmt, detection=detection, decisions=decisions,
                          metadata_quant={}, metadata_ext={}, output_entries=[])
    plan.device = effective_backend
    if mixed_planner is not None:
        plan.mixed_plan = mixed_planner.summary
    entries, total_out = build_output_entries(info, decisions, fmt, out_dtype)
    plan.output_entries = entries
    plan.total_out_bytes = total_out
    plan.n_quantized = len(plan.quantized_layers())
    plan.n_kept = len(decisions) - plan.n_quantized
    plan.chunked_layers = {
        d.name for d in plan.quantized_layers()
        if _quant_work_bytes(info.by_name(d.name), d.format) > args.max_memory
    }
    plan.metadata_quant = build_quant_metadata(info, plan)

    events.emit("plan", architecture=detection.architecture,
                       format=args.format, quantized=plan.n_quantized,
                       preserved=plan.n_kept, output_bytes=total_out,
                       formats={name: sum(d.format == name for d in plan.quantized_layers())
                                for name in sorted({d.format for d in plan.quantized_layers()})})
    _emit_plan_layers(info, decisions)

    if plan.n_quantized == 0:
        raise PolicyError(
            "no tensors selected for quantization under the "
            f"{detection.architecture!r} policy with --components "
            f"{args.components!r} after safety/quality gates "
            "(check --components, adjust thresholds, or use --include to "
            "force eligible layers)")

    quant_rows = []
    for d in plan.quantized_layers():
        m = info.by_name(d.name)
        quant_rows.append(
            f"{d.layer}: {tuple(m.shape)} component={d.component} gs={d.group_size} "
            f"cgs={d.convrot_groupsize} fmt={d.format}")

    comp_stats = compression_stats(info, plan, detection)
    effective_group_size = (plan.quantized_layers()[0].group_size
                            if plan.quantized_layers() else
                            (args.group_size or 16))
    low_comp = low_compression_warning(comp_stats, detection.architecture)
    if low_comp:
        warnings.append(low_comp)
    miss = policy_miss_warning(comp_stats, detection.architecture)
    if miss:
        warnings.append(miss)
    fail_threshold = args.min_quantized_byte_fraction
    if fail_threshold is None and args.fail_on_low_compression:
        fail_threshold = 0.10
    frac = comp_stats.get("quantized_fraction")
    if fail_threshold is not None and frac is not None and frac < fail_threshold:
        flag = ("--min-quantized-byte-fraction" if args.min_quantized_byte_fraction
                else "--fail-on-low-compression")
        raise CompressionGateError(
            f"quantized byte fraction {100*frac:.1f}% is below the required "
            f"{100*fail_threshold:.0f}% ({flag}); nothing was written. Lower "
            "the threshold or drop the flag to allow the conversion.")

    if args.metadata_only:
        metadata_input_hashes = hash_checkpoint_files(info)
        meta = dict(info.metadata)
        meta[METADATA_KEY_QUANT] = json_dumps(plan.metadata_quant)
        meta[METADATA_KEY_EXT] = json_dumps(build_extension_metadata(
            info, plan, env, args, calibration, sensitivity,
            metadata_input_hashes, "", {"status": "metadata-only"},
            warnings))
        if args.output:
            _atomic_write_json(args.output + ".metadata.json", meta, indent=1)
            log().info("metadata written to %s.metadata.json", args.output)
        else:
            print(json_dumps(meta))
        report = build_report(info, plan, env, args, detection, None, {}, {},
                              {}, "", time.time() - t_start, warnings, quant_rows)
        _write_reports(args, report)
        return 0

    if args.dry_run:
        report = build_report(info, plan, env, args, detection, None, {}, {},
                              {}, "", time.time() - t_start, warnings, quant_rows)
        _write_reports(args, report)
        print_console_summary(
            out_path=out_path, input_bytes=info.total_bytes,
            output_bytes=total_out, architecture=detection.architecture,
            confidence=detection.confidence, n_quantized=plan.n_quantized,
            n_kept=plan.n_kept, comp=comp_stats, group_size=effective_group_size,
            validation=None, elapsed=time.time() - t_start, warnings=warnings,
            report_path=os.path.abspath(args.report) if args.report else None,
            dry_run=True, mixed_plan=plan.mixed_plan, format_name=plan.fmt)
        return 0

    # ---- conversion ----
    events.emit("phase", phase="conversion")
    engine = ConversionEngine(info, plan, args, state_path, tmp_path, staged_path)
    try:
        engine.run()
    except Exception:
        try:
            engine.save_state()
        finally:
            engine.close()
        log().error("conversion failed; state saved to %s (rerun with --resume)",
                    state_path)
        raise
    engine.close()
    input_hashes = engine.input_hashes
    metrics = sensitivity.results if sensitivity else {}

    # ---- provisional metadata + validation ----
    tensor_payload_sha = sha256_safetensors_payload(staged_path)
    # Preserve benign source metadata, overriding only the two keys owned by
    # this converter.  Quantized inputs are refused earlier, so replacement is
    # defensive rather than a re-quantization path.
    qm_meta = dict(info.metadata)
    qm_meta[METADATA_KEY_QUANT] = json_dumps(serialized_quant_metadata(plan.metadata_quant))
    ext_meta = build_extension_metadata(
        info, plan, env, args, calibration, sensitivity, input_hashes,
        tensor_payload_sha, {"status": "pending"}, warnings)
    qm_meta[METADATA_KEY_EXT] = json_dumps(ext_meta)
    log().info("building validation copy at %s", validation_path)
    republish_with_metadata(staged_path, validation_path, qm_meta, entries)

    # ---- validation ----
    events.emit("phase", phase="validation")
    validator = Validator(info, plan, validation_path, args, env)
    with CheckpointReader(info) as validation_reader:
        summary = validator.run(reader=validation_reader, metrics=metrics,
                                input_hashes=input_hashes)

    # Never expose a newly generated checkpoint at the requested output path
    # when standalone validation failed.  The checksummed staged file and state
    # remain available for --resume; an older --overwrite target is untouched.
    if summary["n_failed"]:
        warnings.append(
            f"validation failed {summary['n_failed']} checks; new output was not "
            "published and any pre-existing output remains unchanged")
        report = build_report(
            info, plan, env, args, detection, calibration, metrics, summary,
            input_hashes, "", time.time() - t_start, warnings, quant_rows)
        _write_reports(args, report)
        if os.path.exists(validation_path):
            _remove_temp_path(validation_path)
        return 2

    # Embed the completed tensor/schema validation.  Full-file SHA256 is not
    # embedded because changing the header changes that hash; the stable tensor
    # payload hash is embedded instead.
    embedded_summary = json.loads(json_dumps(summary))
    embedded_summary["checks"] = [
        check for check in embedded_summary.get("checks", [])
        if check.get("name") != "output-hash"
    ]
    embedded_summary.pop("output_sha256", None)
    embedded_summary["scope"] = (
        "tensor payload, reconstruction, policy and metadata schema before final "
        "metadata publication; tensor payload is unchanged by publication")
    _refresh_validation_summary(embedded_summary)
    ext_meta = build_extension_metadata(
        info, plan, env, args, calibration, sensitivity, input_hashes,
        tensor_payload_sha, embedded_summary, warnings)
    qm_meta[METADATA_KEY_EXT] = json_dumps(ext_meta)
    # Build and inspect the exact final checkpoint at the private validation
    # path.  Only a fully reopened, payload-bound candidate may replace the
    # requested output (including an existing --overwrite target).
    republish_with_metadata(staged_path, validation_path, qm_meta, entries)

    final_ok = False
    try:
        with RawSafetensorsFile(validation_path) as final_raw:
            final_meta = final_raw.metadata
        final_ext = json.loads(final_meta[METADATA_KEY_EXT])
        final_ok = (final_ext.get("output", {}).get("tensor_data_sha256") ==
                    sha256_safetensors_payload(validation_path))
        summary.setdefault("checks", []).append({
            "name": "final-publication", "status": "passed" if final_ok else "failed",
            "detail": "final file reopens; embedded tensor payload hash matches"
                      if final_ok else "final tensor payload hash mismatch",
            "reason": "",
        })
    except Exception as e:
        summary.setdefault("checks", []).append({
            "name": "final-publication", "status": "failed",
            "detail": f"final reopen failed: {e}", "reason": "",
        })
    _refresh_validation_summary(summary)
    events.emit("validation", ok=final_ok and not summary.get("n_failed", 0),
                       checks=len(summary.get("checks", [])))
    if not final_ok:
        warnings.append(
            "final checkpoint candidate failed integrity checks; requested output "
            "was not replaced")
        report = build_report(
            info, plan, env, args, detection, calibration, metrics, summary,
            input_hashes, "", time.time() - t_start, warnings, quant_rows)
        _write_reports(args, report)
        if os.path.exists(validation_path):
            _remove_temp_path(validation_path)
        return 2

    events.emit("phase", phase="publication")
    output_sha = sha256_file(validation_path)
    _publish_output(validation_path, out_path,
                    overwrite=bool(args.overwrite),
                    expected_identity=expected_dest_identity)
    _fsync_parent(out_path)
    summary["output_sha256"] = output_sha
    output_hash_check = next(
        (check for check in summary.get("checks", [])
         if check.get("name") == "output-hash"), None)
    if output_hash_check is not None:
        output_hash_check["detail"] = f"sha256={output_sha}"
    elif args.validate:
        summary.setdefault("checks", []).append({
            "name": "output-hash", "status": "passed",
            "detail": f"sha256={output_sha}", "reason": "",
        })
    _refresh_validation_summary(summary)

    # ---- report ----
    report = build_report(info, plan, env, args, detection, calibration, metrics,
                          summary, input_hashes, output_sha,
                          time.time() - t_start, warnings, quant_rows)
    _write_reports(args, report)
    print_console_summary(
        out_path=out_path, input_bytes=info.total_bytes,
        output_bytes=os.path.getsize(out_path),
        architecture=detection.architecture, confidence=detection.confidence,
        n_quantized=plan.n_quantized, n_kept=plan.n_kept, comp=comp_stats,
        group_size=effective_group_size, validation=summary,
        elapsed=time.time() - t_start, warnings=warnings,
        report_path=os.path.abspath(args.report) if args.report else None,
        dry_run=False, mixed_plan=plan.mixed_plan, format_name=plan.fmt)
    events.emit("artifact", path=out_path, bytes=os.path.getsize(out_path),
                       quantized=plan.n_quantized)

    # ---- cleanup ----
    if os.path.exists(tmp_path):
        _remove_temp_path(tmp_path)
    if os.path.exists(validation_path):
        _remove_temp_path(validation_path)
    if os.path.exists(staged_path):
        _remove_temp_path(staged_path)
    if os.path.exists(state_path):
        _remove_temp_path(state_path)
    return 0 if summary["n_failed"] == 0 else 2


def resolve_input_sources(args) -> None:
    if args.architecture_policy:
        if not args.experimental:
            raise UsageError('custom architectures require --experimental; runtime loading is not certified')
        policy = load_architecture_policy(args.architecture_policy)
        if args.architecture not in ('auto', policy.family):
            raise UsageError('--architecture conflicts with the custom policy family')
        args.architecture = policy.family
    if args.hf_repo:
        args.model = download_hf_component(args)
