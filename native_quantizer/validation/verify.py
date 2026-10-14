"""Source-free output verification and plan reconstruction."""

from __future__ import annotations

import json
import os
import torch
from safetensors import safe_open
from typing import Any, Dict, List, Optional, Tuple

from native_quantizer.architectures.detection import DetectionResult
from native_quantizer.checkpoint_io.types import CheckpointInfo
from native_quantizer.codecs.int8 import dequantize_int8_tensorwise_weight
from native_quantizer.codecs.rotation import _is_power_of_four
from native_quantizer.codecs.w4a4 import dequantize_w4a4_weight
from native_quantizer.codecs.w4a8 import dequantize_w4a8_weight, validate_w4_shape
from native_quantizer.codecs.w6a8 import pack_w6, unpack_w6, validate_w6_shape
from native_quantizer.dtypes import tensor_nbytes, torch_dtype_from_safe
from native_quantizer.errors import PolicyError, ValidationError, UsageError
from native_quantizer.optimization import read_optimization_metadata
from native_quantizer.files import sha256_file, sha256_safetensors_payload
from native_quantizer.identity import (
    DEFAULT_W4A4_LINEAR_DTYPE,
    FORMAT_INT8,
    FORMAT_MIXED,
    FORMAT_W4A4,
    FORMAT_W4A8,
    FORMAT_W6A8,
    METADATA_KEY_EXT,
    METADATA_KEY_QUANT,
    MIXED_FORMATS,
    W4A4_QUANT_GROUP_SIZE,
    W4A8_CONVROT_GROUPSIZE,
)
from native_quantizer.metadata import build_output_entries, read_quant_metadata
from native_quantizer.planning.types import ConversionPlan, DecisionKind, TensorDecision
from native_quantizer.utils import human_bytes
from native_quantizer.validation.contracts import w6a8_checkpoint_contract, w6a8_extension_contract
from native_quantizer.validation.native import verify_native_output
from native_quantizer.validation.reference import reference_decode_w6a8


def _fmt_display(fmt: Optional[str]) -> Optional[str]:
    return {FORMAT_W4A8: "w4a8", FORMAT_W6A8: "w6a8"}.get(fmt, fmt)


def plan_from_output(output_path: str, detection: DetectionResult,
                     fmt: str, info: Optional[CheckpointInfo] = None) -> ConversionPlan:
    """Reconstruct a minimal plan from an existing output checkpoint (validation-only)."""
    with safe_open(output_path, framework="pt") as st:
        names = list(st.keys())
        meta = st.metadata() or {}
        actual_entries = []
        for name in names:
            sl = st.get_slice(name)
            shape = tuple(sl.get_shape())
            dt = torch_dtype_from_safe(sl.get_dtype())
            actual_entries.append({
                "name": name,
                "dtype": dt,
                "shape": shape,
                "nbytes": tensor_nbytes(dt, shape),
            })
    try:
        qm = read_quant_metadata(output_path)
    except (TypeError, ValueError) as e:
        raise ValidationError(
            f"{output_path}: invalid {METADATA_KEY_QUANT} metadata: {e}") from e
    if not isinstance(qm, dict) or not isinstance(qm.get("layers"), dict):
        raise ValidationError(
            f"{output_path}: {METADATA_KEY_QUANT} must contain a layers object")
    layers = qm.get("layers", {})
    if not layers:
        raise ValidationError(
            f"{output_path}: no quantized layers recorded in {METADATA_KEY_QUANT}")
    decisions: List[TensorDecision] = []
    for layer, conf in layers.items():
        if not isinstance(layer, str) or not layer or not isinstance(conf, dict):
            raise ValidationError(
                f"{output_path}: malformed quantized-layer metadata entry")
        lfmt = conf.get("format")
        if lfmt not in MIXED_FORMATS:
            raise ValidationError(
                f"{output_path}: incompatible format metadata for {layer!r} "
                f"(unknown per-layer format {lfmt!r})")
        if fmt == FORMAT_MIXED:
            pass
        elif lfmt != fmt:
            raise ValidationError(
                f"{output_path}: incompatible format metadata for {layer!r}")
        try:
            group_size = int(conf.get("group_size", 16))
            convrot_groupsize = int(conf.get("convrot_groupsize", 256))
        except (TypeError, ValueError) as e:
            raise ValidationError(
                f"{output_path}: invalid group metadata for {layer!r}") from e
        if group_size <= 0 or convrot_groupsize <= 0:
            raise ValidationError(
                f"{output_path}: invalid group metadata for {layer!r}")
        source_name = f"{layer}.weight"
        if info is not None:
            source_meta = info.by_name(source_name)
            if source_meta is None or len(source_meta.shape) != 2:
                raise ValidationError(
                    f"{output_path}: quantized layer {layer!r} has no matching "
                    "2D weight in the supplied original checkpoint")
            k = int(source_meta.shape[1])
            if lfmt == FORMAT_W6A8:
                try:
                    validate_w6_shape(k, group_size, convrot_groupsize)
                except PolicyError as exc:
                    raise ValidationError(f"{layer}: {exc}") from exc
            elif lfmt == FORMAT_W4A8:
                if convrot_groupsize != W4A8_CONVROT_GROUPSIZE:
                    raise ValidationError(
                        f"{output_path}: layer {layer!r} uses convrot_groupsize "
                        f"{convrot_groupsize}, but the comfy-kitchen 0.2.27 CUDA "
                        f"runtime only supports {W4A8_CONVROT_GROUPSIZE} (K must "
                        "be divisible by 256). Re-convert the checkpoint with "
                        "converter >= 1.2.2; incompatible layers now pass through.")
                try:
                    validate_w4_shape(k, group_size, convrot_groupsize)
                except PolicyError as e:
                    raise ValidationError(
                        f"{output_path}: invalid W4A8 shape metadata for "
                        f"{layer!r}: {e}") from e
            elif lfmt == FORMAT_W4A4:
                try:
                    if k % W4A4_QUANT_GROUP_SIZE != 0:
                        raise PolicyError(
                            f"K={k} not divisible by quant_group_size "
                            f"{W4A4_QUANT_GROUP_SIZE}")
                    if k % convrot_groupsize != 0:
                        raise PolicyError(
                            f"K={k} not divisible by convrot_groupsize "
                            f"{convrot_groupsize}")
                    if int(conf.get("quant_group_size", W4A4_QUANT_GROUP_SIZE)) \
                            != W4A4_QUANT_GROUP_SIZE:
                        raise PolicyError("quant_group_size must be 64")
                    if conf.get("linear_dtype", DEFAULT_W4A4_LINEAR_DTYPE) \
                            not in ("int4", "int8"):
                        raise PolicyError("linear_dtype must be int4 or int8")
                    if convrot_groupsize < 16 or \
                            not _is_power_of_four(convrot_groupsize):
                        raise PolicyError(
                            f"convrot_groupsize {convrot_groupsize} not a "
                            "supported power of 4")
                except PolicyError as e:
                    raise ValidationError(
                        f"{output_path}: invalid W4A4 shape metadata for "
                        f"{layer!r}: {e}") from e
            elif lfmt == FORMAT_INT8:
                pass  # int8_tensorwise without ConvRot has no shape constraint
        decisions.append(TensorDecision(
            source_name, DecisionKind.QUANTIZE, "reconstructed from output metadata",
            layer=layer, group_size=group_size,
            convrot_groupsize=convrot_groupsize, format=lfmt))
    quantized_names = {f"{layer}.weight" for layer in layers}
    if info is not None:
        for tensor in info.tensors:
            if tensor.name not in quantized_names:
                decisions.append(TensorDecision(
                    tensor.name, DecisionKind.KEEP, "reconstructed passthrough"))
    else:
        for name in names:
            if name.endswith(".weight") and name not in quantized_names:
                decisions.append(TensorDecision(
                    name, DecisionKind.KEEP, "reconstructed passthrough"))
    if info is not None:
        passthrough_dtype = None
        try:
            ext = json.loads(meta.get(METADATA_KEY_EXT, "{}"))
            if not isinstance(ext, dict) or not isinstance(
                    ext.get("quantization", {}), dict):
                raise ValidationError(
                    f"{output_path}: invalid {METADATA_KEY_EXT} metadata object")
            recorded_dtype = ext.get("quantization", {}).get(
                "passthrough_output_dtype", "auto")
            if recorded_dtype not in ("auto", "fp16", "bf16"):
                raise ValidationError(
                    f"{output_path}: invalid passthrough_output_dtype "
                    f"{recorded_dtype!r}")
            passthrough_dtype = {
                "auto": None, "fp16": torch.float16, "bf16": torch.bfloat16,
            }[recorded_dtype]
        except json.JSONDecodeError as e:
            raise ValidationError(
                f"{output_path}: invalid {METADATA_KEY_EXT} metadata: {e}") from e
        entries, expected_total = build_output_entries(
            info, decisions, fmt, passthrough_dtype)
    else:
        entries = actual_entries
        expected_total = sum(entry["nbytes"] for entry in actual_entries)
    plan = ConversionPlan(fmt=fmt, detection=detection, decisions=decisions,
                          metadata_quant=qm, metadata_ext={},
                          output_entries=entries, total_out_bytes=expected_total)
    plan.n_quantized = len([d for d in decisions if d.kind == DecisionKind.QUANTIZE])
    plan.n_kept = len(decisions) - plan.n_quantized
    return plan


def verify_output(output_path: str) -> Dict[str, Any]:
    """Source-free verification of an existing output checkpoint.

    Unlike ``--validation-only`` (which reconstructs the plan and compares the
    output against the original checkpoint), this operation needs no source
    model. It reopens the output, validates the tensor inventory, checks every
    quantized layer's metadata against the per-format tensor contracts
    (shapes, dtypes, group constraints), checks the native payload hash or
    W4A8/mixed packing roundtrip, and reports the file sha256. Intended for quick post-download or
    post-publish checks on machines that do not have the original model.
    """
    import math as _math
    checks: List[Dict[str, Any]] = []
    n_failed = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal n_failed
        if not ok:
            n_failed += 1
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    if not os.path.exists(output_path):
        check("output-exists", False, f"{output_path} missing")
        return {"ok": False, "checks": checks, "n_failed": 1,
                "output_sha256": "", "n_tensors": 0, "total_bytes": 0,
                "quantized_layers": 0, "formats": {}}

    total_bytes = os.path.getsize(output_path)
    try:
        st = safe_open(output_path, framework="pt")
        out_names = list(st.keys())
    except Exception as e:  # noqa: BLE001 - report any reopen failure
        check("output-reopen", False, f"{type(e).__name__}: {e}")
        return {"ok": False, "checks": checks, "n_failed": n_failed,
                "output_sha256": "", "n_tensors": 0,
                "total_bytes": total_bytes, "quantized_layers": 0,
                "formats": {}}
    meta = st.metadata() or {}
    check("output-reopen", True, f"{len(out_names)} tensors, {human_bytes(total_bytes)}")
    check("tensor-inventory", len(out_names) > 0, f"{len(out_names)} tensors")

    # Native formats have their own storage contracts and provenance hash.
    # Keep one source-free entry point for all checkpoints this script writes.
    if "comfy_quantizer" in meta:
        try:
            native = verify_native_output(output_path)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            native = {"ok": False, "errors": [f"{type(exc).__name__}: {exc}"]}
        check("native-metadata-and-payload", native["ok"],
              "; ".join(native["errors"]) or "native layout and payload hash valid")
        try:
            layers = read_quant_metadata(output_path)["layers"]
            formats = {}
            for conf in layers.values():
                fmt = conf["format"]
                formats[fmt] = formats.get(fmt, 0) + 1
        except (KeyError, TypeError, ValueError):
            layers, formats = {}, {}
        return {
            "ok": n_failed == 0, "checks": checks, "n_failed": n_failed,
            "n_tensors": len(out_names), "total_bytes": total_bytes,
            "quantized_layers": len(layers), "formats": formats,
            "output_sha256": sha256_file(output_path),
        }

    try:
        qm = read_quant_metadata(output_path)
    except (TypeError, ValueError) as e:
        qm = None
        check("quant-metadata", False, f"invalid {METADATA_KEY_QUANT}: {e}")
    layers = {}
    if isinstance(qm, dict) and isinstance(qm.get("layers"), dict):
        layers = qm["layers"]
        check("quant-metadata", len(layers) > 0,
              f"{len(layers)} quantized layer(s) recorded")
    else:
        check("quant-metadata", False, f"{METADATA_KEY_QUANT} missing layers object")

    if any(isinstance(conf, dict) and conf.get("format") == FORMAT_W6A8 for conf in layers.values()):
        try:
            extension = json.loads(meta[METADATA_KEY_EXT])
            expected_hash = extension["output"]["tensor_data_sha256"]
            check("tensor-payload-hash", expected_hash == sha256_safetensors_payload(output_path),
                  "W6A8 output payload integrity")
            groups = {conf.get("group_size") for conf in layers.values()
                      if isinstance(conf, dict) and conf.get("format") == FORMAT_W6A8}
            schema_ok = w6a8_extension_contract(extension, groups)
            check("w6a8-extension-contract", schema_ok, "six-bit format, revision, packing and group provenance")
        except (KeyError, TypeError, ValueError) as exc:
            check("tensor-payload-hash", False, str(exc))
    by_name = {n: st.get_slice(n) for n in out_names}
    fmt_counts: Dict[str, int] = {}
    n_quant = 0

    def dtype_of(name: str) -> torch.dtype:
        try:
            return torch_dtype_from_safe(by_name[name].get_dtype())
        except Exception:  # noqa: BLE001 - dtype lookup is informational
            return None

    def shape_of(name: str) -> Optional[Tuple[int, ...]]:
        try:
            return tuple(by_name[name].get_shape())
        except Exception:  # noqa: BLE001
            return None

    for layer, conf in layers.items():
        if not isinstance(layer, str) or not layer or not isinstance(conf, dict):
            check("layer-metadata", False, f"{layer!r}: malformed entry")
            continue
        lfmt = conf.get("format")
        if lfmt not in MIXED_FORMATS:
            check("layer-format", False, f"{layer}: unknown format {lfmt!r}")
            continue
        fmt_counts[lfmt] = fmt_counts.get(lfmt, 0) + 1
        n_quant += 1
        try:
            group_size = int(conf.get("group_size", 16))
            convrot_groupsize = int(conf.get("convrot_groupsize", 256))
        except (TypeError, ValueError):
            check("layer-groups", False, f"{layer}: invalid group metadata")
            continue
        wname = f"{layer}.weight"
        wshape = shape_of(wname)
        wdtype = dtype_of(wname)
        problems = []
        if wshape is None or len(wshape) != 2:
            problems.append(f"weight missing or not 2D ({wname})")
        else:
            n, kp = int(wshape[0]), int(wshape[1])
            if lfmt == FORMAT_W6A8:
                try:
                    n, k = w6a8_checkpoint_contract(st, layer, conf)
                except (PolicyError, KeyError, TypeError, ValueError, RuntimeError) as exc:
                    k = kp * 4 // 3
                    problems.append(str(exc))
            elif lfmt == FORMAT_W4A8:
                k = 2 * kp
                if wdtype is not None and wdtype != torch.int8:
                    problems.append(f"weight dtype {wdtype} != int8")
                if k % W4A8_CONVROT_GROUPSIZE != 0:
                    problems.append(f"K={k} not divisible by {W4A8_CONVROT_GROUPSIZE}")
                if convrot_groupsize != W4A8_CONVROT_GROUPSIZE:
                    problems.append(f"cgs={convrot_groupsize} != {W4A8_CONVROT_GROUPSIZE}")
                srel = shape_of(f"{layer}.weight_s_rel")
                if srel != (n, k // 16):
                    problems.append(f"weight_s_rel {srel} != [{n}, {k // 16}]")
                if dtype_of(f"{layer}.weight_s_rel") != torch.float8_e4m3fn:
                    problems.append("weight_s_rel not fp8_e4m3fn")
                sch = shape_of(f"{layer}.weight_s_channel")
                if sch != (n,):
                    problems.append(f"weight_s_channel {sch} != [{n}]")
                cb = shape_of(f"{layer}.weight_codebook")
                if cb != (16,):
                    problems.append(f"weight_codebook {cb} != [16]")
            elif lfmt == FORMAT_W4A4:
                k = 2 * kp
                if wdtype is not None and wdtype != torch.int8:
                    problems.append(f"weight dtype {wdtype} != int8")
                if k % W4A4_QUANT_GROUP_SIZE != 0:
                    problems.append(f"K={k} not divisible by {W4A4_QUANT_GROUP_SIZE}")
                if convrot_groupsize not in (16, 64, 256) or k % convrot_groupsize != 0:
                    problems.append(f"invalid cgs={convrot_groupsize} for K={k}")
                sc = shape_of(f"{layer}.weight_scale")
                if sc != (n,):
                    problems.append(f"weight_scale {sc} != [{n}]")
            else:  # int8_tensorwise
                k = kp
                if wdtype is not None and wdtype != torch.int8:
                    problems.append(f"weight dtype {wdtype} != int8")
                sc = shape_of(f"{layer}.weight_scale")
                if sc not in ((n, 1), (n,)):
                    problems.append(f"weight_scale {sc} != [{n}, 1]")
                if sc == (n,):
                    problems.append("weight_scale is [N]; runtime contract prefers [N, 1]")
        check(f"layer:{layer}", not problems, "; ".join(problems) or
              f"{lfmt} N={n} K={k} gs={group_size} cgs={convrot_groupsize}")

    # Bounded packing roundtrip: dequant a sample of rows for each quantized
    # layer and require finite, sane-magnitude output.
    sample_rows = 32
    roundtrip_fail = 0
    for layer, conf in layers.items():
        if not isinstance(conf, dict):
            continue
        lfmt = conf.get("format")
        if lfmt not in MIXED_FORMATS:
            continue
        wname = f"{layer}.weight"
        wshape = shape_of(wname)
        if wshape is None or len(wshape) != 2:
            continue
        n, kp = int(wshape[0]), int(wshape[1])
        r0 = min(n, sample_rows)
        try:
            packed = by_name[wname][:r0]
            if lfmt == FORMAT_W6A8:
                out = reference_decode_w6a8(packed,
                    by_name[f"{layer}.weight_s_rel"][:r0],
                    by_name[f"{layer}.weight_s_channel"][:r0],
                    conf.get("group_size"), conf.get("convrot_groupsize"))
                if not torch.equal(pack_w6(unpack_w6(packed)), packed):
                    raise PolicyError("W6A8 packing roundtrip differs")
            elif lfmt == FORMAT_W4A8:
                s_rel = by_name[f"{layer}.weight_s_rel"][:r0]
                s_ch = by_name[f"{layer}.weight_s_channel"][:r0]
                cb = (by_name[f"{layer}.weight_codebook"][:]
                      if f"{layer}.weight_codebook" in by_name else None)
                out = dequantize_w4a8_weight(
                    packed, s_rel, s_ch, codebook=cb, group_size=int(conf.get("group_size", 16)),
                    convrot_groupsize=int(conf.get("convrot_groupsize", 256)),
                    output_dtype=torch.float32)
            elif lfmt == FORMAT_W4A4:
                scale = by_name[f"{layer}.weight_scale"][:r0]
                out = dequantize_w4a4_weight(
                    packed, scale, int(conf.get("convrot_groupsize", 256)),
                    output_dtype=torch.float32)
            else:
                scale = by_name[f"{layer}.weight_scale"][:r0]
                scale = scale.reshape(-1, 1)
                out = dequantize_int8_tensorwise_weight(
                    packed, scale, output_dtype=torch.float32)
            finite = bool(torch.isfinite(out).all())
            sane = bool(float(out.abs().max()) <= 1e4)
            if not (finite and sane):
                roundtrip_fail += 1
                check(f"roundtrip:{layer}", False,
                      f"finite={finite} max_abs={float(out.abs().max()):.3e}")
        except Exception as e:  # noqa: BLE001 - packing failures are the finding
            roundtrip_fail += 1
            check(f"roundtrip:{layer}", False, f"{type(e).__name__}: {e}")
    if roundtrip_fail == 0 and n_quant:
        check("packing-roundtrip", True, f"sampled {min(sample_rows, n_quant)} layer(s), all finite")

    # Extension metadata consistency (informational; never fails the check).
    ext = meta.get(METADATA_KEY_EXT)
    if ext:
        try:
            read_optimization_metadata(json.loads(ext).get('optimization'))
            check('optimization-metadata', True, 'optional optimizer provenance valid')
        except (TypeError, ValueError, UsageError, AttributeError) as exc:
            check('optimization-metadata', False, str(exc))
        try:
            ext_obj = json.loads(ext)
            ext_counts = None
            q_ext = ext_obj.get("quantization") or {}
            dist = q_ext.get("distribution") or {}
            if isinstance(dist.get("counts"), dict):
                ext_counts = {str(k): int(v) for k, v in dist["counts"].items()}
            if ext_counts is not None and ext_counts != fmt_counts:
                check("extension-metadata", True,
                      f"{METADATA_KEY_EXT} distribution {ext_counts} differs from "
                      f"{METADATA_KEY_QUANT} {fmt_counts} (informational)")
        except (TypeError, json.JSONDecodeError) as e:
            check("extension-metadata", True, f"unparsable {METADATA_KEY_EXT}: {e}")

    out_sha = sha256_file(output_path)
    ok = n_failed == 0
    return {
        "ok": ok,
        "checks": checks,
        "n_failed": n_failed,
        "n_tensors": len(out_names),
        "total_bytes": total_bytes,
        "quantized_layers": n_quant,
        "formats": fmt_counts,
        "output_sha256": out_sha,
    }
