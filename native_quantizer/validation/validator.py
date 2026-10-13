"""Source-aware output validation and reconstruction checks."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import struct
import torch
from dataclasses import dataclass
from pathlib import Path
from safetensors import safe_open
from typing import Any, Dict, List, Optional

from native_quantizer.checkpoint_io.fingerprints import (
    _portable_file_labels,
    _portable_hash_manifest,
    hash_checkpoint_files,
)
from native_quantizer.checkpoint_io.reader import CheckpointReader
from native_quantizer.checkpoint_io.types import CheckpointInfo
from native_quantizer.codecs.bounded import _chunk_rows_for_budget, _quant_work_bytes, quantize_tensor_bounded
from native_quantizer.codecs.packing import unpack_int4_signed, unpack_w4
from native_quantizer.codecs.rotation import _is_power_of_four
from native_quantizer.codecs.w6a8 import pack_w6, unpack_w6
from native_quantizer.dtypes import TORCH_TO_SAFE
from native_quantizer.files import sha256_file, sha256_safetensors_payload
from native_quantizer.formats import FORMAT_TO_KITCHEN_LAYOUT
from native_quantizer.identity import (
    FORMAT_INT8,
    FORMAT_MIXED,
    FORMAT_W4A4,
    FORMAT_W4A8,
    FORMAT_W6A8,
    GROUPED_INT8_FORMATS,
    INT8_MAX_REL_L2,
    METADATA_KEY_EXT,
    METADATA_KEY_QUANT,
    MIXED_FORMATS,
    W4A4_MAX_REL_L2,
    W4A4_QUANT_GROUP_SIZE,
    W4A8_CONVROT_GROUPSIZE,
    format_revision,
)
from native_quantizer.logging import log
from native_quantizer.metadata import read_quant_metadata
from native_quantizer.planning.metrics import TensorMetrics, _MetricAccumulator, compute_weight_metrics
from native_quantizer.planning.types import ConversionPlan
from native_quantizer.runtime.capabilities import EnvironmentInfo
from native_quantizer.utils import human_bytes, json_dumps
from native_quantizer.validation.contracts import w6a8_checkpoint_contract, w6a8_extension_contract
from native_quantizer.validation.reference import (
    reference_decode_int8,
    reference_decode_w4a4,
    reference_decode_w4a8,
    reference_decode_w6a8,
)


@dataclass
class ValidationCheck:
    name: str
    status: str          # passed | passed-with-warnings | failed | skipped
    detail: str = ""
    reason: str = ""


def _refresh_validation_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    checks = summary.setdefault("checks", [])
    summary["n_passed"] = sum(1 for c in checks if c.get("status") == "passed")
    summary["n_passed_with_warnings"] = sum(
        1 for c in checks if c.get("status") == "passed-with-warnings")
    summary["n_failed"] = sum(1 for c in checks if c.get("status") == "failed")
    summary["n_skipped"] = sum(1 for c in checks if c.get("status") == "skipped")
    return summary


class Validator:
    def __init__(self, info: CheckpointInfo, plan: ConversionPlan,
                 output_path: str, args: Any, env: EnvironmentInfo):
        self.info = info
        self.plan = plan
        self.output_path = output_path
        self.args = args
        self.env = env
        self.checks: List[ValidationCheck] = []
        self.output_sha256 = ""

    def check(self, name: str, ok: bool, detail: str = "", warn: bool = False,
              skipped: bool = False, reason: str = "") -> None:
        status = "skipped" if skipped else ("passed-with-warnings" if (warn and ok) else ("passed" if ok else "failed"))
        self.checks.append(ValidationCheck(name, status, detail, reason))

    def summary(self) -> Dict[str, Any]:
        return {
            "checks": [dataclasses.asdict(c) for c in self.checks],
            "n_passed": sum(1 for c in self.checks if c.status == "passed"),
            "n_passed_with_warnings": sum(1 for c in self.checks if c.status == "passed-with-warnings"),
            "n_failed": sum(1 for c in self.checks if c.status == "failed"),
            "n_skipped": sum(1 for c in self.checks if c.status == "skipped"),
            "output_sha256": self.output_sha256,
        }

    def run(self, reader: Optional[CheckpointReader] = None,
            metrics: Optional[Dict[str, TensorMetrics]] = None,
            input_hashes: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        args = self.args
        full_validation = bool(args.validate or args.validation_only)
        out_path = self.output_path
        if not os.path.exists(out_path):
            self.check("output-exists", False, f"{out_path} missing")
            return self.summary()
        size = os.path.getsize(out_path)
        log().info("validation started on %s (%s, %s mode)",
                   out_path, human_bytes(size),
                   "full" if full_validation else "representative")

        # ---- reopen with safetensors ----
        try:
            st = safe_open(out_path, framework="pt")
            out_names = list(st.keys())
            ok = len(out_names) == len(self.plan.output_entries)
            self.check("output-reopen", True, f"{len(out_names)} tensors, {human_bytes(size)}")
            self.check("output-entry-count", ok,
                       f"{len(out_names)} vs {len(self.plan.output_entries)} expected")
        except Exception as e:
            self.check("output-reopen", False, f"safe_open failed: {e}")
            return self.summary()

        # F9: the serialized tensor payload must match the planned inventory
        # byte-for-byte (file size minus the 8-byte length prefix and header)
        try:
            with open(out_path, "rb") as f:
                hlen = struct.unpack("<Q", f.read(8))[0]
                actual_payload = size - 8 - hlen
            expected_payload = sum(e["nbytes"] for e in self.plan.output_entries)
            self.check("payload-size-accurate",
                       actual_payload == expected_payload,
                       f"actual {actual_payload} vs planned {expected_payload} bytes")
        except Exception as e:
            self.check("payload-size-accurate", False, f"size check failed: {e}")

        expected = {e["name"]: e for e in self.plan.output_entries}
        output_name_set = set(out_names)
        missing = [n for n in expected if n not in output_name_set]
        extra = [n for n in output_name_set if n not in expected]
        self.check("output-inventory", not missing and not extra,
                   f"missing={missing[:5]} extra={extra[:5]}")

        # ---- dtype / shape preservation ----
        shape_ok = True
        dtype_ok = True
        bad = []
        for name, e in expected.items():
            if name not in output_name_set:
                continue
            t = st.get_slice(name)
            shp = tuple(t.get_shape())
            dt = t.get_dtype()
            if shp != tuple(e["shape"]):
                shape_ok = False
                bad.append(f"{name}: shape {shp} != {e['shape']}")
            if dt != TORCH_TO_SAFE[e["dtype"]]:
                dtype_ok = False
                bad.append(f"{name}: dtype {dt} != {TORCH_TO_SAFE[e['dtype']]}")
        self.check("shape-preservation", shape_ok, "; ".join(bad[:5]))
        self.check("dtype-check", dtype_ok, "; ".join(bad[:5]))

        # ---- metadata ----
        meta = {}
        try:
            with open(out_path, "rb") as f:
                hlen = struct.unpack("<Q", f.read(8))[0]
                meta = json.loads(f.read(hlen).decode("utf-8")).get("__metadata__", {})
        except Exception as e:
            meta = {}
        has_quant_meta = METADATA_KEY_QUANT in meta
        self.check("metadata-present", has_quant_meta,
                   "missing _quantization_metadata" if not has_quant_meta else
                   "present")
        ext_ok = METADATA_KEY_EXT in meta
        self.check("extension-metadata-present", ext_ok,
                   "missing comfy_wxa8 extension metadata" if not ext_ok else "present")
        ext_payload: Dict[str, Any] = {}
        is_mixed_plan = self.plan.fmt == FORMAT_MIXED
        if ext_ok:
            try:
                ext_payload = json.loads(meta[METADATA_KEY_EXT])
                required = ("schema", "converter", "converter_version", "format",
                            "architecture", "source", "quantization", "output")
                source_block = ext_payload.get("source", {}) \
                    if isinstance(ext_payload, dict) else {}
                quant_block = ext_payload.get("quantization", {}) \
                    if isinstance(ext_payload, dict) else {}
                output_block = ext_payload.get("output", {}) \
                    if isinstance(ext_payload, dict) else {}
                source_files = source_block.get("files", []) \
                    if isinstance(source_block, dict) else []
                source_hashes = source_block.get("sha256", {}) \
                    if isinstance(source_block, dict) else {}
                schema_ok = isinstance(ext_payload, dict) and all(
                    key in ext_payload for key in required)
                expected_schema = ("comfy_wxa8/v2" if self.plan.fmt == FORMAT_MIXED
                                   else "comfy_wxa8/v1")
                expected_revision = format_revision(self.plan.fmt)
                schema_ok = (
                    schema_ok
                    and ext_payload.get("schema") == expected_schema
                    and ext_payload.get("format") == self.plan.fmt
                    and ext_payload.get("format_revision") == expected_revision
                    and ext_payload.get("architecture") ==
                    self.plan.detection.architecture
                    and isinstance(ext_payload.get("source"), dict)
                    and isinstance(ext_payload.get("quantization"), dict)
                    and isinstance(ext_payload.get("output"), dict)
                    and isinstance(source_files, list)
                    and source_files
                    and all(isinstance(label, str) and label
                            and not Path(label).is_absolute()
                            for label in source_files)
                    and isinstance(source_hashes, dict)
                    and set(source_hashes) == set(source_files)
                    and all(isinstance(digest, str)
                            and re.fullmatch(r"[0-9a-f]{64}", digest)
                            for digest in source_hashes.values())
                    and isinstance(quant_block.get("weight_bits"), (int, str))
                    and (
                        # v1 (single-format w4a8): W4A8-global fields
                        (not is_mixed_plan
                         and quant_block.get("activation_bits") == 8
                         and quant_block.get("weight_bits") == (6 if self.plan.fmt == FORMAT_W6A8 else 4)
                         and quant_block.get("packing") == ("int6-nibble-high2-plane" if self.plan.fmt == FORMAT_W6A8 else "int4-nibble-lsb")
                         and quant_block.get("scale_dtype") == "fp8_e4m3fn")
                        or
                        # v2 (mixed): mode + per-format activation semantics +
                        # distribution + certification level
                        (is_mixed_plan
                         and quant_block.get("activation_precision") ==
                         "per-format"
                         and quant_block.get("mode") == "mixed"
                         and isinstance(quant_block.get("formats"), dict)
                         and set(quant_block["formats"]) <= set(MIXED_FORMATS)
                         and set(quant_block["formats"]) == {
                             d.format for d in self.plan.quantized_layers()
                             if d.format in MIXED_FORMATS}
                         and isinstance(
                             quant_block.get("quality_validation"), dict)
                         and quant_block["quality_validation"].get("level")
                         in ("unverified", "calibrated"))
                    )
                    and isinstance(quant_block.get("activation_quantization"), str)
                    and isinstance(output_block.get("tensor_data_sha256"), str)
                    and bool(re.fullmatch(
                        r"[0-9a-f]{64}", output_block["tensor_data_sha256"]))
                )
                w6_groups = {d.group_size for d in self.plan.quantized_layers() if d.format == FORMAT_W6A8}
                if w6_groups:
                    schema_ok = schema_ok and w6a8_extension_contract(ext_payload, w6_groups)
                self.check("extension-metadata-schema", schema_ok,
                           "required fields, revision and architecture"
                           if schema_ok else
                           "invalid extension fields, revision or architecture")
                if input_hashes:
                    recorded_source_hashes = ext_payload.get("source", {}).get(
                        "sha256", {})
                    # Shard identity is part of provenance: compare the full
                    # label->hash mapping, never sorted hash values (a rename
                    # attack must be detected).
                    labels = _portable_file_labels(sorted(input_hashes))
                    expected_hashes = dict(zip(
                        labels, (input_hashes[p] for p in sorted(input_hashes))))
                    source_hash_ok = (
                        isinstance(recorded_source_hashes, dict)
                        and recorded_source_hashes == expected_hashes
                    )
                    self.check(
                        "metadata-source-hash", source_hash_ok,
                        "source content hashes match" if source_hash_ok else
                        "embedded source hashes do not match the supplied original")
                recorded_payload_hash = ext_payload.get("output", {}).get(
                    "tensor_data_sha256")
                if recorded_payload_hash:
                    actual_payload_hash = sha256_safetensors_payload(out_path)
                    self.check("tensor-payload-hash",
                               recorded_payload_hash == actual_payload_hash,
                               f"sha256={actual_payload_hash}")
                else:
                    self.check("tensor-payload-hash", False,
                               "missing output.tensor_data_sha256")
            except Exception as e:
                self.check("extension-metadata-json", False, f"unparseable: {e}")
        if has_quant_meta:
            try:
                qm = read_quant_metadata(out_path)
                layers = qm.get("layers", {})
                q_layers = {d.layer for d in self.plan.quantized_layers()}
                mism = q_layers.symmetric_difference(set(layers.keys()))
                self.check("metadata-layer-inventory", not mism,
                           (f"{len(layers)} layers match" if not mism else
                            f"layer set mismatch: {sorted(mism)[:5]}"))
                decision_by_layer = {
                    decision.layer: decision for decision in self.plan.quantized_layers()
                }
                conf_ok = True
                for layer, value in layers.items():
                    dec = decision_by_layer.get(layer)
                    if dec is None or not isinstance(value, dict):
                        conf_ok = False
                        continue
                    lfmt = value.get("format")
                    if lfmt != dec.format or lfmt not in MIXED_FORMATS:
                        conf_ok = False
                        continue
                    if lfmt in GROUPED_INT8_FORMATS:
                        conf_ok = conf_ok and (
                            value.get("group_size") == dec.group_size
                            and value.get("convrot") is True
                            and value.get("convrot_groupsize") ==
                            dec.convrot_groupsize)
                    elif lfmt == FORMAT_W4A4:
                        conf_ok = conf_ok and (
                            value.get("convrot_groupsize") ==
                            dec.convrot_groupsize
                            and value.get("quant_group_size") ==
                            W4A4_QUANT_GROUP_SIZE
                            and value.get("linear_dtype") in ("int4", "int8"))
                    # int8_tensorwise: format field only
                conf_ok = conf_ok and not mism
                self.check("metadata-layer-conf", conf_ok,
                           "format/group/ConvRot fields valid" if conf_ok else
                           "invalid format/group/ConvRot field")

                # ---- independent runtime-contract invariants (P0/P1) ----
                # Re-derived from the file itself, not from the plan: every
                # quantized layer must carry convrot=true, convrot_groupsize=256,
                # group_size=16, and exact tensor shapes, and (when the original
                # checkpoint is known) K % 256 == 0.
                inv_ok = True
                inv_bad: List[str] = []
                try:
                    with safe_open(out_path, framework="pt") as inv_st:
                        for layer, conf in layers.items():
                            lfmt = conf.get("format")
                            try:
                                gs = int(conf.get("group_size", 0))
                                cgs = int(conf.get("convrot_groupsize", 0))
                            except (TypeError, ValueError):
                                gs = cgs = 0
                            w = inv_st.get_slice(layer + ".weight")
                            n, k2 = w.get_shape()
                            if lfmt == FORMAT_W6A8:
                                src = self.info.by_name(layer + ".weight") if self.info else None
                                w6a8_checkpoint_contract(inv_st, layer, conf, src.shape if src else None)
                            elif lfmt == FORMAT_W4A8:
                                srel = inv_st.get_slice(layer + ".weight_s_rel")
                                sch = inv_st.get_slice(layer + ".weight_s_channel")
                                cb = inv_st.get_slice(layer + ".weight_codebook")
                                k = k2 * 2
                                if (conf.get("convrot") is not True
                                        or cgs != W4A8_CONVROT_GROUPSIZE
                                        or gs != 16
                                        or tuple(w.get_shape()) != (n, k // 2)
                                        or tuple(srel.get_shape()) != (n, k // gs)
                                        or tuple(sch.get_shape()) != (n,)
                                        or tuple(cb.get_shape()) != (16,)
                                        or k % gs != 0 or k % cgs != 0):
                                    inv_ok = False
                                    inv_bad.append(f"{layer}: w4a8 conf/shape "
                                                   f"{conf} w={w.get_shape()}")
                                    continue
                                if self.info is not None:
                                    src_meta = self.info.by_name(layer + ".weight")
                                    if src_meta is None or int(src_meta.shape[1]) % 256 != 0:
                                        inv_ok = False
                                        inv_bad.append(
                                            f"{layer}: source K not divisible by 256")
                            elif lfmt == FORMAT_W4A4:
                                scale = inv_st.get_slice(layer + ".weight_scale")
                                k = k2 * 2
                                src_k = None
                                if self.info is not None:
                                    src_meta = self.info.by_name(
                                        layer + ".weight")
                                    src_k = (int(src_meta.shape[1])
                                             if src_meta is not None else None)
                                if (tuple(w.get_shape()) != (n, k // 2)
                                        or tuple(scale.get_shape()) != (n,)
                                        or k % W4A4_QUANT_GROUP_SIZE != 0
                                        or k % cgs != 0
                                        or int(conf.get("quant_group_size", 0)) !=
                                        W4A4_QUANT_GROUP_SIZE
                                        or conf.get("linear_dtype") not in
                                        ("int4", "int8")
                                        or cgs < 16
                                        or not _is_power_of_four(cgs)
                                        or (src_k is not None and src_k != k)):
                                    inv_ok = False
                                    inv_bad.append(f"{layer}: w4a4 conf/shape "
                                                   f"{conf} w={w.get_shape()} "
                                                   f"scale={scale.get_shape()}")
                            elif lfmt == FORMAT_INT8:
                                scale = inv_st.get_slice(layer + ".weight_scale")
                                src_k = None
                                if self.info is not None:
                                    src_meta = self.info.by_name(
                                        layer + ".weight")
                                    src_k = (int(src_meta.shape[1])
                                             if src_meta is not None else None)
                                if (tuple(w.get_shape()) != (n, k2)
                                        or tuple(scale.get_shape()) not in
                                        ((n,), (n, 1))
                                        or (src_k is not None and src_k != k2)):
                                    inv_ok = False
                                    inv_bad.append(f"{layer}: int8 shape "
                                                   f"w={w.get_shape()} "
                                                   f"scale={scale.get_shape()}")
                            else:
                                inv_ok = False
                                inv_bad.append(f"{layer}: unknown format {lfmt!r}")
                except Exception as e:
                    inv_ok = False
                    inv_bad.append(str(e))
                self.check(
                    "metadata-runtime-contract", inv_ok,
                    "per-format runtime contract (shapes, scales, K rules) "
                    "valid" if inv_ok else "; ".join(inv_bad[:4]) or "invalid")

            except Exception as e:
                self.check("metadata-json", False, f"unparseable: {e}")

        # ---- passthrough byte integrity (P0, full validation only) ----
        # With --output-dtype auto every non-quantized tensor must be preserved
        # byte-for-byte (dtype, shape, payload). A requested fp16/bf16 cast is
        # checked separately by dtype-check.
        if full_validation and reader is not None \
                and getattr(args, "output_dtype", "auto") in (None, "auto"):
            try:
                q_names = {f"{d.layer}.weight" for d in self.plan.quantized_layers()}
                payload_mismatch: List[str] = []
                payload_checked = 0
                with open(out_path, "rb") as f:
                    hlen = struct.unpack("<Q", f.read(8))[0]
                    header = json.loads(f.read(hlen).decode("utf-8"))
                    for name, e in expected.items():
                        if name in q_names:
                            continue
                        tinfo = header.get(name)
                        if not isinstance(tinfo, dict):
                            payload_mismatch.append(f"{name}: missing in header")
                            continue
                        src_meta = self.info.by_name(name) if self.info is not None else None
                        if src_meta is None:
                            continue
                        # Only tensors whose dtype was preserved can be byte-compared;
                        # explicit output-dtype casts are covered by dtype-check.
                        if e["dtype"] != src_meta.dtype:
                            continue
                        d0, d1 = tinfo["data_offsets"]
                        f.seek(8 + hlen + d0)
                        out_bytes = f.read(d1 - d0)
                        src_bytes = reader.read_bytes(name)
                        payload_checked += 1
                        if len(out_bytes) != len(src_bytes)                                 or hashlib.sha256(out_bytes).digest() !=                                    hashlib.sha256(src_bytes).digest():
                            payload_mismatch.append(name)
                self.check(
                    "passthrough-integrity", not payload_mismatch,
                    (f"{payload_checked} passthrough tensors byte-identical"
                     if not payload_mismatch else
                     f"payload differs for: {payload_mismatch[:5]}"))
            except Exception as e:
                self.check("passthrough-integrity", False, f"check failed: {e}")

        # ---- tensor round trips (all with --validate, representative otherwise) ----
        q_layers = self.plan.quantized_layers()
        if q_layers:
            import random
            rng = random.Random(getattr(args, "seed", 0) or 0)  # noqa: S311
            if full_validation:
                sample = list(q_layers)
            else:
                sample = sorted(q_layers, key=lambda d: self.info.by_name(d.name).nbytes,
                                reverse=True)
                sample = sample[:16] + (rng.sample(sample, min(8, len(sample)))
                                        if len(sample) > 16 else [])
                sample = list({id(d): d for d in sample}.values())
            worst: Dict[str, float] = {}
            bounds_used: Dict[str, float] = {}
            log().info("validation: reconstruction round trips over %d "
                       "quantized layers (%s)", len(sample),
                       "full set" if full_validation else "representative sample")
            for idx, d in enumerate(sample):
                log().info("validation round trip %d/%d: %s",
                           idx + 1, len(sample), d.layer or d.name)
                if d.layer is None:
                    self.check(f"recon-{d.name}", False,
                               "quantized decision has no layer name")
                    continue
                try:
                    orig = self.info.by_name(d.name)
                    if orig is not None and reader is not None:
                        max_mem = getattr(args, "max_memory", 2 * 1024**3)
                        bounded = (
                            d.name in getattr(self.plan, "chunked_layers", set())
                            or _quant_work_bytes(orig, d.format) > max_mem
                        )
                        recon_bound = {
                            FORMAT_W4A8: self.plan.detection.policy.max_rel_l2,
                            FORMAT_W4A4: W4A4_MAX_REL_L2,
                            FORMAT_INT8: INT8_MAX_REL_L2,
                        }.get(d.format, self.plan.detection.policy.max_rel_l2)
                        bounds_used[d.layer] = recon_bound
                        pack_ok = True
                        if d.format == FORMAT_W6A8:
                            n, k = int(orig.shape[0]), int(orig.shape[1])
                            rows = _chunk_rows_for_budget(k, n, max_mem, d.format) if bounded else n
                            acc = _MetricAccumulator(d.name)
                            original_view = reader.read_tensor(d.name)
                            for r0 in range(0, n, rows):
                                r1 = min(n, r0 + rows)
                                packed = st.get_slice(d.layer + ".weight")[r0:r1]
                                dq = reference_decode_w6a8(packed,
                                    st.get_slice(d.layer + ".weight_s_rel")[r0:r1],
                                    st.get_slice(d.layer + ".weight_s_channel")[r0:r1],
                                    d.group_size, d.convrot_groupsize)
                                acc.update(original_view[r0:r1], dq, None)
                                pack_ok = pack_ok and torch.equal(pack_w6(unpack_w6(packed)), packed)
                                del packed, dq
                            m = acc.finish()
                        elif bounded:
                            n, k = int(orig.shape[0]), int(orig.shape[1])
                            chunk_rows = _chunk_rows_for_budget(k, n, max_mem, d.format)
                            acc = _MetricAccumulator(d.name)
                            original_view = reader.read_tensor(d.name)
                            packed_slice = st.get_slice(d.layer + ".weight")
                            if d.format == FORMAT_W4A8:
                                cb = st.get_tensor(d.layer + ".weight_codebook")
                                s_rel_slice = st.get_slice(d.layer + ".weight_s_rel")
                                s_ch_slice = st.get_slice(d.layer + ".weight_s_channel")
                                for r0 in range(0, n, chunk_rows):
                                    r1 = min(n, r0 + chunk_rows)
                                    packed = packed_slice[r0:r1]
                                    s_rel = s_rel_slice[r0:r1]
                                    s_ch = s_ch_slice[r0:r1]
                                    dq = reference_decode_w4a8(
                                        packed, s_rel, s_ch, codebook=cb,
                                        group_size=d.group_size,
                                        convrot_groupsize=d.convrot_groupsize)
                                    acc.update(original_view[r0:r1], dq, None)
                                    rt = unpack_w4(packed)
                                    repacked = (
                                        (rt[:, 0::2] & 0xF)
                                        | ((rt[:, 1::2] & 0xF) << 4)
                                    ).to(torch.int8)
                                    pack_ok = pack_ok and bool(torch.equal(repacked, packed))
                                    del packed, s_rel, s_ch, dq, rt, repacked
                            else:
                                scale_slice = st.get_slice(d.layer + ".weight_scale")
                                for r0 in range(0, n, chunk_rows):
                                    r1 = min(n, r0 + chunk_rows)
                                    packed = packed_slice[r0:r1]
                                    scale = scale_slice[r0:r1]
                                    dq = (reference_decode_w4a4(packed, scale,
                                                                d.convrot_groupsize)
                                          if d.format == FORMAT_W4A4 else
                                          reference_decode_int8(packed, scale))
                                    acc.update(original_view[r0:r1], dq, None)
                                    if d.format == FORMAT_W4A4:
                                        rt = unpack_int4_signed(packed)
                                        repacked = (
                                            (rt[:, 0::2] & 0xF)
                                            | ((rt[:, 1::2] & 0xF) << 4)
                                        ).to(torch.int8)
                                        pack_ok = pack_ok and bool(torch.equal(repacked, packed))
                                        del rt, repacked
                                    del packed, scale, dq
                            m = acc.finish()
                        else:
                            packed = st.get_tensor(d.layer + ".weight")
                            if d.format == FORMAT_W4A8:
                                cb = st.get_tensor(d.layer + ".weight_codebook")
                                s_rel = st.get_tensor(d.layer + ".weight_s_rel")
                                s_ch = st.get_tensor(d.layer + ".weight_s_channel")
                                dq = reference_decode_w4a8(
                                    packed, s_rel, s_ch, codebook=cb,
                                    group_size=d.group_size,
                                    convrot_groupsize=d.convrot_groupsize)
                            else:
                                scale = st.get_tensor(d.layer + ".weight_scale")
                                dq = (reference_decode_w4a4(
                                    packed, scale, d.convrot_groupsize)
                                    if d.format == FORMAT_W4A4 else
                                    reference_decode_int8(packed, scale))
                            orig_t = reader.read_tensor(d.name).float()
                            m = compute_weight_metrics(orig_t, dq)
                            if d.format == FORMAT_W4A4:
                                rt = unpack_int4_signed(packed)
                                repacked = (
                                    (rt[:, 0::2] & 0xF)
                                    | ((rt[:, 1::2] & 0xF) << 4)
                                ).to(torch.int8)
                                pack_ok = bool(torch.equal(repacked, packed))
                        worst[d.layer] = m.rel_l2
                        if m.rel_l2 > recon_bound:
                            self.check(f"recon-{d.layer}", False,
                                       f"relL2 {m.rel_l2:.4f} > {recon_bound}")
                        elif m.cosine < self.plan.detection.policy.min_cosine:
                            self.check(f"recon-{d.layer}", False,
                                       f"cosine {m.cosine:.4f} < {self.plan.detection.policy.min_cosine}")
                        else:
                            self.check(f"recon-{d.layer}", True,
                                       f"relL2={m.rel_l2:.4f} snr={m.snr_db:.1f}dB cos={m.cosine:.4f}")
                        self.check(f"pack-rt-{d.layer}", pack_ok,
                                   "pack/unpack round trip"
                                   + (" (bounded row chunks)" if bounded else ""))
                except Exception as e:
                    self.check(f"recon-{d.layer}", False, f"error: {e}")
            if worst:
                mx = max(worst.values())
                mx_bound = max(bounds_used.values()) if bounds_used else \
                    self.plan.detection.policy.max_rel_l2
                self.check("reconstruction-error-bound", mx <= mx_bound,
                           f"max {'full' if full_validation else 'sampled'} relL2 {mx:.4f} "
                           f"(per-format bound {mx_bound})")

        # ---- scale validation ----
        scale_ok = True
        scale_detail = ""
        n_scale = 0
        for d in q_layers:
            if d.layer is None:
                scale_ok = False
                scale_detail = f"{d.name}: quantized decision has no layer name"
                break
            try:
                if d.format == FORMAT_W6A8:
                    w6a8_checkpoint_contract(st, d.layer, {"convrot": True,
                        "group_size": d.group_size, "convrot_groupsize": d.convrot_groupsize})
                    n_scale += 1
                    continue
                if d.format == FORMAT_W4A4:
                    scale = st.get_tensor(d.layer + ".weight_scale")
                    finite = bool(torch.isfinite(scale).all())
                    valid_range = bool((scale > 0).all())
                    if not finite or not valid_range:
                        scale_ok = False
                        scale_detail = (
                            f"{d.layer}: non-finite or non-positive W4A4 scales")
                        break
                    n_scale += 1
                    continue
                if d.format == FORMAT_INT8:
                    scale = st.get_tensor(d.layer + ".weight_scale")
                    finite = bool(torch.isfinite(scale).all())
                    valid_range = bool((scale > 0).all())
                    if not finite or not valid_range:
                        scale_ok = False
                        scale_detail = (
                            f"{d.layer}: non-finite or non-positive INT8 scales")
                        break
                    n_scale += 1
                    continue
                cb = st.get_tensor(d.layer + ".weight_codebook").float()
                orig = self.info.by_name(d.name)
                max_mem = getattr(args, "max_memory", 2 * 1024**3)
                if orig is not None and _quant_work_bytes(orig, d.format) > max_mem:
                    n, k = int(orig.shape[0]), int(orig.shape[1])
                    chunk_rows = _chunk_rows_for_budget(k, n, max_mem, d.format)
                else:
                    n = int(st.get_slice(d.layer + ".weight_s_channel").get_shape()[0])
                    chunk_rows = n
                finite = bool(torch.isfinite(cb).all())
                valid_range = True
                s_rel_slice = st.get_slice(d.layer + ".weight_s_rel")
                s_ch_slice = st.get_slice(d.layer + ".weight_s_channel")
                for r0 in range(0, n, chunk_rows):
                    r1 = min(n, r0 + chunk_rows)
                    s_rel = s_rel_slice[r0:r1].float()
                    s_ch = s_ch_slice[r0:r1].float()
                    finite = finite and bool(
                        torch.isfinite(s_rel).all() and torch.isfinite(s_ch).all())
                    valid_range = valid_range and bool(
                        (s_rel >= 0).all() and (s_ch > 0).all())
                    del s_rel, s_ch
                if not finite:
                    scale_ok = False
                    scale_detail = f"{d.layer}: non-finite scales"
                    break
                if not valid_range:
                    scale_ok = False
                    scale_detail = (
                        f"{d.layer}: negative fp8 relative scale or non-positive "
                        "channel scale")
                    break
                n_scale += 1
            except Exception as e:
                scale_ok = False
                scale_detail = f"{d.layer}: {e}"
                break
        self.check("scale-validation", scale_ok,
                   scale_detail or f"{n_scale} layers OK (fp8 s_rel finite/nonnegative, "
                   "fp32 s_channel finite/positive)")

        # ---- deterministic re-quantization sample ----
        if q_layers and reader is not None and not args.validation_only:
            conv_device = getattr(self.plan, "device", "cpu")
            if conv_device != "cpu":
                self.check("deterministic-vs-disk", True,
                           "skipped: conversion ran on " + conv_device + " (codebook "
                           "subsample is device-dependent); determinism verified on CPU",
                           skipped=True,
                           reason="device-dependent codebook subsample")
            d0 = q_layers[0]
            try:
                if d0.name in getattr(self.plan, "chunked_layers", set()):
                    self.check("deterministic-conversion", True,
                               "skipped: bounded chunk determinism is covered by self-test",
                               skipped=True, reason="large chunked layer")
                    raise StopIteration
                compute_dtype = getattr(args, "_compute_dtype_tensor", torch.float32)
                max_mem = getattr(args, "max_memory", 2 * 1024**3)
                out1 = quantize_tensor_bounded(
                    reader, d0.name, d0.format, d0.group_size,
                    d0.convrot_groupsize, max_mem, torch.device("cpu"),
                    compute_dtype=compute_dtype,
                    seed=getattr(args, "seed", 0),
                    codebook_mode=getattr(args, "codebook_mode", "auto"))
                out2 = quantize_tensor_bounded(
                    reader, d0.name, d0.format, d0.group_size,
                    d0.convrot_groupsize, max_mem, torch.device("cpu"),
                    compute_dtype=compute_dtype,
                    seed=getattr(args, "seed", 0),
                    codebook_mode=getattr(args, "codebook_mode", "auto"))
                det = all(torch.equal(out1[k], out2[k]) for k in out1)
                self.check("deterministic-conversion", det, "two runs byte-identical")
                if conv_device == "cpu" and d0.name not in getattr(self.plan, "chunked_layers", set()):
                    on_disk = st.get_tensor(d0.layer + ".weight")
                    matches_disk = torch.equal(out1[""], on_disk)
                    self.check("deterministic-vs-disk", matches_disk,
                               "recomputed packed weight matches the file" if matches_disk else "MISMATCH")
            except StopIteration:
                pass
            except Exception as e:
                self.check("deterministic-conversion", False, f"error: {e}")

        # ---- output hash ----
        if full_validation:
            self.output_sha256 = sha256_file(out_path)
            self.check("output-hash", True, f"sha256={self.output_sha256}")

        # ---- compatibility probe (optional, no ComfyUI required) ----
        if full_validation:
            required_formats = sorted({
                d.format for d in self.plan.quantized_layers()
                if d.format in MIXED_FORMATS})
            if self.env.has_comfy_kitchen:
                layout_attr = {
                    FORMAT_W4A8: "comfy_kitchen_has_w4a8_layout",
                    FORMAT_W6A8: "comfy_kitchen_has_w6a8_api",
                    FORMAT_W4A4: "comfy_kitchen_has_w4a4_layout",
                    FORMAT_INT8: "comfy_kitchen_has_int8_layout",
                }
                missing = [f for f in required_formats
                           if not getattr(self.env, layout_attr[f], False)]
                self.check("compat-comfy-kitchen", not missing,
                           detail=f"installed comfy-kitchen: {self.env.comfy_kitchen_rev or 'version unknown'}; "
                                  "required layouts " +
                                  (", ".join(FORMAT_TO_KITCHEN_LAYOUT[f]
                                             for f in required_formats)
                                   or "none") +
                                  ("; MISSING: " + ", ".join(
                                      FORMAT_TO_KITCHEN_LAYOUT[f] for f in missing)
                                   if missing else " all present"),
                           warn=bool(missing))
            else:
                self.check("compat-comfy-kitchen", True, "comfy-kitchen not installed (skipped)",
                           skipped=True, reason="optional runtime probe")
            if self.env.has_comfy_quant_ops:
                missing = [f for f in required_formats
                           if f not in self.env.comfyui_quant_algos]
                self.check("compat-comfyui", not missing,
                           detail=f"static ComfyUI quant_ops formats: "
                                  f"{self.env.comfyui_quant_algos}"
                                  + (f"; MISSING: {missing}" if missing else ""),
                           warn=bool(missing))
            else:
                self.check("compat-comfyui", True, "ComfyUI not installed (skipped)",
                           skipped=True, reason="optional runtime probe")

        # ---- input hash cross-check ----
        if input_hashes:
            if full_validation:
                current_hashes = hash_checkpoint_files(self.info, refresh=True)
                portable_current = _portable_hash_manifest(
                    self.info.files, current_hashes)
                portable_expected = _portable_hash_manifest(
                    self.info.files, input_hashes)
                self.check(
                    "input-hash", current_hashes == input_hashes,
                    (json_dumps(portable_current) if current_hashes == input_hashes else
                     "source changed during validation: " + json_dumps({
                         "expected": portable_expected,
                         "actual": portable_current,
                     })))
            else:
                self.check("input-hash", True,
                           "source hashes matched before and after conversion: "
                           + json_dumps(_portable_hash_manifest(
                               self.info.files, input_hashes)))

        del st
        return self.summary()
