"""Calibration artifact ingestion, coverage and content-addressed caches."""

from __future__ import annotations

import contextlib
import hashlib
import json
import numpy as np
import os
import tempfile
import torch
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from native_quantizer.checkpoint_io.fingerprints import _portable_file_labels, hash_checkpoint_files
from native_quantizer.checkpoint_io.reader import CheckpointReader, _resolve_under
from native_quantizer.checkpoint_io.types import CheckpointInfo
from native_quantizer.dtypes import FLOAT_DTYPES
from native_quantizer.errors import CalibrationError, InputError
from native_quantizer.files import _fsync_parent, _load_json_object, _open_regular_nofollow, sha256_file
from native_quantizer.identity import CALIBRATION_REVISION
from native_quantizer.logging import log
from native_quantizer.planning.types import DecisionKind, TensorDecision
from native_quantizer.utils import human_bytes, json_dumps


@dataclass
class CalibrationStats:
    source: str
    files: List[str]
    n_samples: int
    layers: Dict[str, Dict[str, Any]]      # key -> {samples: Tensor[S,K], rows:int}
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "files": self.files,
            "n_samples": self.n_samples,
            "layers": {k: {
                "rows": v["rows"],
                "features": int(v["samples"].shape[1]),
                "absmax_mean": float(v["samples"].abs().amax(dim=0).mean().item()),
            } for k, v in self.layers.items()},
            "provenance": self.provenance,
        }


def _load_npz_calibration(path: str, layer_keys: set,
                          max_rows: int) -> Dict[str, torch.Tensor]:
    out = {}
    with np.load(path, allow_pickle=False) as data:
        for k in data.files:
            arr = data[k]
            if arr.ndim != 2:
                continue
            if k in layer_keys or (k + ".weight") in layer_keys:
                out[k] = torch.from_numpy(
                    np.asarray(arr[:max_rows], dtype=np.float32))
    return out


def _load_pt_calibration(path: str, layer_keys: set,
                         max_rows: int) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict):
        raise CalibrationError(f"{path}: calibration .pt must contain a dict")
    out = {}
    for k, v in obj.items():
        if isinstance(v, torch.Tensor) and v.ndim == 2 and (k in layer_keys or (k + ".weight") in layer_keys):
            out[k] = v.detach()[:max_rows].float().cpu()
    return out


def _check_calibration_file_budget(path: str, max_memory: int) -> int:
    """Reject calibration files that can expand beyond the working budget."""
    file_size = os.path.getsize(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) > 100_000:
                raise CalibrationError(
                    f"{path}: archive has too many members ({len(members)})")
            expanded = [member.file_size for member in members]
        # The loader retains selected activation rows from multiple members;
        # bounding the total expanded archive prevents a many-member zip bomb
        # even when each individual array would fit.
        required = sum(expanded)
    else:
        required = file_size
    if required > max_memory:
        raise CalibrationError(
            f"{path}: calibration data may expand to {human_bytes(required)}, "
            f"above --max-memory {human_bytes(max_memory)}")
    return required


def _calibration_checkpoint_fingerprint(info: CheckpointInfo) -> Dict[str, str]:
    """Content-addressed binding of a calibration cache to its checkpoint:
    portable file labels -> sha256 of each input file. Same layer names and
    widths are NOT sufficient (they are not content)."""
    labels = _portable_file_labels(sorted(info.files))
    hashes = hash_checkpoint_files(info)
    return dict(zip(labels, (hashes[p] for p in sorted(info.files))))


def _calibration_layer_mapping_fingerprint(info: CheckpointInfo) -> str:
    """Fingerprint of the linear-layer inventory (name/dtype/shape), so a
    cache cannot silently survive a layer-mapping change that keeps content."""
    lines = sorted(
        f"{t.name}:{t.dtype}:{','.join(str(s) for s in t.shape)}"
        for t in info.tensors if t.name.endswith(".weight"))
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def apply_nonfinite_policy(info: CheckpointInfo,
                           decisions: List[TensorDecision],
                           policy: str) -> List[str]:
    """Explicit non-finite weight policy, applied before planning/entries.

    policy="error": any non-finite value in a quantize candidate is a hard
    error naming the layer. policy="keep": the layer is demoted to
    passthrough at original precision (recorded in the report). Returns the
    list of affected tensor names.
    """
    affected: List[str] = []
    with CheckpointReader(info) as reader:
        for d in decisions:
            if d.kind != DecisionKind.QUANTIZE:
                continue
            t = reader.read_tensor(d.name)
            if t.dtype in FLOAT_DTYPES and not torch.isfinite(t).all():
                if policy == "keep":
                    d.kind = DecisionKind.KEEP
                    d.reason = "non-finite values; --nonfinite-policy keep " \
                               "passes the layer through at original precision"
                    affected.append(d.name)
                else:
                    raise InputError(
                        f"non-finite values in quantizable layer {d.name!r} "
                        "(NaN/Inf); pass --nonfinite-policy keep to preserve "
                        "the layer at original precision")
    return affected


def load_calibration(source: str, info: CheckpointInfo, max_samples: Optional[int],
                     cache_path: Optional[str],
                     max_memory: int = 2 * 1024**3) -> CalibrationStats:
    """Load local calibration activations for the checkpoint's linear layers."""
    effective_max_samples = 64 if max_samples is None else int(max_samples)
    if effective_max_samples <= 0:
        raise CalibrationError("--calibration-samples must be positive")
    layer_keys = set()
    for t in info.tensors:
        if t.name.endswith(".weight"):
            layer_keys.add(t.name)
    # fast path: load a precomputed statistics cache before touching the source
    if cache_path is not None and os.path.exists(cache_path):
        try:
            stats = {}
            prov: Dict[str, Any] = {}
            _check_calibration_file_budget(cache_path, max_memory)
            cache_hash_before = sha256_file(cache_path)
            cache_fd, _ = _open_regular_nofollow(cache_path, os.O_RDONLY)
            with os.fdopen(cache_fd, "rb") as cache_probe:
                is_npz = cache_probe.read(4) == b"PK\x03\x04"
                cache_probe.seek(0)
                if is_npz:
                    with np.load(cache_probe, allow_pickle=False) as payload:
                        if "__provenance__" in payload.files:
                            prov = json.loads(
                                np.asarray(payload["__provenance__"], dtype=np.uint8)
                                .tobytes().decode("utf-8"))
                        for key in payload.files:
                            if key == "__provenance__" or key not in layer_keys:
                                continue
                            array = payload[key]
                            tensor = torch.from_numpy(np.asarray(
                                array[:effective_max_samples], dtype=np.float32))
                            meta = info.by_name(key)
                            if tensor.ndim == 2 and tensor.shape[0] > 0 and meta is not None \
                                    and tensor.shape[1] == meta.shape[1]:
                                if not torch.isfinite(tensor).all():
                                    raise CalibrationError(
                                        f"{cache_path}: non-finite calibration "
                                        f"activations for {key}")
                                tensor = tensor.contiguous()
                                stats[key] = {"samples": tensor,
                                              "rows": int(tensor.shape[0])}
            if not is_npz:
                # Backward-compatible reader for early JSON caches.  v1.1
                # summary-only caches are ignored because they cannot reproduce
                # activation-aware output error.
                if os.path.getsize(cache_path) > min(64 * 1024 * 1024,
                                                     max(1, max_memory // 4)):
                    raise CalibrationError(
                        f"{cache_path}: JSON cache exceeds the working-memory limit")
                payload = _load_json_object(
                    cache_path, "calibration cache", nofollow=True)
                prov = payload.get("provenance", {})
                for key, value in payload.get("stats", {}).items():
                    samples = value.get("samples")
                    if samples is None or key not in layer_keys:
                        continue
                    tensor = torch.as_tensor(
                        samples[:effective_max_samples], dtype=torch.float32)
                    meta = info.by_name(key)
                    if tensor.ndim == 2 and tensor.shape[0] > 0 and meta is not None \
                            and tensor.shape[1] == meta.shape[1]:
                        if not torch.isfinite(tensor).all():
                            raise CalibrationError(
                                f"{cache_path}: non-finite calibration "
                                f"activations for {key}")
                        tensor = tensor.contiguous()
                        stats[key] = {"samples": tensor,
                                      "rows": int(tensor.shape[0])}
            cache_hash_after = sha256_file(cache_path)
            if cache_hash_after != cache_hash_before:
                raise CalibrationError(
                    "calibration cache changed while it was being read")
            if not isinstance(prov, dict):
                raise CalibrationError("calibration cache provenance must be an object")
            prov = dict(prov)
            prov["cache_file_sha256"] = cache_hash_after
            if prov.get("schema") == "comfy-wxa8-calibration/v2":
                # content-addressed: the cache is bound to the checkpoint it
                # was built from; a different checkpoint (even with identical
                # layer names and widths) must not reuse it
                expected_fp = _calibration_checkpoint_fingerprint(info)
                if prov.get("checkpoint_fingerprint") != expected_fp:
                    log().warning(
                        "calibration cache %s: checkpoint fingerprint mismatch "
                        "(model changed); cache ignored, fresh load", cache_path)
                    stats = {}
            elif not prov.get("checkpoint_fingerprint"):
                log().warning(
                    "calibration cache %s has no v2 checkpoint fingerprint; "
                    "accepting as legacy (rebuild the cache to bind it)",
                    cache_path)
            if stats:
                log().info("calibration activation rows loaded from cache %s", cache_path)
                return CalibrationStats(source=source, files=[cache_path],
                                        n_samples=max(v["rows"] for v in stats.values()),
                                        layers=stats, provenance=prov)
        except Exception as e:
            log().warning("could not read calibration cache %s: %s", cache_path, e)

    src_path = Path(source)
    if not src_path.exists():
        raise CalibrationError(f"calibration source not found: {source}")

    files: List[str] = []
    tensors: Dict[str, torch.Tensor] = {}
    if src_path.is_dir():
        for f in sorted(src_path.iterdir()):
            if f.suffix in (".npz", ".pt", ".npy"):
                files.append(str(f))
    else:
        files = [str(src_path)]

    manifest: Optional[Dict[str, Any]] = None
    if src_path.is_file() and src_path.suffix == ".json":
        manifest = _load_json_object(str(src_path), "calibration manifest")
        for entry in manifest.get("layers", {}).values():
            raw_path = entry.get("path", "")
            if not isinstance(raw_path, str) or not raw_path:
                raise CalibrationError("calibration manifest contains an invalid path")
            if Path(raw_path).is_absolute():
                raise CalibrationError(
                    "absolute paths are not allowed inside calibration manifests")
            p = _resolve_under(src_path.parent, raw_path)
            if p.suffix in (".npz", ".pt", ".npy") and str(p) not in files:
                files.append(str(p))

    expanded_total = 0
    for f in files:
        expanded_total += _check_calibration_file_budget(f, max_memory)
        if expanded_total > max_memory:
            raise CalibrationError(
                "combined calibration data may expand to "
                f"{human_bytes(expanded_total)}, above --max-memory "
                f"{human_bytes(max_memory)}")
    calibration_hashes = {f: sha256_file(f) for f in files}
    for f in files:
        try:
            if f.endswith(".npz"):
                data = _load_npz_calibration(
                    f, layer_keys, effective_max_samples)
            elif f.endswith(".pt"):
                data = _load_pt_calibration(
                    f, layer_keys, effective_max_samples)
            elif f.endswith(".npy"):
                arr = np.load(f, allow_pickle=False, mmap_mode="r")
                if arr.ndim == 2:
                    # single-layer file: name derived from the file name
                    name = Path(f).stem
                    if name in layer_keys or (name + ".weight") in layer_keys:
                        data = {name: torch.from_numpy(np.asarray(
                            arr[:effective_max_samples], dtype=np.float32))}
                    else:
                        data = {}
                else:
                    data = {}
            else:
                data = {}
        except Exception as e:
            raise CalibrationError(
                f"failed to read calibration file {f}: {e}") from e
        for k, v in data.items():
            canonical = k if k in layer_keys else k + ".weight"
            meta = info.by_name(canonical)
            if meta is None or len(meta.shape) != 2:
                continue
            if int(v.shape[1]) != int(meta.shape[1]):
                raise CalibrationError(
                    f"{f}: activation width for {canonical} is {v.shape[1]}, "
                    f"expected {meta.shape[1]}")
            part = v.detach()[:effective_max_samples].float().cpu()
            if not torch.isfinite(part).all():
                raise CalibrationError(
                    f"{f}: non-finite calibration activations for {canonical}")
            if canonical in tensors:
                remaining = effective_max_samples - int(tensors[canonical].shape[0])
                if remaining > 0:
                    tensors[canonical] = torch.cat(
                        (tensors[canonical], part[:remaining]), dim=0)
            else:
                tensors[canonical] = part.contiguous()
    final_calibration_hashes = {f: sha256_file(f) for f in files}
    if final_calibration_hashes != calibration_hashes:
        raise CalibrationError(
            "one or more calibration files changed while they were being read")

    if not tensors:
        raise CalibrationError(
            f"no usable calibration activations found in {source} (expected arrays "
            "named exactly like the checkpoint's linear layer keys)")

    stats: Dict[str, Dict[str, Any]] = {}
    for k, v in tensors.items():
        stats[k] = {"samples": v.contiguous(), "rows": int(v.shape[0])}

    provenance = {
        "schema": "comfy-wxa8-calibration/v2",
        "checkpoint_fingerprint": _calibration_checkpoint_fingerprint(info),
        "layer_mapping_fingerprint": _calibration_layer_mapping_fingerprint(info),
        "calibration_files": {
            Path(f).name: h for f, h in final_calibration_hashes.items()},
        "preprocessing_revision": CALIBRATION_REVISION,
        "max_samples": effective_max_samples,
        "compute_precision": "float32",
        "source": source,
        "files": files,
        "n_files": len(files),
        "n_layers_with_data": len(stats),
        "max_samples_per_layer": effective_max_samples,
        "method": "recorded local activation rows used directly for output-error analysis",
        "synthetic": False,
        "file_sha256": final_calibration_hashes,
    }

    if cache_path:
        try:
            parent = os.path.dirname(os.path.abspath(cache_path)) or "."
            os.makedirs(parent, exist_ok=True)
            fd, cache_tmp = tempfile.mkstemp(
                prefix=f".{Path(cache_path).name}.", suffix=".tmp", dir=parent)
            try:
                arrays = {key: value["samples"].numpy()
                          for key, value in stats.items()}
                arrays["__provenance__"] = np.frombuffer(
                    json_dumps(provenance).encode("utf-8"), dtype=np.uint8)
                with os.fdopen(fd, "wb") as cache_file:
                    np.savez_compressed(cache_file, **arrays)
                    cache_file.flush()
                    os.fsync(cache_file.fileno())
                os.replace(cache_tmp, cache_path)
                _fsync_parent(cache_path)
            except Exception:
                with contextlib.suppress(OSError):
                    os.close(fd)
                with contextlib.suppress(OSError):
                    os.unlink(cache_tmp)
                raise
            log().info("calibration activation rows cached to %s", cache_path)
        except Exception as e:
            log().warning("could not write calibration cache %s: %s", cache_path, e)

    return CalibrationStats(source=source, files=files,
                            n_samples=max(v["rows"] for v in stats.values()),
                            layers=stats, provenance=provenance)
