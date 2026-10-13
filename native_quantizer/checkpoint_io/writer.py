"""Streaming tensor serialization and metadata republishing."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import struct
import tempfile
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from native_quantizer.checkpoint_io.reader import CheckpointReader, RawSafetensorsFile
from native_quantizer.checkpoint_io.types import TensorMeta
from native_quantizer.dtypes import TORCH_TO_SAFE, tensor_nbytes, tensor_to_bytes
from native_quantizer.errors import OutputError, QualityGateError
from native_quantizer.files import _fsync_parent, _open_regular_nofollow
from native_quantizer.identity import MAX_SAFETENSORS_HEADER_SIZE
from native_quantizer.logging import log
from native_quantizer.utils import human_bytes


class SafetensorsStreamWriter:
    """Writes a .safetensors file incrementally.

    The header is computed up front (all output tensor names / dtypes / shapes /
    byte sizes are known after planning), then tensor data is streamed in
    exactly the planned order.  Supports resume via an external state file.
    """

    def __init__(self, path: str, entries: List[Dict[str, Any]],
                 resume_offsets: Optional[Dict[str, int]] = None,
                 resume_mode: bool = False,
                 expected_identity: Optional[Tuple[int, int]] = None):
        self.path = path
        self.entries = list(entries)
        # deterministic write order
        self.order = [e["name"] for e in self.entries]
        self._by_name = {e["name"]: e for e in self.entries}
        self._resume_offsets = resume_offsets or {}
        self._resume_mode = resume_mode
        self._expected_identity = expected_identity
        self._offsets: Dict[str, int] = {}
        self._header = self._build_header()
        # convert relative (data-relative) offsets to absolute file offsets
        self._offsets = {n: len(self._header) + rel for n, rel in self._offsets.items()}
        self._fh: Optional[Any] = None
        self._pos = 0
        self.identity: Optional[Tuple[int, int]] = None

    def _build_header(self) -> bytes:
        header: Dict[str, Any] = {"__metadata__": {}}
        offset = 0
        # Safetensors data offsets are contiguous.  The JSON header itself is
        # padded to an 8-byte boundary; inserting holes between tensors would
        # make otherwise-valid odd-byte tensors unreadable by strict loaders.
        for e in self.entries:
            start = offset
            end = start + e["nbytes"]
            header[e["name"]] = {
                "dtype": TORCH_TO_SAFE[e["dtype"]],
                "shape": list(e["shape"]),
                "data_offsets": [start, end],
            }
            self._offsets[e["name"]] = start
            offset = end
        self._data_size = offset
        raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(raw) > MAX_SAFETENSORS_HEADER_SIZE:
            raise OutputError(
                "output header exceeds the safetensors 100 MB safety limit")
        raw += b" " * ((-len(raw)) % 8)
        return struct.pack("<Q", len(raw)) + raw

    def header_bytes(self) -> bytes:
        return self._header

    def data_size(self) -> int:
        return self._data_size

    def open(self) -> None:
        if self._fh is not None:
            return
        if os.path.exists(self.path):
            if not self._resume_mode:
                raise OutputError(f"output file already exists: {self.path} (use --overwrite)")
            fd, st = _open_regular_nofollow(self.path, os.O_RDWR)
            identity = (int(st.st_dev), int(st.st_ino))
            if self._expected_identity is not None and identity != self._expected_identity:
                os.close(fd)
                raise OutputError(
                    f"resume temp identity changed for {self.path}; refusing possible "
                    "symlink/hardlink replacement")
            self._fh = os.fdopen(fd, "r+b")
            self.identity = identity
            expected_size = len(self._header) + self._data_size
            if st.st_size != expected_size:
                self.close()
                raise OutputError(
                    f"resume temp size mismatch: {st.st_size} != {expected_size}")
            self._fh.seek(0)
            if self._fh.read(len(self._header)) != self._header:
                self.close()
                raise OutputError("resume temp header does not match the conversion plan")
            self._pos = expected_size
        else:
            if self._resume_mode:
                raise OutputError(f"resume temp file missing: {self.path}")
            fd, st = _open_regular_nofollow(
                self.path, os.O_RDWR | os.O_CREAT | os.O_EXCL)
            self._fh = os.fdopen(fd, "w+b")
            self.identity = (int(st.st_dev), int(st.st_ino))
            self._fh.write(self._header)
            self._fh.truncate(len(self._header) + self._data_size)
            self._pos = len(self._header) + self._data_size

    def write_tensor_bytes(self, name: str, data: bytes) -> int:
        """Write one complete tensor at its planned offset."""
        self.open()
        if self._fh is None:
            raise OutputError("output writer failed to open")
        e = self._by_name[name]
        expected_off = self._offsets[name]     # deterministic, header-derived
        if self._resume_offsets and name in self._resume_offsets:
            # already written by a previous run: the recorded offset must equal
            # the header-derived offset (a real interruption writes strictly in
            # plan order, so this also detects corrupted/out-of-order states)
            off = self._resume_offsets[name]
            if off != expected_off:
                raise OutputError(
                    f"resume state inconsistent for {name}: recorded offset {off} "
                    f"!= header offset {expected_off}")
            self._fh.seek(off + e["nbytes"])
            self._pos = off + e["nbytes"]
            return off
        if len(data) != e["nbytes"]:
            raise OutputError(f"tensor {name}: expected {e['nbytes']} bytes, got {len(data)}")
        self._fh.seek(expected_off)
        self._fh.write(data)
        self._pos = expected_off + e["nbytes"]
        return expected_off

    def write_tensor_slice(self, name: str, byte_offset: int, data: bytes) -> int:
        """Write a byte slice inside a tensor, enabling bounded row-chunk output."""
        self.open()
        if self._fh is None:
            raise OutputError("output writer failed to open")
        e = self._by_name[name]
        if byte_offset < 0 or byte_offset + len(data) > e["nbytes"]:
            raise OutputError(
                f"tensor slice for {name} exceeds planned range: "
                f"{byte_offset}+{len(data)} > {e['nbytes']}")
        absolute = self._offsets[name] + byte_offset
        self._fh.seek(absolute)
        self._fh.write(data)
        self._pos = absolute + len(data)
        return absolute

    def offset_for(self, name: str) -> int:
        return self._offsets[name]

    def invalidate_resume_tensor(self, name: str) -> None:
        """Allow a partially completed logical layer to rewrite this tensor."""
        self._resume_offsets.pop(name, None)

    def tensor_sha256(self, name: str, chunk: int = 1 << 20) -> str:
        self.open()
        if self._fh is None:
            raise OutputError("output writer failed to open")
        entry = self._by_name[name]
        self._fh.flush()
        self._fh.seek(self._offsets[name])
        remaining = entry["nbytes"]
        digest = hashlib.sha256()
        while remaining:
            data = self._fh.read(min(chunk, remaining))
            if not data:
                raise OutputError(f"resume temp is truncated inside tensor {name!r}")
            digest.update(data)
            remaining -= len(data)
        return digest.hexdigest()

    def current_pos(self) -> int:
        return self._pos

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def finalize(self, final_path: str) -> None:
        """Flush, fsync and atomically publish to final_path."""
        self.flush()
        self.close()
        # verify total size
        expected = len(self._header) + self._data_size
        actual = os.path.getsize(self.path)
        if actual != expected:
            raise OutputError(
                f"output size mismatch: {actual} != {expected} (incomplete write?)")
        os.replace(self.path, final_path)
        _fsync_parent(final_path)
        log().info("published output to %s (%s)", final_path, human_bytes(actual))


def republish_with_metadata(src_path: str, final_path: str, metadata: Dict[str, str],
                            entries: List[Dict[str, Any]]) -> str:
    """Rewrite src -> final with the final header (including __metadata__).

    Streaming copy; bounded memory; atomic publish.  The data payload is copied
    verbatim; only the header changes.  `src_path` may equal `final_path` (the
    published file is replaced atomically via a sibling temp file).
    """
    header: Dict[str, Any] = {"__metadata__": metadata}
    offset = 0
    for e in entries:
        start = offset
        end = start + e["nbytes"]
        header[e["name"]] = {
            "dtype": TORCH_TO_SAFE[e["dtype"]],
            "shape": list(e["shape"]),
            "data_offsets": [start, end],
        }
        offset = end
    # Canonical key ordering makes a metadata-only republish byte-identical
    # regardless of the mapping order returned by a safetensors implementation.
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(raw) > MAX_SAFETENSORS_HEADER_SIZE:
        raise OutputError("metadata header exceeds the safetensors 100 MB safety limit")
    raw += b" " * ((-len(raw)) % 8)
    parent = os.path.dirname(os.path.abspath(final_path)) or "."
    fd, final_tmp = tempfile.mkstemp(
        prefix=f".{Path(final_path).name}.metadata.", suffix=".tmp", dir=parent)
    try:
        with open(src_path, "rb") as src:
            head = src.read(8)
            if len(head) != 8:
                raise OutputError(f"{src_path}: truncated source header during republish")
            old_hlen = struct.unpack("<Q", head)[0]
            payload_start = 8 + old_hlen
            if payload_start > os.fstat(src.fileno()).st_size:
                raise OutputError(f"{src_path}: invalid source header during republish")
            payload_bytes = os.fstat(src.fileno()).st_size - payload_start
            if payload_bytes != offset:
                raise OutputError(
                    f"{src_path}: payload size {payload_bytes} != planned {offset}")
            src.seek(payload_start)
            with os.fdopen(fd, "wb") as dst:
                dst.write(struct.pack("<Q", len(raw)))
                dst.write(raw)
                shutil.copyfileobj(src, dst, length=1 << 20)
                dst.flush()
                os.fsync(dst.fileno())
        with RawSafetensorsFile(final_tmp) as candidate:
            if candidate.metadata != metadata:
                raise OutputError("republished metadata did not round-trip")
            if set(candidate.entries) != {entry["name"] for entry in entries}:
                raise OutputError("republished tensor inventory did not round-trip")
            for entry in entries:
                dtype, shape, _, _ = candidate.get(entry["name"])
                if dtype != entry["dtype"] or tuple(shape) != tuple(entry["shape"]):
                    raise OutputError(
                        f"republished shape/dtype mismatch for {entry['name']!r}")
        os.replace(final_tmp, final_path)
        _fsync_parent(final_path)
    except Exception:
        with contextlib.suppress(OSError):
            os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(final_tmp)
        raise
    return final_path


class NativeSpool:
    """Append tensors to disk; serialize the final header once after quality gates.

    Fragments are temporary files, so rejected layers and swizzled backend tensors
    never require retaining a complete model or concatenating all row chunks.
    """
    def __init__(self, directory: str):
        self.file = tempfile.TemporaryFile(dir=directory)
        self.entries: List[Dict[str, Any]] = []
        self.names: set = set()

    def append(self, name: str, dtype: torch.dtype, shape: Sequence[int], stream: Any) -> None:
        if name in self.names:
            raise OutputError(f'duplicate output tensor {name}')
        size = tensor_nbytes(dtype, shape)
        stream.seek(0, os.SEEK_END)
        if stream.tell() != size:
            raise OutputError(f'incomplete tensor {name}')
        stream.seek(0)
        shutil.copyfileobj(stream, self.file, length=1 << 20)
        self.entries.append({'name': name, 'dtype': dtype, 'shape': tuple(shape), 'nbytes': size})
        self.names.add(name)

    def publish_candidate(self, path: str, metadata: Dict[str, str]) -> None:
        header: Dict[str, Any] = {'__metadata__': metadata}
        offset = 0
        for e in self.entries:
            header[e['name']] = {'dtype': TORCH_TO_SAFE[e['dtype']], 'shape': list(e['shape']),
                                  'data_offsets': [offset, offset + e['nbytes']]}
            offset += e['nbytes']
        raw = json.dumps(header, sort_keys=True, separators=(',', ':')).encode()
        if len(raw) > MAX_SAFETENSORS_HEADER_SIZE:
            raise OutputError('output metadata exceeds safetensors header limit')
        raw += b' ' * (-len(raw) % 8)
        with open(path, 'xb') as dst:
            dst.write(struct.pack('<Q', len(raw)))
            dst.write(raw)
            self.file.seek(0)
            shutil.copyfileobj(self.file, dst, length=1 << 20)
            dst.flush()
            os.fsync(dst.fileno())


def _native_copy(reader: CheckpointReader, meta: TensorMeta, spool: NativeSpool,
                 dtype: torch.dtype, budget: int, directory: str) -> None:
    with tempfile.TemporaryFile(dir=directory) as fragment:
        if dtype == meta.dtype:
            data = reader.read_bytes(meta.name)
            for offset in range(0, len(data), min(budget, 1 << 20)):
                fragment.write(data[offset:offset + min(budget, 1 << 20)])
            del data
        else:
            flat = reader.read_tensor(meta.name).reshape(-1)
            step = max(1, budget // 32)
            for start in range(0, flat.numel(), step):
                source = flat[start:start + step]
                converted = source.to(dtype)
                if not torch.isfinite(converted.float()).all() and torch.isfinite(source.float()).all():
                    raise QualityGateError(f'{meta.name}: {dtype} conversion overflow; choose BF16 or auto')
                fragment.write(tensor_to_bytes(converted))
            del flat
        spool.append(meta.name, dtype, meta.shape, fragment)
