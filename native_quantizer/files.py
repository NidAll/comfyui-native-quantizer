"""File hashing, bounded JSON reads and atomic artifact publication."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import stat
import struct
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from native_quantizer.errors import InputError, OutputError

# Hashing and file reads release the GIL, so a few threads overlap independent
# digests without changing any result.
HASH_WORKERS = 4


class HashCancelled(Exception):
    """A background digest was abandoned because its caller failed first."""


def sha256_file(path: str, chunk: int = 1 << 22,
                cancel: Optional[threading.Event] = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            if cancel is not None and cancel.is_set():
                raise HashCancelled(path)
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def sha256_files(paths: Sequence[str],
                 cancel: Optional[threading.Event] = None) -> Dict[str, str]:
    """Full-file digests keyed by path; independent files are hashed concurrently."""
    ordered = sorted(paths)
    if len(ordered) <= 1:
        return {path: sha256_file(path, cancel=cancel) for path in ordered}
    with ThreadPoolExecutor(min(HASH_WORKERS, len(ordered))) as pool:
        digests = list(pool.map(lambda path: sha256_file(path, cancel=cancel), ordered))
    return dict(zip(ordered, digests))


class BackgroundDigest:
    """Run one digest function on a daemon thread while the caller keeps working.

    ``result`` re-raises the worker's exception. ``cancel`` lets a failing caller
    exit without waiting for a full read of a large checkpoint.
    """

    def __init__(self, function, *args):
        self._cancel = threading.Event()
        self._value: Any = None
        self._error: Optional[BaseException] = None

        def run() -> None:
            try:
                self._value = function(*args, cancel=self._cancel)
            except BaseException as exc:  # re-raised in the caller's thread
                self._error = exc
        self._thread = threading.Thread(target=run, name="sha256", daemon=True)
        self._thread.start()

    def result(self) -> Any:
        self._thread.join()
        if self._error is not None:
            raise self._error
        return self._value

    def cancel(self) -> None:
        self._cancel.set()


def copy_file_bytes(src: Any, dst: Any, count: int) -> None:
    """Copy ``count`` bytes between file objects at their current positions.

    Uses in-kernel ``copy_file_range`` where available and falls back to a
    bounded userspace copy. Both positions advance by ``count``.
    """
    src.flush()
    dst.flush()
    src_start, dst_start = src.tell(), dst.tell()
    copied = 0
    if hasattr(os, "copy_file_range"):
        with contextlib.suppress(OSError):
            while copied < count:
                step = os.copy_file_range(src.fileno(), dst.fileno(), count - copied,
                                          src_start + copied, dst_start + copied)
                if step == 0:
                    break
                copied += step
    src.seek(src_start + copied)
    dst.seek(dst_start + copied)
    remaining = count - copied
    while remaining:
        data = src.read(min(1 << 20, remaining))
        if not data:
            raise OutputError(f"source ended {remaining} bytes early during copy")
        dst.write(data)
        remaining -= len(data)


def sha256_safetensors_payload(path: str, chunk: int = 1 << 22) -> str:
    """Hash only the tensor-data section, which is stable across metadata rewrites."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) != 8:
            raise InputError(f"{path}: truncated safetensors header")
        header_len = struct.unpack("<Q", head)[0]
        if header_len > os.fstat(f.fileno()).st_size - 8:
            raise InputError(f"{path}: safetensors header exceeds file size")
        f.seek(8 + header_len)
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def _publish_output(validation_path: str, out_path: str, *,
                    overwrite: bool,
                    expected_identity: Optional[Tuple[int, int]]) -> None:
    """Race-safe publication of the validated candidate.

    Without --overwrite (or when the destination did not exist at conversion
    start) publication is a no-clobber hard link: if another process created
    the destination meanwhile, the link fails and nothing is replaced.
    With --overwrite the destination identity captured before conversion must
    still match: a concurrent swap, deletion or symlink replacement aborts.
    """
    if not overwrite or expected_identity is None:
        try:
            os.link(validation_path, out_path)
        except FileExistsError:
            raise OutputError(
                f"destination was created concurrently: {out_path}; refusing "
                "to replace it (rerun the conversion, or pass --overwrite to "
                "allow replacement)") from None
        except OSError as exc:
            raise OutputError(
                f"cannot publish without overwrite: {exc}") from None
        os.unlink(validation_path)
        return
    try:
        st = os.lstat(out_path)
    except FileNotFoundError:
        raise OutputError(
            f"destination disappeared during conversion: {out_path}; "
            "refusing overwrite") from None
    if stat.S_ISLNK(st.st_mode):
        raise OutputError(
            f"destination became a symlink: {out_path}; refusing overwrite")
    current = (int(st.st_dev), int(st.st_ino))
    if current != expected_identity:
        raise OutputError(
            f"destination changed concurrently: {out_path}; refusing overwrite")
    os.replace(validation_path, out_path)


def _fsync_parent(path: str) -> None:
    """Best-effort directory fsync after atomic publication (POSIX)."""
    if os.name == "nt":
        return
    parent = os.path.dirname(os.path.abspath(path)) or "."
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(parent, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_json(path: str, payload: Any, *, indent: Optional[int] = None) -> None:
    """Write JSON through an unpredictable sibling temp and atomically replace."""
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{Path(path).name}.", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_parent(path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _open_regular_nofollow(path: str, flags: int, mode: int = 0o600):
    """Open a regular file without following a final-component symlink."""
    flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, mode)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise OutputError(f"refusing non-regular file: {path}")
        return fd, st
    except Exception:
        os.close(fd)
        raise


def _remove_temp_path(path: str) -> None:
    """Remove only the named temp entry; never follow it if it is a symlink."""
    try:
        os.lstat(path)
    except FileNotFoundError:
        return
    os.unlink(path)


def _same_path(left: str, right: str) -> bool:
    if os.path.realpath(left) == os.path.realpath(right):
        return True
    if os.path.exists(left) and os.path.exists(right):
        try:
            return os.path.samefile(left, right)
        except OSError:
            pass
    return False


def _load_json_object(path: str, label: str, *, nofollow: bool = False) -> Dict[str, Any]:
    def _no_duplicates(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise InputError(f"{path}: duplicate JSON key {key!r}")
            obj[key] = value
        return obj
    if nofollow:
        fd, file_stat = _open_regular_nofollow(path, os.O_RDONLY)
        file_obj = os.fdopen(fd, "r", encoding="utf-8")
        file_size = file_stat.st_size
    else:
        file_obj = open(path, "r", encoding="utf-8")
        file_size = os.fstat(file_obj.fileno()).st_size
    with file_obj as f:
        if file_size > 64 * 1024 * 1024:
            raise InputError(f"{path}: JSON file exceeds 64 MiB safety limit")
        data = json.load(f, object_pairs_hook=_no_duplicates)
    if not isinstance(data, dict):
        raise InputError(f"{path}: malformed {label}")
    return data
