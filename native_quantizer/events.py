"""Optional, versioned JSONL events for local frontends.

This module has no torch or UI dependencies. Events are best-effort telemetry;
the converter's exit status and published output remain authoritative.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from native_quantizer.errors import UsageError
from native_quantizer.files import _same_path

SCHEMA = "quantizer-events/v1"
_path: str | None = None
_last_progress = 0.0


def prepare_event_stream(options, *, verification_input: str | None = None,
                         source_files=None) -> None:
    # Input discovery must complete before truncating telemetry files. A model
    # may be an index/directory and the event target may alias one of its shards.
    configure(None)
    if options.events_jsonl:
        protected = (options.model, options.output, verification_input,
                     options.calibration_source, options.calibration_cache,
                     options.kernel_costs, options.runtime_certificate,
                     options.report, options.json_log,
                     getattr(options, 'config', None), options.architecture_policy,
                     options.key_map)
        if any(value and _same_path(options.events_jsonl, value) for value in protected):
            raise UsageError('--events-jsonl must not overwrite an input or output path')
        if source_files and any(_same_path(options.events_jsonl, value) for value in source_files):
            raise UsageError('--events-jsonl must not overwrite an input shard')
    if (options.model or options.hf_repo) and source_files is None:
        return
    configure(options.events_jsonl)
    emit('phase', phase='starting', format=options.format)


def enabled() -> bool:
    return _path is not None


def configure(path: str | None) -> None:
    global _path, _last_progress
    _path = None
    _last_progress = 0.0
    if path:
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags, 0o600)
        os.close(fd)
        _path = path


def emit(kind: str, **fields: Any) -> None:
    global _last_progress, _path
    if not _path:
        return
    now = time.time()
    tick = time.monotonic()
    if kind == "progress":
        if tick - _last_progress < 0.25 and fields.get("done") != fields.get("total"):
            return
        _last_progress = tick
    payload = {"schema": SCHEMA, "ts": now, "type": kind, **fields}
    flags = os.O_WRONLY | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(_path, flags)
        with os.fdopen(fd, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, default=str, separators=(",", ":")) + "\n")
    except OSError as exc:
        _path = None
        logging.getLogger(__name__).warning("event stream disabled: %s", exc)


class WarningEvents(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.WARNING:
            emit("warning", message=record.getMessage())
