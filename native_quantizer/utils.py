"""Scalar sizes, JSON formatting, regular expressions and process accounting."""

from __future__ import annotations

import json
import math
import re
import sys
from typing import Any, Sequence

from native_quantizer.errors import UsageError


def parse_size(text: str) -> int:
    """Parse a size like 2G, 512M, 1024K or a plain byte count."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGTP]?B?)?\s*", text, re.I)
    if not m:
        raise UsageError(f"invalid size {text!r}")
    value = float(m.group(1))
    unit = (m.group(2) or "").upper().rstrip("B")
    mult = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3,
            "T": 1024**4, "P": 1024**5}[unit]
    result = value * mult
    if not math.isfinite(result) or result > sys.maxsize:
        raise UsageError(f"size {text!r} exceeds this platform's supported range")
    return int(result)


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if abs(n) < 1024 or unit == "PiB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.2f} PiB"  # pragma: no cover


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def flatten_regex(patterns: Sequence[str]) -> re.Pattern:
    try:
        return re.compile("|".join(f"(?:{p})" for p in patterns))
    except re.error as e:
        raise UsageError(f"invalid regular expression: {e}") from e


def _peak_rss_bytes() -> int:
    try:
        import resource
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS/BSD reports bytes.
        return peak if sys.platform == "darwin" else peak * 1024
    except Exception:
        return 0
