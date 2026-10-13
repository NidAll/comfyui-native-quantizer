"""Reference fixtures loaded only by explicitly requested regression checks."""

from __future__ import annotations

import json
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, Optional


_GOLDEN_BLOB = None


_GOLDEN_DATA_FILE = Path(__file__).resolve().parent / 'fixtures' / 'v1.blob.txt'


_GOLDEN: Optional[Dict[str, Dict[str, Any]]] = None


def _read_golden_blob() -> str:
    """Return the embedded blob, or read the versioned data file."""
    if _GOLDEN_BLOB is not None:
        return _GOLDEN_BLOB
    if not _GOLDEN_DATA_FILE.is_file():
        raise FileNotFoundError(
            f"golden data missing: {_GOLDEN_DATA_FILE} (reinstall the package "
            "with its reference fixtures)")
    return _GOLDEN_DATA_FILE.read_text(encoding="utf-8")


def get_golden() -> Dict[str, Dict[str, Any]]:
    """Lazily load and cache the golden reference vectors."""
    global _GOLDEN  # noqa: PLW0603
    if _GOLDEN is None:
        _GOLDEN = _load_golden()
    return _GOLDEN


def _load_golden() -> Dict[str, Dict[str, Any]]:
    import base64 as _b64
    import zlib as _zlib
    payload = json.loads(_zlib.decompress(_b64.b64decode(_read_golden_blob())).decode("utf-8"))
    out = {}
    np_dtypes = {torch.float16: np.float16, torch.float32: np.float32,
                 torch.uint8: np.uint8, torch.int8: np.int8}

    def _tensor(record, key, dtype, shape):
        return torch.from_numpy(
            np.frombuffer(_b64.b64decode(record[key]),
                          dtype=np_dtypes[dtype]).copy()).view(shape)

    for name, g in payload.items():
        shape = tuple(g["shape"])
        out[name] = {
            "weight": _tensor(g, "weight_b64", torch.float16, shape),
            "packed": _tensor(
                g, "packed_b64", torch.int8, (shape[0], shape[1] // 2)),
            "s_rel_u8": _tensor(
                g, "s_rel_u8_b64", torch.uint8,
                (shape[0], shape[1] // g["gs"])),
            "s_channel": _tensor(
                g, "s_channel_b64", torch.float32, (shape[0],)),
            "codebook": _tensor(g, "codebook_b64", torch.float32, (16,)),
            "gs": g["gs"], "cgs": g["cgs"],
        }
    return out


GOLDEN: Optional[Dict[str, Dict[str, Any]]] = None


GOLDEN_W4: Optional[Dict[str, Any]] = None


GOLDEN = _load_golden()


GOLDEN_W4 = GOLDEN["w4a8_default"]
