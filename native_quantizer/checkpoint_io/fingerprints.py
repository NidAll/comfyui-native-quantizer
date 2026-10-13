"""Checkpoint content identities and portable file labels."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence

from native_quantizer.checkpoint_io.types import CheckpointInfo
from native_quantizer.errors import OutputError
from native_quantizer.files import sha256_file


def hash_checkpoint_files(info: CheckpointInfo, *, refresh: bool = False) -> Dict[str, str]:
    if refresh or not info.source_hashes:
        info.source_hashes = {path: sha256_file(path) for path in sorted(info.files)}
    return dict(info.source_hashes)


def _portable_file_labels(paths: Sequence[str]) -> List[str]:
    """Stable, non-secret file labels for metadata embedded in moved models."""
    basenames = [Path(path).name for path in paths]
    if len(set(basenames)) == len(basenames):
        return basenames
    absolute = [os.path.abspath(path) for path in paths]
    try:
        common = os.path.commonpath([os.path.dirname(path) for path in absolute])
        relative = [os.path.relpath(path, common).replace(os.sep, "/")
                    for path in absolute]
        if len(set(relative)) == len(relative) and not any(
                label == ".." or label.startswith("../") for label in relative):
            return relative
    except ValueError:
        pass
    # Different volumes or adversarial duplicate names: retain deterministic
    # identity without embedding machine-specific absolute paths.
    return [f"{index:04d}-{name}" for index, name in enumerate(basenames)]


def _portable_hash_manifest(paths: Sequence[str], hashes: Dict[str, str]) -> Dict[str, str]:
    by_absolute = {os.path.abspath(path): digest for path, digest in hashes.items()}
    manifest: Dict[str, str] = {}
    for path, label in zip(paths, _portable_file_labels(paths), strict=True):
        absolute = os.path.abspath(path)
        if absolute not in by_absolute:
            raise OutputError(f"missing source hash for metadata: {path}")
        manifest[label] = by_absolute[absolute]
    return manifest
