"""Isolated test preferences and explicitly configured consumer integration."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def isolated_module() -> None:
    """Never read or write the developer's CLI or Studio preferences."""
    directory = tempfile.TemporaryDirectory(prefix='quantizer-tests-')
    unittest.addModuleCleanup(directory.cleanup)
    root = Path(directory.name)
    preferences = patch.dict(os.environ, {
        'XDG_CONFIG_HOME': str(root / 'config'),
        'XDG_STATE_HOME': str(root / 'state'),
        'XDG_CACHE_HOME': str(root / 'cache'),
        'XDG_DATA_HOME': str(root / 'data'),
    })
    preferences.start()
    unittest.addModuleCleanup(preferences.stop)


def comfyui_root() -> Path:
    """Skip unrequested integration; fail if an explicit consumer is missing."""
    configured = os.environ.get('COMFYUI_ROOT')
    if not configured:
        if os.environ.get('COMFYUI_REQUIRED') == '1':
            raise RuntimeError('COMFYUI_REQUIRED=1 requires COMFYUI_ROOT')
        raise unittest.SkipTest('optional integration: set COMFYUI_ROOT')
    root = Path(configured).expanduser().resolve()
    if not (root / 'comfy/ops.py').is_file():
        raise RuntimeError(f'COMFYUI_ROOT is not a ComfyUI source checkout: {root}')
    return root


def load_comfy_cpu():
    """Import the configured consumer on CPU without starting its services."""
    root = comfyui_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
        unittest.addModuleCleanup(lambda: sys.path.remove(str(root)))
    from comfy.cli_args import args
    args.cpu = True
    import comfy.ops
    import comfy.utils
    return comfy.ops, comfy.utils
