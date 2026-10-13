"""Small, data-only XDG defaults for command-line and local frontend users."""
from __future__ import annotations

import json
import os
from pathlib import Path

from native_quantizer.errors import UsageError
from native_quantizer.utils import parse_size


DEFAULT_CHOICES = {
    'device': ('auto', 'cpu', 'cuda', 'rocm'),
    'target_runtime': ('auto', 'nvidia', 'amd', 'cpu'),
    'compute_dtype': ('auto', 'fp32', 'fp16', 'bf16'),
    'codebook_mode': ('auto', 'fixed', 'fit'),
    'rotation_backend': ('matrix', 'butterfly'),
    'progress': ('auto', 'on', 'off'),
    'log_level': ('debug', 'info', 'warning', 'error'),
}


def config_path() -> Path:
    return Path(os.environ.get('XDG_CONFIG_HOME') or Path.home() / '.config') / 'comfyui-native-quantizer/config.json'


def load_defaults(path: str | Path | None = None) -> dict:
    target = Path(path) if path is not None else config_path()
    if not target.exists() and path is None:
        return {}
    try:
        if target.stat().st_size > 65536:
            raise UsageError(f'{target}: configuration exceeds 64 KiB')
        data = json.loads(target.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise UsageError(f'cannot read quantizer configuration {target}: {exc}') from exc
    if not isinstance(data, dict) or type(data.get('schema')) is not int or data['schema'] != 1:
        raise UsageError(f'{target}: expected a JSON object with schema=1')
    unknown = set(data) - {'schema', 'max_memory', 'offline', *DEFAULT_CHOICES}
    if unknown:
        raise UsageError(f'{target}: unknown defaults: {", ".join(sorted(unknown))}')
    for key, choices in DEFAULT_CHOICES.items():
        if key in data and data[key] not in choices:
            raise UsageError(f'{target}: {key} must be one of {", ".join(choices)}')
    if 'max_memory' in data:
        if not isinstance(data['max_memory'], str) or parse_size(data['max_memory']) < 4096:
            raise UsageError(f'{target}: max_memory must be a size of at least 4K')
    if 'offline' in data and type(data['offline']) is not bool:
        raise UsageError(f'{target}: offline must be true or false')
    return {key: value for key, value in data.items() if key != 'schema'}


def apply_cli_defaults(parser, args, argv: list[str]) -> dict:
    defaults = {} if args.no_config else load_defaults(args.config)
    supplied = {token.split('=', 1)[0] for token in argv if token.startswith('-')}
    for action in parser._actions:
        if action.dest in defaults and not supplied.intersection(action.option_strings):
            setattr(args, action.dest, defaults[action.dest])
    return defaults
