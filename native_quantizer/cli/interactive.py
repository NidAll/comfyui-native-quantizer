"""Optional terminal setup for command-line users."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from native_quantizer.errors import QuantizerError, UsageError
from native_quantizer.formats import FORMAT_ALIASES, NATIVE_FORMATS
from native_quantizer.reporting import native_format_table
from native_quantizer.utils import parse_size


def interactive_setup(args: Any) -> None:
    if not sys.stdin.isatty():
        raise UsageError('--interactive requires a terminal; pass command-line options for batch use')
    print('\nComfyUI checkpoint quantizer\n')
    if not args.model and not args.hf_repo:
        source = input('Local model path, or hf://owner/repo: ').strip().strip('"')
        if source.startswith('hf://'):
            args.hf_repo = source[5:]
            args.hf_subfolder = input('HF component subfolder [repository root]: ').strip()
        else:
            args.model = os.path.expanduser(source)
    native_format_table()
    while True:
        value = input(f'Format [{args.format}]: ').strip() or args.format
        value = FORMAT_ALIASES.get(value, value)
        if value in ('w4a8', 'w6a8', 'mixed', *NATIVE_FORMATS):
            args.format = value
            break
        print('Choose a format from the table.')
    if args.format == 'mixed':
        args.experimental = True
    if not args.output:
        default = f'{Path(args.model).stem if args.model else args.hf_repo.split("/")[-1]}-{args.format}.safetensors'
        args.output = os.path.expanduser(input(f'Output [{default}]: ').strip() or default)
    while True:
        value = input(f'Working memory [{args.max_memory}]: ').strip() or args.max_memory
        try:
            if parse_size(value) <= 0:
                raise ValueError()
            args.max_memory = value
            break
        except (ValueError, QuantizerError):
            print('Enter a positive size, such as 512M or 2G.')
    policy = input('Custom architecture policy JSON [built-in detection]: ').strip()
    if policy:
        args.architecture_policy = os.path.expanduser(policy)
        args.experimental = True
    print(f'\n{args.model or args.hf_repo} -> {args.output}\nFormat: {args.format}; budget: {args.max_memory}')
    if not args.yes and input('Start conversion? [y/N]: ').strip().lower() not in ('y', 'yes'):
        raise UsageError('conversion cancelled')
