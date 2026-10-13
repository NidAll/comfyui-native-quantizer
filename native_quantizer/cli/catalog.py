"""Lightweight catalog and shell completion presentation."""
from __future__ import annotations

import argparse
import dataclasses
import shlex

from native_quantizer.formats import NATIVE_FORMATS


def format_records() -> list[dict]:
    return [
        {'cli': 'w4a8', 'algorithm': 'asym_w4a8_int8', 'backend': 'embedded reference'},
        {'cli': 'w6a8', 'algorithm': 'w6a8_int8', 'backend': 'comfy-kitchen >= 0.2.36'},
        {'cli': 'mixed', 'algorithm': 'per-layer native formats', 'backend': 'W6A8 opt-in (--enable-w6a8)'},
        *[{**dataclasses.asdict(spec), 'backend': 'comfy-kitchen' if spec.kitchen_class else 'PyTorch / embedded'}
          for spec in NATIVE_FORMATS.values()]]


def print_formats() -> None:
    print('format          ComfyUI algorithm             conversion backend')
    for row in format_records():
        print(f'{row["cli"]:16}{(row["algorithm"] or "precision conversion"):30}{row["backend"]}')
    print('\nKernel availability depends on the runtime; native safetensors formats do not imply GGUF or custom-loader support.')


def completions(parser, shell: str) -> str:
    actions = [a for a in parser._actions if a.option_strings]
    commands = ('comfyui-native-quantizer', 'comfyquant')
    options = ' '.join(option for a in actions for option in a.option_strings)
    if shell == 'bash':
        cases = '\n'.join(f'    {"|".join(a.option_strings)}) COMPREPLY=( $(compgen -W {shlex.quote(" ".join(map(str, a.choices)))} -- "$cur") ); return ;;'
                          for a in actions if a.choices)
        return f'''_comfyquant_complete() {{
  local cur="${{COMP_WORDS[COMP_CWORD]}}" prev="${{COMP_WORDS[COMP_CWORD-1]}}"
  case "$prev" in
{cases}
  esac
  if [[ "$cur" == -* ]]; then
    COMPREPLY=( $(compgen -W {shlex.quote(options)} -- "$cur") )
  else
    mapfile -t COMPREPLY < <(compgen -f -- "$cur")
    compopt -o filenames
  fi
}}
complete -F _comfyquant_complete {' '.join(commands)}
'''
    if shell == 'fish':
        rows = []
        for command in commands:
            for action in actions:
                for option in action.option_strings:
                    flag = '-l ' + option[2:] if option.startswith('--') else '-s ' + option[1:]
                    row = f'complete -c {command} {flag}'
                    if not isinstance(action, (argparse._HelpAction, argparse._VersionAction, argparse._StoreTrueAction, argparse.BooleanOptionalAction)):
                        row += ' -r'
                    if action.choices:
                        row += ' -f -a ' + shlex.quote(' '.join(map(str, action.choices)))
                    rows.append(row)
        return '\n'.join(rows) + '\n'
    specs = []
    for action in actions:
        for option in action.option_strings:
            spec = option + '[quantizer option]'
            if not isinstance(action, (argparse._HelpAction, argparse._VersionAction, argparse._StoreTrueAction, argparse.BooleanOptionalAction)):
                spec += ':value:' + ('(' + ' '.join(map(str, action.choices)) + ')' if action.choices else '_files')
            specs.append(shlex.quote(spec))
    return '#compdef ' + ' '.join(commands) + '\n_arguments -s \\\n  ' + ' \\\n  '.join([*specs, "'*:checkpoint:_files'"]) + '\n'
