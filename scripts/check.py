#!/usr/bin/env python3
"""Run bounded release checks without user preferences or model downloads."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_TESTS = (
    'test_quantizer_native', 'test_kitchen_pipeline', 'test_architecture_detection',
    'test_quant_metadata', 'test_w6a8', 'test_project_structure', 'test_linux_cli',
    'test_release_contracts',
    'test_added_architectures', 'test_cli_safety',
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfy-root', type=Path, help='require this ComfyUI consumer checkout')
    parser.add_argument('--iris-root', type=Path, help='require the installed native Iris consumer')
    parser.add_argument('--cuda', action='store_true', help='require CUDA and run native W6A8 forward checks')
    args = parser.parse_args(argv)
    if args.cuda and not args.comfy_root:
        parser.error('--cuda requires --comfy-root')
    if args.iris_root and not args.comfy_root:
        parser.error('--iris-root requires --comfy-root')
    with tempfile.TemporaryDirectory(prefix='quantizer-check-') as directory:
        env = dict(os.environ)
        for key in ('CONFIG', 'STATE', 'CACHE', 'DATA'):
            env[f'XDG_{key}_HOME'] = str(Path(directory) / key.lower())
        env.pop('COMFYUI_ROOT', None)
        env.pop('COMFYUI_REQUIRED', None)
        env.pop('IRIS_ROOT', None)
        if args.comfy_root:
            env['COMFYUI_ROOT'] = str(args.comfy_root.resolve())
            env['COMFYUI_REQUIRED'] = '1'
        if args.iris_root:
            env['IRIS_ROOT'] = str(args.iris_root.resolve())
        cpu_env = {**env, 'CUDA_VISIBLE_DEVICES': ''}

        def run(*arguments, environment=cpu_env):
            print('+ python ' + ' '.join(arguments), flush=True)
            subprocess.run([sys.executable, *arguments], cwd=ROOT,
                           env=environment, check=True, timeout=600)

        run('-m', 'compileall', '-q', 'native_quantizer', 'scripts')
        run('-m', 'native_quantizer', '--no-config', '--self-test')
        run('-m', 'unittest', '-q', *CORE_TESTS)
        if args.cuda:
            # Required integration must fail, never pass by skipping the suite.
            run('-c', "import torch; from native_quantizer.runtime.kitchen import require_w6a8_api; "
                "assert torch.cuda.is_available(), 'CUDA is required'; require_w6a8_api()",
                environment=env)
            run('-m', 'unittest', '-v', 'test_w6a8_cuda', environment=env)
    print('Release checks passed. See docs/support.md for evidence limits.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f'Check failed: {exc}', file=sys.stderr)
        raise SystemExit(1)
