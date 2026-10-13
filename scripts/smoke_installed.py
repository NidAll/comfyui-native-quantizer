#!/usr/bin/env python3
"""Exercise installed distributions from outside the source checkout."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.resources
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    if not __debug__:
        raise RuntimeError('Release checks require assertions; run without -O/PYTHONOPTIMIZE')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compute', action='store_true', help='requires environment-supplied PyTorch')
    args = parser.parse_args()
    import native_quantizer
    from native_quantizer import api
    from native_quantizer.identity import get_converter_version
    assert 'torch' not in sys.modules
    assert 'textual' not in sys.modules
    assert 'quantizer_tui' not in sys.modules
    assert not any(name.startswith('native_quantizer.testing') for name in sys.modules)
    version = importlib.metadata.version('comfyui-native-quantizer')
    assert get_converter_version() == version
    assert 'site-packages' in str(native_quantizer.__file__), native_quantizer.__file__
    fixture = importlib.resources.files('native_quantizer.testing').joinpath('fixtures/v1.blob.txt')
    assert fixture.is_file()
    for command in ('comfyquant', 'comfyui-native-quantizer'):
        result = subprocess.run([str(Path(sys.executable).parent / command), '--no-config',
                                 '--list-formats', '--json'], check=True, capture_output=True, text=True)
        assert len(json.loads(result.stdout)) >= 13
    result = subprocess.run([sys.executable, '-m', 'native_quantizer', '--no-config', '--version'],
                            check=True, capture_output=True, text=True)
    assert version in result.stdout
    missing = subprocess.run([sys.executable, '-m', 'native_quantizer', '--no-config'],
                             capture_output=True, text=True)
    assert missing.returncode == 2 and 'ORIGINAL_MODEL is required' in missing.stderr
    if args.compute:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
        from native_quantizer.options import ConversionOptions
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(prefix='quantizer-wheel-') as directory:
            root = Path(directory)
            source, output = root / 'input.safetensors', root / 'output.safetensors'
            weights = {
                'double_blocks.0.img_attn.qkv.weight': torch.randn(
                    64, 256, generator=torch.Generator().manual_seed(72)),
                'double_blocks.0.img_attn.norm.key_norm.weight': torch.ones(64),
                'img_in.weight': torch.ones(64, 64),
            }
            save_file(weights, str(source))
            options = ConversionOptions(model=str(source), output=str(output), architecture='flux',
                                        format='int8', device='cpu', target_runtime='cpu',
                                        max_memory='4M', validate=True, progress='off', log_level='error')
            assert api.convert(options) == 0
            assert api.verify_output(str(output))['ok']
            with safe_open(str(output), framework='pt') as checkpoint:
                assert checkpoint.get_tensor('double_blocks.0.img_attn.qkv.weight').dtype == torch.int8
                assert torch.equal(checkpoint.get_tensor('img_in.weight'), weights['img_in.weight'])
    print(f'Installed distribution smoke passed: {version}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
