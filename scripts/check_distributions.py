#!/usr/bin/env python3
"""Audit the built CLI wheel and source archive before release."""
from __future__ import annotations

import argparse
import hashlib
import re
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = re.compile(rb'/home/[A-Za-z0-9_.-]+/|\b(?:' + b'Ge' + b'Force' + rb'|RTX\s*[0-9]{4})\b', re.I)
FORBIDDEN_SUFFIXES = ('.pyc', '.pyo', '.log', '.safetensors', '.ckpt', '.pt', '.pth', '.bin', '.gguf')


def entries(path):
    if path.suffix == '.whl':
        with zipfile.ZipFile(path) as archive:
            return {name: archive.read(name) for name in archive.namelist() if not name.endswith('/')}
    with tarfile.open(path, 'r:gz') as archive:
        return {'/'.join(member.name.split('/')[1:]): archive.extractfile(member).read()
                for member in archive.getmembers() if member.isfile()}


def main() -> int:
    if not __debug__:
        raise RuntimeError('Release checks require assertions; run without -O/PYTHONOPTIMIZE')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    artifacts = sorted([*args.directory.glob('*.whl'), *args.directory.glob('*.tar.gz')])
    assert len(artifacts) == 2, 'expected only the CLI wheel and sdist'
    observed = set()
    checksums = []
    for path in artifacts:
        files = entries(path)
        is_wheel = path.suffix == '.whl'
        metadata_name = next(name for name in files if name.endswith('.dist-info/METADATA')) if is_wheel else 'PKG-INFO'
        metadata = BytesParser().parsebytes(files[metadata_name])
        name, version = metadata['Name'], metadata['Version']
        assert name == 'comfyui-native-quantizer', name
        observed.add((name, is_wheel))
        config = ROOT / 'pyproject.toml'
        assert re.search(r'^version = "([^"]+)"', config.read_text(), re.M)[1] == version
        assert metadata['License-Expression'] == 'Apache-2.0'
        licenses = [payload for file, payload in files.items() if file.endswith('/LICENSE') or file == 'LICENSE']
        assert (ROOT / 'LICENSE').read_bytes() in licenses, f'{path.name}: missing complete license'
        for file, payload in files.items():
            assert '..' not in Path(file).parts and not Path(file).is_absolute(), file
            assert not file.endswith(FORBIDDEN_SUFFIXES), file
            assert Path(file).name != 'AGENTS.md', file
            assert '__pycache__' not in Path(file).parts, file
            assert not {'build', 'dist'}.intersection(Path(file).parts), file
            assert not file.startswith('quantizer_tui/'), file
            assert Path(file).name not in {'requirements-tui.txt', 'test_quantizer_tui.py'}, file
            if file.endswith(('.py', '.md', '.toml', '.json', '.yaml', '.yml')):
                assert not PRIVATE.search(payload), f'{file}: private checkout or hardware identity'
        if name == 'comfyui-native-quantizer':
            assert 'native_quantizer/testing/fixtures/v1.blob.txt' in files
            assert 'native_quantizer/cli/entrypoint.py' in files
            if is_wheel:
                assert not any(file.startswith('quantizer_tui/') for file in files)
                assert not any(file.startswith('tests/') for file in files)
                assert not any('textual' in dep for dep in metadata.get_all('Requires-Dist', []))
            else:
                for required in ('scripts/install_linux.py', 'scripts/check.py', 'scripts/smoke_installed.py',
                                 'docs/support.md', 'docs/releasing.md', 'tests/cpu/test_project_structure.py',
                                 'tests/support.py', 'tests/cpu/test_added_architectures.py', 'docs/models/iris.md',
                                 'tests/__init__.py', 'tests/cpu/__init__.py', 'tests/cuda/__init__.py',
                                 'tests/cpu/test_mse_optimization.py', 'tests/cuda/test_mse_cuda.py',
                                 'tests/cuda/test_w6a8_cuda.py'):
                    assert required in files, required
        print(f'{path.name}: archive contract passed ({len(files)} files)')
        checksums.append(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}')
    assert len(observed) == 2, 'duplicate or missing distribution type'
    (args.directory / 'SHA256SUMS').write_text('\n'.join(checksums) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
