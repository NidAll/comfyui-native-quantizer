"""Regression checks for package boundaries and the public conversion API."""
import ast
import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from tests_support import isolated_module as setUpModule
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / 'native_quantizer'


class ArchitectureTests(unittest.TestCase):
    def test_importing_public_api_keeps_computation_and_frontend_unloaded(self):
        code = '''
import sys
import native_quantizer
from native_quantizer import api
import comfyui_native_quantizer
assert 'torch' not in sys.modules
assert 'textual' not in sys.modules
assert 'quantizer_tui' not in sys.modules
assert not any(name.startswith('native_quantizer.testing') for name in sys.modules)
'''
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)

    def test_normal_cli_does_not_load_regression_fixtures(self):
        code = '''
import contextlib, io, sys
from native_quantizer.cli.commands import main
with contextlib.redirect_stdout(io.StringIO()):
    assert main(['--list-formats']) == 0
assert not any(name.startswith('native_quantizer.testing') for name in sys.modules)
assert 'textual' not in sys.modules
assert 'quantizer_tui' not in sys.modules
'''
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)

    def test_backend_imports_do_not_point_into_frontend_or_compatibility_commands(self):
        forbidden = {'quantizer_tui', 'textual', 'comfyui_native_quantizer',
                     'kitchen_policy', 'calibration_capture', 'quantizer_events', 'benchmark_kitchen'}
        for path in PACKAGE.rglob('*.py'):
            with self.subTest(module=path.relative_to(ROOT)):
                for node in ast.walk(ast.parse(path.read_text())):
                    names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                             else [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
                    self.assertFalse(forbidden.intersection(name.split('.')[0] for name in names))

    def test_library_layers_do_not_import_command_or_application_layers(self):
        libraries = ('checkpoint_io', 'architectures', 'codecs', 'planning',
                     'execution', 'runtime', 'validation')
        forbidden = ('native_quantizer.cli', 'native_quantizer.api',
                     'native_quantizer.application', 'native_quantizer.testing')
        for directory in libraries:
            for path in (PACKAGE / directory).rglob('*.py'):
                for node in ast.walk(ast.parse(path.read_text())):
                    names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                             else [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
                    for name in names:
                        self.assertFalse(any(name == item or name.startswith(item + '.')
                                             for item in forbidden), f'{path.name}: {name}')

    def test_internal_import_graph_has_no_cycles(self):
        modules = {'.'.join(path.relative_to(ROOT).with_suffix('').parts): path
                   for path in PACKAGE.rglob('*.py')}
        graph = {name: set() for name in modules}
        for name, path in modules.items():
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    targets = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    targets = [node.module or ''] + [(node.module or '') + '.' + alias.name
                                                    for alias in node.names]
                else:
                    continue
                graph[name].update(target for target in targets if target in modules and target != name)
        visited, active = set(), []

        def visit(name):
            self.assertNotIn(name, active, ' -> '.join([*active, name]))
            if name in visited:
                return
            active.append(name)
            for child in sorted(graph[name]):
                visit(child)
            active.pop()
            visited.add(name)
        for name in sorted(graph):
            visit(name)

    def test_script_and_module_help_match(self):
        script = subprocess.run([sys.executable, str(ROOT / 'comfyui_native_quantizer.py'), '--help'],
                                check=True, capture_output=True, text=True)
        package = subprocess.run([sys.executable, '-m', 'native_quantizer', '--help'],
                                 check=True, capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(script.stdout, package.stdout)



class PublicApiTests(unittest.TestCase):
    def test_conversion_requests_exclude_frontend_commands_and_validate_formats(self):
        from dataclasses import fields
        from native_quantizer.errors import UsageError
        from native_quantizer.options import ConversionOptions, validate_conversion_options

        names = {item.name for item in fields(ConversionOptions)}
        self.assertFalse(names.intersection({'interactive', 'self_test', 'list_formats',
                                            'list_architectures', 'verify_output'}))
        alias = ConversionOptions(format='int8_convrot')
        validate_conversion_options(alias)
        self.assertEqual(alias.format, 'int8-convrot')
        with self.assertRaises(UsageError):
            validate_conversion_options(ConversionOptions(format='unknown'))

    def test_typed_api_and_cli_emit_the_same_payload_without_mutating_request(self):
        import torch
        import safetensors.torch
        from native_quantizer import api
        from native_quantizer.cli.commands import main
        from native_quantizer.files import sha256_safetensors_payload
        from native_quantizer.options import ConversionOptions

        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input.safetensors'
            generator = torch.Generator().manual_seed(28)
            safetensors.torch.save_file({
                'double_blocks.0.img_attn.qkv.weight': torch.randn(64, 256, generator=generator),
                'double_blocks.0.img_attn.norm.key_norm.weight': torch.ones(64),
                'img_in.weight': torch.randn(64, 64, generator=generator),
            }, str(source))
            options = ConversionOptions(model=str(source), output=str(root / 'api.safetensors'),
                                        architecture='flux', format='int8', max_memory='4M',
                                        device='cpu', target_runtime='cpu', error_threshold=.35,
                                        progress='off', log_level='error')
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(api.convert(options), 0)
                self.assertEqual(main([str(source), '--output', str(root / 'cli.safetensors'),
                                       '--architecture', 'flux', '--format', 'int8', '--max-memory', '4M',
                                       '--device', 'cpu', '--target-runtime', 'cpu', '--error-threshold', '.35',
                                       '--progress', 'off', '--log-level', 'error']), 0)
            self.assertEqual(options.max_memory, '4M')
            self.assertEqual(options.include, [])
            self.assertEqual(sha256_safetensors_payload(options.output),
                             sha256_safetensors_payload(str(root / 'cli.safetensors')))
            self.assertTrue(api.verify_output(options.output)['ok'])
            self.assertEqual(api.inspect_checkpoint(str(source))['architecture'], 'flux')


if __name__ == '__main__':
    unittest.main()
