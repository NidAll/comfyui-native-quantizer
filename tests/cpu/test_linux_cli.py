"""Linux CLI, local configuration and conversion integrity regressions."""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from tests.support import isolated_module as setUpModule
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.environment = mock.patch.dict(os.environ, {'XDG_CONFIG_HOME': str(self.root / 'config'), 'XDG_DATA_HOME': ''})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.tmp.cleanup()

    def config(self, **values):
        path = self.root / 'config/comfyui-native-quantizer/config.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'schema': 1, **values}))
        return path

    def test_light_commands_and_help_do_not_load_compute_dependencies(self):
        code = '''
import contextlib, io, sys
from native_quantizer.cli.commands import main
for argv in [['--help'], ['--version'], ['--list-formats', '--json'],
             ['--list-architectures', '--json'], ['--completions','fish'], ['--show-config']]:
    with contextlib.redirect_stdout(io.StringIO()):
        try: assert main(argv) == 0
        except SystemExit as exc: assert exc.code == 0
assert 'torch' not in sys.modules
assert 'textual' not in sys.modules
'''
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)

    def test_cli_flags_override_local_defaults_and_no_config_disables_them(self):
        from native_quantizer.cli.commands import main
        self.config(device='cuda', max_memory='512M', offline=True)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(main(['--show-config', '--device=cpu', '--max-memory', '64M', '--no-offline']), 0)
        result = json.loads(stream.getvalue())['effective']
        self.assertEqual((result['device'], result['max_memory'], result['offline']), ('cpu', '64M', False))
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            main(['--no-config', '--show-config'])
        self.assertEqual(json.loads(stream.getvalue())['effective']['device'], 'auto')

    def test_invalid_config_has_usage_exit_and_help_still_works(self):
        self.config(output='do-not-write.safetensors')
        command = [sys.executable, '-m', 'native_quantizer']
        result = subprocess.run([*command, '--list-formats'], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('unknown defaults', result.stderr)
        self.assertNotIn('Traceback', result.stderr)
        help_result = subprocess.run([*command, '--help'], cwd=ROOT, capture_output=True)
        self.assertEqual(help_result.returncode, 0)
        bypass = subprocess.run([*command, '--no-config', '--list-formats'], cwd=ROOT, capture_output=True)
        self.assertEqual(bypass.returncode, 0)

    def test_json_catalogs_and_operation_conflicts(self):
        from native_quantizer.cli.commands import main
        from native_quantizer.errors import UsageError
        for flag in ('--list-formats', '--list-architectures'):
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(main([flag, '--json']), 0)
            self.assertGreater(len(json.loads(stream.getvalue())), 5)
        with self.assertRaises(UsageError):
            main(['--list-formats', '--diagnose'])
        with self.assertRaises(UsageError):
            main(['--config', 'ignored.json', '--no-config', '--show-config'])

    def test_audit_reads_source_without_executing_it_and_reports_missing(self):
        from native_quantizer.architectures.audit import audit_comfy_architectures
        source = self.root / 'comfy/supported_models.py'
        source.parent.mkdir()
        source.write_text("raise RuntimeError('must not execute')\nmodels = [Flux, UnreleasedModel]\n")
        result = audit_comfy_architectures(str(self.root))
        self.assertEqual(result['model_classes'], 2)
        self.assertEqual(result['missing'], ['UnreleasedModel'])
        self.assertEqual(result['covered'][0]['family'], 'flux')

    def test_offline_rejects_model_download_before_hub_access(self):
        from native_quantizer.errors import UsageError
        from native_quantizer.options import ConversionOptions
        from native_quantizer.sources.huggingface import download_hf_component
        with self.assertRaisesRegex(UsageError, 'downloads are disabled'):
            download_hf_component(ConversionOptions(hf_repo='owner/model', offline=True))

    def test_linux_install_and_generated_commands_work_from_another_directory(self):
        prefix = self.root / 'prefix with spaces'
        config = self.root / 'config'
        subprocess.run([sys.executable, str(ROOT / 'scripts/install_linux.py'),
                        '--prefix', str(prefix), '--config-home', str(config),
                        '--offline-default'], check=True, capture_output=True)
        command = prefix / 'bin/comfyquant'
        result = subprocess.run([str(command), '--show-config'], cwd=self.root, capture_output=True, text=True, check=True)
        self.assertTrue(json.loads(result.stdout)['effective']['offline'])
        self.assertEqual({p.name for p in (prefix / 'bin').iterdir()},
                         {'comfyquant', 'comfyui-native-quantizer'})
        for path in (prefix / 'bin').iterdir():
            subprocess.run(['sh', '-n', str(path)], check=True)
        for shell, path in [('bash', prefix / 'share/bash-completion/completions/comfyquant'),
                            ('fish', config / 'fish/completions/comfyquant.fish'),
                            ('zsh', prefix / 'share/zsh/site-functions/_comfyquant')]:
            if shutil.which(shell):
                subprocess.run([shell, '-n', str(path)], check=True)


class ConversionIntegrityTests(unittest.TestCase):
    def test_explicit_gpu_never_silently_falls_back_and_cpu_skips_probe(self):
        from native_quantizer.errors import RuntimeCompatibilityError
        from native_quantizer.runtime.conversion import conversion_device
        with mock.patch('torch.cuda.is_available', return_value=False) as available:
            self.assertEqual(conversion_device('cpu')[1], 'cpu')
            available.assert_not_called()
            self.assertEqual(conversion_device('auto')[1], 'cpu')
            with self.assertRaises(RuntimeCompatibilityError):
                conversion_device('cuda')
        with mock.patch('torch.cuda.is_available', return_value=True), mock.patch('torch.version.hip', None):
            self.assertEqual(conversion_device('auto')[1], 'cuda')
            with self.assertRaises(RuntimeCompatibilityError):
                conversion_device('rocm')
        with mock.patch('torch.cuda.is_available', return_value=True), mock.patch('torch.version.hip', '7.0'):
            self.assertEqual(conversion_device('auto')[1], 'rocm')
            self.assertEqual(conversion_device('rocm')[1], 'rocm')
            with self.assertRaises(RuntimeCompatibilityError):
                conversion_device('cuda')

    def test_native_conversion_rejects_source_mutation_before_publication(self):
        import torch
        import safetensors.torch
        from native_quantizer import api
        from native_quantizer.errors import InputError
        from native_quantizer.options import ConversionOptions
        from native_quantizer.execution import native
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.safetensors'
            out = Path(directory) / 'output.safetensors'
            safetensors.torch.save_file({'double_blocks.0.img_attn.qkv.weight': torch.randn(64, 256)}, str(source))
            encode = native._native_encode

            def mutate(*args, **kwargs):
                result = encode(*args, **kwargs)
                with source.open('r+b') as stream:
                    stream.seek(-1, 2)
                    value = stream.read(1)
                    stream.seek(-1, 2)
                    stream.write(bytes([value[0] ^ 1]))
                return result

            with mock.patch.object(native, '_native_encode', side_effect=mutate), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(InputError, 'source files changed'):
                    api.convert(ConversionOptions(model=str(source), output=str(out), architecture='flux',
                                format='int8', device='cpu', max_memory='1M'))
            self.assertFalse(out.exists())

    def test_native_row_buffers_do_not_alias_the_checkpoint(self):
        import torch
        import safetensors.torch
        from native_quantizer.checkpoint_io.reader import CheckpointReader, discover_checkpoint
        from native_quantizer.codecs.bounded import _native_rows
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.safetensors'
            for dtype in (torch.float32, torch.bfloat16):
                safetensors.torch.save_file({'weight': torch.ones(4, 256, dtype=dtype)}, str(path))
                info = discover_checkpoint(str(path))
                with CheckpointReader(info) as reader:
                    row = next(_native_rows(reader, info.by_name('weight'), 1 << 20))
                    self.assertEqual(row.dtype, torch.float32)
                    row.zero_()
                    self.assertTrue(bool(reader.read_tensor('weight').eq(1).all()))


if __name__ == '__main__':
    unittest.main()
