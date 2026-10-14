"""Quantizer output compatibility with unmodified ComfyUI prefix handling."""
import contextlib
import io
import json
import tempfile
import unittest
from tests.support import isolated_module as setUpModule, load_comfy_cpu
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import comfyui_native_quantizer as q


class QuantMetadataTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def convert(self, prefix, fmt, *extra):
        source = self.root / 'source.safetensors'
        output = self.root / 'output.safetensors'
        generator = torch.Generator().manual_seed(72)
        weights = {prefix + f'double_blocks.{i}.img_attn.qkv.weight':
                   torch.randn(64, 256, generator=generator) for i in range(6)}
        weights[prefix + 'img_in.weight'] = torch.randn(64, 256, generator=generator)
        weights[prefix + 'double_blocks.0.img_attn.norm.key_norm.weight'] = torch.ones(64)
        save_file(weights, str(source))
        with contextlib.redirect_stdout(io.StringIO()):
            code = q.main([str(source), '--output', str(output), '--format', fmt,
                           '--device', 'cpu', '--target-runtime', 'cpu', '--yes',
                           '--max-memory', '4M', '--progress', 'off', '--overwrite',
                           *extra])
        self.assertEqual(code, 0)
        self.assertTrue(q.verify_output(str(output))['ok'])
        with safe_open(str(output), framework='pt') as st:
            return source, output, weights, {k: st.get_tensor(k) for k in st.keys()}, st.metadata()

    def test_prefixes_and_formats_load_without_duplicate_metadata(self):
        self.ops, self.utils = load_comfy_cpu()
        for prefix in ('model.diffusion_model.', 'model.model.', 'net.', 'model.'):
            for fmt in ('w6a8', 'w4a8', 'mixed', 'int8', 'w4a4', 'fp8-e4m3'):
                with self.subTest(prefix=prefix, format=fmt):
                    extra = ('--profile', 'balanced', '--enable-w6a8') if fmt == 'mixed' else ()
                    source, output, weights, state, metadata = self.convert(prefix, fmt, *extra)
                    layers = q.read_quant_metadata(str(output))['layers']
                    self.assertEqual(len(layers), 6)
                    self.assertEqual(json.loads(metadata[q.METADATA_KEY_QUANT]), {'layers': {}})
                    self.assertTrue(torch.equal(state[prefix + 'img_in.weight'], weights[prefix + 'img_in.weight']))
                    # This is the original ComfyUI loading sequence, including its second conversion.
                    state, _ = self.utils.convert_old_quants(state, metadata=metadata)
                    state = self.utils.state_dict_prefix_replace(state, {prefix: ''}, filter_keys=True)
                    state, _ = self.utils.convert_old_quants(state, metadata=metadata)
                    configs = {name for name in state if name.endswith('.comfy_quant')}
                    self.assertEqual(configs, {layer[len(prefix):] + '.comfy_quant' for layer in layers})
                    for layer, conf in layers.items():
                        name = layer[len(prefix):]
                        module = self.ops.mixed_precision_ops(compute_dtype=torch.float32).Linear(256, 64, bias=False, device='cpu')
                        payload = {key[len(name) + 1:]: value for key, value in state.items() if key.startswith(name + '.')}
                        module.load_state_dict(payload, strict=True)
                        self.assertEqual(module.quant_format, conf['format'])

    def test_validation_only_reconstructs_inline_inventory(self):
        source, output, _, _, _ = self.convert('model.diffusion_model.', 'w6a8', '--validate')
        with contextlib.redirect_stdout(io.StringIO()):
            code = q.main([str(source), '--format', 'w6a8', '--output', str(output),
                           '--validation-only', '--target-runtime', 'cpu', '--progress', 'off'])
        self.assertEqual(code, 0)

    def test_conflicting_or_malformed_inline_configuration_is_rejected(self):
        _, output, _, state, metadata = self.convert('model.diffusion_model.', 'w6a8')
        layers = q.read_quant_metadata(str(output))['layers']
        layer = next(iter(layers))
        metadata[q.METADATA_KEY_QUANT] = json.dumps({'layers': {layer: {'format': 'int8_tensorwise'}}})
        bad = self.root / 'conflict.safetensors'
        save_file(state, str(bad), metadata=metadata)
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            q.read_quant_metadata(str(bad))
        self.assertFalse(q.verify_output(str(bad))['ok'])
        state[layer + '.comfy_quant'] = torch.ones(1, dtype=torch.float32)
        save_file(state, str(bad), metadata=metadata)
        self.assertFalse(q.verify_output(str(bad))['ok'])

    def test_interrupted_inline_metadata_write_resumes(self):
        source, output, _, _, _ = self.convert('model.diffusion_model.', 'w6a8')
        output.unlink()
        original = q.ConversionEngine._record_entry
        interrupted = False

        def interrupt(engine, name, digest, entries):
            nonlocal interrupted
            original(engine, name, digest, entries)
            if name.endswith('.comfy_quant') and not interrupted:
                interrupted = True
                raise q._SimulatedCrash('metadata write interrupted')

        argv = [str(source), '--output', str(output), '--format', 'w6a8', '--device', 'cpu',
                '--target-runtime', 'cpu', '--max-memory', '4M', '--yes', '--progress', 'off']
        with patch.object(q.ConversionEngine, '_record_entry', interrupt):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(q._SimulatedCrash):
                    q.main(argv + ['--overwrite'])
        self.assertFalse(output.exists())
        with contextlib.redirect_stdout(io.StringIO()):
            code = q.main(argv + ['--resume', '--validate'])
        self.assertEqual(code, 0)
        self.assertTrue(q.verify_output(str(output))['ok'])


if __name__ == '__main__':
    unittest.main()
