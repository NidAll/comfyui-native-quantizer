"""LFM3-VL header, policy, conversion and offline bundle contracts."""
import contextlib
import dataclasses
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tests.support import isolated_module as setUpModule
import torch
from safetensors import safe_open
from safetensors.torch import save_file
import comfyui_native_quantizer as q
from native_quantizer.options import ConversionOptions, validate_conversion_options
from native_quantizer.planning.classification import classify_tensors
from native_quantizer.planning.types import DecisionKind


SIGNATURE = {
    'language_model.embed_tokens.weight': (128000, 2048),
    'language_model.layers.0.conv.in_proj.weight': (6144, 2048),
    'language_model.layers.0.feed_forward.w1.weight': (7680, 2048),
    'language_model.layers.2.self_attn.k_proj.weight': (512, 2048),
    'vision_tower.patchifier.weight': (1024, 1536),
    'multi_modal_projector.linear_2.weight': (2048, 4096),
}


class Lfm3Tests(unittest.TestCase):
    def test_shape_checked_complete_signature(self):
        for prefix in ('', 'model.'):
            cases = {prefix + k: v for k, v in SIGNATURE.items()}
            self.assertEqual(q.detect_architecture(q._ckpt(list(cases.items()))).architecture, 'lfm3_vl_2b')
            for key in cases:
                with self.subTest(prefix=prefix, key=key):
                    partial = dict(cases)
                    del partial[key]
                    with self.assertRaises(q.UnknownArchitectureError):
                        q.detect_architecture(q._ckpt(list(partial.items())))
            for key in cases:
                partial = dict(cases)
                partial[key] = (17, 23)
                with self.assertRaises(q.UnknownArchitectureError):
                    q.detect_architecture(q._ckpt(list(partial.items())))

    def test_policy_boundaries(self):
        selected = ['model.language_model.layers.1.feed_forward.w1.weight',
                    'model.language_model.layers.0.conv.in_proj.weight',
                    'model.language_model.layers.2.self_attn.k_proj.weight']
        protected = ['model.language_model.layers.2.self_attn.gate_proj.weight',
                     'model.language_model.layers.0.conv.conv.weight',
                     'model.language_model.embed_tokens.weight', 'lm_head.weight',
                     'model.vision_tower.layers.0.conv.in_proj.weight',
                     'model.multi_modal_projector.linear_2.weight']
        info = q._ckpt([(name, (256, 256)) for name in selected + protected])
        decisions = classify_tensors(info, q.detect_architecture(info, override='lfm3_vl_2b'),
                                     'mixed', None, [], [], [], None, None)
        self.assertEqual({d.name for d in decisions if d.kind == DecisionKind.QUANTIZE}, set(selected))

    def test_convrot_sizes_and_preserved_tensors(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / 'source.safetensors'
            gen = torch.Generator().manual_seed(34)
            selected = 'model.language_model.layers.0.feed_forward.w1.weight'
            narrow = 'model.language_model.layers.1.feed_forward.w2.weight'
            protected = 'model.multi_modal_projector.linear_2.weight'
            tensors = {selected: torch.randn(256, 256, generator=gen, dtype=torch.bfloat16),
                       narrow: torch.randn(256, 64, generator=gen, dtype=torch.bfloat16),
                       protected: torch.randn(256, 256, generator=gen, dtype=torch.bfloat16)}
            save_file(tensors, str(src))
            for group in (16, 64, 256):
                out = root / f'{group}.safetensors'
                with contextlib.redirect_stdout(io.StringIO()) as log:
                    code = q.main([str(src), '--architecture', 'lfm3_vl_2b', '--format', 'int8-convrot',
                        '--int8-convrot-groupsize', str(group), '--output', str(out), '--device', 'cpu',
                        '--max-memory', '4M', '--validate', '--progress', 'off', '--no-config'])
                self.assertEqual(code, 0, log.getvalue())
                with safe_open(str(out), framework='pt') as reader:
                    self.assertEqual(reader.get_tensor(selected).dtype, torch.int8)
                    conf = bytes(reader.get_tensor(selected[:-7] + '.comfy_quant').tolist())
                    self.assertEqual(json.loads(conf)['convrot_groupsize'], group)
                    self.assertTrue(torch.equal(reader.get_tensor(protected), tensors[protected]))
                    if group == 256:
                        self.assertTrue(torch.equal(reader.get_tensor(narrow), tensors[narrow]))
                    else:
                        self.assertEqual(reader.get_tensor(narrow).dtype, torch.int8)
                self.assertTrue(q.verify_output(str(out))['ok'])

    def test_option_rejects_misuse(self):
        for opts in (ConversionOptions(format='int8', int8_convrot_groupsize=16),
                     ConversionOptions(format='int8-convrot', int8_convrot_groupsize=32),
                     ConversionOptions(format='int8-convrot', int8_convrot_groupsize=16, reference_recipe='x')):
            with self.assertRaises(q.UsageError):
                validate_conversion_options(opts)

    @unittest.skipUnless((Path(__file__).resolve().parents[3] / 'ComfyUI/custom_nodes/'
                          'ComfyUI-PromptStyler-Next/scripts/prepare_lfm3.py').is_file(),
                         'optional PromptStyler consumer checkout is absent')
    def test_offline_bundle_and_quantized_provenance(self):
        script = (Path(__file__).resolve().parents[3] / 'ComfyUI/custom_nodes/'
                  'ComfyUI-PromptStyler-Next/scripts/prepare_lfm3.py')
        spec = importlib.util.spec_from_file_location('prepare_lfm3', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in module.ASSET_FILES:
                (root / name).write_text('{}' if name.endswith('.json') else 'template', encoding='utf-8')
            (root / 'config.json').write_text('{"model_type":"lfm3_vl"}', encoding='utf-8')
            embedding = torch.randn(64, 256, dtype=torch.bfloat16)
            selected = torch.randn(256, 256, dtype=torch.bfloat16)
            save_file({module.EMBEDDING: embedding,
                       'model.language_model.layers.0.feed_forward.w1.weight': selected},
                      str(root / 'model.safetensors'))
            for fmt in ('bf16', 'int8-convrot'):
                out = root / f'{fmt}.safetensors'
                from native_quantizer import api
                convert = api.convert
                # Tiny fixture deliberately cannot satisfy real-model shape detection.
                def convert_tiny(options):
                    return convert(dataclasses.replace(options, architecture='lfm3_vl_2b'))
                with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(api, 'convert', convert_tiny):
                    module.prepare(root, out, format=fmt, convrot_groupsize=16 if fmt != 'bf16' else None,
                                   max_memory='4M')
                with safe_open(str(out), framework='pt') as reader:
                    manifest = json.loads(reader.metadata()['lfm3_vl'])
                    self.assertEqual(manifest['kind'], 'lfm3_vl')
                    self.assertNotIn(module.HEAD, reader.keys())
                    self.assertTrue(torch.equal(reader.get_tensor(module.EMBEDDING), embedding))
                    payload = bytes(reader.get_tensor(module.ASSET_KEY).tolist())
                    import hashlib
                    self.assertEqual(hashlib.sha256(payload).hexdigest(), manifest['assets_sha256'])
                    self.assertEqual(set(json.loads(payload)), set(module.ASSET_FILES))
                if fmt == 'int8-convrot':
                    self.assertTrue(q.verify_output(str(out))['ok'])
                with self.assertRaises(ValueError):
                    module.prepare(root, out, format=fmt)


if __name__ == '__main__':
    unittest.main()
