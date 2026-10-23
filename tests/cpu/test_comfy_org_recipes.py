"""Exact reference selection, native decoding, routing and failure boundaries."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from native_quantizer import api
from native_quantizer.architectures.detection import detect_architecture
from native_quantizer.errors import PolicyError, QualityGateError, UsageError, OutputError, UnknownArchitectureError
from native_quantizer.metadata import read_quant_metadata
from native_quantizer.options import ConversionOptions
from native_quantizer.planning.classification import classify_tensors
from native_quantizer.planning.types import DecisionKind
from native_quantizer.testing.self_tests import _ckpt
from tests.cpu.test_architecture_detection import headers
from tests.support import isolated_module as setUpModule, load_comfy_cpu


class ComfyOrgTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source.safetensors'
        self.output = self.root / 'output.safetensors'
        self.recipe_path = self.root / 'recipe.json'
        g = torch.Generator().manual_seed(120)
        self.sd = {
            'shared_block_modulation.weight': torch.randn(1536, 256, generator=g).bfloat16(),
            'layers.0.proj.qkv.weight': torch.randn(768, 256, generator=g).bfloat16(),
            'layers.0.proj.ffn_up_gate.weight': torch.randn(1024, 256, generator=g).bfloat16(),
            'layers.0.proj.ffn_down.weight': torch.randn(256, 192, generator=g).bfloat16(),
            'layers.0.proj.o.weight': torch.randn(256, 272, generator=g).bfloat16(),
            'video_embedder.weight': torch.randn(256, 128, generator=g).bfloat16(),
            'text_embedder.blocks.0.ffn_up.weight': torch.randn(256, 256, generator=g).bfloat16(),
        }
        save_file(self.sd, str(self.source))
        self.recipe = {
            'schema': 'comfy-org-int8-recipe/v1', 'architecture': 'moss_image',
            'source': {'repo': 'Comfy-Org/MOSS-Image', 'revision': 'a' * 40,
                       'file': 'diffusion_models/fixture_int8_convrot.safetensors'},
            'weights': {k: list(v.shape) for k, v in self.sd.items()},
            'layers': {
                'layers.0.proj.qkv': {'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 256},
                'layers.0.proj.ffn_down': {'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 64},
                'layers.0.proj.ffn_up_gate': {'format': 'int8_tensorwise'},
                'layers.0.proj.o': {'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 16},
            },
        }
        self.write_recipe()

    def write_recipe(self):
        self.recipe_path.write_text(json.dumps(self.recipe))

    def convert(self, **changes):
        options = dict(model=str(self.source), output=str(self.output), format='int8-convrot',
                       reference_recipe=str(self.recipe_path), device='cpu', target_runtime='cpu',
                       max_memory='4M', progress='off', validate=True)
        options.update(changes)
        with contextlib.redirect_stdout(io.StringIO()):
            return api.convert(ConversionOptions(**options))

    def test_exact_selection_rotations_and_kitchen_decode(self):
        self.assertEqual(self.convert(), 0)
        quant = read_quant_metadata(str(self.output))['layers']
        self.assertEqual(quant, self.recipe['layers'])
        self.assertTrue(api.verify_output(str(self.output))['ok'])
        sd = load_file(str(self.output))
        from comfy_kitchen.tensor import TensorWiseINT8Layout as Layout
        for layer, conf in quant.items():
            weight = layer + '.weight'
            params = Layout.Params(scale=sd[weight + '_scale'], orig_dtype=torch.bfloat16,
                                   orig_shape=tuple(self.sd[weight].shape),
                                   convrot=conf.get('convrot', False), convrot_groupsize=conf.get('convrot_groupsize', 256))
            restored = Layout.dequantize(sd[weight], params)
            error = (restored.float() - self.sd[weight].float()).norm() / self.sd[weight].float().norm()
            self.assertLess(float(error), 0.025)
        for name in self.sd:
            if name[:-7] not in quant:
                self.assertEqual(sd[name].dtype, self.sd[name].dtype)
                self.assertTrue(torch.equal(sd[name], self.sd[name]))

    def test_recipe_refuses_wrong_layout_or_settings(self):
        original = copy.deepcopy(self.recipe)
        for change in ('shape', 'missing', 'groupsize', 'expert', 'architecture'):
            self.recipe = copy.deepcopy(original)
            if change == 'shape': self.recipe['weights']['video_embedder.weight'][0] += 1
            if change == 'missing': del self.recipe['weights']['video_embedder.weight']
            if change == 'groupsize': self.recipe['layers']['layers.0.proj.qkv']['convrot_groupsize'] = 128
            if change == 'expert': self.recipe['layers']['layers.0.proj.qkv']['num_experts'] = 128
            if change == 'architecture': self.recipe['architecture'] = 'fibo2'
            self.write_recipe()
            with self.subTest(change=change), self.assertRaises(PolicyError): self.convert()
            self.assertFalse(self.output.exists())

    def test_recipe_does_not_silently_drop_layers_on_quality_failure(self):
        with self.assertRaises(QualityGateError): self.convert(error_threshold=0.0)
        self.assertFalse(self.output.exists())

    def test_recipe_guards_and_input_preservation(self):
        before = self.recipe_path.read_bytes()
        for options in ({'include': ['.*']}, {'format': 'mixed'}, {'full_precision_matmul': True}):
            with self.subTest(options=options), self.assertRaises(UsageError): self.convert(**options)
        for field in ('output', 'report', 'json_log', 'events_jsonl'):
            with self.subTest(field=field), self.assertRaises((OutputError, UsageError)):
                self.convert(**{field: str(self.recipe_path)})
            self.assertEqual(self.recipe_path.read_bytes(), before)

    def test_new_family_partial_signatures_and_variants(self):
        for family in ('moss_image', 'fibo2'):
            tensors = headers()[family]
            for missing in tensors:
                partial = {k: v for k, v in tensors.items() if k != missing}
                with self.subTest(family=family, missing=missing), self.assertRaises(UnknownArchitectureError):
                    detect_architecture(_ckpt(list(partial.items())))
            variant = dict(tensors)
            if family == 'moss_image':
                del variant['layers.0.proj.qkv.weight']
                variant['layers.0.proj.q.weight'] = (256, 256)
            else:
                del variant['layers.0.feed_forward.w13.weight']
                variant['layers.0.feed_forward.w1.weight'] = (512, 256)
            self.assertEqual(detect_architecture(_ckpt(list(variant.items()))).architecture, family)

    def test_nested_qwen_and_fused_ming(self):
        for prefix in ('', 'model.'):
            tensors = [(prefix + 'language_model.layers.0.self_attn.q_proj.weight', (4096, 2560)),
                       (prefix + 'language_model.layers.0.mlp.gate_proj.weight', (9728, 2560))]
            self.assertEqual(detect_architecture(_ckpt(tensors)).architecture, 'text_encoder_llm')
        tensors = dict(headers()['ming_image'])
        del tensors['layers.0.attention.to_q.weight']
        tensors['layers.0.attention.qkv.weight'] = (11520, 3840)
        info = _ckpt(list(tensors.items()))
        detection = detect_architecture(info)
        self.assertEqual(detection.architecture, 'ming_image')
        selected = classify_tensors(info, detection, 'mixed', None, [], [], [], None, None)
        self.assertIn('layers.0.attention.qkv.weight', {d.name for d in selected if d.kind == DecisionKind.QUANTIZE})
        hybrid = _ckpt([
            ('model.language_model.layers.0.linear_attn.A_log', (32,)),
            ('model.language_model.layers.0.linear_attn.in_proj_qkv.weight', (8192, 4096)),
            ('model.language_model.layers.0.mlp.gate_proj.weight', (12288, 4096)),
            ('model.language_model.layers.0.linear_attn.in_proj_a.weight', (32, 4096)),
            ('model.language_model.layers.0.linear_attn.in_proj_b.weight', (32, 4096)),
        ])
        detection = detect_architecture(hybrid)
        self.assertEqual(detection.architecture, 'text_encoder_qwen35')
        selected = classify_tensors(hybrid, detection, 'mixed', None, [], [], [], None, None)
        quantized = {d.name for d in selected if d.kind == DecisionKind.QUANTIZE}
        self.assertEqual(len(quantized), 2)

    def test_metadata_reader_never_reads_interleaved_weights(self):
        from scripts.read_comfy_org_recipe import read_quant_configs
        conf = json.dumps({'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 64}).encode()
        payload = conf + b'WEIGHTS' + conf
        header = {
            'a.comfy_quant': {'dtype': 'U8', 'shape': [len(conf)], 'data_offsets': [0, len(conf)]},
            'b.weight': {'dtype': 'I8', 'shape': [1, 7], 'data_offsets': [len(conf), len(conf) + 7]},
            'c.comfy_quant': {'dtype': 'U8', 'shape': [len(conf)], 'data_offsets': [len(conf) + 7, len(payload)]},
        }
        calls = []
        def read(start, end, label):
            calls.append((start, end))
            self.assertTrue(end < len(conf) or start >= len(conf) + 7)
            return payload[start:end+1]
        self.assertEqual(set(read_quant_configs(header, 0, read)), {'a', 'c'})
        self.assertEqual(len(calls), 2)
        header['b.weight']['data_offsets'] = [0, 7]
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            read_quant_configs(header, 0, read)

    def test_installed_new_native_consumers(self):
        ops, utils = load_comfy_cpu()
        from comfy.ldm.moss_image.model import MOSSImage
        from comfy.ldm.fibo2.model import Fibo2
        for model_class, dimensions in (
            (MOSSImage, dict(dim=256, ffn_dim=512, num_heads=2, num_layers=1,
                             text_dim=256, latent_channels=128, fused_qkv=True)),
            (Fibo2, dict(dim=256, ffn_dim=512, n_heads=2, n_layers=1, n_refiner_layers=1,
                         injection_layers=(0,), cap_feat_dim=256, perceiver_layers=1,
                         max_gist_tokens=256, in_channels=128, fused_ffn=True)),
        ):
            with self.subTest(model=model_class.__name__):
                base = model_class(**dimensions, operations=ops.disable_weight_init,
                                   dtype=torch.float32, device='cpu')
                g = torch.Generator().manual_seed(20)
                state = {name: (torch.ones_like(value) if 'norm' in name and name.endswith('.weight')
                                else torch.randn(value.shape, generator=g) * .02)
                         for name, value in base.state_dict().items()}
                source = self.root / (model_class.__name__ + '.safetensors')
                output = self.root / (model_class.__name__ + '-int8.safetensors')
                save_file(state, str(source))
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(api.convert(ConversionOptions(model=str(source), output=str(output),
                        format='int8-convrot', device='cpu', target_runtime='cpu', max_memory='8M',
                        validate=True, progress='off', log_level='error')), 0)
                with safe_open(str(output), framework='pt') as st:
                    quantized = {name: st.get_tensor(name) for name in st.keys()}
                    metadata = st.metadata()
                quantized, _ = utils.convert_old_quants(quantized, metadata=metadata)
                native = model_class(**dimensions, operations=ops.mixed_precision_ops(compute_dtype=torch.float32),
                                     dtype=torch.float32, device='cpu')
                native.load_state_dict(quantized, strict=True)
                with torch.inference_mode():
                    context_dim = 512 if model_class is Fibo2 else 256
                    actual = native(torch.randn(1, 128, 2, 2), torch.tensor([.5]), context=torch.randn(1, 4, context_dim))
                self.assertEqual(tuple(actual.shape), (1, 128, 2, 2))
                self.assertTrue(torch.isfinite(actual).all())
