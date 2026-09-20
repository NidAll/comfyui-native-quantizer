"""Regression tests for native formats, quality gates and data-only adapters.
Run: python -m unittest -v test_quantizer_native.py
Optional NVFP4/MXFP8 tests require comfy-kitchen.
"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
import safetensors.torch
from safetensors import safe_open
import comfyui_native_quantizer as q

WEIGHT = 'double_blocks.0.img_attn.qkv.weight'

class NativeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp = tempfile.TemporaryDirectory()
        self.p = Path(self.tmp.name)
        self.src = self.p / 'source.safetensors'
        g = torch.Generator().manual_seed(23)
        self.w = torch.randn(64, 256, generator=g)
        self.sd = {WEIGHT: self.w,
                   'double_blocks.0.img_attn.norm.key_norm.weight': torch.ones(64),
                   'img_in.weight': torch.randn(64, 64, generator=g),
                   'zero_buffer': torch.empty(0),
                   'counter': torch.tensor(7, dtype=torch.int64)}
        safetensors.torch.save_file(self.sd, str(self.src), metadata={'title': 'fixture'})

    def tearDown(self):
        self.tmp.cleanup()

    def convert(self, fmt, *extra, budget='4M'):
        out = self.p / (fmt + '-' + str(len(list(self.p.iterdir()))) + '.safetensors')
        with contextlib.redirect_stdout(io.StringIO()):
            code = q.main([str(self.src), '--format', fmt, '--output', str(out),
                           '--max-memory', budget, '--progress', 'off', *extra])
        self.assertEqual(code, 0)
        return out

    def load(self, out):
        with safe_open(str(out), framework='pt') as f:
            meta = f.metadata()
            sd = {k: f.get_tensor(k) for k in f.keys()}
        return sd, meta

    def test_native_formats_decode_with_kitchen(self):
        try:
            import comfy_kitchen.tensor as kt
        except ImportError:
            self.skipTest('comfy-kitchen not installed')
        classes = {'int8-convrot': kt.TensorWiseINT8Layout, 'int4-convrot': kt.TensorCoreConvRotW4A4Layout,
                   'int8': kt.TensorWiseINT8Layout, 'w4a4': kt.TensorCoreConvRotW4A4Layout,
                   'fp8-e4m3': kt.TensorCoreFP8Layout, 'fp8-e5m2': kt.TensorCoreFP8Layout,
                   'nvfp4': kt.TensorCoreNVFP4Layout, 'mxfp8': kt.TensorCoreMXFP8Layout}
        for fmt, cls in classes.items():
            with self.subTest(format=fmt):
                out = self.convert(fmt)
                sd, meta = self.load(out)
                conf = json.loads(meta['_quantization_metadata'])['layers'][WEIGHT[:-7]]
                self.assertEqual(conf['format'], q.NATIVE_FORMATS[fmt].algorithm)
                scales = {'scale': sd[WEIGHT + '_scale']}
                if fmt == 'nvfp4':
                    scales = {'scale': sd[WEIGHT + '_scale_2'], 'block_scale': sd[WEIGHT + '_scale']}
                if fmt == 'mxfp8':
                    scales['scale'] = scales['scale'].view(torch.float8_e8m0fnu)
                if fmt == 'int8-convrot':
                    self.assertIs(conf['convrot'], True)
                    self.assertEqual(conf['convrot_groupsize'], 256)
                    scales.update(convrot=True, convrot_groupsize=256)
                if fmt in ('w4a4', 'int4-convrot'):
                    scales.update(convrot_groupsize=conf['convrot_groupsize'], quant_group_size=64,
                                  linear_dtype=conf['linear_dtype'])
                if fmt == 'int4-convrot':
                    self.assertEqual(conf['linear_dtype'], 'int4')
                params = cls.Params(**scales, orig_dtype=torch.float32, orig_shape=tuple(self.w.shape))
                decoded = cls.dequantize(sd[WEIGHT], params)
                error = torch.linalg.vector_norm(decoded - self.w) / torch.linalg.vector_norm(self.w)
                self.assertLess(error.item(), 0.25)
                self.assertTrue(q.verify_native_output(str(out))['ok'])
                self.assertEqual(meta['title'], 'fixture')
                self.assertTrue(torch.equal(sd['img_in.weight'], self.sd['img_in.weight']))
                self.assertEqual(sd['zero_buffer'].numel(), 0)

    def test_chunk_invariance(self):
        for fmt in ('int8', 'w4a4', 'int8-convrot', 'int4-convrot', 'fp8-e4m3', 'fp8-e5m2'):
            with self.subTest(format=fmt):
                extra = ('--rotation-backend', 'butterfly') if fmt in ('w4a4', 'int8-convrot', 'int4-convrot') else ()
                a = self.convert(fmt, *extra, budget='32K')
                b = self.convert(fmt, *extra, budget='4M')
                self.assertEqual(q.sha256_safetensors_payload(str(a)), q.sha256_safetensors_payload(str(b)))

    def test_convrot_aliases(self):
        for alias, canonical in (('int8_convrot', 'int8-convrot'), ('int4_convrot', 'int4-convrot')):
            self.assertEqual(q.build_arg_parser().parse_args(['--format', alias]).format, canonical)

    def test_text_embedding_opt_in_and_lookup(self):
        try:
            from comfy_kitchen.tensor import TensorWiseINT8Layout
        except ImportError:
            self.skipTest('comfy-kitchen not installed')
        embedding = 'model.embed_tokens.weight'
        projection = 'model.layers.0.self_attn.q_proj.weight'
        weights = {
            embedding: torch.randn(128, 255),
            projection: self.w,
            'model.position_embeddings.weight': torch.randn(32, 255),
            'model.lm_head.weight': torch.randn(128, 255),
        }
        safetensors.torch.save_file(weights, str(self.src))
        plain, _ = self.load(self.convert('int8'))
        self.assertEqual(plain[embedding].dtype, torch.float32)
        for fmt in ('int8', 'int8-convrot'):
            with self.subTest(format=fmt):
                out = self.convert(fmt, '--quantize-text-embeddings')
                sd, meta = self.load(out)
                conf = json.loads(meta['_quantization_metadata'])['layers']
                self.assertEqual(conf[embedding[:-7]]['format'], q.FORMAT_INT8)
                self.assertNotIn('convrot', conf[embedding[:-7]])
                self.assertEqual(sd[embedding].dtype, torch.int8)
                self.assertTrue(torch.equal(sd['model.position_embeddings.weight'],
                                            weights['model.position_embeddings.weight']))
                self.assertTrue(torch.equal(sd['model.lm_head.weight'], weights['model.lm_head.weight']))
                params = TensorWiseINT8Layout.Params(
                    scale=sd[embedding + '_scale'], orig_dtype=torch.float32,
                    orig_shape=tuple(weights[embedding].shape))
                indices = torch.tensor([[0, 7, 127], [4, 0, 18]])
                actual = TensorWiseINT8Layout.dequantize_embedding(sd[embedding], params, indices)
                expected = TensorWiseINT8Layout.dequantize(sd[embedding], params)[indices]
                torch.testing.assert_close(actual, expected)
                self.assertTrue(q.verify_native_output(str(out))['ok'])

    def test_new_family_policies(self):
        cases = {
            'trellis2': ('img2shape.t_embedder.mlp.0.weight',
                         'img2shape.blocks.0.self_attn.to_qkv.weight'),
            'sensenova_u15': ('fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight',
                              'language_model.model.layers.0.self_attn.q_proj_mot_gen.weight'),
            'yue2': ('vae2llm.weight', 'model.layers.0.self_attn.qkv_proj.weight'),
            'text_encoder_qwen35': ('model.layers.0.linear_attn.A_log',
                                    'model.layers.0.linear_attn.in_proj_qkv.weight'),
            'text_encoder_jina_clip2': ('model.encoder.layers.0.mixer.Wqkv.weight',
                                        'model.encoder.layers.0.mlp.fc1.weight'),
            'text_encoder_t5gemma': ('model.encoder.layers.0.pre_self_attn_layernorm.weight',
                                     'model.encoder.layers.0.self_attn.q_proj.weight'),
            'text_encoder_yue2': ('yue2_tokenizer_json',
                                  'model.layers.0.self_attn.qkv_proj.weight'),
        }
        for family, (marker, target) in cases.items():
            with self.subTest(family=family):
                info = q._ckpt([(marker, (64, 256) if marker.endswith('.weight') else (64,)),
                                (target, (64, 256))])
                detected = q.detect_architecture(info)
                self.assertEqual(detected.architecture, family)
                decisions = q.classify_tensors(info, detected, q.FORMAT_MIXED,
                                               None, [], [], [], None, None)
                self.assertEqual(next(d.kind for d in decisions if d.name == target),
                                 q.DecisionKind.QUANTIZE)

    def test_ace_step15_policy(self):
        targets = [
            *(f'decoder.layers.0.{attention}.{projection}.weight'
              for attention in ('self_attn', 'cross_attn')
              for projection in ('q_proj', 'k_proj', 'v_proj', 'o_proj')),
            *(f'decoder.layers.0.mlp.{projection}.weight'
              for projection in ('gate_proj', 'up_proj', 'down_proj')),
            *(f'encoder.{tower}_encoder.layers.0.{block}.{projection}.weight'
              for tower in ('lyric', 'timbre')
              for block, projection in (('self_attn', 'q_proj'), ('mlp', 'gate_proj'))),
        ]
        kept = [
            'encoder.text_projector.weight',
            'encoder.lyric_encoder.embed_tokens.weight',
            'encoder.timbre_encoder.embed_tokens.weight',
            'decoder.proj_in.weight', 'decoder.proj_out.weight',
            'decoder.time_embed.0.weight', 'decoder.time_embed_r.0.weight',
            'tokenizer.layers.0.proj.weight',
            'detokenizer.layers.0.proj.weight',
        ]
        marker = 'encoder.lyric_encoder.layers.0.input_layernorm.weight'
        info = q._ckpt([(marker, (64,)), *((name, (64, 256)) for name in targets + kept)])
        detected = q.detect_architecture(info)
        self.assertEqual(detected.architecture, 'ace_step')
        self.assertIn(marker, detected.evidence)
        self.assertIn('decoder.layers.0.self_attn.q_proj.weight', detected.evidence)
        self.assertIn('decoder.layers.0.cross_attn.q_proj.weight', detected.hints)
        self.assertIn('decoder.layers.0.mlp.gate_proj.weight', detected.hints)
        decisions = {d.name: d.kind for d in q.classify_tensors(
            info, detected, q.FORMAT_MIXED, None, [], [], [], None, None)}
        for name in targets:
            with self.subTest(target=name):
                self.assertEqual(decisions[name], q.DecisionKind.QUANTIZE)
        for name in kept:
            with self.subTest(keep=name):
                self.assertEqual(decisions[name], q.DecisionKind.KEEP_PRECISION)

        legacy = q._ckpt([
            ('genre_embedder.weight', (64, 256)),
            ('encoder.layers.0.self_attn.q_proj.weight', (64, 256)),
            ('encoder.layers.0.mlp.gate_proj.weight', (64, 256)),
            ('lyric_proj.weight', (64, 256)),
        ])
        legacy_detection = q.detect_architecture(legacy)
        self.assertEqual(legacy_detection.architecture, 'ace_step')
        legacy_decisions = {d.name: d.kind for d in q.classify_tensors(
            legacy, legacy_detection, q.FORMAT_MIXED, None, [], [], [], None, None)}
        self.assertEqual(legacy_decisions['encoder.layers.0.self_attn.q_proj.weight'],
                         q.DecisionKind.QUANTIZE)
        self.assertEqual(legacy_decisions['encoder.layers.0.mlp.gate_proj.weight'],
                         q.DecisionKind.QUANTIZE)
        self.assertEqual(legacy_decisions['lyric_proj.weight'], q.DecisionKind.KEEP_PRECISION)

    def test_qwen_image21_policy(self):
        marker = [
            ('txt_in.text_norm.weight', (256,)),
            ('modulation.1.weight', (64, 256)),
            ('transformer_blocks.0.attn.norm_q.weight', (64,)),
            ('img_in.weight', (64, 256)),
            ('proj_out.weight', (64, 256)),
        ]
        attention = [f'transformer_blocks.0.attn.{name}.weight'
                     for name in ('to_q', 'to_k', 'to_v', 'to_out.0')]
        sensitive = [
            'txt_in.in_layer.weight', 'txt_in.out_layer.weight',
            'time_text_embed.timestep_embedder.linear_1.weight',
            'time_text_embed.timestep_embedder.linear_2.weight',
            'norm_out.linear.weight',
            'transformer_blocks.0.attn.norm_k.weight',
        ]
        for mlp in (('gate_layer', 'proj', 'out'), ('gate_up', 'out')):
            with self.subTest(mlp=mlp):
                targets = attention + [f'transformer_blocks.0.img_mlp.{name}.weight'
                                       for name in mlp]
                info = q._ckpt(marker + [(name, (64, 256))
                                         for name in targets + sensitive])
                detected = q.detect_architecture(info)
                self.assertEqual(detected.architecture, 'qwen_image21')
                self.assertEqual(detected.confidence, 'high')
                self.assertEqual(detected.policy.runtime_status, 'experimental')
                for fmt in (q.FORMAT_W4A8, q.FORMAT_MIXED):
                    decisions = {d.name: d.kind for d in q.classify_tensors(
                        info, detected, fmt, None, [], [], [], None, None)}
                    for name in targets:
                        self.assertEqual(decisions[name], q.DecisionKind.QUANTIZE, (fmt, name))
                    for name in sensitive + [name for name, _ in marker]:
                        self.assertNotEqual(decisions[name], q.DecisionKind.QUANTIZE,
                                            (fmt, name))

        incomplete = q._ckpt([('modulation.1.weight', (64, 256)),
                              ('transformer_blocks.0.attn.to_q.weight', (64, 256))])
        with self.assertRaises(q.UnknownArchitectureError):
            q.detect_architecture(incomplete)

        prefixed = q._ckpt([(f'model.diffusion_model.{name}', shape)
                            for name, shape in marker] + [
            ('model.diffusion_model.transformer_blocks.0.attn.to_q.weight', (64, 256)),
            ('model.diffusion_model.transformer_blocks.0.img_mlp.proj.weight', (64, 256)),
        ])
        prefixed_detection = q.detect_architecture(prefixed)
        self.assertEqual(prefixed_detection.architecture, 'qwen_image21')
        prefixed_decisions = {d.name: d.kind for d in q.classify_tensors(
            prefixed, prefixed_detection, q.FORMAT_MIXED,
            None, [], [], [], None, None)}
        self.assertEqual(prefixed_decisions[
            'model.diffusion_model.transformer_blocks.0.img_mlp.proj.weight'],
            q.DecisionKind.QUANTIZE)

    def test_qwen_image21_native_output(self):
        weights = {
            'txt_in.text_norm.weight': torch.ones(256),
            'modulation.1.weight': self.w.clone(),
            'transformer_blocks.0.attn.norm_q.weight': torch.ones(64),
            'img_in.weight': self.w.clone(),
            'proj_out.weight': self.w.clone(),
            'transformer_blocks.0.attn.to_q.weight': self.w.clone(),
            'transformer_blocks.0.img_mlp.proj.weight': self.w.clone(),
            'norm_out.linear.weight': self.w.clone(),
        }
        safetensors.torch.save_file(weights, str(self.src))
        for fmt, algorithm in (('int8-convrot', q.FORMAT_INT8),
                               ('w4a8', q.FORMAT_W4A8)):
            with self.subTest(format=fmt):
                out = self.convert(fmt)
                sd, meta = self.load(out)
                layers = json.loads(meta['_quantization_metadata'])['layers']
                for layer in ('transformer_blocks.0.attn.to_q',
                              'transformer_blocks.0.img_mlp.proj'):
                    self.assertEqual(layers[layer]['format'], algorithm)
                    if fmt == 'int8-convrot':
                        self.assertTrue(layers[layer]['convrot'])
                for name in ('modulation.1.weight', 'img_in.weight', 'proj_out.weight',
                             'norm_out.linear.weight'):
                    self.assertTrue(torch.equal(sd[name], weights[name]))
                if fmt == 'int8-convrot':
                    self.assertTrue(q.verify_native_output(str(out))['ok'])

    def test_qwen_image21_text_encoder_policy(self):
        info = q._ckpt([
            ('model.layers.0.self_attn.q_proj.weight', (64, 256)),
            ('model.layers.0.mlp.gate_proj.weight', (64, 256)),
            ('model.embed_tokens.weight', (64, 256)),
            ('visual.blocks.0.attn.qkv.weight', (64, 256)),
        ])
        detected = q.detect_architecture(info)
        self.assertEqual(detected.architecture, 'text_encoder_llm')
        decisions = {d.name: d.kind for d in q.classify_tensors(
            info, detected, q.FORMAT_MIXED, None, [], [], [], None, None)}
        self.assertEqual(decisions['model.layers.0.self_attn.q_proj.weight'],
                         q.DecisionKind.QUANTIZE)
        self.assertEqual(decisions['model.layers.0.mlp.gate_proj.weight'],
                         q.DecisionKind.QUANTIZE)
        self.assertNotEqual(decisions['model.embed_tokens.weight'], q.DecisionKind.QUANTIZE)
        self.assertNotEqual(decisions['visual.blocks.0.attn.qkv.weight'],
                            q.DecisionKind.QUANTIZE)

    def test_qwen_image21_format_dry_runs(self):
        weights = {
            'txt_in.text_norm.weight': torch.ones(256),
            'modulation.1.weight': self.w.clone(),
            'transformer_blocks.0.attn.norm_q.weight': torch.ones(64),
            'img_in.weight': self.w.clone(),
            'proj_out.weight': self.w.clone(),
            'transformer_blocks.0.attn.to_q.weight': self.w.clone(),
            'transformer_blocks.0.img_mlp.proj.weight': self.w.clone(),
        }
        safetensors.torch.save_file(weights, str(self.src))
        formats = ('int8', 'int8-convrot', 'w4a4', 'int4-convrot',
                   'w4a8', 'mixed', 'fp8-e4m3', 'fp8-e5m2',
                   'nvfp4', 'mxfp8', 'fp16', 'bf16')
        for fmt in formats:
            with self.subTest(format=fmt):
                args = [str(self.src), '--format', fmt, '--output',
                        str(self.p / f'{fmt}.safetensors'), '--dry-run',
                        '--progress', 'off']
                if fmt == 'mixed':
                    args += ['--experimental', '--target-runtime', 'cpu']
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(q.main(args), 0)
                self.assertIn('qwen_image21', output.getvalue())
                if fmt in q.NATIVE_FORMATS and fmt not in ('fp16', 'bf16'):
                    self.assertIn('candidate layers: 2', output.getvalue())

    def test_mixed_embedding_uses_int8_only(self):
        embedding = 'model.embed_tokens.weight'
        self.sd = {
            embedding: torch.randn(128, 255),
            'model.layers.0.self_attn.q_proj.weight': self.w,
            'model.position_embeddings.weight': torch.randn(32, 255),
        }
        safetensors.torch.save_file(self.sd, str(self.src))
        out = self.convert('mixed', '--experimental', '--target-runtime', 'cpu',
                           '--quantize-text-embeddings')
        sd, meta = self.load(out)
        conf = json.loads(meta['_quantization_metadata'])['layers']
        self.assertEqual(conf[embedding[:-7]]['format'], q.FORMAT_INT8)
        self.assertEqual(sd[embedding].dtype, torch.int8)
        self.assertTrue(torch.equal(sd['model.position_embeddings.weight'],
                                    self.sd['model.position_embeddings.weight']))

    def test_gpu_warnings_preserve_formats(self):
        for fmt, threshold in (('fp8-e4m3', (8, 9)), ('fp8-e5m2', (8, 9)),
                               ('nvfp4', (10, 0)), ('mxfp8', (10, 0))):
            with self.subTest(format=fmt):
                self.assertIn('SM 8.6', q.native_gpu_warning(fmt, 'nvidia', (8, 6)))
                self.assertIsNone(q.native_gpu_warning(fmt, 'nvidia', threshold))
                self.assertIsNone(q.native_gpu_warning(fmt, 'cpu', (8, 6)))

    def test_int8_convrot_unaligned_width_preserved(self):
        name = 'double_blocks.1.img_attn.qkv.weight'
        self.sd[name] = torch.randn(64, 257)
        safetensors.torch.save_file(self.sd, str(self.src))
        out = self.convert('int8-convrot')
        sd, meta = self.load(out)
        self.assertTrue(torch.equal(sd[name], self.sd[name]))
        self.assertNotIn(name[:-7], json.loads(meta['_quantization_metadata'])['layers'])
        conf = json.loads(meta['_quantization_metadata'])
        conf['layers'][WEIGHT[:-7]]['convrot_groupsize'] = 128
        meta['_quantization_metadata'] = json.dumps(conf)
        safetensors.torch.save_file(sd, str(out), metadata=meta)
        self.assertFalse(q.verify_native_output(str(out))['ok'])

    def test_convrot_zero_weights(self):
        self.sd[WEIGHT] = torch.zeros_like(self.w)
        safetensors.torch.save_file(self.sd, str(self.src))
        for fmt in ('int8-convrot', 'int4-convrot'):
            sd, meta = self.load(self.convert(fmt))
            self.assertEqual(torch.count_nonzero(sd[WEIGHT]).item(), 0)
            self.assertTrue(torch.isfinite(sd[WEIGHT + '_scale']).all())

    def test_rotation_equivalence(self):
        for size in (16, 64, 256):
            matrix = q.rotate_weight(self.w, q.build_hadamard(size), size)
            torch.testing.assert_close(q._native_convrot(self.w, size), matrix, atol=5e-6, rtol=1e-5)

    def test_precision_and_preservation(self):
        for fmt in ('fp16', 'bf16'):
            out = self.convert(fmt, '--keep-precision', 'img_in')
            sd, meta = self.load(out)
            self.assertEqual(sd[WEIGHT].dtype, torch.float16 if fmt == 'fp16' else torch.bfloat16)
            self.assertEqual(sd['img_in.weight'].dtype, torch.float32)
            self.assertEqual(sd['counter'].dtype, torch.int64)
            self.assertEqual(json.loads(meta['_quantization_metadata'])['layers'], {})

    def test_failed_gate_does_not_publish(self):
        with self.assertRaises(q.QualityGateError):
            self.convert('int8', '--error-threshold', '0')
        self.assertEqual(list(self.p.glob('int8-*.safetensors')), [])

    def test_nonfinite_keep(self):
        second = 'double_blocks.1.img_attn.qkv.weight'
        self.sd[second] = torch.full((64, 256), float('nan'))
        safetensors.torch.save_file(self.sd, str(self.src))
        out = self.convert('int8', '--nonfinite-policy', 'keep')
        sd, meta = self.load(out)
        self.assertTrue(torch.isnan(sd[second]).all())
        self.assertNotIn(second[:-7], json.loads(meta['_quantization_metadata'])['layers'])

    def test_corrupted_payload_and_metadata_rejected(self):
        out = self.convert('int8')
        with open(out, 'r+b') as f:
            f.seek(-1, 2); b = f.read(1); f.seek(-1, 2); f.write(bytes([b[0] ^ 1]))
        self.assertFalse(q.verify_native_output(str(out))['ok'])
        out = self.convert('int8')
        sd, meta = self.load(out)
        meta['_quantization_metadata'] = json.dumps({'layers': {WEIGHT[:-7]: {'format': 'nvfp4'}}})
        safetensors.torch.save_file(sd, str(out), metadata=meta)
        self.assertFalse(q.verify_native_output(str(out))['ok'])

    def test_remap_and_collision(self):
        mapping = self.p / 'mapping.json'
        mapping.write_text(json.dumps({WEIGHT: 'double_blocks.1.img_attn.qkv.weight'}))
        out = self.convert('int8', '--key-map', str(mapping))
        sd, meta = self.load(out)
        self.assertIn('double_blocks.1.img_attn.qkv.weight', sd)
        self.assertNotIn(WEIGHT, sd)
        mapping.write_text(json.dumps({WEIGHT: 'counter'}))
        with self.assertRaises(q.InputError):
            self.convert('int8', '--key-map', str(mapping))

    def test_custom_policy_unknown_architecture(self):
        unknown = self.p / 'unknown.safetensors'
        safetensors.torch.save_file({'novel.projection.weight': self.w}, str(unknown))
        policy = q.architecture_template()
        policy['family'] = 'regression_custom'
        policy['quantize'] = [r'^novel\.projection\.weight$']
        path = self.p / 'policy.json';path.write_text(json.dumps(policy))
        self.src = unknown
        try:
            out = self.convert('int8', '--architecture-policy', str(path), '--experimental')
            sd, meta = self.load(out)
            self.assertEqual(sd['novel.projection.weight'].dtype, torch.int8)
        finally:
            q.REGISTRY.pop('regression_custom', None)
            q.REGISTRY_ORDER.remove('regression_custom')

    def test_sharded_input(self):
        first = {WEIGHT: self.w}
        second = {k: v for k, v in self.sd.items() if k != WEIGHT}
        safetensors.torch.save_file(first, str(self.p / 'a.safetensors'))
        safetensors.torch.save_file(second, str(self.p / 'b.safetensors'))
        index = self.p / 'model.safetensors.index.json'
        index.write_text(json.dumps({'weight_map': {k: 'a.safetensors' if k == WEIGHT else 'b.safetensors' for k in self.sd}}))
        self.src = index
        out = self.convert('int8')
        self.assertTrue(q.verify_native_output(str(out))['ok'])

    def test_input_output_alias_and_unsupported_flags(self):
        with self.assertRaises(q.OutputError):
            q.main([str(self.src), '--output', str(self.src), '--format', 'int8', '--overwrite'])
        for flag in ('--resume', '--require-calibration'):
            with self.assertRaises(q.UsageError):
                self.convert('int8', flag)

    def test_hf_pinned_download(self):
        try:
            import huggingface_hub
        except ImportError:
            self.skipTest('huggingface-hub not installed')
        import types
        calls = []
        args = q.build_arg_parser().parse_args(['--hf-repo', 'owner/model', '--hf-subfolder', 'transformer'])
        def fetch(repo, name, **kw):
            calls.append((name, kw))
            file = self.p / Path(name).name
            if name.endswith('.json'):
                file.write_text(json.dumps({'weight_map': {WEIGHT: 'part.safetensors'}}))
            else:
                safetensors.torch.save_file({WEIGHT: self.w}, str(file))
            return str(file)
        with patch('huggingface_hub.HfApi') as api, patch('huggingface_hub.hf_hub_download', side_effect=fetch):
            api.return_value.model_info.return_value = types.SimpleNamespace(sha='a' * 40)
            api.return_value.list_repo_files.return_value = ['transformer/model.safetensors.index.json',
                                                            'transformer/part.safetensors', 'transformer/config.json']
            result = q.download_hf_component(args)
        self.assertTrue(result.endswith('.index.json'))
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(kw['revision'] == 'a' * 40 and 'local_dir' in kw for _, kw in calls))

if __name__ == '__main__':
    unittest.main()
