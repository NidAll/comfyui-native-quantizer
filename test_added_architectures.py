"""Independent headers, protected tensors and native consumers for new families."""
import contextlib
import importlib.util
import io
import os
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from native_quantizer import api
from native_quantizer.options import ConversionOptions
from native_quantizer.errors import UnknownArchitectureError
import comfyui_native_quantizer as q
from test_architecture_detection import headers
from tests_support import isolated_module as setUpModule, load_comfy_cpu


class AddedArchitectureTests(unittest.TestCase):
    def test_complete_signatures_and_shape_boundaries(self):
        for family in ('iris', 'kandinsky6', 'kandinsky6_sr'):
            fixture = headers()[family]
            for missing in fixture:
                if family == 'iris' and missing == 'blocks.0.mlp_x.w3.weight':
                    continue
                with self.subTest(family=family, missing=missing):
                    incomplete = {k: v for k, v in fixture.items() if k != missing}
                    with self.assertRaises(UnknownArchitectureError):
                        q.detect_architecture(q._ckpt(list(incomplete.items())))
            malformed = dict(fixture, **{'visual_embeddings.in_layer.weight': (255, 33)}) if family != 'iris' else dict(fixture, **{'s_embedder.proj.weight': (256, 13)})
            with self.assertRaises(UnknownArchitectureError):
                q.detect_architecture(q._ckpt(list(malformed.items())))
            if family == 'iris':
                mismatched = {**fixture, 'blocks.0.mlp_x.w3.weight': (511, 256)}
                with self.assertRaises(UnknownArchitectureError):
                    q.detect_architecture(q._ckpt(list(mismatched.items())))

    def test_iris_fused_and_unaligned_original_mlp(self):
        fixture = headers()['iris']
        for fused in (False, True):
            state = dict(fixture)
            state['blocks.0.mlp_x.w2.weight'] = (2560, 6826)
            state['s_embedder.proj.weight'] = (2560, 12)
            state['y_embedder.refiner.proj.weight'] = (2560, 64)
            state['y_pos_embedding'] = (1, 8, 2560)
            state['blocks.0.attn.k_proj_x.weight'] = (640, 2560)
            state['blocks.0.mlp_x.w1.weight'] = state['blocks.0.mlp_x.w3.weight'] = (6826, 2560)
            if fused:
                del state['blocks.0.mlp_x.w1.weight'], state['blocks.0.mlp_x.w3.weight']
                state['blocks.0.mlp_x.gate_up.weight'] = (13652, 2560)
            detection = q.detect_architecture(q._ckpt(list(state.items())))
            self.assertEqual(detection.architecture, 'iris')
            self.assertEqual(detection.classifier_info['layout'], 'fused' if fused else 'unfused')
            self.assertEqual(detection.classifier_info['mlp_hidden'], 6826)
            # Neither grouped packing nor ConvRot may silently change this K.
            ok, _ = q.w6_weight_is_quantizable((2560, 6826), torch.bfloat16, 16, 256)
            self.assertFalse(ok)
            info = q._ckpt(list(state.items()))
            from native_quantizer.planning.classification import classify_tensors
            from native_quantizer.planning.types import DecisionKind
            # Explicit native formats share the FORMAT_MIXED classification pass.
            for fmt in (q.FORMAT_MIXED, q.FORMAT_W4A8, q.FORMAT_W6A8):
                for includes in ([], [r'.*mlp_x.*']):
                    decisions = classify_tensors(info, detection, fmt, None, includes, [], [], None, None)
                    for decision in decisions:
                        if '.mlp_x.' in decision.name:
                            self.assertEqual(decision.kind, DecisionKind.KEEP_PRECISION)

    def test_new_policies_convert_and_preserve_boundaries(self):
        torch.set_num_threads(2)
        for family in ('iris', 'kandinsky6', 'kandinsky6_sr'):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                state = {name: torch.randn(shape, generator=torch.Generator().manual_seed(62)) * .05
                         for name, shape in headers()[family].items()}
                source, output = root / 'source.safetensors', root / 'output.safetensors'
                save_file(state, str(source))
                with contextlib.redirect_stdout(io.StringIO()):
                    status = api.convert(ConversionOptions(model=str(source), output=str(output),
                        format='int8-convrot', device='cpu', target_runtime='cpu', max_memory='16M',
                        validate=True, progress='off', log_level='error'))
                self.assertEqual(status, 0)
                self.assertTrue(api.verify_output(str(output))['ok'])
                self.assertEqual(api.inspect_checkpoint(str(source))['architecture'], family)
                layers = q.read_quant_metadata(str(output))['layers']
                self.assertTrue(layers)
                with safe_open(str(output), framework='pt') as result:
                    for name, value in state.items():
                        if name.removesuffix('.weight') not in layers:
                            self.assertTrue(torch.equal(value, result.get_tensor(name)), name)
                policy = q.get_family(family)
                for protected in ('modulation_cores.adaln_img.weight', 'blocks.0.attn_gate_x.weight') if family == 'iris' else ('visual_blocks.0.va_modulation.out_layer.weight', 'out_layer.out_layer.weight'):
                    self.assertFalse(policy.quantize_re().search(protected), protected)

    def test_installed_iris_native_loader_and_tiny_forward(self):
        ops, _ = load_comfy_cpu()
        configured = os.environ.get('IRIS_ROOT')
        if not configured:
            self.skipTest('optional Iris consumer: set IRIS_ROOT')
        model_file = Path(configured) / 'iris/model.py'
        if not model_file.is_file():
            raise RuntimeError('IRIS_ROOT must contain the native Iris model source')
        spec = importlib.util.spec_from_file_location('_quantizer_iris_model', model_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        dimensions = dict(hidden_size=256, depth=2, dual_depth=1, num_heads=4, num_kv_heads=2,
            mlp_hidden=512, patch_size=2, text_dim=64, text_len=8, text_layers=2,
            text_layer_heads=4, pixel_hidden_size=64, pixel_attn_hidden_size=256,
            pixel_num_heads=4, pixel_depth=1, dtype=torch.float32, device='cpu')
        base = module.IrisDiT(**dimensions, operations=ops.disable_weight_init)
        generator = torch.Generator().manual_seed(123)
        state = {name: (torch.ones_like(value) if 'norm' in name and name.endswith('.weight')
                        else torch.randn(value.shape, generator=generator) * .02)
                 for name, value in base.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'iris.safetensors', Path(directory) / 'int8.safetensors'
            save_file(state, str(source))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(api.convert(ConversionOptions(model=str(source), output=str(output),
                    format='int8-convrot', device='cpu', target_runtime='cpu', max_memory='16M',
                    validate=True, progress='off', log_level='error')), 0)
            with safe_open(str(output), framework='pt') as st:
                quantized = {name: st.get_tensor(name) for name in st.keys()}
                metadata = st.metadata()
            quantized, _ = __import__('comfy.utils', fromlist=['convert_old_quants']).convert_old_quants(quantized, metadata=metadata)
            native = module.IrisDiT(**dimensions, operations=ops.mixed_precision_ops(compute_dtype=torch.float32))
            native.load_state_dict(quantized, strict=True)
            with torch.inference_mode():
                actual = native(torch.randn(1, 3, 4, 4), torch.tensor([500.]),
                                context=torch.randn(1, 8, 128), attention_mask=torch.ones(1, 8))
            self.assertEqual(tuple(actual.shape), (1, 3, 4, 4))
            self.assertTrue(torch.isfinite(actual).all())


if __name__ == '__main__':
    unittest.main()
