"""Independent native Flux 3 layouts, safe policy and bounded consumer checks."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import comfyui_native_quantizer as q
from tests.support import isolated_module as setUpModule, load_comfy_cpu


SCOPES = ('content_mode_blocks.image', 'content_mode_blocks.image_cond',
          'txt_mode_blocks', 'single_blocks')


def layout(fused=True, flow_map=True):
    shapes = {}
    selected = set()
    for scope in SCOPES:
        projections = {'attn_out': (256, 256), 'mlp_in': (1536, 256),
                       'mlp_out': (256, 768)}
        projections.update({'qkv': (768, 256)} if fused else
                           {name: (256, 256) for name in ('q_proj', 'k_proj', 'v_proj')})
        for projection, shape in projections.items():
            name = f'{scope}.0.{projection}.weight'
            shapes[name] = shape
            selected.add(name)
        for norm in ('query_norm', 'key_norm'):
            shapes[f'{scope}.0.norm.{norm}.weight'] = (128,)
    shapes.update({'emb_in.image.weight': (256, 128),
                   'emb_in.image_cond.weight': (256, 128), 'txt_in.weight': (256, 256),
                   'time_in.in_layer.weight': (256, 256),
                   'time_in.out_layer.weight': (256, 256),
                   'final_layer.image.linear.weight': (128, 256)})
    for stage in ('early_stream_modulations', 'single_stream_modulations'):
        for mode in ('image', 'image_cond', 'txt'):
            shapes[f'{stage}.{mode}.lin.weight'] = (768, 256)
    if flow_map:
        shapes.update({'ref_time_in.in_layer.weight': (256, 256),
                       'ref_time_in.out_layer.weight': (256, 256)})
    return shapes, selected


class Flux3Tests(unittest.TestCase):
    def test_variants_prefixes_policy_and_boundaries(self):
        for fused in (False, True):
            for flow_map in (False, True):
                shapes, selected = layout(fused, flow_map)
                # Unsupported mode names and unrelated auxiliary towers stay untouched.
                shapes['content_mode_blocks.audio.0.qkv.weight'] = (768, 256)
                shapes['aux.single_blocks.0.qkv.weight'] = (768, 256)
                shapes['single_blocks.1.attn_out.weight'] = (256, 256)
                selected.add('single_blocks.1.attn_out.weight')
                for prefix in ('', 'model.', *q.UNET_PREFIX_CANDIDATES):
                    with self.subTest(fused=fused, flow_map=flow_map, prefix=prefix):
                        info = q._ckpt([(prefix + k, v) for k, v in shapes.items()])
                        detection = q.detect_architecture(info)
                        self.assertEqual(detection.architecture, 'flux3')
                        self.assertEqual(detection.confidence, 'high')
                        self.assertEqual(detection.classifier_info,
                                         {'hidden_size': 256, 'qkv_layout': 'fused' if fused else 'split',
                                          'flow_map': flow_map})
                        decisions = q.classify_tensors(info, detection, q.FORMAT_MIXED,
                                                       None, [], [], [], None, None)
                        actual = {d.name for d in decisions if d.kind == q.DecisionKind.QUANTIZE}
                        self.assertEqual(actual, {prefix + name for name in selected})

    def test_partial_malformed_and_unrelated_signatures_fail(self):
        first = 'content_mode_blocks.image.0.attn_out.weight'
        second = 'single_stream_modulations.txt.lin.weight'
        cases = [{first: (256, 256)}, {second: (768, 256)},
                 {first: (256, 128), second: (768, 256)},
                 {first: (256, 256), second: (256, 256)},
                 {first: (192, 192), second: (576, 192)},
                 {'aux.' + first: (256, 256), 'aux.' + second: (768, 256)},
                 {'model.diffusion_model.' + first: (256, 256),
                  'other.' + second: (768, 256)}]
        for shapes in cases:
            with self.subTest(shapes=shapes), self.assertRaises(q.UnknownArchitectureError):
                q.detect_architecture(q._ckpt(list(shapes.items())))
        # A complete second architecture is ambiguous, rather than guessed by order.
        shapes = {first: (256, 256), second: (768, 256),
                  'joint_blocks.0.context_block.attn.qkv.weight': (768, 256),
                  'x_embedder.proj.weight': (256, 128)}
        with self.assertRaisesRegex(q.UnknownArchitectureError, 'ambiguous'):
            q.detect_architecture(q._ckpt(list(shapes.items())))
        # Flux 2 Klein remains Flux 2, not this distinct layout.
        shapes = {'double_blocks.0.img_attn.norm.key_norm.weight': (128,),
                  'double_stream_modulation_img.lin.weight': (1536, 256),
                  'img_in.weight': (256, 128)}
        self.assertEqual(q.detect_architecture(q._ckpt(list(shapes.items()))).architecture, 'flux2')

    def test_conversion_metadata_and_protected_bytes(self):
        torch.set_num_threads(2)
        for fused in (False, True):
            for prefix in ('', 'model.diffusion_model.'):
                with self.subTest(fused=fused, prefix=prefix), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    source, output = root / 'source.safetensors', root / 'output.safetensors'
                    shapes, selected = layout(fused)
                    generator = torch.Generator().manual_seed(813)
                    tensors = {prefix + name: torch.randn(shape, generator=generator).to(torch.bfloat16)
                               for name, shape in shapes.items()}
                    save_file(tensors, str(source))
                    with contextlib.redirect_stdout(io.StringIO()) as log:
                        status = q.main([str(source), '--no-config', '--format', 'int8-convrot',
                                         '--device', 'cpu', '--target-runtime', 'cpu', '--output', str(output),
                                         '--max-memory', '8M', '--validate', '--progress', 'off'])
                    self.assertEqual(status, 0, log.getvalue())
                    layers = q.read_quant_metadata(str(output))['layers']
                    self.assertEqual(set(layers), {prefix + k.removesuffix('.weight') for k in selected})
                    with safe_open(str(output), framework='pt') as checkpoint:
                        self.assertEqual(json.loads(checkpoint.metadata()['comfy_quantizer'])['architecture'], 'flux3')
                        for name, value in tensors.items():
                            actual = checkpoint.get_tensor(name)
                            if name.removeprefix(prefix) in selected:
                                self.assertEqual(actual.dtype, torch.int8)
                                self.assertEqual(actual.shape, value.shape)
                            else:
                                self.assertEqual(actual.dtype, value.dtype)
                                self.assertTrue(torch.equal(actual, value), name)
                    # Verify from serialized output alone, with the source unavailable.
                    source.unlink()
                    self.assertTrue(q.verify_output(str(output))['ok'])

    def test_native_block_consumer_preserves_quantized_projections(self):
        ops, utils = load_comfy_cpu()
        from comfy.ldm.flux3.model import Block
        torch.set_num_threads(2)
        for fused in (False, True):
            with self.subTest(fused=fused), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source, output = root / 'source.safetensors', root / 'output.safetensors'
                shapes, selected = layout(fused)
                prefix = 'model.diffusion_model.'
                generator = torch.Generator().manual_seed(942)
                weights = {prefix + k: torch.randn(v, generator=generator) for k, v in shapes.items()}
                save_file(weights, str(source))
                with contextlib.redirect_stdout(io.StringIO()) as log:
                    code = q.main([str(source), '--no-config', '--format', 'int8-convrot', '--device', 'cpu',
                                   '--target-runtime', 'cpu', '--output', str(output), '--max-memory', '8M',
                                   '--validate', '--progress', 'off'])
                self.assertEqual(code, 0, log.getvalue())
                with safe_open(str(output), framework='pt') as st:
                    metadata = st.metadata()
                    state = {key: st.get_tensor(key) for key in st.keys()}
                state, _ = utils.convert_old_quants(state, metadata=metadata)
                state = utils.state_dict_prefix_replace(state, {prefix: ''}, filter_keys=True)
                state, _ = utils.convert_old_quants(state, metadata=metadata)
                for scope in SCOPES:
                    module = Block(256, 2, 3.0, fused_qkv=fused, dtype=torch.float32,
                                   device='cpu', operations=ops.mixed_precision_ops(compute_dtype=torch.float32))
                    block_prefix = scope + '.0.'
                    payload = {k[len(block_prefix):]: v for k, v in state.items() if k.startswith(block_prefix)}
                    module.load_state_dict(payload, strict=True)
                    for name in selected:
                        if name.startswith(block_prefix):
                            projection = getattr(module, name[len(block_prefix):].removesuffix('.weight'))
                            self.assertEqual(projection.quant_format, 'int8_tensorwise')


if __name__ == '__main__':
    unittest.main()
