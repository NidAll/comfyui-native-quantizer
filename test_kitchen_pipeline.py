"""Policy, packing and bounded calibration regression tests."""
import json
import tempfile
import unittest
from tests_support import isolated_module as setUpModule
from pathlib import Path
from unittest.mock import patch

import torch
import safetensors.torch
import comfyui_native_quantizer as q
from calibration_capture import CalibrationCapture
from kitchen_policy import kernel_contract, load_kernel_costs


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_kernel_contracts(self):
        direct = kernel_contract(q.FORMAT_INT8, backend='nvidia', accelerated=True)
        native = kernel_contract(q.FORMAT_W4A4, backend='nvidia', accelerated=True,
                                 activation_bits=4, certain=True)
        expanded = kernel_contract(q.FORMAT_W4A8, backend='nvidia', accelerated=True)
        self.assertFalse(direct.weight_expansion)
        self.assertFalse(native.weight_expansion)
        self.assertLess(direct.preference, expanded.preference)
        unknown = kernel_contract(q.FORMAT_W4A4, backend='nvidia', accelerated=None,
                                  activation_bits=4, certain=False)
        self.assertGreater(unknown.preference, expanded.preference)

    def test_anima_signature_and_attention_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            source = str(Path(temp) / 'anima.safetensors')
            keys = {
                'net.blocks.0.mlp.layer1.weight': (64, 256),
                'net.blocks.0.self_attn.q_proj.weight': (64, 256),
                'net.blocks.0.cross_attn.k_proj.weight': (64, 256),
                'net.blocks.0.adaln_modulation_mlp.1.weight': (64, 256),
                'net.llm_adapter.blocks.0.cross_attn.q_proj.weight': (64, 256),
                'net.x_embedder.proj.1.weight': (64, 256),
            }
            safetensors.torch.save_file({name: torch.zeros(shape)
                                         for name, shape in keys.items()}, source)
            info = q.discover_checkpoint(source)
            detected = q.detect_architecture(info)
            self.assertEqual(detected.architecture, 'anima')
            decisions = q.classify_tensors(info, detected, q.FORMAT_MIXED, None,
                                           [], [], [], None, None)
            by_name = {decision.name: decision for decision in decisions}
            for suffix in ('mlp.layer1', 'self_attn.q_proj', 'cross_attn.k_proj'):
                self.assertEqual(by_name[f'net.blocks.0.{suffix}.weight'].kind,
                                 q.DecisionKind.QUANTIZE)
            self.assertEqual(by_name['net.llm_adapter.blocks.0.cross_attn.q_proj.weight'].kind,
                             q.DecisionKind.KEEP_PRECISION)
            self.assertEqual(by_name['net.blocks.0.adaln_modulation_mlp.1.weight'].kind,
                             q.DecisionKind.KEEP_PRECISION)

    def test_budgeted_execution_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp) / 'source.safetensors')
            names = [f'double_blocks.{i}.img_attn.qkv.weight' for i in range(2)]
            safetensors.torch.save_file({name: torch.zeros(64, 256) for name in names}, path)
            info = q.discover_checkpoint(path)
            detection = q.detect_architecture(info, override="flux")
            decisions = q.classify_tensors(info, detection, q.FORMAT_MIXED, None,
                                           [], [], [], None, None)
            cap = q.FormatRuntimeCapability(True, True, True)
            runtime = q.RuntimeCapabilities('nvidia', cap, cap, cap, cuda_capability=(8, 0))
            planner = q.MixedPlanner('balanced', None, 32 << 20, torch.device('cpu'), None,
                                     runtime=runtime, compression_target_bpp=.9)
            # Stable synthetic errors isolate budget and kernel decisions.
            def evaluate(reader, decision, fmt):
                error = {q.FORMAT_INT8: .01, q.FORMAT_W4A8: .07, q.FORMAT_W4A4: .14}[fmt]
                return q.TensorMetrics(decision.name, error, 20, .99)
            with patch.object(planner, '_evaluate_format', side_effect=evaluate):
                summary = planner.plan(info, decisions)
            self.assertEqual(summary['counts'][q.FORMAT_INT8], 1)
            self.assertEqual(summary['counts'][q.FORMAT_W4A8], 1)
            self.assertLessEqual(summary['effective_bpp'], .9)
            self.assertLessEqual(summary['global_mean_error'], .08)
            self.assertEqual(summary['execution_optimizations'], [names[0]])
            self.assertEqual(summary['calibration_coverage']['sampled_layers'], 0)
            planner.kernel_costs = {(64, 256, q.FORMAT_INT8, 8): 2.,
                                    (64, 256, q.FORMAT_W4A8, 8): 1.}
            decisions = q.classify_tensors(info, detection, q.FORMAT_MIXED, None,
                                           [], [], [], None, None)
            with patch.object(planner, '_evaluate_format', side_effect=evaluate):
                summary = planner.plan(info, decisions)
            self.assertEqual(summary['counts'], {q.FORMAT_W4A8: 2})


    def test_cost_profile_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'costs.json'
            data = {'schema': 'kitchen-costs/v1', 'kitchen_version': 'test',
                    'benchmarks': [{'shape': {'m': 8, 'n': 64, 'k': 256},
                                    'results': [{'format': q.FORMAT_INT8,
                                                 'activation_bits': 8, 'warm_ms': .1}]}]}
            path.write_text(json.dumps(data))
            costs, workloads = load_kernel_costs(path, 'test')
            self.assertEqual(costs[(64, 256, q.FORMAT_INT8, 8)], .1)
            self.assertEqual(workloads[0]['m'], 8)
            with self.assertRaises(ValueError):
                load_kernel_costs(path, 'different')
            data['benchmarks'][0]['results'][0]['warm_ms'] = float('nan')
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_kernel_costs(path, 'test')

    def test_capture_roundtrip_and_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            layer = torch.nn.Linear(16, 8, bias=False)
            key = 'double_blocks.0.img_attn.qkv.weight'
            source = str(Path(temp) / 'source.safetensors')
            safetensors.torch.save_file({key: layer.weight.detach()}, source)
            capture = CalibrationCapture({key: layer}, rows=8, max_bytes=512)
            with capture:
                for step in range(12):
                    # Noncontiguous input: capture only selected rows.
                    layer(torch.full((3, 2, 16), float(step)).transpose(0, 1))
            self.assertFalse(layer._forward_pre_hooks)
            self.assertEqual(capture.used_bytes, 512)
            self.assertGreater(float(capture.samples[key].max()), 0)
            output = Path(temp) / 'activations.npz'
            capture.save(output)
            stats = q.load_calibration(str(output), q.discover_checkpoint(source), 8, None)
            torch.testing.assert_close(stats.layers[key]['samples'], capture.samples[key])
            with self.assertRaises(FileExistsError):
                capture.save(output)
            with self.assertRaises(MemoryError):
                with CalibrationCapture({key: layer}, rows=8, max_bytes=16):
                    layer(torch.zeros(1, 16))
            self.assertFalse(layer._forward_pre_hooks)

    def test_shared_signed_packing(self):
        rotated = torch.tensor([[-7., -3., 0., 1., 4., 7.]])
        packed, scale = q.quantize_rotated_int4(rotated)
        self.assertTrue(packed.is_contiguous())
        self.assertEqual(packed.dtype, torch.int8)
        torch.testing.assert_close(q.unpack_int4_signed(packed).float() * scale[:, None], rotated)
        with self.assertRaises(q.PolicyError):
            q.quantize_rotated_int4(torch.full((2, 8), float('nan')))


if __name__ == '__main__':
    unittest.main()
