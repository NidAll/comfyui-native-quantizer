"""Constrained selection, measured exchanges and native publication regressions."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from safetensors import safe_open

from native_quantizer import api
from native_quantizer import application
from native_quantizer.architectures.detection import detect_architecture
from native_quantizer.calibration import CalibrationStats, split_calibration
from native_quantizer.calibration_capture import CalibrationCapture
from native_quantizer.checkpoint_io.reader import CheckpointReader, discover_checkpoint
from native_quantizer.codecs.bounded import quantize_tensor_bounded
from native_quantizer.checkpoint_io.types import CheckpointInfo, TensorMeta
from native_quantizer.codecs.dispatch import quantize_weight_by_format, dequantize_weight_by_format
from native_quantizer.errors import CalibrationError, CompressionGateError
from native_quantizer.execution.state import _plan_hash
from native_quantizer.formats import quantized_format_bytes
from native_quantizer.identity import FORMAT_INT8, FORMAT_ORIGINAL, FORMAT_W4A4, FORMAT_W4A8, FORMAT_W6A8
from native_quantizer.metadata import read_quant_metadata
from native_quantizer.options import ConversionOptions
from native_quantizer.planning.metrics import compute_weight_metrics, _OutputErrorAccumulator, TensorMetrics
from native_quantizer.planning.mixed import CandidateResult, MixedPlanner
from native_quantizer.planning.types import ConversionPlan, DecisionKind, TensorDecision
from native_quantizer.runtime.capabilities import runtime_capabilities_for
from native_quantizer.runtime.costs import load_kernel_costs, runtime_fingerprint
from native_quantizer.runtime.kitchen import kitchen_version
from tests.support import isolated_module as setUpModule


def inventory(shapes):
    return CheckpointInfo('safetensors', [], {}, [
        TensorMeta(name, torch.bfloat16, shape, 2 * shape[0] * shape[1], '', 0, 0)
        for name, shape in shapes.items()])


class AdaptiveMixedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_weighted_promotion_keeps_feasible_plan_with_unequal_layers(self):
        info = inventory({'large.weight': (1024, 256), 'small.weight': (64, 256)})
        decisions = [TensorDecision(t.name, DecisionKind.QUANTIZE, '', format=FORMAT_W4A4)
                     for t in info.tensors]
        planner = MixedPlanner('balanced', None, 32 << 20, torch.device('cpu'), None,
            layer_gate=.1, global_gate=.025, compression_target_bpp=1.)
        planner._targeted = decisions
        for t in info.tensors:
            n, k = t.shape
            planner.candidates[t.name] = {
                fmt: CandidateResult(fmt, True, estimated_bytes=size, weight_rel_l2=error)
                for fmt, size, error in (
                    (FORMAT_W4A4, quantized_format_bytes(n, k, FORMAT_W4A4), .08),
                    (FORMAT_INT8, quantized_format_bytes(n, k, FORMAT_INT8),
                     .02 if t.name.startswith('large') else .001),
                    (FORMAT_ORIGINAL, t.nbytes, 0.))}
        planner.select(info, decisions)
        planner.promote(info, decisions)
        self.assertTrue(planner.repair_feasibility(info))
        planner._enforce_gates(info, decisions)
        self.assertEqual([d.format for d in decisions], [FORMAT_INT8, FORMAT_W4A4])
        self.assertLessEqual(planner.global_mean_error(info, decisions), .025)
        self.assertLessEqual(planner._effective_bpp, 1.)

    def test_group32_mixed_budget_matches_real_output_payload(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / 'source.safetensors', root / 'output.safetensors'
            save_file({'double_blocks.0.img_attn.qkv.weight': torch.randn(128, 256).bfloat16()}, str(source))
            options = ConversionOptions(model=str(source), output=str(output), architecture='flux',
                format='mixed', profile='balanced', device='cpu', target_runtime='cpu',
                enable_w6a8=True, w6a8_group_size=32, disable_w4a4=True, disable_w4a8=True,
                disable_int8=True, quality_gate=.5, global_error_gate=.5,
                max_linear_bytes_per_param=.81, max_memory='32M', validate=True,
                progress='off', log_level='error')
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(api.convert(options), 0)
            self.assertTrue(api.verify_output(str(output))['ok'])
            self.assertEqual(read_quant_metadata(str(output))['layers']['double_blocks.0.img_attn.qkv']['group_size'], 32)

    def test_exchange_recovers_quality_with_a_compensating_layer(self):
        info = inventory({'a.weight': (128, 256), 'b.weight': (128, 256)})
        a = TensorDecision('a.weight', DecisionKind.QUANTIZE, '', format=FORMAT_W6A8)
        b = TensorDecision('b.weight', DecisionKind.QUANTIZE, '', format=FORMAT_W4A8)
        planner = MixedPlanner('adaptive', None, 32 << 20, torch.device('cpu'), None,
            enable_w6a8=True, layer_gate=.08, global_gate=.04, compression_target_bpp=.8)
        planner._targeted = [a, b]
        errors = {'a.weight': {FORMAT_W4A8: .03, FORMAT_W6A8: .02},
                  'b.weight': {FORMAT_W4A8: .06, FORMAT_W6A8: .04}}
        costs = {'a.weight': {FORMAT_W4A8: .1, FORMAT_W6A8: .3},
                 'b.weight': {FORMAT_W4A8: .2, FORMAT_W6A8: .25}}
        for d in (a, b):
            planner.candidates[d.name] = {fmt: CandidateResult(fmt, True,
                estimated_bytes=quantized_format_bytes(128, 256, fmt), act_rel_l2=error)
                for fmt, error in errors[d.name].items()}
        before_error = planner.global_mean_error(info, [a, b])
        before_bytes = planner.selected_bytes(info)
        with patch.object(planner, 'candidate_cost', side_effect=lambda info, d, c: costs[d.name][c.format]):
            planner.optimize_exchanges(info)
        planner._enforce_gates(info, [a, b])
        self.assertEqual([a.format, b.format], [FORMAT_W4A8, FORMAT_W6A8])
        self.assertLessEqual(planner.global_mean_error(info, [a, b]), before_error)
        self.assertEqual(planner.selected_bytes(info), before_bytes)
        self.assertAlmostEqual(sum(m['delta_ms'] for m in planner.exchanges[0]), -.15)
        # A single unpaired W6->W4 change cannot spend quality without compensation.
        a.format = FORMAT_W6A8
        planner._targeted = [a]
        with patch.object(planner, 'candidate_cost', side_effect=lambda info, d, c: costs[d.name][c.format]):
            planner.optimize_exchanges(info)
        self.assertEqual(a.format, FORMAT_W6A8)

    def test_w6_reference_caps_the_plan_before_any_speed_exchanges(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'source.safetensors'
            names = [f'double_blocks.{i}.img_attn.qkv.weight' for i in range(2)]
            save_file({name: torch.zeros(64, 256) for name in names}, str(source))
            info = discover_checkpoint(str(source))
            decisions = [TensorDecision(name, DecisionKind.QUANTIZE, '', layer=name.removesuffix('.weight'))
                         for name in names]
            planner = MixedPlanner('adaptive', None, 32 << 20, torch.device('cpu'), None,
                enable_w6a8=True, quality_reference='w6a8')
            def evaluate(reader, decision, fmt):
                error = {FORMAT_W4A4: .15, FORMAT_W4A8: .06, FORMAT_W6A8: .02,
                         FORMAT_INT8: .01, 'int8-convrot': .01}[fmt]
                return TensorMetrics(decision.name, error, 20., .99, act_rel_l2=error)
            with patch.object(planner, '_evaluate_format', side_effect=evaluate):
                summary = planner.plan(info, decisions)
            self.assertEqual(summary['reference_error'], .02)
            self.assertLessEqual(summary['global_mean_error'], summary['reference_error'])
            self.assertEqual(summary['counts'], {FORMAT_W6A8: 2})

    def test_workload_costs_keep_multiple_m_and_layer_frequencies(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'costs.json'
            samples = []
            for m, weight, times in ((16, 3, (.1, .2)), (256, 1, (.4, .2))):
                samples.append({'shape': dict(m=m, n=64, k=256), 'dtype': 'bfloat16', 'weight': weight,
                    'results': [dict(format=fmt, activation_bits=8, group_size=group, warm_ms=time)
                        for fmt, group, time in ((FORMAT_W4A8, 16, times[0]), (FORMAT_INT8, None, times[1]))]})
            payload = dict(schema='kitchen-costs/v3', kitchen_version=kitchen_version(),
                           runtime_fingerprint=runtime_fingerprint(), benchmarks=samples)
            path.write_text(json.dumps(payload))
            costs, workloads = load_kernel_costs(path, kitchen_version())
            planner = MixedPlanner('balanced', None, 32 << 20, torch.device('cpu'), None,
                                   kernel_costs=costs, workloads=workloads)
            self.assertAlmostEqual(planner.measured_cost((64, 256), FORMAT_W4A8, 8), .7)
            self.assertAlmostEqual(planner.measured_cost((64, 256), FORMAT_INT8, 8), .8)
            payload['runtime_fingerprint'] = 'wrong-runtime'
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'Runtime differs'):
                load_kernel_costs(path, kitchen_version())

    def test_runtime_cost_distinguishes_plain_and_rotated_int8(self):
        info = inventory({'layer.weight': (64, 256)})
        d = TensorDecision('layer.weight', DecisionKind.QUANTIZE, '', format=FORMAT_INT8, convrot=True)
        costs = {(64, 256, FORMAT_INT8, 8, None): .1, (64, 256, 'int8-convrot', 8, None): .3}
        planner = MixedPlanner('balanced', None, 32 << 20, torch.device('cpu'), None,
                               kernel_costs=costs, compression_target_bpp=1.1)
        planner._targeted = [d]
        planner.candidates[d.name] = {fmt: CandidateResult(fmt, True,
            estimated_bytes=quantized_format_bytes(64, 256, fmt), act_rel_l2=.02)
            for fmt in (FORMAT_INT8, 'int8-convrot')}
        planner.optimize_execution(info)
        self.assertFalse(d.convrot)

    def test_mixed_resume_binds_calibration_costs_and_reference(self):
        name = 'double_blocks.0.img_attn.qkv.weight'
        info = inventory({name: (64, 256)})
        plan = ConversionPlan('mixed', detect_architecture(info, override='flux'),
            [TensorDecision(name, DecisionKind.QUANTIZE, '', layer=name.removesuffix('.weight'), format=FORMAT_INT8)],
            {}, {}, [], mixed_plan={'calibration_digest': 'a', 'kernel_costs_sha256': 'b'})
        args = ConversionOptions(format='mixed')
        hashes = {'fixture': 'c'}
        original = _plan_hash(info, plan, args, hashes)
        for key in ('calibration_digest', 'kernel_costs_sha256'):
            old = plan.mixed_plan[key]
            plan.mixed_plan[key] = 'changed'
            self.assertNotEqual(original, _plan_hash(info, plan, args, hashes))
            plan.mixed_plan[key] = old
        args.quality_reference = 'w6a8'
        self.assertNotEqual(original, _plan_hash(info, plan, args, hashes))

    def test_holdout_is_disjoint_and_p95_uses_heldout_rows(self):
        rows = torch.arange(64, dtype=torch.float32).reshape(16, 4)
        stats = CalibrationStats('fixture', [], 16, {'layer.weight': {'samples': rows, 'rows': 16}})
        train, heldout = split_calibration(stats)
        self.assertEqual(train.layers['layer.weight']['rows'], 12)
        self.assertEqual(heldout.layers['layer.weight']['rows'], 4)
        self.assertFalse(set(train.layers['layer.weight']['samples'][:, 0].tolist()) &
                         set(heldout.layers['layer.weight']['samples'][:, 0].tolist()))
        stats.layers['layer.weight']['samples'] = rows[:7]
        with self.assertRaises(CalibrationError):
            split_calibration(stats)
        acc = _OutputErrorAccumulator(10)
        reference = torch.ones(10, 1)
        quantized = reference.clone()
        quantized[0] = 2
        acc.update(quantized, reference)
        self.assertGreater(acc.finish('p95'), acc.finish())

    def test_stratified_capture_reserves_rows_for_short_stages(self):
        with tempfile.TemporaryDirectory() as temp:
            layer = torch.nn.Linear(16, 8, bias=False)
            capture = CalibrationCapture({'layer.weight': layer}, rows=12, rows_per_call=4,
                max_bytes=1024, strata=('early', 'middle', 'late'))
            with capture:
                for label, count, value in (('early', 40, 0.), ('middle', 2, 10.), ('late', 1, 20.)):
                    with capture.stratum(label):
                        for _ in range(count):
                            layer(torch.full((8, 16), value))
            output = Path(temp) / 'calibration.npz'
            capture.save(output)
            import numpy as np
            with np.load(output, allow_pickle=False) as data:
                self.assertEqual(data['layer.weight'][:, 0].tolist(), [0.] * 4 + [10.] * 4 + [20.] * 4)
            self.assertFalse(layer._forward_pre_hooks)

    def test_reused_codebook_matches_fresh_encoding_without_fitting_twice(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'source.safetensors'
            name = 'double_blocks.0.img_attn.qkv.weight'
            weight = torch.randn(64, 256, generator=torch.Generator().manual_seed(11))
            save_file({name: weight}, str(source))
            info = discover_checkpoint(str(source))
            for mode in ('fixed', 'auto', 'fit'):
                baseline = quantize_weight_by_format(weight, FORMAT_W4A8, 16, 256, codebook_mode=mode)
                with CheckpointReader(info) as reader, \
                        patch('native_quantizer.codecs.bounded._gather_codebook_samples', side_effect=AssertionError('refit')):
                    reused = quantize_tensor_bounded(reader, name, FORMAT_W4A8, 16, 256,
                        32 << 20, 'cpu', codebook_mode=mode, codebook=baseline['_codebook'])
                for key in baseline:
                    torch.testing.assert_close(reused[key], baseline[key], atol=0, rtol=0)

    def test_rotated_int8_selects_native_metadata_and_validates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / 'source.safetensors', root / 'output.safetensors'
            w = (torch.randn(64, 256, generator=torch.Generator().manual_seed(37)) * .02)
            w[:, 0] = 100
            w = w.bfloat16()
            errors = []
            for fmt in (FORMAT_INT8, 'int8-convrot'):
                q = quantize_weight_by_format(w.float(), fmt, 16, 256)
                dq = dequantize_weight_by_format(q, fmt, 16, 256, torch.float32)
                errors.append(compute_weight_metrics(w, dq).rel_l2)
            self.assertLess(errors[1], errors[0])
            save_file({'net.double_blocks.0.img_attn.qkv.weight': w}, str(source))
            gate = sum(errors) / 2
            options = ConversionOptions(model=str(source), output=str(output), architecture='flux',
                format='mixed', profile='balanced', device='cpu', target_runtime='cpu',
                enable_int8_convrot=True, disable_w4a4=True, disable_w4a8=True,
                quality_gate=gate, global_error_gate=gate, max_linear_bytes_per_param=1.1,
                max_memory='32M', validate=True, progress='off', log_level='error')
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(api.convert(options), 0)
            layer = read_quant_metadata(str(output))['layers']['net.double_blocks.0.img_attn.qkv']
            self.assertEqual(layer['format'], FORMAT_INT8)
            self.assertTrue(layer['convrot'])
            self.assertTrue(api.verify_output(str(output))['ok'])

    def test_whole_file_gate_includes_protected_tensors(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / 'source.safetensors', root / 'output.safetensors'
            save_file({'double_blocks.0.img_attn.qkv.weight': torch.randn(64, 256).bfloat16(),
                       'img_in.weight': torch.zeros(1024, 1024, dtype=torch.bfloat16)}, str(source))
            options = ConversionOptions(model=str(source), output=str(output), architecture='flux',
                format='mixed', profile='balanced', target_compression=2.5, device='cpu',
                target_runtime='cpu', max_memory='32M', progress='off', log_level='error')
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(CompressionGateError):
                api.convert(options)
            self.assertFalse(output.exists())

    def test_final_file_gate_preserves_existing_output_when_metadata_grows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / 'source.safetensors', root / 'output.safetensors'
            save_file({'double_blocks.0.img_attn.qkv.weight': torch.randn(512, 1024).bfloat16()}, str(source))
            output.write_bytes(b'previous-output')
            options = ConversionOptions(model=str(source), output=str(output), architecture='flux',
                format='mixed', profile='balanced', target_compression=2.5, device='cpu',
                target_runtime='cpu', max_memory='32M', progress='off', log_level='error', overwrite=True)
            original_publish = application.republish_with_metadata
            calls = []
            def growing_header(source_path, destination, metadata, entries):
                calls.append(destination)
                if len(calls) == 2:
                    metadata = {**metadata, 'large-final-metadata': 'x' * 500000}
                return original_publish(source_path, destination, metadata, entries)
            with contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(application, 'republish_with_metadata', side_effect=growing_header), \
                    self.assertRaisesRegex(CompressionGateError, 'whole-file compression gate failed'):
                api.convert(options)
            self.assertEqual(output.read_bytes(), b'previous-output')

    def test_adaptive_reuses_only_matching_selected_encodings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output, calibration = root / 'source.safetensors', root / 'output.safetensors', root / 'calibration.pt'
            name = 'double_blocks.0.img_attn.qkv.weight'
            save_file({name: torch.randn(64, 256).bfloat16()}, str(source))
            torch.save({name: torch.randn(16, 256)}, calibration)
            costs = {(64, 256, fmt, 8, group, 16, None): 1.
                     for fmt, group in ((FORMAT_INT8, None), ('int8-convrot', None), (FORMAT_W4A8, 16), (FORMAT_W4A4, None))}
            workloads = [dict(m=16, n=64, k=256, weight=1., layer=None)]
            options = ConversionOptions(model=str(source), output=str(output), architecture='flux',
                format='mixed', profile='adaptive', calibration_source=str(calibration), kernel_costs='fixture',
                device='cpu', target_runtime='nvidia', quality_gate=.2, global_error_gate=.2,
                max_memory='32M', validate=True, progress='off', log_level='error')
            with patch.object(application, 'load_kernel_costs', return_value=(costs, workloads)), \
                    patch.object(application, 'runtime_capabilities_for', return_value=runtime_capabilities_for('cpu')), \
                    patch('native_quantizer.execution.mixed.quantize_tensor_bounded', side_effect=AssertionError('selected payload encoded twice')), \
                    contextlib.redirect_stdout(io.StringIO()):
                # The hash still identifies an actual input file; the cost loader is isolated above.
                options.kernel_costs = str(source)
                self.assertEqual(api.convert(options), 0)
            self.assertTrue(api.verify_output(str(output))['ok'])

            with safe_open(str(output), framework='pt') as checkpoint:
                record = json.loads(checkpoint.metadata()['comfy_wxa8'])['quantization']['planning']
            self.assertEqual(record['error_metric'], 'heldout-p95-runtime-output-relL2')
            self.assertTrue(record['calibration_digest'])


if __name__ == '__main__':
    unittest.main()
