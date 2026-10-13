"""Native W6A8 storage, planner, publication and ComfyUI interoperability tests."""
import contextlib
import dataclasses
import io
import json
import tempfile
import unittest
from tests_support import isolated_module as setUpModule, load_comfy_cpu
from pathlib import Path
from unittest.mock import patch
import torch
from safetensors import safe_open
from safetensors.torch import save_file
import comfyui_native_quantizer as q
from contextlib import ExitStack
from native_quantizer.codecs import bounded as bounded_codecs, dispatch
from native_quantizer.execution import mixed as mixed_execution
from native_quantizer.execution import state as conversion_state
from native_quantizer.planning import mixed as mixed_planning
from native_quantizer.runtime import kitchen
from kitchen_policy import load_kernel_costs, kernel_contract

WEIGHT = 'double_blocks.0.img_attn.qkv.weight'

@unittest.skipUnless(q.w6a8_api_available(), 'Kitchen >= 0.2.36 six-bit API required')
class W6A8Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.safetensors'
        self.weight = torch.randn(65, 256, generator=torch.Generator().manual_seed(91))
        self.weights = {WEIGHT: self.weight, 'img_in.weight': torch.randn(32, 256),
                        'double_blocks.0.img_mod.lin.weight': torch.randn(32, 256),
                        'double_blocks.0.img_attn.norm.key_norm.weight': torch.ones(32),
                        'zero_buffer': torch.empty(0)}
        save_file(self.weights, str(self.source))

    def tearDown(self):
        self.temp.cleanup()

    def convert(self, fmt='w6a8', *extra, filename='out.safetensors', budget='32M'):
        output = self.root / filename
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            result = q.main([str(self.source), '--output', str(output), '--format', fmt,
                             '--device', 'cpu', '--target-runtime', 'cpu', '--yes',
                             '--max-memory', budget, '--progress', 'off', *extra])
        self.assertEqual(result, 0)
        if fmt in ('w6a8', 'w6a8_int8'):
            self.assertIn('format      : w6a8_int8', captured.getvalue())
        return output

    def load(self, path):
        with safe_open(str(path), framework='pt') as st:
            return {key: st.get_tensor(key) for key in st.keys()}, st.metadata()

    def test_all_code_boundaries_and_plane_order(self):
        codes = torch.arange(64).repeat(3, 4)
        packed = q.pack_w6(codes)
        self.assertEqual(tuple(packed.shape), (3, 192))
        self.assertTrue(torch.equal(q.unpack_w6(packed), codes))
        self.assertEqual(int(packed[0, 0].item()) & 255, 0x10)
        self.assertEqual(int(packed[0, 128 + 4].item()) & 255, 0x55)
        for bad in (torch.zeros(3, 191, dtype=torch.int8), torch.zeros(3, 192)):
            with self.assertRaises(q.PolicyError):
                q.unpack_w6(bad)
        with self.assertRaises(q.PolicyError):
            q.pack_w6(torch.full((3, 256), 64))

    def test_kitchen_decoder_consensus_zero_ordinary_heavy_tail(self):
        cls = q.require_w6a8_api()
        fixtures = (torch.zeros(13, 256), self.weight[:13],
                    self.weight[:13] * torch.exp(self.weight[:13].abs()))
        for group in (16, 32, 64):
            for w in fixtures:
                with self.subTest(group=group, zero=not bool(w.any())):
                    tensors = q.quantize_weight_by_format(w, q.FORMAT_W6A8, group, 256)
                    packed, params = cls.quantize(w.float(), bits=6, group_size=group,
                        convrot_groupsize=256, codebook=False, scale_search=True)
                    reference = q.reference_decode_w6a8(tensors[''], tensors['_s_rel'], tensors['_s_channel'], group)
                    torch.testing.assert_close(reference, cls.dequantize(packed, params).float(), rtol=1e-5, atol=2e-6)
                    self.assertTrue(torch.equal(q.pack_w6(q.unpack_w6(packed)), packed))
                    self.assertEqual(sum(v.numel()*v.element_size() for v in tensors.values()),
                                     q.quantized_format_bytes(len(w), 256, q.FORMAT_W6A8, group))

    def test_standalone_validation_alias_and_protected_tensors(self):
        for group in (16, 32, 64):
            output = self.convert('w6a8_int8', '--w6a8-group-size', str(group), '--validate',
                                  filename=f'g{group}.safetensors')
            self.assertTrue(q.verify_output(str(output))['ok'])
            sd, metadata = self.load(output)
            layers = json.loads(metadata[q.METADATA_KEY_QUANT])['layers']
            self.assertEqual(layers[WEIGHT[:-7]]['group_size'], group)
            self.assertEqual(set(sd), {WEIGHT, WEIGHT+'_s_rel', WEIGHT+'_s_channel',
                                      *set(self.weights)-{WEIGHT}})
            for name in set(self.weights)-{WEIGHT}:
                self.assertTrue(torch.equal(sd[name], self.weights[name]))
            ext = json.loads(metadata[q.METADATA_KEY_EXT])
            self.assertEqual(ext['format_revision'], q.FORMAT_W6A8_REVISION)
            self.assertEqual(ext['quantization']['weight_bits'], 6)
            self.assertNotIn('w4a8_format_revision', ext['algorithm_identity'])
            self.assertEqual(ext['reproducibility']['kitchen_version'], q.kitchen_version())
            with contextlib.redirect_stdout(io.StringIO()):
                result = q.main([str(self.source), '--format', 'w6a8', '--output', str(output),
                                 '--validation-only', '--progress', 'off'])
            self.assertEqual(result, 0)

    def test_mixed_opt_in_and_group_selected(self):
        shared = ['--profile', 'balanced', '--quality-gate', '.05', '--global-error-gate', '.05',
                  '--max-linear-bytes-per-param', '1.1', '--mixed-objective', 'size', '--validate']
        before = self.convert('mixed', *shared, filename='off.safetensors')
        after = self.convert('mixed', *shared, '--enable-w6a8', '--w6a8-group-size', '32',
                             filename='on.safetensors')
        _, old_meta = self.load(before)
        _, new_meta = self.load(after)
        old_layers = json.loads(old_meta[q.METADATA_KEY_QUANT])['layers']
        new_layers = json.loads(new_meta[q.METADATA_KEY_QUANT])['layers']
        self.assertNotIn(q.FORMAT_W6A8, {v['format'] for v in old_layers.values()})
        self.assertEqual(new_layers[WEIGHT[:-7]]['format'], q.FORMAT_W6A8)
        self.assertEqual(new_layers[WEIGHT[:-7]]['group_size'], 32)
        self.assertTrue(q.verify_output(str(after))['ok'])

    def test_runtime_activation_scoring_includes_rotation_and_a8(self):
        acts = torch.randn(8, 256)
        tensors = q.quantize_weight_by_format(self.weight, q.FORMAT_W6A8, 16, 256)
        h = q.build_hadamard(256)
        aq, scale = q._act_quant_int8(q.rotate_activation(acts, h, 256))
        wint = q.decode_w6a8_runtime_weight(tensors[''], tensors['_s_rel'], 16)
        output = (aq.float() @ wint.float().T) * scale * tensors['_s_channel'][None, :]
        ref = acts @ self.weight.T
        expected = float(((output-ref).norm(dim=1)/ref.norm(dim=1)).mean())
        self.assertAlmostEqual(q.runtime_output_rel_l2(self.weight, tensors, q.FORMAT_W6A8, 16, 256, acts), expected)
        info = q.discover_checkpoint(str(self.source))
        decision = q.TensorDecision(WEIGHT, q.DecisionKind.QUANTIZE, '', layer=WEIGHT[:-7], format=q.FORMAT_W6A8)
        calibration = q.CalibrationStats('fixture', [], 8, {WEIGHT: {'samples': acts, 'rows': 8}})
        analyzer = q.SensitivityAnalyzer(.001, .5, calibration)
        q.apply_sensitivity_prepass(info, [decision], analyzer, 32*1024**2, torch.device('cpu'), torch.float32)
        self.assertEqual(decision.kind, q.DecisionKind.KEEP_PRECISION)
        self.assertAlmostEqual(analyzer.results[WEIGHT].act_rel_l2, expected, places=6)

    def test_bounded_scoring_and_chunked_publication(self):
        info = q.discover_checkpoint(str(self.source))
        d = q.TensorDecision(WEIGHT, q.DecisionKind.QUANTIZE, '', layer=WEIGHT[:-7], format=q.FORMAT_W6A8)
        planner = q.MixedPlanner('balanced', None, 32*1024**2, torch.device('cpu'), torch.float32,
                                 enable_w6a8=True, w6a8_group_size=32)
        with q.CheckpointReader(info) as reader:
            full = planner._evaluate_format(reader, d, q.FORMAT_W6A8)
            original_budget = q._chunk_rows_for_budget
            with ExitStack() as patches:
                for module in (bounded_codecs, mixed_execution, mixed_planning):
                    patches.enter_context(patch.object(module, '_quant_work_bytes', return_value=64*1024**2))
                    patches.enter_context(patch.object(module, '_chunk_rows_for_budget', return_value=7))
                bounded = planner._evaluate_format(reader, d, q.FORMAT_W6A8)
                output = self.convert('w6a8', '--w6a8-group-size', '32', '--validate')
            self.assertAlmostEqual(full.rel_l2, bounded.rel_l2, places=6)
            self.assertTrue(q.verify_output(str(output))['ok'])
            self.assertLess(original_budget(256, 10000, 32*1024**2, q.FORMAT_W6A8),
                            original_budget(256, 10000, 32*1024**2, q.FORMAT_W4A8))

    def test_real_memory_budget_streams_and_validates(self):
        weight = torch.randn(1025, 256, generator=torch.Generator().manual_seed(54))
        save_file({WEIGHT: weight, 'img_in.weight': self.weights['img_in.weight'],
                   'double_blocks.0.img_attn.norm.key_norm.weight': torch.ones(32)}, str(self.source))
        output = self.convert('w6a8', '--validate', budget='32M')
        sd, metadata = self.load(output)
        self.assertEqual(tuple(sd[WEIGHT].shape), (1025,192))
        extension = json.loads(metadata[q.METADATA_KEY_EXT])
        self.assertIn(WEIGHT, extension['quantization']['chunked_layers'])
        self.assertTrue(q.verify_output(str(output))['ok'])
        with contextlib.redirect_stdout(io.StringIO()):
            status = q.main([str(self.source), '--format', 'w6a8', '--output', str(output),
                             '--max-memory', '32M', '--validation-only', '--progress', 'off'])
        self.assertEqual(status, 0)

    def test_mixed_promotes_to_w6a8_for_global_gate(self):
        output = self.convert('mixed', '--enable-w6a8', '--profile', 'balanced',
            '--quality-gate', '.2', '--global-error-gate', '.025',
            '--max-linear-bytes-per-param', '1.1', '--mixed-objective', 'size', '--validate')
        _, metadata = self.load(output)
        layers = json.loads(metadata[q.METADATA_KEY_QUANT])['layers']
        self.assertEqual(layers[WEIGHT[:-7]]['format'], q.FORMAT_W6A8)
        extension = json.loads(metadata[q.METADATA_KEY_EXT])
        self.assertTrue(extension['quantization']['distribution']['promotions'])

    def test_repeated_payload_and_budget_numerics(self):
        first = self.convert(filename='one.safetensors')
        second = self.convert(filename='two.safetensors')
        self.assertEqual(q.sha256_safetensors_payload(str(first)), q.sha256_safetensors_payload(str(second)))
        full = q.quantize_weight_by_format(self.weight, q.FORMAT_W6A8, 16, 256)
        parts = [q.quantize_weight_by_format(w, q.FORMAT_W6A8, 16, 256) for w in self.weight.split(7)]
        joined = {key: torch.cat([part[key] for part in parts]) for key in full}
        torch.testing.assert_close(q.dequantize_weight_by_format(full, q.FORMAT_W6A8, 16, 256, torch.float32),
                                   q.dequantize_weight_by_format(joined, q.FORMAT_W6A8, 16, 256, torch.float32), rtol=1e-5, atol=1e-6)

    def test_malformed_outputs_fail_source_free_and_validation(self):
        output = self.convert()
        sd, meta = self.load(output)
        variants = []
        for suffix in ('_codebook', '_correction'):
            variants.append(({**sd, WEIGHT+suffix: torch.ones(1)}, meta))
        variants.append(({key:value for key,value in sd.items() if key != WEIGHT+'_s_rel'}, meta))
        variants.append(({**sd, WEIGHT: sd[WEIGHT][:, :-1].contiguous()}, meta))
        changed = sd[WEIGHT].clone(); changed[0,0] ^= 1
        variants.append(({**sd, WEIGHT: changed}, meta))
        variants.append(({**sd, WEIGHT+'_s_channel': sd[WEIGHT+'_s_channel'].half()}, meta))
        invalid = sd[WEIGHT+'_s_rel'].float(); invalid[0,0] = float('nan')
        variants.append(({**sd, WEIGHT+'_s_rel': invalid.to(torch.float8_e4m3fn)}, meta))
        for field, value in [('group_size', 0), ('group_size', 48), ('convrot_groupsize', 64), ('convrot', False), ('group_size', True)]:
            qm = json.loads(meta[q.METADATA_KEY_QUANT]); qm['layers'][WEIGHT[:-7]][field] = value
            variants.append((sd, {**meta, q.METADATA_KEY_QUANT: json.dumps(qm)}))
        for field, value in [('weight_bits',4), ('packing','int4-nibble-lsb'), ('group_size',64)]:
            extension = json.loads(meta[q.METADATA_KEY_EXT]); extension['quantization'][field] = value
            variants.append((sd, {**meta, q.METADATA_KEY_EXT: json.dumps(extension)}))
        for index, (bad_sd, bad_meta) in enumerate(variants):
            with self.subTest(variant=index):
                bad = self.root / f'bad{index}.safetensors'
                save_file(bad_sd, str(bad), metadata=bad_meta)
                self.assertFalse(q.verify_output(str(bad))['ok'])
                with contextlib.redirect_stdout(io.StringIO()):
                    try:
                        result = q.main([str(self.source), '--format', 'w6a8', '--output', str(bad), '--validation-only'])
                    except q.ValidationError:
                        result = 2
                self.assertNotEqual(result, 0)

    def test_dependency_gate_never_accepts_old_kwargs_api(self):
        with patch.object(kitchen, 'w6a8_api_available', return_value=False):
            with self.assertRaises(q.RuntimeCompatibilityError):
                q.quantize_weight_by_format(self.weight, q.FORMAT_W6A8, 16, 256)
            self.assertEqual(q.quantize_weight_by_format(self.weight, q.FORMAT_INT8, 16, 256)[''].shape, self.weight.shape)
        class Older:
            @staticmethod
            def quantize(tensor, **kwargs):
                return None
        with patch('comfy_kitchen.tensor.w4a8_int8.AsymW4A8Int8Layout', Older):
            with self.assertRaises(q.RuntimeCompatibilityError):
                q.require_w6a8_api()
        cls = q.require_w6a8_api()
        old_packed, old_params = cls.quantize(self.weight, bits=4, codebook=False)
        with patch.object(dispatch, 'require_w6a8_api', return_value=cls), patch.object(cls, 'quantize', return_value=(old_packed, old_params)):
            with self.assertRaises(q.PolicyError):
                q.quantize_weight_by_format(self.weight, q.FORMAT_W6A8, 16, 256)

    def test_shape_and_flag_gates(self):
        for group in (-16, 0, 1, 17):
            with self.assertRaises(q.UsageError):
                q.main([str(self.source), '--format', 'w6a8', '--w6a8-group-size', str(group)])
        with self.assertRaises(q.UsageError):
            q.main([str(self.source), '--format', 'int8', '--enable-w6a8'])
        info = q.discover_checkpoint(str(self.source)); det = q.detect_architecture(info)
        kept = q.classify_tensors(info, det, q.FORMAT_W6A8, 48, [], [], [], None, None)
        self.assertFalse(any(d.kind == q.DecisionKind.QUANTIZE for d in kept))
        with self.assertRaises(q.CalibrationError):
            self.convert('w6a8', '--require-calibration')

    def test_resume_and_dependency_identity(self):
        info = q.discover_checkpoint(str(self.source)); det = q.detect_architecture(info)
        dec = q.classify_tensors(info, det, q.FORMAT_W6A8, 32, [], [], [], None, None)
        entries, total = q.build_output_entries(info, dec, q.FORMAT_W6A8, None)
        plan = q.ConversionPlan(q.FORMAT_W6A8, det, dec, {}, {}, entries,
                                total_in_bytes=info.total_bytes, total_out_bytes=total,
                                n_quantized=1, n_kept=len(dec)-1)
        output = str(self.root/'resume.safetensors')
        args = q._selftest_args(output, q.FORMAT_W6A8, extra={'w6a8_group_size':32, 'max_memory':32*1024**2})
        state, temp = output+'.state.json', output+'.tmp'
        engine = q.ConversionEngine(info, plan, args, state, temp, output)
        engine._crash_after = 1
        try:
            with self.assertRaises(q._SimulatedCrash):
                engine.run()
            engine.save_state()
        finally:
            engine.close()
        identity = q._plan_hash(info, plan, args, {})
        with patch.object(conversion_state, 'kitchen_version', return_value='0.2.99'):
            self.assertNotEqual(identity, q._plan_hash(info, plan, args, {}))
        args.resume=True
        for changed in (dataclasses.replace(plan),):
            drift = q._selftest_args(output, q.FORMAT_W6A8, resume=True,
                extra={'w6a8_group_size':64, 'max_memory':32*1024**2})
            rejected = q.ConversionEngine(info, changed, drift, state, temp, output)
            try:
                with self.assertRaises(q.OutputError):
                    rejected.run()
            finally:
                rejected.close()
        engine = q.ConversionEngine(info, plan, args, state, temp, output)
        try:
            engine.run()
        finally:
            engine.close()
        sd, _ = self.load(output)
        self.assertEqual(tuple(sd[WEIGHT].shape), (65, 192))
        self.assertTrue(torch.equal(sd['img_in.weight'], self.weights['img_in.weight']))

    def test_failed_publication_preserves_destination(self):
        destination = self.root/'out.safetensors'; destination.write_bytes(b'existing destination')
        original = destination.read_bytes()
        with patch.object(q.Validator, 'run', return_value={'n_failed':1, 'checks':[], 'output_sha256':''}):
            with contextlib.redirect_stdout(io.StringIO()):
                result = q.main([str(self.source), '--output', str(destination), '--format', 'w6a8',
                                 '--overwrite', '--validate', '--yes', '--progress', 'off'])
            self.assertEqual(result, 2)
        self.assertEqual(destination.read_bytes(), original)

    def test_group_aware_costs_and_distinct_runtime_certificate(self):
        cost = self.root/'cost.json'
        base = {'kitchen_version':q.kitchen_version(), 'benchmarks':[{'shape':{'m':1,'n':65,'k':256},
                'results':[{'format':q.FORMAT_W6A8,'activation_bits':8,'group_size':32,'warm_ms':.1}]}]}
        cost.write_text(json.dumps({**base,'schema':'kitchen-costs/v2'}))
        costs, _ = load_kernel_costs(cost, q.kitchen_version())
        planner = q.MixedPlanner('balanced', None, 32*1024**2, torch.device('cpu'), torch.float32,
                                 enable_w6a8=True, w6a8_group_size=32, kernel_costs=costs)
        self.assertEqual(planner.measured_cost((65,256),q.FORMAT_W6A8,8), .1)
        planner.w6a8_group_size=16
        self.assertIsNone(planner.measured_cost((65,256),q.FORMAT_W6A8,8))
        cost.write_text(json.dumps({**base,'schema':'kitchen-costs/v1'}))
        with self.assertRaises(ValueError):
            load_kernel_costs(cost,q.kitchen_version())
        self.assertEqual(kernel_contract(q.FORMAT_W6A8,backend='nvidia',accelerated=True).path,'int6-via-int8')
        runtime=q.runtime_capabilities_for('cpu')
        certificate=q.RuntimeCertificate('cpu', None, None, None, {q.FORMAT_W4A8:{'load':True,'forward':True}})
        with self.assertRaises(q.RuntimeCompatibilityError):
            q._check_runtime_certificate(certificate,runtime,[q.FORMAT_W6A8])

    def test_comfyui_cpu_load_save_and_device_movement(self):
        ops, _ = load_comfy_cpu()
        output=self.convert()
        sd, metadata=self.load(output)
        conf=json.loads(metadata[q.METADATA_KEY_QUANT])['layers'][WEIGHT[:-7]]
        module=ops.mixed_precision_ops(compute_dtype=torch.float32).Linear(256,65,bias=False,device='cpu')
        state={'weight'+suffix:sd[WEIGHT+suffix] for suffix in ('','_s_rel','_s_channel')}
        state['comfy_quant']=torch.tensor(list(json.dumps(conf).encode()),dtype=torch.uint8)
        module.load_state_dict(state,strict=True)
        x=torch.randn(2,256)
        with torch.inference_mode():
            actual=module(x)
        self.assertTrue(torch.isfinite(actual).all())
        saved=module.state_dict()
        clone=ops.mixed_precision_ops(compute_dtype=torch.float32).Linear(256,65,bias=False,device='cpu')
        clone.load_state_dict(saved,strict=True)
        clone.to('cpu')
        with torch.inference_mode():
            torch.testing.assert_close(actual,clone(x),atol=0,rtol=0)

if __name__ == '__main__':
    unittest.main()
