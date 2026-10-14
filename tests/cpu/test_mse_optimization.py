"""Bounded native MSE regressions; synthetic fixtures, no downloaded models."""
import contextlib
import dataclasses
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tests.support import isolated_module as setUpModule
from native_quantizer import api
from native_quantizer.checkpoint_io.reader import CheckpointReader, discover_checkpoint
from native_quantizer.codecs import bounded, dispatch
from native_quantizer.codecs.dispatch import quantize_weight_by_format, dequantize_weight_by_format
from native_quantizer.codecs.native import _native_encode
from native_quantizer.codecs.optimization import ErrorObjective
from native_quantizer.errors import CalibrationError, PolicyError, UsageError, OutputError
from native_quantizer.execution.mixed import ConversionEngine, _SimulatedCrash
from native_quantizer.formats import NATIVE_FORMATS
from native_quantizer.identity import FORMAT_INT8, FORMAT_W4A4, FORMAT_W4A8, FORMAT_W6A8
from native_quantizer.optimization import OptimizationConfig
from native_quantizer.options import ConversionOptions, validate_conversion_options
from native_quantizer.runtime.kitchen import _native_kitchen, w6a8_api_available
from native_quantizer.validation.reference import reference_decode_w4a8, reference_decode_w4a4, reference_decode_w6a8

WEIGHT = 'double_blocks.0.img_attn.qkv.weight'
FORMATS = (FORMAT_INT8, FORMAT_W4A4, FORMAT_W4A8, FORMAT_W6A8)


def same_bytes(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))


class MSETests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        g = torch.Generator().manual_seed(123)
        self.w = torch.randn(32, 256, generator=g)
        self.x = torch.randn(19, 256, generator=g) * torch.linspace(.05, 3, 256)
        self.source = self.root / 'source.safetensors'
        self.kept = {'img_in.weight': torch.randn(32, 256, generator=g),
                     'double_blocks.0.img_mod.lin.weight': torch.randn(32, 256, generator=g),
                     'counter': torch.tensor(3)}
        save_file({WEIGHT: self.w, **self.kept}, str(self.source))
        self.cfg = OptimizationConfig('mse', 'weight')

    def tearDown(self):
        self.temp.cleanup()

    def encode(self, w, fmt, cfg=None, x=None):
        return quantize_weight_by_format(w, fmt, 16, 256, optimization=cfg, activations=x, codebook_mode='fixed')

    def decode(self, payload, fmt):
        return dequantize_weight_by_format(payload, fmt, 16, 256, torch.float32)

    def convert(self, fmt, **kw):
        out = self.root / f'{fmt}-{len(list(self.root.iterdir()))}.safetensors'
        options = ConversionOptions(model=str(self.source), output=str(out), architecture='flux',
                                    format=fmt, device='cpu', target_runtime='cpu', optimizer='mse',
                                    mse_steps=9, max_memory='32M', validate=True, progress='off', **kw)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(api.convert(options), 0)
        self.assertTrue(api.verify_output(str(out))['ok'])
        with safe_open(str(out), framework='pt') as f:
            sd, metadata = {key: f.get_tensor(key) for key in f.keys()}, f.metadata()
        for name, value in self.kept.items():
            self.assertTrue(same_bytes(sd[name], value))
        return out, sd, metadata

    def test_optional_mode_preserves_baseline_payloads(self):
        for fmt in FORMATS:
            if fmt == FORMAT_W6A8 and not w6a8_api_available():
                continue
            with self.subTest(fmt=fmt):
                a, b = self.encode(self.w, fmt), self.encode(self.w, fmt, OptimizationConfig())
                self.assertTrue(all(same_bytes(a[k], b[k]) for k in a))

    def test_weight_objective_never_regresses_distributions_or_zero_rows(self):
        fixtures = [self.w, self.w * torch.exp(self.w.abs()), torch.zeros_like(self.w),
                    torch.where(self.w > 1, self.w, 0), self.w * 1e-8]
        for fmt in FORMATS:
            if fmt == FORMAT_W6A8 and not w6a8_api_available():
                continue
            for w in fixtures:
                with self.subTest(fmt=fmt, maximum=float(w.abs().max())):
                    a, b = self.decode(self.encode(w, fmt), fmt), self.decode(self.encode(w, fmt, self.cfg), fmt)
                    self.assertTrue(torch.isfinite(b).all())
                    baseline = (w.double() - a.double()).square().sum(-1)
                    candidate = (w.double() - b.double()).square().sum(-1)
                    torch.testing.assert_close(torch.minimum(candidate, baseline), candidate, rtol=2e-4, atol=1e-16)

    def test_activation_objective_uses_correlated_samples_and_auto(self):
        x = self.x.clone(); x[:, 1:10] = x[:, :1]
        cfg = OptimizationConfig('mse', 'activation')
        for fmt in FORMATS:
            if fmt == FORMAT_W6A8 and not w6a8_api_available():
                continue
            with self.subTest(fmt=fmt):
                a = self.decode(self.encode(self.w, fmt), fmt)
                b = self.decode(self.encode(self.w, fmt, cfg, x), fmt)
                ea = ((self.w - a) @ x.T).double().square().sum(-1)
                eb = ((self.w - b) @ x.T).double().square().sum(-1)
                self.assertTrue(bool((eb <= ea * 1.0002 + 1e-10).all()))
                auto = self.encode(self.w, fmt, OptimizationConfig('mse'), x)
                explicit = self.encode(self.w, fmt, cfg, x)
                self.assertTrue(all(same_bytes(auto[k], explicit[k]) for k in auto))
        with self.assertRaises(CalibrationError):
            self.encode(self.w, FORMAT_INT8, cfg)

    def test_row_chunk_determinism_and_source_immutability(self):
        original = self.w.clone()
        for fmt in FORMATS:
            if fmt == FORMAT_W6A8 and not w6a8_api_available():
                continue
            with self.subTest(fmt=fmt):
                full = self.encode(self.w, fmt, self.cfg)
                pieces = [self.encode(w, fmt, self.cfg) for w in self.w.split(5)]
                if fmt == FORMAT_W4A8:
                    # auto codebook is fixed on this Gaussian fixture.
                    joined = {key: full[key] if key == '_codebook' else torch.cat([p[key] for p in pieces]) for key in full}
                else:
                    joined = {key: torch.cat([p[key] for p in pieces]) for key in full}
                torch.testing.assert_close(self.decode(full, fmt), self.decode(joined, fmt), rtol=2e-5, atol=2e-6)
                repeat = self.encode(self.w, fmt, self.cfg)
                self.assertTrue(all(same_bytes(full[k], repeat[k]) for k in full))
        self.assertTrue(torch.equal(self.w, original))

    def test_grouped_native_reference_decode_consensus(self):
        for fmt, reference in ((FORMAT_W4A8, reference_decode_w4a8), (FORMAT_W6A8, reference_decode_w6a8)):
            if fmt == FORMAT_W6A8 and not w6a8_api_available():
                continue
            p = self.encode(self.w, fmt, self.cfg)
            args = (p[''], p['_s_rel'], p['_s_channel'])
            if fmt == FORMAT_W4A8:
                args += (p['_codebook'],)
            expected = reference(*args, 16, 256)
            torch.testing.assert_close(self.decode(p, fmt), expected, rtol=1e-5, atol=2e-6)
        p = self.encode(self.w, FORMAT_W4A4, self.cfg)
        torch.testing.assert_close(self.decode(p, FORMAT_W4A4), reference_decode_w4a4(p[''], p['_scale'], 256))

    def test_native_block_formats_exact_kitchen_decode_and_objectives(self):
        for fmt in ('nvfp4', 'mxfp8'):
            spec, cfg = NATIVE_FORMATS[fmt], OptimizationConfig('mse')
            kitchen = _native_kitchen(spec)
            for w in (self.w, torch.zeros_like(self.w), self.w * 1e-8):
                for x in (None, self.x):
                    with self.subTest(fmt=fmt, zero=not bool(w.any()), calibrated=x is not None):
                        scale = (w.abs().max() / 2688).clamp(min=1e-30) if fmt == 'nvfp4' else None
                        _, base = _native_encode(w, spec, scale, 0, kitchen)
                        p, result = _native_encode(w, spec, scale, 0, kitchen, optimization=cfg, activations=x)
                        params = (kitchen.Params(scale=p['_scale_2'], block_scale=p['_scale'], orig_dtype=torch.float32, orig_shape=tuple(w.shape))
                                  if fmt == 'nvfp4' else kitchen.Params(scale=p['_scale'].view(torch.float8_e8m0fnu), orig_dtype=torch.float32, orig_shape=tuple(w.shape)))
                        decoded = kitchen.dequantize(p[''], params)
                        torch.testing.assert_close(result, decoded, rtol=2e-6, atol=1e-12)
                        objective = ErrorObjective(w, cfg, x)
                        self.assertLessEqual(float(objective.score(decoded).sum()), float(objective.score(base).sum()) * 1.00001 + 1e-16)

    def test_all_native_formats_end_to_end_and_scalar_fp8_scale_streaming(self):
        for fmt in ('int8', 'int8-convrot', 'int4-convrot', 'w4a4', 'fp8-e4m3', 'fp8-e5m2', 'nvfp4', 'mxfp8'):
            with self.subTest(fmt=fmt):
                _, sd, metadata = self.convert(fmt)
                provenance = json.loads(metadata['comfy_quantizer'])
                self.assertEqual(provenance['optimization']['mode'], 'mse')
                if fmt.startswith('fp8'):
                    self.assertEqual(sd[WEIGHT + '_scale'].numel(), 1)
                    with patch('native_quantizer.execution.native._native_rows', wraps=bounded._native_rows) as rows:
                        self.convert(fmt)
                        self.assertGreater(rows.call_count, 1)

    def test_streamed_fp8_matches_full_budget(self):
        for fmt in ('fp8-e4m3', 'fp8-e5m2'):
            _, first, _ = self.convert(fmt)
            # 256 bytes/element makes this one working row at a time.
            with patch('native_quantizer.execution.native._native_rows', side_effect=lambda reader, meta, budget, device='cpu', optimization=None:
                       bounded._native_rows(reader, meta, min(budget, meta.shape[1] * 256), device, optimization)):
                _, second, _ = self.convert(fmt)
            self.assertTrue(all(same_bytes(first[k], second[k]) for k in first))

    def test_planned_formats_and_mixed_use_selected_optimizer(self):
        for fmt in ('w4a8', 'w6a8', 'mixed'):
            if fmt == 'w6a8' and not w6a8_api_available():
                continue
            with self.subTest(fmt=fmt):
                output, _, metadata = self.convert(fmt, error_threshold=.25,
                    **({'profile': 'size-first', 'global_error_gate': .25} if fmt == 'mixed' else {}))
                extension = json.loads(metadata['comfy_wxa8'])
                self.assertEqual(extension['optimization']['mode'], 'mse')
                captured = io.StringIO()
                with contextlib.redirect_stdout(captured):
                    self.assertEqual(api.convert(ConversionOptions(model=str(self.source), output=str(output),
                        architecture='flux', format=fmt, validation_only=True, device='cpu', max_memory='32M')), 0)
                self.assertIn('optimizer: MSE', captured.getvalue())

    def test_bounded_fitted_codebook_and_writer_match_full(self):
        for mode in ('fixed', 'auto', 'fit'):
            for fmt in ('w4a8', 'w6a8'):
                if fmt == 'w6a8' and not w6a8_api_available():
                    continue
                with self.subTest(mode=mode, fmt=fmt):
                    _, first, _ = self.convert(fmt, codebook_mode=mode)
                    with contextlib.ExitStack() as stack:
                        for module in ('native_quantizer.codecs.bounded', 'native_quantizer.execution.mixed',
                                       'native_quantizer.planning.mixed', 'native_quantizer.planning.sensitivity'):
                            stack.enter_context(patch(module + '._quant_work_bytes', return_value=64 << 20))
                            stack.enter_context(patch(module + '._chunk_rows_for_budget', return_value=5))
                        _, second, _ = self.convert(fmt, codebook_mode=mode)
                    for key in first:
                        torch.testing.assert_close(first[key].float(), second[key].float(), rtol=2e-5, atol=2e-6)

    def test_calibration_all_engines_and_explicit_missing_coverage(self):
        path = self.root / 'calibration.npz'
        np.savez(path, **{WEIGHT: self.x.numpy()})
        for fmt in ('int8-convrot', 'fp8-e4m3', 'nvfp4', 'mxfp8', 'w4a8', 'w6a8', 'mixed'):
            if fmt == 'w6a8' and not w6a8_api_available():
                continue
            with self.subTest(fmt=fmt):
                self.convert(fmt, calibration_source=str(path), mse_objective='activation', error_threshold=.3,
                             **({'global_error_gate': .3} if fmt == 'mixed' else {}))
                with self.assertRaises(CalibrationError):
                    self.convert(fmt, mse_objective='activation')

    def test_resume_rejects_changed_optimizer_and_calibration(self):
        calibration = self.root / 'calibration.npz'
        np.savez(calibration, **{WEIGHT: self.x.numpy()})
        out = self.root / 'resume.safetensors'
        options = ConversionOptions(model=str(self.source), output=str(out), architecture='flux',
            format='w4a8', optimizer='mse', mse_steps=5, device='cpu', target_runtime='cpu',
            max_memory='32M', progress='off', calibration_source=str(calibration))
        original = ConversionEngine._record_entry
        def crash(engine, *args):
            original(engine, *args)
            engine.save_state()
            raise _SimulatedCrash('partial layer')
        with contextlib.redirect_stdout(io.StringIO()):
            with patch.object(ConversionEngine, '_record_entry', crash), self.assertRaises(_SimulatedCrash):
                api.convert(options)
            for change in ({'mse_steps': 7}, {'mse_objective': 'weight'}, {'optimizer': 'none'}, {'mse_iterations': 1}):
                with self.subTest(change=change), self.assertRaises(OutputError):
                    api.convert(dataclasses.replace(options, resume=True, **change))
            np.savez(calibration, **{WEIGHT: (self.x * 2).numpy()})
            with self.assertRaises(OutputError):
                api.convert(dataclasses.replace(options, resume=True))
            np.savez(calibration, **{WEIGHT: self.x.numpy()})
            self.assertEqual(api.convert(dataclasses.replace(options, resume=True)), 0)
        self.assertTrue(api.verify_output(str(out))['ok'])

    def test_option_config_validation_and_precision_cast_rejection(self):
        for change in ({'mse_steps': 1}, {'mse_steps': True}, {'mse_iterations': -1}, {'optimizer': 'bad'}, {'mse_objective': 'bad'}):
            with self.subTest(change=change), self.assertRaises(UsageError):
                validate_conversion_options(ConversionOptions(**change))
        for fmt in ('fp16', 'bf16'):
            with self.assertRaises(UsageError):
                self.convert(fmt)
        from native_quantizer.config import load_defaults
        config = self.root / 'config.json'
        config.write_text(json.dumps(dict(schema=1, optimizer='mse', mse_steps=5, mse_iterations=1)))
        self.assertEqual(load_defaults(config)['mse_steps'], 5)

    def test_cli_api_payload_parity_and_xdg_override(self):
        from native_quantizer.files import sha256_safetensors_payload
        from native_quantizer.cli.arguments import build_arg_parser
        from native_quantizer.config import apply_cli_defaults
        config = self.root / 'defaults.json'
        config.write_text(json.dumps(dict(schema=1, optimizer='mse', mse_steps=5, mse_iterations=0)))
        argv = ['--config', str(config), '--optimizer', 'none', '--mse-steps', '11']
        parser = build_arg_parser()
        args = parser.parse_args(argv)
        apply_cli_defaults(parser, args, argv)
        self.assertEqual((args.optimizer, args.mse_steps, args.mse_iterations), ('none', 11, 0))
        for fmt in ('int8-convrot', 'w4a8'):
            first, _, _ = self.convert(fmt)
            second = self.root / f'cli-{fmt}.safetensors'
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(api.run_cli([str(self.source), '--no-config', '--output', str(second),
                    '--architecture', 'flux', '--format', fmt, '--device', 'cpu', '--target-runtime', 'cpu',
                    '--optimizer', 'mse', '--mse-steps', '9', '--max-memory', '32M', '--validate', '--progress', 'off']), 0)
            self.assertEqual(sha256_safetensors_payload(str(first)), sha256_safetensors_payload(str(second)))

    def test_optimizer_provenance_corruption_fails_source_free_verification(self):
        for fmt, key in (('int8', 'comfy_quantizer'), ('w4a8', 'comfy_wxa8')):
            _, sd, metadata = self.convert(fmt)
            provenance = json.loads(metadata[key]); provenance['optimization']['revision'] = 'invalid'
            metadata[key] = json.dumps(provenance)
            corrupted = self.root / f'bad-{fmt}.safetensors'
            save_file(sd, str(corrupted), metadata=metadata)
            result = api.verify_output(str(corrupted))
            self.assertFalse(result['ok'])
            self.assertTrue(any('optimizer revision' in c.get('detail', '') for c in result['checks']))

    def test_saved_mse_files_load_in_native_comfyui_consumer(self):
        from tests.support import load_comfy_cpu
        from native_quantizer.metadata import read_quant_metadata
        ops, _ = load_comfy_cpu()
        import comfy_kitchen as kitchen
        for fmt in ('int8', 'int8-convrot', 'w4a4', 'fp8-e4m3', 'fp8-e5m2', 'nvfp4', 'mxfp8', 'w4a8', 'w6a8'):
            if fmt == 'w6a8' and not w6a8_api_available():
                continue
            with self.subTest(fmt=fmt):
                output, sd, _ = self.convert(fmt)
                conf = read_quant_metadata(str(output))['layers'][WEIGHT[:-7]]
                state = {'weight' + name.removeprefix(WEIGHT): value for name, value in sd.items() if name.startswith(WEIGHT)}
                state['comfy_quant'] = torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8)
                module = ops.mixed_precision_ops(compute_dtype=torch.bfloat16).Linear(256, 32, bias=False, device='cpu')
                module.load_state_dict(state, strict=True)
                self.assertEqual(module.quant_format, conf['format'])
                with torch.inference_mode(), kitchen.use_backend('eager'):
                    x = self.x.bfloat16()
                    actual = module(x)
                    from comfy_kitchen.tensor import QuantizedTensor
                    qinput = (QuantizedTensor.from_float(x, module.layout_type, scale=getattr(module, 'input_scale', None))
                              if ops.QUANT_ALGOS[conf['format']].get('quantize_input', True) else x)
                    direct = torch.nn.functional.linear(qinput, module.weight)
                    torch.testing.assert_close(actual, direct, rtol=0, atol=0)
                    self.assertTrue(torch.isfinite(actual).all())
                    self.assertEqual(tuple(actual.shape), (len(x), 32))

    def test_block_optimizer_budget_fails_before_whole_layer_allocation(self):
        for fmt in ('nvfp4', 'mxfp8'):
            with self.subTest(fmt=fmt), self.assertRaisesRegex(UsageError, 'whole-layer'):
                options = ConversionOptions(model=str(self.source), output=str(self.root / f'{fmt}.safetensors'),
                    format=fmt, architecture='flux', optimizer='mse', device='cpu', max_memory='1700K')
                with contextlib.redirect_stdout(io.StringIO()):
                    api.convert(options)

    def test_nonfinite_and_shape_safeguards_stay_enforced(self):
        for fmt in FORMATS:
            if fmt == FORMAT_W6A8 and not w6a8_api_available():
                continue
            broken = self.w.clone(); broken[0, 0] = float('nan')
            with self.subTest(fmt=fmt), self.assertRaises(PolicyError):
                self.encode(broken, fmt, self.cfg)
        with self.assertRaises(PolicyError):
            self.encode(self.w[:, :255], FORMAT_W4A4, self.cfg)
        from native_quantizer.architectures.policies import get_family
        from native_quantizer.architectures.detection import DetectionResult
        from native_quantizer.planning.classification import classify_tensors
        from native_quantizer.planning.types import DecisionKind
        iris = self.root / 'iris.safetensors'
        save_file({'blocks.0.mlp.w1.weight': torch.zeros(32, 6826)}, str(iris))
        info = discover_checkpoint(str(iris))
        # The architecture safeguard applies before any optimizer gets a tensor.
        detection = DetectionResult(architecture='iris', confidence='high', policy=get_family('iris'), unet_prefix='')
        decisions = classify_tensors(info, detection, 'mixed', None, ['.*'], [], [], None, 1)
        self.assertTrue(all(d.kind != DecisionKind.QUANTIZE for d in decisions))


if __name__ == '__main__':
    unittest.main()
