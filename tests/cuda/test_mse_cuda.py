"""Native CUDA regression for optimized payloads (no model weights/services)."""
import unittest
from unittest.mock import patch
import torch
from tests.support import isolated_module as setUpModule, comfyui_root


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class MSECudaTests(unittest.TestCase):
    def test_low_budget_streams_optimized_cuda_payloads_and_calibration(self):
        import contextlib
        import io
        import json
        import tempfile
        from pathlib import Path
        import numpy as np
        from safetensors import safe_open
        from safetensors.torch import save_file
        from native_quantizer import api
        from native_quantizer.options import ConversionOptions
        name = 'double_blocks.0.img_attn.qkv.weight'
        g = torch.Generator().manual_seed(72)
        with tempfile.TemporaryDirectory(prefix='mse-cuda-stream-') as directory:
            root = Path(directory)
            source, calibration = root / 'source.safetensors', root / 'calibration.npz'
            save_file({name: torch.randn(160, 1024, generator=g)}, str(source))
            np.savez(calibration, **{name: torch.randn(8, 1024, generator=g).numpy()})
            for fmt in ('w4a8', 'w6a8', 'mixed'):
                with self.subTest(format=fmt):
                    output = root / (fmt + '.safetensors')
                    with contextlib.redirect_stdout(io.StringIO()):
                        status = api.convert(ConversionOptions(model=str(source), output=str(output),
                            architecture='flux', format=fmt, optimizer='mse', mse_objective='activation',
                            calibration_source=str(calibration), device='cuda', target_runtime='nvidia',
                            max_memory='32M', validate=True, progress='off', mse_steps=5,
                            global_error_gate=.3 if fmt == 'mixed' else None,
                            error_threshold=.3, profile='size-first' if fmt == 'mixed' else 'auto'))
                    self.assertEqual(status, 0)
                    self.assertTrue(api.verify_output(str(output))['ok'])
                    with safe_open(str(output), framework='pt') as st:
                        extension = json.loads(st.metadata()['comfy_wxa8'])
                    self.assertIn(name, extension['quantization']['chunked_layers'])

    def test_fp32_block_encoding_matches_cpu_without_half_precision_rounding(self):
        from native_quantizer.codecs.native import _native_encode
        from native_quantizer.formats import NATIVE_FORMATS
        from native_quantizer.optimization import OptimizationConfig
        from native_quantizer.runtime.kitchen import _native_kitchen
        w = torch.randn(64, 256, generator=torch.Generator().manual_seed(71))
        for fmt in ('nvfp4', 'mxfp8'):
            spec, kitchen = NATIVE_FORMATS[fmt], _native_kitchen(NATIVE_FORMATS[fmt])
            scale = (w.abs().max() / 2688).clamp(min=1e-30) if fmt == 'nvfp4' else None
            for mode in ('none', 'mse'):
                with self.subTest(fmt=fmt, optimizer=mode), torch.inference_mode():
                    cfg = OptimizationConfig(mode, 'weight')
                    a, da = _native_encode(w, spec, scale, 0, kitchen, optimization=cfg)
                    b, db = _native_encode(w.cuda(), spec, scale.cuda() if scale is not None else None,
                                           0, kitchen, optimization=cfg)
                    torch.testing.assert_close(da, db.cpu(), rtol=2e-5, atol=2e-6)
                    self.assertEqual(a[''].shape, b[''].shape)
                    self.assertEqual(a[''].dtype, b[''].dtype)

    def test_native_integer_forward_graph_and_no_float_weight_fallback(self):
        from native_quantizer.benchmark import run
        from native_quantizer.optimization import OptimizationConfig
        from comfy_kitchen.tensor import (TensorWiseINT8Layout, TensorCoreConvRotW4A4Layout,
                                         AsymW4A8Int8Layout)
        from contextlib import ExitStack
        root = str(comfyui_root())
        with ExitStack() as stack:
            for layout in (TensorWiseINT8Layout, TensorCoreConvRotW4A4Layout, AsymW4A8Int8Layout):
                stack.enter_context(patch.object(layout, 'dequantize', side_effect=AssertionError('float weight fallback')))
            for batch in (1, 16):
                result = run(64, 256, batch, 3, root, optimization=OptimizationConfig('mse', 'weight'))
                self.assertEqual(len(result['results']), 5)
                self.assertTrue(all(r['native_loader'] and r['graph_replay'] for r in result['results']))


if __name__ == '__main__':
    unittest.main()
