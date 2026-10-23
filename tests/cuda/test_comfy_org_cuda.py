"""Reference-recipe ConvRot sizes through the installed native CUDA consumer."""
import json
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file

from tests.cpu import test_comfy_org_recipes as recipe_fixtures
from tests.support import isolated_module as setUpModule, load_comfy_cpu


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class ComfyOrgCudaTests(unittest.TestCase):
    def test_reference_payloads_native_integer_dispatch(self):
        fixture = recipe_fixtures.ComfyOrgTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.assertEqual(fixture.convert(device='cuda', target_runtime='nvidia'), 0)
        state = load_file(str(fixture.output))
        ops, _ = load_comfy_cpu()
        from comfy_kitchen.tensor import TensorWiseINT8Layout
        for name, conf in fixture.recipe['layers'].items():
            weight = name + '.weight'
            n, k = fixture.sd[weight].shape
            module = ops.mixed_precision_ops(compute_dtype=torch.bfloat16).Linear(k, n, bias=False, device='cpu')
            module.load_state_dict({'weight': state[weight], 'weight_scale': state[weight + '_scale'],
                'comfy_quant': torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8)}, strict=True)
            module.to('cuda')
            with self.subTest(layer=name, group=conf.get('convrot_groupsize', 0)):
                with torch.inference_mode(), patch.object(TensorWiseINT8Layout, 'dequantize',
                        side_effect=AssertionError('float weight fallback')):
                    for batch in (1, 16):
                        actual = module(torch.randn(batch, k, device='cuda', dtype=torch.bfloat16))
                        self.assertEqual(tuple(actual.shape), (batch, n))
                        self.assertTrue(torch.isfinite(actual).all())
            del module
