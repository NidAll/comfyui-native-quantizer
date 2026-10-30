import unittest
from tests.support import isolated_module as setUpModule
import comfyui_native_quantizer as q
from native_quantizer.planning.classification import classify_tensors
from native_quantizer.planning.types import DecisionKind


class MiniCPMTests(unittest.TestCase):
    def signature(self):
        return {
            'language_model.layers.0.linear_attn.in_proj_qkv.weight': (6144, 1024),
            'language_model.layers.3.self_attn.q_proj.weight': (4096, 1024),
            'vision_tower.encoder.layers.0.self_attn.q_proj.weight': (1152, 1152),
            'merger.mlp.0.linear_2.weight': (1024, 4608),
        }

    def test_complete_signature_required(self):
        tensors = self.signature()
        self.assertEqual(q.detect_architecture(q._ckpt(list(tensors.items()))).architecture, 'minicpm_v47')
        for missing in tensors:
            with self.subTest(missing=missing), self.assertRaises(q.UnknownArchitectureError):
                q.detect_architecture(q._ckpt([(k, s) for k, s in tensors.items() if k != missing]))

    def test_other_model_size_refused(self):
        tensors = self.signature()
        tensors['language_model.layers.3.self_attn.q_proj.weight'] = (8192, 2048)
        with self.assertRaises(q.UnknownArchitectureError):
            q.detect_architecture(q._ckpt(list(tensors.items())))

    def test_recipe_protects_sensitive_weights(self):
        names = ['model.language_model.layers.1.mlp.up_proj.weight',
                 'model.language_model.layers.1.linear_attn.in_proj_qkv.weight',
                 'model.language_model.layers.1.linear_attn.in_proj_a.weight',
                 'model.language_model.layers.1.linear_attn.in_proj_b.weight',
                 'model.language_model.layers.0.mlp.up_proj.weight',
                 'model.language_model.layers.23.mlp.up_proj.weight',
                 'model.merger.mlp.0.linear_2.weight', 'lm_head.weight',
                 'model.vision_tower.encoder.layers.0.self_attn.q_proj.weight']
        info = q._ckpt([(name, (256, 256)) for name in names])
        detection = q.detect_architecture(info, override='minicpm_v47')
        decisions = classify_tensors(info, detection, 'mixed', None, [], [], [], None, None)
        selected = {d.name for d in decisions if d.kind == DecisionKind.QUANTIZE}
        self.assertEqual(selected, set(names[:2]))


if __name__ == '__main__':
    unittest.main()
