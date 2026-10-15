"""Architecture header fixtures from ComfyUI detectors and native block names.

These tests exercise routing and protected boundaries without loading models.
"""
import ast
import contextlib
import io
import json
import tempfile
import unittest
from tests.support import isolated_module as setUpModule, comfyui_root
from pathlib import Path

import safetensors.torch
import torch
from safetensors import safe_open

import comfyui_native_quantizer as q


# Independent checkpoint signatures: do not derive these from the registry.
SIGNATURES = {
    'mmdit_sd3': 'joint_blocks.0.context_block.attn.qkv.weight x_embedder.proj.weight',
    'stable_cascade': 'clf.1.weight clip_txt_mapper.weight',
    'stable_audio': 'transformer.rotary_pos_emb.inv_freq to_global_embed.0.weight',
    'aura_flow': 'double_layers.0.attn.w1q.weight single_layers.0.attn.w1q.weight',
    'hydit': 'mlp_t5.0.weight blocks.0.attn1.Wqkv.weight',
    'hunyuan_video': 'txt_in.individual_token_refiner.blocks.0.norm1.weight double_blocks.0.img_attn.norm.key_norm.weight img_in.proj.weight',
    'flux': 'double_blocks.0.img_attn.norm.key_norm.weight img_in.weight txt_in.weight',
    'flux2': 'double_blocks.0.img_attn.norm.key_norm.weight double_stream_modulation_img.lin.weight img_in.weight',
    'chroma': 'double_blocks.0.img_attn.norm.key_norm.weight distilled_guidance_layer.norms.0.weight',
    'mochi': 't5_yproj.weight blocks.0.attn.qkv_x.weight',
    'minimax_h3': 'video_patch_proj.weight audio_patch_proj.weight blocks.0.attn.qkv_proj.weight',
    'minimax_music3': 'cond_layer_logits latent_conditioners.0.weight diffusion_transformer.transformer.layers.0.self_attn.to_qkv.weight',
    'ltxv': 'adaln_single.emb.timestep_embedder.linear_1.bias transformer_blocks.0.attn2.to_k.weight',
    'ace_step': 'encoder.lyric_encoder.layers.0.input_layernorm.weight decoder.layers.0.self_attn.q_proj.weight',
    'pixart': 't_block.1.weight ar_embedder.mlp.0.weight blocks.0.attn.qkv.weight',
    'cosmos': 'blocks.block0.blocks.0.block.attn.to_q.0.weight x_embedder.proj.1.weight',
    'cosmos_predict2': 'blocks.0.mlp.layer1.weight x_embedder.proj.1.weight',
    'anima': 'blocks.0.mlp.layer1.weight llm_adapter.blocks.0.cross_attn.q_proj.weight',
    'pixeldit': 'core.pixel_embedder.proj.weight core.patch_blocks.0.attn.qkv_x.weight',
    'wan': 'head.modulation blocks.0.self_attn.q.weight',
    'hunyuan3d': 'latent_in.weight cond_in.weight double_blocks.0.img_attn.norm.key_norm.weight',
    'hunyuan3d21': 't_embedder.mlp.2.weight blocks.0.attn1.k_norm.weight',
    'triposplat': 'cam_out_layer.weight repo_layers.0.final_map.weight',
    'hidream': 'caption_projection.0.linear.weight double_stream_blocks.0.block.attn1.to_q.weight',
    'hidream_o1': 't_embedder1.mlp.0.weight x_embedder.proj1.weight',
    'seedvr2': 'blocks.35.mlp.vid.proj_out.weight blocks.0.attn.proj_qkv.all.weight',
    'omnigen2': 'time_caption_embed.timestep_embedder.linear_1.bias layers.0.attn.to_q.weight',
    'boogu': 'double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight double_stream_layers.0.img_self_attn.to_q.weight double_stream_layers.0.img_feed_forward.linear_1.weight time_caption_embed.timestep_embedder.linear_1.bias',
    'lens': 'transformer_blocks.0.attn.norm_added_q.weight transformer_blocks.0.img_mlp.w1.weight',
    'ideogram4': 'embed_image_indicator.weight layers.0.attention.qkv.weight',
    'krea2': 'txtfusion.projector.weight blocks.0.attn.wq.weight',
    'kandinsky5': 'visual_transformer_blocks.0.cross_attention.key_norm.weight visual_embeddings.in_layer.weight',
    'cogvideox': 'blocks.0.norm1.linear.weight blocks.0.q.weight',
    'ernie_image': 'layers.0.mlp.linear_fc2.weight layers.0.self_attention.to_q.weight',
    'joyimage': 'double_blocks.0.attn.img_attn_qkv.weight double_blocks.0.attn.img_attn_q_norm.weight',
    'trellis2': 'img2shape.t_embedder.mlp.0.weight img2shape.blocks.0.self_attn.to_qkv.weight',
    'sensenova_u15': 'fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight language_model.model.layers.0.self_attn.q_proj_mot_gen.weight',
    'yue2': 'vae2llm.weight llm2vae.weight model.layers.0.self_attn.qkv_proj.weight',
    'rt_detr_v4': 'encoder.pan_blocks.1.cv4.conv.weight',
    'depth_anything3': 'backbone.embeddings.patch_embeddings.projection.weight',
    'sam3': 'detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight detector.transformer.decoder.query_embed.weight',
    'minimax_music3_text_encoder': 'audio_decoder.projection.weight audio_extra_embedding.weight layers.0.self_attn.q_proj.weight',
    'ming_image_text_encoder': 'thinker.norm.weight thinker.layers.0.attention.query_key_value.weight connector.layers.0.self_attn.q_proj.weight proj_out.weight',
    'text_encoder_llm': 'layers.0.self_attn.q_proj.weight layers.0.mlp.gate_proj.weight',
    'text_encoder_qwen35': 'layers.0.linear_attn.A_log layers.0.linear_attn.in_proj_qkv.weight',
    'text_encoder_jina_clip2': 'encoder.layers.0.mixer.Wqkv.weight encoder.layers.0.mlp.fc1.weight',
    'text_encoder_t5gemma': 'encoder.layers.0.pre_self_attn_layernorm.weight encoder.layers.0.self_attn.q_proj.weight',
    'text_encoder_yue2': 'layers.0.self_attn.qkv_proj.weight layers.0.mlp.gate_up_proj.weight',
    'text_encoder_t5': 'encoder.block.0.layer.0.SelfAttention.q.weight',
    'text_encoder_clip': 'text_model.encoder.layers.0.self_attn.q_proj.weight',
    'text_encoder_bert': 'encoder.layer.0.attention.self.query.weight',
    'chroma_radiance': 'double_blocks.0.img_attn.norm.key_norm.weight distilled_guidance_layer.norms.0.weight nerf_blocks.0.norm.weight',
    'ltxav': 'adaln_single.emb.timestep_embedder.linear_1.bias audio_adaln_single.linear.weight transformer_blocks.0.attn2.to_k.weight',
    'pixart_sigma': 't_block.1.weight blocks.0.attn.qkv.weight',
    'pid': 'lq_proj.latent_proj.0.weight core.pixel_embedder.proj.weight patch_blocks.0.attn.qkv_x.weight',
    'sam31': 'detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight detector.transformer.decoder.query_embed.weight detector.backbone.vision_backbone.propagation_convs.0.conv_1x1.weight',
}


def headers():
    cases = {family: dict.fromkeys(names.split(), (256, 256))
             for family, names in SIGNATURES.items()}
    lumina = {'cap_embedder.1.weight': (2304, 2560),
              'noise_refiner.0.attention.k_norm.weight': (128,),
              'layers.0.attention.qkv.weight': (64, 256)}
    cases['lumina2'] = lumina
    cases['z_image'] = dict(lumina, **{
        'cap_embedder.1.weight': (3840, 2560),
        'cap_pad_token': (1, 3840), 'x_pad_token': (1, 3840)})
    cases['z_image_pixel_space'] = dict(lumina, **{
        'cap_embedder.1.weight': (3840, 3072),
        'dec_net.cond_embed.weight': (256, 3840)})
    cases['ming_image'] = dict(lumina, **{
        'cap_embedder.1.weight': (3840, 2560),
        'x_embedder.weight': (3840, 64),
        'layers.0.attention.to_q.weight': (64, 256),
        'layers.0.feed_forward.w1.weight': (64, 256)})
    cases['hydit1'] = dict(cases['hydit'], **{'extra_embedder.0.weight': (256, 3968)})
    cases['mage_flow'] = {'txt_norm.weight': (2560,), 'proj_out.weight': (128, 256),
                          'transformer_blocks.0.attn.to_q.weight': (256, 256)}
    cases['qwen_image'] = {'txt_norm.weight': (3584,), 'proj_out.weight': (64, 256)}
    cases['qwen_image21'] = {
        'txt_in.text_norm.weight': (3584,), 'modulation.1.weight': (256, 256),
        'transformer_blocks.0.attn.norm_q.weight': (128,), 'img_in.weight': (256, 64),
        'proj_out.weight': (64, 256), 'transformer_blocks.0.img_mlp.gate_up.weight': (512, 256)}
    for family, context in (('sd15', 768), ('sd20', 1024), ('sdxl', 2048),
                            ('sdxl_refiner', 1280), ('svd', 1024)):
        cases[family] = {
            'input_blocks.0.0.weight': (320, 8 if family == 'svd' else 4, 3, 3),
            'input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight': (320, context)}
    cases['iris'] = {
        's_embedder.proj.weight': (256, 12), 'pixel_embedder.proj.weight': (64, 3),
        'y_embedder.layer_pool.weight': (1, 2), 'y_embedder.refiner.proj.weight': (256, 64),
        'y_pos_embedding': (1, 8, 256), 'blocks.0.attn.q_norm_x.weight': (64,),
        'blocks.0.attn.k_proj_x.weight': (128, 256), 'pixel_blocks.0.compress.weight': (256, 256),
        'pixel_blocks.0.attn.q_norm.weight': (64,),
        'blocks.0.mlp_x.w1.weight': (512, 256), 'blocks.0.mlp_x.w3.weight': (512, 256),
        'blocks.0.mlp_x.w2.weight': (256, 512),
    }
    cases['flux3'] = {
        'content_mode_blocks.image.0.attn_out.weight': (256, 256),
        'single_stream_modulations.txt.lin.weight': (768, 256),
        'emb_in.image.weight': (256, 128),
        'single_blocks.0.qkv.weight': (768, 256),
    }
    cases['kandinsky6'] = {
        'visual_embeddings.in_layer.weight': (256, 33), 'audio_embeddings.in_layer.weight': (128, 64),
        'video_time_embeddings.out_layer.weight': (64, 256), 'audio_time_embeddings.out_layer.weight': (64, 256),
        'video_text_embeddings.in_layer.weight': (256, 64), 'video_pooled_text_embeddings.in_layer.weight': (64, 64),
        'out_layer.out_layer.weight': (64, 256),
        'visual_blocks.0.videoT.feed_forward.in_layer.weight': (512, 256),
        'visual_blocks.0.audioT.feed_forward.in_layer.weight': (256, 128),
        'visual_blocks.0.videoT.self_attention.query_norm.weight': (64,),
        'visual_blocks.0.audioT.self_attention.query_norm.weight': (64,),
        'visual_blocks.0.va_modulation.out_layer.weight': (640, 64),
        'visual_blocks.0.va_cross_attention.to_key.weight': (256, 128),
    }
    cases['kandinsky6_sr'] = {
        'pooled_bias': (64,), 'visual_embeddings.in_layer.weight': (256, 33),
        'out_layer.out_layer.weight': (64, 256),
        'visual_transformer_blocks.0.visual_modulation.out_layer.weight': (1536, 64),
        'visual_transformer_blocks.0.self_attention.query_norm.weight': (64,),
        'visual_transformer_blocks.0.self_attention.to_query.weight': (256, 256),
        'visual_transformer_blocks.0.feed_forward.in_layer.weight': (512, 256),
    }
    return cases


class ArchitectureDetectionTests(unittest.TestCase):
    def test_header_matrix_bare_and_wrapped(self):
        cases = headers()
        self.assertEqual(set(cases), set(q.REGISTRY))
        for family, tensors in cases.items():
            for prefix in ('', 'model.', *q.UNET_PREFIX_CANDIDATES):
                with self.subTest(family=family, prefix=prefix):
                    info = q._ckpt([(prefix + name, shape) for name, shape in tensors.items()])
                    detected = q.detect_architecture(info)
                    self.assertEqual(detected.architecture, family)
                    self.assertEqual(detected.policy.component, q.REGISTRY[family].component)

    def test_variant_class_aliases_are_unique(self):
        owners = {}
        for family, policy in q.REGISTRY.items():
            for name in policy.comfyui_classes:
                self.assertNotIn(name, owners, (name, owners.get(name), family))
                owners[name] = family
                self.assertEqual(q.get_family(name).family, family)
        for name, family in (('ZImage', 'z_image'), ('ZImagePixelSpace', 'z_image_pixel_space'),
                             ('HunyuanDiT1', 'hydit1'), ('ChromaRadiance', 'chroma_radiance'),
                             ('LTXAV', 'ltxav'), ('PixArtSigma', 'pixart_sigma'),
                             ('PiD', 'pid'), ('SAM31', 'sam31')):
            self.assertEqual(q.get_family(name).family, family)

    def test_installed_comfy_supported_class_coverage(self):
        source = comfyui_root() / 'comfy/supported_models.py'
        tree = ast.parse(source.read_text())
        supported = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Name) and target.id == 'models'
                                 for target in node.targets))
        classes = {node.id for node in supported.elts}
        covered = {name for policy in q.REGISTRY.values() for name in policy.comfyui_classes}
        self.assertFalse(classes - covered, sorted(classes - covered))

    def test_diffusion_component_isolation(self):
        z = headers()['z_image']
        keys = [('model.diffusion_model.' + name, shape) for name, shape in z.items()]
        # A text encoder/other component must not outscore the diffusion model.
        keys += [('text_encoder.' + name, shape) for name, shape in headers()['ming_image'].items()]
        keys += list(headers()['flux'].items())
        self.assertEqual(q.detect_architecture(q._ckpt(keys)).architecture, 'z_image')

    def test_nested_and_suffix_names_are_not_signatures(self):
        for name in ('backup.layers.0.self_attn.q_proj.weight',
                     'layers.0.self_attn.q_proj.weight.extra',
                     'backup.cap_embedder.1.weight'):
            with self.subTest(name=name), self.assertRaises(q.UnknownArchitectureError):
                q.detect_architecture(q._ckpt([(name, (256, 256))]))

    def test_partial_z_image_and_ming_signatures_refuse_guessing(self):
        for removed in ('cap_pad_token', 'x_pad_token'):
            tensors = headers()['z_image']
            del tensors[removed]
            with self.subTest(removed=removed), self.assertRaises(q.UnknownArchitectureError):
                q.detect_architecture(q._ckpt(list(tensors.items())))
        tensors = headers()['lumina2']
        tensors['cap_embedder.1.weight'] = (4096, 2560)
        with self.assertRaises(q.UnknownArchitectureError):
            q.detect_architecture(q._ckpt(list(tensors.items())))

    def test_ltxav_audio_linears_and_protected_boundaries(self):
        targets = [f'transformer_blocks.0.{attention}.{projection}.weight'
                   for attention in ('audio_attn1', 'audio_attn2', 'audio_to_video_attn', 'video_to_audio_attn')
                   for projection in ('to_q', 'to_k', 'to_v', 'to_out.0')]
        targets += ['transformer_blocks.0.audio_ff.net.0.proj.weight',
                    'transformer_blocks.0.audio_ff.net.2.weight']
        kept = ['audio_patchify_proj.weight', 'audio_caption_projection.linear_1.weight',
                'audio_proj_out.weight', 'audio_adaln_single.linear.weight',
                'transformer_blocks.0.audio_attn1.q_norm.weight']
        tensors = headers()['ltxav']
        tensors.update(dict.fromkeys(targets + kept, (64, 256)))
        info = q._ckpt(list(tensors.items()))
        det = q.detect_architecture(info)
        decisions = {d.name: d.kind for d in q.classify_tensors(
            info, det, q.FORMAT_MIXED, None, [], [], [], None, None)}
        for name in targets:
            self.assertEqual(decisions[name], q.DecisionKind.QUANTIZE, name)
        for name in kept:
            self.assertNotEqual(decisions[name], q.DecisionKind.QUANTIZE, name)

    def test_variant_block_policies_preserve_boundaries(self):
        layouts = {
            'z_image': ('layers.0.attention.qkv.weight', 'x_embedder.weight'),
            'z_image_pixel_space': ('noise_refiner.0.feed_forward.w3.weight', 'dec_net.res_blocks.0.linear_1.weight'),
            'hydit1': ('blocks.0.attn1.Wqkv.weight', 'extra_embedder.0.weight'),
            'chroma_radiance': ('double_blocks.0.img_attn.qkv.weight', 'distilled_guidance_layer.norms.0.weight'),
            'pixart_sigma': ('blocks.0.attn.qkv.weight', 'final_layer.linear.weight'),
            'pid': ('patch_blocks.0.attn.qkv_x.weight', 'lq_proj.latent_proj.0.weight'),
        }
        for family, (target, boundary) in layouts.items():
            for prefix in ('', 'model.diffusion_model.'):
                with self.subTest(family=family, prefix=prefix):
                    tensors = headers()[family]
                    tensors[target] = (64, 256)
                    tensors.setdefault(boundary, (64, 256))
                    info = q._ckpt([(prefix + name, shape) for name, shape in tensors.items()])
                    det = q.detect_architecture(info)
                    self.assertEqual(det.architecture, family)
                    decisions = {d.name: d.kind for d in q.classify_tensors(
                        info, det, q.FORMAT_MIXED, None, [], [], [], None, None)}
                    self.assertEqual(decisions[prefix + target], q.DecisionKind.QUANTIZE)
                    self.assertIn(decisions[prefix + boundary],
                                  (q.DecisionKind.KEEP, q.DecisionKind.KEEP_PRECISION))

    def test_z_image_int8_convrot_conversion(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            src, out = (Path(tmp) / name for name in ('source.safetensors', 'output.safetensors'))
            generator = torch.Generator().manual_seed(93)
            tensors = {
                'cap_embedder.1.weight': torch.randn(3840, 1, generator=generator),
                'noise_refiner.0.attention.k_norm.weight': torch.ones(64),
                'cap_pad_token': torch.randn(1, 3840, generator=generator),
                'x_pad_token': torch.randn(1, 3840, generator=generator),
                'layers.0.attention.qkv.weight': torch.randn(64, 256, generator=generator),
                'layers.0.feed_forward.w1.weight': torch.randn(64, 256, generator=generator),
                'layers.0.adaLN_modulation.0.weight': torch.randn(64, 256, generator=generator),
                'x_embedder.weight': torch.randn(64, 256, generator=generator),
                'final_layer.linear.weight': torch.randn(64, 256, generator=generator),
            }
            tensors = {'model.diffusion_model.' + name: value for name, value in tensors.items()}
            safetensors.torch.save_file(tensors, str(src))
            with contextlib.redirect_stdout(io.StringIO()) as log:
                code = q.main([str(src), '--format', 'int8-convrot', '--output', str(out),
                               '--device', 'cpu', '--max-memory', '4M', '--validate',
                               '--progress', 'off'])
            self.assertEqual(code, 0, log.getvalue())
            self.assertIn('Architecture: z_image;', log.getvalue())
            with safe_open(str(out), framework='pt') as checkpoint:
                provenance = json.loads(checkpoint.metadata()['comfy_quantizer'])
                self.assertEqual(provenance['architecture'], 'z_image')
                for name, tensor in tensors.items():
                    loaded = checkpoint.get_tensor(name)
                    if name.endswith(('attention.qkv.weight', 'feed_forward.w1.weight')):
                        self.assertEqual(loaded.dtype, torch.int8)
                    else:
                        self.assertTrue(torch.equal(loaded, tensor), name)
            self.assertTrue(q.verify_output(str(out))['ok'])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(q.main(['--verify-output', str(out)]), 0)


if __name__ == '__main__':
    unittest.main()
