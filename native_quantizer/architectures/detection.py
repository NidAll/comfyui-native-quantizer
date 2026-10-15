"""Architecture detection from checkpoint names, shapes and config."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from native_quantizer.architectures.policies import FamilyPolicy, REGISTRY, REGISTRY_ORDER, get_family
from native_quantizer.checkpoint_io.types import CheckpointInfo
from native_quantizer.errors import UnknownArchitectureError
from native_quantizer.identity import COMPONENT_TEXT_ENCODER


UNET_PREFIX_CANDIDATES = ("model.diffusion_model.", "model.model.", "net.")


def unet_prefix_from_keys(keys: Iterable[str]) -> str:
    keys = list(keys)
    counts = {c: 0 for c in UNET_PREFIX_CANDIDATES}
    for k in keys:
        for c in UNET_PREFIX_CANDIDATES:
            if k.startswith(c):
                counts[c] += 1
                break
    top = max(counts, key=counts.get)
    # YuE acoustic checkpoints have model.layers plus root boundary tensors;
    # an outer model. wrapper makes only the layers begin with model.model.
    # Do not discard those boundary tensors by mistaking that inner path for
    # the wrapper of the entire model.
    if top == "model.model." and any(k.startswith("model.") and not k.startswith(top)
                                     for k in keys):
        return "model."
    if counts[top] > 0:
        return top
    return "model."


@dataclass
class DetectionResult:
    architecture: str
    confidence: str                 # high | medium | low
    policy: FamilyPolicy
    unet_prefix: str
    evidence: List[str] = field(default_factory=list)
    hints: List[str] = field(default_factory=list)
    competing: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    classifier_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture,
            "component": self.policy.component,
            "confidence": self.confidence,
            "unet_prefix": self.unet_prefix,
            "evidence": self.evidence,
            "hints": self.hints,
            "competing": self.competing,
            "warnings": self.warnings,
            "classifier_info": self.classifier_info,
        }


def _match_signatures(keys: Iterable[str], prefix: str,
                      signatures: Sequence[str]) -> List[str]:
    """Match complete tensor names within the selected model component."""
    keys = set(keys)
    scoped = {k for k in keys if k.startswith(prefix)} if prefix in UNET_PREFIX_CANDIDATES else keys
    if not scoped:
        scoped = keys
    names = scoped | {k[len(prefix):] for k in scoped if prefix and k.startswith(prefix)}
    return [sig for sig in signatures if sig in names]


def detect_architecture(info: CheckpointInfo, override: Optional[str] = None,
                        shape_lookup: Optional[Callable[[str], Optional[Tuple[int, ...]]]] = None,
                        ) -> DetectionResult:
    """Detect the model architecture from checkpoint structure alone."""
    keys = list(info.key_set())
    prefix = unet_prefix_from_keys(keys)
    if override:
        policy = get_family(override)
        return DetectionResult(
            architecture=policy.family, confidence="high",
            policy=policy,
            unet_prefix=("" if policy.component == COMPONENT_TEXT_ENCODER
                         else prefix),
            evidence=[f"user override --architecture {override}"],
            warnings=["architecture supplied by the user; detection was not used"])
    if shape_lookup is None:
        def shape_lookup(name: str) -> Optional[Tuple[int, ...]]:
            tensor = info.by_name(name)
            return tuple(tensor.shape) if tensor is not None else None

    component_scoped = prefix in UNET_PREFIX_CANDIDATES and any(k.startswith(prefix) for k in keys)

    def model_shape(name: str) -> Optional[Tuple[int, ...]]:
        if component_scoped:
            return shape_lookup(prefix + name)
        return shape_lookup(prefix + name) or shape_lookup(name)

    # ---- structural families first (mirroring ComfyUI's branch order) ----
    scoped_keys = [k for k in keys if k.startswith(prefix)] if component_scoped else keys
    if not scoped_keys:
        scoped_keys = keys
    stripped = {k[len(prefix):] if k.startswith(prefix) else k for k in scoped_keys}

    def has(*subs: str) -> bool:
        return all(s in stripped for s in subs)

    def any_of(*subs: str) -> bool:
        return any(s in stripped for s in subs)

    raw_metadata_config = info.metadata.get("config")
    metadata_config: Dict[str, Any] = {}
    if isinstance(raw_metadata_config, str):
        try:
            parsed_config = json.loads(raw_metadata_config)
        except json.JSONDecodeError:
            parsed_config = None
        if isinstance(parsed_config, dict):
            metadata_config = parsed_config
    transformer_config = metadata_config.get("transformer")
    ming_metadata = (isinstance(transformer_config, dict)
                     and transformer_config.get("image_model") == "ming_image")

    candidates: List[Tuple[str, List[str], List[str]]] = []  # (family, evidence, hints)

    if has("vae2llm.weight"):
        candidates.append(("yue2", ["vae2llm.weight"],
                           [s for s in ("llm2vae.weight", "latent_pos_embed.pe") if has(s)]))
    if has("encoder.layers.0.pre_self_attn_layernorm.weight"):
        candidates.append(("text_encoder_t5gemma",
                           ["encoder.layers.0.pre_self_attn_layernorm.weight"],
                           ["encoder.layers.0.self_attn.q_proj.weight"]
                           if has("encoder.layers.0.self_attn.q_proj.weight") else []))
    if has("thinker.norm.weight",
           "thinker.layers.0.attention.query_key_value.weight",
           "connector.layers.0.self_attn.q_proj.weight", "proj_out.weight"):
        candidates.append(("ming_image_text_encoder",
                           ["thinker.norm.weight",
                            "thinker.layers.0.attention.query_key_value.weight",
                            "connector.layers.0.self_attn.q_proj.weight"],
                           [s for s in ("tokenizer_json", "vision.patch_embed.proj.weight",
                                        "thinker.layers.0.mlp.gate_up_proj.weight") if has(s)]))

    # 1. mmdit (SD3 / SD3.5)
    if has("joint_blocks.0.context_block.attn.qkv.weight", "x_embedder.proj.weight"):
        candidates.append(("mmdit_sd3", ["joint_blocks.0.context_block.attn.qkv.weight",
                                         "x_embedder.proj.weight"],
                           [s for s in ("final_layer.linear.weight", "y_embedder.mlp.0.weight",
                                        "context_embedder.weight") if has(s)]))
    # 2. stable cascade
    if has("clf.1.weight"):
        candidates.append(("stable_cascade", ["clf.1.weight"],
                           [s for s in ("clip_txt_mapper.weight", "clip_mapper.weight",
                                        "clip_img_mapper.weight") if has(s)]))
    # 3. stable audio
    if has("transformer.rotary_pos_emb.inv_freq"):
        candidates.append(("stable_audio", ["transformer.rotary_pos_emb.inv_freq"],
                           [s for s in ("to_global_embed.0.weight", "to_timestep_embed.0.weight") if has(s)]))
    # 4. aura flow
    if has("double_layers.0.attn.w1q.weight"):
        candidates.append(("aura_flow", ["double_layers.0.attn.w1q.weight"],
                           [s for s in ("single_layers.0.attn.w1q.weight", "cond_seq_linear.weight") if has(s)]))
    # 5. hydit
    if has("mlp_t5.0.weight"):
        extra_shape = model_shape("extra_embedder.0.weight")
        hydit_family = "hydit1" if extra_shape and len(extra_shape) == 2 and extra_shape[1] == 3968 else "hydit"
        candidates.append((hydit_family, ["mlp_t5.0.weight"],
                           [s for s in ("blocks.0.attn1.Wqkv.weight", "x_embedder.proj.weight",
                                        "extra_embedder.0.weight") if has(s)]))
    # 6. hunyuan video / image
    if has("txt_in.individual_token_refiner.blocks.0.norm1.weight"):
        candidates.append(("hunyuan_video", ["txt_in.individual_token_refiner.blocks.0.norm1.weight"],
                           [s for s in ("img_in.proj.weight", "final_layer.linear.weight",
                                        "double_blocks.0.img_attn.qkv.weight") if has(s)]))
    # Flux 3 has mode-specific blocks rather than Flux/Flux 2 double blocks.
    # Require both native boundary signatures with coherent logical dimensions;
    # neither a lone projection nor an unrelated nested component is sufficient.
    if has("content_mode_blocks.image.0.attn_out.weight",
           "single_stream_modulations.txt.lin.weight"):
        out_shape = model_shape("content_mode_blocks.image.0.attn_out.weight")
        mod_shape = model_shape("single_stream_modulations.txt.lin.weight")
        if (out_shape and len(out_shape) == 2 and out_shape[0] > 0
                and out_shape[0] % 128 == 0 and out_shape[0] == out_shape[1]
                and mod_shape == (3 * out_shape[0], out_shape[0])):
            candidates.append(("flux3", ["content_mode_blocks.image.0.attn_out.weight",
                                         "single_stream_modulations.txt.lin.weight"],
                               [s for s in ("emb_in.image.weight", "txt_in.weight",
                                            "single_blocks.0.qkv.weight",
                                            "single_blocks.0.q_proj.weight",
                                            "ref_time_in.in_layer.weight") if has(s)]))

    # 7. flux / chroma / flux2
    # Hunyuan variants also use Flux-style attention blocks. Their boundary
    # projections identify the model; a shared block norm cannot override it.
    if has("latent_in.weight"):
        candidates.append(("hunyuan3d", ["latent_in.weight"],
                           [s for s in ("cond_in.weight", "double_blocks.0.img_attn.qkv.weight") if has(s)]))
    elif (not has("txt_in.individual_token_refiner.blocks.0.norm1.weight")
          and any_of("double_blocks.0.img_attn.norm.key_norm.weight",
                     "double_blocks.0.img_attn.norm.key_norm.scale")):
        if has("double_stream_modulation_img.lin.weight"):
            candidates.append(("flux2", ["double_stream_modulation_img.lin.weight"],
                               [s for s in ("double_blocks.0.img_attn.qkv.weight",) if has(s)]))
        elif any_of("distilled_guidance_layer.norms.0.weight",
                    "distilled_guidance_layer.0.norms.0.weight"):
            chroma_family = "chroma_radiance" if any_of("nerf_blocks.0.norm.weight", "nerf_blocks.0.norm.scale") else "chroma"
            candidates.append((chroma_family, ["distilled_guidance_layer.norms.0.weight"],
                               [s for s in ("nerf_blocks.0.norm.weight", "img_in.weight") if has(s)]))
        else:
            candidates.append(("flux", ["double_blocks.0.img_attn.norm.key_norm.weight"],
                               [s for s in ("img_in.weight", "txt_in.weight",
                                            "single_blocks.0.linear1.weight") if has(s)]))
    # 8. mochi
    if has("t5_yproj.weight"):
        candidates.append(("mochi", ["t5_yproj.weight"],
                           [s for s in ("blocks.0.attn.qkv_x.weight",) if has(s)]))
    # 9. minimax h3 (checked before ltxv, like ComfyUI)
    if has("video_patch_proj.weight", "audio_patch_proj.weight"):
        candidates.append(("minimax_h3", ["video_patch_proj.weight", "audio_patch_proj.weight"],
                           [s for s in ("blocks.0.attn.qkv_proj.weight", "final_layer.video_out.weight",
                                        "adaln_t_table") if has(s)]))
    # 10. minimax music 3
    if has("cond_layer_logits", "latent_conditioners.0.weight",
           "diffusion_transformer.transformer.layers.0.self_attn.to_qkv.weight"):
        candidates.append((
            "minimax_music3",
            ["cond_layer_logits", "latent_conditioners.0.weight",
             "diffusion_transformer.transformer.layers.0.self_attn.to_qkv.weight"],
            [s for s in (
                "diffusion_transformer.transformer.layers.0.ff.ff.0.proj.weight",
                "diffusion_transformer.transformer.layers.0.ff.ff.2.weight",
                "diffusion_transformer.transformer.project_in.weight",
            ) if has(s)],
        ))
    # 11. ltxv / ltxav
    if has("adaln_single.emb.timestep_embedder.linear_1.bias"):
        ltx_family = "ltxav" if has("audio_adaln_single.linear.weight") else "ltxv"
        candidates.append((ltx_family, ["adaln_single.emb.timestep_embedder.linear_1.bias"],
                           [s for s in ("transformer_blocks.0.attn2.to_k.weight",
                                        "audio_adaln_single.linear.weight") if has(s)]))
    # 12. ACE-Step / ACE-Step 1.5
    if has(
        "encoder.lyric_encoder.layers.0.input_layernorm.weight",
        "decoder.layers.0.self_attn.q_proj.weight",
    ):
        candidates.append((
            "ace_step",
            [
                "encoder.lyric_encoder.layers.0.input_layernorm.weight",
                "decoder.layers.0.self_attn.q_proj.weight",
            ],
            [
                s for s in (
                    "decoder.layers.0.cross_attn.q_proj.weight",
                    "decoder.layers.0.mlp.gate_proj.weight",
                )
                if has(s)
            ],
        ))

    # Legacy ACE-Step
    elif has("genre_embedder.weight"):
        candidates.append((
            "ace_step",
            ["genre_embedder.weight"],
            [],
        ))
    # 13. pixart
    if has("t_block.1.weight"):
        pixart_family = "pixart" if has("ar_embedder.mlp.0.weight") else "pixart_sigma"
        candidates.append((pixart_family, ["t_block.1.weight"],
                           [s for s in ("blocks.0.attn.qkv.weight", "x_embedder.proj.weight",
                                        "y_embedder.y_embedding") if has(s)]))
    # 14. cosmos
    if has("blocks.block0.blocks.0.block.attn.to_q.0.weight"):
        candidates.append(("cosmos", ["blocks.block0.blocks.0.block.attn.to_q.0.weight"],
                           [s for s in ("x_embedder.proj.1.weight",) if has(s)]))
    # 14. PiD (checked before PixelDiT)
    if has("lq_proj.latent_proj.0.weight"):
        candidates.append(("pid", ["lq_proj.latent_proj.0.weight"],
                           [s for s in ("lq_proj.gate_modules.0.content_proj.weight",
                                        "lq_proj.pit_head.weight") if has(s)]))
    # Iris is a distinct dual/single-stream trunk, not PixelDiT patch_blocks.
    iris_signature = REGISTRY['iris'].detect_primary
    if has(*iris_signature):
        fused = has('blocks.0.mlp_x.gate_up.weight')
        mlp_input = 'blocks.0.mlp_x.gate_up.weight' if fused else 'blocks.0.mlp_x.w1.weight'
        complete_mlp = (has(mlp_input, 'blocks.0.mlp_x.w2.weight') and
                        (fused or has('blocks.0.mlp_x.w3.weight')))
        mixed_mlp = fused and any_of('blocks.0.mlp_x.w1.weight', 'blocks.0.mlp_x.w3.weight')
        shapes = {name: model_shape(name) for name in (*iris_signature, mlp_input,
                                                      'blocks.0.mlp_x.w2.weight')}
        if complete_mlp and not mixed_mlp and all(shapes.values()):
            patch, pixel = shapes['s_embedder.proj.weight'], shapes['pixel_embedder.proj.weight']
            qnorm, knorm = shapes['blocks.0.attn.q_norm_x.weight'], shapes['blocks.0.attn.k_proj_x.weight']
            refiner, positions = shapes['y_embedder.refiner.proj.weight'], shapes['y_pos_embedding']
            compress, pnorm = shapes['pixel_blocks.0.compress.weight'], shapes['pixel_blocks.0.attn.q_norm.weight']
            up, down = shapes[mlp_input], shapes['blocks.0.mlp_x.w2.weight']
            pool = shapes['y_embedder.layer_pool.weight']
            valid = (all(len(s) == 2 and all(n > 0 for n in s) for s in (patch, pixel, knorm, refiner, compress, up, down))
                     and len(pool) == 2 and pool[0] == 1 and pool[1] > 0
                     and len(qnorm) == len(pnorm) == 1 and qnorm[0] > 0 and pnorm[0] > 0
                     and len(positions) == 3 and positions[0] == 1 and positions[1] > 0
                     and patch[0] == refiner[0] == positions[2] == knorm[1] == up[1] == down[0]
                     and patch[0] % qnorm[0] == 0 and knorm[0] % qnorm[0] == 0
                     and knorm[0] <= patch[0] and compress[0] % pnorm[0] == 0
                     and patch[1] % pixel[1] == 0
                     and math.isqrt(patch[1] // pixel[1]) ** 2 == patch[1] // pixel[1]
                     and compress[1] == pixel[0] * (patch[1] // pixel[1])
                     and up[0] == down[1] * (2 if fused else 1)
                     and (fused or model_shape('blocks.0.mlp_x.w3.weight') == up))
            if valid:
                candidates.append(('iris', list(iris_signature), [mlp_input]))
                classifier_info = {'layout': 'fused' if fused else 'unfused',
                                   'hidden_size': patch[0], 'mlp_hidden': down[1],
                                   'native_loader': 'ComfyUI-Iris'}
    for family in ('kandinsky6', 'kandinsky6_sr'):
        signature = REGISTRY[family].detect_primary
        if not has(*signature):
            continue
        dimensions = {name: model_shape(name) for name in signature}
        if not all(dimensions.values()):
            continue
        matrices = [s for name, s in dimensions.items() if not name.endswith('query_norm.weight') and name != 'pooled_bias']
        vectors = [s for name, s in dimensions.items() if name.endswith('query_norm.weight') or name == 'pooled_bias']
        if not all(len(s) == 2 and all(v > 0 for v in s) for s in matrices):
            continue
        if not all(len(s) == 1 and s[0] > 0 for s in vectors):
            continue
        visual = dimensions['visual_embeddings.in_layer.weight'][0]
        if dimensions['out_layer.out_layer.weight'][1] != visual:
            continue
        if family == 'kandinsky6':
            audio = dimensions['audio_embeddings.in_layer.weight'][0]
            valid = (dimensions['visual_blocks.0.videoT.feed_forward.in_layer.weight'][1] == visual
                     and dimensions['visual_blocks.0.audioT.feed_forward.in_layer.weight'][1] == audio
                     and dimensions['visual_blocks.0.va_cross_attention.to_key.weight'] == (visual, audio)
                     and visual % dimensions['visual_blocks.0.videoT.self_attention.query_norm.weight'][0] == 0
                     and audio % dimensions['visual_blocks.0.audioT.self_attention.query_norm.weight'][0] == 0
                     and dimensions['visual_blocks.0.va_modulation.out_layer.weight'][0] in (2 * visual + audio, 3 * visual))
        else:
            valid = (dimensions['visual_transformer_blocks.0.feed_forward.in_layer.weight'][1] == visual
                     and dimensions['visual_transformer_blocks.0.self_attention.to_query.weight'] == (visual, visual)
                     and visual % dimensions['visual_transformer_blocks.0.self_attention.query_norm.weight'][0] == 0
                     and dimensions['visual_transformer_blocks.0.visual_modulation.out_layer.weight'] == (6 * visual, dimensions['pooled_bias'][0])
                     and not any(k.startswith(('text_embeddings.', 'audio_embeddings.', 'video_text_embeddings.')) for k in stripped))
        if valid:
            candidates.append((family, list(signature), []))
    # 15. PixelDiT T2I
    if has("core.pixel_embedder.proj.weight") and not has("lq_proj.latent_proj.0.weight"):
        candidates.append(("pixeldit", ["core.pixel_embedder.proj.weight"],
                           [s for s in ("cap_embedder.1.weight", "noise_refiner.0.attention.k_norm.weight") if has(s)]))
    elif (has("pixel_embedder.proj.weight", "patch_blocks.0.attn.qkv_x.weight")
          and not has("lq_proj.latent_proj.0.weight")):
        candidates.append(("pixeldit", ["pixel_embedder.proj.weight",
                                         "patch_blocks.0.attn.qkv_x.weight"], []))
    # 16. Ming / lumina2 / zimage. Native Ming Design can retain the
    # patch-indexed input/output projections instead of x_embedder.weight.
    ming_embedder_keys = ["x_embedder.weight"]
    if not has(*ming_embedder_keys):
        ming_embedder_keys = ["all_x_embedder.2-1.weight",
                              "all_final_layer.2-1.linear.weight"]
    has_ming_marker = has("__ming_image__")
    has_pixel_space_decoder = has("dec_net.cond_embed.weight")
    has_cap_pad_token = has("cap_pad_token")
    has_x_pad_token = has("x_pad_token")
    if (has(*ming_embedder_keys, "cap_embedder.1.weight",
            "layers.0.attention.to_q.weight", "layers.0.feed_forward.w1.weight")
            and not has_pixel_space_decoder
            and has_cap_pad_token == has_x_pad_token
            and (has_ming_marker or ming_metadata or not has_cap_pad_token)):
        candidates.append(("ming_image",
                           (["__ming_image__"] if has_ming_marker else []) +
                           (["config.transformer.image_model=ming_image"]
                            if ming_metadata else []) +
                           ming_embedder_keys +
                           ["cap_embedder.1.weight",
                            "layers.0.attention.to_q.weight",
                            "layers.0.feed_forward.w1.weight"],
                           [s for s in ("context_refiner.0.attention.to_q.weight",
                                        "noise_refiner.0.attention.to_q.weight") if has(s)]))
    if (has("cap_embedder.1.weight", "noise_refiner.0.attention.k_norm.weight")
            and not any(c[0] == "ming_image" for c in candidates)):
        caption_shape = model_shape("cap_embedder.1.weight")
        dim = caption_shape[0] if caption_shape and len(caption_shape) == 2 else None
        lumina_family = None
        evidence = ["cap_embedder.1.weight", "noise_refiner.0.attention.k_norm.weight"]
        if dim == 3840 and has_pixel_space_decoder:
            lumina_family = "z_image_pixel_space"
            evidence.append("dec_net.cond_embed.weight")
        elif dim == 3840 and has_cap_pad_token and has_x_pad_token and not (has_ming_marker or ming_metadata):
            lumina_family = "z_image"
            evidence.extend(("cap_pad_token", "x_pad_token"))
        elif dim == 2304 and not (has_cap_pad_token or has_x_pad_token or has_pixel_space_decoder or has_ming_marker or ming_metadata):
            lumina_family = "lumina2"
        if lumina_family:
            candidates.append((lumina_family, evidence,
                               [s for s in ("layers.0.attention.qkv.weight", "x_embedder.weight") if has(s)]))
    # 17. cogvideox
    if has("blocks.0.norm1.linear.weight"):
        candidates.append(("cogvideox", ["blocks.0.norm1.linear.weight"],
                           [s for s in ("patch_embed.proj.weight", "transformer_blocks.0.attn1.to_q.weight") if has(s)]))
    # 18. wan
    if has("head.modulation"):
        candidates.append(("wan", ["head.modulation"],
                           [s for s in ("blocks.0.self_attn.q.weight", "blocks.0.cross_attn.k.weight",
                                        "patch_embedding.weight") if has(s)]))
    # 19. seedvr2 (must be checked before generic lumina2-ish catches; ComfyUI order)
    if any_of("blocks.35.mlp.vid.proj_out.weight", "blocks.35.mlp.all.proj_in_gate.weight",
              "blocks.31.mlp.all.proj_in_gate.weight"):
        candidates.append(("seedvr2", ["blocks.35.mlp.vid.proj_out.weight"
                                       if has("blocks.35.mlp.vid.proj_out.weight")
                                       else "blocks.35.mlp.all.proj_in_gate.weight"],
                           [s for s in ("x_embedder.proj.1.weight",) if has(s)]))
    # 20. cosmos predict2 / anima
    if has("blocks.0.mlp.layer1.weight"):
        if has("llm_adapter.blocks.0.cross_attn.q_proj.weight"):
            candidates.append(("anima", ["blocks.0.mlp.layer1.weight",
                                          "llm_adapter.blocks.0.cross_attn.q_proj.weight"],
                               [s for s in ("x_embedder.proj.1.weight",) if has(s)]))
        else:
            candidates.append(("cosmos_predict2", ["blocks.0.mlp.layer1.weight"],
                               [s for s in ("x_embedder.proj.1.weight",) if has(s)]))
    # 21. boogu (checked before omnigen2; both share the embedder skeleton)
    if has("double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight",
           "double_stream_layers.0.img_self_attn.to_q.weight",
           "double_stream_layers.0.img_feed_forward.linear_1.weight"):
        candidates.append(("boogu",
                           ["double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight",
                            "double_stream_layers.0.img_self_attn.to_q.weight",
                            "double_stream_layers.0.img_feed_forward.linear_1.weight"],
                           [s for s in ("single_stream_layers.0.attn.to_q.weight",
                                        "single_stream_layers.0.feed_forward.linear_1.weight",
                                        "context_refiner.0.attn.to_q.weight",
                                        "ref_image_patch_embedder.weight") if has(s)]))
    # 22. omnigen2
    if has("time_caption_embed.timestep_embedder.linear_1.bias", "layers.0.attn.to_q.weight"):
        candidates.append(("omnigen2",
                           ["time_caption_embed.timestep_embedder.linear_1.bias",
                            "layers.0.attn.to_q.weight"],
                           [s for s in ("layers.0.feed_forward.linear_1.weight",
                                        "context_refiner.0.attn.to_q.weight",
                                        "ref_image_patch_embedder.weight",
                                        "x_embedder.weight") if has(s)]))
    # 23. lens
    if has("transformer_blocks.0.attn.norm_added_q.weight",
           "transformer_blocks.0.img_mlp.w1.weight"):
        candidates.append(("lens", ["transformer_blocks.0.attn.norm_added_q.weight"],
                           [s for s in ("img_in.weight", "proj_out.weight") if has(s)]))
    # 24. mage flow (shape-disambiguated from qwen image)
    if has("txt_norm.weight") and has("proj_out.weight"):
        tn_shape = model_shape("txt_norm.weight")
        po_shape = model_shape("proj_out.weight")
        if (tn_shape is not None and len(tn_shape) >= 1 and tn_shape[0] == 2560
                and po_shape is not None and len(po_shape) >= 1 and po_shape[0] == 128):
            candidates.append(("mage_flow", ["txt_norm.weight", "proj_out.weight"],
                               [s for s in ("transformer_blocks.0.attn.to_q.weight",) if has(s)]))
    # 25. qwen image 2.1 (match ComfyUI's distinctive shared-modulation layout)
    if (has("txt_in.text_norm.weight", "modulation.1.weight",
            "transformer_blocks.0.attn.norm_q.weight", "img_in.weight",
            "proj_out.weight")
            and any_of("transformer_blocks.0.img_mlp.gate_up.weight",
                       "transformer_blocks.0.img_mlp.proj.weight")):
        candidates.append(("qwen_image21",
                           ["txt_in.text_norm.weight", "modulation.1.weight",
                            "transformer_blocks.0.attn.norm_q.weight"],
                           [s for s in ("img_in.weight", "proj_out.weight",
                                        "transformer_blocks.0.attn.to_q.weight",
                                        "transformer_blocks.0.img_mlp.gate_up.weight",
                                        "transformer_blocks.0.img_mlp.proj.weight")
                            if has(s)]))
    # 26. qwen image
    if has("txt_norm.weight") and has("proj_out.weight") and not any(c[0] == "mage_flow" for c in candidates):
        candidates.append(("qwen_image", ["txt_norm.weight", "proj_out.weight"],
                           [s for s in ("img_in.weight", "transformer_blocks.0.attn.to_q.weight",
                                        "time_text_embed.addition_t_embedding.weight") if has(s)]))
    # 26. ideogram4
    if has("embed_image_indicator.weight"):
        candidates.append(("ideogram4", ["embed_image_indicator.weight"],
                           [s for s in ("input_proj.weight", "layers.0.attn.qkv.weight") if has(s)]))
    # 27. krea2
    if has("txtfusion.projector.weight"):
        candidates.append(("krea2", ["txtfusion.projector.weight"],
                           [s for s in ("txtfusion.layerwise_blocks.0.prenorm.scale",
                                        "layers.0.attn.wq.weight") if has(s)]))
    # 28. kandinsky5
    if has("visual_transformer_blocks.0.cross_attention.key_norm.weight"):
        candidates.append(("kandinsky5", ["visual_transformer_blocks.0.cross_attention.key_norm.weight"],
                           [s for s in ("visual_embeddings.in_layer.weight",) if has(s)]))
    # 29. ace 1.5 (music)
    if has("encoder.lyric_encoder.layers.0.input_layernorm.weight") and not any(c[0] == "ace_step" for c in candidates):
        candidates.append(("ace_step", ["encoder.lyric_encoder.layers.0.input_layernorm.weight"],
                           [s for s in ("decoder.layers.0.self_attn.q_proj.weight",) if has(s)]))
    # 30. RT-DETR / DepthAnything / SAM3 (perception: unsupported)
    if has("encoder.pan_blocks.1.cv4.conv.weight"):
        candidates.append(("rt_detr_v4", ["encoder.pan_blocks.1.cv4.conv.weight"], []))
    if has("backbone.embeddings.patch_embeddings.projection.weight"):
        candidates.append(("depth_anything3", ["backbone.embeddings.patch_embeddings.projection.weight"],
                           [s for s in ("head.scratch.refinenet1.out_conv.weight",) if has(s)]))
    if has("detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight",
           "detector.transformer.decoder.query_embed.weight"):
        sam_family = "sam31" if has("detector.backbone.vision_backbone.propagation_convs.0.conv_1x1.weight") else "sam3"
        candidates.append((sam_family, ["detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight",
                                      "detector.transformer.decoder.query_embed.weight"], []))
    # 31. ernie image
    if has("layers.0.mlp.linear_fc2.weight"):
        candidates.append(("ernie_image", ["layers.0.mlp.linear_fc2.weight"],
                           [s for s in ("text_proj.weight",) if has(s)]))
    # Hunyuan3D v2.1 has a separate DiT layout from the Flux-style v2 model.
    if has("t_embedder.mlp.2.weight", "blocks.0.attn1.k_norm.weight"):
        candidates.append(("hunyuan3d21",
                           ["t_embedder.mlp.2.weight",
                            "blocks.0.attn1.k_norm.weight"],
                           [s for s in ("x_embedder.weight", "blocks.0.attn2.to_q.weight")
                            if has(s)]))
    # 32. classic SD UNets
    sd_family = None
    if "input_blocks.0.0.weight" in stripped:
        mc_shape = model_shape("input_blocks.0.0.weight")
        model_channels = int(mc_shape[0]) if mc_shape else None
        # context dim from the first transformer block's cross-attention to_k
        ctx_dim = None
        lin_attn = False
        # real checkpoints place the transformer sub-block at index 1
        # (input_blocks.1.1 / 2.1 / middle_block.1); the synthetic fixtures
        # use index 0. Probe the real layouts first, then fall back.
        for probe in ("input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight",
                      "input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight",
                      "input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight",
                      "input_blocks.5.1.transformer_blocks.0.attn2.to_k.weight",
                      "middle_block.1.transformer_blocks.0.attn2.to_k.weight",
                      "input_blocks.1.0.transformer_blocks.0.attn2.to_k.weight",
                      "input_blocks.1.0.transformer_blocks.0.attn2.q.weight",
                      "input_blocks.2.0.transformer_blocks.0.attn2.to_k.weight"):
            shp = model_shape(probe)
            if shp is not None:
                ctx_dim = int(shp[1]) if len(shp) == 2 else None
                lin_attn = probe.endswith("to_k.weight")
                break
        in_ch = None
        in_shp = model_shape("input_blocks.0.0.weight")
        if in_shp is not None:
            in_ch = int(in_shp[1])
        temporal = any(("time_stack" in k or "temporal_transformer" in k) for k in stripped)
        has_label_emb = has("label_emb.0.0.weight")
        # Local config is supporting evidence only.  Tensor shapes remain the
        # primary signal, but a standard HF field can resolve a missing probe.
        cfg_ctx = info.config.get("cross_attention_dim") or info.config.get("context_dim")
        if ctx_dim is None and isinstance(cfg_ctx, int):
            ctx_dim = int(cfg_ctx)
        if in_ch is None and isinstance(info.config.get("in_channels"), int):
            in_ch = int(info.config["in_channels"])
        classifier_info = {"model_channels": model_channels, "context_dim": ctx_dim,
                           "in_channels": in_ch, "linear_attention": lin_attn,
                           "temporal": temporal, "label_emb": has_label_emb,
                           "config_keys": sorted(info.config.keys())[:32]}
        if model_channels == 256:
            sd_family = "sd20"  # SD_X4Upscaler
        elif ctx_dim == 2048:
            sd_family = "sdxl"
        elif ctx_dim == 1280:
            sd_family = "sdxl_refiner"
        elif ctx_dim == 1024:
            sd_family = "svd" if (temporal or in_ch == 8) else "sd20"
        elif ctx_dim == 768:
            sd_family = "sd15"
        elif model_channels == 384:
            sd_family = "sdxl_refiner"
        if sd_family is not None:
            candidates.append((sd_family, ["input_blocks.0.0.weight"],
                               ["config.cross_attention_dim"]
                               if cfg_ctx is not None else []))

    # ---- score candidates ----
    best: Optional[Tuple[str, int, List[str], List[str]]] = None
    competing = []
    for fam, ev, hints in candidates:
        score = 2 * len(ev) + len(hints)
        if best is None or score > best[1]:
            best = (fam, score, ev, hints)
            competing = [fam]
        elif score == best[1] and fam != best[0]:
            competing.append(fam)

    if best is None:
        # try generic signature matching against the registry for anything missed
        for fam in REGISTRY_ORDER:
            if fam in {'flux3', 'iris', 'kandinsky6', 'kandinsky6_sr'}:
                # These families require complete, shape-checked signatures.
                continue
            if fam == 'pixeldit' and any_of('y_embedder.layer_pool.weight',
                                            'blocks.0.attn.q_norm_x.weight'):
                continue
            pol = REGISTRY[fam]
            ev = _match_signatures(keys, prefix, pol.detect_primary)
            if ev:
                hints = _match_signatures(keys, prefix, pol.detect_hints)
                score = 2 * len(ev) + len(hints)
                if best is None or score > best[1]:
                    best = (fam, score, ev, hints)
                    competing = [fam]
                elif score == best[1] and fam != best[0]:
                    competing.append(fam)

    if best is None:
        raise UnknownArchitectureError(
            "could not identify the model architecture from checkpoint structure. "
            "Use --architecture NAME (see --list-architectures) to supply one "
            "explicitly, or --inspect to review the checkpoint contents.")

    fam, score, ev, hints = best
    competing = [c for c in competing if c != fam]
    if competing:
        choices = ", ".join([fam] + sorted(set(competing)))
        raise UnknownArchitectureError(
            f"architecture detection is ambiguous between: {choices}. "
            "Pass --architecture NAME after inspecting the checkpoint; refusing "
            "to guess.")
    policy = REGISTRY[fam]
    if policy.component == COMPONENT_TEXT_ENCODER:
        prefix = ""
    if len(ev) >= 2:
        confidence = "high"
    elif len(ev) == 1 and len(hints) >= 1:
        confidence = "medium"
    else:
        confidence = "low"
    warnings = []
    if confidence == "low":
        warnings.append("low detection confidence; consider --architecture override")
    if policy.runtime_status == "unsupported":
        warnings.append(
            f"architecture {fam!r} has no ComfyUI quantized-loading path; conversion "
            "would not be consumable by ComfyUI (refusing unless forced)")
    detected_classifier_info = dict(locals().get("classifier_info", {}))
    if fam == "flux3":
        detected_classifier_info.update(
            hidden_size=model_shape("content_mode_blocks.image.0.attn_out.weight")[0],
            qkv_layout="fused" if has("single_blocks.0.qkv.weight") else "split",
            flow_map=has("ref_time_in.in_layer.weight"),
        )
    if fam == "ming_image":
        variant = "design-layer" if has_cap_pad_token else "design"
        detected_classifier_info["variant"] = variant
        if variant == "design-layer":
            warnings.append(
                "Ming-Image Design-Layer requires a layer-decomposition workflow; "
                "the bundled Design workflow produces invalid output with it")

    result = DetectionResult(
        architecture=fam, confidence=confidence, policy=policy,
        unet_prefix=prefix, evidence=ev, hints=hints,
        competing=competing, warnings=warnings,
        classifier_info=detected_classifier_info)
    return result
