"""Built-in architecture policies and precision-preservation rules."""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from native_quantizer.errors import UnknownArchitectureError
from native_quantizer.identity import COMPONENT_DIFFUSION, COMPONENT_TEXT_ENCODER
from native_quantizer.utils import flatten_regex


KNOWN_GATE_DIMS: Dict[str, Tuple[int, ...]] = {
    "boogu": (3360,),
    "omnigen2": (2520,),        # fails K % 16 too
    "pixart": (1152,),
    "hydit": (1408,),
    "cogvideox": (1920,),       # only the 2b variant; 5b (3072) is clean
    "minimax_h3": (1152,),      # mlp.fc2 only
    "sdxl": (320, 640),
    "sd15": (320,),
    "svd": (320, 640),
    "stable_cascade": (320, 640),
    "ace_step": (1152,),
}


@dataclass(frozen=True)
class FamilyPolicy:
    family: str
    comfyui_classes: Tuple[str, ...]
    detect_primary: Tuple[str, ...]
    detect_hints: Tuple[str, ...] = ()
    quantize: Tuple[str, ...] = ()
    keep: Tuple[str, ...] = ()
    exclude: Tuple[str, ...] = ()
    group_size: int = 16
    min_weight_numel: int = 4096
    max_rel_l2: float = 0.25
    min_cosine: float = 0.95
    runtime_status: str = "experimental"
    notes: str = ""
    component: str = COMPONENT_DIFFUSION
    linear_alignment: Tuple[int, int] = (1, 1)

    def quantize_re(self) -> re.Pattern:
        return (flatten_regex(self.quantize) if self.quantize
                else re.compile(r"(?!x)x"))

    def keep_re(self) -> re.Pattern:
        return (flatten_regex(self.keep) if self.keep
                else re.compile(r"(?!x)x"))

    def exclude_re(self) -> re.Pattern:
        return (flatten_regex(self.exclude) if self.exclude
                else re.compile(r"(?!x)x"))

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        return d


UNIVERSAL_EXCLUDE = (
    r"(^|\.)(norm|norm1|norm2|norm3|ln\w*|layer_norm|rms_norm|final_norm|final_layer_norm|"
    r"q_norm|k_norm|query_norm|key_norm|pre_norm|post_norm|adaln_norm|norm_added_q|"
    r"prenorm|input_layernorm|self_attn_norm|post_attention_layernorm|"
    r"ln_final|ln1|ln2|ln_pre|ln_post|emb_norm|token_norm|txt_norm|ffn_norm1|ffn_norm2)"
    r"\.(weight|bias|scale|shift)$",
    r"(^|\.)(pos_embed|pos_embedding|positional_encoding|position_ids|"
    r"emb_pos|pos_emb|rotary_pos_emb|rope|inv_freq|freqs_cis|emb_tokens|"
    r"embed_positions|patch_embedding|patch_embed|adaln_t_table|cap_pad_token|"
    r"__x0__|__sequential__|memory_tokens|timestep_features)(\.|$)",
    r"(^|\.)(time_embed|t_embedder|timestep_embedder|time_embedder|time_embeddings|"
    r"ofs_embedding|fps_embedding|style_embedding|cond_embedding|guidance_in|"
    r"vector_in|txt_in|img_in|input_embedder|x_embedder|y_embedder|context_embedder|"
    r"patch_embedder|patchify_proj|video_patch_proj|audio_patch_proj|condition_proj|"
    r"cond_proj|cond_embed|input_proj|img_emb|text_proj|caption_proj|t5_yproj|"
    r"final_layer|output_layer|head|out_layer|final_linear|"
    r"audio_out|video_out|linear_fc2|to_gate_logits|genre_embedder|speaker_embedder|"
    r"lyric_proj|text_embedding|ref_image_patch_embedder|ofs_embedding_linear_1|"
    r"ofs_embedding_linear_2|time_embedding_linear_1|time_embedding_linear_2|"
    r"adaln_proj|adaln_modulation|adaLN_modulation|adaln_single|time_caption_embed|extra_embedder|"
    r"text_embedder|label_emb|clip_txt_mapper|clip_img_mapper|clip_mapper|"
    r"clip_txt_pooled_mapper|cond_type_embedding|distilled_guidance_layer|"
    r"control_adapter|ref_conv|latent_in|cond_in|cam_out_layer|repo_layers|"
    r"content_map|gate_map|final_map|input_layer|cam_enc|cam_dec|"
    r"visual_embeddings|time_embeddings|ofs_embedding_linear|patch_embedding_mask|"
    r"patch_embedding_pose|patch_embedding_global|emb|embed|embedding|"
    r"condition_embedder|adaln_curve|llm_cond_proj|input_embedder|pos_embed_proj|"
    r"visual_transformer_blocks|text_transformer_blocks|"
    r"encoder|decoder|lyric_encoder|ssl_|vocoder|first_stage|cond_stage)(\.|$)",
)


UNET_ATTN_Q = (
    r"(input_blocks|output_blocks|middle_block)\.\d+(\.\d+)?\.transformer_blocks\.\d+\.attn[12]\.(to_q|to_k|to_v|to_out\.0)\.weight",
)


UNET_ATTN_K = (
    r"(input_blocks|output_blocks|middle_block)\.\d+(\.\d+)?\.transformer_blocks\.\d+\.attn[12]\.(to_q|to_k|to_v|to_out\.0)\.weight",
)


UNET_FF = (
    r"(input_blocks|output_blocks|middle_block)\.\d+(\.\d+)?\.transformer_blocks\.\d+\.ff\.net\.(0|2)\.(proj)?\.?weight$",
)


UNET_TIME_EMBED = (r"(^|\.)time_embed\.(0|2)\.weight$",)


UNET_LABEL_EMB = (r"(^|\.)label_emb\.0\.0\.weight$",)


def _sd_unet_policy(family: str, classes: Tuple[str, ...], notes: str = "") -> FamilyPolicy:
    return FamilyPolicy(
        family=family, comfyui_classes=classes,
        detect_primary=(),
        detect_hints=(),
        quantize=UNET_ATTN_Q + UNET_FF,
        keep=UNET_TIME_EMBED + UNET_LABEL_EMB + (
            r"(^|\.)(out\.\d+|output_blocks\.\d+\.\d+\.conv|input_blocks\.\d+\.\d+\.conv)\.weight$",
        ),
        exclude=UNIVERSAL_EXCLUDE,
        runtime_status="experimental",
        notes=notes,
    )


REGISTRY_ORDER: List[str] = []


REGISTRY: Dict[str, FamilyPolicy] = {}


def _register(policy: FamilyPolicy) -> None:
    REGISTRY[policy.family] = policy
    REGISTRY_ORDER.append(policy.family)


_register(FamilyPolicy(
    family="mmdit_sd3",
    comfyui_classes=("SD3",),
    detect_primary=("joint_blocks.0.context_block.attn.qkv.weight",),
    detect_hints=("x_embedder.proj.weight", "final_layer.linear.weight", "y_embedder.mlp.0.weight"),
    quantize=(
        r"joint_blocks\.\d+\.(x_block|context_block)\.attn\.(qkv|proj)\.weight$",
        r"joint_blocks\.\d+\.(x_block|context_block)\.mlp\.(fc1|fc2)\.weight$",
    ),
    keep=(r"(^|\.)(x_embedder|y_embedder|context_embedder|final_layer|pos_embed)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
    notes="SD3 / SD3.5 MMDiT joint-block family (also covers sd3.5 medium/large).",
))


_register(FamilyPolicy(
    family="stable_cascade",
    comfyui_classes=("Stable_Cascade_C", "Stable_Cascade_B"),
    detect_primary=("clf.1.weight",),
    detect_hints=("clip_txt_mapper.weight", "clip_mapper.weight", "clip_img_mapper.weight"),
    quantize=(
        r"^(down_blocks|up_blocks)\.\d+\.\d+\."
        r"attention\.attn\.(to_q|to_k|to_v|out_proj)\.weight$",
        r"^(down_blocks|up_blocks)\.\d+\.\d+\."
        r"channelwise\.(0|4)\.weight$",
    ),
    keep=(r"(^|\.)(clip_txt_mapper|clip_img_mapper|clip_mapper|clip_txt_pooled_mapper)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
    notes="Würstchen stage B/C DiT.",
))


_register(FamilyPolicy(
    family="stable_audio",
    comfyui_classes=("StableAudio", "StableAudio3"),
    detect_primary=("transformer.rotary_pos_emb.inv_freq",),
    detect_hints=("to_global_embed.0.weight", "to_cond_embed.0.weight", "to_timestep_embed.0.weight"),
    quantize=(
        r"transformer\.layers\.\d+\.(self_attn|cross_attn)\.(to_qkv|to_q|to_kv|to_out)\.weight$",
        r"transformer\.layers\.\d+\.ff\.ff\.(0\.proj|0\.1|2)\.weight$",
        r"transformer\.layers\.\d+\.to_local_embed\.(0|2)\.weight$",
    ),
    keep=(r"(^|\.)(to_global_embed|to_cond_embed|to_timestep_embed|postprocess_conv|project_in|project_out|transformer\.project_in|transformer\.project_out|global_cond_embedder)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
    notes="Stable Audio 1/3 DiT (audio).",
))


_register(FamilyPolicy(
    family="aura_flow",
    comfyui_classes=("AuraFlow",),
    detect_primary=("double_layers.0.attn.w1q.weight",),
    detect_hints=("single_layers.0.attn.w1q.weight", "cond_seq_linear.weight", "positional_encoding"),
    quantize=(
        r"(double_layers|single_layers)\.\d+\.attn\.(w1q|w1k|w1v|w2|o_proj|w2q|w2k|w2v|w1o|w2o)\.weight$",
        r"double_layers\.\d+\.mlp(C|X)\.(c_fc1|c_fc2|c_proj)\.weight$",
        r"single_layers\.\d+\.mlp\.(c_fc1|c_fc2|c_proj)\.weight$",
    ),
    keep=(r"(^|\.)(cond_seq_linear|init_x_linear|final_linear|positional_encoding)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))


_register(FamilyPolicy(
    family="hydit",
    comfyui_classes=("HunyuanDiT", "HunyuanDiT1"),
    detect_primary=("mlp_t5.0.weight",),
    detect_hints=("blocks.0.attn1.Wqkv.weight", "x_embedder.proj.weight", "extra_embedder.0.weight"),
    quantize=(
        r"blocks\.\d+\.attn1\.(Wqkv|out_proj)\.weight$",
        r"blocks\.\d+\.attn2\.(q_proj|kv_proj|out_proj)\.weight$",
        r"blocks\.\d+\.mlp\.(fc1|fc2)\.weight$",
    ),
    keep=(r"(^|\.)(x_embedder|t_embedder|extra_embedder|text_embedder|final_layer|pos_embed|mlp_t5)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))


_register(FamilyPolicy(
    family="hunyuan_video",
    comfyui_classes=("HunyuanVideo", "HunyuanVideoI2V", "HunyuanVideoSkyreelsI2V",
                     "HunyuanImage21", "HunyuanImage21Refiner", "HunyuanVideo15",
                     "HunyuanVideo15_SR_Distilled"),
    detect_primary=("txt_in.individual_token_refiner.blocks.0.norm1.weight",),
    detect_hints=("img_in.proj.weight", "final_layer.linear.weight",
                  "double_blocks.0.img_attn.qkv.weight", "txt_in.input_embedder.weight"),
    quantize=(
        r"double_blocks\.\d+\.(img_attn|txt_attn)\.(qkv|proj)\.weight$",
        r"double_blocks\.\d+\.(img_mlp|txt_mlp)\."
        r"(0|2|w1|w2|gate_proj|up_proj|down_proj)\.weight$",
        r"single_blocks\.\d+\.(linear1|linear2)\.weight$",
    ),
    keep=(r"(^|\.)(img_in|txt_in|vector_in|guidance_in|fps_embedding|style_embedding|"
          r"cond_embedding|txt_std|txt_emb|final_layer|individual_token_refiner|"
          r"byt5_in|time_r_in|vision_in|cond_type_embedding|time_embed|extra_embedder|"
          r"audio_embed|adaln_)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
    notes="HunyuanVideo 1.x + HunyuanImage 2.1 families share this DiT structure.",
))


_register(FamilyPolicy(
    family="flux",
    comfyui_classes=("Flux", "FluxInpaint", "FluxSchnell", "LongCatImage"),
    detect_primary=("double_blocks.0.img_attn.norm.key_norm.weight",
                    "double_blocks.0.img_attn.norm.key_norm.scale"),
    detect_hints=("img_in.weight", "txt_in.weight", "single_blocks.0.linear1.weight",
                  "guidance_in.in_layer.weight", "vector_in.in_layer.weight"),
    quantize=(
        r"double_blocks\.\d+\.img_attn\.(qkv|proj)\.weight$",
        r"double_blocks\.\d+\.txt_attn\.(qkv|proj)\.weight$",
        r"double_blocks\.\d+\.img_mlp\.(0|2|w1|w2|gate_proj|up_proj|down_proj)\.weight$",
        r"double_blocks\.\d+\.txt_mlp\.(0|2|w1|w2|gate_proj|up_proj|down_proj)\.weight$",
        r"single_blocks\.\d+\.(linear1|linear2)\.weight$",
    ),
    keep=(r"(^|\.)(img_in|txt_in|vector_in|guidance_in|time_text_embed|txt_embed|"
          r"final_layer|distilled_guidance_layer|txt_norm)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="verified",
    notes="Flux family; runtime verified via the reference W4A8 pipeline (same "
          "double_blocks/single_blocks structure as MiniMax H3).",
))


_register(FamilyPolicy(
    family="flux2",
    comfyui_classes=("Flux2",),
    detect_primary=("double_stream_modulation_img.lin.weight",),
    detect_hints=("double_blocks.0.img_attn.qkv.weight", "img_in.weight"),
    quantize=(
        r"double_blocks\.\d+\.(img_attn|txt_attn)\.(qkv|proj)\.weight$",
        r"double_blocks\.\d+\.(img_mlp|txt_mlp)\."
        r"(0|2|w1|w2|gate_proj|up_proj|down_proj)\.weight$",
        r"single_blocks\.\d+\.(linear1|linear2)\.weight$",
    ),
    keep=(r"(^|\.)(img_in|txt_in|vector_in|guidance_in|time_text_embed|final_layer)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))


_register(FamilyPolicy(
    family="flux3",
    comfyui_classes=("Flux3",),
    detect_primary=("content_mode_blocks.image.0.attn_out.weight",
                    "single_stream_modulations.txt.lin.weight"),
    detect_hints=("emb_in.image.weight", "emb_in.image_cond.weight",
                  "txt_in.weight", "single_blocks.0.qkv.weight",
                  "single_blocks.0.q_proj.weight", "ref_time_in.in_layer.weight"),
    quantize=(
        r"^(content_mode_blocks\.(image|image_cond)|txt_mode_blocks|single_blocks)"
        r"\.\d+\.(qkv|q_proj|k_proj|v_proj|attn_out|mlp_in|mlp_out)\.weight$",
    ),
    keep=(r"(^|\.)(emb_in|txt_in|time_in|ref_time_in|early_stream_modulations|"
          r"single_stream_modulations|final_layer|pe_embedder)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
    notes="Native Flux 3 / Klein, including step-distilled flow-map weights. "
          "Fused or split QKV block linears only; preserve boundary projections, "
          "time/ref-time embeddings, modulation and norms. Not the Flux 2 Klein layout.",
))


_register(FamilyPolicy(
    family="chroma",
    comfyui_classes=("Chroma", "ChromaRadiance"),
    detect_primary=("distilled_guidance_layer.norms.0.weight",),
    detect_hints=("distilled_guidance_layer.0.norms.0.weight", "nerf_blocks.0.norm.weight"),
    quantize=(
        r"(double_blocks|nerf_blocks)\.\d+\.(img_attn|txt_attn)\.(qkv|proj)\.weight$",
        r"(double_blocks|nerf_blocks)\.\d+\.(img_mlp|txt_mlp)\.(0|2|w1|w2)\.weight$",
        r"(single_blocks|nerf_blocks)\.\d+\.(linear1|linear2)\.weight$",
    ),
    keep=(r"(^|\.)(img_in|txt_in|vector_in|guidance_in|final_layer|nerf_final_layer|nerf_embedder|img_in_patch)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))


_register(FamilyPolicy(
    family="mochi",
    comfyui_classes=("GenmoMochi",),
    detect_primary=("t5_yproj.weight",),
    detect_hints=("blocks.0.attn.qkv_x.weight", "x_embedder.proj.weight"),
    quantize=(
        r"^blocks\.\d+\.attn\.(qkv_x|qkv_y|proj_x|proj_y)\.weight$",
        r"^blocks\.\d+\.mlp_(x|y)\.(w1|w2)\.weight$",
    ),
    keep=(r"(^|\.)(patch_embed|t5_yproj|final_layer|cond_embedder|timestep_embedder|pos_embed|mod)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))


_register(FamilyPolicy(
    family="minimax_h3",
    comfyui_classes=("MiniMaxH3",),
    detect_primary=("video_patch_proj.weight", "audio_patch_proj.weight"),
    detect_hints=("blocks.0.attn.qkv_proj.weight", "final_layer.video_out.weight",
                  "adaln_t_table", "rope.inv_freq"),
    quantize=(
        r"blocks\.\d+\.attn\.(qkv_proj|out_proj)\.weight$",
        r"blocks\.\d+\.mlp\.(fc1|fc2)\.weight$",
    ),
    keep=(r"(^|\.)(video_patch_proj|audio_patch_proj|condition_proj|final_layer|"
          r"adaln_proj|adaln_t_table|token_refiner|time_embedder|rope)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="verified",
    notes="Reference family: the Kijai W4A8 test model quantizes exactly "
          "attn.qkv_proj / attn.out_proj / mlp.fc1 / mlp.fc2 per block.",
))


_register(FamilyPolicy(
    family="minimax_music3",
    comfyui_classes=("MiniMaxMusic3",),
    detect_primary=(
        "cond_layer_logits",
        "latent_conditioners.0.weight",
        "diffusion_transformer.transformer.layers.0.self_attn.to_qkv.weight",
    ),
    detect_hints=(
        "diffusion_transformer.transformer.layers.0.ff.ff.0.proj.weight",
        "diffusion_transformer.transformer.layers.0.ff.ff.2.weight",
        "diffusion_transformer.transformer.project_in.weight",
    ),
    quantize=(
        r"diffusion_transformer\.transformer\.layers\.\d+\.self_attn\."
        r"(to_qkv|to_out)\.weight$",
        r"diffusion_transformer\.transformer\.layers\.\d+\.ff\.ff\."
        r"(0\.proj|2)\.weight$",
    ),
    keep=(
        r"(^|\.)(latent_conditioners|timestep_features|to_timestep_embed|"
        r"preprocess_conv|postprocess_conv|project_in|project_out|"
        r"rotary_pos_emb)(\.|$)",
    ),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
    notes="MiniMax Music 3 flow-matching DiT. The policy mirrors the 144 "
          "transformer linears selected by Comfy-Org's reference ConvRot "
          "checkpoint and keeps conditioning, time, input/output and convolution "
          "layers at source precision.",
))


_register(FamilyPolicy(
    family="ltxv",
    comfyui_classes=("LTXV", "LTXAV"),
    detect_primary=("adaln_single.emb.timestep_embedder.linear_1.bias",),
    detect_hints=("transformer_blocks.0.attn2.to_k.weight", "audio_adaln_single.linear.weight"),
    quantize=(
        r"transformer_blocks\.\d+\.attn[12]\.(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"transformer_blocks\.\d+\.ff\.net\.(0|2)\.(proj)?\.?weight$",
        r"transformer_blocks\.\d+\.(attn1|attn2)\.(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"transformer_blocks\.\d+\.ff\.(0|2)\.weight$",
    ),
    keep=(r"(^|\.)(patchify_proj|time_embed|cond_proj|caption_proj|proj_out|adaln_single|"
          r"audio_adaln_single|pos_embed|rope)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))


_register(FamilyPolicy(
    family="ace_step",
    comfyui_classes=("ACEStep", "ACEStep15"),

    # ACE-Step 1.5 / XL native ComfyUI layout.
    # Legacy ACE-Step detection is still handled by the structural detector.
    detect_primary=(
        "encoder.lyric_encoder.layers.0.input_layernorm.weight",
    ),
    detect_hints=(
        "decoder.layers.0.self_attn.q_proj.weight",
        "decoder.layers.0.cross_attn.q_proj.weight",
        "decoder.layers.0.mlp.gate_proj.weight",
        "genre_embedder.weight",
    ),

    quantize=(
        # ACE-Step 1.5 / XL main DiT.
        r"decoder\.layers\.\d+\.(self_attn|cross_attn)\."
        r"(q_proj|k_proj|v_proj|o_proj)\.weight$",

        r"decoder\.layers\.\d+\.mlp\."
        r"(gate_proj|up_proj|down_proj)\.weight$",

        # Internal lyric + timbre conditioning transformers.
        r"encoder\.(lyric_encoder|timbre_encoder)\.layers\.\d+\.self_attn\."
        r"(q_proj|k_proj|v_proj|o_proj)\.weight$",

        r"encoder\.(lyric_encoder|timbre_encoder)\.layers\.\d+\.mlp\."
        r"(gate_proj|up_proj|down_proj)\.weight$",

        # Legacy ACE-Step encoder layout.
        r"encoder\.layers\.\d+\.self_attn\."
        r"(q_proj|k_proj|v_proj|o_proj)\.weight$",

        r"encoder\.layers\.\d+\.mlp\."
        r"(gate_proj|up_proj|down_proj)\.weight$",
    ),

    keep=(
        # Legacy ACE-Step sensitive paths.
        r"(^|\.)(genre_embedder|speaker_embedder|lyric_proj|ssl_|enc|dec)\.",

        # ACE-Step 1.5 conditioning/projector layers.
        r"^encoder\.(text_projector|"
        r"lyric_encoder\.embed_tokens|"
        r"timbre_encoder\.embed_tokens)(\.|$)",

        # Input/output/time projections.
        r"^decoder\.(proj_in|proj_out|condition_embedder|"
        r"time_embed|time_embed_r)(\.|$)",

        # Keep the audio-code path untouched initially.
        r"^(tokenizer|detokenizer)(\.|$)",
    ),

    # IMPORTANT:
    # UNIVERSAL_EXCLUDE contains encoder/decoder subtree exclusions,
    # which would suppress the ACE-Step 1.5 transformer itself.
    # This policy is already a narrow allow-list, so no broad exclude is needed.
    exclude=(),

    runtime_status="experimental",
    notes=(
        "ACE-Step / ACE-Step 1.5 music diffusion. Supports the native "
        "ComfyUI ACE-Step 1.5 Base/Turbo/XL layout, including decoder "
        "self-attention, cross-attention, gated MLPs, and lyric/timbre encoders."
    ),
))


_register(FamilyPolicy(
    family="pixart",
    comfyui_classes=("PixArtAlpha", "PixArtSigma"),
    detect_primary=("t_block.1.weight",),
    detect_hints=("blocks.0.attn.qkv.weight", "x_embedder.proj.weight",
                  "y_embedder.y_embedding", "ar_embedder.mlp.0.weight"),
    quantize=(
        r"blocks\.\d+\.attn\.(qkv|proj)\.weight$",
        r"blocks\.\d+\.cross_attn\.(q_linear|kv_linear|proj)\.weight$",
        r"blocks\.\d+\.mlp\.(fc1|fc2|w1|w2)\.weight$",
    ),
    keep=(r"(^|\.)(x_embedder|y_embedder|t_block|pos_embed|final_layer|csize_embedder|"
          r"ar_embedder|pe_interpolation)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))


_register(FamilyPolicy(
    family="cosmos",
    comfyui_classes=("CosmosT2V", "CosmosI2V"),
    detect_primary=("blocks.block0.blocks.0.block.attn.to_q.0.weight",),
    detect_hints=("x_embedder.proj.1.weight", "adaln_lora"),
    quantize=(
        r"blocks\.block\d+\.blocks\.\d+\.block\.attn\.(to_q|to_k|to_v|to_out)\.\d+\.weight$",
        r"blocks\.block\d+\.blocks\.\d+\.block\.mlp\.(w1|w2|fc1|fc2)\.weight$",
        r"blocks\.block\d+\.blocks\.\d+\.block\.cross_attn\.(to_q|to_k|to_v|to_out)\.\d+\.weight$",
    ),
    keep=(r"(^|\.)(x_embedder|t_embedder|adaln|pos_emb|final_layer|patch_embed|cond_embed|"
          r"cross_attn_norm|norm)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))


_register(FamilyPolicy(
    family="cosmos_predict2",
    comfyui_classes=("CosmosT2IPredict2", "CosmosI2VPredict2"),
    detect_primary=("blocks.0.mlp.layer1.weight",),
    detect_hints=("x_embedder.proj.1.weight",),
    quantize=(
        r"blocks\.\d+\.(self_attn|cross_attn)\."
        r"(q_proj|k_proj|v_proj|output_proj)\.weight$",
        r"blocks\.\d+\.mlp\.(layer1|layer2)\.weight$",
    ),
    keep=(r"(^|\.)(x_embedder|t_embedder|adaln|pos_emb|final_layer)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))


_register(FamilyPolicy(
    family="anima",
    comfyui_classes=("Anima",),
    detect_primary=("llm_adapter.blocks.0.cross_attn.q_proj.weight",),
    detect_hints=("blocks.0.mlp.layer1.weight", "x_embedder.proj.1.weight"),
    quantize=(
        r"blocks\.\d+\.(self_attn|cross_attn)\."
        r"(q_proj|k_proj|v_proj|output_proj)\.weight$",
        r"blocks\.\d+\.mlp\.(layer1|layer2)\.weight$",
    ),
    keep=(r"^llm_adapter\.",
          r"^blocks\.\d+\.adaln_modulation_(self_attn|cross_attn|mlp)\.",
          r"(^|\.)(x_embedder|t_embedder|cond_embed|final_layer)\."),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
    notes="Anima shares its main DiT block layout with Cosmos Predict2 and adds an LLM adapter.",
))


_register(FamilyPolicy(
    family="lumina2",
    comfyui_classes=("Lumina2", "ZImage", "ZImagePixelSpace"),
    # The structural detector requires both caption and noise-refiner keys.
    # A caption projection alone also occurs in incomplete Ming checkpoints.
    detect_primary=(),
    detect_hints=("noise_refiner.0.attention.k_norm.weight", "layers.0.attn.qkv.weight",
                  "layers.0.attention.qkv.weight", "dec_net.cond_embed.weight"),
    quantize=(
        r"layers\.\d+\.attn\.(qkv|o_proj|proj)\.weight$",
        r"layers\.\d+\.mlp\.(w1|w2|fc1|fc2)\.weight$",
        # real Lumina2 / Z-Image naming (comfy/ldm/lumina/model.py):
        # JointTransformerBlock.attention.{qkv,out}, FeedForward.{w1,w2,w3}
        r"(layers|context_refiner|noise_refiner)\.\d+\.attention\.(qkv|out)\.weight$",
        r"(layers|context_refiner|noise_refiner)\.\d+\.feed_forward\.(w1|w2|w3)\.weight$",
    ),
    keep=(r"(^|\.)(cap_embedder|clip_text_pooled_proj|siglip_embedder|x_embedder|t_embedder|"
          r"cond_embed|final_layer|dec_net|pos_embed|adaLN_modulation)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))


_register(FamilyPolicy(
    family="ming_image",
    comfyui_classes=("MingImage",),
    # Design-Layer carries a Ming marker/config and paired learned pad tokens;
    # unmarked Design uses the same block signature without learned padding.
    detect_primary=(),
    detect_hints=("__ming_image__", "x_embedder.weight",
                  "layers.0.attention.to_q.weight",
                  "layers.0.feed_forward.w1.weight"),
    quantize=(
        r"^(layers|context_refiner|noise_refiner)\.\d+\.attention\."
        r"(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"^(layers|context_refiner|noise_refiner)\.\d+\.feed_forward\."
        r"(w1|w2|w3)\.weight$",
    ),
    keep=(r"^(x_embedder|t_embedder|cap_embedder|final_layer|"
          r"all_x_embedder|all_final_layer|siglip_embedder)(\.|$)",
          r"(^|\.)(adaLN_modulation|attention_norm\d*|ffn_norm\d*|"
          r"norm_[qk])(\.|$)"),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
    notes="ComfyUI Ming-Image Design and Design-Layer repacks; block attention "
          "and feed-forward linears only. Design-Layer requires a layer "
          "decomposition workflow, not the bundled Design workflow.",
))


_register(FamilyPolicy(
    family="pixeldit",
    comfyui_classes=("PixelDiTT2I", "PiD"),
    detect_primary=("core.pixel_embedder.proj.weight", "lq_proj.latent_proj.0.weight",
                    "pixel_embedder.proj.weight"),
    detect_hints=("core.patch_blocks.0.attn.qkv_x.weight",
                  "patch_blocks.0.attn.qkv_x.weight"),
    quantize=(
        r"^(core\.|net\.)?patch_blocks\.\d+\.attn\."
        r"(qkv_x|qkv_y|proj_x|proj_y)\.weight$",
        r"^(core\.|net\.)?patch_blocks\.\d+\.mlp_(x|y)\."
        r"(w1|w2|w3)\.weight$",
        r"^(core\.|net\.)?pixel_blocks\.\d+\."
        r"(compress_to_attn|expand_from_attn|attn\.(qkv|proj)|mlp\.(fc1|fc2))\.weight$",
    ),
    keep=(r"(^|\.)(pixel_embedder|s_embedder|y_embedder|t_embedder|"
          r"final_layer|lq_proj)(\.|$)",),
    exclude=(),
    runtime_status="experimental",
))


_register(FamilyPolicy(
    family="iris",
    linear_alignment=(16, 256),
    comfyui_classes=("Iris", "Iris3B"),
    detect_primary=("y_embedder.layer_pool.weight", "s_embedder.proj.weight",
                    "pixel_embedder.proj.weight", "pixel_blocks.0.compress.weight",
                    "blocks.0.attn.q_norm_x.weight", "blocks.0.attn.k_proj_x.weight",
                    "y_embedder.refiner.proj.weight", "y_pos_embedding",
                    "pixel_blocks.0.attn.q_norm.weight"),
    quantize=(
        r"^blocks\.\d+\.attn\.(q_proj_[xy]|k_proj_[xy]|v_proj_[xy]|proj_[xy])\.weight$",
        r"^blocks\.\d+\.(q_proj|k_proj|v_proj|attn_proj)\.weight$",
        r"^blocks\.\d+\.mlp(?:_[xy])?\.(w1|w2|w3|gate_up)\.weight$",
        r"^pixel_blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(fc1|fc2))\.weight$",
    ),
    keep=(r"(^|\.)(s_embedder|pixel_embedder|y_embedder|t_embedder|y_pos_embedding|"
          r"modulation_cores|adaln\w*|attn_gate\w*|compress|expand|final_layer)(\.|$)",),
    exclude=UNIVERSAL_EXCLUDE,
    max_rel_l2=0.03,
    notes="Iris-3B fused/unfused trunk and pixel head. Requires a native Iris ComfyUI loader; "
          "gate/modulation, stage boundaries and text adapter remain dense. No padding.",
))


_register(FamilyPolicy(
    family="kandinsky6",
    comfyui_classes=("Kandinsky6",),
    detect_primary=("audio_embeddings.in_layer.weight", "visual_embeddings.in_layer.weight",
                    "video_time_embeddings.out_layer.weight", "audio_time_embeddings.out_layer.weight",
                    "visual_blocks.0.videoT.feed_forward.in_layer.weight",
                    "visual_blocks.0.audioT.feed_forward.in_layer.weight",
                    "visual_blocks.0.videoT.self_attention.query_norm.weight",
                    "visual_blocks.0.audioT.self_attention.query_norm.weight",
                    "visual_blocks.0.va_modulation.out_layer.weight",
                    "visual_blocks.0.va_cross_attention.to_key.weight", "out_layer.out_layer.weight",
                    "video_text_embeddings.in_layer.weight", "video_pooled_text_embeddings.in_layer.weight"),
    quantize=(
        r"^visual_blocks\.\d+\.(videoT|audioT)\.(self_attention|cross_attention)\."
        r"(to_query|to_key|to_value|out_layer)\.weight$",
        r"^visual_blocks\.\d+\.(videoT|audioT)\.feed_forward\.(in_layer|out_layer)\.weight$",
        r"^visual_blocks\.\d+\.(va_cross_attention|av_cross_attention)\."
        r"(to_query|to_key|to_value|out_layer)\.weight$",
    ),
    keep=(r"^(?!visual_blocks\.)", r"(^|\.)(\w*modulation|\w*norm|\w*gate\w*)(\.|$)"),
    exclude=UNIVERSAL_EXCLUDE[:2],
    max_rel_l2=0.03,
    notes="Kandinsky 6 Lite/Pro native video+audio DiT. Requires the official native ComfyUI extension; "
          "external encoders, embeddings, text towers, modality gates and modulation stay dense. "
          "Diffusers keys require an explicit reviewed key map before native conversion.",
))


_register(FamilyPolicy(
    family="kandinsky6_sr",
    comfyui_classes=("Kandinsky6SR",),
    detect_primary=("pooled_bias", "visual_embeddings.in_layer.weight", "out_layer.out_layer.weight",
                    "visual_transformer_blocks.0.visual_modulation.out_layer.weight",
                    "visual_transformer_blocks.0.self_attention.query_norm.weight",
                    "visual_transformer_blocks.0.self_attention.to_query.weight",
                    "visual_transformer_blocks.0.feed_forward.in_layer.weight"),
    quantize=(
        r"^visual_transformer_blocks\.\d+\.self_attention\.(to_query|to_key|to_value|out_layer)\.weight$",
        r"^visual_transformer_blocks\.\d+\.feed_forward\.(in_layer|out_layer)\.weight$",
    ),
    keep=(r"^(?!visual_transformer_blocks\.)", r"(^|\.)(\w*modulation|\w*norm|\w*gate\w*)(\.|$)"),
    exclude=UNIVERSAL_EXCLUDE[:2],
    max_rel_l2=0.03,
    notes="Kandinsky 6 video SR text-free native DiT only; official native extension required. "
          "Pooled bias, conditioning, embeddings, output heads, KVAE and latent upscaler remain dense.",
))


_register(FamilyPolicy(
    family="wan",
    comfyui_classes=("WAN21_T2V", "WAN21_CausalAR_T2V", "WAN21_I2V", "WAN21_FunControl2V",
                     "WAN21_Camera", "WAN22_Camera", "WAN21_Vace", "WAN21_HuMo",
                     "WAN22_S2V", "WAN22_Animate", "WAN22_T2V", "WAN_Animate2", "WAN21_FlowRVS",
                     "WAN21_SCAIL", "WAN21_SCAIL2", "WAN22_WanDancer"),
    detect_primary=("head.modulation",),
    detect_hints=("blocks.0.self_attn.q.weight", "blocks.0.cross_attn.k.weight",
                  "patch_embedding.weight", "txt_embedding.weight"),
    quantize=(
        r"blocks\.\d+\.(self_attn|cross_attn)\.(q|k|v|o)\.weight$",
        r"blocks\.\d+\.feed_forward\.(w1|w2)\.weight$",
        r"blocks\.\d+\.ffn\.(0|2)\.weight$",
        r"blocks\.\d+\.(self_attn|cross_attn)\.(q_img|k_img|v_img|o_img)\.weight$",
    ),
    keep=(r"(^|\.)(patch_embedding|text_embedding|time_embedding|time_projection|"
          r"final_linear|head|before_proj|after_proj|audio_proj|cond_embedding|"
          r"vace_patch_embedding|control_adapter|img_emb|face_adapter|latent_in|"
          r"patch_embedding_mask|patch_embedding_pose|patch_embedding_global)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))


_register(FamilyPolicy(
    family="hunyuan3d",
    comfyui_classes=("Hunyuan3Dv2", "Hunyuan3Dv2mini"),
    detect_primary=("latent_in.weight",),
    detect_hints=("cond_in.weight", "double_blocks.0.img_attn.qkv.weight"),
    quantize=(
        r"^double_blocks\.\d+\.(img_attn|txt_attn)\.(qkv|proj)\.weight$",
        r"^double_blocks\.\d+\.(img_mlp|txt_mlp)\.(0|2|w1|w2)\.weight$",
        r"^single_blocks\.\d+\.(linear1|linear2)\.weight$",
    ),
    keep=(r"^(latent_in|cond_in|time_in|guidance_in|final_layer)(\.|$)",),
    exclude=(),
    runtime_status="experimental",
))


_register(FamilyPolicy(
    family="hunyuan3d21",
    comfyui_classes=("Hunyuan3Dv2_1",),
    # The structural detector requires both keys, as ComfyUI does.
    detect_primary=(),
    detect_hints=("t_embedder.mlp.2.weight", "blocks.0.attn1.k_norm.weight"),
    quantize=(
        r"^blocks\.\d+\.attn[12]\.(to_q|to_k|to_v|out_proj)\.weight$",
        r"^blocks\.\d+\.mlp\.(fc1|fc2)\.weight$",
        r"^blocks\.\d+\.moe\.(shared_experts|experts\.\d+)\."
        r"net\.(0\.proj|2)\.weight$",
    ),
    keep=(r"^(x_embedder|t_embedder|final_layer)(\.|$)",),
    exclude=(),
    runtime_status="experimental",
))


_register(FamilyPolicy(
    family="triposplat",
    comfyui_classes=("TripoSplat",),
    detect_primary=("cam_out_layer.weight",),
    detect_hints=("repo_layers.0.final_map.weight", "cond_embedder.weight"),
    quantize=(
        r"^(blocks|noise_refiner|context_refiner)\.\d+\.attn\.(qkv|out)\.weight$",
        r"^(blocks|noise_refiner|context_refiner)\.\d+\.mlp\.mlp\.(0|2)\.weight$",
    ),
    keep=(r"(^|\.)(cond_embedder|input_layer|out_layer|cam_out_layer|pos_embed)\.",),
    exclude=(),
    runtime_status="experimental",
))


_register(FamilyPolicy(
    family="hidream",
    comfyui_classes=("HiDream",),
    detect_primary=("caption_projection.0.linear.weight",),
    detect_hints=("x_embedder.proj.weight", "double_stream_blocks.0.block.attn1.to_q.weight"),
    quantize=(
        r"^(double_stream_blocks|single_stream_blocks)\.\d+\.block\.attn1\."
        r"(to_q|to_k|to_v|to_out)(|_t)\.weight$",
        r"^(double_stream_blocks|single_stream_blocks)\.\d+\.block\."
        r"(ff_i\.(shared_experts|experts\.\d+)|ff_t)\.(w1|w2|w3)\.weight$",
    ),
    keep=(r"(^|\.)(x_embedder|t_embedder|caption_projection|time_embed|pos_embed|final_layer|cond_embed)\.",),
    exclude=(),
    runtime_status="experimental",
))


_register(FamilyPolicy(
    family="hidream_o1",
    comfyui_classes=("HiDreamO1",),
    detect_primary=("t_embedder1.mlp.0.weight", "x_embedder.proj1.weight"),
    detect_hints=("language_model.layers.0.self_attn.q_proj.weight",),
    quantize=(
        r"^language_model\.layers\.\d+\.self_attn\."
        r"(q_proj|k_proj|v_proj|o_proj)\.weight$",
        r"^language_model\.layers\.\d+\.mlp\."
        r"(gate_proj|up_proj|down_proj)\.weight$",
    ),
    keep=(r"^(visual|x_embedder|t_embedder1|final_layer2|language_model\.embed_tokens)(\.|$)",),
    exclude=(),
    runtime_status="experimental",
    notes="HiDream-O1 pixel-space transformer: quantize the repeated language-model "
          "blocks; preserve the vision tower and input/output projections.",
))


_register(FamilyPolicy(
    family="seedvr2",
    comfyui_classes=("SeedVR2",),
    detect_primary=("blocks.35.mlp.vid.proj_out.weight",
                    "blocks.35.mlp.all.proj_in_gate.weight",
                    "blocks.31.mlp.all.proj_in_gate.weight"),
    detect_hints=("blocks.0.attn.proj_qkv.all.weight", "vid_in.proj.weight"),
    quantize=(
        r"^blocks\.\d+\.attn\.(proj_qkv|proj_out)\.(all|vid|txt)\.weight$",
        r"^blocks\.\d+\.mlp\.(all|vid|txt)\."
        r"(proj_in|proj_in_gate|proj_out)\.weight$",
    ),
    keep=(r"^(vid_in|txt_in|emb_in|vid_out)(\.|$)",),
    exclude=(),
    runtime_status="experimental",
    notes="SeedVR2 NaDiT shared and split video/text attention and MLP weights.",
))


_register(FamilyPolicy(
    family="omnigen2",
    comfyui_classes=("Omnigen2",),
    detect_primary=("time_caption_embed.timestep_embedder.linear_1.bias",
                    "layers.0.attn.to_q.weight"),
    detect_hints=("layers.0.feed_forward.linear_1.weight", "context_refiner.0.attn.to_q.weight",
                  "ref_image_patch_embedder.weight", "x_embedder.weight"),
    quantize=(
        r"layers\.\d+\.attn\.(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"layers\.\d+\.feed_forward\.(linear_1|linear_2|linear_3)\.weight$",
        r"(context_refiner|noise_refiner|ref_image_refiner)\.\d+\.attn\.(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"(context_refiner|noise_refiner|ref_image_refiner)\.\d+\.feed_forward\.(linear_1|linear_2|linear_3)\.weight$",
    ),
    keep=(r"(^|\.)(norm_out|image_index_embedding)\.", r"norm\d+\.linear\.(weight|bias)$"),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))


_register(FamilyPolicy(
    family="boogu",
    comfyui_classes=("Boogu",),
    detect_primary=("double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight",
                    "double_stream_layers.0.img_self_attn.to_q.weight",
                    "double_stream_layers.0.img_feed_forward.linear_1.weight"),
    detect_hints=("single_stream_layers.0.attn.to_q.weight",
                  "single_stream_layers.0.feed_forward.linear_1.weight",
                  "context_refiner.0.attn.to_q.weight",
                  "ref_image_patch_embedder.weight", "x_embedder.weight"),
    quantize=(
        r"double_stream_layers\.\d+\.(img_self_attn|img_instruct_attn)\.(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"double_stream_layers\.\d+\.img_instruct_attn\.processor\.(img_to_q|img_to_k|img_to_v|img_out|instruct_to_q|instruct_to_k|instruct_to_v|instruct_out)\.weight$",
        r"double_stream_layers\.\d+\.(img_feed_forward|instruct_feed_forward)\.(linear_1|linear_2|linear_3)\.weight$",
        r"single_stream_layers\.\d+\.attn\.(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"single_stream_layers\.\d+\.feed_forward\.(linear_1|linear_2|linear_3)\.weight$",
        r"(context_refiner|noise_refiner|ref_image_refiner)\.\d+\.attn\.(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"(context_refiner|noise_refiner|ref_image_refiner)\.\d+\.feed_forward\.(linear_1|linear_2|linear_3)\.weight$",
    ),
    keep=(r"(^|\.)(norm_out|image_index_embedding|time_caption_embed)\.", r"norm\d+\.linear\.(weight|bias)$"),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
    notes="Real Boogu-Image-0.1 naming (double/single_stream_layers, img_instruct_attn.processor).",
))


_register(FamilyPolicy(
    family="lens",
    comfyui_classes=("Lens",),
    detect_primary=("transformer_blocks.0.attn.norm_added_q.weight",),
    detect_hints=("transformer_blocks.0.img_mlp.w1.weight", "img_in.weight"),
    quantize=(
        r"transformer_blocks\.\d+\.attn\."
        r"(img_qkv|txt_qkv|to_out\.0|to_add_out)\.weight$",
        r"transformer_blocks\.\d+\.(img_mlp|txt_mlp)\.(w1|w2|w3)\.weight$",
    ),
    keep=(r"(^|\.)(img_in|txt_in|proj_out|txt_norm|time_text_embed|final_layer)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))


_register(FamilyPolicy(
    family="mage_flow",
    comfyui_classes=("MageFlow",),
    detect_primary=(),
    detect_hints=("proj_out.weight", "transformer_blocks.0.attn.to_q.weight"),
    quantize=(
        r"transformer_blocks\.\d+\.attn\.(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"transformer_blocks\.\d+\.(img_mlp|txt_mlp)\."
        r"net\.(0\.proj|2)\.weight$",
    ),
    keep=(r"(^|\.)(img_in|txt_in|proj_out|txt_norm|time_text_embed|final_layer)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
    notes="Disambiguated from qwen_image by txt_norm dim 2560 / proj_out 128.",
))


_register(FamilyPolicy(
    family="qwen_image",
    comfyui_classes=("QwenImage",),
    detect_primary=(),
    detect_hints=("proj_out.weight", "transformer_blocks.0.attn.to_q.weight",
                  "img_in.weight", "time_text_embed.addition_t_embedding.weight"),
    quantize=(
        r"transformer_blocks\.\d+\.attn\.(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"transformer_blocks\.\d+\.(img_mlp|txt_mlp)\."
        r"net\.(0\.proj|2)\.weight$",
    ),
    keep=(r"(^|\.)(img_in|txt_in|proj_out|txt_norm|time_text_embed|final_layer|"
          r"time_caption_embed|add_k_proj|add_q_proj|add_v_proj|to_add_out)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))


_register(FamilyPolicy(
    family="qwen_image21",
    comfyui_classes=("QwenImage21",),
    # Detection requires the combined signature below, not one isolated key.
    detect_primary=(),
    detect_hints=("img_in.weight", "proj_out.weight",
                  "transformer_blocks.0.attn.to_q.weight",
                  "transformer_blocks.0.img_mlp.proj.weight",
                  "transformer_blocks.0.img_mlp.gate_up.weight"),
    quantize=(
        r"transformer_blocks\.\d+\.attn\.(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"transformer_blocks\.\d+\.img_mlp\."
        r"(gate_up|gate_layer|proj|out)\.weight$",
    ),
    keep=(r"^(img_in|txt_in|time_text_embed|modulation|norm_out|proj_out)(\.|$)",
          r"^transformer_blocks\.\d+\.attn\.norm_(q|k)(\.|$)"),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
    notes="Qwen-Image 2.1 single-stream DiT; block attention and fused or "
          "unfused SwiGLU linears only. Keep inputs, shared modulation, "
          "normalization, timestep projection, and output at source precision.",
))


_register(FamilyPolicy(
    family="ideogram4",
    comfyui_classes=("Ideogram4",),
    detect_primary=("embed_image_indicator.weight",),
    detect_hints=("input_proj.weight", "layers.0.attention.qkv.weight"),
    quantize=(
        r"layers\.\d+\.attention\.(qkv|o)\.weight$",
        r"layers\.\d+\.feed_forward\.(w1|w2|w3)\.weight$",
    ),
    keep=(r"(^|\.)(input_proj|embed_image_indicator|adaln_proj|adaln_modulation|"
          r"llm_cond_proj|time_embed|pos_embed|final_layer)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))


_register(FamilyPolicy(
    family="krea2",
    comfyui_classes=("Krea2",),
    detect_primary=("txtfusion.projector.weight",),
    detect_hints=("txtfusion.layerwise_blocks.0.prenorm.scale",
                  "blocks.0.attn.wq.weight"),
    quantize=(
        r"blocks\.\d+\.attn\.(wq|wk|wv|wo|gate)\.weight$",
        r"blocks\.\d+\.mlp\.(gate|up|down)\.weight$",
        r"txtfusion\.(layerwise_blocks|refiner_blocks)\.\d+\.(attn|mlp)\.\w+\.weight$",
    ),
    keep=(r"(^|\.)(first|projector|txtfusion\.projector|input_proj|time_embed|"
          r"pos_embed|final_layer|adaln)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
    notes="Real Kroma v0.2 (Krea2 fine-tune) naming: blocks.N.attn.wq/wk/wv/wo/gate, "
          "blocks.N.mlp.gate/up/down, txtfusion layerwise+refiner blocks. Verified "
          "against the published kroma-v0.2-turbo checkpoint header (K 6144/16384/"
          "2560/6912 are all ConvRot-256 compatible).",
))


_register(FamilyPolicy(
    family="kandinsky5",
    comfyui_classes=("Kandinsky5", "Kandinsky5Image"),
    detect_primary=("visual_transformer_blocks.0.cross_attention.key_norm.weight",),
    detect_hints=("visual_embeddings.in_layer.weight", "text_transformer_blocks.0.self_attention.q_norm.weight"),
    quantize=(
        r"^(visual|text)_transformer_blocks\.\d+\."
        r"(self_attention|cross_attention)\.(to_query|to_key|to_value|out_layer)\.weight$",
        r"^(visual|text)_transformer_blocks\.\d+\."
        r"feed_forward\.(in_layer|out_layer)\.weight$",
    ),
    keep=(r"^(visual_embeddings|text_embeddings|pooled_text_embeddings|"
          r"time_embeddings|out_layer|rope_embedder_1d|rope_embedder_3d)(\.|$)",),
    exclude=(),
    runtime_status="experimental",
))


_register(FamilyPolicy(
    family="cogvideox",
    comfyui_classes=("CogVideoX_T2V", "CogVideoX_I2V", "CogVideoX_Inpaint"),
    detect_primary=("blocks.0.norm1.linear.weight",),
    detect_hints=("patch_embed.proj.weight", "blocks.0.q.weight"),
    quantize=(
        r"^blocks\.\d+\.(q|k|v|attn_out|ff_proj|ff_out)\.weight$",
    ),
    keep=(r"(^|\.)(patch_embed|time_embed|text_proj|proj_out|final_layer|ofs_embedding|"
          r"norm1\.linear|cond_embed)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))


_register(FamilyPolicy(
    family="ernie_image",
    comfyui_classes=("ErnieImage",),
    detect_primary=("layers.0.mlp.linear_fc2.weight",),
    detect_hints=("text_proj.weight", "layers.0.self_attention.to_q.weight"),
    quantize=(
        r"^layers\.\d+\.self_attention\.(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"^layers\.\d+\.mlp\.(gate_proj|up_proj|linear_fc2)\.weight$",
    ),
    keep=(r"(^|\.)(text_proj|visual_embeddings|time_embeddings|patch_embed|final_linear|pos_embed)\.",),
    exclude=(),
    runtime_status="experimental",
))


_register(_sd_unet_policy(
    "sd15", ("SD15", "SD15_instructpix2pix", "Stable_Zero123"),
    "SD1.5 UNet (context_dim 768). Attention q/k/v/out are 1x1 convs in the "
    "non-linear-attention layout; only feed-forward linears plus linear-attention "
    "variants are quantized."))


_register(_sd_unet_policy(
    "sd20", ("SD20", "SD21UnclipL", "SD21UnclipH", "SD_X4Upscaler", "LotusD"),
    "SD2.x UNet (context_dim 1024, linear attention)."))


_register(_sd_unet_policy(
    "sdxl", ("SDXL", "SSD1B", "Segmind_Vega", "KOALA_700M", "KOALA_1B",
             "SDXL_instructpix2pix"),
    "SDXL / SSD-1B / Segmind-Vega / KOALA UNets (context_dim 2048)."))


_register(_sd_unet_policy(
    "sdxl_refiner", ("SDXLRefiner",),
    "SDXL refiner UNet (model_channels 384, context_dim 1280)."))


_register(_sd_unet_policy(
    "svd", ("SVD_img2vid", "SV3D_u", "SV3D_p"),
    "SVD / SV3D video UNets (temporal attention)."))


_register(FamilyPolicy(
    family="joyimage",
    comfyui_classes=("JoyImage",),
    detect_primary=("double_blocks.0.attn.img_attn_qkv.weight",
                    "double_blocks.0.attn.img_attn_q_norm.weight"),
    detect_hints=("condition_embedder.time_embedder.linear_1.weight", "img_in.weight"),
    quantize=(
        r"double_blocks\.\d+\.attn\.(img_attn_qkv|img_attn_proj|txt_attn_qkv|txt_attn_proj)\.weight$",
        r"double_blocks\.\d+\.(img_mlp|txt_mlp)\."
        r"net\.(0\.proj|2)\.weight$",
    ),
    keep=(r"(^|\.)(img_in|txt_in|condition_embedder|time_embedder|proj_out|"
          r"final_layer|pos_embed|adaln)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))


_register(FamilyPolicy(
    family="trellis2", comfyui_classes=("Trellis2",),
    detect_primary=("img2shape.t_embedder.mlp.0.weight",
                    "shape2txt.t_embedder.mlp.0.weight"),
    detect_hints=("img2shape.blocks.0.self_attn.to_qkv.weight",),
    quantize=(
        r"^(img2shape|img2shape_512|shape2txt|structure_model)\.blocks\.\d+\."
        r"(self_attn\.(to_qkv|to_out)|cross_attn\.(to_q|to_kv|to_out)|mlp\.\d+)\.weight$",
    ),
    keep=(r"(^|\.)(t_embedder|input_layer|output_layer|adaLN_modulation|"
          r"conditioner|vae|proj_linear)(\.|$)",),
    exclude=UNIVERSAL_EXCLUDE, runtime_status="experimental",
    notes="Trellis2 transformer blocks; VAE and conditioning stay at source precision.",
))


_register(FamilyPolicy(
    family="sensenova_u15", comfyui_classes=("SenseNovaU15",),
    detect_primary=("fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight",),
    detect_hints=("language_model.model.layers.0.self_attn.q_proj_mot_gen.weight",),
    quantize=(
        r"^language_model\.model\.layers\.\d+\.self_attn\."
        r"(q_proj|k_proj|v_proj|o_proj)(_mot_gen)?\.weight$",
        r"^language_model\.model\.layers\.\d+\.mlp\."
        r"(gate_proj|up_proj|down_proj)(_mot_gen)?\.weight$",
    ),
    keep=(r"^(fm_modules|vae|text_encoders|language_model\.lm_head)(\.|$)",),
    exclude=UNIVERSAL_EXCLUDE, runtime_status="experimental",
    notes="SenseNova language backbone only; vision and output modules stay unchanged.",
))


_register(FamilyPolicy(
    family="yue2", comfyui_classes=("YuE2",),
    detect_primary=("vae2llm.weight", "llm2vae.weight"),
    detect_hints=("latent_pos_embed.pe", "model.layers.0.self_attn.qkv_proj.weight"),
    quantize=(
        r"^(model\.)?layers\.\d+\.self_attn\.(qkv_proj|o_proj)\.weight$",
        r"^(model\.)?layers\.\d+\.mlp\.(gate_up_proj|down_proj)\.weight$",
    ),
    keep=(r"^(vae2llm|llm2vae|time_embedder|latent_pos_embed|"
          r"text_encoders|vae|model\.embed_tokens|lm_head)(\.|$)",),
    exclude=UNIVERSAL_EXCLUDE, runtime_status="experimental",
    notes="YuE2 acoustic transformer; text encoder uses a separate policy.",
))


for _fam, _cls, _signature in (
        ("rt_detr_v4", ("RT_DETR_v4",), "encoder.pan_blocks.1.cv4.conv.weight"),
        ("depth_anything3", ("DepthAnything3",), "backbone.embeddings.patch_embeddings.projection.weight"),
        ("sam3", ("SAM3", "SAM31"), "detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight")):
    _register(FamilyPolicy(
        family=_fam, comfyui_classes=_cls,
        detect_primary=(_signature,),
        detect_hints=(),
        quantize=(), keep=(), exclude=UNIVERSAL_EXCLUDE,
        runtime_status="unsupported",
        notes="Perception model: ComfyUI loads it through its own node, not through "
              "the mixed-precision (quantized) loader; W4A8 output would not "
              "be consumable. Conversion refused unless --architecture forces it.",
    ))


TEXT_ENCODER_EXCLUDE = (
    r"(^|\.)(embeddings?|embed_tokens|token_embedding|word_embeddings|"
    r"position_embeddings|token_type_embeddings|shared)(\.|$)",
    r"(^|\.)(norm|layer_norm|layernorm|rms_norm|final_layer_norm|"
    r"input_layernorm|post_attention_layernorm|ln_final|ln_\d+|"
    r"q_norm|k_norm)\.(weight|bias)$",
    r"(^|\.)(lm_head|output_head|classifier|score|pooler|text_projection|"
    r"projection_head|language_model_head)(\.|$)",
    r"(^|\.)(visual|vision_model|vision_tower|image_encoder|audio|"
    r"multi_modal_projector|multimodal_projector)(\.|$)",
)


TEXT_TOKEN_EMBEDDING = re.compile(
    r"(^|\.)(embed_tokens|token_embedding|word_embeddings|shared)\.weight$")


TEXT_LLM_QUANTIZE = (
    r"(^|\.)(model(\.language_model)?|language_model\.model|transformer)?\."
    r"?layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj|out_proj)\.weight$",
    r"(^|\.)(model(\.language_model)?|language_model\.model|transformer)?\."
    r"?layers\.\d+\.linear_attn\."
    r"(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)\.weight$",
    r"(^|\.)(model(\.language_model)?|language_model\.model|transformer)?\."
    r"?layers\.\d+\.mlp\."
    r"(gate_proj|up_proj|down_proj|fc1|fc2|w1|w2|w3)\.weight$",
)


TEXT_YUE2_QUANTIZE = (
    r"(^|\.)model\.layers\.\d+\.self_attn\.(qkv_proj|o_proj)\.weight$",
    r"(^|\.)model\.layers\.\d+\.mlp\.(gate_up_proj|down_proj)\.weight$",
)


TEXT_JINA_CLIP2_QUANTIZE = (
    r"^model\.encoder\.layers\.\d+\.mixer\.(Wqkv|out_proj)\.weight$",
    r"^model\.encoder\.layers\.\d+\.mlp\.(fc1|fc2)\.weight$",
)


TEXT_T5GEMMA_QUANTIZE = (
    r"^model\.encoder\.layers\.\d+\.self_attn\."
    r"(q_proj|k_proj|v_proj|o_proj)\.weight$",
    r"^model\.encoder\.layers\.\d+\.mlp\."
    r"(gate_proj|up_proj|down_proj)\.weight$",
)


TEXT_T5_QUANTIZE = (
    r"(^|\.)(encoder|decoder)\.block\.\d+\.layer\.\d+\."
    r"(SelfAttention|EncDecAttention)\.(q|k|v|o)\.weight$",
    r"(^|\.)(encoder|decoder)\.block\.\d+\.layer\.\d+\."
    r"DenseReluDense\.(wi|wi_0|wi_1|wo)\.weight$",
)


TEXT_CLIP_QUANTIZE = (
    r"(^|\.)text_model\.encoder\.layers\.\d+\.self_attn\."
    r"(q_proj|k_proj|v_proj|out_proj)\.weight$",
    r"(^|\.)text_model\.encoder\.layers\.\d+\.mlp\.(fc1|fc2)\.weight$",
)


TEXT_BERT_QUANTIZE = (
    r"(^|\.)encoder\.layer\.\d+\.attention\.self\."
    r"(query|key|value)\.weight$",
    r"(^|\.)encoder\.layer\.\d+\.attention\.output\.dense\.weight$",
    r"(^|\.)encoder\.layer\.\d+\.(intermediate|output)\.dense\.weight$",
)


MINIMAX_MUSIC3_TEXT_QUANTIZE = (
    r"^model\.layers\.\d+\.self_attn\."
    r"(qkv_proj|q_proj|k_proj|v_proj|o_proj)\.weight$",
    r"^model\.layers\.\d+\.mlp\."
    r"(gate_up_proj|gate_proj|up_proj|down_proj)\.weight$",
    r"^model\.audio_decoder\.layers\.\d+\.self_attn\."
    r"(qkv_proj|q_proj|k_proj|v_proj|o_proj)\.weight$",
    r"^model\.audio_decoder\.layers\.\d+\.mlp\."
    r"(gate_up_proj|gate_proj|up_proj|down_proj)\.weight$",
)


_register(FamilyPolicy(
    family="minimax_music3_text_encoder",
    comfyui_classes=("MiniMaxMusic3TEModel", "MiniMaxMusic3TextEncoder"),
    detect_primary=(
        "audio_decoder.projection.weight",
        "audio_extra_embedding.weight",
    ),
    detect_hints=(
        "layers.0.self_attn.qkv_proj.weight",
        "layers.0.self_attn.q_proj.weight",
        "audio_decoder.layers.0.self_attn.qkv_proj.weight",
        "tokenizer_json",
    ),
    quantize=MINIMAX_MUSIC3_TEXT_QUANTIZE,
    keep=(
        r"^model\.(embed_tokens|embed_tokens_prefill|embed_tokens_audio|"
        r"audio_extra_embedding|lm_head|lm_head_pruned)(\.|$)",
        r"^model\.audio_decoder\.(audio_heads|pos_embedding|projection)(\.|$)",
    ),
    exclude=TEXT_ENCODER_EXCLUDE,
    runtime_status="experimental",
    notes="MiniMax Music 3 Qwen3 autoregressive conditioner plus RVQ depth "
          "decoder. Supports the full split-projection checkpoint and the "
          "pruned merged-QKV/merged-MLP checkpoint; embeddings, norms, decoder "
          "projection/audio heads and LM heads remain at source precision.",
    component=COMPONENT_TEXT_ENCODER,
))


_register(FamilyPolicy(
    family="ming_image_text_encoder",
    comfyui_classes=("MingImageTEModel", "MingImageEncoder"),
    detect_primary=(),
    detect_hints=("thinker.norm.weight", "thinker.layers.0.attention.query_key_value.weight",
                  "connector.layers.0.self_attn.q_proj.weight", "tokenizer_json"),
    quantize=(
        r"^thinker\.layers\.\d+\.attention\."
        r"(query_key_value|dense)\.weight$",
        r"^thinker\.layers\.\d+\.mlp\."
        r"(gate_up_proj|gate_proj|up_proj|down_proj)\.weight$",
        r"^thinker\.layers\.\d+\.mlp\.shared_experts\."
        r"(gate_up_proj|gate_proj|up_proj|down_proj)\.weight$",
        r"^connector\.layers\.\d+\.self_attn\."
        r"(q_proj|k_proj|v_proj|o_proj)\.weight$",
        r"^connector\.layers\.\d+\.mlp\."
        r"(gate_up_proj|gate_proj|up_proj|down_proj)\.weight$",
    ),
    keep=(r"^(thinker\.embed_tokens|thinker\.norm|vision|linear_proj|"
          r"query_tokens|proj_in|proj_out|proj_directvlm|connector\.norm)(\.|$)",
          r"^thinker\.layers\.\d+\.mlp\.(gate|image_gate|experts)(\.|$)"),
    exclude=TEXT_ENCODER_EXCLUDE,
    runtime_status="experimental",
    notes="ComfyUI Ming-Image Ling Mini conditioner. Quantizes dense thinker "
          "and connector linears; keeps vision, embeddings, MoE routing and "
          "expert banks, norms, and output projections at source precision.",
    component=COMPONENT_TEXT_ENCODER,
))


_register(FamilyPolicy(
    family="text_encoder_llm",
    comfyui_classes=("QwenTextEncoder", "LlamaTextEncoder", "GemmaTextEncoder"),
    detect_primary=("layers.0.self_attn.q_proj.weight",),
    detect_hints=("layers.0.mlp.gate_proj.weight",
                  "layers.0.self_attn.o_proj.weight"),
    quantize=TEXT_LLM_QUANTIZE,
    exclude=TEXT_ENCODER_EXCLUDE,
    runtime_status="verified",
    notes="Decoder-only transformer text encoders (Qwen/Llama/Gemma/Mistral naming).",
    component=COMPONENT_TEXT_ENCODER,
))


_register(FamilyPolicy(
    family="text_encoder_qwen35",
    comfyui_classes=("Qwen35TEModel",),
    detect_primary=("layers.0.linear_attn.A_log",),
    detect_hints=("layers.0.linear_attn.in_proj_qkv.weight",),
    quantize=TEXT_LLM_QUANTIZE, exclude=TEXT_ENCODER_EXCLUDE,
    runtime_status="experimental", component=COMPONENT_TEXT_ENCODER,
    notes="Qwen3.5 hybrid full and linear attention; visual tower stays unchanged.",
))


_register(FamilyPolicy(
    family="text_encoder_jina_clip2",
    comfyui_classes=("JinaClip2TextModelWrapper",),
    detect_primary=("encoder.layers.0.mixer.Wqkv.weight",),
    detect_hints=("encoder.layers.0.mlp.fc1.weight",),
    quantize=TEXT_JINA_CLIP2_QUANTIZE, exclude=TEXT_ENCODER_EXCLUDE,
    runtime_status="experimental", component=COMPONENT_TEXT_ENCODER,
    notes="JinaCLIP2 text tower only.",
))


_register(FamilyPolicy(
    family="text_encoder_t5gemma",
    comfyui_classes=("T5GemmaModel",),
    detect_primary=("encoder.layers.0.pre_self_attn_layernorm.weight",),
    detect_hints=("encoder.layers.0.self_attn.q_proj.weight",),
    quantize=TEXT_T5GEMMA_QUANTIZE, exclude=TEXT_ENCODER_EXCLUDE,
    runtime_status="experimental", component=COMPONENT_TEXT_ENCODER,
    notes="T5Gemma encoder stack with Gemma-style projections.",
))


_register(FamilyPolicy(
    family="text_encoder_yue2",
    comfyui_classes=("YuE2TextEncoder",),
    detect_primary=("layers.0.self_attn.qkv_proj.weight",),
    detect_hints=("layers.0.mlp.gate_up_proj.weight", "yue2_tokenizer_json"),
    quantize=TEXT_YUE2_QUANTIZE, exclude=TEXT_ENCODER_EXCLUDE,
    runtime_status="experimental", component=COMPONENT_TEXT_ENCODER,
    notes="YuE2 fused-projection text encoder, separate from acoustic diffusion.",
))


_register(FamilyPolicy(
    family="text_encoder_t5",
    comfyui_classes=("T5", "T5XXL", "UMT5"),
    detect_primary=("encoder.block.0.layer.0.SelfAttention.q.weight",),
    detect_hints=("encoder.block.0.layer.1.DenseReluDense.wi.weight",
                  "encoder.block.0.layer.1.DenseReluDense.wi_0.weight"),
    quantize=TEXT_T5_QUANTIZE,
    exclude=TEXT_ENCODER_EXCLUDE,
    runtime_status="verified",
    notes="T5/UMT5 encoder and encoder-decoder text models.",
    component=COMPONENT_TEXT_ENCODER,
))


_register(FamilyPolicy(
    family="text_encoder_clip",
    comfyui_classes=("CLIPTextModel", "CLIPTextModelWithProjection"),
    detect_primary=("text_model.encoder.layers.0.self_attn.q_proj.weight",),
    detect_hints=("text_model.encoder.layers.0.mlp.fc1.weight",),
    quantize=TEXT_CLIP_QUANTIZE,
    exclude=TEXT_ENCODER_EXCLUDE,
    runtime_status="verified",
    notes="Hugging Face CLIP text towers with split attention projections.",
    component=COMPONENT_TEXT_ENCODER,
))


_register(FamilyPolicy(
    family="text_encoder_bert",
    comfyui_classes=("BertModel", "JinaBertModel"),
    detect_primary=("encoder.layer.0.attention.self.query.weight",),
    detect_hints=("encoder.layer.0.intermediate.dense.weight",),
    quantize=TEXT_BERT_QUANTIZE,
    exclude=TEXT_ENCODER_EXCLUDE,
    runtime_status="experimental",
    notes="BERT/Jina-style encoder text models.",
    component=COMPONENT_TEXT_ENCODER,
))


for _base, _variant, _classes in (
        ("lumina2", "z_image", ("ZImage",)),
        ("lumina2", "z_image_pixel_space", ("ZImagePixelSpace",)),
        ("hydit", "hydit1", ("HunyuanDiT1",)),
        ("chroma", "chroma_radiance", ("ChromaRadiance",)),
        ("ltxv", "ltxav", ("LTXAV",)),
        ("pixart", "pixart_sigma", ("PixArtSigma",)),
        ("pixeldit", "pid", ("PiD",)),
        ("sam3", "sam31", ("SAM31",))):
    _parent = REGISTRY[_base]
    _register(dataclasses.replace(_parent, family=_variant, comfyui_classes=_classes,
                                  detect_primary=(), detect_hints=()))
    REGISTRY[_base] = dataclasses.replace(
        _parent, comfyui_classes=tuple(c for c in _parent.comfyui_classes if c not in _classes))


REGISTRY["ltxav"] = dataclasses.replace(REGISTRY["ltxav"], quantize=REGISTRY["ltxv"].quantize + (
    r"transformer_blocks\.\d+\.(audio_attn[12]|audio_to_video_attn|video_to_audio_attn)\."
    r"(to_q|to_k|to_v|to_out\.0)\.weight$",
    r"transformer_blocks\.\d+\.audio_ff\.net\.(0\.proj|2)\.weight$",
), keep=REGISTRY["ltxv"].keep + (
    r"(^|\.)(audio_patchify_proj|audio_caption_projection|audio_proj_out|audio_scale_shift_table)(\.|$)",
))


REGISTRY["ltxv"] = dataclasses.replace(
    REGISTRY["ltxv"], detect_hints=("transformer_blocks.0.attn2.to_k.weight",))


REGISTRY["chroma"] = dataclasses.replace(REGISTRY["chroma"], detect_hints=(
    "distilled_guidance_layer.0.norms.0.weight",))


REGISTRY["lumina2"] = dataclasses.replace(REGISTRY["lumina2"], detect_hints=(
    "noise_refiner.0.attention.k_norm.weight", "layers.0.attention.qkv.weight"))


REGISTRY["pixeldit"] = dataclasses.replace(REGISTRY["pixeldit"], detect_primary=(
    "core.pixel_embedder.proj.weight", "pixel_embedder.proj.weight"))


REGISTRY["sam3"] = dataclasses.replace(REGISTRY["sam3"], detect_primary=(), detect_hints=(
    "detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight",
    "detector.transformer.decoder.query_embed.weight"))


TEXT_ENCODER_FAMILIES = tuple(
    family for family in REGISTRY_ORDER
    if REGISTRY[family].component == COMPONENT_TEXT_ENCODER)


def family_names() -> List[str]:
    return list(REGISTRY_ORDER)


def get_family(name: str) -> FamilyPolicy:
    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    for family, policy in REGISTRY.items():
        aliases = (family,) + policy.comfyui_classes
        if any(re.sub(r"[^a-z0-9]", "", alias.lower()) == normalized
               for alias in aliases):
            return policy
    raise UnknownArchitectureError(
        f"unknown architecture {name!r}; use --list-architectures")
