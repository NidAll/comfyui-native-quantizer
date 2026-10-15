# Flux 3 Klein quantization

The experimental `flux3` policy targets the native ComfyUI **Flux 3 / Klein**
layout, including step-distilled flow-map checkpoints. It is separate from
`flux2`: Flux 2 Klein uses a different block layout. T2I and reference-image
editing use the same diffusion weights and policy.

Start from original floating-point diffusion weights. Already quantized
checkpoints are not a source for another quantization pass. Qwen3-VL text
encoders use the separate text-encoder policies; this policy does not convert
the VAE or change ComfyUI sampling settings.

## Detection and tensor selection

Detection requires the complete native names
`content_mode_blocks.image.0.attn_out.weight` and
`single_stream_modulations.txt.lin.weight`, with coherent square hidden-width
and three-times-hidden modulation shapes. Bare checkpoints and supported model
prefixes are recognized; partial signatures and conflicting architecture
evidence are rejected. The detector result records hidden width, fused/split QKV layout
and whether the optional `ref_time_in` flow-map embedding is present.

Eligible 2D block linears are `qkv` or separate `q_proj`/`k_proj`/`v_proj`,
`attn_out`, `mlp_in` and `mlp_out` under:

- `content_mode_blocks.image.N`
- `content_mode_blocks.image_cond.N`
- `txt_mode_blocks.N`
- `single_blocks.N`

Input embeddings (`emb_in`, `txt_in`), time/ref-time embeddings, early/single
stream modulation, Q/K norms, positional components and final output layers
stay at source precision. Shape/alignment and reconstruction gates can preserve
additional weights. No padding or QKV fusion is introduced by conversion.

The policy reuses existing native codecs, optional MSE, mixed planning and
metadata. ConvRot requires its existing input alignment; W6A8 and other formats
retain their existing consumer/Kitchen requirements. A policy listing is not a
guarantee that every format accelerates inference on a particular device.

## Use

```bash
comfyquant --no-config /path/to/flux3_klein_bf16.safetensors --inspect
comfyquant --no-config /path/to/flux3_klein_bf16.safetensors \
  --format int8-convrot --dry-run
comfyquant --no-config /path/to/flux3_klein_bf16.safetensors \
  --format int8-convrot --device cpu --target-runtime nvidia \
  --max-memory 512M --validate \
  --output /path/to/flux3_klein_int8_convrot.safetensors
comfyquant --no-config --verify-output /path/to/flux3_klein_int8_convrot.safetensors
```

Conversion device and target inference runtime are separate choices. Add
`--optimizer mse` for optional native-grid optimization; see
[MSE optimization](../mse-optimization.md). Use `--architecture flux3` only after
checking a checkpoint's native layout. It does not translate arbitrary
Diffusers names into the ComfyUI layout.

## Consumer and evidence

Requires a ComfyUI installation with `supported_models.Flux3`, native
`comfy/ldm/flux3/model.py` and the corresponding quantized operations. The loader
keeps these projection names; it supports fused or split QKV and optional
flow-map embeddings. Reference-image and text blocks are part of the diffusion
model even when their conditioning is prepared before denoising.

Reviewed against ComfyUI revision `29e28e2b488e7739c351c331d73da25aa435da5c`:
[detector](https://github.com/Comfy-Org/ComfyUI/blob/29e28e2b488e7739c351c331d73da25aa435da5c/comfy/model_detection.py),
[native model](https://github.com/Comfy-Org/ComfyUI/blob/29e28e2b488e7739c351c331d73da25aa435da5c/comfy/ldm/flux3/model.py),
[supported model](https://github.com/Comfy-Org/ComfyUI/blob/29e28e2b488e7739c351c331d73da25aa435da5c/comfy/supported_models.py).

Regression coverage uses independent reduced-shape layouts for fused/split QKV,
flow-map presence, supported prefixes, partial/malformed signatures, Flux 2
separation and protected tensors. Bounded conversion checks native serialization,
source-free verification and precision preservation. Configured consumer tests
load quantized projections into actual native Flux 3 blocks after ComfyUI's
metadata/prefix handling. These checks do not establish full-model image quality,
generation, VRAM use, inference speed or LoRA compatibility.
