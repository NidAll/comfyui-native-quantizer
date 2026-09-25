# Ming-Image support

The `ming_image` and `ming_image_text_encoder` policies target the native
ComfyUI single-file checkpoints distributed in
[Comfy-Org/Ming-Image](https://huggingface.co/Comfy-Org/Ming-Image). They cover
both Ming-Image-0.1-Design and Design-Layer repacks. The original
[inclusionAI release](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design)
uses a component directory with sharded weights; use a ComfyUI repack for a
native ComfyUI output.

Preview each local checkpoint before conversion:

```bash
python comfyui_native_quantizer.py ming_image_0.1_design_bf16.safetensors --inspect
python comfyui_native_quantizer.py ming_image_0.1_design_bf16.safetensors \
  --format int8-convrot --dry-run
python comfyui_native_quantizer.py ming_image_0.1_ling_mini_2.0_bf16.safetensors \
  --components text_encoder --format int8-convrot --dry-run
```

The diffusion policy selects attention and feed-forward linears in the main
transformer, context refiner, and noise refiner. It preserves embeddings,
modulation, normalization, and final projections. Detection requires the
combined ComfyUI block signature and either a Ming marker or the unmarked
Design layout without Z-Image padding and decoder markers.

The text-encoder policy selects dense linears in the thinker and connector.
Token embeddings, vision, MoE routing and expert banks, normalization, and
output projections remain unchanged. The expert banks are large, so this
conservative policy may achieve less compression than a specialized converter.

The policies are experimental. Synthetic conversion and native metadata tests
do not establish full-checkpoint loading, generated-image quality, or runtime
speed. Inspect the dry-run plan and validate a completed output in ComfyUI
before relying on it.
