# Kandinsky 6 native policies

Two independent diffusion families are provided:

| Family | Checkpoint contract | Native consumer |
| --- | --- | --- |
| `kandinsky6` | Native `visual_blocks.*.videoT/audioT`, separate video/audio embeddings and cross-stream attention | Official Kandinsky 6 ComfyUI extension |
| `kandinsky6_sr` | Text-free SR DiT, `pooled_bias` and `visual_transformer_blocks.*` | Official Kandinsky 6 SR ComfyUI extension |

Both extensions build their relevant Linear layers through native ComfyUI
operations. They must be installed independently; these policies do not install
a runtime, add nodes or fetch weights.

```sh
comfyquant native-dit.safetensors --no-config --inspect --json
comfyquant native-dit.safetensors --no-config --format int8 --dry-run
comfyquant native-dit.safetensors --no-config --format int8 \
  --device cpu --target-runtime cpu --max-memory 256M \
  --output native-dit-int8.safetensors --validate
comfyquant --no-config --verify-output native-dit-int8.safetensors --json
```

The video/audio policy selects repeated video/audio self/cross-attention, FFNs
and cross-stream attention. Internal/external text encoders, embeddings,
modulation, modality gates, norms and output heads remain dense. Lite/Pro share
the naming contract; detection checks both streams and compatible dimensions.

The SR policy selects repeated transformer self-attention and FFNs, preserving
pooled conditioning, embeddings, modulation, norms and multi-grid output heads.
KVAE and latent-upscaler components are outside this policy. A text/audio tower
is not an SR signature. SR is not aliased to Kandinsky 5 or the base-video family.

Only the native checkpoint namespaces are accepted automatically. Diffusers
renaming requires a reviewed explicit `--key-map` that transforms the entire
input inventory before detection/quantization. Automatic conversion does not
emit quantization metadata in a namespace that the consumer then renames. Keep
packing/ConvRot alignment gates; unsupported dimensions retain source precision.

Independent header/shape, protected-tensor and synthetic conversion/verification
tests pass. The common native consumer metadata/Linear contract is exercised
separately. Full extension forwards and real audio/video/SR quality have not been
evaluated, so these policies remain experimental.

Reviewed primary sources:

- [Video/audio detector and contract](https://github.com/kandinskylab/kandinsky-6/blob/119d58ee5c5de0157644f09ef20bc9b2ab68a064/comfyui/kandinsky6/register.py)
- [Video/audio native model](https://github.com/kandinskylab/kandinsky-6/blob/119d58ee5c5de0157644f09ef20bc9b2ab68a064/comfyui/kandinsky6/ldm/model.py)
- [SR detector and contract](https://github.com/kandinskylab/kandinsky-6-sr/blob/326756b36d7ca5a3a1af0abd2b34edad9fe6cb2d/comfyui/kandinsky6_vsr/register.py)
- [SR native model](https://github.com/kandinskylab/kandinsky-6-sr/blob/326756b36d7ca5a3a1af0abd2b34edad9fe6cb2d/comfyui/kandinsky6_vsr/ldm/model.py)
