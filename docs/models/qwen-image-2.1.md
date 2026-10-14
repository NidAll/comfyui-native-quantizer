# Qwen-Image 2.1 quantization

The built-in `qwen_image21` policy is **experimental**. It targets the
32-block, single-stream diffusion transformer in the [original model](https://huggingface.co/Qwen/Qwen-Image-2.1)
and the raw BF16 diffusion checkpoint in [Comfy-Org's repackaging](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/blob/main/README.md).
It does not convert the VAE. The Qwen3-VL text encoder uses the converter's
separate text-encoder policy, which selects language-model linears and keeps
the vision tower and, by default, token embeddings. Input checkpoints that already
contain ComfyUI quantization metadata are rejected, so start from raw weights.

The [published weight index](https://huggingface.co/Qwen/Qwen-Image-2.1/commit/0501a7ee6495902d7bd2daf59406ae5fbd5c77c1)
shows attention `to_q`, `to_k`, `to_v`, `to_out.0` and unfused SwiGLU
`gate_layer`, `proj`, `out` in each block. [ComfyUI's model](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/qwen_image21/model.py)
also accepts a saved fused `gate_up`, `out` layout. Only these block linears
are candidates. `img_in`, `txt_in`, timestep embeddings, shared modulation,
attention norms, `norm_out`, and `proj_out` stay at source precision. Shape
eligibility and reconstruction gates can retain additional weights.

| Format | Path for this model |
| --- | --- |
| `bf16` | Reference precision for comparing images; no quantization. |
| `fp16` | Precision cast only. Compare against BF16 for overflow or quality changes. |
| `int8` | Row-scaled INT8 without rotation. |
| `int8-convrot` | INT8 with ConvRot; requires input width divisible by 256. |
| `w4a8` | Codebook weights with INT8 activations; use reconstruction gates and model-quality comparisons. |
| `w6a8` | Six-bit weights and INT8 activations; explicit Kitchen six-bit API and native consumer registration required. |
| `mixed` | Per-layer W4A8/W4A4/INT8 choice; W6A8 requires `--enable-w6a8`. Set `--target-runtime` to the intended inference backend and review the dry-run and report. |
| `w4a4` | Aggressive 4-bit weights with the default INT8 activation request. Test image quality and actual runtime dispatch. |
| `int4-convrot` | Aggressive 4-bit weight and INT4 activation request. Test on the target GPU; fallback dispatch can differ. |
| `fp8-e4m3` | E4M3 floating-point weights; accelerated inference requires compatible hardware and dispatch. |
| `fp8-e5m2` | Wider range and less precision than E4M3; native dtype casting is preserved. |
| `mxfp8` | Block-scaled FP8; comfy-kitchen serializer and suitable hardware needed for accelerated inference. |
| `nvfp4` | Aggressive 4-bit float storage experiment; comfy-kitchen serializer, high packing memory, and suitable hardware needed. |

Format support means native serialization, not a verified image-quality or
speed result. The local ComfyUI loader, comfy-kitchen, GPU, and model workflow
determine execution. No full Qwen-Image 2.1 checkpoint is included in tests.

Use the Python environment supplied with ComfyUI. Inspect the raw checkpoint
and its planned layers before conversion:

```bash
comfyquant --no-config /path/to/qwen_image_2.1_bf16.safetensors --inspect
comfyquant --no-config /path/to/qwen_image_2.1_bf16.safetensors \
  --format int8-convrot --dry-run
comfyquant --no-config /path/to/qwen_image_2.1_bf16.safetensors \
  --format int8-convrot --output /path/to/qwen_image_2.1_int8_convrot.safetensors \
  --max-memory 512M
comfyquant --no-config --verify-output /path/to/qwen_image_2.1_int8_convrot.safetensors
```

Add `--optimizer mse` to eligible quantized formats for optional scale and
clipping search. It preserves protected tensors and native layouts; precision
casts have no optimizable quantization scale. See [MSE optimization](../mse-optimization.md).
