# MiniCPM-V 4.7 1B visual prompt analysis

The `minicpm_v47` recipe supports the released MiniCPM-V 4.7 **1B** hybrid
Qwen3.5 checkpoint, including the Heretic derivative. It requires the
package-local consumer in `ComfyUI-Anima-Prompt-Architect`. Stock ComfyUI text
encoder loaders do not implement this model's visual preprocessing or generation.

The complete shape-checked signature combines language recurrent/full-attention
projections, the 1152-wide vision encoder and the 1024-by-4608 merger output.
Other model sizes, incomplete signatures and arbitrary remapped architectures
are not inferred as this family.

The conservative recipe selects repeated language MLP, full-attention Q/K/V/O,
and recurrent QKV/Z/output matrices. It preserves blocks 0 and 23, the complete
vision tower and merger, embeddings, output head, norms, recurrent A/dt controls,
depthwise convolution and small recurrent A/B projections. On the inspected
1.55B total-parameter model this produces 137 native INT8 ConvRot layers and
retains 643 original BF16 tensors exactly. FP32 row scales are quantization
metadata, not FP32 model weights.

```bash
comfyquant --no-config /path/to/model.safetensors --inspect
comfyquant --no-config /path/to/model.safetensors \
  --architecture minicpm_v47 --components text_encoder \
  --format int8-convrot --device cuda --output-dtype bf16 \
  --optimizer mse --mse-objective weight --mse-steps 9 --mse-iterations 2 \
  --max-memory 512M --validate --output /path/to/model-int8.safetensors
comfyquant --no-config --verify-output /path/to/model-int8.safetensors
```

Conversion math is FP32; inference uses BF16 and native W8A8 ConvRot kernels.
The recipe never pads incompatible matrices or modifies the original checkpoint.
Weight-MSE is an optional scale search, not evidence of visual equivalence.
Other quantization formats remain outside this consumer's deployment contract.
