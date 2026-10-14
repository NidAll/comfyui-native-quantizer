# Iris-3B native checkpoint conversion

Family: `iris` (`Iris3B` is an alias). The policy handles the released Iris-3B
dual/single-stream trunk and pixel head, with unfused `w1/w3/w2` or fused
`gate_up/w2` trunk MLPs. Base and full Turbo checkpoints share this structure;
a community Turbo LoRA is a separate adapter and is not converted by this policy.

Use a compatible native Iris ComfyUI loader. Stock ComfyUI alone does not
register this architecture; the reviewed `ComfyUI-Iris` implementation creates
its Linear layers through native `comfy.ops` without checkpoint key conversion.
This quantizer does not install or modify that loader.

```sh
comfyquant iris.safetensors --no-config --inspect --json
comfyquant iris.safetensors --no-config --format int8-convrot --dry-run
comfyquant iris.safetensors --no-config --format int8-convrot \
  --device auto --target-runtime auto --compute-dtype fp32 \
  --rotation-backend butterfly --max-memory 256M \
  --output iris-int8.safetensors --validate
comfyquant --no-config --verify-output iris-int8.safetensors --json
```

Detection checks the full text/pixel/trunk signature and consistent projection,
patch, norm and fused-MLP shapes. Incomplete or mixed fused/unfused signatures
are rejected. Detection does not fall back to PixelDiT for Iris-like partial inputs.

The allowlist selects repeated trunk attention/MLP projections and pixel-head
attention/MLP projections. Text adapters, embeddings, positional tensors, norms,
modulation, sigmoid attention gates, compress/expand stage boundaries and output
heads remain at source precision. Selected linears require N divisible by 16 and
K divisible by 256 in every planning mode. Original 6,826-wide MLP projections
therefore remain dense; aligned prepared checkpoints can reach the corresponding
projections. `--include` does not bypass this consumer safety boundary. No padding,
fusing or adapter merging is performed here.

Conversion and source-free verification use independent existing native codecs.
A tiny complete Iris checkpoint is tested through the installed native loader,
strict state loading and a full forward. Separate CUDA checks cover aligned
real projection shapes with weight dequantization blocked. These checks establish
loader/dispatch contracts, not full-checkpoint visual quality or throughput.

Architecture sources: [official Iris implementation](https://github.com/speridlabs/iris-3b/blob/main/src/iris3b/models/dit.py)
and [official model card](https://huggingface.co/speridlabs/iris-3b).
