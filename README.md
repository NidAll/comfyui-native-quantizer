# ComfyUI Native Quantizer

Convert compatible ComfyUI checkpoints and text encoders to ComfyUI's native
quantized safetensors formats. The converter is self-contained, streams large
files, preserves unsupported tensors, and validates its output before publishing
it.

It does not modify ComfyUI core and does not produce GGUF, GPTQ, bitsandbytes,
AWQ, or other custom-loader formats.

## Quick start

Run the converter with the Python environment used by ComfyUI:

```bash
python comfyui_native_quantizer.py --list-formats
python comfyui_native_quantizer.py model.safetensors \
  --format int8 \
  --output model-int8.safetensors \
  --max-memory 512M
```

Preview detection and layer decisions first:

```bash
python comfyui_native_quantizer.py model.safetensors --inspect
python comfyui_native_quantizer.py model.safetensors \
  --format mixed --experimental --dry-run
```

The input is never overwritten. Use `--overwrite` only when replacing an
existing output is intentional.

## Native formats

| CLI format | ComfyUI algorithm | Use |
| --- | --- | --- |
| `int8` | `int8_tensorwise` | Row-wise INT8 weights |
| `int8-convrot` | `int8_tensorwise` | INT8 weights with ConvRot metadata |
| `w4a4` | `convrot_w4a4` | Packed 4-bit weights |
| `int4-convrot` | `convrot_w4a4` | Packed 4-bit weights with an INT4 execution request |
| `w4a8` | `asym_w4a8_int8` | 4-bit weights with INT8 activations |
| `mixed` | Per-layer native formats | Quality-aware W4A8/W4A4/INT8 selection |
| `fp8-e4m3` | `float8_e4m3fn` | Tensor-scaled FP8 |
| `fp8-e5m2` | `float8_e5m2` | Tensor-scaled FP8 |
| `nvfp4` | `nvfp4` | Comfy-kitchen NVFP4 layout |
| `mxfp8` | `mxfp8` | Comfy-kitchen MXFP8 layout |
| `fp16`, `bf16` | — | Precision conversion without quantization |

The serialized format and accelerated execution are separate concerns. ComfyUI,
comfy-kitchen, the selected loader, and the target runtime determine which
kernel is used. Unsupported accelerated formats remain selectable and produce a
warning when the runtime capability is known.

Comfy-kitchen also contains AWQ W4A16 and SVDQuant W4A4 layouts. They are not
registered as native loadable algorithms by stock ComfyUI, so this project does
not emit them.

## Text encoders

Built-in policies cover decoder-only LLM text encoders, T5/UMT5, T5Gemma,
CLIP, JinaCLIP2, BERT, Qwen3.5 hybrid attention, and YuE2. Diffusion policies
are included for Trellis2, SenseNova U1.5, and YuE2. Use:

```bash
python comfyui_native_quantizer.py text_encoder.safetensors \
  --components text_encoder \
  --format int8 \
  --output text_encoder-int8.safetensors
```

Token embeddings remain unchanged by default. Opt in to row-wise INT8 token
embedding lookup with:

```bash
python comfyui_native_quantizer.py text_encoder.safetensors \
  --components text_encoder \
  --format int8 \
  --quantize-text-embeddings \
  --output text_encoder-int8.safetensors
```

The embedding option is supported with `int8`, `int8-convrot`, and `mixed`.
Embeddings use plain INT8 even when linears use another INT8 variant. Positional
embeddings, output heads, vision/audio towers, and expert banks remain at source
precision.

## Installation

Use the Python environment that contains the ComfyUI PyTorch build:

```bash
python -m pip install numpy safetensors
```

Optional features:

```bash
python -m pip install huggingface-hub
python -m pip install comfy-kitchen
```

The converter accepts local safetensors files, indexed shards, model
directories, and explicitly trusted local pickle checkpoints. Hugging Face
inputs are pinned to a resolved revision and only the requested component is
downloaded.

## Useful commands

```bash
# List built-in architecture policies
python comfyui_native_quantizer.py --list-architectures

# Check a completed native output
python comfyui_native_quantizer.py --verify-output model-int8.safetensors

# Run the embedded regression suite
python comfyui_native_quantizer.py --self-test
python -m unittest -v test_quantizer_native.py

# Generate a data-only policy for an architecture not yet listed
python comfyui_native_quantizer.py --write-architecture-template my-model.json
```

Use `--include`, `--exclude`, and `--keep-precision` for narrow layer
overrides. A custom policy is data-only and does not install a model loader;
the target ComfyUI architecture must already understand the tensor names and
metadata.

## Design boundaries

- Conversion is local and deterministic for the same backend, options, and input.
- Unsupported tensors pass through unchanged.
- Quantization uses reconstruction gates; a layer that fails its gate stays at
  source precision.
- Memory limits control temporary conversion work, not total process RSS.
- Output publication is atomic and output validation runs before the final path
  is made visible.
- ComfyUI core is never edited for this project. New integrations belong in a
  dedicated `ComfyUI/custom_nodes/<name>/` directory.

## Project layout

```text
comfyui_native_quantizer.py  converter
test_quantizer_native.py     regression tests
architecture_template.json   data-only policy template
requirements-optional.txt    optional dependencies
```

## License

No license is asserted by this repository yet. Add the license that matches the
intended distribution before publishing a release.
