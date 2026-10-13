# ComfyUI Native Quantizer

Native checkpoint conversion for ComfyUI, with a modular Python API, streaming
I/O, precision policies, independent verification and a command-line interface.

**Current candidate: 2.3.0rc2.** See the [changelog](CHANGELOG.md),
[support and evidence matrix](docs/support.md) and [V2 migration guide](docs/migration-v2.md).
This candidate establishes the maintained modular base. Architecture policy
coverage does not certify generation quality or accelerated execution for every model.

Convert compatible ComfyUI checkpoints and text encoders to ComfyUI's native
quantized safetensors formats. The converter streams large
files, preserves unsupported tensors, and validates its output before publishing
it.

It does not modify ComfyUI core and does not produce GGUF, GPTQ, bitsandbytes,
AWQ, or other custom-loader formats.

## Quick start

Use Python 3.12–3.13 in an environment that already supplies a compatible
PyTorch build. Choose PyTorch for your backend using its
[official installation guide](https://pytorch.org/get-started/locally/).
PyTorch is intentionally supplied by the environment, so installing the
quantizer does not replace a working ComfyUI compute stack.

From a source checkout:

```sh
python -m pip install -e .
comfyquant --no-config --list-formats
comfyquant model.safetensors --inspect
comfyquant model.safetensors --format int8 --dry-run
comfyquant model.safetensors --format int8 --output model-int8.safetensors \
  --max-memory 512M --validate
comfyquant --verify-output model-int8.safetensors
```

Install optional native layout and Hub dependencies only when needed:

```sh
python -m pip install -e '.[kitchen,hub]'
```

Built distributions can be installed with `python -m pip install path/to/wheel.whl`.
There is no PyPI publication implied by this source release.

For user-local commands, XDG defaults and Fish/Bash/Zsh completion, see
[Linux integration](docs/linux-cli.md). Catalog commands start without PyTorch.

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
| `w6a8` | `w6a8_int8` | Uniform 6-bit weights with INT8 activations; Kitchen >= 0.2.36 |
| `mixed` | Per-layer native formats | Kitchen-aware W4A8/W4A4/INT8 selection; W6A8 opt-in |
| `fp8-e4m3` | `float8_e4m3fn` | Tensor-scaled FP8 |
| `fp8-e5m2` | `float8_e5m2` | Tensor-scaled FP8 |
| `nvfp4` | `nvfp4` | Comfy-kitchen NVFP4 layout |
| `mxfp8` | `mxfp8` | Comfy-kitchen MXFP8 layout |
| `fp16`, `bf16` | — | Precision conversion without quantization |

Prefixed checkpoints use native per-layer `.comfy_quant` tensors and an empty
header layer map. This avoids duplicate metadata warnings when ComfyUI removes
the model prefix. Bare checkpoints retain header metadata. Verification and
validation support both representations; no ComfyUI changes are required.

The serialized format and accelerated execution are separate concerns. ComfyUI,
comfy-kitchen, the selected loader, and the target runtime determine which
kernel is used. Unsupported accelerated formats remain selectable and produce a
warning when the runtime capability is known.

Comfy-kitchen also contains AWQ W4A16 and SVDQuant W4A4 layouts. They are not
registered as native loadable algorithms by stock ComfyUI, so this project does
not emit them.

Mixed mode now budgets runtime weight expansion as well as storage and error.
Use `--profile throughput` for more INT8 headroom, and `--kernel-costs` for
shape-specific measured selection. See the [pipeline guide](docs/kitchen-pipeline.md)
for calibration capture, layout contracts, validation and limits.

W6A8 defaults to group size 16. Use `--enable-w6a8` to add it to mixed
planning; the opt-in is off by default. See the [W6A8 guide](docs/w6a8.md) for
packing, compatibility, examples, validation and runtime limits.

## Architecture support

Iris-3B has its own fused/unfused trunk and pixel-head policy, with native Iris
loader validation and conservative shape guards. Kandinsky 6 Lite/Pro video/audio
and Kandinsky 6 SR have separate native-layout policies. See the
[Iris guide](docs/iris.md) and [Kandinsky 6 guide](docs/kandinsky6.md) for loader
requirements, preserved boundaries and supported checkpoint namespaces.


Built-in policies cover a broad range of ComfyUI diffusion, video, audio, and
text-encoder architectures, including experimental native ACE-Step 1.5 support,
Flux-family models, MiniMax, LTXV/LTXAV, YuE2, Trellis2, Qwen text encoders,
T5, and CLIP. Qwen-Image 2.1 has an experimental native policy; see the
[format guide](docs/qwen-image-2.1.md). Ming-Image Design and Design-Layer
ComfyUI checkpoints and their Ling Mini text encoders also have experimental
policies; see the [Ming-Image guide](docs/ming-image.md). Run
`--list-architectures` for the authoritative current list.

Detection uses checkpoint tensor names and shapes, rather than filenames.
Z-Image (`z_image`), pixel-space Z-Image (`z_image_pixel_space`), and Lumina2
have separate policies and architecture labels. The same distinction applies
to HunyuanDiT1, Chroma Radiance, LTXAV, PixArt Sigma, PiD, and SAM31. Class
names such as `--architecture ZImage` work as aliases. Automatic detection
rejects incomplete or ambiguous signatures; use `--inspect` to review them
before choosing an override. Input/output projections, embeddings, norms,
modulation, and pixel-space decoders remain at source precision.

## Text encoders

Use `--components text_encoder` to select a text-encoder checkpoint:

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

## Input sources

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

# Run the regression suite
python scripts/check.py

# Generate a data-only policy for an architecture not yet listed
python comfyui_native_quantizer.py --write-architecture-template my-model.json
```

Use `--include`, `--exclude`, and `--keep-precision` for narrow layer
overrides. A custom policy is data-only and does not install a model loader;
the target ComfyUI architecture must already understand the tensor names and
metadata.

`--runtime-certificate` accepts a JSON certificate from observed native load
and forward execution on the target backend. Static capability checks alone do
not qualify as certification.

## Design boundaries

- Conversion is local and deterministic for the same backend, options, and input.
- Unsupported tensors pass through unchanged.
- Quantization uses reconstruction gates; a layer that fails its gate stays at
  source precision.
- Memory limits control temporary conversion work, not total process RSS.
- Output publication is atomic and output validation runs before the final path
  is made visible.

## Project layout

```text
comfyui_native_quantizer.py   compatibility CLI and Python facade
native_quantizer/
  api.py, options.py         public operations and conversion requests
  application.py            conversion orchestration
  checkpoint_io/            discovery, shards, streaming reads/writes
  architectures/            detection and precision-preservation policies
  codecs/, formats.py       native encoding, packing and layout contracts
  planning/                 metrics, sensitivity and mixed selection
  execution/                conversion engines and resume journals
  runtime/                  Kitchen probes, capabilities and measured costs
  validation/               independent reconstruction and output checks
  cli/                      arguments, commands and process exit handling
  testing/                  on-demand regression suite and reference fixtures
pyproject.toml               CLI/library packaging
```

The original script remains supported. The equivalent module command is
`python -m native_quantizer`. An editable core install with
`python -m pip install -e .` also provides `comfyui-native-quantizer`.
Use the Python environment that already supplies the intended PyTorch build.

See [docs/architecture.md](docs/architecture.md) for ownership, dependency rules,
Python API examples and regression commands. Conversion algorithms and format
contracts retain their existing behavior.

## Development and releases

[CONTRIBUTING.md](CONTRIBUTING.md) describes reproducible checks and code ownership.
Default checks use synthetic fixtures and isolated preferences. Studio is deferred and excluded from this release. ComfyUI integration
is explicitly configured with `COMFYUI_ROOT`; CUDA checks are a separate required
run when selected. No check downloads model weights or starts ComfyUI services.

The [release guide](docs/releasing.md) covers source review, distribution builds,
installed-package smoke checks, checksums and promotion from candidate to stable.
This release ships one CLI/library wheel and one source archive containing
the CLI guides and checks.

Licensed under [Apache-2.0](LICENSE).
