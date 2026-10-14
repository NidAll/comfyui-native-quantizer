# ComfyUI Native Quantizer

A CLI and Python library for converting diffusion checkpoints and text encoders
to native ComfyUI quantization formats. It selects eligible weights by architecture,
preserves sensitive tensors, converts within a working-memory budget, and validates
reconstruction before publishing the output.

Optional MSE optimization searches scales and clipping on each format's actual
numerical grid. Inference uses the existing ComfyUI/Kitchen layouts and kernels.
The target model architecture must be supported by your ComfyUI installation;
some architectures require their own native extensions.

## Installation

Use Python 3.12–3.13 with your intended PyTorch build already installed. From this
checkout, install the package and native-layout dependency:

```sh
python -m pip install '.[kitchen]'
comfyquant --version
comfyquant --no-config --list-formats
```

You can also install a release wheel with `python -m pip install /path/to/package.whl`.
PyTorch is supplied by your environment. Add the `hub` extra for Hugging Face
inputs, or `dev` for development and packaging tools. See the
[installation guide](docs/linux-cli.md) for command paths, completions and defaults.

## Quick start

Inspect the checkpoint and preview the selected layers:

```sh
comfyquant --no-config input.safetensors --inspect
comfyquant --no-config input.safetensors --format int8-convrot --dry-run
```

Convert, validate against the source, then verify the serialized output:

```sh
comfyquant --no-config input.safetensors --format int8-convrot \
  --output output-int8.safetensors --max-memory 512M --validate
comfyquant --no-config --verify-output output-int8.safetensors
```

The source is preserved. Replacing an output requires `--overwrite`. Unsupported
shapes and protected tensors keep their source precision. Review the dry-run and
report for actual coverage; a format name does not guarantee native acceleration
on every consumer, shape or backend.

## MSE optimization

Enable `--optimizer mse` for eligible quantized weights in any registered
architecture. The default remains `--optimizer none`. The original encoding is
always considered, and the optimizer retains it when no valid candidate improves
the selected objective.

For Z-Image INT4 ConvRot with weight reconstruction MSE:

```sh
comfyquant --no-config z-image.safetensors --architecture z_image \
  --format int4-convrot --optimizer mse --mse-objective weight \
  --device auto --max-memory 512M --output z-image-int4-mse.safetensors --validate
```

For mixed precision with recorded activations:

```sh
comfyquant --no-config input.safetensors --format mixed --profile balanced \
  --optimizer mse --mse-objective activation \
  --calibration-source activations.npz --require-calibration \
  --max-memory 512M --output output-mixed.safetensors --validate
```

| Objective | Selection |
| --- | --- |
| `weight` | Minimize weight reconstruction error |
| `activation` | Minimize reconstructed-weight output error on recorded activation rows; complete coverage required |
| `auto` (default) | Use activation error where samples exist and weight error elsewhere |

`--mse-steps` controls applicable scale/clipping searches; `--mse-iterations`
controls applicable least-squares refinement. Searches respect integer rounding,
FP8 scale storage and block-format representations. Improvement depends on the
weights, format and objective; reconstruction MSE does not establish image quality.
See [MSE optimization](docs/mse-optimization.md) for controls, calibration,
benchmarks and numerical limits.

## Formats

| Format | Stored weights and execution request |
| --- | --- |
| `int8`, `int8-convrot` | INT8 with FP32 row scales; optional ConvRot |
| `w4a4` | Packed INT4 with FP32 row scales; INT8 execution request by default |
| `int4-convrot` | Packed INT4 with ConvRot and an INT4 execution request |
| `w4a8` | Codebook INT4 with FP8 group scales and FP32 channel scales; INT8 decode grid |
| `w6a8` | Uniform six-bit weights with FP8 group scales and FP32 channel scales; INT8 decode grid |
| `mixed` | Per-layer W4A4/W4A8/INT8; W6A8 added with `--enable-w6a8` |
| `fp8-e4m3`, `fp8-e5m2` | FP8 with one FP32 tensor scale |
| `nvfp4` | E2M1 values, E4M3 block scales and an FP32 tensor scale |
| `mxfp8` | E4M3 values and E8M0 block scales |
| `fp16`, `bf16` | Precision casts; no MSE quantization-scale optimization |

W6A8 requires Kitchen's explicit six-bit API and native ComfyUI registration.
Block formats require compatible layout APIs; accelerated inference needs an
appropriate backend. See [support](docs/support.md) and [W6A8](docs/w6a8.md).

## Architecture and input coverage

List policies with `comfyquant --no-config --list-architectures`. Detection uses
tensor names and shapes, with separate policies for distinct model variants.
`--architecture NAME` provides an explicit override. Include/exclude controls
still obey shape, precision and consumer safeguards.

Inputs can be safetensors files, indexed shards or model directories. Pickle
input requires `--trust-pickle`. Hugging Face input requires the `hub` extra and
explicit download options; `--offline` prevents model downloads.

Use `--components text_encoder` for supported text towers. Token embeddings stay
at source precision unless explicitly enabled with `--quantize-text-embeddings`
for supported INT8/mixed paths. Norms, modulation, input/output boundaries and
protected auxiliary towers retain their architecture policies.

`--max-memory` budgets temporary conversion work, excluding mmap pages, framework
overhead and allocator caches. W4A8, W6A8 and mixed conversions support matching
resume journals. Other formats restart atomically. NVFP4/MXFP8 require a sufficient
whole-layer budget. These limits describe conversion, not inference VRAM.

## Python API

```python
from native_quantizer import api
from native_quantizer.options import ConversionOptions

status = api.convert(ConversionOptions(
    model="input.safetensors", output="output.safetensors",
    format="int8-convrot", optimizer="mse", mse_objective="weight",
    max_memory="512M", validate=True,
))
verification = api.verify_output("output.safetensors")
```

API requests are explicit and do not read CLI preferences. See
[architecture and API](docs/architecture.md) for operations and module ownership.

## Project layout

```text
native_quantizer/   CLI/library implementation and packaged self-test fixtures
tests/            Regression suites (cpu/, cuda/) and shared support
scripts/          Checks, installed smoke, benchmarks and Linux setup
docs/             Usage, numerical contracts, support and release guides
docs/models/      Architecture-specific conversion guides
skills/           Architecture contributor workflow
.github/          CI and contribution templates
```

The root Python entry points remain thin compatibility interfaces. Maintained
implementation lives in `native_quantizer/`; dependencies are declared in
`pyproject.toml`. Generate a custom architecture starter with
`comfyquant --no-config --write-architecture-template policy.json`.

## Documentation and development

- [Installation and configuration](docs/linux-cli.md)
- [MSE objectives and benchmarks](docs/mse-optimization.md)
- [Mixed precision, calibration and runtime costs](docs/kitchen-pipeline.md)
- [Iris](docs/models/iris.md), [Kandinsky 6](docs/models/kandinsky6.md),
  [Ming-Image](docs/models/ming-image.md), [Qwen-Image 2.1](docs/models/qwen-image-2.1.md)
- [Support](docs/support.md) and [validation](docs/release-validation.md)
- [Contributing](CONTRIBUTING.md), [release process](docs/releasing.md) and [changelog](CHANGELOG.md)

```sh
python scripts/check.py
python scripts/check.py --comfy-root /path/to/ComfyUI --cuda
```

Default checks use CPU fixtures with isolated preferences. Requested consumer and
CUDA checks fail when prerequisites are missing. Synthetic tests establish
serialization, numerical and consumer contracts; model quality, workflow latency,
offload and adapters need separate evaluation. Development checks do not download
model weights or start services.

Licensed under [Apache-2.0](LICENSE).
