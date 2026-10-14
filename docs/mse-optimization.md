# Optional native MSE optimization

Use `--optimizer mse` to search native quantization parameters before writing
the checkpoint. The default is `--optimizer none`: existing encoding algorithms,
precision policies, and serialized tensor layouts stay in use.

```sh
comfyquant --no-config input.safetensors --output output.safetensors --format w4a4 \
  --optimizer mse --mse-objective weight --max-memory 512M --validate
comfyquant --no-config input.safetensors --output output-mixed.safetensors --format mixed \
  --optimizer mse --enable-w6a8 --calibration-source activations.npz --validate
comfyquant --no-config --verify-output output.safetensors
```

The work was motivated by the scale-selection approach described in
[Anima INT4 ConvRot MSE](https://civitai.com/models/2998292/anima-int4-quants).
This implementation independently searches the framework's existing numerical
representations. The linked page's reported improvement is not a guarantee for
other weights, formats, or objectives.

## Coverage and numerical representation

Optimization runs after architecture selection and precision safeguards. It
therefore applies to eligible linears in every registered architecture, custom
data-only policy, supported component, and supported prefix. Norms, modulation,
inputs/outputs, protected embeddings, unsupported shapes and unaligned Iris
projections retain their existing policies. Explicit embedding quantization
remains an opt-in.

| Format | Optimized parameters and exact scoring representation |
| --- | --- |
| INT8 and INT8 ConvRot | FP32 row scales, clipping and alternating least squares; decoded INT8 values |
| INT4 ConvRot / W4A4 | FP32 row scales, clipping and alternating least squares; native signed nibbles and existing ConvRot |
| W4A8 | Existing adaptive codebook/ALS seed, FP8 E4M3 relative group scales and FP32 channel scales; nearest **rounded, saturated INT8** decoded levels |
| W6A8 | Existing Kitchen six-bit scale search seed, FP8 relative group scales and FP32 channel scales; native six-bit packing and INT8 decode grid |
| FP8 E4M3 and E5M2 | One FP32 tensor scale, clipping and least squares; actual dtype casts including subnormals and saturation |
| NVFP4 | E4M3 block scales and FP32 tensor scale; actual E2M1 packing, native swizzle and Kitchen decode |
| MXFP8 | Neighboring representable E8M0 block exponents, E4M3 data, native swizzle and Kitchen decode |
| Mixed | Optimizes every eligible candidate in the existing W4A4/W4A8/INT8/W6A8 candidate set before quality/runtime/storage selection |

The original encoding is always a candidate. Ties preserve it. Searches select
the lowest measured error among valid proposals, including clipping, group
refinement and least-squares proposals where supported. This is a bounded search,
not a proof of the global optimum. W4A8 codebook fitting and W6A8's existing
grid-aware search remain the starting methods rather than being replaced by
an absmax-only encoder. MXFP8 often retains its original encoding: power-of-two
block scales already constrain the useful alternatives strongly. FP16/BF16
casts have no quantization scale and reject `--optimizer mse`.

No loader, inference kernel, activation quantizer, group size, rotation, shape,
or native per-layer metadata contract is added. Native NVFP4/MXFP8 offline
encoding uses Kitchen's FP32 eager tensor operations on CUDA because its CUDA
block encoders accept only FP16/BF16 inputs. The FP32 source is preserved during
encoding.
Inference continues through the existing ComfyUI/Kitchen dispatch.

## Objectives and controls

| Option | Default | Meaning |
| --- | --- | --- |
| `--optimizer` | `none` | Enable with `mse` |
| `--mse-objective` | `auto` | `weight`, `activation`, or calibration-dependent `auto` |
| `--mse-steps` | `17` | 2–129 scale/clipping candidates for applicable formats, spanning 0.5–1.05 of the seed scale |
| `--mse-iterations` | `3` | 0–16 least-squares refinement iterations where supported |

MXFP8 uses fixed neighboring block-exponent proposals (offsets −2, −1, +1, +2)
rather than the scale grid or least-squares refinement.

Weight optimization minimizes `sum((W_decoded - W)^2)`. Activation optimization
minimizes `sum(((W_decoded - W) @ X.T)^2)` on recorded calibration rows. Selection
uses sums because the MSE denominator is constant across candidates. ConvRot
transforms weights and calibration consistently; it never removes the matching
runtime activation transform.

`auto` uses activation MSE for covered layers and weight MSE elsewhere.
`activation` requires coverage for every targeted layer. `--require-calibration`
also enforces complete coverage. Use the existing calibration capture and
NPZ/manifest/cache formats from the [pipeline guide](kitchen-pipeline.md).
For native formats, calibration options are accepted when MSE is enabled.

Activation MSE can increase weight MSE while reducing sampled output error.
It measures reconstructed-weight output error on full-precision calibration
rows. Dynamic activation rounding, finite compute precision and generated-image
quality are separate effects. Mixed planning still evaluates its existing
runtime-output error gates on the optimized payloads. Embedding lookup quality
requires embedding-specific evaluation; ordinary linear activation samples do
not represent token lookup usage.

Python callers use the same settings:

```python
from native_quantizer import api
from native_quantizer.options import ConversionOptions

api.convert(ConversionOptions(
    model="input.safetensors", output="output.safetensors", format="w6a8",
    optimizer="mse", mse_objective="weight", mse_steps=17, mse_iterations=3,
    max_memory="512M", validate=True,
))
```

These four settings are also supported in schema-1 XDG defaults. CLI flags
override saved settings; `--no-config` provides an isolated invocation. The
library continues to use explicit requests without reading XDG defaults.

## Memory, determinism and recovery

Candidates are processed sequentially. Working estimates increase when MSE is
enabled, making row chunks smaller. Calibration is transferred in small tiles;
no Hessian or stack of candidate tensors is allocated. Tensor-wide FP8 scale
selection streams all rows with bounded buffers. Swizzled NVFP4/MXFP8 still
require a whole-layer budget and fail the preflight if it is too small.
The budget excludes mmap pages, Python/framework overhead and allocator caches.

The seed and search order are fixed. Repeated conversions on the same backend,
precision and budget are deterministic. Matrix multiplication rounding can
vary with row-chunk shape or device; cross-budget/device bit identity is not
promised. Native `--rotation-backend butterfly` retains its existing fixed
elementwise reduction option. Resume identities bind the optimizer revision,
all search settings and the actual selected calibration rows, including cache
hits. Rejected resume identities leave the journal unchanged.

W4A8, W6A8 and mixed retain resumable execution. Other native formats retain
atomic publication and restart from the beginning after interruption. Optimizer
provenance lives in the converter extension, separate from loader configuration;
source-free verification validates it when present. Outputs without optimizer
provenance remain readable.

## Regression tests and benchmarks

```sh
python scripts/check.py
python scripts/check.py --comfy-root /path/to/ComfyUI --cuda
python scripts/benchmark_mse.py --output comparison.json
python scripts/benchmark_mse.py --objective activation --output activation.json
python scripts/benchmark_mse.py --device cuda --output cuda-conversion.json
quantizer-kitchen-benchmark --comfy-root /path/to/ComfyUI --optimizer mse
```

The conversion benchmark uses fresh processes for each format/mode and small
seeded Gaussian/outlier checkpoints. It records weight/activation MSE, first and
warm complete conversion time with validation, peak process RSS, CUDA allocated
memory and file size. RSS includes imported libraries; CUDA allocated memory
excludes reserved caches. Errors are recorded and cause a nonzero exit.

Tests cover baseline preservation, native decode/packing, correlated activation
samples, zero/sparse/outlier/tiny weights, bounded/fitted-codebook execution,
CLI/API parity, provenance corruption, native consumer loading, shape/precision
safeguards and resume mismatch/recovery. CUDA checks exercise native integer
forward/graph replay with floating-weight dequantization blocked, plus FP32
NVFP4/MXFP8 CPU/CUDA encoding agreement. The native runtime benchmark accepts
`--optimizer mse` for matched measurements of existing kernels and temporary
allocations. Synthetic checks do not establish full-model generation quality,
full-workflow memory, speed, or accelerated NVFP4/MXFP8 execution.
