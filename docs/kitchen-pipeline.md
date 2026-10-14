# Kitchen-targeted mixed precision

The checkpoint is the runtime boundary. The converter emits native ComfyUI
packed tensors and metadata. ComfyUI owns loading, patching, casting and
offloading; Kitchen owns execution and temporary workspaces.

## Planning and native execution

Architecture policies select eligible linears and protect sensitive projections,
normalization, embeddings and unsupported tensors. The mixed planner evaluates
weight reconstruction and, when supplied, recorded activation output error,
including activation quantization. Optional [MSE optimization](mse-optimization.md)
optimizes candidate payloads before these quality gates and format selection.

The planner satisfies quality and compression constraints, then uses available
storage headroom to improve estimated execution cost without increasing error.
Shape-specific measurements override expansion-cost heuristics when both
candidates have measurements. Layer and format names resolve ties deterministically.
`size-first` or `--mixed-objective size` selects for minimum storage. Explicit
byte/error limits take precedence; an unsatisfied gate aborts publication.

| Format | Serialized contract | Runtime implication |
| --- | --- | --- |
| INT8 | INT8 `[N,K]`, FP32 row scales | Direct INT8 linear |
| W4A4 | Signed nibbles in INT8 `[N,K/2]`, even index in low nibble, FP32 row scales; rotation group 64 | Native INT4 on compatible dispatch with `linear_dtype=int4`; otherwise packed GEMV or INT8 expansion |
| W4A8 | INT8 `[N,K/2]`, FP8 group scales, FP32 channel scales and 16-entry codebook; group 16, ConvRot 256 | Fused packed GEMV where available; otherwise INT8 decode workspace |
| W6A8 | INT8 `[N,3K/4]`, FP8 `[N,K/G]`, FP32 channel scales; G defaults to 16, ConvRot 256 | Six-bit codes decode onto INT8; explicit mixed opt-in |

W4A4 scoring follows effective activation precision. Uncertain dispatch evaluates
both activation variants conservatively. A storage name alone does not certify
native INT4 execution. Small-M packed paths may avoid the expansion workspace;
latency depends on workload shape, dtype and the installed backend.

Rotated weights require the matching runtime activation transform. Kitchen can
fuse rotation and activation quantization; rotations cannot be folded through
arbitrary nonlinearities or residuals. W4A8 relative group scales remain FP8 for
native chunked CUDA execution. No correction matrix or custom runtime is added.
Embedding opt-in uses indexed-row dequantization and does not certify integer
GEMM. Forced full-precision operations and patches requiring floating weights
retain their necessary dequantization paths.

FP8, NVFP4 and MXFP8 are explicit conversion formats. Mixed plans use the
W4A4/W4A8/INT8 candidate set, with W6A8 enabled only by `--enable-w6a8`.

## Conversion and calibration

```sh
comfyquant --no-config model.safetensors --format mixed \
  --profile throughput --target-runtime nvidia --optimizer mse \
  --calibration-source activations.npz --require-calibration \
  --max-memory 512M --output model-mixed.safetensors --validate
```

`balanced` budgets 0.90 bytes per parameter; `throughput` allows 1.10, permitting
row-scaled INT8 for most targeted layers. Review the dry-run and report for the
actual selected formats and protected precision.

`--require-calibration` requires samples for every targeted layer. Otherwise,
missing layers use weight reconstruction and the report exposes coverage. Capture
from an existing eager baseline runner, mapping modules to checkpoint weight keys:

```python
from native_quantizer.calibration_capture import CalibrationCapture

with CalibrationCapture(selected_modules, rows=64, rows_per_call=4,
                        max_bytes=256 << 20, seed=0) as capture:
    run_representative_inference()
capture.save("activations.npz")
```

Capture samples calls and steps using a deterministic reservoir, transferring
selected rows to CPU. Hooks exist only within the context manager and are removed
on failure. A budget overflow fails explicitly. Capture runs outside CUDA graphs;
it does not provide a workflow runner. Choose representative prompts, resolutions
and denoising steps in the owning baseline runner. Ingestion checks finite values,
widths, budgets and checkpoint-bound caches. Neither reconstruction metric proves
image quality.

## Runtime measurements

```sh
quantizer-kitchen-benchmark --comfy-root /path/to/ComfyUI \
  --m 256 --n 1024 --k 4096 --repeats 30 > kernel-costs.json
comfyquant --no-config model.safetensors --format mixed \
  --profile throughput --target-runtime nvidia --kernel-costs kernel-costs.json \
  --output model-mixed.safetensors --validate
```

The CUDA probe tests native loading and device movement, independent output-error
simulation, first/warm timings, temporary allocator peaks and graph replay.
Simulation estimates quantization error; reduced-precision fused rotation can
produce different tensors. Replay is compared exactly with the native forward.
These synthetic checks do not establish full-workflow quality, latency, memory,
dynamic-shape graph safety, patching or offload behavior.

Cost files are version-checked and shape-specific. Combine their `benchmarks`
arrays to cover multiple N/K shapes, with representative M workloads. The probe
uses BF16 activations and records workload shapes and the cost-file hash without
device identity. Use measurements from the intended environment and workload;
unmatched shapes use heuristics. `kitchen-costs/v2` keys grouped formats by group
size. Files using v1 remain readable for formats other than W6A8.

Streaming serialization, independent decoding, quality gates, resume integrity
and atomic publication apply to mixed conversion. Kernel workspaces and live
activation/output allocations remain owned by Kitchen. See [W6A8](w6a8.md) for
six-bit requirements and the observed odd-row graph limitation, and
[validation](release-validation.md) for required checks.
