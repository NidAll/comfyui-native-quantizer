# Kitchen-targeted mixed precision

The checkpoint is the runtime boundary. The converter emits ComfyUI's native
packed tensors and metadata; ComfyUI owns loading, patching, casting and
offloading, and Kitchen owns execution and temporary workspaces. There is no
custom loader, persistent dequantized-weight cache or model monkeypatch.

## Findings and architecture

The old mixed planner minimized serialized bytes subject to error gates. It did
not distinguish native integer execution from packed formats that expand weights
before GEMM. In particular, the W4A4 storage name does not determine activation
precision. Repeated transfers during evaluation, duplicated signed INT4 packing,
and lack of bounded activation capture also complicated the pipeline.

The pipeline now has these responsibilities:

1. Architecture policies select linear layers and protect sensitive projections,
   normalization, embeddings and unsupported tensors. Explicit embedding opt-in
   remains an indexed-row dequantization path, not integer GEMM coverage.
2. `calibration_capture.py` records a bounded deterministic reservoir from supplied
   modules during representative inference. Existing calibration ingestion checks
   widths, finite values, budgets and checkpoint-bound caches. Capture hooks exist
   only within the context manager and are removed on exceptions.
3. `MixedPlanner` evaluates reconstruction and, when supplied, activation output
   error including activation quantization. W4A4 evaluation uses the effective
   activation precision; uncertain dispatch evaluates both variants conservatively.
4. `kitchen_policy.py` describes execution contracts separately from quality and
   storage. The planner first satisfies quality/compression gates, then spends
   available storage budget to improve execution without increasing estimated
   error. Shape-specific measurements override the expansion-cost heuristic when
   both candidates have measurements. Ties are resolved by layer/format names.
5. Offline transforms and quantization emit the existing native layouts. Native
   and mixed INT4 share one signed nibble encoder. Weight metrics stay on the
   evaluation device and return four scalar reductions together; activation
   scoring avoids transferring full weights and decoded weights back to CPU.
   Chunked CPU reference scoring reuses one transfer per tensor per chunk.
6. Streaming serialization, independent reference decoders, resume integrity,
   quality gates and atomic publication remain in place. The planner revision
   changes to invalidate plans made with the old policy.
7. ComfyUI loads `comfy_quant` metadata into its native `QuantizedTensor` layouts.
   Normal model patching and device movement continue through ComfyUI. Forced
   full-precision execution and patches that require floating weights retain
   their necessary dequantization path.

## Runtime formats

| Format | Serialized contract | Runtime implication |
| --- | --- | --- |
| INT8 | Row-major INT8 `[N,K]`, FP32 row scales | Direct INT8 linear, no weight rotation or nibble expansion |
| W4A4 | Row-major signed INT8 `[N,K/2]`, even index in low nibble, FP32 row scales; quant group 64 | Native INT4 only on a compatible dispatch and `linear_dtype=int4`; otherwise packed GEMV or expansion into INT8 |
| W4A8 | INT8 `[N,K/2]`, FP8 E4M3 group scales, FP32 channel scales and 16-entry codebook; group 16, ConvRot 256 | Small-M fused decode GEMV where available; otherwise INT8 decode workspace |

Weights are rotated once offline. Removing the matching activation transform
would change the model's math. The normal runtime uses Kitchen's fused rotation
and quantization when supported; the converter cannot safely fold rotations
through arbitrary nonlinearities or residual connections. W4A8 group scales stay
FP8, which is required by the installed CUDA chunked fast path. No correction
matrix or runtime transpose/repacking format is added.

The expansion contract is a conservative general-GEMM cost indicator. It is not
a claim that every invocation allocates expanded weights: small-M packed GEMV
paths can avoid that workspace. Actual latency depends on M, N, K, dtype and the
installed backend. Native INT4 is not inherently faster than INT8.

FP8, NVFP4 and MXFP8 remain explicit formats. They are not automatically inserted
into mixed plans without matching target acceleration and comparable quality
scoring. Unsupported AWQ/SVDQuant native loading is not invented by the converter.

## Use

Run from the repository checkout; retain the supporting Python modules alongside
the entry point. Mixed conversion no longer needs `--experimental`.

```bash
python comfyui_native_quantizer.py model.safetensors \
  --format mixed --profile throughput --target-runtime nvidia \
  --calibration-source activations.npz --require-calibration \
  --max-memory 512M --output model-mixed.safetensors
```

`balanced` keeps the existing 0.90-byte/parameter budget and spends available
headroom on execution improvements. `throughput` allows 1.10 bytes/parameter,
which permits row-scaled INT8 for most targeted layers. Quality thresholds remain
the same. `size-first` or `--mixed-objective size` retains minimum-storage
selection. Hard user-specified byte/error limits still take precedence; failure
to satisfy a gate aborts publication.

`--require-calibration` requires samples for **every targeted layer**. Without
it, missing layers use weight reconstruction error; reports expose calibration
coverage. Neither metric proves image quality.

Capture from an existing eager baseline runner, mapping the actual modules to
checkpoint weight keys:

```python
from calibration_capture import CalibrationCapture

with CalibrationCapture(selected_modules, rows=64, rows_per_call=4,
                        max_bytes=256 << 20, seed=0) as capture:
    run_representative_inference()
capture.save("activations.npz")
```

Sampling covers calls/steps rather than only the first activation tensor. It
copies selected rows to CPU during calibration. Capture is intentionally outside
CUDA graph capture and normal inference; it is not itself a ComfyUI workflow
runner. Choose representative prompts, resolutions and denoising steps in the
owning baseline runner. A capture budget overflow fails explicitly.

## Runtime validation and measured policy

```bash
python benchmark_kitchen.py --comfy-root /path/to/ComfyUI \
  --m 256 --n 1024 --k 4096 --repeats 30 > kernel-costs.json
python comfyui_native_quantizer.py model.safetensors \
  --format mixed --profile throughput --target-runtime nvidia \
  --kernel-costs kernel-costs.json --output model-mixed.safetensors
```

The probe forces Kitchen CUDA dispatch, tests native ComfyUI loading and device
round-trip, compares outputs with independent runtime simulation, records first
and warm timings and temporary allocator peak, then checks graph replay. These
are synthetic linear tests, not full workflow throughput, LoRA/offload stress
or generated-image quality tests.

The FP32 activation simulation predicts quantization error; it is not a bit-exact
reference for fused reduced-precision rotation. The probe checks agreement of
output-error estimates and separately reports the difference between actual and
simulated output tensors. Graph replay itself is checked against the native
forward output exactly.

Cost files are version-checked and shape-specific; combine their `benchmarks`
arrays for multiple weight shapes, with one representative M per N/K shape.
The probe uses BF16 activations. Only supply costs measured on the intended
inference device, software and workload. The converter cannot infer that a cost
file from a different device or activation workload is representative. Unmatched
shapes use the documented heuristic. Reports record workload shapes and the cost
file hash. No device identity is written by the probe.

```bash
python -m unittest -q test_quantizer_native.py test_kitchen_pipeline.py
python comfyui_native_quantizer.py --self-test
```

Kernel workspaces remain owned by Kitchen, and live input quantization still
allocates activation/output buffers. This redesign avoids adding a competing
runtime and does not claim to eliminate upstream allocations. Graph replay for
fixed synthetic shapes does not establish graph safety across ComfyUI patching,
dynamic shapes, lazy loading or offloading. Validate those with the actual model
and workflow before deployment.
