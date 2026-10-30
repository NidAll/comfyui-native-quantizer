# Adaptive mixed precision

`--profile adaptive` selects native formats using recorded activations, measured
workload costs and hard storage limits. It is opt-in; existing profiles and
`optimizer=none` defaults remain unchanged. W6A8 still requires `--enable-w6a8`.

## Quality-preserving precision exchanges

The planner can pair a fast W6A8→W4A8 change on one layer with a compensating
W4A8→W6A8/INT8 change elsewhere. A pair is accepted only when measured cost
improves by at least 2% of the affected layers' current cost, parameter-weighted
held-out error does not increase, each layer remains within its error gate,
and storage/original-precision limits hold. BF16/source precision can also
participate when it has matching measurements.

This differs from assuming that fewer bits are always faster or that six bits
are always more accurate. Codebook W4A8 and uniform W6A8 have different grids;
their quality and speed rankings depend on weights, activations and workload.
The measured cost is a linear-kernel estimate, not an end-to-end speed guarantee.

Adaptive planning reserves every fourth recorded row for candidate evaluation.
The other rows are used by optional MSE encoding. Every targeted layer needs at
least eight rows. The gate metric is the 95th percentile of sampled relative
runtime-output errors per layer, including rotation and activation quantization.
Its global score is the parameter-weighted average of these layer metrics.
It is neither a whole-model 95th percentile nor a perceptual-quality score.
Scoring simulates the native integer grids in FP32; fused reduced-precision
rotation can differ, so final native forwards remain a separate acceptance step.

`--quality-reference w6a8` caps that aggregate error at the measured W6A8
reference as well as the configured global gate. Ineligible W6 layers use plain
INT8, or source precision if INT8 is unavailable; the report lists fallbacks.
`--quality-reference int8` uses the same mechanism with an INT8 reference.
The default `none` uses the profile/global gate. Shape and architecture guards
are preserved; embeddings remain opt-in. Adaptive mode preserves source dtype
on passthrough tensors and rejects an explicit `--output-dtype` cast.

## Measure the intended workload

Create a workload list using actual linear shapes, flattened activation-row
counts and invocation frequencies. Optional `layer` keys must exactly match
checkpoint weight keys. A per-layer workload overrides generic samples for
the same N/K. Weights represent relative invocation frequency; costs are summed
with those weights, so cached conditioning and repeated denoiser calls can
receive different weights.

```json
[
  {"shape": {"m": 16, "n": 1024, "k": 4096}, "weight": 1},
  {"shape": {"m": 256, "n": 1024, "k": 4096}, "weight": 4}
]
```

```sh
quantizer-kitchen-benchmark --comfy-root /path/to/ComfyUI \
  --workloads workloads.json --repeats 50 > kernel-costs.json
```

The command emits `kitchen-costs/v3`, including multiple M workloads, BF16
baseline costs and INT8 ConvRot. It currently measures BF16 activations and the
configured W6 group size. The private runtime fingerprint binds measurements
to the device, PyTorch/CUDA/HIP and Kitchen version without printing a device
name. Rerun measurements after changing that runtime. Keep measurement profiles
from different devices or runtime versions separate.

Legacy v1/v2 files remain readable for existing profiles, with their single-M
assumption. Adaptive mode requires v3. An unmeasured candidate cannot drive a
precision exchange; missing measurements are not inferred from bit width.

## Convert with explicit quality and size limits

```sh
comfyquant --no-config input.safetensors --format mixed --profile adaptive \
  --enable-w6a8 --quality-reference w6a8 \
  --calibration-source activations.npz --kernel-costs kernel-costs.json \
  --target-runtime nvidia --target-compression 2.5 \
  --optimizer mse --mse-objective activation --max-memory 512M \
  --output output-adaptive.safetensors --report conversion --validate
```

`--target-compression` limits the final file to the summed source file sizes
divided by the requested ratio. Planning includes protected tensors and reserves
space for metadata; final publication checks the actual complete file. The
reserve is conservative, and a plan can fail at the final check if metadata
outgrows it. Existing outputs are preserved on failure. The selected-linear
budget remains a separate hard constraint; override it explicitly when needed.

Infeasible quality/size combinations fail rather than relaxing a quality gate.
W6-equivalent error and 2.5× compression are not always achievable together.
The bounded feasibility repair improves greedy allocation but does not promise
the globally optimal assignment. Review `reference_error`, `global_mean_error`,
`precision_exchanges`, `int8_convrot_layers`, and `reference_fallbacks` in the
mixed-plan report.

INT8 ConvRot is automatically considered by the adaptive profile on eligible
256-aligned inputs. Other mixed profiles can opt in with `--enable-int8-convrot`.
It retains native `int8_tensorwise` metadata with per-layer rotation options.

Adaptive planning reuses selected encodings within a bounded cache and reuses
fitted W4A8 codebooks when payloads do not fit. Source hashes are checked across
planning; resume also binds the planner revision, cost-file hash, calibration
contents and quality reference. Chunked jobs retain streaming and independent
validation. Reports identify payload/codebook reuse counts.

## Capture representative stages

The capture helper can reserve separate reservoirs for known stages or branches:

```python
from native_quantizer.calibration_capture import CalibrationCapture

with CalibrationCapture(selected_modules, rows=96, rows_per_call=8,
                        strata=("early", "middle", "late")) as capture:
    for stage in ("early", "middle", "late"):
        with capture.stratum(stage):
            run_baseline_stage(stage)
capture.save("activations.npz")
```

The owning eager baseline runner must define the stages using the actual
schedule. The helper does not drive ComfyUI workflows. Inputs sample randomized
token positions, preserve per-stratum capacity despite unequal call counts,
and record sampled counts in the NPZ manifest. Layers executed in only some
strata report zero counts for the others. Inspect coverage and use all steps
for short distilled schedules where practical. Hold out additional prompts,
seeds, resolutions and reference inputs for final model acceptance.

## Evidence limits

This policy applies to eligible linears through the shared architecture
classification path. It does not add loaders for previously unsupported models,
quantize protected convolutions/expert banks, or certify every architecture and
backend. Native serialization, synthetic forwards, block quality, generated
image/video/audio quality, offload and adapters remain separate checks.

Use a matched BF16 workflow to validate complete outputs before describing a
result as near lossless. Calibration rows guide the planner; repeated selection
on them still requires a separate final evaluation set.
