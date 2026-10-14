# Validation coverage

Run checks against the exact source or installed artifact being evaluated. Keep
software versions, required consumers, skipped checks and model-level results
explicit; a source test is not an installed-package or generation test.

## Reproduce checks

```sh
python scripts/check.py
python scripts/check.py --comfy-root /path/to/ComfyUI --cuda
```

The default run isolates preferences and uses CPU fixtures. `--comfy-root`
requires the named consumer. Add `--iris-root /path/to/ComfyUI-Iris` to require
Iris integration. `--cuda` requires CUDA and Kitchen APIs rather than passing by
skipping native execution. These commands do not start services or fetch models.

## Exercised contracts

| Area | Checks |
| --- | --- |
| Source/core | 57 embedded self-tests and 120 CPU tests covering APIs, CLI, policies, codecs, MSE, metadata and failure paths |
| Consumer integration | Native ComfyUI metadata/load/save, complete tiny Iris forward and protected precision |
| Integer CUDA | W6A8 shape/group/batch cases, aligned Iris projections, native MSE forwards and graph replay with floating-weight dequantization blocked |
| Bounded CUDA conversion | W4A8/W6A8/mixed streaming with recorded-activation objectives and a small work budget |
| Block formats | FP32 NVFP4/MXFP8 CPU/CUDA encoding agreement and native CPU consumer loading |
| Recovery | Source/option/calibration identity, partial writes, corruption and rejected resume journal preservation |
| Distributions | Source archive/extracted-source wheel contents, license/resources/privacy, lazy imports and installed conversion/verification outside the checkout |

The 2.3.0rc3 MSE source is validated locally in Python 3.12 with PyTorch 2.11.0 and
comfy-kitchen 0.2.37. The configured CI matrix covers Python 3.12 and 3.13; its
status must be checked for each pushed commit. Local results do not establish a
fresh Python 3.13 or hosted-CI result for an unpushed change.

## Numerical and runtime benchmarks

[MSE benchmarks](mse-optimization.md#regression-tests-and-benchmarks) use seeded
Gaussian/outlier weights and synthetic activation rows. They compare baseline
and optimized weight/output MSE, first/warm conversion time, process RSS and CUDA
allocations. Native integer probes measure forwards, graph replay and temporary
allocations independently of conversion. Results depend on the selected fixture,
format, objective, backend and software; no fixed percentage improvement is
promised.

Full-model quality, workflow throughput, inference memory, complete activation
capture, offload/LoRA behavior and accelerated NVFP4/MXFP8 inference require
separate evaluation. Kandinsky extension forwards and real video/audio/SR quality
remain unevaluated. Publish only the support established by the exact release's
checks; see [support](support.md) and [releasing](releasing.md).
