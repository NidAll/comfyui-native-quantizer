# Changelog

## Unreleased

- Remove the optional Textual frontend (Quantizer Studio); the CLI and the
  Python API are the only interfaces.
- Speed up conversion I/O without changing output bytes: hash source shards in
  parallel, overlap native source hashing with encoding, copy payloads in the
  kernel, write passthrough tensors straight to the spool, compute the final
  output digest alongside the payload re-check, and drop an unused full-output
  hash in the planned engine. W4A8 codebook search uses int32 indices.

- Add native Fibo2/MOSS-Image policies and correct fused Ming attention, nested
  Qwen encoder detection and Trellis2/Pixal3D block selection.
- Add exact, offline Comfy-Org reference recipes with per-layer 16/64/256
  ConvRot settings, strict layout/quality gates and a metadata-only extractor.

- Add the conservative MiniCPM-V 4.7 1B INT8 ConvRot recipe, complete model
  detection and protected vision/recurrent/boundary tensors for the separate
  ComfyUI-Anima-Prompt-Architect consumer.

- Fix nondefault W6A8 group-size accounting and parameter-weighted promotions;
  add bounded feasibility repair when greedy allocation exhausts a budget.
- Add opt-in adaptive planning with disjoint calibration rows, per-layer p95
  output gates, measured paired precision exchanges and W6A8/INT8 references.
- Add whole-file compression limits, mixed INT8 ConvRot, workload-aware v3
  costs with BF16 baselines, stratified capture and bounded encoding/codebook reuse.
- Bind mixed resume identity to calibration, planner/cost/reference settings and
  validate rotated INT8 independently through serialization and native forwards.
- Add a separate experimental native Flux 3 Klein policy with complete
  shape-checked detection, fused/split QKV and step-distilled flow-map layouts.
- Select image/reference/text/joint block linears while preserving boundary
  projections, modulation, embeddings and norms; add bounded conversion and
  native block-loading regression coverage.

## 2.3.0rc3

- Add optional MSE scale/clipping and least-squares optimization across native
  quantized formats and mixed candidates, preserving default encoders and native
  inference contracts.
- Support weight, recorded-activation and automatic objectives through CLI, API
  and configuration, including bounded conversion and validation.
- Bind optimizer settings and selected calibration contents to resume identity;
  preserve valid journals when a resume request is rejected.
- Use Kitchen's FP32 eager block encoding on CUDA for NVFP4/MXFP8 without rounding
  source weights to half precision.
- Add synthetic accuracy/conversion/memory benchmarks and native consumer/CUDA
  regressions.
- Rewrite the README around conversion, MSE, native formats and the Python API.
- Group CPU/CUDA regressions under `tests/` and architecture guides under
  `docs/models/`; discover tests through the maintained check runner.
- Consolidate dependency declarations in `pyproject.toml` and generate custom
  architecture templates through the CLI.

See [MSE optimization](docs/mse-optimization.md) for controls and numerical limits.
Published versions and source history are available in the repository's
[releases](https://github.com/NidAll/comfyui-native-quantizer/releases).
