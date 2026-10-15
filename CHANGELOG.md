# Changelog

## Unreleased

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
