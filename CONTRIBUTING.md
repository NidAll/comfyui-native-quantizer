# Contributing

Use Python 3.12–3.13 and a separate environment with the intended PyTorch build.
The core does not choose or replace that build.

```sh
python -m pip install -e '.[hub,kitchen,dev]'
python scripts/check.py
```

The default run compiles packages, runs embedded self-tests and exercises
backend, CLI, policy, metadata, packing, API and publication failure paths.
Optional consumer tests skip when no ComfyUI checkout is requested. To require
an installed consumer and native CUDA execution:

```sh
python scripts/check.py --comfy-root /path/to/ComfyUI --cuda
```

This mode fails if its required consumer, CUDA device or six-bit Kitchen API is
missing. It does not launch services. Direct unittest runs can also use
`COMFYUI_ROOT` and `COMFYUI_REQUIRED=1`. The test environment isolates CLI
preferences and stores fixtures in temporary directories.

Read [architecture.md](docs/architecture.md) before backend changes. Keep
frontends out of library modules, reference reconstruction independent from
production encoders, and compatibility facades thin. Preserve source precision
boundaries, bounded working memory, quality gates, input identities and atomic
publication. Add regression checks for the changed behavior and failure modes.

Use source, headers and tiny synthetic checkpoints for routine development.
Tests and benchmarks must not download model weights. Treat architecture
coverage, serialization, consumer loading, execution and model-quality evidence
as separate results; update [support.md](docs/support.md) when that evidence changes.

Issues should include a minimal reproducer, command/options, Python/PyTorch/
Kitchen versions and the consumer revision where relevant. Remove credentials,
personal paths, private prompts and machine identity from submitted material.
Do not attach weights or generated outputs by default.

Build and publication instructions are in [releasing.md](docs/releasing.md).
