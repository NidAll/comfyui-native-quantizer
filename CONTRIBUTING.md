# Contributing

Use Python 3.12–3.13 and the compute environment that supplies the intended
PyTorch build. Install development dependencies from the checkout:

```sh
python -m pip install -e '.[hub,kitchen,dev]'
python scripts/check.py
```

The default checks compile sources and run synthetic backend, CLI, API, metadata,
packing, policy, MSE and failure-path tests with isolated preferences. Consumer
integration is optional unless requested explicitly:

```sh
python scripts/check.py --comfy-root /path/to/ComfyUI --cuda
```

Add `--iris-root /path/to/ComfyUI-Iris` to require that consumer too. Requested
integration fails when its consumer, CUDA backend or required Kitchen API is
missing. Checks do not launch services or download model weights.

Read [module ownership](docs/architecture.md) before implementation changes.
Keep library imports independent of the CLI and optional frontends. Preserve
native layouts, protected precision, bounded work, quality gates, input identity,
resume integrity and atomic publication. Keep reference reconstruction separate
from production encoding. Add focused regressions for changed behavior under `tests/cpu/` or
`tests/cuda/`; shared consumer/preference helpers live in `tests/support.py`.
Run a focused CPU suite with `python -m unittest -q tests.cpu.test_quantizer_native`.

Use source, headers and small fixtures for routine development. Record
serialization, consumer loading, native execution and model-quality evidence
separately. Update [support](docs/support.md) and [validation](docs/release-validation.md)
when the evidence changes. Documentation examples must match the current CLI/API;
check local links and built archives after moving or removing guides.

Reports and submissions should include a minimal reproducer, command/options,
software versions, relevant consumer revision and redacted failure evidence.
Exclude credentials, personal paths, machine identity, model weights and private
inputs or generated media.

See the [release process](docs/releasing.md) for source review, distribution
checks, installed-package testing and publication gates.
