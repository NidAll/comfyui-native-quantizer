# Release process

The prepared identity is CLI/library `2.3.0rc2`. Do not reuse
`v2.2.0`, replace an existing tag or call an unevaluated support set stable.

## Prepare a reviewed source snapshot

Review tracked and non-ignored new files, including the modular backend and
CLI. Keep model weights, generated media, reports with private inputs,
machine details and local instructions out of the public payload. Preserve
unrelated work. Use SSH for GitHub Git transport and verify the configured
remote and authentication before any authorized push.

Update runtime/package identities, changelog, migration,
support matrix and validation record together. Existing format revisions need
their own compatibility review when payload contracts change. Old conversion
journals must remain version-bound.

## Run local gates

Use an environment with the intended PyTorch build:

```sh
python scripts/check.py
python scripts/check.py --comfy-root /path/to/ComfyUI --cuda
git diff --check
```

Default checks are CPU-only. To require the native Iris consumer, add
`--iris-root /path/to/ComfyUI-Iris` alongside `--comfy-root`. Explicit consumer checks fail if the consumer is
missing; CUDA checks fail if CUDA or the six-bit API is unavailable. No check
downloads weights or starts services. Record optional skips and required failures.

## Build and verify artifacts

Use a clean snapshot/output directory and isolated build tooling:

```sh
python -m pip install 'build>=1.2' 'twine>=5'
python -m build --outdir dist
python -m twine check --strict dist/*.whl dist/*.tar.gz
python scripts/check_distributions.py dist
```

`build` creates a source archive and rebuilds its wheel from that archive.
The CLI wheel and source archive include the full license. Both exclude Studio,
its tests, requirements and packaging. The source archive includes CLI tests, guides and scripts. Archive checks verify identities, resources, license,
isolation, payload hygiene and generate `SHA256SUMS`.

Install the CLI wheel into a fresh environment, run `pip check`, and copy
`scripts/smoke_installed.py` outside the source checkout. Run it
before supplying PyTorch; then run `--compute` with the tested compute stack (PyTorch 2.11.0).
The compute smoke must publish and independently verify a tiny checkpoint and
preserve protected tensors. Also rebuild/test the source distributions rather
than relying on an editable install alone. CI performs these gates separately
from source tests.

The package metadata follows the [Python Packaging User Guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/).
Archive contents are controlled using [setuptools manifests](https://setuptools.pypa.io/en/latest/userguide/miscellaneous.html).

## Publish and promote

After an authorized source push, require every hosted CI job to pass at the
reviewed commit. Tag the exact candidate commit and publish a GitHub prerelease
with the two validated distributions, checksums and their evidence limits.
Publication and PyPI upload are separate actions; neither is automatic here.

A stable release additionally needs the named model/runtime acceptance set
described in [support.md](support.md). Keep untested combinations experimental.
If a gate fails, fix the candidate and rerun the affected checks before promotion.
