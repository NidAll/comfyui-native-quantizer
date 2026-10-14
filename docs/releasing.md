# Release process

## Review the source

Review tracked changes and new files as one intentional payload. Preserve
unrelated work. Exclude model weights, generated media, local instructions,
private reports, credentials, personal paths and machine details. Keep a
reproducible inventory of the source used to build artifacts.

For a new release, choose an unused version and update runtime/package identities
together. Update the changelog, support matrix and validation record. Payload or
metadata contract changes require a format-revision review; resume journals remain
version-bound. Do not replace existing tags or published artifacts.

GitHub Git transport uses SSH. Check the remote and authentication before an
authorized push, preserving host-key verification. Preparing files and artifacts
does not authorize publishing them.

## Run source checks

```sh
python scripts/check.py
python scripts/check.py --comfy-root /path/to/ComfyUI --cuda
git diff --check
```

Add `--iris-root /path/to/ComfyUI-Iris` when that consumer is required. Default
checks use CPU and synthetic fixtures. Requested consumer/CUDA checks fail when
unavailable. Record software versions, allowed skips and actual results; no check
should start services or download model weights. Check documentation links and
CLI examples after changing guides.

## Build and inspect distributions

Use a clean output directory separate from existing release assets:

```sh
python -m pip install 'build>=1.2' 'twine>=5'
python -m build --outdir /path/to/release-artifacts
python -m twine check --strict /path/to/release-artifacts/*.whl /path/to/release-artifacts/*.tar.gz
python scripts/check_distributions.py /path/to/release-artifacts
```

The build creates a source archive and builds its wheel from that archive.
Archive checks verify package identity, complete license, resources, payload
privacy and frontend isolation, then generate `SHA256SUMS`. Confirm changed
source files are present byte-for-byte. The source archive includes the tests,
CLI guides and scripts needed to reproduce validation.

Install the wheel in a fresh environment and run `pip check`. Copy
`scripts/smoke_installed.py` outside the checkout and run it before adding
PyTorch. Then supply the tested compute dependencies and run it with `--compute`.
Also exercise MSE conversion/verification from the installed package. Test the
extracted source archive rather than relying only on an editable installation.

## Publish

After an authorized source push, require all hosted jobs to pass at the reviewed
commit. Tag that exact commit, then publish the validated wheel, source archive,
checksums and support limits. Use prerelease status while the support set remains
under evaluation. A PyPI upload is a separate publishing action.

Promote model/format/backend combinations only with the evidence defined in
[support](support.md). Fix failed gates and rerun the affected checks before
publishing.
