# V2 base and release-readiness assessment

**Historical assessment.** The 2026-10-13 preparation implements the release
engineering changes below for `2.3.0rc2` (CLI-only). Current status and remaining gates
are in [support.md](support.md) and [release-validation.md](release-validation.md).
The following findings describe the earlier snapshot.

Assessment date: **2026-10-08**. This records the local candidate's readiness,
not a completed GitHub release or a certification of every model/runtime.

## Verdict

The modular implementation is a sound **V2 foundation**. The current checkout
still needs release engineering and a defined support scope before a stable
production claim. The recommended next deliverable is a V2 release candidate
built from a reviewed source snapshot, followed by a stable release for the
specific combinations that have passed consumer and model-level evaluation.

The existing modular code should become the maintained base. Existing native
codecs and their regression evidence should be retained. New development should
enter the owning backend modules, public API or optional frontend. The original
script remains a compatibility adapter with a documented support policy.

V2 does not require discarding tested algorithms or rewriting them in another
language. It needs enforceable boundaries, portable distribution, versioned
contracts and evidence that matches each advertised capability.

## What is already strong

- Backend ownership is explicit: checkpoint I/O, detection/policies, codecs,
  planning, execution, runtime and independent validation have separate modules.
- The typed conversion request and public API provide an integration surface.
  Core import and lightweight CLI commands avoid compute/frontend startup.
- Core and Studio are separately packaged; the core excludes Textual and the UI.
- Sources, protected tensor boundaries, reconstruction gates, resume identities
  and atomic publication are preserved. Explicit unavailable GPU requests fail.
- Linux commands, completion, diagnostics, JSON inspection and configuration
  are implemented. CUDA preference belongs to local configuration.
- W6A8, prefix-safe metadata, detector regressions, Studio state/worker behavior
  and source-mutation checks have meaningful development evidence.
- Heavy model downloads are prohibited for development. Source/header analysis
  and tiny fixtures support routine work without acquiring large checkpoints.

These strengths justify keeping this architecture as the V2 base. They do not
certify all architecture policies or all advertised formats for generation.

## Confirmed release gaps

| Gap | Current evidence | Required result |
| --- | --- | --- |
| CI coverage | `.github/workflows/tests.yml` runs self-tests and `test_quantizer_native.py`; it compiles only the compatibility script. | CI must cover the modular package, Kitchen, detection, metadata, W6A8, boundaries, Linux CLI and separate Studio suites; build and exercise distributions outside the checkout. |
| Portable integration tests | Three test files hardcode a private ComfyUI checkout: `test_quant_metadata.py`, `test_w6a8.py`, `test_w6a8_cuda.py`. | Use an explicit external ComfyUI root and reproducible integration setup. Standalone tests must work without that checkout; integration jobs must fail clearly when their required consumer is missing. Remove private paths from the release payload. |
| Distribution completeness | Both wheels/sdists build. Studio wheel and sdist contain no license file. Core sdist contains no Linux installer, guides or external tests. | Ship the Studio license and intentionally define each archive's contents. Ensure documented setup/test commands are available in the artifact for which they are advertised. A core-only sdist can intentionally exclude Studio if its documentation points to an available separate distribution. |
| Release identity | Runtime/core metadata is `2.2.0`; a local `v2.2.0` tag already exists for the earlier single-file implementation. Studio pins core `==2.2.0`. | Assign a new unreused prerelease identity; align runtime, package metadata and frontend dependency. Never replace the existing tag. Add a changelog and migration note. |
| Python/dependency contract | Packages declare Python >=3.10; existing CI covers 3.12/3.13. PyTorch is deliberately supplied by the selected environment. Kitchen extra permits >=0.2.33 while W6A8 needs >=0.2.36 and matching ComfyUI registration. | Test the advertised Python range or narrow it. Document external PyTorch setup and feature-specific dependency floors. Dependency metadata alone must not imply that every native format will execute in every consumer. |
| Support claims | Detector and synthetic coverage is broad; current records do not certify full-model behavior for every architecture/format/backend combination. | Publish an evidence matrix separating policy coverage, file integrity, consumer loading, real forward execution and model-quality/performance evaluation. Mark unevaluated combinations experimental. |
| Release snapshot | The new modular tree, packaging, Studio and several tests remain untracked or modified locally. | Review an intentional file inventory, preserve unrelated work, and create a reproducible candidate snapshot with its exact validation results. |

The archive-content findings concern generated Python distributions. A GitHub
source checkout contains files that the current core sdist omits. Lack of those
files in the sdist does not mean they are absent from the repository.

## Fresh checks performed for this assessment

1. Read the current guidance, package configuration, source boundaries, CI,
   documentation and Git status.
2. Ran `test_project_structure.py` and `test_linux_cli.py`: **20 tests passed**.
3. Built core and Studio wheels and source distributions from a temporary
   snapshot of the current non-ignored tracked/untracked files, without network
   installation. Inspected archive contents and dependency metadata.
4. Extracted the wheels to a temporary import location and ran help, JSON format
   listing and resource/import checks from outside the checkout.
5. Converted an approximately 82 KB synthetic checkpoint through the packaged
   typed API on CPU with validation enabled; conversion returned 0 and
   source-free output verification passed.
6. Ran `git diff --check`: passed.

The package smoke checks used the existing development interpreter and its
dependencies. They prove independence from the source checkout; they do not
prove dependency resolution in a fresh virtual environment. GPU tests,
full-model generation and GitHub-hosted CI were not rerun for this assessment.

Historical 7 October evidence includes 57 self-tests, 115 unique relevant
unit/interaction tests and one synthetic CUDA loader regression after fixes.
Those counts overlap the current targeted checks and must not be added together.

## Proposed V2 maintenance contract

| Area | V2 contract |
| --- | --- |
| Maintained core | `native_quantizer/` owns behavior; compatibility facades only adapt old entry points. |
| Public API | Typed requests, documented errors/results and an explicit compatibility policy; frontend code consumes public operations. |
| File contracts | Versioned native metadata, format revisions, input identities and independent validation; existing native payloads remain covered by regression fixtures. |
| Frontends | CLI is an adapter; Studio is independently installable and optional. GUI dependencies do not enter core. |
| Architectures | Complete component-scoped signatures, safe precision boundaries, variant separation and explicit evidence levels. |
| Execution | Preserve the two current executor contracts. A shared executor is a separate migration with parity and failure-path evidence. |
| Runtime | Portable default selection, local CUDA preferences and clear unavailable-backend failures. Keep accelerated execution claims consumer-specific. |
| Development | No heavy model downloads; use source, headers and small fixtures. Required checks accompany behavior changes. |
| Release evidence | Exact source version, dependency/consumer versions, reproducible commands, allowed skips and explicit limitations. Public payloads omit private paths and machine identity. |

## Concrete route to a release

### Gate A: GitHub V2 release candidate

Prepare the reviewed source inventory; remove private paths; fix archive/license
contents; assign a new prerelease version; update the frontend dependency;
add V2 installation/migration/changelog documentation; and extend CI to the new
surface. Test installed wheels and rebuilt sdists in fresh environments using
the declared dependency setup. Keep optional consumer integration in a distinct,
configured job and synthetic offline fixtures in the default job.

An RC may expose experimental policies when their status and limitations are
clear. A universal executor, additional candidate architectures and every
proposed quality-search optimization are not prerequisites for that RC.

### Gate B: stable support claims

Select a small representative initial support set. Use already installed models
when possible and compare original versus converted output through the intended
ComfyUI consumer. Record real forward/generation success, quality comparison,
throughput, peak memory, offload and applicable LoRA behavior separately. Exercise
cancellation, failed validation, existing-output protection and resume where the
engine supports it. Promote only the evaluated combinations; retain the other
policies as experimental.

Do not acquire heavy models to fill this matrix without a direct artifact
request. Stable scope can remain deliberately narrower than detector coverage.

## Sources and scope

Source ownership is documented in [architecture.md](architecture.md), command
installation in [linux-cli.md](linux-cli.md), and W6A8 conditions in
[w6a8.md](w6a8.md). The packaging/release review also consulted the official
[Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
and [GitHub release documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).

This task changed assessment/guidance documents only. It did not fix the gaps
listed above, rename versions, rewrite conversion code, create tags, commit,
push, create a repository or publish a release.
