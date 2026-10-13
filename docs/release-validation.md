# CLI candidate validation — 2026-10-13

Prepared identity: **CLI/library 2.3.0rc2**. This supersedes the combined RC1
candidate. Studio is preserved as deferred local work and excluded from this
release's source inventory, distributions, dependencies, installer and CI.
At completion of local preparation, no candidate commit, push, tag, GitHub
release or PyPI upload had been created. Main-branch source publication does not
promote this candidate to a stable release.

## Local source checks

| Check | Result |
| --- | --- |
| Embedded self-tests | 57/57 passed |
| Fresh Python 3.12 CPU environment | 102 tests passed; four optional consumer tests skipped |
| Fresh Python 3.13 CPU environment | 102 tests passed; four optional consumer tests skipped |
| Required installed ComfyUI and Iris consumers | 102 tests passed without skips |
| Native W6A8 CUDA | One test covering 12 shape/group/batch cases passed |
| Native Iris INT8 CUDA | One test covering six aligned real-shape/batch cases passed with weight dequantization blocked |
| Architecture and CLI failure-path follow-ups | Passed after final detector and telemetry guards |

Both fresh CPU environments resolved the CLI's declared dependencies separately
and used PyTorch 2.11.0+cpu, NumPy 2.5.4, safetensors 0.8.0 and comfy-kitchen
0.2.37. Python versions were 3.12.14 and 3.13.15; neither environment supplied
Studio or Textual. The package's supported Python range is narrowed to 3.12–3.13.

The installed consumer revisions were ComfyUI
`7bf5df9e38763dbdd94a0f9c939273019025c2bc` and native Iris
`4acb9e9110c591e74cfffc481fa1d2fb0dc75dda`. Iris validation performs strict
native loading and a complete tiny forward after conversion. Separate CUDA
probes use aligned projection shapes `(N,K)=(2560,2560), (6912,2560),
(2560,6912)` at batch sizes 1 and 16. Original 6,826-wide MLPs are guarded and
remain dense; explicit includes do not bypass this boundary.

W6A8 probes use row counts 64/65, K=256, groups 16/32/64 and batches 1/64,
covering native forwards, direct Kitchen agreement, movement, save/reload and
aligned graph replay. The odd-row fallback capture exception remains documented
in [w6a8.md](w6a8.md).

Kandinsky 6 video/audio and SR have separate, shape-checked native policies,
independent synthetic conversion and protected-tensor verification. Full
extension forwards remain untested; see [kandinsky6.md](kandinsky6.md).

Telemetry and output-path tests reject source/shard/index, hard-link,
symlink-directory, defaults and journal collisions before writing. Rejected CLI
requests clear previous event sinks. Typed API enum validation precedes telemetry.
Self-test scratch is reclaimed after each case, including failure paths.

## Distributions

The CLI wheel is rebuilt from its source archive. Strict metadata and archive
checks pass for both artifacts, including full Apache-2.0 license, identities,
resources, private-path/hardware hygiene and exclusion of Studio. The wheel
contains 87 files; the source archive contains 133 files, including CLI guides,
scripts and tests. Checksums accompany these two artifacts.

A fresh minimal environment passed dependency checks and installed CLI/resource
smoke outside the checkout with no PyTorch, Studio or Textual. Installed-wheel
CPU conversions also passed on both Python 3.12 and 3.13 using independently
resolved compute dependencies, preserving protected tensors and independently
verifying the output. The extracted source archive passed all 57 self-tests and
102 core tests (four explicitly optional consumer skips). Workflow YAML, local
documentation links and Git whitespace checks pass.

The installed-class AST audit reports 105/105 covered model classes and four
explicitly unsupported auxiliary classes. This is policy coverage, not complete
model/runtime quality acceptance.

## Remaining publication and stable gates

The CLI-only GitHub matrix is configured for Python 3.12/3.13 and clean
installation/build checks. Hosted CI was not run during local preparation;
check [GitHub Actions](https://github.com/NidAll/comfyui-native-quantizer/actions)
for the published commit's results.
The tested dependency and consumer versions do not certify other combinations.

No model weights were downloaded, services restarted or full-model generations
run. Model-level quality, full workload throughput, peak inference memory,
offload and LoRA compatibility were not re-evaluated. Keep unevaluated
architecture/format/backend combinations experimental. Require hosted CI and a
named model/runtime acceptance set before publishing a stable tag.
