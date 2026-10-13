# Changelog

## 2.3.0rc2 — CLI candidate prepared 2026-10-13

Supersedes the combined core/Studio candidate; tagged distribution publication
remains separate from main-branch source publication.

- Exclude Studio source, packaging, tests, dependencies and launchers from the
  CLI release. Preserve deferred local frontend work separately.
- Add distinct native-layout Iris-3B, Kandinsky 6 video/audio and Kandinsky 6 SR
  policies with complete shape-checked signatures and conservative boundaries.
- Preserve unaligned Iris projections at source precision in all planning modes;
  never pad or select a known unsafe native integer shape automatically.
- Delay event-file opening until source/shard discovery and destination checks.
  Reject aliases to sources, defaults, journals and control inputs before writing.
- Validate typed API enum options and close replaced log handlers.
- Release self-test scratch after each case, including failures, to bound disk use.
- Runtime-test clean CPU dependency setups on Python 3.12/3.13; scope the package
  support range to those tested interpreters. Add CLI-only installed archive checks.

## 2.3.0rc1 — prepared 2026-10-13

Release candidate for the maintained modular V2 base; publication is pending.

- Move backend ownership into `native_quantizer/`, with typed conversion requests,
  a lazy public API and thin compatibility entry points.
- Package Quantizer Studio independently as `0.2.0rc1`; keep Textual out of core.
- Add native W6A8 conversion and explicit mixed-mode opt-in, preserving grouped
  scales, ConvRot, independent decoding and consumer compatibility gates.
- Preserve per-layer metadata through ComfyUI prefix stripping; retain safe
  architecture detection and protected precision boundaries.
- Add portable Linux commands, XDG defaults, diagnostics and shell completion.
- Isolate tests from user preferences and require explicit ComfyUI integration.
- Expand CI to the modular backend, CLI, Studio, distributions and installed API.
- Include licenses, guides, tests and build inputs in intentional source archives;
  add wheel audits, checksums, migration and support documentation.

Conversion engines retain their distinct resume capabilities. Existing native
payload revisions are unchanged. Older in-progress journals are version-bound
and must not be resumed across this converter identity change.

## Earlier releases

`v2.2.0` belongs to the preceding script-based release. This candidate does not
replace that tag or reissue its package identity. Consult repository history
for earlier changes.
