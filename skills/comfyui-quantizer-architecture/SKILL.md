---
name: comfyui-quantizer-architecture
description: Add or audit architecture detection and safe tensor policies in the ComfyUI Native Quantizer using source inspection, headers and small fixtures. Use for new diffusion, video, audio or text-encoder architecture support in this quantizer, not model downloads or ComfyUI loader implementation.
---

# Quantizer architecture integration

Work in the requested native-quantizer checkout. Read workspace and project AGENTS.md, workspace MEMORY.md and `docs/architecture.md`. Use the workspace's ComfyUI Python environment. Preserve uncommitted work and source checkpoints.

Do not download heavy models for architecture research, tests or benchmarks. Prefer installed model definitions, authoritative source/configuration and headers, and tiny synthetic fixtures. Do not run upstream examples that automatically fetch weights. Respect explicit user scope: an audit or proposal does not authorize implementation or a loader installation.

## Identify the integration boundary

Run `comfyquant --audit-architectures COMFY_ROOT --json` against installed ComfyUI and compare `--list-architectures --json`. Registry coverage is only a starting point: a listed family can still have stale or unreachable tensor patterns. Read the relevant `comfy/model_detection.py`, `comfy/supported_models.py`, model definitions and native load/state-dict conversion path. For a new release outside installed ComfyUI, consult primary upstream sources and record the reviewed revision.

Determine checkpoint namespace, component, input/output projections, repeated block linears, fused or split projections, sensitive tensors, loader key transforms, and architecture variants. Inspect header shapes rather than filenames. An architecture policy does not create a model loader; if the actual consumer lacks native `.comfy_quant`/quantized-operation support, report that dependency and keep the support experimental or deferred. Never label a custom-loader format as native ComfyUI output.

## Implement in the owning modules

- `native_quantizer/architectures/detection.py`: add complete component-scoped names and necessary shape evidence. Handle bare and supported prefixes. Do not use isolated markers or broad substring guesses; partial and ambiguous signatures must fail. Add variant discrimination before selecting a workflow or policy.
- `native_quantizer/architectures/policies.py`: register exact supported model classes and patterns for actual repeated 2D Linear weights. Preserve input/output, embeddings, normalization, modulation, routers, auxiliary towers and unsupported expert layouts. Quantize refiners or fused projections only when the real loader keeps their format/shape contracts intact. Avoid widening another family's rules to absorb a distinct architecture.
- `native_quantizer/architectures/custom.py`: use the existing data-only policy/key-map interface for a custom namespace. Generate a starter with `--write-architecture-template`; do not install executable Python through a policy. Exact remaps must stay collision-checked. Built-in support needs a real detector and regression coverage, not just a JSON template.

Reuse existing codecs, formats, metadata, planning and runtime capabilities. An architecture addition normally does not require changing quantization algorithms. Preserve source dimensions: unsupported ConvRot/packing alignment keeps tensors at source precision instead of silently padding them. Keep per-projection Q/K/V metadata separate when the loader requires it. Preserve the distinction between conversion device and target inference runtime.

## Prove routing and policy behavior without large weights

Extend `test_architecture_detection.py` with independent header fixtures based on inspected source, including bare/prefixed keys, partial signatures, realistic shape boundaries, variants and collisions with the nearest families. Add real block-name selection fixtures that prove intended linears are reached while sensitive tensors are kept. Include ordinary and malformed custom policy/remapping cases if those paths change.

Create a tiny synthetic checkpoint with the architecture's exact names and packing-compatible reduced dimensions. Run inspection/dry-run, conversion with `--validate`, and source-free `--verify-output`. Assert protected tensors retain their original dtype/bytes and selected metadata survives the consumer's prefix/key transforms. When a genuine native loader can be exercised using synthetic tensors, test that contract too; otherwise state the missing evidence.

Required checks, using the workspace interpreter:

```sh
python comfyui_native_quantizer.py --no-config --self-test
python -m unittest -q test_quantizer_native.py test_kitchen_pipeline.py test_architecture_detection.py
```

Add metadata, W6A8, CLI or project-structure suites when their contracts change. Recheck CUDA before GPU tests in the same access context; a sandbox failure can hide the device. Use bounded fixtures and output directories outside ComfyUI core. Never claim synthetic/header proof establishes generation quality, full-model latency, peak VRAM or LoRA behavior.

Update the format/architecture guide with supported variants and actual validation limits. Record the task in workspace AGENTS.md and MEMORY.md, including source revision, affected paths, implemented/deferred status and remaining loader/runtime evidence. Do not commit, push or publish unless separately authorized.
