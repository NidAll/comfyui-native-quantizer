# Migrating to the modular V2 base

`2.3.0rc2` introduces a new release identity for the modular implementation.
Existing `v2.2.0` remains the preceding script-based release.

| Previous integration | Maintained integration |
| --- | --- |
| `python comfyui_native_quantizer.py ...` | Still supported; equivalent to `python -m native_quantizer ...` |
| Import helpers from the script | Use `native_quantizer.api` and `ConversionOptions` for new integrations |
| Script copied into an environment | Install the core wheel or source package; all owning modules are required |
| Frontend alongside conversion code | Studio remains deferred local work and is excluded from this release |
| Hardcoded command defaults | Explicit API options or validated CLI XDG defaults |

`comfyquant` and `comfyui-native-quantizer` invoke the same CLI. The source-based
Linux installer provides CLI commands and completion; keep its checkout
available while using those generated launchers.

Native format names and payload revisions retain their existing contracts.
W6A8 is an additional format and remains off by default in mixed planning.
Prefixed checkpoints store native per-layer `.comfy_quant` tensors; bare
checkpoints can keep header metadata. The verifier accepts both forms.

Resume identities include the converter version. Finish an old conversion with
its original version or start a new output with the new version; do not reuse
an older journal as a new candidate conversion. W4A8, W6A8 and mixed conversions
support matching resume state. Other native engines restart atomically.

An explicit unavailable CUDA/ROCm request fails instead of silently using CPU.
For reproducible CLI scripts, use `--no-config` and explicit device/runtime options.
Library conversion requests are independent of CLI preference files.

Python support is scoped to the runtime-tested 3.12–3.13 range. PyTorch remains environment-supplied.
The Kitchen extra requires >=0.2.36,<0.3 and W6A8 also probes the explicit
six-bit API; matching ComfyUI algorithm registration remains a consumer requirement.
