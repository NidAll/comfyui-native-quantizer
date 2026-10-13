# Linux command-line integration

Use the existing ComfyUI Python environment; the installer does not invoke pip,
fetch weights or change ComfyUI. From this checkout:

```sh
python scripts/install_linux.py --cuda-default --offline-default
```

`--cuda-default` verifies the live CUDA backend before saving the preference.
Omit it to retain portable auto/CPU settings. `--python` selects an existing
interpreter. The default user install provides:

- `comfyquant` and `comfyui-native-quantizer` in `~/.local/bin`, preserving the
  caller's working directory and relative input paths.
- Fish, Bash and Zsh completion definitions. Fish loads its per-command files
  automatically. Bash can source the file under
  `~/.local/share/bash-completion/completions/comfyquant`. For Zsh, add
  `~/.local/share/zsh/site-functions` to `fpath` before `compinit` if necessary.
  No shell startup files are rewritten.

## Configuration and scripting

Defaults are data-only JSON in
`$XDG_CONFIG_HOME/comfyui-native-quantizer/config.json`, falling back to
`~/.config/comfyui-native-quantizer/config.json`. CLI options override the file.
`--config PATH` chooses a file; `--no-config` gives reproducible built-in
defaults. Unsupported keys and invalid values fail clearly. Destructive paths,
overwrite and resume cannot be enabled through this defaults file.

```sh
comfyquant --show-config
comfyquant --diagnose --json
comfyquant --list-formats --json
comfyquant --list-architectures --json
comfyquant --audit-architectures /path/to/ComfyUI --json
comfyquant input.safetensors --format int8-convrot --output output.safetensors --validate
comfyquant input.safetensors --inspect --json
comfyquant --verify-output output.safetensors --json
```

`device=auto` selects CUDA/ROCm when available, otherwise CPU. An explicit
`device=cuda` or `rocm` fails if that backend is unavailable. `--device cpu`
overrides a saved CUDA preference. CUDA math stays FP32 unless an applicable
format is explicitly configured otherwise. The memory limit bounds temporary
conversion work, not total process/device residency.

`offline=true` prevents CLI model downloads; repository metadata
browsing remains available. A deliberate `--no-offline` overrides the CLI
setting. Development and architecture evaluation must still follow the
no-heavy-download rule in AGENTS.md and the integration skill.

Help, version, catalog lists, completions and registry audits do not import
PyTorch. Usage errors exit with 2, conversion/verification failures
with 1, success with 0 and keyboard cancellation with 130. JSON is supported
for the operations shown above; conversion progress remains on stderr and
the existing JSONL event stream remains available.

## Conversion changes and evidence

Native row chunks are copied/cast directly into the selected device, avoiding
an intermediate FP32 CPU buffer. Reconstruction checks group their scalar
results into one device-to-host transfer. Quality gates and format payloads
remain unchanged. Native publication rejects source inode/size/mtime/ctime
changes observed during conversion, supplementing the existing output race
checks; source content identities remain embedded in provenance.

Use bounded synthetic fixtures to measure cold and warm behavior separately
and compare payload hashes on the same backend. Timing is workload-dependent;
these optimizations do not establish whole-model image quality, conversion
throughput on every format, inference speed or peak VRAM.

The `comfyui-quantizer-architecture` skill lives under `skills/` and is installed
in the user's Codex skills directory. See
[architecture-candidates.md](architecture-candidates.md) for the current
source-only shortlist. The registry audit finds newly added ComfyUI classes;
it does not prove every policy's tensor selection or native loader behavior.
