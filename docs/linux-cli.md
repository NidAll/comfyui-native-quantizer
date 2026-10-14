# Installation, commands and configuration

Use an existing Python environment with the appropriate PyTorch build. A package
installation creates `comfyquant` and `comfyui-native-quantizer` console scripts
in that environment's `bin` directory:

```sh
/path/to/environment/bin/python -m pip install /path/to/package.whl
/path/to/environment/bin/comfyquant --version
/path/to/environment/bin/python -m native_quantizer --no-config --list-formats
```

Activate the environment or add its `bin` directory to `PATH` to use the short
commands. For a local wheel rebuilt with the same version, force replacement
without changing compute dependencies:

```sh
/path/to/environment/bin/python -m pip install --no-index --no-deps \
  --force-reinstall /path/to/package.whl
```

Check installed imports and run a synthetic conversion outside the source
checkout. An editable install loads the checkout and does not validate wheel
contents.

## Shell completion and source-checkout launchers

`comfyquant --completions fish`, `bash` or `zsh` emits completion definitions.
Place them in the shell's normal completion directory.

The optional `scripts/install_linux.py` helper installs user-local launchers,
completion files and XDG defaults from a source checkout:

```sh
python scripts/install_linux.py --offline-default
```

Its generated launchers load that checkout directly; keep the checkout present.
For installed-wheel testing, use the environment's console scripts instead.
`--python` chooses the interpreter. `--cuda-default` verifies CUDA before saving
that preference. The helper does not install packages, download models or modify
shell startup files.

## Defaults

CLI defaults live in `$XDG_CONFIG_HOME/comfyui-native-quantizer/config.json`, or
`~/.config/comfyui-native-quantizer/config.json` when XDG is unset:

```json
{
  "schema": 1,
  "device": "auto",
  "target_runtime": "auto",
  "max_memory": "512M",
  "offline": true,
  "optimizer": "none",
  "mse_objective": "auto",
  "mse_steps": 17,
  "mse_iterations": 3
}
```

CLI flags override the file. `--config PATH` selects another file; `--no-config`
uses built-in defaults. Invalid or unknown values fail clearly. Overwrite and
resume cannot be enabled through this file. Python API requests remain explicit.

`device=auto` uses an available accelerator or CPU. Explicit CUDA/ROCm requests
fail when unavailable. `--device cpu` overrides a saved accelerator preference.
The memory budget estimates temporary conversion work, excluding mmap pages,
framework overhead and allocator caches.

`offline=true` prevents model downloads; a deliberate `--no-offline` overrides
it. Repository metadata browsing remains available. Development checks and
benchmarks use synthetic fixtures rather than downloading weights.

## Inspection and scripting

```sh
comfyquant --no-config --diagnose --json
comfyquant --no-config --list-formats --json
comfyquant --no-config --list-architectures --json
comfyquant --no-config --audit-architectures /path/to/ComfyUI --json
comfyquant --no-config input.safetensors --inspect --json
comfyquant --no-config --verify-output output.safetensors --json
```

Help, version, catalogs, completion and architecture audits start without
PyTorch. Exit codes are 0 for success, 1 for operation failure, 2 for usage errors
and 130 for keyboard cancellation. Conversion progress uses stderr; JSONL events
are available with `--events-jsonl PATH`.
