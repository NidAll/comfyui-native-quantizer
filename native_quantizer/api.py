"""Stable operations for Python callers and optional frontends.

Importing this module does not initialize PyTorch, checkpoints or regression
fixtures. Computation dependencies are loaded when an operation is requested.
"""
from __future__ import annotations

from native_quantizer.options import ConversionOptions, validate_conversion_options


def local_defaults() -> dict:
    """Read local CLI/frontend preferences without importing compute libraries."""
    from native_quantizer.config import load_defaults
    return load_defaults()


def convert(options: ConversionOptions) -> int:
    import dataclasses
    from native_quantizer.application import prepare_event_stream, resolve_input_sources, run_conversion
    from native_quantizer.errors import UsageError
    from native_quantizer.logging import setup_logging
    from native_quantizer.runtime.capabilities import inspect_environment

    options = dataclasses.replace(options)
    from native_quantizer.events import configure
    configure(None)
    validate_conversion_options(options)
    resolve_input_sources(options)
    if options.model is None:
        raise UsageError('ORIGINAL_MODEL is required')
    prepare_event_stream(options)
    setup_logging(options.log_level)
    return run_conversion(options, inspect_environment())


def run_cli(argv: list[str]) -> int:
    """Execute a recorded CLI command in a frontend's isolated worker."""
    from native_quantizer.cli.commands import main
    return main(argv)


def inspect_checkpoint(path: str) -> dict:
    from native_quantizer.architectures.inspection import inspect_checkpoint as inspect
    return inspect(path)


def probe_runtime() -> dict:
    from native_quantizer.runtime.diagnostics import probe_runtime as probe
    return probe()


def inspect_calibration(model: str, source: str, samples: int | None = None) -> dict:
    from native_quantizer.calibration import load_calibration
    from native_quantizer.checkpoint_io.reader import discover_checkpoint

    info = discover_checkpoint(model)
    stats = load_calibration(source, info, samples, None, max_memory=64 << 20)
    return {'source': stats.source, 'files': len(stats.files), 'layers': len(stats.layers),
            'samples': stats.n_samples,
            'coverage': len(stats.layers) / max(1, sum(t.name.endswith('.weight') for t in info.tensors))}


def inspect_runtime_certificate(path: str) -> dict:
    from native_quantizer.runtime.capabilities import load_runtime_certificate
    certificate = load_runtime_certificate(path)
    return {'path': path, 'formats': sorted(certificate.formats)}


def verify_output(path: str) -> dict:
    from native_quantizer.validation.verify import verify_output as verify
    return verify(path)


def download_hf_component(options) -> str:
    from native_quantizer.sources.huggingface import download_hf_component as download
    return download(options)


def browse_huggingface(repo: str, revision: str = 'main', subfolder: str = '') -> dict:
    from native_quantizer.sources.huggingface import browse_huggingface as browse
    return browse(repo, revision, subfolder)


def benchmark(n: int, k: int, m: int, repeats: int, comfy_root: str | None = None,
              w6a8_group_size: int = 16) -> dict:
    from native_quantizer.benchmark import run
    return run(n, k, m, repeats, comfy_root, w6a8_group_size)
