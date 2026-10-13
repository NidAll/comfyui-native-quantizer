#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compatibility command and Python facade for the modular native quantizer."""

from importlib import import_module


def main(argv=None):
    from native_quantizer.cli.commands import main as run
    return run(argv)


def __getattr__(name):
    from native_quantizer.compat import EXPORTS
    if name in EXPORTS:
        return getattr(import_module("native_quantizer." + EXPORTS[name]), name)
    if name in {"torch", "safetensors", "np"}:
        return import_module("numpy" if name == "np" else name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    from native_quantizer.compat import EXPORTS
    return sorted(set(globals()) | set(EXPORTS))


if __name__ == "__main__":
    from native_quantizer.cli.entrypoint import entrypoint
    raise SystemExit(entrypoint())
