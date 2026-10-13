"""Command-line arguments and native format options."""

from __future__ import annotations

import argparse

from native_quantizer.errors import UsageError
from native_quantizer.options import validate_conversion_options
from native_quantizer.formats import FORMAT_ALIASES, NATIVE_FORMATS
from native_quantizer.identity import (
    CODEBOOK_MODES,
    COMPONENT_AUTO,
    COMPONENT_CHOICES,
    MIXED_PROFILES,
    get_converter_version,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="comfyui-native-quantizer",
        allow_abbrev=False,
        description=(
            "Native-format checkpoint converter for ComfyUI-compatible "
            "diffusion models and text encoders. See the module docstring for "
            "the verified format specification and exact source revisions."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version",
                   version=f"%(prog)s {get_converter_version()}")
    p.add_argument('--config', metavar='PATH', help='read defaults from this JSON file instead of the XDG config')
    p.add_argument('--no-config', action='store_true', help='ignore local defaults (CLI flags still apply)')
    p.add_argument('--show-config', action='store_true', help='show the configuration path and resolved defaults')
    p.add_argument('--diagnose', action='store_true', help='check Python, CUDA and conversion dependencies without loading weights')
    p.add_argument('--completions', choices=('bash', 'fish', 'zsh'), help='print shell completion definitions')
    p.add_argument('--audit-architectures', metavar='COMFY_ROOT', help='compare installed ComfyUI model classes to policy coverage, without importing ComfyUI')
    p.add_argument('--json', action='store_true', help='machine-readable output for inspection, verification and catalog/diagnostic commands')
    p.add_argument("model", nargs="?", metavar="ORIGINAL_MODEL",
                   help="path to the original checkpoint: a .safetensors file, a "
                        "sharded safetensors directory, an HF-style model directory, "
                        "or (with --trust-pickle) a torch pickle checkpoint")
    p.add_argument("--output", metavar="PATH", default=None,
                   help="output checkpoint path (required for conversion)")
    p.add_argument("--format", type=lambda v: FORMAT_ALIASES.get(v, v),
                   choices=["w4a8", "w6a8", "mixed", *NATIVE_FORMATS], default="w4a8",
                   help="output format; --list-formats shows backend requirements and native algorithm names")
    p.add_argument("--components",
                   type=lambda value: value.strip().lower().replace("-", "_"),
                   choices=list(COMPONENT_CHOICES),
                   default=COMPONENT_AUTO,
                   help="components to quantize: auto selects the detected "
                        "standalone diffusion model or text encoder; all also "
                        "recognizes embedded text towers outside a prefixed "
                        "diffusion model")
    p.add_argument("--quantize-text-embeddings", action="store_true",
                   help="opt in to rowwise INT8 token embeddings with int8, "
                        "int8-convrot, or mixed; positional embeddings stay unchanged")
    p.add_argument("--codebook-mode", choices=list(CODEBOOK_MODES),
                   default="auto",
                   help="W4A8 codebook strategy: auto uses the fast fixed "
                        "Gaussian table and fits only heavy-tailed layers; "
                        "fixed never fits; fit preserves the slower legacy "
                        "per-layer Lloyd-Max path")
    p.add_argument("--enable-w6a8", action="store_true", help="mixed mode: opt in to native W6A8 candidates")
    p.add_argument("--w6a8-group-size", type=int, default=16, help="W6A8 group size: positive multiple of 16 dividing K (default 16)")
    p.add_argument("--profile", choices=list(MIXED_PROFILES), default="auto",
                   help="mixed-mode profile: auto detects the runtime (GPU -> "
                        "balanced, CPU -> conservative); balanced, conservative "
                        "and size-first set the quality/compression tradeoff; "
                        "throughput permits more direct INT8 storage")
    p.add_argument("--kernel-costs", metavar="JSON",
                   help="mixed mode: shape-specific measurements from benchmark_kitchen.py")
    p.add_argument("--mixed-objective", choices=("kitchen", "size"), default="kitchen",
                   help="spend available byte budget to avoid Kitchen runtime weight expansion, or minimize storage")
    p.add_argument("--target-runtime", choices=["auto", "nvidia", "amd", "cpu"],
                   default="auto",
                   help="target inference runtime used for format eligibility; "
                        "auto probes torch (CUDA/ROCm/CPU)")
    p.add_argument("--runtime-certificate", metavar="PATH", default=None,
                   help="mixed mode: JSON certificate from actual native "
                        "load and forward probes on the target inference "
                        "machine; overrides static W4A4 dispatch guesses with "
                        "observed behavior")
    p.add_argument("--require-runtime-certificate", action="store_true",
                   help="mixed mode: refuse conversion unless every selected "
                        "format is runtime-certified on the target machine")
    p.add_argument("--strip-gpu-identity", action="store_true",
                   help="omit GPU identity (device name, compute capability, "
                        "ROCm architecture) from the checkpoint extension "
                        "metadata, for publishing converted checkpoints "
                        "without revealing the target or conversion machine")
    p.add_argument("--quality-gate", type=float, default=None, metavar="F",
                   help="mixed mode: per-layer error gate override (relL2 "
                        "fraction; default from the profile)")
    p.add_argument("--global-error-gate", type=float, default=None, metavar="F",
                   help="mixed mode: global mean-error gate override; the "
                        "planner promotes layers until this passes")
    p.add_argument("--max-linear-bytes-per-param", type=float, default=None,
                   metavar="F",
                   help="mixed mode: hard compression target in bytes/parameter "
                        "for the targeted linear payload (overrides the profile; "
                        "conversion fails when exceeded)")
    p.add_argument("--max-bf16-fraction", "--max-original-byte-fraction",
                   type=float, default=None, dest="max_bf16_fraction",
                   metavar="F",
                   help="mixed mode: hard limit on the fraction of final "
                        "targeted serialized bytes occupied by "
                        "original-precision tensors (legacy option name "
                        "--max-bf16-fraction; the alias "
                        "--max-original-byte-fraction is preferred)")
    p.add_argument("--w4a4-linear-dtype", choices=["int4", "int8"], default="int8",
                   help="convrot_w4a4 execution variant: int4 (true W4A4) or "
                        "int8 (4-bit weights through the int8 path; the "
                        "default). Execution property only, never a quality "
                        "fallback")
    p.add_argument("--disable-w4a4", action="store_true",
                   help="mixed mode: never select convrot_w4a4 layers")
    p.add_argument("--disable-w4a8", action="store_true",
                   help="mixed mode: never select asym_w4a8_int8 layers")
    p.add_argument("--disable-int8", action="store_true",
                   help="mixed mode: never select int8_tensorwise layers")
    p.add_argument("--require-calibration", action="store_true",
                   help="mixed mode: refuse planning without calibration "
                        "activations (--calibration-source)")
    p.add_argument("--architecture", default="auto",
                   help="auto or an architecture name from --list-architectures")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "rocm"], default="auto",
                   help="conversion device; auto uses CUDA/ROCm when available, otherwise CPU; explicit unavailable devices fail")
    p.add_argument("--compute-dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto",
                   help="precision of the quantization math (default fp32, matching "
                        "the reference implementation)")
    p.add_argument("--output-dtype", choices=["auto", "fp16", "bf16"], default="auto",
                   help="cast passthrough (non-quantized) float tensors to this "
                        "dtype; auto keeps the original dtype")
    p.add_argument("--group-size", type=int, default=None,
                   help="quantization group size; only 16 is supported (the "
                        "validated W4A8 configuration with convrot 256)")
    p.add_argument("--min-quantized-byte-fraction", type=float, default=None,
                   metavar="F",
                   help="abort when the quantized share of policy-targeted 2D "
                        "linear bytes is below F (0..1); used with "
                        "--fail-on-low-compression or alone as the threshold")
    p.add_argument("--fail-on-low-compression", action="store_true",
                   help="refuse to convert when the quantized byte fraction is "
                        "below 0.10 (or --min-quantized-byte-fraction)")
    p.add_argument("--calibration-source", metavar="PATH", default=None,
                   help="local calibration data (.npz/.pt/.npy files or a directory; "
                        "arrays named exactly like the layer keys, shape [S, K])")
    p.add_argument("--calibration-samples", type=int, default=None,
                   help="limit calibration rows used per layer")
    p.add_argument("--calibration-cache", metavar="PATH", default=None,
                   help="read/write a compressed cache of the activation rows")
    p.add_argument("--seed", type=int, default=0, help="reproducibility seed")
    p.add_argument("--include", action="append", default=[],
                   metavar="PATTERN", help="regex; select matching eligible layer weights "
                        "for quantization (shape/dtype safety gates still apply)")
    p.add_argument("--exclude", action="append", default=[],
                   metavar="PATTERN", help="regex; never quantize matching tensors")
    p.add_argument("--keep-precision", action="append", default=[],
                   metavar="PATTERN", help="regex; keep matching weights at original precision")
    p.add_argument("--sensitivity-threshold", type=float, default=None,
                   help="keep layers at original precision when their (activation-aware "
                        "if calibration given, else weight-only) error exceeds this")
    p.add_argument("--error-threshold", type=float, default=None,
                   help="hard reconstruction relL2 fallback during a calibration/"
                        "sensitivity prepass (default: architecture policy bound)")
    p.add_argument("--max-memory", default="2G", metavar="SIZE",
                   help="per-tensor working-memory budget (e.g. 512M, 2G); larger "
                        "tensors are quantized in chunks")
    p.add_argument("--streaming", action="store_true", default=True,
                   help="stream the conversion with bounded memory (always enabled)")
    p.add_argument("--resume", action="store_true",
                   help="resume an interrupted conversion from its state file")
    p.add_argument("--overwrite", action="store_true",
                   help="allow replacing an existing output file")
    p.add_argument("--dry-run", action="store_true",
                   help="detect, plan and report without writing the output")
    p.add_argument("--inspect", action="store_true",
                   help="inspect the input checkpoint and exit")
    p.add_argument("--list-architectures", action="store_true",
                   help="list the embedded architecture registry and exit")
    p.add_argument("--validate", action="store_true",
                   help="run full standalone validation after conversion (all layers, "
                        "output hash, optional runtime compatibility probe)")
    p.add_argument("--validation-only", action="store_true",
                   help="validate an existing output checkpoint (with --model and --output)")
    p.add_argument("--verify-output", metavar="PATH", default=None,
                   help="source-free verification of an existing output "
                        "checkpoint: structural, metadata, packing and "
                        "payload-hash checks with no original model required")
    p.add_argument("--metadata-only", action="store_true",
                   help="generate the metadata and report only; do not write the model")
    p.add_argument("--report", metavar="PATH", default=None,
                   help="write the human-readable report to PATH")
    p.add_argument("--log-level", default="info",
                   choices=["debug", "info", "warning", "error"])
    p.add_argument("--json-log", metavar="PATH", default=None,
                   help="also emit structured JSON log lines to PATH")
    p.add_argument("--events-jsonl", metavar="PATH", default=None,
                   help="write quantizer-events/v1 JSON lines for local frontends")
    p.add_argument("--trust-pickle", action="store_true",
                   help="allow deserializing pickle-based checkpoints (unsafe for "
                        "untrusted files)")
    p.add_argument("--allow-extra-shard-tensors", action="store_true",
                   help="when a shard index is present, tolerate tensors in "
                        "shards that the index does not list (default: error; "
                        "the index is authoritative)")
    p.add_argument("--nonfinite-policy", choices=("error", "keep"),
                   default="error",
                   help="policy for non-finite (NaN/Inf) values in quantizable "
                        "layers: error aborts naming the layer, keep passes it "
                        "through at original precision")
    p.add_argument("--experimental", action="store_true",
                   help="allow data-only custom architecture policies without runtime certification")
    p.add_argument("--yes", action="store_true", help="assume yes for confirmations")
    p.add_argument("--self-test", action="store_true",
                   help="run the embedded engineering self-tests and exit")
    add_native_arguments(p)
    return p


def add_native_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument('--offline', action=argparse.BooleanOptionalAction, default=False,
                   help='refuse model downloads; local checkpoints and metadata inspection remain available')
    p.add_argument('--interactive', action='store_true', help='guided terminal wizard')
    p.add_argument('--list-formats', action='store_true', help='show native format contracts')
    p.add_argument('--architecture-policy', help='JSON architecture policy; data only, no Python execution')
    p.add_argument('--write-architecture-template', metavar='PATH', help='write a starter JSON policy and exit')
    p.add_argument('--key-map', help='JSON {source_key: target_key}; exact, collision-checked renaming before quantization')
    p.add_argument('--hf-repo', help='download raw safetensors from owner/repository')
    p.add_argument('--hf-revision', default='main', help='HF revision; resolved to a commit before downloading')
    p.add_argument('--hf-subfolder', default='', help='one component, e.g. transformer or text_encoder')
    p.add_argument('--hf-filename', help='one safetensors file or safetensors index inside the component')
    p.add_argument('--hf-cache', help='Hugging Face download cache directory')
    p.add_argument('--full-precision-matmul', action='store_true',
                   help='native formats: keep compressed storage, ask ComfyUI for full precision matmul')
    p.add_argument('--progress', choices=('auto', 'on', 'off'), default='auto')
    p.add_argument('--rotation-backend', choices=('matrix', 'butterfly'), default='matrix',
                   help='ConvRot formats: matrix is faster in CPU tests; butterfly is chunk-invariant')


def validate_operation_options(args) -> None:
    if args.config and args.no_config:
        raise UsageError('--config and --no-config are mutually exclusive')
    operation_modes = {
        "--inspect": args.inspect,
        "--validation-only": args.validation_only,
        "--verify-output": args.verify_output is not None,
        "--metadata-only": args.metadata_only,
        "--dry-run": args.dry_run,
    }
    selected_modes = [name for name, selected in operation_modes.items() if selected]
    if len(selected_modes) > 1:
        raise UsageError(
            "operation modes are mutually exclusive: " + ", ".join(selected_modes))
    if args.list_architectures and args.self_test:
        raise UsageError("--list-architectures and --self-test are mutually exclusive")
    if (args.list_architectures or args.self_test) and selected_modes:
        special = "--list-architectures" if args.list_architectures else "--self-test"
        raise UsageError(
            f"{special} cannot be combined with " + ", ".join(selected_modes))
    validate_conversion_options(args)
