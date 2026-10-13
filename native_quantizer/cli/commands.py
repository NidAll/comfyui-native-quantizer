"""CLI command dispatch, setup, inspection and verification presentation."""

from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

from native_quantizer.cli.arguments import build_arg_parser, validate_operation_options
from native_quantizer.cli.catalog import completions, format_records, print_formats
from native_quantizer.config import apply_cli_defaults, config_path
from native_quantizer.errors import UsageError
from native_quantizer.events import prepare_event_stream


def main(argv: Optional[List[str]] = None) -> int:
    # Even rejected arguments must not reuse telemetry from an earlier call.
    from native_quantizer.events import configure
    configure(None)
    parser = build_arg_parser()
    effective_argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(effective_argv)
    defaults = apply_cli_defaults(parser, args, effective_argv)
    validate_operation_options(args)
    if not args.no_config:
        from native_quantizer.files import _same_path
        defaults_path = str(args.config or config_path())
        for path in (args.events_jsonl, args.json_log, args.report,
                     args.report + '.json' if args.report else None,
                     args.output, args.calibration_cache):
            if path and _same_path(path, defaults_path):
                raise UsageError('artifact paths must not overwrite the CLI defaults file')
    commands = [args.list_formats, args.list_architectures, args.self_test,
                args.show_config, args.diagnose, bool(args.completions),
                bool(args.audit_architectures), bool(args.write_architecture_template)]
    operations = [args.inspect, args.validation_only, args.verify_output is not None,
                  args.metadata_only, args.dry_run]
    if sum(commands) > 1 or (any(commands) and any(operations)):
        raise UsageError('choose one catalog, diagnostic or conversion operation')
    if args.json and not any((args.list_formats, args.list_architectures, args.show_config,
                              args.diagnose, args.audit_architectures, args.inspect, args.verify_output)):
        raise UsageError('--json requires an inspection, verification, catalog or diagnostic operation')
    if (args.model is None and not args.hf_repo and not any(commands)
            and args.verify_output is None and not args.interactive
            and not (not effective_argv and sys.stdin.isatty())):
        parser.error('ORIGINAL_MODEL is required')
    prepare_event_stream(args, verification_input=args.verify_output)
    if args.completions:
        print(completions(parser, args.completions), end='')
        return 0
    if args.show_config:
        from native_quantizer.config import DEFAULT_CHOICES
        result = {'path': str(args.config or config_path()), 'enabled': not args.no_config,
                  'file_defaults': defaults,
                  'effective': {key: getattr(args, key) for key in (*DEFAULT_CHOICES, 'max_memory', 'offline')}}
        print(json.dumps(result, indent=2))
        return 0
    if args.audit_architectures:
        from native_quantizer.architectures.audit import audit_comfy_architectures
        result = audit_comfy_architectures(args.audit_architectures)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f'{len(result["covered"])}/{result["model_classes"]} installed model classes have policies')
            print('Missing: ' + (', '.join(result['missing']) or 'none'))
            print('Known unsupported: ' + (', '.join(result['known_unsupported']) or 'none'))
            print(result['scope'])
        return 1 if result['missing'] else 0
    if args.diagnose:
        from native_quantizer.runtime.diagnostics import probe_runtime
        from native_quantizer.runtime.conversion import conversion_device
        from native_quantizer.errors import RuntimeCompatibilityError
        result = probe_runtime()
        result['python'] = sys.executable
        result['requested_device'] = args.device
        result['offline'] = args.offline
        try:
            _, result['conversion_backend'] = conversion_device(args.device)
            result['ok'] = True
        except RuntimeCompatibilityError as exc:
            result.update(ok=False, error=str(exc))
        print(json.dumps(result, indent=2))
        return 0 if result['ok'] else 1
    if args.list_formats:
        if args.json:
            print(json.dumps(format_records(), indent=2))
        else:
            print_formats()
        return 0
    if args.list_architectures:
        from native_quantizer.architectures.policies import family_names, get_family
        rows = [get_family(name).to_dict() for name in family_names()]
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            print(f"{'family':24s} {'component':14s} {'runtime':12s} classes")
            for row in rows:
                print(f"{row['family']:24s} {row['component']:14s} {row['runtime_status']:12s} "
                      f"{', '.join(row['comfyui_classes'])}")
        return 0
    if args.write_architecture_template:
        from native_quantizer.architectures.custom import architecture_template
        with open(args.write_architecture_template, 'x', encoding='utf-8') as f:
            json.dump(architecture_template(), f, indent=2)
            f.write('\n')
        return 0
    if args.interactive or (not effective_argv and sys.stdin.isatty()):
        from native_quantizer.cli.interactive import interactive_setup
        interactive_setup(args)
    if args.model is None and not args.hf_repo and not args.self_test and args.verify_output is None:
        parser.error('ORIGINAL_MODEL is required')
    from native_quantizer.application import resolve_input_sources
    from native_quantizer.logging import setup_logging
    from native_quantizer.runtime.capabilities import inspect_environment
    from native_quantizer.checkpoint_io.reader import RawSafetensorsFile
    from native_quantizer.utils import human_bytes
    from native_quantizer.validation.native import verify_native_output
    from native_quantizer.validation.verify import verify_output

    resolve_input_sources(args)
    validate_operation_options(args)
    setup_logging(args.log_level,
                  args.json_log if (args.list_architectures or args.self_test) else None)
    env = inspect_environment()

    if args.self_test:
        from native_quantizer.testing.self_tests import run_self_tests
        return run_self_tests()

    if args.verify_output is not None:
        if not os.path.exists(args.verify_output):
            parser.error(f"--verify-output file not found: {args.verify_output}")
        with RawSafetensorsFile(args.verify_output) as probe:
            native = 'comfy_quantizer' in probe.metadata
        if args.json:
            result = verify_native_output(args.verify_output) if native else verify_output(args.verify_output)
            print(json.dumps(result, indent=2))
            return 0 if result['ok'] else 1
        if native:
            result = verify_native_output(args.verify_output)
            print(json.dumps(result, indent=2))
            return 0 if result['ok'] else 1
        summary = verify_output(args.verify_output)
        print("=" * 70)
        print("output checkpoint verification (source-free)")
        print("=" * 70)
        print(f"tensors         : {summary['n_tensors']} "
              f"({human_bytes(summary['total_bytes'])})")
        print(f"quantized layers: {summary['quantized_layers']} "
              f"{summary['formats'] or ''}")
        print(f"payload sha256  : {summary['output_sha256']}")
        print("-" * 70)
        for c in summary["checks"]:
            print(f"[{'PASS' if c['ok'] else 'FAIL'}] {c['name']} -- {c['detail']}")
        print(f"verify-output: {summary['n_failed']} failure(s)")
        return 0 if summary["ok"] else 1

    if args.model is None and args.verify_output is None:
        parser.error("ORIGINAL_MODEL is required")

    if args.inspect and args.json:
        from native_quantizer.architectures.inspection import inspect_checkpoint
        from native_quantizer.application import _validate_destination_paths
        from native_quantizer.checkpoint_io.reader import discover_checkpoint
        info = discover_checkpoint(args.model, trust_pickle=args.trust_pickle,
                                   allow_extra_shard_tensors=args.allow_extra_shard_tensors)
        _validate_destination_paths(info, args)
        prepare_event_stream(args, source_files=info.files)
        print(json.dumps(inspect_checkpoint(args.model), indent=2))
        return 0

    from native_quantizer.application import run_conversion
    from native_quantizer.options import ConversionOptions
    return run_conversion(ConversionOptions.from_namespace(args), env, parser.error)
