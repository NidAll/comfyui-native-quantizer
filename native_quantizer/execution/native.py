"""Native conversion with bounded encoding, quality gates and atomic restart."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import math
import os
import shutil
import stat
import sys
import tempfile
import time
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from native_quantizer import events
from native_quantizer.architectures.detection import DetectionResult
from native_quantizer.architectures.policies import TEXT_TOKEN_EMBEDDING
from native_quantizer.architectures.reference import load_reference_recipe
from native_quantizer.calibration import load_calibration
from native_quantizer.checkpoint_io.fingerprints import _portable_hash_manifest, hash_checkpoint_files
from native_quantizer.checkpoint_io.reader import CheckpointReader
from native_quantizer.checkpoint_io.types import CheckpointInfo, TensorMeta
from native_quantizer.checkpoint_io.writer import NativeSpool, _native_copy
from native_quantizer.codecs.bounded import _native_rows
from native_quantizer.codecs.native import _native_encode
from native_quantizer.codecs.w4a4 import _pick_w4a4_convrot_group, w4a4_weight_is_quantizable
from native_quantizer.dtypes import FLOAT_DTYPES, tensor_to_bytes
from native_quantizer.errors import (
    CompressionGateError,
    InputError,
    OutputError,
    PolicyError,
    QualityGateError,
    UsageError,
    ValidationError,
)
from native_quantizer.files import BackgroundDigest, _atomic_write_json, _fsync_parent, _publish_output
from native_quantizer.formats import NATIVE_ENGINE_REVISION, NATIVE_FORMATS, NATIVE_UNSUPPORTED_OPTIONS, NativeFormat
from native_quantizer.identity import (
    COMPONENT_TEXT_ENCODER,
    FORMAT_INT8,
    FORMAT_MIXED,
    FORMAT_W4A4,
    LAYER_CONF_KEY,
    METADATA_KEY_EXT,
    get_converter_version,
)
from native_quantizer.logging import log
from native_quantizer.metadata import inline_quant_metadata, serialized_quant_metadata
from native_quantizer.optimization import (OptimizationConfig, calibration_digest,
                                          check_optimization_calibration, layer_activations)
from native_quantizer.planning.classification import _emit_plan_layers, classify_tensors
from native_quantizer.options import low_compression_threshold
from native_quantizer.planning.types import DecisionKind, TensorDecision
from native_quantizer.runtime.capabilities import warn_native_gpu
from native_quantizer.runtime.conversion import conversion_device
from native_quantizer.runtime.kitchen import _native_kitchen
from native_quantizer.utils import human_bytes, json_dumps
from native_quantizer.validation.native import verify_native_output


@dataclass
class _NativeRun:
    """Settings and accumulated results shared by every tensor of one conversion."""
    args: Any
    spec: NativeFormat
    device: torch.device
    kitchen: Any
    optimization: OptimizationConfig
    calibration: Any
    reference: Optional[Dict[str, Any]]
    gate: float
    spool: NativeSpool
    tmp: str
    layers: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)


def convert_native(info: CheckpointInfo, detection: DetectionResult, args: Any) -> int:
    spec = NATIVE_FORMATS[args.format]
    optimization = OptimizationConfig.from_options(args)
    device = _check_native_request(args, spec, optimization)
    decisions, reference, reference_identity = _select_native_layers(info, detection, args, spec)
    kitchen = _native_kitchen(spec) if spec.kitchen_class and not args.dry_run else None
    planned = _announce_native_plan(info, detection, decisions, spec)
    if args.dry_run:
        for d in decisions:
            print(f'{d.kind.value if spec.algorithm else "cast"}: {d.name}: {d.reason}')
        return 0
    if spec.algorithm and not planned:
        raise PolicyError('no eligible linears; inspect the model and supply a matching architecture policy')
    calibration = None
    if optimization.enabled and args.calibration_source:
        calibration = load_calibration(args.calibration_source, info, args.calibration_samples,
                                       args.calibration_cache, args.max_memory)
    if optimization.enabled:
        check_optimization_calibration(optimization, calibration, decisions, args.require_calibration)
    out, identity = _prepare_native_output(args, info)
    gate = args.error_threshold if args.error_threshold is not None else detection.policy.max_rel_l2
    started = time.monotonic()
    events.emit('phase', phase='conversion')
    initial_identity = _source_identity(info)
    # Provenance hashing overlaps encoding; the identity check before publication
    # still rejects any source change made while either was reading.
    source_digest = BackgroundDigest(hash_checkpoint_files, info)
    with tempfile.TemporaryDirectory(prefix='.comfy-quant-', dir=os.path.dirname(out)) as tmp:
        run = _NativeRun(args, spec, device, kitchen, optimization, calibration, reference, gate,
                         NativeSpool(tmp), tmp)
        try:
            with CheckpointReader(info) as reader, torch.inference_mode():
                for number, d in enumerate(decisions, 1):
                    m = info.by_name(d.name)
                    _convert_native_tensor(run, reader, m, d)
                    if args.progress == 'on' or (args.progress == 'auto' and sys.stderr.isatty()):
                        print(f'[{number}/{len(decisions)}] {m.name}', file=sys.stderr)
                    events.emit('progress', phase='conversion', done=number,
                                total=len(decisions), tensor=m.name,
                                rate=number / max(time.monotonic() - started, 1e-9))
            _check_native_gates(run, info, decisions)
            metadata = _native_metadata(run, info, detection, decisions, reference_identity,
                                        source_digest.result())
            check = _publish_native(run, info, metadata, out, identity, initial_identity)
        finally:
            source_digest.cancel()
            run.spool.file.close()
    result = {'output': out, 'format': spec.cli, 'quantized_layers': len(run.layers),
              'input_bytes': info.total_bytes, 'output_bytes': os.path.getsize(out),
              'seconds': time.monotonic() - started, 'validation': check, 'layers': run.metrics,
              'runtime_verified': False}
    if args.report:
        _atomic_write_json(args.report, result)
    events.emit('artifact', path=out, bytes=result['output_bytes'],
                quantized=len(run.layers))
    print(f'Saved {out}\n{len(run.layers)} quantized layers; {human_bytes(result["output_bytes"])}; '
          f'{result["seconds"]:.1f}s. Payload/metadata verified; runtime inference untested.')
    return 0


# ---- request and layer selection ----

def _check_native_request(args: Any, spec: NativeFormat, optimization: OptimizationConfig) -> torch.device:
    if optimization.enabled and spec.algorithm is None:
        raise UsageError('MSE optimization applies to quantized formats; FP16/BF16 casts have no scale to optimize')
    events.emit('phase', phase='planning')
    warn_native_gpu(args)
    # Refuse controls belonging to the legacy planner instead of silently ignoring them.
    for option in NATIVE_UNSUPPORTED_OPTIONS:
        if optimization.enabled and option in ('calibration_source', 'calibration_cache',
                                               'calibration_samples', 'require_calibration'):
            continue
        if getattr(args, option, None) is not None and getattr(args, option) is not False:
            raise UsageError(f'--{option.replace("_", "-")} is a W4A8/mixed-engine option; '
                             'native formats use streamed weight-error gates and atomic restart')
    if args.profile != 'auto':
        raise UsageError('--profile belongs to mixed mode; use --error-threshold for native formats')
    if args.compute_dtype not in ('auto', 'fp32'):
        raise UsageError('native quantization math uses FP32; omit --compute-dtype or use fp32')
    if not args.output and not args.dry_run:
        raise UsageError('--output is required')
    if args.max_memory < 4096:
        raise UsageError('--max-memory must be at least 4K')
    device, _ = conversion_device(args.device)
    if device.type == 'cuda':
        free, _ = torch.cuda.mem_get_info()
        if args.max_memory > int(free * 0.75):
            log().warning('limiting conversion budget to 75%% of currently free GPU memory')
            args.max_memory = max(4096, int(free * 0.75))
    if args.group_size is not None:
        raise UsageError('--group-size belongs to W4A8; native layout group sizes are fixed')
    return device


def _is_token_embedding(d: TensorDecision) -> bool:
    return d.component == COMPONENT_TEXT_ENCODER and bool(TEXT_TOKEN_EMBEDDING.search(d.name))


def _select_native_layers(info: CheckpointInfo, detection: DetectionResult, args: Any,
                          spec: NativeFormat) -> Tuple[List[TensorDecision], Any, Any]:
    decisions = classify_tensors(info, detection, FORMAT_MIXED, None, args.include,
                                 args.exclude, args.keep_precision, None, getattr(args, "min_numel_override", None),
                                 components=args.components,
                                 quantize_text_embeddings=args.quantize_text_embeddings)
    reference = None
    reference_identity = None
    if getattr(args, 'reference_recipe', None):
        if (args.include or args.exclude or args.keep_precision or args.quantize_text_embeddings
                or args.full_precision_matmul
                or args.components not in ('auto', detection.policy.component)):
            raise UsageError('reference recipes own layer selection; omit include/exclude/keep/embedding/component overrides')
        reference, reference_identity = load_reference_recipe(args.reference_recipe, info, detection)
        for d in decisions:
            d.layer = d.name[:-len('.weight')] if d.name.endswith('.weight') else None
            if d.layer in reference['layers']:
                d.kind, d.reason = DecisionKind.QUANTIZE, 'exact reference recipe layer'
            else:
                d.kind, d.reason = DecisionKind.KEEP, 'reference floating tensor; passthrough'
    for d in decisions:
        m = info.by_name(d.name)
        if d.kind == DecisionKind.QUANTIZE and spec.algorithm:
            token_embedding = _is_token_embedding(d)
            d.format = FORMAT_INT8 if token_embedding else spec.algorithm
            if spec.cli == 'int8-convrot' and not token_embedding:
                group = (reference['layers'][d.layer].get('convrot_groupsize', 1)
                         if reference else (args.int8_convrot_groupsize or 256))
                ok = m.shape[1] > 0 and m.shape[1] % group == 0
                reason = f'INT8 ConvRot requires input width divisible by {group}; source precision retained'
            elif spec.algorithm == FORMAT_W4A4:
                ok, reason = w4a4_weight_is_quantizable(m.shape, m.dtype)
            else:
                alignment = 1 if token_embedding else spec.alignment
                ok = all(x % alignment == 0 for x in m.shape)
                reason = f'shape must align to {alignment}; padding would change checkpoint shape detection'
            if not ok:
                d.kind, d.reason = DecisionKind.KEEP, reason
    return decisions, reference, reference_identity


def _announce_native_plan(info: CheckpointInfo, detection: DetectionResult,
                          decisions: List[TensorDecision], spec: NativeFormat) -> int:
    planned = sum(d.kind == DecisionKind.QUANTIZE for d in decisions) if spec.algorithm else 0
    variant = detection.classifier_info.get('variant')
    variant_label = f' ({variant})' if variant else ''
    print(f'Architecture: {detection.architecture}{variant_label}; '
          f'format: {spec.cli}; candidate layers: {planned}')
    events.emit('plan', architecture=detection.architecture, format=spec.cli,
                quantized=planned,
                preserved=len(decisions) - planned,
                formats={spec.algorithm: planned} if spec.algorithm else {})
    _emit_plan_layers(info, decisions, cast_format=spec.cli if spec.algorithm is None else None)
    print('Working budget excludes mmap pages, Python/PyTorch overhead and backend allocator caches.')
    return planned


def _prepare_native_output(args: Any, info: CheckpointInfo) -> Tuple[str, Optional[Tuple[int, int]]]:
    out = os.path.abspath(args.output)
    parent = os.path.dirname(out)
    os.makedirs(parent, exist_ok=True)
    identity = None
    if os.path.lexists(out):
        st = os.lstat(out)
        if not stat.S_ISREG(st.st_mode) or not args.overwrite:
            raise OutputError('output exists or is not a regular file; choose another path or use --overwrite')
        identity = (st.st_dev, st.st_ino)
    # Two output copies plus one layer of fragments; conservative space preflight.
    if shutil.disk_usage(parent).free < max(1 << 20, info.total_bytes * 3):
        raise OutputError('insufficient temporary disk space (need up to 3x input bytes)')
    return out, identity


def _source_identity(info: CheckpointInfo) -> Dict[str, Tuple[int, ...]]:
    return {path: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            for path in info.files for s in [os.stat(path)]}


# ---- per-tensor encoding ----

def _convert_native_tensor(run: _NativeRun, reader: CheckpointReader, m: TensorMeta,
                           d: TensorDecision) -> None:
    quantize = run.spec.algorithm is not None and d.kind == DecisionKind.QUANTIZE
    if quantize:
        quantize = _quantize_native_layer(run, reader, m, d)
    if not quantize:
        dtype = m.dtype if d.kind == DecisionKind.KEEP_PRECISION else (_native_output_dtype(run) or m.dtype)
        if m.dtype not in FLOAT_DTYPES:
            dtype = m.dtype
        _native_copy(reader, m, run.spool, dtype, run.args.max_memory)


def _native_output_dtype(run: _NativeRun) -> Optional[torch.dtype]:
    if run.spec.algorithm is None:
        return torch.float16 if run.spec.cli == 'fp16' else torch.bfloat16
    return {'auto': None, 'fp16': torch.float16, 'bf16': torch.bfloat16}[run.args.output_dtype]


def _quantize_native_layer(run: _NativeRun, reader: CheckpointReader, m: TensorMeta,
                           d: TensorDecision) -> bool:
    """Encode one layer into fragments; spool it if the gate accepts, else report a keep."""
    args, spec, optimization = run.args, run.spec, run.optimization
    layer_spec = NATIVE_FORMATS['int8'] if _is_token_embedding(d) else spec
    reference_conf = run.reference['layers'][d.layer] if run.reference else None
    if reference_conf is not None:
        layer_spec = NATIVE_FORMATS['int8-convrot' if reference_conf.get('convrot') else 'int8']
    with contextlib.ExitStack() as stack:
        convrot = (_pick_w4a4_convrot_group(m.shape[1]) if layer_spec.algorithm == FORMAT_W4A4
                   else (args.int8_convrot_groupsize or 256) if layer_spec.cli == 'int8-convrot' else 0)
        if reference_conf is not None:
            convrot = reference_conf.get('convrot_groupsize', 0)
        scale, finite = _tensor_scale(run, reader, m)
        kitchen_work = max(m.nbytes * 48, math.prod(m.shape) * 256 if optimization.enabled else 0)
        if run.kitchen and kitchen_work > args.max_memory:
            raise UsageError(f'{m.name}: swizzled {spec.cli} packing needs a whole-layer '
                             f'budget of about {human_bytes(kitchen_work)}; '
                             'increase --max-memory or use INT8/FP8/W4A4')
        row_budget = args.max_memory - (convrot * convrot * 4 if convrot and args.rotation_backend == 'matrix' else 0)
        if row_budget < m.shape[1] * (256 if optimization.enabled else 96):
            raise UsageError('rotation and row workspace exceed budget; increase --max-memory or use --rotation-backend butterfly')
        activations = layer_activations(run.calibration, m.name)
        if optimization.enabled and layer_spec.cli.startswith('fp8-') and finite:
            from native_quantizer.codecs.optimization import tensor_scale_search
            dtype = torch.float8_e4m3fn if layer_spec.cli == 'fp8-e4m3' else torch.float8_e5m2
            scale = tensor_scale_search(
                lambda: _native_rows(reader, m, row_budget, run.device, optimization),
                scale, dtype, optimization, activations)
        chunks = ([reader.read_tensor(m.name).to(device=run.device, dtype=torch.float32, copy=True)] if run.kitchen else
                  _native_rows(reader, m, row_budget, run.device, optimization)) if finite else ()
        fragments, fragment_meta, finite, error = _encode_rows(
            run, stack, m, chunks, layer_spec, scale, convrot, activations, finite)
        if not finite and args.nonfinite_policy == 'error':
            raise InputError(f'{m.name}: source contains NaN/Inf')
        accepted = finite and error <= run.gate
        run.metrics[m.name] = {'relative_l2': error if finite else None,
                               'optimization_objective': ('activation' if activations is not None and optimization.objective != 'weight' else 'weight') if optimization.enabled else None,
                               'accepted': accepted,
                               'reason': 'quality gate passed' if accepted else
                                         ('nonfinite source' if not finite else 'weight error exceeds gate')}
        if not accepted:
            if run.reference is not None:
                raise QualityGateError(f'{m.name}: reference layer failed quality gate; no file published')
            d.kind = DecisionKind.KEEP_PRECISION
            log().warning('%s retained at source precision: %s', m.name, run.metrics[m.name]['reason'])
            return False
        if scale is not None and spec.cli.startswith('fp8-'):
            fragment = stack.enter_context(tempfile.TemporaryFile(dir=run.tmp))
            fragment.write(tensor_to_bytes(scale))
            fragments['_scale'] = fragment
            fragment_meta['_scale'] = (torch.float32, ())
        for suffix, fragment in fragments.items():
            dt, shape = fragment_meta[suffix]
            run.spool.append(m.name + suffix, dt, shape, fragment)
        run.layers[d.layer] = _layer_config(run, layer_spec, convrot, reference_conf)
    return True


def _tensor_scale(run: _NativeRun, reader: CheckpointReader,
                  m: TensorMeta) -> Tuple[Optional[torch.Tensor], bool]:
    """Per-tensor scale for FP8/NVFP4 from one bounded scan; None elsewhere."""
    spec = run.spec
    if not (spec.cli.startswith('fp8-') or spec.cli == 'nvfp4'):
        return None, True
    finite = True
    maximum = 0.0
    for row in _native_rows(reader, m, run.args.max_memory):
        if not bool(torch.isfinite(row).all()):
            finite = False
            break
        maximum = max(maximum, row.abs().max().item())
    denominator = (448 * 6 if spec.cli == 'nvfp4' else
                   torch.finfo(torch.float8_e4m3fn if spec.cli == 'fp8-e4m3'
                               else torch.float8_e5m2).max)
    return torch.tensor(max(maximum / denominator, 1e-30), device=run.device), finite


def _encode_rows(run: _NativeRun, stack: contextlib.ExitStack, m: TensorMeta, chunks: Any,
                 layer_spec: NativeFormat, scale: Optional[torch.Tensor], convrot: int,
                 activations: Any, finite: bool) -> Tuple[Dict[str, Any], Dict[str, Tuple[torch.dtype, Tuple[int, ...]]], bool, float]:
    """Encode row chunks into per-suffix fragments and return the relative L2 error."""
    fragments: Dict[str, Any] = {}
    fragment_meta: Dict[str, Tuple[torch.dtype, Tuple[int, ...]]] = {}
    squared_error, squared_weight = 0.0, 0.0
    for row in chunks:
        if not bool(torch.isfinite(row).all()):
            finite = False
            break
        w = row
        payload, reconstructed = _native_encode(w, layer_spec, scale, convrot, run.kitchen, run.args.rotation_backend,
                                                run.optimization, activations)
        reconstructed_finite, error_sum, weight_sum = torch.stack((
            torch.isfinite(reconstructed).all().to(torch.float64),
            (w - reconstructed).double().square().sum(),
            w.double().square().sum())).tolist()
        if not reconstructed_finite:
            raise QualityGateError(f'{m.name}: backend produced nonfinite reconstruction')
        squared_error += error_sum
        squared_weight += weight_sum
        for suffix, tensor in payload.items():
            # E8M0 scales are stored as bytes, as expected by ComfyUI's dtype view.
            if str(tensor.dtype) == 'torch.float8_e8m0fnu':
                tensor = tensor.view(torch.uint8)
            if suffix not in fragments:
                fragments[suffix] = stack.enter_context(tempfile.TemporaryFile(dir=run.tmp))
                fragment_meta[suffix] = (tensor.dtype, tuple(tensor.shape))
            else:
                dt, shape = fragment_meta[suffix]
                if dt != tensor.dtype or shape[1:] != tuple(tensor.shape[1:]):
                    raise OutputError('backend returned inconsistent row-chunk layout')
                fragment_meta[suffix] = (dt, (shape[0] + tensor.shape[0], *shape[1:]))
            fragments[suffix].write(tensor_to_bytes(tensor))
        del w, payload, reconstructed, row
    error = math.sqrt(squared_error / max(squared_weight, 1e-300))
    return fragments, fragment_meta, finite, error


def _layer_config(run: _NativeRun, layer_spec: NativeFormat, convrot: int,
                  reference_conf: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    args = run.args
    conf: Dict[str, Any] = {'format': layer_spec.algorithm}
    if layer_spec.algorithm == FORMAT_W4A4:
        conf.update(convrot_groupsize=convrot, quant_group_size=64,
                    linear_dtype='int4' if run.spec.cli == 'int4-convrot' else args.w4a4_linear_dtype)
    elif layer_spec.cli == 'int8-convrot':
        conf.update(convrot=True, convrot_groupsize=convrot)
    if args.full_precision_matmul:
        conf['full_precision_matrix_mult'] = True
    if reference_conf is not None and 'full_precision_matrix_mult' in reference_conf:
        conf['full_precision_matrix_mult'] = reference_conf['full_precision_matrix_mult']
    return conf


# ---- gates, metadata and publication ----

def _check_native_gates(run: _NativeRun, info: CheckpointInfo, decisions: List[TensorDecision]) -> None:
    if run.spec.algorithm and not run.layers:
        raise QualityGateError('all candidate layers failed quality gates; no quantized file published')
    original_candidate_bytes = sum(info.by_name(d.name).nbytes for d in decisions
                                   if d.name in run.metrics)
    accepted_bytes = sum(info.by_name(d.name).nbytes for d in decisions if d.layer in run.layers)
    fraction = accepted_bytes / max(1, original_candidate_bytes)
    threshold = low_compression_threshold(run.args)
    if threshold is not None and fraction < threshold:
        raise CompressionGateError(f'accepted candidate byte fraction {fraction:.3f} < {threshold}')


def _native_metadata(run: _NativeRun, info: CheckpointInfo, detection: DetectionResult,
                     decisions: List[TensorDecision], reference_identity: Any,
                     source_hashes: Dict[str, str]) -> Dict[str, str]:
    """Append inline layer configs, then build header metadata bound to the spooled payload."""
    args, spec, layers = run.args, run.spec, run.layers
    metadata = dict(info.metadata)
    metadata.pop(METADATA_KEY_EXT, None)
    if inline_quant_metadata(layers):
        for layer, conf in layers.items():
            data = json_dumps(conf).encode('utf-8')
            run.spool.append_chunks(f'{layer}.{LAYER_CONF_KEY}', torch.uint8, (len(data),), [data])
    metadata['_quantization_metadata'] = json_dumps(serialized_quant_metadata({'layers': layers}))
    provenance = {'engine': NATIVE_ENGINE_REVISION, 'converter_version': get_converter_version(),
                  'optimization': {**run.optimization.to_dict(), 'calibration_digest': calibration_digest(run.calibration)},
                  'format': spec.cli, 'architecture': detection.architecture,
                  'quant_metadata_transport': 'inline' if inline_quant_metadata(layers) else 'header',
                  'layer_shapes': {d.layer: list(info.by_name(d.name).shape) for d in decisions if d.layer in layers},
                  'target_runtime': args.target_runtime,
                  'architecture_policy': detection.policy.to_dict(), 'source_config': info.config,
                  'hf': getattr(args, '_hf_provenance', None),
                  'reference_recipe': reference_identity,
                  'key_map': getattr(info, '_source_names', {}),
                  'source_hashes': _portable_hash_manifest(info.files, source_hashes),
                  'quality_gate': run.gate, 'metrics': run.metrics,
                  'rotation': ('regular-h4-' + args.rotation_backend + '-fp32-v1') if (spec.algorithm == FORMAT_W4A4 or spec.cli == 'int8-convrot') else None,
                  'torch_version': torch.__version__,
                  'kitchen_version': importlib.metadata.version('comfy-kitchen') if run.kitchen else None,
                  'encoding_backend': 'kitchen-eager-fp32' if run.kitchen else 'torch-fp32',
                  'memory_budget': args.max_memory,
                  'runtime_verified': False,
                  'offloading': 'ComfyUI controls residency and transfers; no offload scheduler is embedded'}
    if detection.classifier_info.get('variant'):
        provenance['model_variant'] = detection.classifier_info['variant']
    spool_file = run.spool.file
    spool_file.flush()
    spool_file.seek(0)
    digest = hashlib.sha256()
    while chunk := spool_file.read(1 << 20):
        digest.update(chunk)
    provenance['payload_sha256'] = digest.hexdigest()
    metadata['comfy_quantizer'] = json_dumps(provenance)
    return metadata


def _publish_native(run: _NativeRun, info: CheckpointInfo, metadata: Dict[str, str], out: str,
                    identity: Optional[Tuple[int, int]], initial_identity: Dict[str, Any]) -> Dict[str, Any]:
    candidate = os.path.join(run.tmp, 'candidate.safetensors')
    run.spool.publish_candidate(candidate, metadata)
    events.emit('phase', phase='validation')
    check = verify_native_output(candidate)
    events.emit('validation', ok=check['ok'],
                checks=1, errors=check['errors'][:5])
    if not check['ok']:
        raise ValidationError('; '.join(check['errors']))
    if _source_identity(info) != initial_identity:
        raise InputError('source files changed during conversion; refusing to publish')
    events.emit('phase', phase='publication')
    _publish_output(candidate, out, overwrite=run.args.overwrite, expected_identity=identity)
    _fsync_parent(out)
    return check
