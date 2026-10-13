"""Native payload, scale and metadata integrity checks."""

from __future__ import annotations

import json
import torch
from typing import Any, Dict, List

from native_quantizer.architectures.policies import TEXT_TOKEN_EMBEDDING
from native_quantizer.checkpoint_io.reader import RawSafetensorsFile
from native_quantizer.errors import InputError
from native_quantizer.files import sha256_safetensors_payload
from native_quantizer.formats import NATIVE_FORMATS
from native_quantizer.identity import FORMAT_INT8, FORMAT_W4A4
from native_quantizer.metadata import read_quant_metadata


def verify_native_output(path: str) -> Dict[str, Any]:
    errors: List[str] = []
    with RawSafetensorsFile(path) as raw:
        try:
            provenance = json.loads(raw.metadata['comfy_quantizer'])
            layers = read_quant_metadata(path)['layers']
            if not isinstance(layers, dict):
                raise ValueError('layers must be an object')
            source_format = provenance.get('format')
            spec = NATIVE_FORMATS.get(source_format)
            if spec is None:
                errors.append(f'unknown native output format {source_format!r}')
            elif spec.algorithm is None and layers:
                errors.append(f'{source_format}: precision conversion has quantized layers')
            elif spec.algorithm is not None and not layers:
                errors.append(f'{source_format}: no quantized layers recorded')
            if sha256_safetensors_payload(path) != provenance.get('payload_sha256'):
                errors.append('payload hash mismatch')
            algorithms = {f.algorithm: f for f in NATIVE_FORMATS.values() if f.algorithm}
            for layer, conf in layers.items():
                if not isinstance(layer, str) or not layer or not isinstance(conf, dict):
                    errors.append('malformed layer metadata')
                    continue
                fmt = conf.get('format')
                if fmt not in algorithms:
                    errors.append(f'{layer}: unknown format {fmt}')
                    continue
                if spec is not None and fmt != spec.algorithm:
                    errors.append(f'{layer}: format {fmt} disagrees with {source_format}')
                # Token embeddings deliberately use plain INT8, including in
                # an INT8 ConvRot conversion, because lookup has no input
                # activation to rotate.
                token_embedding = bool(TEXT_TOKEN_EMBEDDING.search(layer + '.weight'))
                if source_format == 'int8-convrot' and not token_embedding and conf.get('convrot') is not True:
                    errors.append(f'{layer}: missing INT8 ConvRot metadata')
                if source_format == 'int4-convrot' and conf.get('linear_dtype') != 'int4':
                    errors.append(f'{layer}: missing INT4 activation request')
                weight = layer + '.weight'
                dtype, shape, _, _ = raw.get(weight)
                expected = {FORMAT_INT8: torch.int8, FORMAT_W4A4: torch.int8,
                            'float8_e4m3fn': torch.float8_e4m3fn,
                            'float8_e5m2': torch.float8_e5m2,
                            'nvfp4': torch.uint8, 'mxfp8': torch.float8_e4m3fn}[fmt]
                if dtype != expected or len(shape) != 2:
                    errors.append(f'{layer}: invalid weight dtype/shape')
                logical_shape = provenance.get('layer_shapes', {}).get(layer)
                if not isinstance(logical_shape, list) or len(logical_shape) != 2 or not all(type(x) is int and x > 0 for x in logical_shape):
                    errors.append(f'{layer}: missing original dimensions')
                    continue
                n, k = logical_shape
                expected_shape = (n, k // 2) if fmt in (FORMAT_W4A4, 'nvfp4') else (n, k)
                if tuple(shape) != expected_shape:
                    errors.append(f'{layer}: storage dimensions disagree with original dimensions')
                if fmt == FORMAT_INT8 and ('convrot' in conf or 'convrot_groupsize' in conf):
                    if conf.get('convrot') is not True or conf.get('convrot_groupsize') != 256 or k % 256:
                        errors.append(f'{layer}: invalid INT8 ConvRot parameters')
                if fmt == FORMAT_W4A4:
                    group = conf.get('convrot_groupsize')
                    if group not in (16, 64, 256) or k % group or k % 64 or conf.get('quant_group_size') != 64 or conf.get('linear_dtype') not in ('int4', 'int8'):
                        errors.append(f'{layer}: invalid W4A4 execution parameters')
                scale = raw.read_tensor(weight + '_scale')
                if fmt in ('nvfp4', 'mxfp8'):
                    group = 16 if fmt == 'nvfp4' else 32
                    expected_scale_shape = (((n + 127) // 128) * 128, (((k // group) + 3) // 4) * 4)
                    if n % group or k % group or tuple(scale.shape) != expected_scale_shape:
                        errors.append(f'{layer}: invalid swizzled block-scale dimensions')
                    if fmt == 'nvfp4' and scale.dtype != torch.float8_e4m3fn:
                        errors.append(f'{layer}: NVFP4 needs E4M3 block scales')
                if fmt == 'mxfp8':
                    if scale.dtype != torch.uint8 or bool((scale == 255).any()):
                        errors.append(f'{layer}: invalid E8M0 scales')
                else:
                    sf = scale.float()
                    if not bool(torch.isfinite(sf).all()) or bool((sf < 0).any()):
                        errors.append(f'{layer}: invalid scales')
                    if fmt != 'nvfp4' and bool((sf == 0).any()):
                        errors.append(f'{layer}: zero scales')
                if fmt in ('float8_e4m3fn', 'float8_e5m2') and (scale.dtype != torch.float32 or scale.numel() != 1):
                    errors.append(f'{layer}: FP8 needs one FP32 tensor scale')
                if fmt == FORMAT_INT8 and (scale.dtype != torch.float32 or tuple(scale.shape) != (shape[0], 1)):
                    errors.append(f'{layer}: INT8 needs FP32 [N,1] scales')
                if fmt == FORMAT_W4A4 and (scale.dtype != torch.float32 or tuple(scale.shape) != (shape[0],)):
                    errors.append(f'{layer}: W4A4 needs FP32 [N] scales')
                if fmt == 'nvfp4':
                    global_scale = raw.read_tensor(weight + '_scale_2').float()
                    if global_scale.numel() != 1 or not bool(torch.isfinite(global_scale).all()) or not bool((global_scale > 0).all()):
                        errors.append(f'{layer}: invalid NVFP4 tensor scale')
        except (KeyError, ValueError, TypeError, AttributeError, InputError) as exc:
            errors.append(str(exc))
    return {'ok': not errors, 'errors': errors, 'runtime_verified': False}
