"""Data-only, exact-layout reference recipes for native INT8 conversion."""
from __future__ import annotations

import hashlib
import json
import re

from native_quantizer.dtypes import FLOAT_DTYPES
from native_quantizer.errors import PolicyError
from native_quantizer.files import _load_json_object


def load_reference_recipe(path, info, detection):
    recipe = _load_json_object(path, 'reference recipe')
    return validate_reference_recipe(recipe, info, detection)


def validate_reference_recipe(recipe, info, detection):
    if (set(recipe) != {'schema', 'source', 'architecture', 'weights', 'layers'}
            or recipe['schema'] != 'comfy-org-int8-recipe/v1'):
        raise PolicyError('reference recipe needs the comfy-org-int8-recipe/v1 schema')
    source = recipe['source']
    if (not isinstance(source, dict) or set(source) != {'repo', 'revision', 'file'}
            or not isinstance(source['repo'], str)
            or not re.fullmatch(r'Comfy-Org/[A-Za-z0-9._-]+', source['repo'])
            or not isinstance(source['revision'], str)
            or not re.fullmatch(r'[0-9a-f]{40}', source['revision'])
            or not isinstance(source['file'], str) or not source['file'].endswith('.safetensors')):
        raise PolicyError('reference recipe requires a pinned Comfy-Org artifact identity')
    if recipe['architecture'] != detection.architecture:
        raise PolicyError('reference recipe architecture disagrees with detected architecture')
    weights, layers = recipe['weights'], recipe['layers']
    if not isinstance(weights, dict) or not weights or not isinstance(layers, dict) or not layers:
        raise PolicyError('reference recipe requires explicit weight shapes and layer settings')
    actual = {m.name: list(m.shape) for m in info.tensors if m.name.endswith('.weight')}
    if weights != actual:
        raise PolicyError('reference recipe tensor layout mismatch; use the matching native repack, '
                          'or convert without a reference recipe (no fusion/padding is performed)')
    for layer, conf in layers.items():
        if not isinstance(layer, str) or not isinstance(conf, dict):
            raise PolicyError('malformed reference layer')
        weight = info.by_name(layer + '.weight')
        if weight is None or len(weight.shape) != 2 or weight.dtype not in FLOAT_DTYPES:
            raise PolicyError(f'{layer}: reference recipes require original 2D floating weights')
        if (set(conf) - {'format', 'convrot', 'convrot_groupsize', 'full_precision_matrix_mult'}
                or conf.get('format') != 'int8_tensorwise'
                or ('full_precision_matrix_mult' in conf and type(conf['full_precision_matrix_mult']) is not bool)):
            raise PolicyError(f'{layer}: unsupported reference quantization settings')
        if 'convrot' in conf or 'convrot_groupsize' in conf:
            group = conf.get('convrot_groupsize')
            if (conf.get('convrot') is not True or type(group) is not int
                    or group not in (16, 64, 256) or weight.shape[1] % group):
                raise PolicyError(f'{layer}: invalid reference ConvRot alignment')
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return recipe, {'source': source, 'sha256': digest, 'layers': layers}


def recipe_from_header(header, configs, source):
    """Build an exact recipe from metadata, checking a source-precision view."""
    from native_quantizer.architectures.detection import detect_architecture
    from native_quantizer.checkpoint_io.types import CheckpointInfo, TensorMeta
    from native_quantizer.dtypes import torch_dtype_from_safe
    import torch

    tensors = []
    for name, tensor in header.items():
        if name == '__metadata__' or name.endswith(('.weight_scale', '.comfy_quant', '.config')):
            continue
        dtype = torch_dtype_from_safe(tensor['dtype'])
        if name.endswith('.weight') and dtype == torch.int8:
            dtype = torch.float32
        tensors.append(TensorMeta(name, dtype, tuple(tensor['shape']), 0, '', 0, 0))
    info = CheckpointInfo('safetensors', [], header.get('__metadata__', {}), tensors)
    detection = detect_architecture(info)
    if detection.policy.runtime_status == 'unsupported':
        raise PolicyError('reference architecture has no supported native consumer')
    int8_weights = {name[:-7] for name, tensor in header.items()
                    if name != '__metadata__' and name.endswith('.weight') and tensor['dtype'] == 'I8'}
    if set(configs) != int8_weights:
        raise PolicyError('reference metadata does not cover exactly the INT8 weights')
    recipe = {'schema': 'comfy-org-int8-recipe/v1', 'source': source,
              'architecture': detection.architecture,
              'weights': {m.name: list(m.shape) for m in tensors if m.name.endswith('.weight')},
              'layers': configs}
    validate_reference_recipe(recipe, info, detection)
    return recipe
