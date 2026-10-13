"""Data-only architecture policies and exact checkpoint key mapping."""

from __future__ import annotations

import dataclasses
import math
import re
from typing import Any, Dict, Optional

from native_quantizer.architectures.policies import FamilyPolicy, REGISTRY, _register
from native_quantizer.checkpoint_io.types import CheckpointInfo
from native_quantizer.errors import InputError, PolicyError
from native_quantizer.files import _load_json_object


def architecture_template() -> Dict[str, Any]:
    return {'schema_version': 1, 'family': 'custom_transformer',
            'component': 'diffusion', 'comfyui_classes': [],
            'detect_primary': [],
            'quantize': [r'(^|\.)(blocks|layers|transformer_blocks)\.\d+\..*(to_q|to_k|to_v|q_proj|k_proj|v_proj|o_proj|up_proj|down_proj|gate_proj|fc1|fc2)\.weight$'],
            'keep': [r'(^|\.)(embed_tokens|lm_head|norm|patch_embed|final_layer)(\.|$)'],
            'exclude': [], 'min_weight_numel': 4096,
            'notes': 'Edit patterns using --inspect output. Requires a compatible ComfyUI architecture loader.'}


def load_architecture_policy(path: str) -> FamilyPolicy:
    obj = _load_json_object(path, 'architecture policy')
    allowed = {'schema_version', 'family', 'component', 'comfyui_classes',
               'detect_primary', 'detect_hints', 'quantize', 'keep', 'exclude',
               'min_weight_numel', 'max_rel_l2', 'min_cosine', 'notes'}
    if set(obj) - allowed or obj.get('schema_version') != 1:
        raise PolicyError('architecture policy needs schema_version=1 and only documented fields')
    name = obj.get('family', '')
    if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,79}', name) or name in REGISTRY:
        raise PolicyError('policy family must be a new lowercase identifier; built-in policies cannot be replaced')
    if obj.get('component', 'diffusion') not in ('diffusion', 'text_encoder'):
        raise PolicyError('policy component must be diffusion or text_encoder')
    for field_name in ('comfyui_classes', 'detect_primary', 'detect_hints', 'quantize', 'keep', 'exclude'):
        value = obj.get(field_name, [])
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            raise PolicyError(f'{field_name} must be an array of strings')
        if field_name != 'comfyui_classes':
            for pattern in value:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise PolicyError(f'invalid {field_name} regex: {exc}') from exc
        obj[field_name] = tuple(value)
    if not obj['quantize']:
        raise PolicyError('an explicit quantize allowlist is required')
    minimum = obj.get('min_weight_numel', 4096)
    if type(minimum) is not int or minimum < 1:
        raise PolicyError('min_weight_numel must be a positive integer')
    for key in ('max_rel_l2', 'min_cosine'):
        value = obj.get(key, 0.25 if key == 'max_rel_l2' else 0.95)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise PolicyError(f'{key} must be finite and in [0, 1]')
    obj.pop('schema_version')
    obj.setdefault('comfyui_classes', ())
    obj.setdefault('detect_primary', ())
    policy = FamilyPolicy(**obj, runtime_status='experimental')
    _register(policy)
    return policy


def apply_key_mapping(info: CheckpointInfo, path: Optional[str]) -> None:
    if not path:
        return
    mapping = _load_json_object(path, 'key map')
    if not all(isinstance(k, str) and isinstance(v, str) and v for k, v in mapping.items()):
        raise InputError('key map must contain nonempty target names')
    if set(mapping) - info.key_set():
        raise InputError('key map references missing source tensors')
    metas = [dataclasses.replace(m, name=mapping.get(m.name, m.name)) for m in info.tensors]
    if len({m.name for m in metas}) != len(metas):
        raise InputError('key mapping creates duplicate tensor names')
    info._source_names = {mapping.get(m.name, m.name): m.name for m in info.tensors}
    info.tensors = metas
    info._tensor_by_name = {m.name: m for m in metas}
    info.input_warnings.append('explicit key mapping applied; no tensor splitting, fusion or transpose is performed')
