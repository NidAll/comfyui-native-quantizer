"""Compare policy coverage with ComfyUI source without executing model code."""
from __future__ import annotations

import ast
from pathlib import Path

from native_quantizer.architectures.policies import REGISTRY
from native_quantizer.errors import InputError


def audit_comfy_architectures(root: str) -> dict:
    source = Path(root) / 'comfy/supported_models.py'
    try:
        tree = ast.parse(source.read_text(encoding='utf-8'))
    except (OSError, SyntaxError) as exc:
        raise InputError(f'cannot read ComfyUI model registry: {exc}') from exc
    models = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'models' for t in node.targets):
            if isinstance(node.value, (ast.List, ast.Tuple)) and all(isinstance(n, ast.Name) for n in node.value.elts):
                models = [n.id for n in node.value.elts]
    if models is None:
        raise InputError('ComfyUI source has no static models list; audit the registry manually')
    owners = {name: policy for policy in REGISTRY.values() for name in policy.comfyui_classes}
    covered = [{'class': name, 'family': owners[name].family,
                'status': owners[name].runtime_status} for name in models if name in owners]
    return {'schema': 'quantizer-architecture-audit/v1', 'model_classes': len(models),
            'covered': covered, 'missing': sorted(set(models) - owners.keys()),
            'known_unsupported': sorted(row['class'] for row in covered if row['status'] == 'unsupported'),
            'scope': 'Registry coverage only; tensor policies, loader dispatch and generation require separate validation.'}
