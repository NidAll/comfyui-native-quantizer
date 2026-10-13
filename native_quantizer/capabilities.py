"""Machine-readable conversion capabilities shared by optional frontends."""

from __future__ import annotations

from typing import Any, Dict

from native_quantizer.formats import FORMAT_TO_KITCHEN_LAYOUT, NATIVE_FORMATS, NATIVE_UNSUPPORTED_OPTIONS
from native_quantizer.identity import FORMAT_W4A8, FORMAT_W6A8, MIXED_PROFILES


def frontend_capabilities() -> Dict[str, Any]:
    """Capabilities for optional local frontends; CLI policies stay authoritative."""
    formats = {
        'w4a8': {'algorithm': FORMAT_W4A8, 'engine': 'mixed', 'resume': True,
                 'description': 'Kitchen W4A8',
                 'kitchen_layouts': [FORMAT_TO_KITCHEN_LAYOUT[FORMAT_W4A8]]},
        'w6a8': {'algorithm': FORMAT_W6A8, 'engine': 'mixed', 'resume': True,
                 'description': 'Kitchen uniform W6A8',
                 'kitchen_layouts': [FORMAT_TO_KITCHEN_LAYOUT[FORMAT_W6A8]]},
        'mixed': {'algorithm': 'per-layer', 'engine': 'mixed', 'resume': True,
                  'description': 'Kitchen-aware precision selection',
                  'kitchen_layouts': list(FORMAT_TO_KITCHEN_LAYOUT.values())},
    }
    formats.update({name: {'algorithm': spec.algorithm, 'engine': 'native',
                           'resume': False, 'description': spec.description,
                           'kitchen_layouts': ([FORMAT_TO_KITCHEN_LAYOUT[spec.algorithm]]
                                               if spec.algorithm in FORMAT_TO_KITCHEN_LAYOUT
                                               else [spec.kitchen_class] if spec.kitchen_class
                                               else [])}
                    for name, spec in NATIVE_FORMATS.items()})
    return {'schema': 'quantizer-capabilities/v1', 'formats': formats,
            'profiles': list(MIXED_PROFILES),
            'native_unsupported_options': list(NATIVE_UNSUPPORTED_OPTIONS)}
