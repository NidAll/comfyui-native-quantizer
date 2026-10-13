"""Header-only checkpoint inventory and architecture inspection."""
from native_quantizer.architectures.detection import detect_architecture
from native_quantizer.checkpoint_io.reader import discover_checkpoint
from native_quantizer.errors import UnknownArchitectureError


def inspect_checkpoint(path: str) -> dict:
    info = discover_checkpoint(path)
    try:
        found = detect_architecture(info)
        architecture = found.architecture
        variant = found.classifier_info.get('variant')
        confidence = found.confidence
        warnings = found.warnings
    except UnknownArchitectureError as exc:
        architecture, variant, confidence, warnings = 'unknown', None, 'none', [str(exc)]
    return {'path': path, 'kind': info.kind, 'files': len(info.files),
            'tensors': len(info.tensors), 'bytes': info.total_bytes,
            'metadata_keys': sorted(info.metadata), 'quantized': info.is_quantized_input,
            'architecture': architecture, 'variant': variant,
            'confidence': confidence, 'warnings': warnings,
            'first_tensors': [tensor.name for tensor in info.tensors[:30]]}
