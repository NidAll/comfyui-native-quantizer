"""Portable optimizer configuration; safe to import without PyTorch."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from native_quantizer.errors import CalibrationError, UsageError

MSE_REVISION = 'native-mse-v1'


@dataclass(frozen=True)
class OptimizationConfig:
    mode: str = 'none'
    objective: str = 'auto'
    steps: int = 17
    iterations: int = 3

    def __post_init__(self):
        if self.mode not in ('none', 'mse') or self.objective not in ('weight', 'activation', 'auto'):
            raise UsageError('invalid optimizer or MSE objective')
        if type(self.steps) is not int or not 2 <= self.steps <= 129:
            raise UsageError('--mse-steps must be an integer in [2,129]')
        if type(self.iterations) is not int or not 0 <= self.iterations <= 16:
            raise UsageError('--mse-iterations must be an integer in [0,16]')

    @classmethod
    def from_options(cls, args):
        return cls(getattr(args, 'optimizer', 'none'), getattr(args, 'mse_objective', 'auto'),
                   getattr(args, 'mse_steps', 17), getattr(args, 'mse_iterations', 3))

    @property
    def enabled(self):
        return self.mode == 'mse'

    def to_dict(self):
        return {**asdict(self), 'revision': MSE_REVISION}

    def factors(self):
        # Include modest expansion as well as clipping. Baseline is always separate.
        return tuple(0.5 + 0.55 * i / (self.steps - 1) for i in range(self.steps))


def layer_activations(calibration, name):
    if calibration is not None:
        for key in (name, name.removesuffix('.weight')):
            if key in calibration.layers:
                return calibration.layers[key]['samples']
    return None


def check_optimization_calibration(config, calibration, decisions, require=False):
    from native_quantizer.planning.types import DecisionKind
    if require or (config.enabled and config.objective == 'activation'):
        missing = [d.name for d in decisions if d.kind == DecisionKind.QUANTIZE
                   and layer_activations(calibration, d.name) is None]
        if missing:
            raise CalibrationError('calibration missing for targeted layers: ' + ', '.join(missing[:5]))


def calibration_digest(calibration):
    """Bind resume to the actual selected activation rows, including cache hits."""
    if calibration is None:
        return None
    import hashlib
    digest = hashlib.sha256()
    for name, stats in sorted(calibration.layers.items()):
        samples = stats['samples'].detach().contiguous().cpu()
        digest.update(repr((name, tuple(samples.shape))).encode())
        digest.update(memoryview(samples.numpy()).cast('B'))
    return digest.hexdigest()


def read_optimization_metadata(record):
    """Validate optional provenance; checkpoints predating MSE remain valid."""
    if record is None:
        return OptimizationConfig()
    import re
    if not isinstance(record, dict) or not {'mode', 'objective', 'steps', 'iterations', 'revision'} <= record.keys():
        raise UsageError('incomplete optimization metadata')
    if record['revision'] != MSE_REVISION:
        raise UsageError('unsupported MSE optimizer revision')
    digest = record.get('calibration_digest')
    if digest is not None and (not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest)):
        raise UsageError('invalid optimizer calibration digest')
    return OptimizationConfig(record['mode'], record['objective'], record['steps'], record['iterations'])
