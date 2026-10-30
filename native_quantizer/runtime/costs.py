"""Offline execution contracts for native Comfy Kitchen checkpoint layouts.

These describe storage and dispatch costs, not measured latency. The runtime
owns kernel selection, workspace allocation, patching and device movement.
"""
import json
import math
import hashlib
import importlib.metadata
from dataclasses import dataclass


@dataclass(frozen=True)
class KernelContract:
    path: str
    activation_bits: int | None
    weight_expansion: bool
    activation_transform: bool
    accelerated: bool | None

    @property
    def preference(self) -> int:
        if self.accelerated is not True:
            return 3
        return 1 if self.weight_expansion else 0


def kernel_contract(fmt: str, *, backend: str, accelerated: bool | None,
                    activation_bits: int | None = None,
                    certain: bool = False, embedding: bool = False) -> KernelContract:
    if embedding:
        return KernelContract('indexed-row-dequantization', None, False, False, False)
    if backend == 'cpu' or accelerated is not True:
        return KernelContract('eager-or-unverified', activation_bits, True,
                              fmt != 'int8_tensorwise', accelerated)
    if fmt == 'int8_tensorwise':
        return KernelContract('int8-linear', 8, False, False, accelerated)
    if fmt == 'convrot_w4a4':
        native = certain and activation_bits == 4 and backend == 'nvidia'
        return KernelContract('native-int4' if native else 'int4-via-int8-or-unverified',
                              activation_bits, not native, True, accelerated)
    if fmt == 'w6a8_int8':
        return KernelContract('int6-via-int8', 8, True, True, accelerated)
    if fmt == 'asym_w4a8_int8':
        return KernelContract('codebook-via-int8', 8, True, True, accelerated)
    raise ValueError(f'Unsupported mixed format: {fmt}')


def load_kernel_costs(path, kitchen_version):
    """Read shape-specific warm CUDA costs produced by benchmark_kitchen.py.

    The file declares the activation workload. It is intentionally opt-in:
    measurements from another workload must never masquerade as universal costs.
    """
    with open(path, encoding='utf-8') as stream:
        payload = json.load(stream)
    schema = payload.get('schema')
    if schema not in ('kitchen-costs/v1', 'kitchen-costs/v2', 'kitchen-costs/v3'):
        raise ValueError('Expected kitchen-costs/v1, v2 or v3 benchmark data')
    if payload.get('kitchen_version') != kitchen_version:
        raise ValueError('Kitchen version differs from the benchmark; rerun the probe')
    if schema == 'kitchen-costs/v3' and payload.get('runtime_fingerprint') != runtime_fingerprint():
        raise ValueError('Runtime differs from the benchmark; rerun the probe on the intended runtime')
    costs = {}
    workloads = []
    for sample in payload['benchmarks']:
        shape = sample['shape']
        m, n, k = (shape[key] for key in ('m', 'n', 'k'))
        if any(type(value) is not int or value <= 0 for value in (m, n, k)):
            raise ValueError('Benchmark shapes must be positive integers')
        weight = sample.get('weight', 1.)
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
            raise ValueError('Workload weight must be finite and positive')
        layer = sample.get('layer')
        if layer is not None and (not isinstance(layer, str) or not layer.endswith('.weight')):
            raise ValueError('Workload layer must be a checkpoint weight key')
        workloads.append(dict(m=m, n=n, k=k, weight=weight, layer=layer))
        for result in sample['results']:
            fmt, bits = result['format'], result['activation_bits']
            if fmt not in ('int8_tensorwise', 'int8-convrot', 'convrot_w4a4', 'asym_w4a8_int8', 'w6a8_int8', 'original') or bits not in (4, 8, 16):
                raise ValueError('Unknown measured kernel format or activation precision')
            if (fmt == 'original') != (bits == 16) or (fmt == 'original' and schema != 'kitchen-costs/v3'):
                raise ValueError('BF16 measurements require v3, original format and 16 activation bits')
            if fmt == 'int8-convrot' and (schema != 'kitchen-costs/v3' or bits != 8 or k % 256):
                raise ValueError('Rotated INT8 costs require v3, A8 and K divisible by 256')
            value = result['warm_ms']
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError('Benchmark latency must be finite and positive')
            group = result.get('group_size')
            if fmt == 'w6a8_int8':
                if schema == 'kitchen-costs/v1' or type(group) is not int or group <= 0 or group % 16 or k % group or k % 256 or bits != 8:
                    raise ValueError('W6A8 costs require v2/v3, A8 and a valid group size dividing K')
            if schema != 'kitchen-costs/v1':
                if fmt == 'asym_w4a8_int8' and group != 16:
                    raise ValueError('W4A8 costs require group_size=16')
                if fmt not in ('asym_w4a8_int8', 'w6a8_int8') and group is not None:
                    raise ValueError('Ungrouped formats must use group_size=null')
                key = (n, k, fmt, bits, group)
                if schema == 'kitchen-costs/v3':
                    if sample.get('dtype') != 'bfloat16':
                        raise ValueError('v3 currently requires bfloat16 activation measurements')
                    key += (m, layer)
            else:
                key = (n, k, fmt, bits)
            if key in costs:
                raise ValueError('Duplicate kernel workload; v1/v2 allow only one representative M per weight shape')
            costs[key] = float(value)
    if not costs:
        raise ValueError('No kernel measurements found')
    return costs, workloads


def runtime_fingerprint():
    """Bind private measurements to the software/device without publishing its name."""
    import torch
    device = None
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        device = (p.name, p.major, p.minor, p.total_memory, p.multi_processor_count)
    identity = (torch.__version__, torch.version.cuda, torch.version.hip, device,
                importlib.metadata.version('comfy-kitchen'))
    return hashlib.sha256(repr(identity).encode()).hexdigest()
