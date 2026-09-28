"""Offline execution contracts for native Comfy Kitchen checkpoint layouts.

These describe storage and dispatch costs, not measured latency. The runtime
owns kernel selection, workspace allocation, patching and device movement.
"""
from dataclasses import dataclass
import json
import math


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
    if payload.get('schema') != 'kitchen-costs/v1':
        raise ValueError('Expected kitchen-costs/v1 benchmark data')
    if payload.get('kitchen_version') != kitchen_version:
        raise ValueError('Kitchen version differs from the benchmark; rerun the probe')
    costs = {}
    workloads = []
    for sample in payload['benchmarks']:
        shape = sample['shape']
        m, n, k = (shape[key] for key in ('m', 'n', 'k'))
        if any(type(value) is not int or value <= 0 for value in (m, n, k)):
            raise ValueError('Benchmark shapes must be positive integers')
        workloads.append(dict(m=m, n=n, k=k))
        for result in sample['results']:
            fmt, bits = result['format'], result['activation_bits']
            if fmt not in ('int8_tensorwise', 'convrot_w4a4', 'asym_w4a8_int8') or bits not in (4, 8):
                raise ValueError('Unknown measured kernel format or activation precision')
            value = result['warm_ms']
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError('Benchmark latency must be finite and positive')
            key = (n, k, fmt, bits)
            if key in costs:
                raise ValueError('Use one representative M per weight shape in a cost profile')
            costs[key] = float(value)
    if not costs:
        raise ValueError('No kernel measurements found')
    return costs, workloads
