"""Bounded activation sampling for an explicitly supplied inference model.

Use as a context around representative eager inference. Hooks are removed on
exit, including exceptions. Capture is an offline operation, not graph-safe.
"""
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


class CalibrationCapture:
    def __init__(self, modules, *, rows=64, rows_per_call=4,
                 max_bytes=256 << 20, seed=0):
        if rows <= 0 or rows_per_call <= 0 or max_bytes <= 0:
            raise ValueError('Capture budgets must be positive')
        self.modules = dict(modules)  # checkpoint weight key -> module
        self.rows = rows
        self.rows_per_call = rows_per_call
        self.max_bytes = max_bytes
        self.seed = seed
        self.samples = {}
        self.seen = {}
        self.rng = {}
        self.handles = []
        self.used_bytes = 0

    def __enter__(self):
        if self.handles:
            raise RuntimeError('Capture is already active')
        try:
            for name, module in self.modules.items():
                if not name.endswith('.weight'):
                    raise ValueError('Capture keys must match checkpoint .weight keys')
                self.handles.append(module.register_forward_pre_hook(self._hook(name)))
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _hook(self, name):
        def capture(module, inputs):
            x = inputs[0].detach()
            if x.ndim < 2 or not x.is_floating_point():
                raise ValueError(f'{name}: expected floating linear activations')
            width = x.shape[-1]
            count = x.numel() // width
            if count == 0:
                return
            if name not in self.samples:
                size = self.rows * width * 4
                if self.used_bytes + size > self.max_bytes:
                    raise MemoryError(f'{name}: calibration budget exhausted; select fewer layers or rows')
                self.samples[name] = torch.empty(self.rows, width, dtype=torch.float32)
                self.used_bytes += size
                self.seen[name] = 0
                self.rng[name] = random.Random(f'{self.seed}:{name}')
            elif self.samples[name].shape[1] != width:
                raise ValueError(f'{name}: activation width changed during capture')
            # Equal samples per invocation cover diffusion steps/prompts instead
            # of retaining only the first call. Gather before the device transfer.
            take = min(count, self.rows_per_call)
            destinations, indices = [], []
            for index in range(take):
                seen = self.seen[name]
                dest = seen if seen < self.rows else self.rng[name].randrange(seen + 1)
                self.seen[name] += 1
                if dest < self.rows:
                    destinations.append(dest)
                    indices.append(index * count // take)
            if indices:
                index = torch.tensor(indices, device=x.device)
                coords = torch.unravel_index(index, x.shape[:-1])
                selected = x[coords].to(device='cpu', dtype=torch.float32)
                # Sequential writes preserve reservoir semantics on duplicate slots.
                for dest, row in zip(destinations, selected):
                    self.samples[name][dest].copy_(row)
        return capture

    def save(self, path):
        if self.handles:
            raise RuntimeError('Exit capture before serializing calibration')
        if not self.samples:
            raise ValueError('No activations captured')
        path = Path(path)
        arrays = {}
        for name, samples in self.samples.items():
            samples = samples[:min(self.rows, self.seen[name])]
            if not torch.isfinite(samples).all():
                raise ValueError(f'{name}: non-finite calibration samples')
            arrays[name] = samples.numpy()
        fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.npz')
        try:
            with os.fdopen(fd, 'wb') as stream:
                np.savez(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            # Publish without replacing an existing calibration artifact.
            os.link(temporary, path)
        finally:
            os.unlink(temporary)
