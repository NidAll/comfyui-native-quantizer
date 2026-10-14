#!/usr/bin/env python3
"""Compare baseline/MSE on synthetic checkpoints in fresh processes. No downloads.

Measures complete API conversion with validation, independent reconstruction,
first/warm conversion time, process peak RSS and CUDA peak allocated memory.
RSS includes the interpreter and libraries; CUDA allocated bytes exclude caches.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FORMATS = ('int8', 'int8-convrot', 'w4a4', 'w4a8', 'w6a8', 'fp8-e4m3', 'fp8-e5m2', 'nvfp4', 'mxfp8', 'mixed')


def worker(args):
    import resource
    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from native_quantizer import api
    from native_quantizer.metadata import read_quant_metadata
    from native_quantizer.identity import FORMAT_INT8, FORMAT_W4A4, FORMAT_W4A8, FORMAT_W6A8
    from native_quantizer.options import ConversionOptions
    from native_quantizer.validation.reference import (reference_decode_int8, reference_decode_w4a4,
                                                       reference_decode_w4a8, reference_decode_w6a8, _reference_rotate)
    from native_quantizer.formats import NATIVE_FORMATS
    from native_quantizer.runtime.kitchen import _native_kitchen
    torch.set_num_threads(2)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    g = torch.Generator().manual_seed(321)
    w = torch.randn(args.rows, args.cols, generator=g)
    if args.distribution == 'outliers':
        w[:, ::73] *= 8
    x = torch.randn(24, args.cols, generator=g) * torch.linspace(.1, 3, args.cols)
    name = 'double_blocks.0.img_attn.qkv.weight'
    timings = []
    with tempfile.TemporaryDirectory(prefix='quantizer-mse-benchmark-') as directory:
        root = Path(directory)
        source = root / 'source.safetensors'
        save_file({name: w, 'img_in.weight': torch.zeros(16, args.cols)}, str(source))
        calibration = None
        if args.objective == 'activation':
            calibration = str(root / 'calibration.npz')
            np.savez(calibration, **{name: x.numpy()})
        if args.device == 'cuda':
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats()
        for i in range(args.repeats):
            output = root / f'output-{i}.safetensors'
            options = ConversionOptions(model=str(source), output=str(output), architecture='flux',
                format=args.format, optimizer=args.optimizer, mse_objective=args.objective,
                mse_steps=args.steps, device=args.device, target_runtime='cpu',
                max_memory=args.budget, validate=True, progress='off', log_level='error',
                error_threshold=.5, global_error_gate=.5 if args.format == 'mixed' else None,
                profile='size-first' if args.format == 'mixed' else 'auto',
                calibration_source=calibration if args.optimizer == 'mse' or args.format in ('mixed', 'w4a8', 'w6a8') else None)
            if args.device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                if api.convert(options) != 0:
                    raise RuntimeError('conversion/validation failed')
            if args.device == 'cuda':
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
        peak_cuda = torch.cuda.max_memory_allocated() if args.device == 'cuda' else None
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if not api.verify_output(str(output))['ok']:
            raise RuntimeError('source-free verification failed')
        conf = read_quant_metadata(str(output))['layers'][name[:-7]]
        with safe_open(str(output), framework='pt') as st:
            p = {key.removeprefix(name): st.get_tensor(key) for key in st.keys() if key.startswith(name)}
        fmt = conf['format']
        if fmt == FORMAT_INT8:
            dq = reference_decode_int8(p[''], p['_scale'])
            if conf.get('convrot'):
                dq = _reference_rotate(dq, conf['convrot_groupsize'])
        elif fmt == FORMAT_W4A4:
            dq = reference_decode_w4a4(p[''], p['_scale'], conf['convrot_groupsize'])
        elif fmt == FORMAT_W4A8:
            dq = reference_decode_w4a8(p[''], p['_s_rel'], p['_s_channel'], p['_codebook'], conf['group_size'], conf['convrot_groupsize'])
        elif fmt == FORMAT_W6A8:
            dq = reference_decode_w6a8(p[''], p['_s_rel'], p['_s_channel'], conf['group_size'], conf['convrot_groupsize'])
        elif fmt.startswith('float8_'):
            dq = p[''].float() * p['_scale']
        else:
            layout = _native_kitchen(NATIVE_FORMATS[fmt])
            fields = ({'scale': p['_scale_2'], 'block_scale': p['_scale']} if fmt == 'nvfp4'
                      else {'scale': p['_scale'].view(torch.float8_e8m0fnu)})
            dq = layout.dequantize(p[''], layout.Params(**fields, orig_dtype=torch.float32, orig_shape=tuple(w.shape)))
        error = dq.double() - w.double()
        return dict(format=args.format, selected_format=fmt, optimizer=args.optimizer,
                    distribution=args.distribution, objective=args.objective, shape=list(w.shape),
                    weight_mse=float(error.square().mean()),
                    activation_mse=float((error @ x.double().T).square().mean()),
                    first_seconds=timings[0], warm_median_seconds=statistics.median(timings[1:] or timings),
                    seconds=timings, process_peak_rss_bytes=peak_rss, peak_cuda_allocated_bytes=peak_cuda,
                    output_bytes=output.stat().st_size, payload_verified=True, native_inference_measured=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    p.add_argument('--formats', nargs='+', choices=FORMATS, default=FORMATS)
    p.add_argument('--distributions', nargs='+', choices=('gaussian', 'outliers'), default=('gaussian', 'outliers'))
    p.add_argument('--objective', choices=('weight', 'activation'), default='weight')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--rows', type=int, default=128)
    p.add_argument('--cols', type=int, default=512)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--steps', type=int, default=17)
    p.add_argument('--budget', default='32M')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--format', choices=FORMATS, help=argparse.SUPPRESS)
    p.add_argument('--distribution', help=argparse.SUPPRESS)
    p.add_argument('--optimizer', choices=('none', 'mse'), help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.rows <= 0 or args.cols <= 0 or args.rows % 32 or args.cols % 256 or args.repeats < 1:
        p.error('rows must be a positive multiple of 32, cols of 256, repeats >= 1')
    if args.worker:
        print(json.dumps(worker(args)))
        return
    if not args.output:
        p.error('--output is required')
    records, errors = [], []
    for distribution in args.distributions:
        for fmt in args.formats:
            for optimizer in ('none', 'mse'):
                command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--format', fmt,
                           '--distribution', distribution, '--optimizer', optimizer, '--objective', args.objective,
                           '--device', args.device, '--rows', str(args.rows), '--cols', str(args.cols),
                           '--repeats', str(args.repeats), '--steps', str(args.steps), '--budget', args.budget]
                result = subprocess.run(command, capture_output=True, text=True, timeout=600,
                                        env={**os.environ, 'OMP_NUM_THREADS': '2'})
                if result.returncode:
                    errors.append(dict(format=fmt, distribution=distribution, optimizer=optimizer,
                                       error=result.stderr[-6000:]))
                    print(f'{distribution} {fmt} {optimizer}: FAILED (recorded)', flush=True)
                    continue
                record = json.loads(result.stdout)
                records.append(record)
                print(f'{distribution} {fmt} {optimizer}: MSE={record["weight_mse"]:.7g}, '
                      f'warm conversion={record["warm_median_seconds"]:.3f}s', flush=True)
    summary = []
    lookup = {(r['distribution'], r['format'], r['optimizer']): r for r in records}
    for base in (r for r in records if r['optimizer'] == 'none'):
        mse = lookup.get((base['distribution'], base['format'], 'mse'))
        if mse is None:
            continue
        summary.append(dict(format=base['format'], distribution=base['distribution'],
                            weight_mse_reduction_percent=100 * (1 - mse['weight_mse'] / max(base['weight_mse'], 1e-300)),
                            activation_mse_reduction_percent=100 * (1 - mse['activation_mse'] / max(base['activation_mse'], 1e-300)),
                            warm_conversion_ratio=mse['warm_median_seconds'] / base['warm_median_seconds'],
                            peak_rss_difference_bytes=mse['process_peak_rss_bytes'] - base['process_peak_rss_bytes']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(schema='native-mse-benchmark/v1', device=args.device,
        steps=args.steps, budget=args.budget, records=records, errors=errors, summary=summary,
        limits='Synthetic checkpoint conversion, not model quality or inference speed; RSS includes interpreter/libraries.\n'), indent=2) + '\n')
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
