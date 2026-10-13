"""Synthetic Kitchen CUDA/native ComfyUI loading and CUDA graph validation."""
import argparse
import comfy_kitchen as kitchen
import comfy_kitchen.tensor as kt
import importlib.metadata
import json
import sys
import torch
from pathlib import Path

from native_quantizer import identity
from native_quantizer.codecs import dispatch, rotation, w4a4, w6a8
from native_quantizer.planning import metrics


def run(n, k, m, repeats, comfy_root=None, w6a8_group_size=16, formats=None):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this probe')
    if min(n, k, m, repeats) <= 0 or k % 256:
        raise ValueError('Positive shapes/repeats and K divisible by 256 are required')
    if comfy_root:
        sys.path.insert(0, str(Path(comfy_root).resolve()))
        from comfy.cli_args import args as comfy_args
        comfy_args.cpu = True
        import comfy.ops as ops
    torch.manual_seed(123)
    w = torch.randn(n, k) * (k ** -0.5)
    x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
    results = []
    w6a8.validate_w6_shape(k, w6a8_group_size)
    for fmt, bits in (formats or ((identity.FORMAT_INT8, 8), (identity.FORMAT_W4A4, 4),
                      (identity.FORMAT_W4A4, 8), (identity.FORMAT_W4A8, 8), (identity.FORMAT_W6A8, 8))):
        group = w6a8_group_size if fmt == identity.FORMAT_W6A8 else 16
        cgs = w4a4._pick_w4a4_convrot_group(k) if fmt == identity.FORMAT_W4A4 else 256
        tensors = dispatch.quantize_weight_by_format(w, fmt, group, cgs)
        conf = {'format': fmt}
        if fmt in identity.GROUPED_INT8_FORMATS:
            cls = kt.AsymW4A8Int8Layout
            params = cls.Params(scale=tensors['_s_rel'], s_channel=tensors['_s_channel'],
                                codebook=tensors.get('_codebook'), group_size=group,
                                convrot_groupsize=256, orig_dtype=x.dtype, orig_shape=(n, k))
            conf.update(group_size=group, convrot_groupsize=256, convrot=True)
        elif fmt == identity.FORMAT_W4A4:
            cls = kt.TensorCoreConvRotW4A4Layout
            params = cls.Params(scale=tensors['_scale'], convrot_groupsize=cgs,
                                linear_dtype=f'int{bits}', orig_dtype=x.dtype, orig_shape=(n, k))
            conf.update(convrot_groupsize=cgs, linear_dtype=f'int{bits}')
        else:
            cls = kt.TensorWiseINT8Layout
            params = cls.Params(scale=tensors['_scale'], orig_dtype=x.dtype, orig_shape=(n, k))
        if comfy_root:
            module = ops.mixed_precision_ops(compute_dtype=x.dtype).Linear(k, n, bias=False, device='cpu')
            sd = {'weight' + suffix: value for suffix, value in tensors.items()}
            sd['comfy_quant'] = torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8)
            module.load_state_dict(sd, strict=True)
            module.to('cuda')
            forward = lambda: module(x)
            packed = module.weight
        else:
            packed = kt.QuantizedTensor(tensors[''], cls.__name__, params).to('cuda')
            forward = lambda: torch.nn.functional.linear(x, packed)
        roundtrip = packed.to('cpu').to('cuda')
        torch.testing.assert_close(roundtrip._qdata, packed._qdata)
        del roundtrip
        with torch.inference_mode(), kitchen.use_backend('cuda'):
            first_start, first_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            first_baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            first_start.record()
            output = forward()
            first_end.record()
            first_end.synchronize()
            first_ms = first_start.elapsed_time(first_end)
            first_peak = torch.cuda.max_memory_allocated() - first_baseline
            for _ in range(3):
                forward()
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                output = forward()
            end.record()
            end.synchronize()
            warm_ms = start.elapsed_time(end) / repeats
            peak = torch.cuda.max_memory_allocated() - baseline
            reference = metrics.runtime_output_rel_l2(w, tensors, fmt, group, cgs, x.float().cpu(), bits)
            activation = x.float().cpu()
            if fmt != identity.FORMAT_INT8:
                activation = rotation.rotate_activation(activation, rotation.build_hadamard(cgs), cgs)
            aq, scale = (metrics._act_quant_int4(activation) if bits == 4
                         else metrics._act_quant_int8(activation))
            expected = metrics._simulate_quantized_chunk(w, tensors, fmt, group, cgs,
                                                  aq, scale, f'int{bits}')
            agreement = float((output.float().cpu() - expected).norm()
                              / expected.norm().clamp_min(1e-8))
            y_ref = x.float() @ w.to('cuda').T
            error = float(((output.float() - y_ref).norm(dim=1) / y_ref.norm(dim=1).clamp_min(1e-8)).mean())
            # Fused reduced-precision rotation can cross INT4 rounding bins.
            # Validate the quality estimate; report tensor agreement separately.
            if not torch.isfinite(output).all() or abs(error - reference) > 0.025:
                raise AssertionError(f'{fmt}: CUDA error disagrees with reference: {error} vs {reference}; tensor agreement {agreement}')
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = forward()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured, output, atol=0, rtol=0)
            results.append(dict(format=fmt, activation_bits=bits,
                                group_size=group if fmt in identity.GROUPED_INT8_FORMATS else None, first_ms=first_ms,
                                warm_ms=warm_ms, first_temporary_peak_bytes=first_peak,
                                temporary_peak_bytes=peak,
                                output_rel_l2=error, reference_rel_l2=reference,
                                simulation_agreement_rel_l2=agreement,
                                graph_replay=True, native_loader=bool(comfy_root)))
            del graph, captured, output, y_ref
    return {'shape': {'m': m, 'n': n, 'k': k}, 'repeats': repeats, 'results': results}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=512)
    parser.add_argument('--k', type=int, default=1024)
    parser.add_argument('--m', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--comfy-root')
    parser.add_argument('--w6a8-group-size', type=int, default=16)
    args = parser.parse_args(argv)
    sample = run(args.n, args.k, args.m, args.repeats, args.comfy_root, args.w6a8_group_size)
    print(json.dumps({'schema': 'kitchen-costs/v2',
                      'kitchen_version': importlib.metadata.version('comfy-kitchen'),
                      'benchmarks': [sample]}, indent=2))
    return 0
