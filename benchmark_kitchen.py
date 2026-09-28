"""Synthetic Kitchen CUDA/native ComfyUI loading and CUDA graph validation."""
import argparse
import json
import importlib.metadata
import sys
from pathlib import Path

import torch
import comfy_kitchen as kitchen
import comfy_kitchen.tensor as kt
import comfyui_native_quantizer as q


def run(n, k, m, repeats, comfy_root=None):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this probe')
    if min(n, k, m, repeats) <= 0 or k % 256:
        raise ValueError('Positive shapes/repeats and K divisible by 256 are required')
    if comfy_root:
        sys.path.insert(0, str(Path(comfy_root).resolve()))
        import comfy.ops as ops
    torch.manual_seed(123)
    w = torch.randn(n, k) * (k ** -0.5)
    x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
    results = []
    for fmt, bits in ((q.FORMAT_INT8, 8), (q.FORMAT_W4A4, 4),
                      (q.FORMAT_W4A4, 8), (q.FORMAT_W4A8, 8)):
        cgs = q._pick_w4a4_convrot_group(k) if fmt == q.FORMAT_W4A4 else 256
        tensors = q.quantize_weight_by_format(w, fmt, 16, cgs)
        conf = {'format': fmt}
        if fmt == q.FORMAT_W4A8:
            cls = kt.AsymW4A8Int8Layout
            params = cls.Params(scale=tensors['_s_rel'], s_channel=tensors['_s_channel'],
                                codebook=tensors['_codebook'], group_size=16,
                                convrot_groupsize=256, orig_dtype=x.dtype, orig_shape=(n, k))
            conf.update(group_size=16, convrot_groupsize=256)
        elif fmt == q.FORMAT_W4A4:
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
            packed = kt.QuantizedTensor(tensors[''], cls, params).to('cuda')
            forward = lambda: torch.nn.functional.linear(x, packed)
        roundtrip = packed.to('cpu').to('cuda')
        torch.testing.assert_close(roundtrip._qdata, packed._qdata)
        del roundtrip
        with torch.inference_mode(), kitchen.use_backend('cuda'):
            first_start, first_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            first_start.record()
            output = forward()
            first_end.record()
            first_end.synchronize()
            first_ms = first_start.elapsed_time(first_end)
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
            reference = q.runtime_output_rel_l2(w, tensors, fmt, 16, cgs, x.float().cpu(), bits)
            activation = x.float().cpu()
            if fmt != q.FORMAT_INT8:
                activation = q.rotate_activation(activation, q.build_hadamard(cgs), cgs)
            aq, scale = (q._act_quant_int4(activation) if bits == 4
                         else q._act_quant_int8(activation))
            expected = q._simulate_quantized_chunk(w, tensors, fmt, 16, cgs,
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
            results.append(dict(format=fmt, activation_bits=bits, first_ms=first_ms,
                                warm_ms=warm_ms, temporary_peak_bytes=peak,
                                output_rel_l2=error, reference_rel_l2=reference,
                                simulation_agreement_rel_l2=agreement,
                                graph_replay=True, native_loader=bool(comfy_root)))
            del graph, captured, output, y_ref
    return {'shape': {'m': m, 'n': n, 'k': k}, 'repeats': repeats, 'results': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=512)
    parser.add_argument('--k', type=int, default=1024)
    parser.add_argument('--m', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--comfy-root')
    args = parser.parse_args()
    sample = run(args.n, args.k, args.m, args.repeats, args.comfy_root)
    print(json.dumps({'schema': 'kitchen-costs/v1',
                      'kitchen_version': importlib.metadata.version('comfy-kitchen'),
                      'benchmarks': [sample]}, indent=2))
