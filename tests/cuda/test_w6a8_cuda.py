"""Synthetic published W6A8 checkpoints through native ComfyUI CUDA loading.
Run separately where CUDA is visible: python -m unittest -v tests.cuda.test_w6a8_cuda
Timing records contain no machine identity and cover small synthetic linears only.
"""
import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from tests.support import isolated_module as setUpModule, load_comfy_cpu
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
import comfyui_native_quantizer as q

@unittest.skipUnless(torch.cuda.is_available() and q.w6a8_api_available(), 'CUDA and Kitchen W6A8 API required')
class W6A8CudaTests(unittest.TestCase):
    def test_published_outputs_native_forward_movement_and_graph(self):
        records = self._probe_rows(64) + self._probe_rows(65)
        self.assertTrue(all(record['graph_replay'] for record in records if record['n'] == 64))
        self.assertEqual(len(records), 12)

    def _probe_rows(self, n):
        import comfy_kitchen as kitchen
        ops, _ = load_comfy_cpu()
        torch.set_num_threads(2)
        records=[]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            name='double_blocks.0.img_attn.qkv.weight'
            weight=torch.randn(n,256,generator=torch.Generator().manual_seed(123)) / 16
            source=root/'source.safetensors'
            save_file({name:weight,'img_in.weight':torch.randn(32,256)},str(source))
            for group in (16,32,64):
                output=root/f'g{group}.safetensors'
                with contextlib.redirect_stdout(io.StringIO()):
                    status=q.main([str(source),'--format','w6a8','--output',str(output),
                                   '--w6a8-group-size',str(group),'--architecture','flux','--validate','--yes',
                                   '--max-memory','32M','--device','cuda','--progress','off'])
                self.assertEqual(status,0)
                self.assertTrue(q.verify_output(str(output))['ok'])
                with safe_open(str(output),framework='pt') as st:
                    state={'weight'+suffix:st.get_tensor(name+suffix) for suffix in ('','_s_rel','_s_channel')}
                    conf=json.loads(st.metadata()[q.METADATA_KEY_QUANT])['layers'][name[:-7]]
                state['comfy_quant']=torch.tensor(list(json.dumps(conf).encode()),dtype=torch.uint8)
                module=ops.mixed_precision_ops(compute_dtype=torch.bfloat16).Linear(256,n,bias=False,device='cpu')
                module.load_state_dict(state,strict=True)
                module.to('cuda').to('cpu').to('cuda')
                cls=q.require_w6a8_api()
                params=cls.Params(scale=state['weight_s_rel'],s_channel=state['weight_s_channel'],
                    group_size=group,convrot_groupsize=256,codebook=None,orig_dtype=torch.bfloat16,orig_shape=(n,256))
                from comfy_kitchen.tensor import QuantizedTensor
                direct=QuantizedTensor(state['weight'],'AsymW4A8Int8Layout',params).to('cuda')
                for batch in (1,64):
                    with self.subTest(group=group,batch=batch), torch.inference_mode(), kitchen.use_backend('cuda'):
                        x=torch.randn(batch,256,device='cuda',dtype=torch.bfloat16)
                        torch.cuda.synchronize()
                        baseline=torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record(); actual=module(x); end.record(); end.synchronize()
                        first=start.elapsed_time(end)
                        first_peak=torch.cuda.max_memory_allocated()-baseline
                        expected=torch.nn.functional.linear(x,direct)
                        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
                        self.assertTrue(torch.isfinite(actual).all())
                        for _ in range(3):
                            module(x)
                        torch.cuda.synchronize()
                        baseline=torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
                        start.record()
                        for _ in range(10):
                            actual=module(x)
                        end.record(); end.synchronize()
                        warm=start.elapsed_time(end)/10
                        peak=torch.cuda.max_memory_allocated()-baseline
                        graph=torch.cuda.CUDAGraph()
                        graph_ok=True
                        graph_note=None
                        try:
                            with torch.cuda.graph(graph):
                                captured=module(x)
                            graph.replay(); torch.cuda.synchronize()
                            torch.testing.assert_close(captured,actual,atol=0,rtol=0)
                        except RuntimeError as exc:
                            if n % 16 == 0 or "Cannot copy between CPU and CUDA tensors during CUDA graph capture" not in str(exc):
                                raise
                            captured=None
                            graph_ok=False
                            graph_note="Kitchen 0.2.36 odd-N larger-M eager fallback creates an unpinned CPU shift tensor"
                        saved=module.state_dict()
                        clone=ops.mixed_precision_ops(compute_dtype=torch.bfloat16).Linear(256,n,bias=False,device='cuda')
                        clone.load_state_dict(saved,strict=True)
                        torch.testing.assert_close(clone(x),actual,atol=0,rtol=0)
                        records.append(dict(format=q.FORMAT_W6A8,group_size=group,n=n,k=256,m=batch,
                            first_ms=first,warm_ms=warm,first_temporary_peak_bytes=first_peak,
                            warm_temporary_peak_bytes=peak,native_loader=True,graph_replay=graph_ok,graph_note=graph_note,
                            exact_direct_agreement=True,native_save_reload=True))
                        del graph,captured,clone,saved,actual,expected,x
                del direct,module
        return records


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class IrisShapeCudaTests(unittest.TestCase):
    def test_aligned_iris_projection_shapes_keep_native_int8(self):
        if not os.environ.get('IRIS_ROOT'):
            self.skipTest('optional Iris shape probe: set IRIS_ROOT')
        ops, _ = load_comfy_cpu()
        from comfy_kitchen.tensor import TensorWiseINT8Layout
        generator = torch.Generator().manual_seed(83)
        for n, k in ((2560, 2560), (6912, 2560), (2560, 6912)):
            state = {'weight': torch.randint(-8, 8, (n, k), generator=generator, dtype=torch.int8),
                     'weight_scale': torch.full((n, 1), .01),
                     'comfy_quant': torch.tensor(list(json.dumps({'format':'int8_tensorwise',
                         'convrot':True,'convrot_groupsize':256}).encode()), dtype=torch.uint8)}
            layer = ops.mixed_precision_ops(compute_dtype=torch.bfloat16).Linear(k, n, bias=False, device='cpu')
            layer.load_state_dict(state, strict=True)
            layer.to('cuda')
            self.assertEqual(layer.quant_format, 'int8_tensorwise')
            for batch in (1, 16):
                with self.subTest(n=n, k=k, batch=batch), torch.inference_mode(), patch.object(
                        TensorWiseINT8Layout, 'dequantize', side_effect=AssertionError('weight dequantization')):
                    actual = layer(torch.randn(batch, k, device='cuda', dtype=torch.bfloat16))
                    self.assertEqual(tuple(actual.shape), (batch, n))
                    self.assertTrue(torch.isfinite(actual).all())
            del layer, state


if __name__ == '__main__':
    unittest.main()
