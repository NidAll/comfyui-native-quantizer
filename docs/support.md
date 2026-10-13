# Support and evidence

Candidate: **CLI/library 2.3.0rc2**. This is a release candidate,
not a certification of every registered architecture, backend or workflow.
Exact checks for this candidate are recorded in [release-validation.md](release-validation.md).

## Compatibility contract

- Python 3.12–3.13; the configured CI matrix covers this range. Both interpreters and their fresh CPU dependencies are exercised locally;
  hosted CI publication remains a separate remote gate.
- PyTorch is supplied by the selected compute environment. CPU CI requests
  PyTorch 2.11.0. GPU builds must match the chosen backend and consumer.
- Core dependencies are NumPy >=1.24 and safetensors >=0.4.3. Lightweight
  imports, help and catalogs work without PyTorch or the optional frontend.
- Native Kitchen layouts require compatible APIs. The optional dependency range
  is >=0.2.36,<0.3; W6A8 requires explicit six-bit parameters and ComfyUI
  registration of `w6a8_int8`.
- Studio is deferred; its source, tests, launchers and dependencies are excluded.
- Linux CLI integration is tested. GPU/ROCm, other operating systems and
  individual loader revisions require their own consumer evidence.

## Evidence matrix

| Surface | Routine proof | Remaining evaluation |
| --- | --- | --- |
| Core import, CLI, typed API | Lazy imports, options, exit codes, source/CLI parity and installed wheel smoke | Other platform/environment combinations |
| Checkpoint publication | Tiny fixtures exercise source preservation, corruption rejection, failed gates, source mutation, interrupted writes and version-bound resume | Full-size operational workloads |
| INT8/ConvRot, W4A4, FP8, NVFP4, MXFP8 and precision conversion | Synthetic codecs, protected tensors, metadata and output verification; optional Kitchen decoding | Every model/consumer/backend combination and accelerated dispatch |
| W4A8, W6A8 and mixed planning | Synthetic packing, independent reconstruction, bounds, calibration/planner and matching resume checks | Model-level quality and workload-specific memory/performance |
| Prefix metadata and W6A8 native consumer | Explicit ComfyUI CPU load/save tests; separate bounded CUDA forward/movement/replay probe | Other loader revisions, offload/LoRA behavior and complete generation |
| Architecture registry | Independent signatures, variants, policy reachability and optional installed-class source audit | Real checkpoint and workflow acceptance per architecture |
| Iris-3B | Fused/unfused headers, precision boundaries, full tiny native forward and aligned real-shape CUDA dispatch without weight dequantization | Full-checkpoint quality and workload/offload/LoRA acceptance |
| Kandinsky 6 video/audio and SR | Independent native signatures, dimensions, repeated-linears and protected-tensor synthetic conversions | Full extension forwards and real video/audio/SR quality |

The architecture catalog is policy coverage. A consumer may require a dedicated
loader; adding a policy does not install one. Unevaluated architecture/format/
backend combinations remain experimental. Reconstruction gates measure weight
or scoped activation error; they are not a guarantee of image/audio/video quality.

Known Kitchen odd-row CUDA fallback behavior is documented in [w6a8.md](w6a8.md).
Serialization preserves original dimensions; graph support remains
consumer-dependent. Working-memory limits cover temporary conversion work,
not total process residency or inference VRAM.

## Stable promotion

Promote a named model/format/consumer/backend combination only after preserving
the source, validating the converted file, loading it in the intended consumer,
executing a real forward/generation and comparing representative output quality.
Measure throughput, peak memory, offload and LoRA behavior when those claims are
advertised. Record consumer/dependency revisions and reproducible commands.

Do not extend stable claims from a passing synthetic test or a registry entry.
The release process must also pass the full hosted CI matrix and installed
distribution checks before a stable tag is published.
