# Compatibility and evidence

Format serialization, architecture selection, native loading, execution and
model quality are separate capabilities. Registry coverage does not certify every
model/format/backend combination.

## Dependencies

| Component | Requirement |
| --- | --- |
| Python | 3.12–3.13 |
| Core | NumPy >=1.24 and safetensors >=0.4.3 |
| Compute | Environment-supplied PyTorch; CPU CI uses 2.11.0 |
| Kitchen extra | comfy-kitchen >=0.2.36,<0.3 with the requested layout APIs |
| W6A8 consumer | Explicit Kitchen six-bit API and ComfyUI registration of `w6a8_int8` |
| Hub input | Optional `hub` extra; downloads require explicit input options |

Lightweight imports, help and catalogs work without PyTorch. GPU builds must
match the intended backend and consumer. A supported file format may need a
separately installed architecture loader. See [Iris](models/iris.md) and
[Kandinsky 6](models/kandinsky6.md) for extension-specific contracts.

## Coverage

| Surface | Regression evidence | Evaluation still needed |
| --- | --- | --- |
| CLI/API | Options, lazy imports, exit codes, config precedence and installed wheel smoke | Additional platform/environment combinations |
| Checkpoint integrity | Protected tensors, source identities, corruption, failed gates, interrupted writes and atomic publication | Full-size operational workloads |
| Quantized formats | Synthetic encoding, packing, independent reconstruction, metadata and output verification | Every model/consumer/backend combination |
| MSE optimization | Native-grid scoring, both objectives, baseline fallback, bounded paths, calibration and resume | Representative model calibration and output quality |
| Native integer runtime | Configured consumer load/save, device movement, forwards and graph probes | Dynamic model shapes, offload, LoRA and complete generation |
| NVFP4/MXFP8 | Encoding/decoding, CPU consumer loading and FP32 CUDA encoding | Accelerated inference and full-workflow behavior |
| Architecture policies | Independent signatures, variant separation, selection and precision guards | Real checkpoint and workflow acceptance per architecture |
| Iris | Tiny complete native forward and aligned projection dispatch | Full-checkpoint quality, workload memory and adapter behavior |
| Kandinsky 6/SR | Native namespaces, dimensions and protected-tensor conversion fixtures | Complete extension forwards and audio/video/SR quality |

Current checks and commands are in [validation](release-validation.md).
Unevaluated combinations remain experimental. Reconstruction gates measure
weight or sampled activation error; they do not guarantee perceptual quality.

W4A8/W6A8/mixed resume only matching jobs. Other formats restart atomically.
Original dimensions are preserved. Odd-row fallback and graph limitations are
covered in [W6A8](w6a8.md). Working-memory budgets do not describe total process
memory or inference residency.

## Making runtime or quality claims

Preserve the source, validate the output, load it in the intended consumer,
execute representative forwards/generations and compare output quality. Record
software/consumer revisions and reproducible inputs. Measure throughput, memory,
offload and adapters separately when advertising those capabilities. A passing
synthetic test or static kernel capability is insufficient for a model-level
claim. Public releases also require the configured CI and installed-distribution
checks in the [release process](releasing.md).
