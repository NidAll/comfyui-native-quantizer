# Comfy-Org native INT8 references

The [official catalogue](https://huggingface.co/Comfy-Org) is the reference for
artifact layouts and per-layer INT8 settings. The 2026-10-23 audit enumerated all
92 public model repositories and read the headers and quantization JSON for all
71 artifacts named INT8/ConvRot. It exported 63 exact, compatible recipes.
Some catalogue entries provide FP8, BF16 or other formats instead of INT8;
their presence does not establish an INT8 recipe.

## Extract and apply an official recipe

From the source checkout, using a Python environment with the converter installed:

```sh
python scripts/read_comfy_org_recipe.py Comfy-Org/Fibo-2.0 \
  diffusion_models/fibo2_edit_turbo_int8_convrot.safetensors \
  --output fibo2-recipe.json

comfyquant original-native-repack.safetensors --no-config \
  --format int8-convrot --reference-recipe fibo2-recipe.json \
  --device cuda --target-runtime nvidia --max-memory 512M \
  --output fibo2-int8.safetensors --validate
```

The extractor resolves and records an immutable repository revision. Supply
`--revision COMMIT` to repeat a particular snapshot. It reads only repository
metadata, the safetensors header, and bounded JSON metadata tensors; it does not
download weights. Unhonored HTTP ranges are refused before reading the response
body. Existing recipe files are preserved.

Conversion reads the local JSON and makes no reference-network requests.
`ConversionOptions(reference_recipe="fibo2-recipe.json", format="int8-convrot", ...)`
provides the same Python API behavior.

The recipe requires the exact weight names and shapes of the reference artifact.
Use the corresponding original **floating-point native repack**, not the already
quantized official file. Fused QKV, fused/padded FFNs and variant boundaries must
already match. The converter performs no fusion, padding or name guessing;
existing `--key-map` can express exact, collision-checked name changes.

Reference recipes explicitly own the quantized layer set, including any official
embedding, vision or auxiliary weights. Other tensors retain source precision;
`--output-dtype bf16` explicitly casts unselected floating tensors if needed.
Include/exclude/keep, component, embedding and matmul overrides cannot be mixed
with a recipe. Each selected layer must pass the numerical gate or publication
fails. The output records the reference identity, recipe digest and layer settings.

## New and corrected architectures

| Family | Native layout and selection |
| --- | --- |
| `moss_image` | Fused or split QKV; main `layers.N.proj` attention and FFN linears. Text refinement, modulation and boundary projections stay floating. Both official Base and SFT artifacts select 168 linears. |
| `fibo2` | Native repacked Dev/Edit Turbo; main and noise-refiner attention/FFN, injection cross-attention, Perceiver attention/FFN. Fused `w13` or split `w1/w3` accepted. Official artifacts select 215 linears. |
| `ming_image` | Recognizes and selects native fused `attention.qkv/out` as well as split attention. Existing Design/Design-Layer workflow distinction remains. |
| Qwen encoders | Recognizes nested `language_model` decoder layouts, including Qwen3.5 hybrid linear attention. Generic policies remain conservative; exact reference recipes can select the official wider tensor set. |
| `trellis2` / Pixal3D | Selects nested `mlp.mlp` and `cross_attn.cross_attn_block` native linears. Generic policies preserve projection heads; an exact reference can reproduce their official selection. |

The installed and reviewed upstream diffusion registry each contain 108 classes.
All have a policy entry; RT-DETR, DepthAnything3, SAM3 and SAM31 remain explicitly
unsupported. Registry coverage is separate from actual tensor selection and runtime
acceptance.

## Model-specific rotation

Official INT8 metadata uses regular ConvRot groups of **256, 64 and 16**, including
within a single file. Exact recipes retain these per-layer settings. The ordinary
`int8-convrot` mode keeps its existing 256-wide rule; incompatible dimensions stay
floating. Native INT8 uses FP32 row scales. The W4A8 `--group-size` option is unrelated
to the ConvRot size and remains unchanged.

Eight audited artifacts were deferred: the bundled Ming TTS checkpoint, two
Ming-Image MoE encoder artifacts, three ControlNets, the MiniMax H3 video VAE and
the SAM 3D Body DINOv3 detection artifact. These require dedicated consumers or
non-2D expert/convolution contracts; the recipe importer refuses them.

The extraction establishes serialized layouts, selected tensors and runtime
settings. It cannot recover unpublished calibration data, clipping or optimization
history, or prove that an official recipe is optimal for a different model.
Optional MSE remains a separate converter choice.

Validation covers complete/partial/prefixed headers, native synthetic conversion,
source-free verification, protected tensor equality, Kitchen reconstruction and
tiny native Fibo2/MOSS forwards. Synthetic CUDA tests exercise 16/64/256 rotation
and plain INT8 with weight dequantization blocked. Full-model image quality,
LoRA behavior, latency and peak VRAM are not established by these tests.
