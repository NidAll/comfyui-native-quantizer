# LFM3-VL-2B

The experimental `lfm3_vl_2b` architecture identifies the upstream multimodal
checkpoint using shape-checked language, vision, and projector tensors. It was
reviewed against upstream revision `f83d2b0a2296aaa277f394990a9b8b949ddf9691`.
The matching header has 424 tensors, including 142 selected language matrices
(1,465,909,248 parameters); its tied `lm_head.weight` is absent. The policy selects
language block feed-forward `w1/w2/w3`, convolution operator
`in_proj/out_proj`, and attention `q/k/v/o` linear weights. It preserves the
attention gate, depthwise convolution, normalizations, tied token embedding and
head, vision tower, and multimodal projector. A policy does not supply a loader;
the intended consumer is the package-local PromptStyler LFM3-VL runtime.

`int8-convrot` defaults to rotation group 256. The optional
`--int8-convrot-groupsize 16`, `64`, or `256` chooses another supported group;
input widths not divisible by that group remain at source precision. This
option cannot be combined with `--reference-recipe`, which owns exact layer
settings. `--exclude` and `--keep-precision` can retain selected blocks for
sensitivity comparisons. For example:

```sh
comfyquant source.safetensors --architecture lfm3_vl_2b \
  --format int8-convrot --int8-convrot-groupsize 64 \
  --exclude 'language_model\.layers\.(0|23)\.' \
  --output lfm3-int8.safetensors --device cpu --validate
```

The consumer's `scripts/prepare_lfm3.py` can convert an offline local upstream
model and embed its configuration, processor, tokenizer, chat template and
generation settings into one safetensors file, or bundle a BF16 reference. The
result contains no executable model code. Preserve the original checkpoint.
Synthetic conversion and verification do not establish generation quality,
runtime compatibility, speed, or a best rotation group; those need separate
full-model comparisons on the intended consumer.

Full-checkpoint text/image inference has also been exercised through the
PromptStyler Next and Anima Prompt Architect consumers using native INT8
operators, with BF16 reference probes and short/long generation checks. The
integration remains experimental: broad prompt fidelity and image-quality
equivalence are not established. Weight-only MSE can improve one probe while
worsening another; it is not enabled by default for this recipe.
