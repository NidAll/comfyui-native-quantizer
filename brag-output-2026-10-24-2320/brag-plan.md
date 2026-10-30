# Brag Plan: ComfyUI Native Quantizer (`comfyquant`)

## What is this app?

A command-line quantizer that converts ComfyUI diffusion models and text encoders to
native ComfyUI quantization formats — and, unlike the usual "pick a bit width and hope",
it reads the architecture first, chooses per-layer what may be crushed, and proves the
result before it publishes.

## The angle

**Not every tensor deserves 8 bits.** The film is a precision-surgery trailer: a model file
is opened like an artifact, its architecture is read out loud, 310 tensors are split into
196 quantized and 114 preserved, the output is measured, and finally the tool admits what
it did *not* verify. The flex is the receipt: every number on screen came out of a real run
on a real checkpoint (`qwen_3_06b_base.safetensors`) or a real catalog in this checkout.

## Hook (first 3 seconds)

Two lines, hard cut, no logo, no music sting yet:
`310 TENSORS.` → `NOT ALL OF THEM DESERVE 8 BITS.`

## Key moments (the middle)

- **It reads the model.** Real `--inspect` output types out: `detected arch : text_encoder_llm
  (confidence medium)`, then three more real detections (`anima`, `fibo2`, `moss_image`, all
  `confidence high`) from the installed library. Counter: **108** ComfyUI model classes covered.
- **The layer plan.** The real Quantizer Studio panel: `196 quantized · 114 preserved`, with the
  real decision table underneath (`keep / bfloat16`, `quantize / int8_tensorwise`). Then the four
  preserved families named in the repo's own language: norms, modulation, embeddings, input/output
  boundaries.
- **The proof.** Real sizes: `1.11 GiB → 718 MiB`, `196` quantized layers, worst-layer
  reconstruction `1.07%` relative L2, source preserved and published atomically.
- **The reach.** All 13 registered native formats, one chip each — `int8-convrot`, `w4a8`,
  `w6a8`, `nvfp4`, `mxfp8`, `mixed`, …
- **The honesty beat.** The real verify payload `"runtime_verified": false` — and the line
  `IT REPORTS WHAT IT DIDN'T TEST.`

## Outro / punchline

`COMFYUI NATIVE QUANTIZER` → `comfyquant` with the two receipts that cost the most to earn:
`57/57 self-tests` and `108 model classes`. Tagline from the app itself: `ComfyUI native quantization`.

## User flow worth showing

entry → key action → result, from the actual tool:
1. **entry** — `comfyquant qwen_3_06b_base.safetensors --inspect` → architecture read out.
2. **key action** — `--format int8-convrot --dry-run` → the layer plan, `196` candidates, protected
   tensors listed as `keep`.
3. **result** — a real conversion + verification (`196 quantized layers; 718.32 MiB`) and the
   Studio panel showing the same plan.

## Tone

- Preset: `cinematic`
- Creative direction: precision-surgery trailer for a command-line instrument; the numbers carry the drama.
- Interpretation: few scenes, big type, hard cuts on the downbeat, no jokes beyond the deadpan
  honesty beat. Motion is slow arrivals and decisive exits; nothing bounces. The terminal and the
  Studio panel are the hero imagery — no abstract filler.

## Format: landscape — 1920x1080
## Duration: 30.0 s (user-requested; the /brag default window is 15–25 s)

## Visual identity (from the project)

- Background: `#101720` (Quantizer Studio ground), deepest frame `#070B10`
- Panel / log surface: `#101A22`, card `#18232D`, hairline `#2B4B57`
- Accent: `#8BE9E8` (Studio page-title / active tab), secondary `#207986`, warning `#FFD179`
- Text: `#D9E7EC`, muted `#8EA3AD`
- Display font: `Oswald` (condensed poster grotesque, uppercase) — bundled
- Data/terminal font: `JetBrains Mono` — bundled, nearest to the Studio's mono
- Strongest visual element: the Studio's `QUANTIZER // STUDIO` header system — `//` separators,
  letter-spaced cyan chapter labels, hairline rules, and the real `LAYER PLAN` panel.

## Share copy (draft)

The quantizer that reads the model before it touches a weight: 196 layers crushed, 114 preserved, and
a report that says `runtime_verified: false` instead of pretending. `comfyquant`.

## Audio direction

- Role: cinematic support, low in the mix.
- Music: `happy-beats-business-moves-vol-12-by-ende-dot-app.mp3` (steady, clean — the cinematic pick).
- Music treatment: 0.30 bed from 0.0, duck to 0.22 under the honesty beat, hard out at 29.4 s with a
  0.6 s fade so the last logo hit rings.
- Music cue guidance: bundled preset read (109.96 BPM). Strong cues used as locks: 8.74 s (scene cut +
  `196 QUANTIZED`), 17.47 s (the size/error payoff), 24.56 s (nothing — it lands inside the honesty beat,
  so the name slam moves to the next detected beat after 25 s via `hyperframes beats`). Beat grid used
  for the 13 format chips (every third beat only) and the preserved-family rows.
- Audio-reactive treatment: subtle; music RMS drives the accent hairline glow, the terminal panel's
  border presence and the final wordmark glow. No waveform/equalizer imagery.
- SFX posture: sparse, motion-matched; a heavy soft impact for the two hook lines, keys for typed
  terminal text, one bell on the size payoff, dry ticks for chips, one bell on the wordmark.
- Audio-coupled moments: typed inspect lines (keys), 196/114 counters (chips), format chips (drops),
  `runtime_verified: false` (single dry hit, no music emphasis), wordmark (bell).
- Restraint rule: no whooshes on every cut, no SFX during the honesty line except one dry hit,
  nothing louder than the voice a trailer would have.

## Storyboard

### Scene 1 — 310 / 8 bits — 4.30 s
Near-black. A single terminal line types in, small, top-left: `$ comfyquant qwen_3_06b_base.safetensors`.
`310 TENSORS.` lands centre in condensed caps, dead still. Hard cut mid-beat to `NOT ALL OF THEM DESERVE 8 BITS.`
in the same weight, cyan on the word `8 BITS`.
Sequential/interaction: yes — command line types character by character.
Audio intent: quiet, mechanical, then weight.
Audio-coupled idea: keyboard ticks while typing; `impactSoft_heavy` under the second line.
Music: bed enters at 0.0, low.
Transition mood: hard → Scene 2.

### Scene 2 — IT READS THE MODEL FIRST — 4.44 s
Headline right of frame; left, a terminal panel types the real inspect block
(`tensors : 310 (1.11 GiB)`, `detected arch : text_encoder_llm (confidence medium)`).
Then three real detections stack in below (`anima`, `fibo2`, `moss_image`, `confidence high`).
`108` counts up in the corner with `COMFYUI MODEL CLASSES`.
Sequential/interaction: yes — inspect block types, then three detections arrive one by one.
Audio intent: clinical, accumulating.
Audio-coupled idea: keys for typing; one soft drop per detection; a chip-stack tick on `108`.
Music: steady.
Transition mood: dramatic cut into the plan → Scene 3 at 8.74 s (strong cue).

### Scene 3 — 196 QUANTIZED / 114 PRESERVED — 5.46 s
`196` and `114` slam in as a pair of oversized numerals, `#8BE9E8` and muted. Under them, the real
Studio `LAYER PLAN` panel slides up (crop of the actual TUI: `text_encoder_llm · 196 quantized · 114 preserved`
and rows `keep / bfloat16`, `quantize / int8_tensorwise`). Four preserved families reveal as a row:
`NORMS · MODULATION · EMBEDDINGS · BOUNDARIES`.
Sequential/interaction: yes — the four families arrive quickly, then hold.
Audio intent: the film's first real weight.
Audio-coupled idea: two card-place hits on the numerals; drops on the four families.
Music: accent on the cut.
Transition mood: hard → Scene 4.

### Scene 4 — 1.11 GiB → 718 MiB — 4.36 s
The real model-list strip appears (`qwen_3_06b_base.safetensors  1.11 GiB`) and the size cuts to
`718 MiB` with `−37%` beside it. Below: `196 LAYERS · WORST RELATIVE L2 1.07%` and
`SOURCE PRESERVED · ATOMIC PUBLISH`.
Sequential/interaction: yes — size swaps on the beat; the two small claims arrive after.
Audio intent: payoff.
Audio-coupled idea: `impactBell_heavy_003` on the number swap at 17.47 s.
Music: strong cue lock.
Transition mood: hard → Scene 5.

### Scene 5 — 13 NATIVE FORMATS — 3.81 s
A grid of 13 real format names, mono, each on a hairline row, arriving in three quick waves
(`int8`, `int8-convrot`, `w4a4`, `int4-convrot`, `w4a8`, `w6a8`, `mixed`, `fp8-e4m3`, `fp8-e5m2`,
`nvfp4`, `mxfp8`, `fp16`, `bf16`). Header: `13` + `REGISTERED NATIVE FORMATS`.
Sequential/interaction: yes — three waves of chips, then the full set holds.
Audio intent: mechanical enumeration, fast.
Audio-coupled idea: drops on the first chip of each wave only.
Music: beat-grid aligned.
Transition mood: hard, drop the music a step → Scene 6.

### Scene 6 — IT REPORTS WHAT IT DIDN'T TEST — 3.83 s
Near-still. The real verification payload appears as live JSON:
`{ "ok": true, "errors": [], "runtime_verified": false }`. One line lands beneath it:
`IT REPORTS WHAT IT DIDN'T TEST.` Nothing moves after that.
Sequential/interaction: none — one arrival, then a hold.
Audio intent: the room goes quiet; one dry hit.
Audio-coupled idea: single `impactWood_light` on the line, no bell.
Music: ducked.
Transition mood: soft crossfade → Scene 7.

### Scene 7 — Wordmark — 3.80 s
`COMFYUI NATIVE QUANTIZER` in condensed caps, then `comfyquant` beneath it in mono with the app's
subtitle `ComfyUI native quantization`. Two receipts sit under a hairline: `57/57 SELF-TESTS` and
`108 MODEL CLASSES`. Hold, fade to black.
Sequential/interaction: none — one arrival, long hold.
Audio intent: resolve.
Audio-coupled idea: `impactBell_heavy_000` on the wordmark; music out at 29.4 s.
Transition mood: fade out.

**Music mood for this video:** cinematic, restrained, driving
**Audio summary:** a low steady bed with mechanical accents that gets louder only where the numbers
land, then deliberately strips back for the honesty beat before one final bell.

## Sources used (all real, captured this run)

- `comfyquant --inspect` on `qwen_3_06b_base.safetensors`, `Anima-Lightning-step400-int8-convrot`,
  `fibo2_edit_turbo_int8_convrot`, `moss_image_sft_int8_convrot` (installed library).
- `comfyquant --dry-run` on the same encoder: 310 decisions, 196 `quantize`, 114 `keep`.
- A real `--format int8-convrot --validate` conversion: 1,192,099,840 B → 753,210,796 B,
  196 layers, per-layer relative L2 min 0.79% / mean 0.84% / max 1.07%.
- `comfyquant --self-test`: `self-tests: 57/57 passed`.
- `comfyquant --audit-architectures`: `"model_classes": 108`.
- `comfyquant --no-config --list-formats`: the 13 format rows quoted above.
- Quantizer Studio screenshots (Textual `export_screenshot`) of the `LAYER PLAN` panel and the
  model list, captured headlessly from the real app.

## Privacy substitutions

The Studio screenshots were captured with the directory field shown as `~/ComfyUI/models/...` and
cropped to drop the absolute `Path` column and the command-log line, so no local username or home
path appears in the film. No performance timings are shown (machine-tied numbers are excluded);
sizes, counts and reconstruction error are artifact facts.
