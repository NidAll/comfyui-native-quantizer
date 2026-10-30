# Hyperframes Composition Brief: ComfyUI Native Quantizer (comfyquant)

## Objective
Create a 30-second cinematic launch film for `comfyquant`, the ComfyUI native quantizer in this
repository. It must feel like a precision instrument trailer — real terminal output, the real
Quantizer Studio layer plan, and numbers taken from a real run — not a feature list.

## Output
- Composition directory: `brag-output-2026-10-24-2320/composition/`
- Rendered video: `brag-output-2026-10-24-2320/brag.mp4`
- Format: landscape — 1920x1080, 30 fps
- Duration: 30.0 s (explicit user request; overrides the 15–25 s brag default)

## Source Material
- Project root: `/home/nidall/AI/comfyui_quantizer_v2`
- Primary files read: `README.md`, `docs/support.md`, `CHANGELOG.md`, `quantizer_tui/app.py`,
  `quantizer_tui/studio.tcss`, plus live CLI runs (see `brag-plan.md` → Sources).
- Product name: ComfyUI Native Quantizer; CLI name `comfyquant`; Studio name `QUANTIZER // STUDIO`
- Tagline / strongest claim: "ComfyUI native quantization" (Studio subtitle). Repository line:
  "converts diffusion checkpoints and text encoders to native ComfyUI quantization formats".
- Key UI/visual moment to recreate: the Studio `LAYER PLAN` panel — `text_encoder_llm · 196 quantized ·
  114 preserved` over a real decision table (`keep / bfloat16`, `quantize / int8_tensorwise`).
- Real assets in the project (already copied):
  - `assets/img/layer-plan.png` — real Studio LAYER PLAN crop
  - `assets/img/model-list.png` — real Studio model list crop (Filename/Size, no paths)
- Copy that must appear verbatim:
  - `310 TENSORS.` / `NOT ALL OF THEM DESERVE 8 BITS.`
  - `detected arch : text_encoder_llm (confidence medium)`
  - `detected arch : anima (confidence high)` / `: fibo2` / `: moss_image`
  - `196 quantized · 114 preserved`
  - `196 LAYERS · WORST RELATIVE L2 1.07%` (real max from the run)
  - `"runtime_verified": false`
  - `IT REPORTS WHAT IT DIDN'T TEST.`
  - `57/57 SELF-TESTS` · `108 MODEL CLASSES`
  - the 13 format ids: `int8`, `int8-convrot`, `w4a4`, `int4-convrot`, `w4a8`, `w6a8`, `mixed`,
    `fp8-e4m3`, `fp8-e5m2`, `nvfp4`, `mxfp8`, `fp16`, `bf16`

## Creative Direction
- Tone preset: `cinematic`
- Creative direction: precision-surgery trailer for a command-line instrument
- Interpretation: 7 short scenes, hard cuts on strong cues, condensed uppercase display type,
  dead-still landed titles, no bounce, no floating/pulsing loops, no abstract filler. Real terminal
  frames and one real Studio panel carry the imagery.
- Angle: not every tensor deserves 8 bits; the tool decides per layer, and then reports what it did
  not verify.
- Hook: `310 TENSORS.` → `NOT ALL OF THEM DESERVE 8 BITS.`
- Outro: `COMFYUI NATIVE QUANTIZER` / `comfyquant` / `57/57 SELF-TESTS · 108 MODEL CLASSES`
- Avoid:
  - Generic SaaS language, "streamline", "supercharge", "10x"
  - Invented metrics, invented benchmarks, machine timings, hardware names
  - Waveform/equalizer graphics, particle sparkles, neon grids, fake CRT scanlines
  - Redesigning the product's UI — reproduce it, don't restyle it

## Visual Identity
- Background: `#101720`; deepest black `#070B10`
- Panel/log: `#101A22`; card `#18232D`; hairline `#2B4B57`
- Text: `#D9E7EC`; muted `#8EA3AD`
- Accent: `#8BE9E8` (Studio title cyan); secondary `#207986`; warning `#FFD179`
- Display font: `Oswald` (700/500, uppercase, tight tracking)
- Mono font: `JetBrains Mono` (400/500/700)
- Visual references from the project: Studio header system (`QUANTIZER // STUDIO`), letter-spaced
  cyan section labels with `//`, hairline rules, RichLog panel (`#101A22`, 1px `#2B4B57` border),
  DataTable rows with the accent selection bar.
- LOOK line: ground `#101720` matte terminal black, one cyan `#8BE9E8` accent used only on the number
  that matters in each scene, condensed Oswald caps at 120–220 px for headlines, 13 px mono labels
  with 0.18em tracking, hairlines at 1px `#2B4B57`, panels with 1px borders and no shadows; no
  gradients except a single radial glow behind the wordmark.

## Storyboard
Use `brag-plan.md` as the creative contract. Scene summary (absolute times):

1. `310 / 8 BITS` — 0.00–4.30 — command types; `310 TENSORS.` then `NOT ALL OF THEM DESERVE 8 BITS.`
2. `READ` — 4.30–8.74 — real inspect block types; three detections stack; `108` counts up.
3. `PLAN` — 8.74–14.20 — `196` / `114` numerals slam; real LAYER PLAN panel; four preserved families.
4. `PROVE` — 14.20–18.56 — `1.11 GiB → 718 MiB`, `−37%`, `WORST RELATIVE L2 1.07%`, atomic publish.
5. `REACH` — 18.56–22.37 — 13 format chips in three waves.
6. `HONESTY` — 22.37–26.20 — real verify JSON with `"runtime_verified": false`; one line; hold.
7. `WORDMARK` — 26.20–30.00 — name, `comfyquant`, two receipts, fade.

## Audio
- Audio role: cinematic support
- Audio arc: steady bed at 0.30; accents only where numbers land; ducked to 0.20 for the honesty
  beat; out at 29.4 s.
- Music: `assets/music/happy-beats-business-moves-vol-12-by-ende-dot-app.mp3`
- Music treatment: 0.0 start, no fade-in (it begins under the typing), duck 22.4–26.2, 0.6 s fade at
  the end; final bell rings over the fade.
- Music cue guidance: bundled preset `/home/nidall/.agents/skills/brag/assets/music/cues/happy-beats-business-moves-vol-12-by-ende-dot-app.music-cues.json`
  (109.96 BPM; strong cues 8.74, 17.47, 19.66, 22.37, 22.93). Cuts at 8.74 and 19.66 are within
  ±0.15 s of strong cues; the 17.47 number swap is locked to its cue. Beats past 25 s: run
  `npx hyperframes beats` on the project and use the grid for the wordmark and the format chips.
- Audio-reactive treatment: subtle; RMS → accent hairline opacity/glow + wordmark radial glow +
  terminal border presence. No waveform visuals.
- Audio-coupled moments:
  - Scene 1 — keypress ticks while the command types; heavy soft impact under line 2
  - Scene 2 — keypress ticks for the inspect block; one soft drop per detection; one tick for `108`
  - Scene 3 — two card placements on `196`/`114`; drops for the four families
  - Scene 4 — bell on the size swap at 17.47 s
  - Scene 5 — drops on the first chip of each wave only
  - Scene 6 — one dry wood knock on the line; nothing else
  - Scene 7 — `impactBell_heavy_000` on the wordmark
- SFX selection guidance: `/home/nidall/.agents/skills/brag/assets/sfx/sfx-analysis.md`; prefer
  low high-frequency-risk files; keyboard set for typing, `casino/chip-lay-*` or `chips-stack-*` for
  counters, `interface/drop_00*` for chips, `impact/impactSoft_heavy_001` for the hook,
  `impact/impactBell_heavy_003` for the size payoff, `impact/impactWood_light_002` for the honesty line.
- Exact SFX choice, timestamps, density and volume: Hyperframes decides from the implemented motion,
  then copies the chosen files into `composition/assets/sfx/`.
- Copy every chosen file into `composition/assets/` before referencing it; root-relative paths.

## Hyperframes Instructions
Load the composition-building Hyperframes skills — `hyperframes` (contract + `data-*` timing,
`references/format.md`; motion in `references/motion.md`; look in `references/look.md`),
`hyperframes-registry` (only if a ready-made effect is genuinely needed), `media-use` (audio,
audio-reactive data) and `hyperframes-cli` (lint/check/snapshot/render). /brag is its own workflow:
skip the storyline step, `references/types/*`, `references/story.md`, `SPEC.md`, `usage`, `history`,
`feedback` and `REVIEW.md`; `check` is the gate.

Requirements:
- Show the real Studio panel and real CLI output; at least one scene is unmistakably this product.
- Keep all text readable; headlines 120–220 px, labels ≥ 20 px at 1080p.
- Total duration exactly 30.0 s.
- Include the planned music/SFX layer.
- Do not invent product numbers; use only the values in `brag-plan.md`.
- Run `npx hyperframes check` before render; fix every error including contrast.
- Keep creation and rendering local.
