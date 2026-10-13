# Architecture candidates

**2026-10-13 update:** Iris-3B, Kandinsky 6 native video/audio and Kandinsky 6 SR
policies are implemented in the CLI candidate. See [iris.md](iris.md),
[kandinsky6.md](kandinsky6.md) and [release-validation.md](release-validation.md).
The October 7 shortlist below is historical; its no-policy statements are superseded.

Source review on 2026-10-07. No model weights were downloaded. The installed
ComfyUI model-class registry is already covered by quantizer policies; rerun
`--audit-architectures COMFY_ROOT --json` after updates to detect additions.
Coverage does not certify every variant or current tensor pattern.

| Candidate | Priority and evidence | Integration condition |
| --- | --- | --- |
| Kandinsky 6 Video Lite / Pro | First research priority: a newly released synchronized audio/video family with official source and ComfyUI extensions. The video/audio streams make it a distinct integration from the existing Kandinsky 5 policy. | Inspect exact checkpoint keys and the extension's load path. Confirm native quantized Linear operations and metadata are consumed before adding a native policy; preserve audio/video embeddings, modulation and cross-stream boundaries. No policy is shipped from this source review alone. |
| Kandinsky 6 Video SR | Separate candidate: the official package combines a tiled SR DiT, KVAE and latent upscaler. | Treat the SR transformer independently. Keep VAE/upscaler components dense and verify the custom consumer accepts native layouts. Do not reuse the base-video policy blindly. |
| FLUX 3 Action | Lower priority for this image-workflow quantizer: the available official weights target robotics/action modeling. | A checkpoint name alone is not Flux/Flux2 compatibility. Require a matching ComfyUI consumer and independent action-model tensor/shape contracts before native conversion support. |

Primary sources:

- [Kandinsky 6 source and ComfyUI integration](https://github.com/kandinskylab/kandinsky-6)
- [Kandinsky 6 SR source and component pipeline](https://github.com/kandinskylab/kandinsky-6-sr)
- [Official FLUX 3 Action base model](https://huggingface.co/black-forest-labs/flux-3-action-base)

Upstream generation/install examples can download large checkpoints on first
use; they were not run. These candidates need consumer-contract inspection,
independent header/policy regressions and tiny synthetic load/format checks.
Real quality/performance remains unverified until an explicitly authorized
installed-model evaluation is performed. Use the
`comfyui-quantizer-architecture` skill for the integration workflow.
