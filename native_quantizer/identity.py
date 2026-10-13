"""Converter and algorithm identities used by metadata and resume checks."""

from __future__ import annotations

from typing import Dict


CONVERTER_NAME = "comfyui_native_quantizer"


_CONVERTER_VERSION = "2.3.0rc2"


def get_converter_version() -> str:
    """Return the current converter version (observable at call time).

    Internal code must read the version through this function (or
    set_converter_version) so that plan-hash/resume checks observe changes;
    the self-tests temporarily mutate it to prove version drift is rejected.
    """
    return _CONVERTER_VERSION


def set_converter_version(version: str) -> None:
    """Override the converter version (self-tests only; not for production use)."""
    global _CONVERTER_VERSION  # noqa: PLW0603
    _CONVERTER_VERSION = version


CONVERTER_VERSION = _CONVERTER_VERSION


FORMAT_W4A8 = "asym_w4a8_int8"


FORMAT_W6A8 = "w6a8_int8"


GROUPED_INT8_FORMATS = (FORMAT_W4A8, FORMAT_W6A8)


FORMAT_MIXED = "mixed"


FORMAT_W4A4 = "convrot_w4a4"


FORMAT_INT8 = "int8_tensorwise"


MIXED_FORMATS = (FORMAT_W4A8, FORMAT_W4A4, FORMAT_INT8, FORMAT_W6A8)


FORMAT_ORIGINAL = "original"   # planner-only candidate: keep at source precision


W4A4_QUANT_GROUP_SIZE = 64


W4A4_EMISSION_MAX = 7


DEFAULT_W4A4_LINEAR_DTYPE = "int8"  # default execution variant: "int4" or "int8"


INT8_SCALE_MAX = 127


CODEBOOK_SAMPLE_COUNT = 300000  # canonical Lloyd-Max sample size (chunk-invariant)


CODEBOOK_MODES = ("auto", "fixed", "fit")


FIXED_W4A8_CODEBOOK = (
    -0.980602, -0.794529, -0.638165, -0.500986,
    -0.377321, -0.263187, -0.155210, -0.050720,
    0.052541, 0.156985, 0.265284, 0.379533,
    0.502636, 0.638953, 0.794876, 0.980671,
)


W4A8_GATE_KURTOSIS = -0.1


CODEBOOK_DECISION_SAMPLE_COUNT = 1 << 22


CODEBOOK_KURTOSIS_SAMPLE_COUNT = 1 << 19


W4A8_ALS_ITERS = 2


W4A4_MAX_REL_L2 = 0.20


INT8_MAX_REL_L2 = 0.05


MIXED_PROFILES = ("auto", "balanced", "conservative", "size-first", "throughput")


FORMAT_MIXED_REVISION = "mixed-r1"


W4A8_CONVROT_GROUPSIZE = 256


QUANT_ALGORITHM_REVISION = "adaptive-codebook-searchsorted-r3"


MIXED_PLANNER_REVISION = "kitchen-planner-r4"


VALIDATION_REVISION = "validator-r5"  # independent reference decoders in recon checks


CALIBRATION_REVISION = "calibration-r2"  # content-addressed cache (v2 fingerprint)


COMPONENT_AUTO = "auto"


COMPONENT_DIFFUSION = "diffusion"


COMPONENT_TEXT_ENCODER = "text_encoder"


COMPONENT_ALL = "all"


COMPONENT_CHOICES = (COMPONENT_AUTO, COMPONENT_DIFFUSION,
                     COMPONENT_TEXT_ENCODER, COMPONENT_ALL)


def get_algorithm_identity() -> Dict[str, str]:
    """Machine-readable algorithm identity recorded in every output."""
    return {
        "quant_algorithm_rev": QUANT_ALGORITHM_REVISION,
        "w4a8_format_revision": FORMAT_W4A8_REVISION,
        "mixed_planner_revision": MIXED_PLANNER_REVISION,
        "validation_revision": VALIDATION_REVISION,
        "calibration_revision": CALIBRATION_REVISION,
    }


FORMAT_W4A8_REVISION = "asym-w4a8-int8-r1"


FORMAT_W6A8_REVISION = "kitchen-w6a8-int8-r1"


def format_revision(fmt):
    return {FORMAT_W6A8: FORMAT_W6A8_REVISION, FORMAT_MIXED: FORMAT_MIXED_REVISION}.get(fmt, FORMAT_W4A8_REVISION)


MAX_SAFETENSORS_HEADER_SIZE = 100_000_000


METADATA_KEY_QUANT = "_quantization_metadata"     # official key read by ComfyUI


METADATA_KEY_EXT = "comfy_wxa8"                   # namespaced extension key (never official)


LAYER_CONF_KEY = "comfy_quant"                    # per-layer blob key used by ComfyUI loader


COMFY_KITCHEN_REV = "aa1ab2263dc06225d9de6702dfc087313d4bc971"   # PR #90 merge commit


COMFYUI_PR = 15308


COMFYUI_PR_HEAD = "8c3a2b27c37bd34e87b58846baf962407c92843c"


COMFYUI_BASE = "bdcb886a4705a03cf40f4a7226de9fc7c059fc90"


W4A8_KERNEL_MIN_SM = (8, 0)


TRITON_MIN_VERSION = (3, 7)
