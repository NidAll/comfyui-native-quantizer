"""Architecture policy selection and precision-preserving decisions."""

from __future__ import annotations

import re
import torch
from typing import List, Optional, Sequence

from native_quantizer import events
from native_quantizer.architectures.detection import DetectionResult
from native_quantizer.architectures.policies import (
    FamilyPolicy,
    REGISTRY,
    TEXT_ENCODER_FAMILIES,
    TEXT_TOKEN_EMBEDDING,
)
from native_quantizer.checkpoint_io.types import CheckpointInfo
from native_quantizer.codecs.w4a8 import w4_weight_is_quantizable
from native_quantizer.codecs.w6a8 import w6_weight_is_quantizable
from native_quantizer.dtypes import FLOAT_DTYPES, torch_dtype_name
from native_quantizer.errors import PolicyError
from native_quantizer.identity import (
    COMPONENT_ALL,
    COMPONENT_AUTO,
    COMPONENT_CHOICES,
    COMPONENT_DIFFUSION,
    COMPONENT_TEXT_ENCODER,
    FORMAT_INT8,
    FORMAT_MIXED,
    FORMAT_W6A8,
    GROUPED_INT8_FORMATS,
    W4A8_CONVROT_GROUPSIZE,
)
from native_quantizer.planning.types import DecisionKind, TensorDecision
from native_quantizer.utils import flatten_regex


def classify_tensors(info: CheckpointInfo, detection: DetectionResult,
                     fmt: str, group_size: Optional[int], include: Sequence[str],
                     exclude: Sequence[str], keep_precision: Sequence[str],
                     output_dtype: Optional[torch.dtype],
                     min_numel: Optional[int],
                     components: str = COMPONENT_AUTO,
                     quantize_text_embeddings: bool = False) -> List[TensorDecision]:
    """Classify diffusion and text-encoder linears without crossing components.

    ``auto`` preserves the historical behavior for diffusion checkpoints and
    automatically switches to text-encoder rules for standalone CLIP/T5/LLM/
    BERT checkpoints. ``all`` additionally finds recognized text towers outside
    an explicitly prefixed diffusion model.
    """
    if components not in COMPONENT_CHOICES:
        raise PolicyError(
            f"unknown component selection {components!r}; "
            f"expected one of {COMPONENT_CHOICES}")
    primary = detection.policy
    if components == COMPONENT_AUTO:
        selected_components = {primary.component}
    elif components == COMPONENT_ALL:
        selected_components = {COMPONENT_DIFFUSION, COMPONENT_TEXT_ENCODER}
    else:
        selected_components = {components}

    prefix = detection.unet_prefix
    if prefix and not any(k.startswith(prefix) for k in info.key_set()):
        prefix = ""
    primary_is_text = primary.component == COMPONENT_TEXT_ENCODER

    try:
        include_re = flatten_regex(list(include)) if include else None
        exclude_re = flatten_regex(list(exclude)) if exclude else None
        keep_re_user = (flatten_regex(list(keep_precision))
                        if keep_precision else None)
    except re.error as exc:
        raise PolicyError(f"invalid user regular expression: {exc}") from exc

    text_matchers = [
        (REGISTRY[family], REGISTRY[family].quantize_re())
        for family in TEXT_ENCODER_FAMILIES
    ]

    decisions: List[TensorDecision] = []
    for meta in info.tensors:
        name = meta.name
        rel = name
        policy: Optional[FamilyPolicy] = None
        component = "other"

        if primary_is_text:
            if COMPONENT_TEXT_ENCODER in selected_components:
                policy = primary
                component = COMPONENT_TEXT_ENCODER
        else:
            in_diffusion_scope = (name.startswith(prefix) if prefix else True)
            if (COMPONENT_DIFFUSION in selected_components
                    and in_diffusion_scope):
                policy = primary
                component = COMPONENT_DIFFUSION
                rel = name[len(prefix):] if prefix else name
            elif COMPONENT_TEXT_ENCODER in selected_components:
                # In ``all`` mode the diffusion prefix always wins. Text rules
                # are only considered elsewhere in the state dict. With an
                # explicit text-only selection they may inspect the full file.
                for text_policy, text_q_re in text_matchers:
                    if text_q_re.search(name):
                        policy = text_policy
                        component = COMPONENT_TEXT_ENCODER
                        rel = name
                        break
                if (policy is None and quantize_text_embeddings
                        and name.startswith("text_encoders.")
                        and TEXT_TOKEN_EMBEDDING.search(name)):
                    policy = REGISTRY["text_encoder_llm"]
                    component = COMPONENT_TEXT_ENCODER
                    rel = name

        if policy is None:
            reason = ("component not selected; passthrough"
                      if primary.component not in selected_components
                      else "outside a recognized selected component; passthrough")
            decisions.append(TensorDecision(
                name, DecisionKind.KEEP, reason, component=component))
            continue
        if not rel.endswith(".weight"):
            decisions.append(TensorDecision(
                name, DecisionKind.KEEP,
                "not a weight tensor (bias/norm/buffer); passthrough",
                component=component))
            continue

        layer = name[:-len(".weight")]
        if exclude_re is not None and exclude_re.search(name):
            decisions.append(TensorDecision(
                name, DecisionKind.KEEP,
                "matched --exclude pattern; passthrough", component=component))
            continue
        if keep_re_user is not None and keep_re_user.search(name):
            decisions.append(TensorDecision(
                name, DecisionKind.KEEP_PRECISION,
                "matched --keep-precision pattern; passthrough",
                component=component))
            continue

        if (quantize_text_embeddings and component == COMPONENT_TEXT_ENCODER
                and TEXT_TOKEN_EMBEDDING.search(rel)):
            if len(meta.shape) != 2 or meta.dtype not in FLOAT_DTYPES:
                decisions.append(TensorDecision(
                    name, DecisionKind.KEEP,
                    "token embedding is not a 2D float weight; passthrough",
                    component=component))
                continue
            decisions.append(TensorDecision(
                name, DecisionKind.QUANTIZE, "opt-in rowwise INT8 token embedding",
                layer=layer, format=FORMAT_INT8, component=component))
            continue

        q_re = policy.quantize_re()
        k_re = policy.keep_re()
        x_re = policy.exclude_re()
        if x_re.search(rel):
            decisions.append(TensorDecision(
                name, DecisionKind.KEEP,
                f"policy exclude for {policy.family}; passthrough",
                component=component))
            continue
        if k_re.search(rel):
            decisions.append(TensorDecision(
                name, DecisionKind.KEEP_PRECISION,
                f"policy keep for {policy.family}; passthrough",
                component=component))
            continue

        candidate = bool(q_re.search(rel)) if policy.quantize else True
        user_forced = bool(include_re is not None and include_re.search(name))
        if user_forced:
            candidate = True
        if not candidate:
            decisions.append(TensorDecision(
                name, DecisionKind.KEEP,
                f"not in {policy.family} quantize set; passthrough",
                component=component))
            continue

        if len(meta.shape) == 2 and any(size % alignment for size, alignment
                                       in zip(meta.shape, policy.linear_alignment)):
            decisions.append(TensorDecision(
                name, DecisionKind.KEEP_PRECISION,
                f'{policy.family} native consumer requires N/K alignment '
                f'{policy.linear_alignment}; source dimensions and precision retained',
                component=component))
            continue

        effective_group_size = (group_size if group_size is not None
                                else policy.group_size)
        effective_min_numel = (min_numel if min_numel is not None
                               else policy.min_weight_numel)
        if fmt in GROUPED_INT8_FORMATS:
            predicate = w6_weight_is_quantizable if fmt == FORMAT_W6A8 else w4_weight_is_quantizable
            ok, why = predicate(
                meta.shape, meta.dtype, effective_group_size,
                W4A8_CONVROT_GROUPSIZE)
            if not ok:
                decisions.append(TensorDecision(
                    name, DecisionKind.KEEP,
                    f"not quantizable: {why}; passthrough",
                    component=component))
                continue
        elif fmt == FORMAT_MIXED:
            if len(meta.shape) != 2 or meta.dtype not in FLOAT_DTYPES:
                decisions.append(TensorDecision(
                    name, DecisionKind.KEEP,
                    "not a 2D float weight; passthrough", component=component))
                continue
        else:
            raise PolicyError(f"unknown quantization format {fmt!r}")
        if meta.nbytes < effective_min_numel * meta.dtype.itemsize:
            decisions.append(TensorDecision(
                name, DecisionKind.KEEP,
                f"small tensor (<{effective_min_numel} elements); passthrough",
                component=component))
            continue
        decisions.append(TensorDecision(
            name, DecisionKind.QUANTIZE,
            "user-forced via --include" if user_forced else
            f"{policy.family} policy candidate",
            layer=layer, group_size=effective_group_size,
            convrot_groupsize=W4A8_CONVROT_GROUPSIZE,
            format=fmt, component=component))

    return decisions


def _emit_plan_layers(info: CheckpointInfo, decisions: Sequence[TensorDecision],
                      cast_format: Optional[str] = None) -> None:
    """Stream bounded plan rows for frontends without changing planner output."""
    if not events.enabled():
        return
    for start in range(0, len(decisions), 200):
        rows = []
        for decision in decisions[start:start + 200]:
            tensor = info.by_name(decision.name)
            casting = bool(cast_format and tensor and tensor.dtype.is_floating_point)
            kind = "cast" if casting else "keep" if cast_format else decision.kind.value
            fmt = (cast_format if casting else decision.format
                   if not cast_format and decision.kind == DecisionKind.QUANTIZE
                   else torch_dtype_name(tensor.dtype) if tensor else None)
            rows.append({"name": decision.name, "kind": kind,
                         "format": fmt, "reason": "precision conversion" if casting else decision.reason,
                         "shape": list(tensor.shape) if tensor else None})
        events.emit("plan_layers", start=start, total=len(decisions), rows=rows)
