"""Conversion requests independent of argparse and optional frontend state."""
from __future__ import annotations

from dataclasses import dataclass, field, fields

from native_quantizer.errors import UsageError


@dataclass
class ConversionOptions:
    """The converter copies this request before normalizing execution options."""
    model: str | None = None
    output: str | None = None
    format: str = 'w4a8'
    components: str = 'auto'
    quantize_text_embeddings: bool = False
    codebook_mode: str = 'auto'
    enable_w6a8: bool = False
    w6a8_group_size: int = 16
    profile: str = 'auto'
    kernel_costs: str | None = None
    mixed_objective: str = 'kitchen'
    target_runtime: str = 'auto'
    runtime_certificate: str | None = None
    require_runtime_certificate: bool = False
    strip_gpu_identity: bool = False
    quality_gate: float | None = None
    global_error_gate: float | None = None
    max_linear_bytes_per_param: float | None = None
    max_bf16_fraction: float | None = None
    w4a4_linear_dtype: str = 'int8'
    disable_w4a4: bool = False
    disable_w4a8: bool = False
    disable_int8: bool = False
    require_calibration: bool = False
    architecture: str = 'auto'
    device: str = 'auto'
    compute_dtype: str = 'auto'
    output_dtype: str = 'auto'
    group_size: int | None = None
    min_quantized_byte_fraction: float | None = None
    fail_on_low_compression: bool = False
    calibration_source: str | None = None
    calibration_samples: int | None = None
    calibration_cache: str | None = None
    seed: int = 0
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    keep_precision: list[str] = field(default_factory=list)
    sensitivity_threshold: float | None = None
    error_threshold: float | None = None
    max_memory: str | int = '2G'
    streaming: bool = True
    resume: bool = False
    overwrite: bool = False
    dry_run: bool = False
    inspect: bool = False
    validate: bool = False
    validation_only: bool = False
    metadata_only: bool = False
    report: str | None = None
    log_level: str = 'info'
    json_log: str | None = None
    events_jsonl: str | None = None
    trust_pickle: bool = False
    allow_extra_shard_tensors: bool = False
    nonfinite_policy: str = 'error'
    experimental: bool = False
    yes: bool = False
    architecture_policy: str | None = None
    key_map: str | None = None
    hf_repo: str | None = None
    hf_revision: str = 'main'
    hf_subfolder: str = ''
    hf_filename: str | None = None
    hf_cache: str | None = None
    offline: bool = False
    full_precision_matmul: bool = False
    progress: str = 'auto'
    rotation_backend: str = 'matrix'
    _hf_provenance: dict | None = None

    @classmethod
    def from_namespace(cls, namespace) -> ConversionOptions:
        values = vars(namespace)
        return cls(**{item.name: values[item.name] for item in fields(cls)
                      if item.name in values})


def validate_conversion_options(args) -> None:
    from native_quantizer.formats import FORMAT_ALIASES, NATIVE_FORMATS
    from native_quantizer.config import DEFAULT_CHOICES

    choices = {**DEFAULT_CHOICES, 'components': ('auto', 'diffusion', 'text_encoder', 'all'),
               'output_dtype': ('auto', 'fp16', 'bf16'),
               'nonfinite_policy': ('error', 'keep'), 'w4a4_linear_dtype': ('int4', 'int8'),
               'profile': ('auto', 'balanced', 'conservative', 'size-first', 'throughput')}
    for key, values in choices.items():
        if getattr(args, key) not in values:
            raise UsageError(f'--{key.replace("_", "-")} must be one of {", ".join(values)}')

    args.format = FORMAT_ALIASES.get(args.format, args.format)
    if args.format not in ('w4a8', 'w6a8', 'mixed') and args.format not in NATIVE_FORMATS:
        raise UsageError(f'unknown conversion format {args.format!r}')
    if args.resume and args.overwrite:
        raise UsageError("--resume and --overwrite are mutually exclusive")
    if args.enable_w6a8 and args.format != "mixed":
        raise UsageError("--enable-w6a8 requires --format mixed")
    if args.w6a8_group_size <= 0 or args.w6a8_group_size % 16:
        raise UsageError("--w6a8-group-size must be a positive multiple of 16")
    if args.group_size is not None and args.group_size != 16:
        raise UsageError(
            "--group-size must be 16: only the validated W4A8 configuration "
            "(group_size=16, convrot_groupsize=256) is supported by the "
            "production runtime")
