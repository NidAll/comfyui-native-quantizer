"""Observed environment and target-runtime compatibility contracts."""

from __future__ import annotations

import dataclasses
import importlib.util
import numpy as np
import os
import safetensors
import safetensors.torch
import sys
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from native_quantizer.errors import RuntimeCompatibilityError
from native_quantizer.files import _load_json_object
from native_quantizer.identity import FORMAT_INT8, FORMAT_W4A4, FORMAT_W4A8, FORMAT_W6A8, MIXED_FORMATS
from native_quantizer.logging import log
from native_quantizer.runtime.kitchen import w6a8_api_available


@dataclass
class EnvironmentInfo:
    python: str
    torch_version: str
    torch_cuda: Optional[str]
    torch_hip: Optional[str]
    cuda_available: bool
    cuda_device: Optional[str]
    cuda_capability: Optional[Tuple[int, int]]
    rocm_arch: Optional[str]
    safetensors_version: str
    numpy_version: str
    platform: str
    cpu_count: int
    has_comfy_kitchen: bool = False
    comfy_kitchen_has_w4a8_layout: bool = False
    comfy_kitchen_has_w6a8_api: bool = False
    comfy_kitchen_has_w4a4_layout: bool = False
    comfy_kitchen_has_int8_layout: bool = False
    comfy_kitchen_rev: Optional[str] = None
    has_comfy_quant_ops: bool = False
    comfyui_quant_algos: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class RuntimeCertificate:
    """Runtime certificate from the target inference machine: which formats
    actually loaded and executed,
    with the observed effective W4A4 activation precision."""
    backend: str
    gpu: Optional[str]
    cuda_capability: Optional[Tuple[int, int]]
    rocm_arch: Optional[str]
    formats: Dict[str, Dict[str, Any]]


def load_runtime_certificate(path: str) -> RuntimeCertificate:
    """Parse and validate a runtime certificate JSON file.

    The certificate must record actual load and forward execution on the
    target inference backend; capability detection alone is insufficient."""
    try:
        payload = _load_json_object(path, "runtime certificate", nofollow=True)
    except Exception as e:
        raise RuntimeCompatibilityError(
            f"cannot read runtime certificate {path}: {e}") from e
    if not isinstance(payload, dict):
        raise RuntimeCompatibilityError(
            f"runtime certificate {path} is not a JSON object")
    if payload.get("schema") not in ("comfy-wxa8-runtime-cert/v1",
                                     "comfy-wxa8-runtime-cert/v2"):
        raise RuntimeCompatibilityError(
            f"unsupported runtime certificate schema in {path}: "
            f"{payload.get('schema')!r}")
    backend = payload.get("backend")
    if backend not in ("nvidia", "amd", "cpu"):
        raise RuntimeCompatibilityError(
            f"runtime certificate has unknown backend {backend!r}")
    formats = payload.get("formats")
    if not isinstance(formats, dict):
        raise RuntimeCompatibilityError(
            "runtime certificate has no formats object")
    known = set(MIXED_FORMATS)
    unknown = set(formats) - known
    if unknown:
        raise RuntimeCompatibilityError(
            f"runtime certificate lists unknown formats: {sorted(unknown)}")
    cap = payload.get("cuda_capability")
    if isinstance(cap, (list, tuple)) and len(cap) == 2:
        cap = (int(cap[0]), int(cap[1]))
    else:
        cap = None
    return RuntimeCertificate(
        backend=backend,
        gpu=payload.get("gpu"),
        cuda_capability=cap,
        rocm_arch=payload.get("rocm_arch"),
        formats={fmt: conf for fmt, conf in formats.items()
                 if isinstance(conf, dict)},
    )


def _check_runtime_certificate(cert: RuntimeCertificate,
                               runtime: RuntimeCapabilities,
                               required_formats: Sequence[str]) -> None:
    """The certificate must cover the target backend and every selected
    format must have actually loaded and executed on the certified machine."""
    if cert.backend != runtime.target:
        raise RuntimeCompatibilityError(
            f"runtime certificate is for backend {cert.backend!r} but the "
            f"target runtime is {runtime.target!r}; regenerate the "
            "certificate on the target inference machine")
    if (runtime.cuda_capability is not None
            and cert.cuda_capability is not None
            and cert.cuda_capability != runtime.cuda_capability):
        raise RuntimeCompatibilityError(
            f"runtime certificate compute capability "
            f"{cert.cuda_capability} does not match the local GPU "
            f"{runtime.cuda_capability}")
    for fmt in required_formats:
        conf = cert.formats.get(fmt)
        if not isinstance(conf, dict):
            raise RuntimeCompatibilityError(
                f"runtime certificate does not cover format {fmt!r}")
        if not conf.get("load") or not conf.get("forward"):
            raise RuntimeCompatibilityError(
                f"runtime certificate shows format {fmt!r} did not load and "
                "forward on the target machine")
    if FORMAT_W4A4 in required_formats:
        conf = cert.formats.get(FORMAT_W4A4, {})
        bits = (conf.get("effective_activation_bits")
                or conf.get("activation_bits"))
        if bits not in (4, 8):
            raise RuntimeCompatibilityError(
                "runtime certificate lacks the observed W4A4 activation "
                "bits")


def inspect_environment() -> EnvironmentInfo:
    info = EnvironmentInfo(
        python=sys.version.split()[0],
        torch_version=torch.__version__,
        torch_cuda=getattr(torch.version, "cuda", None),
        torch_hip=getattr(torch.version, "hip", None),
        cuda_available=torch.cuda.is_available(),
        cuda_device=None,
        cuda_capability=None,
        rocm_arch=None,
        safetensors_version=safetensors.__version__,
        numpy_version=np.__version__,
        platform=sys.platform,
        cpu_count=os.cpu_count() or 1,
    )
    if info.cuda_available:
        try:
            info.cuda_device = torch.cuda.get_device_name(0)
            info.cuda_capability = tuple(int(x) for x in torch.cuda.get_device_capability(0))
            props = torch.cuda.get_device_properties(0)
            info.rocm_arch = getattr(props, "gcnArchName", None)
        except Exception as e:
            log().debug("CUDA environment probe failed: %s", e)
    # Optional runtime compatibility probing is deliberately static.  Importing
    # either project here would execute third-party package initializers and
    # would violate the converter's standalone/runtime-isolation guarantee.
    try:
        dist = importlib.metadata.distribution("comfy-kitchen")
        info.has_comfy_kitchen = True
        info.comfy_kitchen_rev = dist.version
        wanted = {
            "AsymW4A8Int8Layout": "comfy_kitchen_has_w4a8_layout",
            "TensorCoreConvRotW4A4Layout": "comfy_kitchen_has_w4a4_layout",
            "TensorWiseINT8Layout": "comfy_kitchen_has_int8_layout",
        }
        seen = set()
        for rel in dist.files or ():
            rel_text = str(rel).replace("\\", "/")
            if not rel_text.endswith(".py") or "comfy_kitchen" not in rel_text:
                continue
            path = Path(dist.locate_file(rel))
            try:
                if path.stat().st_size > 4 * 1024 * 1024:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for marker, attr in wanted.items():
                if marker in text:
                    setattr(info, attr, True)
                    seen.add(marker)
            if len(seen) == len(wanted):
                break
    except importlib.metadata.PackageNotFoundError:
        pass
    except Exception as e:
        log().debug("comfy-kitchen static compatibility probe failed: %s", e)
    try:
        spec = importlib.util.find_spec("comfy")
        roots = list(spec.submodule_search_locations or ()) if spec is not None else []
        for root in roots:
            quant_ops = Path(root) / "quant_ops.py"
            if not quant_ops.is_file() or quant_ops.stat().st_size > 4 * 1024 * 1024:
                continue
            source = quant_ops.read_text(encoding="utf-8", errors="ignore")
            info.has_comfy_quant_ops = True
            # Only report formats proven present in the static source.  This is
            # not a runtime import or a claim that the whole installation works.
            for fmt in MIXED_FORMATS:
                if fmt in source:
                    info.comfyui_quant_algos.append(fmt)
            break
    except Exception as e:
        log().debug("ComfyUI static compatibility probe failed: %s", e)
    info.comfy_kitchen_has_w6a8_api = w6a8_api_available()
    return info


@dataclass(frozen=True)
class FormatRuntimeCapability:
    """One format's status on one target runtime.

    loadable/executable: True = known supported, False = known unsupported,
    None = UNKNOWN (the environment did not provide the information). Missing
    information is never translated into True.
    accelerated: None = not proven, True/False = expected accelerated /
                 expected fallback from static analysis.
    certified: True only after a real runtime probe executed the format
               on the target inference machine.
    """
    loadable: Optional[bool]
    executable: Optional[bool]
    accelerated: Optional[bool]
    certified: bool = False
    backend: str = "unknown"
    reason: str = ""

    @property
    def supported(self) -> bool:
        return self.loadable is True and self.executable is True

    def describe(self) -> str:
        if self.loadable is None or self.executable is None:
            return f"unknown: {self.reason}" if self.reason else "unknown"
        if not (self.loadable and self.executable):
            return f"unsupported: {self.reason}" if self.reason else "unsupported"
        if self.certified:
            if self.accelerated:
                return "runtime-certified accelerated"
            return "runtime-certified fallback"
        if self.accelerated is True:
            return "expected accelerated (not certified)"
        if self.accelerated is False:
            return "eager/fallback"
        return "supported; acceleration unknown"


@dataclass(frozen=True)
class RuntimeCapabilities:
    """What the target runtime can actually execute, per format, plus the
    hardware the static model was built for."""
    target: str
    w4a4: FormatRuntimeCapability
    w4a8: FormatRuntimeCapability
    int8: FormatRuntimeCapability
    gpu_name: Optional[str] = None
    cuda_capability: Optional[Tuple[int, int]] = None
    rocm_arch: Optional[str] = None
    torch_cuda: Optional[str] = None
    torch_hip: Optional[str] = None
    runtime_certified: bool = False
    w6a8: FormatRuntimeCapability = field(default_factory=lambda: FormatRuntimeCapability(
        None, None, None, False, "unverified", "W6A8 API/consumer capability not probed"))

    _ATTR = {FORMAT_W4A4: "w4a4", FORMAT_W4A8: "w4a8", FORMAT_INT8: "int8", FORMAT_W6A8: "w6a8"}

    def capability(self, fmt: str) -> FormatRuntimeCapability:
        return getattr(self, self._ATTR[fmt])

    def supports(self, fmt: str) -> bool:
        return self.capability(fmt).supported

    def describe(self, fmt: str) -> str:
        return self.capability(fmt).describe()


def rocm_matrix_core_supported(rocm_arch: Optional[str]) -> bool:
    """Whether a ROCm architecture has the matrix-core paths ComfyUI's
    automatic INT8/Triton acceleration relies on. gfx10 (RDNA1/2) is
    intentionally excluded; RDNA3/3.5/4 (gfx11/gfx12) and the CDNA gfx9
    parts are matrix-core capable."""
    if not rocm_arch:
        return False
    arch = rocm_arch.split(":")[0]
    if arch.startswith(("gfx11", "gfx12")):
        return True
    return arch in {
        "gfx908", "gfx90a", "gfx940", "gfx941", "gfx942", "gfx950",
    }


@dataclass(frozen=True)
class W4A4ExecutionMode:
    """Effective W4A4 execution semantics on a target runtime.

    activation_bits: the activation precision the runtime will actually use.
    path: the dispatch path name (eager-int4, cuda-native-int4, cuda-int8,
          hip-int4-request, unknown-conservative).
    certain: True when the mode is proven (eager, explicit int8 request,
             SM8x native INT4, or a runtime certificate). False when the
             dispatch depends on compiled kernels or shapes.
    reason: human-readable justification.
    """
    activation_bits: int
    path: str
    certain: bool
    reason: str


def resolve_w4a4_execution_mode(
    runtime: RuntimeCapabilities,
    requested_linear_dtype: str,
) -> W4A4ExecutionMode:
    """Effective W4A4 execution mode, mirroring the comfy-kitchen CUDA
    dispatcher (verified against 0.2.28):

      linear_dtype int8 -> INT8 activation branch, unconditionally.
      linear_dtype int4 -> native SM8x INT4 MMA when major == 8; Turing
      (SM 7.5) may select the compiled Turing INT4 path depending on shape
      and kernel availability; everything else falls back to INT8.
      eager (cpu)     -> always the int4 activation path.
    """
    if runtime.target == "cpu":
        return W4A4ExecutionMode(
            4, "eager-int4", True,
            "comfy-kitchen eager always executes the int4 activation path")
    if requested_linear_dtype == "int8":
        return W4A4ExecutionMode(
            8, f"{runtime.target}-int8", True,
            "linear_dtype explicitly requests INT8 execution")
    if runtime.target == "nvidia":
        sm = runtime.cuda_capability
        if sm is not None and sm[0] == 8:
            return W4A4ExecutionMode(
                4, "cuda-native-int4", True,
                "current comfy-kitchen native INT4 MMA supports SM8x")
        if sm == (7, 5):
            return W4A4ExecutionMode(
                8, "cuda-dispatch-uncertain", False,
                "Turing may select INT4 depending on compiled kernels and "
                "activation shape; conservative A8 simulation unless "
                "runtime-certified")
        return W4A4ExecutionMode(
            8, "cuda-int8-fallback", True,
            "comfy-kitchen routes non-SM8x INT4 requests through the INT8 "
            "fallback")
    if runtime.target == "amd":
        return W4A4ExecutionMode(
            4, "hip-int4-request", False,
            "must be runtime-certified on the target HIP backend")
    return W4A4ExecutionMode(
        8, "unknown-conservative", False,
        "unknown runtime; conservatively model A8")


def _legacy_runtime_capabilities_for(
        backend: str,
        env: Optional[EnvironmentInfo] = None) -> RuntimeCapabilities:
    """Capability matrix built from the target backend plus (when available)
    the actual hardware data collected by the environment probe.

    nvidia: fused CUDA kernels for W4A8 (ConvRot 256), INT8 tensor-core
    (non-ConvRot rowwise verified at any K), and W4A4 MMA (cgs 16/64/256).
    amd: HIP/triton kernels exist for all three, but "accelerated" is only
    expected on matrix-core-capable architectures (gfx11/gfx12 and CDNA
    gfx9); RDNA1/2 stay on fallback paths, matching ComfyUI's gating.
    cpu: eager implementations of all three; the eager W4A4 path always
    executes int4 activations regardless of linear_dtype.

    Nothing here is runtime-certified; certification comes from
    a real target-runtime probe supplied via --runtime-certificate.
    """
    gpu_name = env.cuda_device if env is not None else None
    cuda_cap = env.cuda_capability if env is not None else None
    rocm_arch = env.rocm_arch if env is not None else None
    torch_cuda = env.torch_cuda if env is not None else None
    torch_hip = env.torch_hip if env is not None else None

    def _fmt(loadable: bool, executable: bool, accelerated: Optional[bool],
             backend: str, reason: str = "") -> FormatRuntimeCapability:
        return FormatRuntimeCapability(loadable, executable, accelerated,
                                       False, backend, reason)

    if backend == "nvidia":
        have_cuda = env is not None and env.cuda_device is not None
        if not have_cuda:
            return RuntimeCapabilities(
                target="nvidia",
                w4a4=_fmt(None, None, None, "cuda",
                          reason="no CUDA device detected; runtime behavior "
                                 "unknown"),
                w4a8=_fmt(None, None, None, "cuda",
                          reason="no CUDA device detected; runtime behavior "
                                 "unknown"),
                int8=_fmt(None, None, None, "cuda",
                          reason="no CUDA device detected; runtime behavior "
                                 "unknown"),
                gpu_name=gpu_name, cuda_capability=cuda_cap,
                rocm_arch=rocm_arch, torch_cuda=torch_cuda, torch_hip=torch_hip)
        accelerated = cuda_cap >= (7, 5) if cuda_cap is not None else None
        return RuntimeCapabilities(
            target="nvidia",
            w4a4=_fmt(True, True, accelerated, "cuda"),
            w4a8=_fmt(True, True, accelerated, "cuda"),
            int8=_fmt(True, True, accelerated, "cuda"),
            gpu_name=gpu_name, cuda_capability=cuda_cap,
            rocm_arch=rocm_arch, torch_cuda=torch_cuda, torch_hip=torch_hip)
    if backend == "amd":
        have_hip = env is not None and env.cuda_device is not None \
            and getattr(env, "torch_hip", None) is not None
        if not have_hip:
            return RuntimeCapabilities(
                target="amd",
                w4a4=_fmt(None, None, None, "hip",
                          reason="no HIP device detected; runtime behavior "
                                 "unknown"),
                w4a8=_fmt(None, None, None, "triton/hip",
                          reason="no HIP device detected; runtime behavior "
                                 "unknown"),
                int8=_fmt(None, None, None, "triton/hip",
                          reason="no HIP device detected; runtime behavior "
                                 "unknown"),
                gpu_name=gpu_name, cuda_capability=cuda_cap,
                rocm_arch=rocm_arch, torch_cuda=torch_cuda, torch_hip=torch_hip)
        matrix_core = rocm_matrix_core_supported(rocm_arch)
        return RuntimeCapabilities(
            target="amd",
            w4a4=_fmt(True, True, matrix_core, "hip",
                      reason=("no matrix-core path on this ROCm arch"
                              if not matrix_core else "")),
            w4a8=_fmt(True, True, matrix_core, "triton/hip",
                      reason=("no matrix-core path on this ROCm arch"
                              if not matrix_core else "")),
            int8=_fmt(True, True, matrix_core, "triton/hip",
                      reason=("no matrix-core path on this ROCm arch"
                              if not matrix_core else "")),
            gpu_name=gpu_name, cuda_capability=cuda_cap,
            rocm_arch=rocm_arch, torch_cuda=torch_cuda, torch_hip=torch_hip)
    return RuntimeCapabilities(
        target="cpu",
        w4a4=_fmt(True, True, False, "eager",
                  reason="eager always executes the int4 activation path"),
        w4a8=_fmt(True, True, False, "eager"),
        int8=_fmt(True, True, False, "eager"),
        gpu_name=gpu_name, cuda_capability=cuda_cap,
        rocm_arch=rocm_arch, torch_cuda=torch_cuda, torch_hip=torch_hip)


def runtime_capabilities_for(backend, env=None):
    caps = _legacy_runtime_capabilities_for(backend, env)
    api = w6a8_api_available()
    consumer = env is None or not env.has_comfy_quant_ops or FORMAT_W6A8 in env.comfyui_quant_algos
    available = api and consumer
    if backend == "cpu":
        accelerated = False
    elif env is None or env.cuda_device is None:
        accelerated = None
    elif backend == "nvidia":
        accelerated = env.cuda_capability >= (8, 0) if env.cuda_capability else None
    else:
        accelerated = rocm_matrix_core_supported(env.rocm_arch)
    known = available if backend == "cpu" or env is not None and env.cuda_device is not None else (None if available else False)
    cap = FormatRuntimeCapability(known, known, accelerated if available else False,
        False, "eager" if backend == "cpu" else "cuda" if backend == "nvidia" else "triton/hip",
        "" if available else "requires Kitchen >= 0.2.36 six-bit API and a W6A8 consumer")
    return dataclasses.replace(caps, w6a8=cap)


def native_gpu_warning(fmt: str, target_runtime: str,
                       capability: Optional[Tuple[int, int]]) -> Optional[str]:
    minimum = ((8, 9) if fmt in ('fp8-e4m3', 'fp8-e5m2') else
               (10, 0) if fmt in ('nvfp4', 'mxfp8') else None)
    if minimum is None or target_runtime not in ('auto', 'nvidia') or capability is None:
        return None
    if capability >= minimum:
        return None
    return (f'{fmt}: local NVIDIA SM {capability[0]}.{capability[1]} is below '
            f'SM {minimum[0]}.{minimum[1]} for accelerated ComfyUI inference; '
            'conversion remains available, but inference may dequantize weights')


def warn_native_gpu(args: Any) -> None:
    if not torch.cuda.is_available() or getattr(torch.version, 'hip', None) is not None:
        return
    warning = native_gpu_warning(args.format, args.target_runtime,
                                 torch.cuda.get_device_capability())
    if warning:
        log().warning(warning)
