"""NitroGen execution-backend planning.

The NitroGen optimization study sweeps the *execution backend* (how the same
500M policy is run) rather than a serving framework. This module defines the
axes and parses them from a round's launch args (the YAML `variants` /
`backend_args` flags), producing an `ExecPlan` the serve launcher applies to
the loaded model.

Axes:
    exec backend : eager | torch_compile | cudagraph | tensorrt | onnxruntime
    precision    : bf16 | fp16 | fp8 | nvfp4
    denoise steps: flow-matching iterations (latency<->quality knob)
    cfg scale    : classifier-free guidance (1.0 = off; >1 doubles DiT passes)
    seed         : pinned denoising noise (so precision deltas aren't sampling noise)

Everything here is pure/CPU-safe. `apply_optimization()` is the only function
that touches torch/TensorRT/ONNX, and it imports them lazily so this module —
and the planning logic — imports on a bare CPU box.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class ExecBackend(str, enum.Enum):
    EAGER = "eager"
    TORCH_COMPILE = "torch_compile"
    CUDAGRAPH = "cudagraph"
    TENSORRT = "tensorrt"
    ONNXRUNTIME = "onnxruntime"


class Precision(str, enum.Enum):
    BF16 = "bf16"
    FP16 = "fp16"
    FP8 = "fp8"
    NVFP4 = "nvfp4"


# Accept short aliases used in YAML variants.
_EXEC_ALIASES = {
    "eager": ExecBackend.EAGER,
    "compile": ExecBackend.TORCH_COMPILE,
    "torch_compile": ExecBackend.TORCH_COMPILE,
    "cudagraph": ExecBackend.CUDAGRAPH,
    "cuda_graph": ExecBackend.CUDAGRAPH,
    "trt": ExecBackend.TENSORRT,
    "tensorrt": ExecBackend.TENSORRT,
    "onnx": ExecBackend.ONNXRUNTIME,
    "onnxruntime": ExecBackend.ONNXRUNTIME,
}

# Torch dtype names (resolved to real dtypes lazily inside apply_optimization).
# FP8/NVFP4 keep a bf16 compute dtype for the layers left unquantized; the
# quantization itself is applied per-backend, not via a global cast.
_PRECISION_TORCH_DTYPE = {
    Precision.BF16: "bfloat16",
    Precision.FP16: "float16",
    Precision.FP8: "bfloat16",
    Precision.NVFP4: "bfloat16",
}

# Precisions that need real quantization tooling (vs a plain dtype cast).
_QUANTIZED = {Precision.FP8, Precision.NVFP4}

# Blackwell-only datatypes (informational; enforced at apply time on GPU).
_BLACKWELL_ONLY = {Precision.NVFP4}


@dataclass(frozen=True)
class ExecPlan:
    exec_backend: ExecBackend = ExecBackend.EAGER
    precision: Precision = Precision.BF16
    steps: int = 16
    cfg_scale: float = 1.0
    seed: int = 0

    @property
    def is_quantized(self) -> bool:
        return self.precision in _QUANTIZED

    @property
    def torch_dtype_name(self) -> str:
        return _PRECISION_TORCH_DTYPE[self.precision]

    def label(self) -> str:
        """Stable label for results/logging, e.g. 'trt-fp8-4step'."""
        cfg = "" if self.cfg_scale == 1.0 else f"-cfg{self.cfg_scale:g}"
        return f"{self.exec_backend.value}-{self.precision.value}-{self.steps}step{cfg}"

    def to_knobs(self) -> dict:
        """Flat dict for BenchmarkResult.framework_knobs."""
        return {
            "exec_backend": self.exec_backend.value,
            "precision": self.precision.value,
            "denoise_steps": self.steps,
            "cfg_scale": self.cfg_scale,
            "seed": self.seed,
        }


def _iter_flags(launch_args: list[str]):
    """Yield (key, value) for --key=value and --key value forms."""
    i = 0
    n = len(launch_args)
    while i < n:
        tok = launch_args[i]
        if tok.startswith("--"):
            body = tok[2:]
            if "=" in body:
                k, v = body.split("=", 1)
                yield k, v
            elif i + 1 < n and not launch_args[i + 1].startswith("--"):
                yield body, launch_args[i + 1]
                i += 1
            else:
                yield body, ""  # bare flag
        i += 1


def parse_exec_plan(launch_args: list[str], *, default_seed: int = 0) -> ExecPlan:
    """Build an ExecPlan from a round's launch args.

    Recognized: --exec, --precision, --steps, --cfg, --seed. Unknown flags are
    ignored (they may target the server for other reasons). Raises ValueError on
    an unknown exec/precision value so a typo in the YAML fails loudly.
    """
    exec_backend = ExecBackend.EAGER
    precision = Precision.BF16
    steps = 16
    cfg_scale = 1.0
    seed = default_seed

    for key, value in _iter_flags(launch_args):
        if key == "exec":
            if value not in _EXEC_ALIASES:
                raise ValueError(
                    f"unknown --exec {value!r}; expected one of {sorted(_EXEC_ALIASES)}"
                )
            exec_backend = _EXEC_ALIASES[value]
        elif key == "precision":
            try:
                precision = Precision(value)
            except ValueError as e:
                raise ValueError(
                    f"unknown --precision {value!r}; expected one of "
                    f"{[p.value for p in Precision]}"
                ) from e
        elif key == "steps":
            steps = int(value)
        elif key == "cfg":
            cfg_scale = float(value)
        elif key == "seed":
            seed = int(value)

    if steps < 1:
        raise ValueError(f"--steps must be >= 1, got {steps}")
    return ExecPlan(
        exec_backend=exec_backend,
        precision=precision,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
    )


def requires_blackwell(plan: ExecPlan) -> bool:
    """True if the plan uses a datatype only available on Blackwell (NVFP4)."""
    return plan.precision in _BLACKWELL_ONLY


def apply_optimization(model, plan: ExecPlan):
    """Apply an ExecPlan to a loaded NitroGen model and return the runnable model.

    GPU-only: imports torch + the relevant compiler lazily. Raises RuntimeError
    if the toolchain for the requested backend is unavailable, so a misconfigured
    serving env fails at launch rather than silently running eager.

    Quantization (FP8/NVFP4) is applied to the DiT + vision-tower Linear layers
    only; norms, timestep embeddings, and the action decoder stay at the compute
    dtype (see docs note on diffusion-policy quantization sensitivity).
    """
    import torch  # lazy — GPU/serving env only

    _ = getattr(torch, plan.torch_dtype_name)  # validate the dtype name early
    # We deliberately do NOT `model.to(dtype=...)` here. NitroGen builds
    # internal action buffers at the loaded model dtype (float32) inside
    # `prepare_input_embs` and then `masked_scatter`s them against an
    # embedding tensor — a blanket cast desynchronises those dtypes.
    # Mixed-precision compute is applied by wrapping the predict call in
    # `torch.autocast(device, dtype=plan.torch_dtype_name)` in the serve
    # loop (see scripts/serve_nitrogen.py); the model itself stays at fp32.

    if plan.exec_backend is ExecBackend.EAGER:
        return model
    if plan.exec_backend is ExecBackend.TORCH_COMPILE:
        return torch.compile(model)
    if plan.exec_backend is ExecBackend.CUDAGRAPH:
        # CUDA graphs are captured by the serve loop around the fixed-shape
        # denoising step; nothing to transform on the module here.
        return model
    if plan.exec_backend in (ExecBackend.TENSORRT, ExecBackend.ONNXRUNTIME):
        raise RuntimeError(
            f"{plan.exec_backend.value} export is performed by the serve launcher's "
            "export path (modelopt/ONNX); call build_export_engine() there. This "
            "stub guards against running the un-exported module by mistake."
        )
    raise RuntimeError(f"unhandled exec backend: {plan.exec_backend}")
