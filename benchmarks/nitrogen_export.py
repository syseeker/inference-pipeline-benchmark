"""Export a calibrated NitroGen model to ONNX / TensorRT for FP8/NVFP4 serving.

Calibration happens in `benchmarks/nitrogen_quant.py` (modelopt PTQ — adds
fake-quant nodes + learns amax). This module **persists** that to a
hardware-accelerable artifact and provides a tiny runtime wrapper the
serve loop can swap in for the PyTorch DiT step.

We deliberately export the **DiT step** only, not the full denoise loop:
- The loop in `NitroGen.get_action` is data-dependent (its iteration
  count is `--steps`, which we sweep over). Unrolling at export time
  would freeze the step count into the engine.
- The vision encoder is called once per request; the DiT runs N times.
  Quantizing the DiT is where the FP8/NVFP4 throughput win is.
- The action-decoder head is excluded from quant (see nitrogen_quant.py)
  so its float32 forward stays in PyTorch — single small matmul.

Cache key: `(precision, denoise_steps)`. A new (precision, steps) combo
re-exports and re-compiles; the cache survives across `bench sweep`
invocations.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

CACHE_ROOT_DEFAULT = Path("/ephemeral/cache/nitrogen-engines")


def _cache_key(precision: str, steps: int) -> str:
    """Stable short key for the (precision, steps) pair."""
    raw = f"{precision}-steps{steps}"
    short = hashlib.sha1(raw.encode()).hexdigest()[:8]
    return f"{raw}-{short}"


def cache_paths(precision: str, steps: int, cache_root: Path | None = None) -> dict[str, Path]:
    """Resolve the on-disk paths for the (precision, steps) engine bundle.

    Returns a dict with `dir`, `onnx`, `plan`, `meta` keys. None of these
    are required to exist — caller can probe via `.exists()`.
    """
    root = (cache_root or CACHE_ROOT_DEFAULT) / _cache_key(precision, steps)
    return {
        "dir":  root,
        "onnx": root / "ng_dit.onnx",
        "plan": root / "ng_dit.plan",
        "meta": root / "meta.json",
    }


def _dummy_dit_inputs(model: Any) -> dict[str, Any]:
    """Build representative inputs matching the real DiT.forward signature.

    Shape source: NitroGen/flow_matching_transformer/modules.py line 251.
    DiT.forward(hidden_states, encoder_hidden_states, timestep,
                encoder_attention_mask=None, return_all_hidden_states=False).

    - `hidden_states`        : (B, T, D)  — action-token embeddings
                                 T ≈ action_horizon + small token overhead;
                                 D = hidden_size from ckpt_config
    - `encoder_hidden_states`: (B, S, D)  — vl tokens (vision + game-id)
                                 S = num_visual_tokens_per_frame (256)
    - `timestep`             : (B,) long  — discretised diffusion timestep,
                                 in [0, num_timestep_buckets)

    The exported engine is shape-static; if any of action_horizon /
    num_visual_tokens / hidden_size change between training runs, the
    cache key (`precision-stepsN-<sha>`) misses and we re-export.
    """
    import torch

    nitrogen_cfg = model.config        # NitroGen_Config
    dit_cfg = model.model.config        # DiTConfig (the submodule we're exporting)

    batch = 1
    hidden = int(dit_cfg.output_dim)    # DiT's working dim
    horizon = int(nitrogen_cfg.action_horizon)
    n_vis_tok = 256                     # tokenizer_cfg.num_visual_tokens_per_frame
    num_buckets = int(nitrogen_cfg.num_timestep_buckets)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    hidden_states = torch.randn(batch, horizon, hidden, device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(batch, n_vis_tok, hidden, device=device, dtype=dtype)
    timestep = torch.randint(0, num_buckets, (batch,), device=device, dtype=torch.long)
    return {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": timestep,
    }


class _DitExportWrapper(__import__("torch").nn.Module):
    """Wraps `DiT.forward` to drop the kwargs / tuple-return that confuse ONNX export.

    The real signature accepts an attention mask + a `return_all_hidden_states`
    flag whose True path returns a tuple — ONNX export hates that. We pin the
    flag to False and treat the mask as always-None.
    """

    def __init__(self, dit) -> None:  # type: ignore[no-untyped-def]
        super().__init__()
        self.dit = dit

    def forward(self, hidden_states, encoder_hidden_states, timestep):  # type: ignore[no-untyped-def]
        return self.dit(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            encoder_attention_mask=None,
            return_all_hidden_states=False,
        )


def export_dit_to_onnx(
    model: Any,
    *,
    precision: str,
    steps: int,
    cache_root: Path | None = None,
    opset: int = 20,
) -> Path:
    """Trace + serialize the DiT step to ONNX with QDQ nodes preserved.

    Requires the model to have already been calibrated via `quantize_for_serving`.
    Returns the path to the written `.onnx`.
    """
    import torch

    paths = cache_paths(precision, steps, cache_root)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    dit = _resolve_dit_submodule(model)
    inputs = _dummy_dit_inputs(model)
    wrapper = _DitExportWrapper(dit).eval()

    # Why `dynamo=False`: modelopt's tensor-quantizer modules carry
    # internal state (`lifted_tensor_0`) that the new (dynamo-based)
    # torch.onnx exporter rejects as a "fake tensor in the constants
    # list." The legacy JIT-tracing path handles QDQ nodes correctly —
    # confirmed against modelopt 0.44 + torch 2.12.
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (inputs["hidden_states"], inputs["encoder_hidden_states"], inputs["timestep"]),
            str(paths["onnx"]),
            opset_version=opset,
            input_names=["hidden_states", "encoder_hidden_states", "timestep"],
            output_names=["pred"],
            dynamic_axes={
                # batch dim only; sequence and hidden are static.
                "hidden_states":          {0: "batch"},
                "encoder_hidden_states":  {0: "batch"},
                "timestep":               {0: "batch"},
                "pred":                   {0: "batch"},
            },
            dynamo=False,
        )

    paths["meta"].write_text(
        json.dumps(
            {
                "precision": precision,
                "steps":     steps,
                "opset":     opset,
                "input_shapes": {k: list(v.shape) for k, v in inputs.items()},
            },
            indent=2,
        )
    )
    return paths["onnx"]


def compile_onnx_to_trt(
    onnx_path: Path,
    *,
    precision: str,
    cache_root: Path | None = None,
) -> Path:
    """Compile the ONNX to a TRT engine via `trtexec`. Returns the `.plan` path.

    `trtexec` ships with TensorRT; the harness's GPU venv (.venv-nitrogen
    here, .venv-trtllm on serving boxes) is expected to have it on PATH.
    Failure surfaces as a RuntimeError with the trtexec stderr.
    """
    if shutil.which("trtexec") is None:
        raise RuntimeError(
            "trtexec not on PATH — install TensorRT in this venv "
            "(pip install tensorrt --extra-index-url https://pypi.nvidia.com) "
            "or compile the engine offline and drop the .plan into the cache dir."
        )

    paths = cache_paths(precision, _meta_steps(onnx_path), cache_root)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    flag = {"fp8": "--fp8", "nvfp4": "--fp4"}.get(precision)
    if flag is None:
        raise ValueError(f"unsupported TRT precision: {precision!r}")

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={paths['plan']}",
        flag,
        # Allow bf16 fallback for layers the FP8/NVFP4 path doesn't cover
        # (e.g. our excluded action-decoder etc.).
        "--bf16",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    if res.returncode != 0:
        raise RuntimeError(
            f"trtexec failed for {onnx_path}:\n{res.stderr.strip()[:800]}"
        )
    return paths["plan"]


def _meta_steps(onnx_path: Path) -> int:
    """Steps recovered from the sibling meta.json (so we cache under the
    same key the export step used)."""
    meta = onnx_path.parent / "meta.json"
    return int(json.loads(meta.read_text())["steps"]) if meta.exists() else 0


def _resolve_dit_submodule(model: Any) -> Any:
    """Return the DiT submodule inside the NitroGen wrapper.

    Centralised so the export and the runtime wrapper agree on which
    Python object they're targeting. NitroGen stores the DiT under
    `model.model` (see flow_matching_transformer/nitrogen.py:193).
    """
    if hasattr(model, "model"):
        return model.model
    raise RuntimeError(
        "could not locate DiT submodule on the NitroGen instance — "
        "checkpoint layout may have changed; check "
        "flow_matching_transformer/nitrogen.py for the new attribute name."
    )


# --------------------------------------------------------------------------- #
# Runtime wrappers — swapped into session.model.model by serve_nitrogen.py    #
# --------------------------------------------------------------------------- #


class OrtDitWrapper:
    """`session.model.model` replacement that runs the DiT step under
    ONNX Runtime instead of PyTorch.

    Only `__call__` (DiT step forward) is replaced — the rest of NitroGen
    (vision encoder, denoise loop control, action decoder) stays in PyTorch.

    **PR #5 status — known limitation.** modelopt 0.44 emits TRT-extension
    ops (`trt:TRT_FP8QuantizeLinear`) in the QDQ'd ONNX graph. ORT's
    schema validator rejects these BEFORE the TRT-EP gets a chance to
    parse them — even with `trt_fp8_enable=True`. The fix is direct
    TensorRT runtime via `tensorrt.Builder` + `execute_async_v3`, which
    is tracked as PR #5.1. The wrapper below stays for the BF16-export
    path (which uses standard ONNX ops) and as scaffolding for the
    PR #5.1 runtime rewrite.
    """

    def __init__(self, onnx_path: Path, *, provider_preference: list[str] | None = None) -> None:
        # Pre-load tensorrt's shared libs so ORT's TensorrtExecutionProvider
        # can dlopen libnvinfer.so.10. The `tensorrt` Python package ships
        # them under `tensorrt_libs/` but doesn't put that on LD_LIBRARY_PATH
        # by default — `import tensorrt` works (it uses ctypes.CDLL) but ORT's
        # dlopen call fails. We force-load `tensorrt` first; that registers
        # the libs into the process and ORT picks them up.
        try:
            import tensorrt  # noqa: F401 - side-effect import
        except ImportError:
            pass

        import onnxruntime as ort  # lazy — only when --exec=onnx

        providers = provider_preference or [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        self._session = ort.InferenceSession(str(onnx_path), providers=providers)
        self._io_binding = self._session.io_binding()

    def __call__(self, hidden_states, encoder_hidden_states, timestep, **_):  # type: ignore[no-untyped-def]
        """Match the DiT's keyword call surface so we can drop-in for `model.model`.

        NitroGen calls the DiT as
            self.model(hidden_states=..., encoder_hidden_states=...,
                       encoder_attention_mask=..., timestep=...,
                       return_all_hidden_states=True/False)
        We swallow the kwargs we don't forward (mask + flag) — the engine
        was traced with the flag pinned False and the mask treated as None.
        """
        import torch  # lazy

        out = self._session.run(
            ["pred"],
            {
                "hidden_states":         hidden_states.detach().cpu().numpy(),
                "encoder_hidden_states": encoder_hidden_states.detach().cpu().numpy(),
                "timestep":              timestep.detach().cpu().numpy(),
            },
        )[0]
        return torch.from_numpy(out).to(hidden_states.device, dtype=hidden_states.dtype)


class TrtDitWrapper(OrtDitWrapper):
    """ORT session pinned to TensorrtExecutionProvider only — forces TRT-EP
    compilation under the hood.

    Distinction from OrtDitWrapper: the parent class lets ORT walk its EP
    preference order (TRT → CUDA → CPU). This subclass refuses anything
    but TRT-EP, so a TRT compile failure surfaces as a load error instead
    of silently falling back to CUDA. Same forward signature.

    Direct TensorRT plan binding (via `trt.Runtime` + `execute_async_v3`)
    has a lower per-call overhead than ORT-with-TRT-EP, but requires
    template-ised binding management that's out of scope for PR #5. The
    `compile_onnx_to_trt` standalone path stays in this module for users
    who want to drop in a pre-compiled `.plan`; it's just not on the
    serve-loop default path.
    """

    def __init__(self, onnx_path: Path) -> None:
        super().__init__(onnx_path, provider_preference=["TensorrtExecutionProvider"])
