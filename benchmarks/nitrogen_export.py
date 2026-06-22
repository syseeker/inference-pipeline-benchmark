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


def _gpu_id_short() -> str:
    """Short device tag for cache keying — TRT plans are GPU-specific.

    A plan compiled on PRO 6000 (SM_120) is not loadable on H200 (SM_90),
    so we key the on-disk cache by GPU name to keep parallel installs
    from clobbering each other on shared-storage boxes.
    """
    import torch
    try:
        return torch.cuda.get_device_name(0).replace(" ", "_")
    except Exception:
        return "unknown-gpu"


def _build_trt_plan_from_onnx(onnx_path: Path, *, precision: str, plan_path: Path) -> Path:
    """Compile an `ng_dit.onnx` (with QDQ nodes) to a `.plan` for this GPU.

    Direct TRT API — no `trtexec` subprocess (which isn't bundled in the
    pip `tensorrt` wheel). Reads the ONNX bytes, runs the OnnxParser,
    sets the precision flag (FP8 or NVFP4 from BuilderFlag), serializes
    to the plan path. Idempotent: caller checks plan existence first.
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    # NVFP4's ONNX op is shipped as a TRT plugin (not a built-in op); the
    # plugin registry has to be initialised before OnnxParser sees the
    # graph, otherwise parsing fails with "Plugin not found". Harmless
    # no-op for FP8.
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    # STRONGLY_TYPED keeps the parser's declared precision sticky —
    # important for QDQ'd graphs where TRT could otherwise promote to bf16.
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    parser = trt.OnnxParser(network, logger)

    onnx_bytes = onnx_path.read_bytes()
    if not parser.parse(onnx_bytes, str(onnx_path)):
        errs = [parser.get_error(i).desc() for i in range(parser.num_errors)]
        raise RuntimeError(f"OnnxParser failed for {onnx_path}:\n  " + "\n  ".join(errs))

    cfg = builder.create_builder_config()
    # 8 GB workspace is plenty for the 466 MB DiT without starving the box.
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    # NOTE: With STRONGLY_TYPED networks (set above), TRT refuses
    # BuilderFlag.FP8 / FP4 / BF16 — precision is declared per-tensor in
    # the QDQ'd ONNX itself. The parser's trt:TRT_FP8QuantizeLinear ops
    # are how FP8 gets dispatched. Validate the requested precision is
    # one we support so a typo in the YAML fails loud.
    if precision not in ("fp8", "nvfp4"):
        raise ValueError(f"_build_trt_plan_from_onnx: unsupported precision {precision!r}")

    # The exported ONNX has batch as a dynamic axis (so the artifact stays
    # reusable across batch sizes). TRT requires an optimization profile
    # for dynamic shapes; we pin batch=1 because that's the only shape the
    # serve loop ever invokes (PR #1 left multi-batch as a TODO).
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        t = network.get_input(i)
        shape = list(t.shape)
        # Replace -1 (dynamic) with 1; static dims stay as-is.
        concrete = tuple(1 if d == -1 else d for d in shape)
        profile.set_shape(t.name, min=concrete, opt=concrete, max=concrete)
    cfg.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, cfg)
    if serialized is None:
        raise RuntimeError(f"build_serialized_network returned None for {onnx_path}")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_bytes(bytes(serialized))
    return plan_path


def _trt_dtype_to_torch(dt):  # type: ignore[no-untyped-def]
    """Map TRT 10's DataType enum to torch dtypes for buffer allocation."""
    import tensorrt as trt
    import torch
    return {
        trt.DataType.FLOAT:  torch.float32,
        trt.DataType.HALF:   torch.float16,
        trt.DataType.BF16:   torch.bfloat16,
        trt.DataType.INT32:  torch.int32,
        trt.DataType.INT64:  torch.int64,
        # FP8 / NVFP4 inputs are wrapped in bf16 at the API boundary —
        # the Q/DQ nodes inside the engine handle the precision drop.
    }.get(dt, torch.float32)


class TrtDitWrapper(__import__("torch").nn.Module):
    """`session.model.model` replacement that runs the DiT step on a
    pre-compiled TensorRT plan.

    Inherits from `nn.Module` so we can drop it into NitroGen's PyTorch
    container via `session.model.model = TrtDitWrapper(...)` —
    `nn.Module.__setattr__` enforces children-must-be-Module.

    Owns:
      - one ICudaEngine compiled from the calibrated ONNX once at startup
      - one IExecutionContext reused across the request's denoise calls
      - a pre-allocated CUDA buffer per engine I/O tensor (alloc once in
        __init__, set_tensor_address once, just memcpy on each call)

    Construction: takes the ONNX path the calibration step produced; if
    a per-GPU `.plan` is already cached next to it, deserialize that.
    Otherwise compile + cache. The customer pays the 30–60 s build cost
    once per (precision, GPU) tuple.

    Call surface matches `DiT.forward`'s inference path exactly so we
    can `session.model.model = TrtDitWrapper(...)` and the upstream
    caller in NitroGen.get_action keeps working unchanged.
    """

    def __init__(self, onnx_path: Path, *, precision: str) -> None:
        import tensorrt as trt
        import torch

        super().__init__()  # nn.Module bookkeeping — required before
                            # any other attribute assignment, otherwise
                            # __setattr__ raises on _parameters / _buffers.

        if not torch.cuda.is_available():  # pragma: no cover - GPU only
            raise RuntimeError("TrtDitWrapper requires CUDA")

        plan_path = onnx_path.parent / f"ng_dit.{_gpu_id_short()}.plan"
        if not plan_path.exists():
            print(f"[ng-trt] compiling {onnx_path.name} -> {plan_path.name} (one-time)...")
            _build_trt_plan_from_onnx(onnx_path, precision=precision, plan_path=plan_path)
            print(f"[ng-trt] compiled: {plan_path}  ({plan_path.stat().st_size:,} B)")
        else:
            print(f"[ng-trt] reusing cached plan: {plan_path}")

        logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(logger)
        with open(plan_path, "rb") as f:
            self._engine = self._runtime.deserialize_cuda_engine(f.read())
        self._context = self._engine.create_execution_context()
        self._device = torch.device("cuda")
        self._stream = torch.cuda.Stream()

        # Discover the engine's I/O tensors + pre-allocate static buffers.
        # The export pinned batch=1; we keep that invariant.
        self._inputs: dict[str, "torch.Tensor"] = {}
        self._outputs: dict[str, "torch.Tensor"] = {}
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = tuple(self._engine.get_tensor_shape(name))
            dtype = _trt_dtype_to_torch(self._engine.get_tensor_dtype(name))
            mode = self._engine.get_tensor_mode(name)
            # Replace any -1 (dynamic) dim with 1 — we promised batch=1.
            shape = tuple(1 if s == -1 else s for s in shape)
            buf = torch.empty(shape, dtype=dtype, device=self._device)
            if mode == trt.TensorIOMode.INPUT:
                self._inputs[name] = buf
            else:
                self._outputs[name] = buf
            self._context.set_tensor_address(name, buf.data_ptr())

    def _copy_in(self, **named_tensors) -> None:  # type: ignore[no-untyped-def]
        """Memcpy each named input into its pre-bound CUDA buffer."""
        for name, src in named_tensors.items():
            if src is None:
                continue
            dst = self._inputs.get(name)
            if dst is None:
                raise KeyError(f"TrtDitWrapper has no input named {name!r}; "
                               f"known: {list(self._inputs)}")
            dst.copy_(src.to(device=self._device, dtype=dst.dtype, non_blocking=True))

    def forward(
        self,
        hidden_states,                       # (B, T, D)
        encoder_hidden_states,               # (B, S, D)
        timestep,                            # (B,) long
        encoder_attention_mask=None,         # ignored — engine traced with None
        return_all_hidden_states=False,      # ignored — engine traced with False
    ):  # type: ignore[no-untyped-def]
        """Override `nn.Module.forward` so `__call__(...)` from
        NitroGen.get_action routes through us. Keep the kwargs that
        NitroGen.get_action passes (encoder_attention_mask,
        return_all_hidden_states) as accept-and-ignore — they were
        baked into the engine trace at export time."""
        self._copy_in(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
        )
        if not self._context.execute_async_v3(self._stream.cuda_stream):
            raise RuntimeError("TrtDitWrapper.execute_async_v3 returned False")
        self._stream.synchronize()
        out = self._outputs["pred"]
        # Caller (NitroGen's get_action) expects float-dtype output —
        # action_decoder downstream is fp32. Match the input dtype.
        return out.to(dtype=hidden_states.dtype)
