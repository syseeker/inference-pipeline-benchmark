#!/usr/bin/env python3
"""Non-interactive NitroGen ZMQ server with execution-backend selection.

NitroGen's stock `scripts/serve.py` picks the game via `input()` at startup and
always runs eager bf16 — neither works for an automated benchmark sweep. This
launcher:

  * selects the game from `--game` (no prompt),
  * applies an ExecPlan (`--exec/--precision/--steps/--cfg/--seed`) to the model,
  * handles `predict` requests carrying `game_id`/`seed` (sent by NitrogenReasoner),

while speaking the same ZMQ/pickle protocol the reasoner expects.

Run on the GPU/serving instance (needs NitroGen + torch + CUDA):
    python scripts/serve_nitrogen.py /path/to/ng.pt \
        --port 5555 --game celeste --exec trt --precision fp8 --steps 4 --seed 0

Heavy imports (zmq, torch, NitroGen) are deferred into `serve()` so this module
imports and `--help`/arg-parses on a bare CPU box (used by the unit tests).
"""

from __future__ import annotations

import argparse

from benchmarks.nitrogen_exec import ExecPlan, parse_exec_plan


def build_plan(args: argparse.Namespace) -> ExecPlan:
    """Pure: assemble the ExecPlan from parsed CLI args (CPU-testable)."""
    flags = [
        f"--exec={args.exec_backend}",
        f"--precision={args.precision}",
        f"--steps={args.steps}",
        f"--cfg={args.cfg}",
        f"--seed={args.seed}",
    ]
    return parse_exec_plan(flags)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("ckpt", help="Path to the NitroGen checkpoint (.pt).")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--game", default=None, help="Game label (no interactive prompt).")
    p.add_argument("--exec", dest="exec_backend", default="eager",
                   help="eager | compile | cudagraph | trt | onnx")
    p.add_argument("--precision", default="bf16", help="bf16 | fp16 | fp8 | nvfp4")
    p.add_argument("--steps", type=int, default=16, help="Denoise steps (flow-matching iters).")
    p.add_argument("--cfg", type=float, default=1.0, help="Classifier-free guidance scale.")
    p.add_argument("--seed", type=int, default=0, help="Pinned denoising seed.")
    return p


def serve(args: argparse.Namespace) -> int:  # pragma: no cover - GPU/serving only
    import pickle

    import torch
    import zmq
    from nitrogen.inference_session import InferenceSession

    from benchmarks.nitrogen_exec import apply_optimization

    plan = build_plan(args)
    print(f"ExecPlan: {plan.label()}")

    session = InferenceSession.from_ckpt(
        args.ckpt, cfg_scale=plan.cfg_scale, context_length=1
    )
    # Non-interactive game selection (bypass the stock input() prompt).
    session.selected_game = args.game

    # For FP8 / NVFP4 + ONNX or TRT exec paths: calibrate, export, and swap
    # the DiT submodule for a runtime wrapper that runs the QDQ'd graph on
    # the accelerator. Done BEFORE apply_optimization so that fn sees the
    # post-quant module shape it expects.
    from benchmarks.nitrogen_exec import ExecBackend, Precision

    if (
        plan.precision in (Precision.FP8, Precision.NVFP4)
        and plan.exec_backend in (ExecBackend.ONNXRUNTIME, ExecBackend.TENSORRT)
    ):
        # FP8 / NVFP4 serving via direct TensorRT runtime (PR #5.1):
        #   1. Resolve the calibrated ONNX — download from HF via
        #      ensure_artifact (PR #5.0.5), or fall back to local
        #      calibration + export via PR #5's path.
        #   2. Compile-or-load a per-GPU TRT plan from that ONNX.
        #   3. Swap session.model.model for TrtDitWrapper so the upstream
        #      NitroGen.get_action denoise loop transparently runs on TRT.
        from pathlib import Path

        from benchmarks.nitrogen_artifacts import ensure_artifact
        from benchmarks.nitrogen_export import (
            TrtDitWrapper,
            cache_paths,
            export_dit_to_onnx,
        )
        from benchmarks.nitrogen_quant import (
            load_calib_images_from_scenarios,
            quantize_for_serving,
        )

        paths = cache_paths(plan.precision.value, plan.steps)
        onnx_path = None

        try:
            onnx_path = ensure_artifact(plan.precision.value, plan.steps)
            print(f"[ng-quant] using pre-built artifact: {onnx_path}")
        except FileNotFoundError as e:
            print(f"[ng-quant] no artifact for {plan.label()}: {e}")
            print(f"[ng-quant] falling back to local calibration + export")
            scen_roots = [
                Path("tests/smoke/scenarios"),
                Path("tests/smoke/scenarios_nitrogen"),
            ]
            calib = load_calib_images_from_scenarios(*scen_roots)
            print(f"[ng-quant] calibrating on {len(calib)} frames")

            def _drive(_model, frame):
                _ = session.predict(frame)

            quantize_for_serving(
                session.model, precision=plan.precision.value,
                calib_images=calib, predict_fn=_drive,
            )
            export_dit_to_onnx(session.model, precision=plan.precision.value, steps=plan.steps)
            onnx_path = paths["onnx"]

        if onnx_path is None or not onnx_path.exists():
            raise SystemExit(f"[ng-quant] no usable ONNX for {plan.label()}; bailing")

        # Compile-or-load the per-GPU TRT plan and swap session.model.model.
        # Vision encoder + action decoder + denoise-loop control stay in
        # PyTorch; only the DiT step (the inner per-iteration forward) runs
        # on TRT, which is where the FP8/NVFP4 win materialises.
        session.model.model = TrtDitWrapper(onnx_path, precision=plan.precision.value)
        print(f"[ng-quant] TRT runtime swapped in for {plan.label()}")
    else:
        session.model = apply_optimization(session.model, plan)

    # Mixed-precision compute via autocast, not a model.to(dtype=...) cast —
    # the cast would desync NitroGen's float32 action buffers with the
    # embedding tensor in `prepare_input_embs.masked_scatter`. bf16/fp16
    # autocast leaves the buffers at fp32 and only the supported ops run in
    # the lower precision. fp8/nvfp4 fall through to the real quant tooling
    # in apply_optimization (TRT/ONNX export); they don't autocast here.
    autocast_dtype: torch.dtype | None = None
    if plan.torch_dtype_name == "bfloat16":
        autocast_dtype = torch.bfloat16
    elif plan.torch_dtype_name == "float16":
        autocast_dtype = torch.float16

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")
    print(f"NitroGen serving on {args.port} (game={args.game}, {plan.label()})")

    try:
        while True:
            request = pickle.loads(socket.recv())
            rtype = request.get("type")
            if rtype == "predict":
                # Per-request game/seed override (sent by NitrogenReasoner).
                if request.get("game_id") is not None:
                    session.selected_game = request["game_id"]
                if autocast_dtype is not None and torch.cuda.is_available():
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                        result = session.predict(request["image"])
                else:
                    result = session.predict(request["image"])
                response = {"status": "ok", "pred": result}
            elif rtype == "reset":
                session.reset()
                response = {"status": "ok"}
            elif rtype == "info":
                response = {"status": "ok", "info": session.info()}
            else:
                response = {"status": "error", "message": f"unknown type {rtype}"}
            socket.send(pickle.dumps(response))
    except KeyboardInterrupt:
        print("\nshutting down")
        return 0
    finally:
        socket.close()
        context.term()


def main() -> int:
    args = _build_parser().parse_args()
    return serve(args)


if __name__ == "__main__":
    raise SystemExit(main())
