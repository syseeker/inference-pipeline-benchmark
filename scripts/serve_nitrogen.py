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
    session.model = apply_optimization(session.model, plan)

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
