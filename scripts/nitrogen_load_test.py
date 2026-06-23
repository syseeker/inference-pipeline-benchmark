#!/usr/bin/env python3
"""Replica-based load test for NitroGen policy backends (PR #8).

NitroGen's ZMQ REP socket is single-flight by design (one request in-flight
per server). Concurrency is achieved by launching N replicas on consecutive
ports and distributing requests across them — one client thread per replica.

Usage:
    python scripts/nitrogen_load_test.py \\
        --ckpt ~/.cache/huggingface/hub/.../ng.pt \\
        --exec eager --precision bf16 --steps 16 \\
        --replicas 1,4,16,32 \\
        --requests 200 --warmup 10 \\
        --scenarios-dir tests/smoke/scenarios_nitrogen \\
        --out benchmarks/results/rtx_pro6000/aiperf/nitrogen-eager-bf16/

Output: one JSON file per replica-count under --out/, compatible with the
summary generator's "Concurrency profile" section (section 9).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from statistics import mean, quantiles
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_PORT = 5600  # load-test replicas use 5600+ to avoid sweeps ports (5560-5564)
NITROGEN_PYTHON = str(REPO_ROOT / ".venv-nitrogen" / "bin" / "python")


def _gpu_ids() -> list[int]:
    """Return list of available CUDA GPU indices via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
        )
        return [int(x.strip()) for x in out.strip().splitlines() if x.strip().isdigit()]
    except Exception:
        return [0]


def _assign_gpus(n_replicas: int, gpu_ids: list[int]) -> list[int]:
    """Round-robin assign n_replicas across available GPUs."""
    return [gpu_ids[i % len(gpu_ids)] for i in range(n_replicas)]


def _start_replica(
    ckpt: str, port: int, exec_mode: str, precision: str, steps: int, gpu_id: int
) -> tuple[subprocess.Popen, object]:
    python = NITROGEN_PYTHON if Path(NITROGEN_PYTHON).exists() else sys.executable
    cmd = [
        python, str(REPO_ROOT / "scripts" / "serve_nitrogen.py"),
        ckpt, "--port", str(port),
        f"--exec={exec_mode}", f"--precision={precision}", f"--steps={steps}",
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    log_path = Path(f"/tmp/nitrogen_replica_{port}.log")
    log_fh = open(log_path, "w")  # noqa: SIM115 — closed by caller in finally
    return subprocess.Popen(cmd, stdout=log_fh, stderr=log_fh, env=env), log_fh


def _wait_for_port(port: int, timeout_s: int = 300) -> bool:
    import socket
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True
        except OSError:
            time.sleep(1)
    return False


def _load_scenarios(scenarios_dir: Path) -> list[dict[str, Any]]:
    import pickle

    import numpy as np
    from PIL import Image

    scenarios = []
    for scenario_path in sorted(scenarios_dir.iterdir()):
        screen = scenario_path / "screen.png"
        request_file = scenario_path / "request.json"
        if not screen.exists() or not request_file.exists():
            continue
        req = json.loads(request_file.read_text())
        img = np.array(Image.open(screen).convert("RGB").resize((256, 256)))
        scenarios.append({"image": img, "game_id": req.get("game_id"), "request": req})
    return scenarios


def _run_client(
    port: int,
    scenarios: list[dict],
    n_warmup: int,
    n_measured: int,
    results: list[float],
    errors: list[str],
) -> None:
    try:
        import pickle
        import zmq

        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.connect(f"tcp://localhost:{port}")
        sock.setsockopt(zmq.RCVTIMEO, 30_000)

        total = n_warmup + n_measured
        for i in range(total):
            sc = scenarios[i % len(scenarios)]
            req = {"type": "predict", "image": sc["image"],
                   "game_id": sc.get("game_id"), "seed": 0}
            t0 = time.perf_counter()
            sock.send(pickle.dumps(req))
            resp = pickle.loads(sock.recv())
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if resp.get("status") != "ok":
                errors.append(f"port {port}: {resp.get('message')}")
                continue
            if i >= n_warmup:
                results.append(elapsed_ms)

        sock.close()
        ctx.term()
    except Exception as e:
        errors.append(f"port {port}: {e}")


def run_phase(
    replicas: int,
    ckpt: str,
    exec_mode: str,
    precision: str,
    steps: int,
    scenarios: list[dict],
    n_warmup: int,
    n_measured: int,
    gpu_ids: list[int],
) -> dict[str, Any]:
    ports = [BASE_PORT + i for i in range(replicas)]
    gpu_assignment = _assign_gpus(replicas, gpu_ids)
    print(f"   GPU assignment: {dict(zip(ports, gpu_assignment))}")
    procs: list[subprocess.Popen] = []
    log_fhs: list = []
    try:
        python = NITROGEN_PYTHON if Path(NITROGEN_PYTHON).exists() else sys.executable
        print(f"   using python: {python}")
        for port, gpu_id in zip(ports, gpu_assignment):
            print(f"   starting replica on port {port} (GPU {gpu_id}), log: /tmp/nitrogen_replica_{port}.log")
            proc, log_fh = _start_replica(ckpt, port, exec_mode, precision, steps, gpu_id)
            procs.append(proc)
            log_fhs.append(log_fh)

        for port in ports:
            if not _wait_for_port(port):
                log_path = Path(f"/tmp/nitrogen_replica_{port}.log")
                tail = ""
                if log_path.exists():
                    lines = log_path.read_text().splitlines()
                    tail = "\n".join(lines[-20:]) if lines else ""
                    print(f"\n-- replica {port} log (last 20 lines) --\n{tail}\n--", file=sys.stderr)
                return {"replicas": replicas, "error": f"replica on port {port} failed to start", "log_tail": tail}

        all_results: list[list[float]] = [[] for _ in ports]
        all_errors: list[list[str]] = [[] for _ in ports]
        threads = [
            threading.Thread(
                target=_run_client,
                args=(port, scenarios, n_warmup, n_measured, all_results[i], all_errors[i]),
                daemon=True,
            )
            for i, port in enumerate(ports)
        ]

        wall_start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall_s = time.perf_counter() - wall_start

        latencies = [ms for r in all_results for ms in r]
        errors = [e for errs in all_errors for e in errs]
        n_ok = len(latencies)
        n_total = replicas * n_measured

        if not latencies:
            return {"replicas": replicas, "error": "all requests failed", "errors": errors}

        qs = quantiles(latencies, n=100)
        return {
            "replicas": replicas,
            "concurrency": replicas,
            "n_gpus_used": len(set(gpu_assignment)),
            "gpu_assignment": gpu_assignment,
            "n_requests": n_ok,
            "n_errors": len(errors),
            "wall_s": round(wall_s, 3),
            "throughput_req_s": round(n_ok / wall_s, 2),
            "latency_p50_ms": round(qs[49], 2),
            "latency_p95_ms": round(qs[94], 2),
            "latency_p99_ms": round(qs[98], 2),
            "latency_mean_ms": round(mean(latencies), 2),
            "error_rate_pct": round(100 * len(errors) / n_total, 1) if n_total else 0,
        }
    finally:
        for p in procs:
            try:
                p.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
        time.sleep(1)
        for p in procs:
            try:
                p.kill()
            except ProcessLookupError:
                pass
        for fh in log_fhs:
            try:
                fh.close()
            except Exception:
                pass


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="Path to ng.pt checkpoint.")
    p.add_argument("--exec", default="eager", dest="exec_mode", help="eager|compile|cudagraph|tensorrt|onnxruntime")
    p.add_argument("--precision", default="bf16", help="bf16|fp8|nvfp4")
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--replicas", default="1,4,16,32", help="Comma-separated replica counts to sweep.")
    p.add_argument("--requests", type=int, default=200, help="Measured requests per replica per phase.")
    p.add_argument("--warmup", type=int, default=10, help="Warmup requests (discarded) per replica.")
    p.add_argument("--scenarios-dir", default="tests/smoke/scenarios_nitrogen", help="Scenario directory.")
    p.add_argument("--out", required=True, help="Output directory for result JSONs.")
    p.add_argument("--backend", default=None, help="Backend label for the result (e.g. nitrogen-eager).")
    p.add_argument("--model", default=None, help="Model label for the result (e.g. nitrogen-500m-bf16).")
    args = p.parse_args()

    scenarios_dir = Path(args.scenarios_dir)
    if not scenarios_dir.exists():
        print(f"scenarios dir not found: {scenarios_dir}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = _load_scenarios(scenarios_dir)
    if not scenarios:
        print(f"no scenarios found under {scenarios_dir}", file=sys.stderr)
        return 1

    gpu_ids = _gpu_ids()
    print(f">> detected GPUs: {gpu_ids}")

    replica_counts = [int(x) for x in args.replicas.split(",")]
    n_gpus = len(gpu_ids)
    all_phases = []
    for n in replica_counts:
        if n > n_gpus:
            print(f">> skipping {n} replicas — only {n_gpus} GPU(s) available (1 replica per GPU max)")
            continue
        print(f">> phase: {n} replica(s), 1 per GPU × {args.requests} requests each")
        result = run_phase(
            replicas=n,
            ckpt=args.ckpt,
            exec_mode=args.exec_mode,
            precision=args.precision,
            steps=args.steps,
            scenarios=scenarios,
            n_warmup=args.warmup,
            n_measured=args.requests,
            gpu_ids=gpu_ids,
        )
        result["backend"] = args.backend or f"nitrogen-{args.exec_mode}"
        result["model"] = args.model or f"nitrogen-500m-{args.precision}"
        all_phases.append(result)
        print(json.dumps(result, indent=2))

    out_file = out_dir / "profile_export_aiperf.json"
    out_file.write_text(json.dumps({"phases": all_phases}, indent=2))
    print(f"\n>> results written to {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
