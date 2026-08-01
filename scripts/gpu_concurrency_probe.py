#!/usr/bin/env python3
"""Phase-0 concurrency gate for the GPU-contention study.

This is the hard gate the whole contention study depends on. Before any
colocation is worth measuring we must prove two things about *this* GPU in
*this* isolation config (MPS on/off, clocks pinned):

  1. Co-resident work genuinely **overlaps** on the SMs rather than
     time-slicing. If the GPU serialises two tenants, every later
     "degradation ratio" would just be measuring a queue, and the study
     changes meaning to "time-slice fairness" (see the skill's failure table).
  2. The run is **not clock-throttled**. Under co-residency power rises, the
     card hits its cap, clocks fall, everything slows — real, but NOT
     contention. A throttled window must be discarded, not published.

It also measures **run-to-run variance**, which sets the repetition policy for
Phases 1-6 from data instead of the customer's assumed <5% (design-decisions §5).

Method — synthetic tenants, on purpose
--------------------------------------
The tenant here is a small, deliberately *under-saturating* GEMM loop, not a
real model server. Two reasons:

  * A hardware-capability gate should isolate the GPU+MPS behaviour from
    model-load noise, OOM tuning, and server warmup. Those belong to the
    per-colocation runs (step 6), not to the gate.
  * Overlap is only *observable* when a single tenant leaves SM headroom. A
    kernel that already saturates the GPU cannot show 2x aggregate throughput
    no matter how well the scheduler overlaps it — so we keep the kernel small
    and check that solo GPU util leaves room (else the verdict is INCONCLUSIVE
    with a "shrink the kernel" hint).

The discriminator uses BOTH signals, because either one alone is ambiguous:

    overlap genuine   → aggregate throughput ~2x solo, per-iter latency ~1x
    serialised/sliced → aggregate throughput ~1x solo, per-iter latency ~2x

Usage
-----
    # default: 2 tenants, 1024^2 fp16 GEMM, 5s each, 5 reps, GPU 0
    python scripts/gpu_concurrency_probe.py --gpu rtx_pro6000 --json

    # tune if the kernel saturates (INCONCLUSIVE) or to stress harder
    python scripts/gpu_concurrency_probe.py --gpu rtx_pro6000 --matrix 768 --reps 5

Workers need torch+CUDA; the orchestrator does not. Workers run under
`--worker-python` (default: repo `.venv-vllm/bin/python`, which has torch).
`--worker` is the internal per-tenant entrypoint; users never pass it.

Output: a JSON report (with `--json`) plus a human summary. Exit code is 0 on a
PASS gate, 2 on FAIL or INCONCLUSIVE — so a wrapper can branch on it, matching
the repo's `bench` exit-code convention.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# Throttle reasons that invalidate a contention measurement. A clock drop from
# any of these is a power/thermal artifact, not a contention finding (§4.2).
FATAL_THROTTLE_REASONS = (
    "sw_power_cap",
    "hw_thermal_slowdown",
    "sw_thermal_slowdown",
    "hw_power_brake_slowdown",
)

# Overlap classification thresholds. Ideal genuine overlap is 2.0x aggregate
# throughput at 1.0x latency; ideal serialisation is 1.0x / 2.0x.
OVERLAP_PASS_RATIO = 1.5     # aggregate throughput >= this ⇒ real overlap
OVERLAP_FAIL_RATIO = 1.2     # aggregate throughput <= this (and latency up) ⇒ serialised
LATENCY_OK_RATIO = 1.5       # per-iter latency <= this under load ⇒ headroom held
SATURATION_UTIL_PCT = 90.0   # solo util above this ⇒ kernel saturates, gate inconclusive


# ─────────────────────────── worker (one tenant) ───────────────────────────

def run_worker(matrix: int, dtype: str, duration_s: float, gpu_index: int) -> dict[str, Any]:
    """One synthetic GPU tenant: a sustained GEMM loop for `duration_s`.

    Emits per-iteration wall times (each iter is synchronised, so the time is
    the kernel's, not a queue's). Returns throughput and latency percentiles.
    Runs in a subprocess under a torch-capable python.
    """
    import torch  # noqa: PLC0415 — only the worker needs torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False in the worker")
    torch.cuda.set_device(gpu_index)
    tdt = torch.float16 if dtype == "fp16" else torch.float32
    a = torch.randn(matrix, matrix, device="cuda", dtype=tdt)
    b = torch.randn(matrix, matrix, device="cuda", dtype=tdt)

    # Warm up: force cuBLAS algo selection and allocator settling out of the
    # measured window, so the first iterations don't skew the percentiles.
    for _ in range(20):
        _ = a @ b
    torch.cuda.synchronize()

    iter_ms: list[float] = []
    t_start = time.perf_counter()
    deadline = t_start + duration_s
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        _ = a @ b
        torch.cuda.synchronize()
        iter_ms.append((time.perf_counter() - t0) * 1000.0)
    elapsed = time.perf_counter() - t_start

    n = len(iter_ms)
    return {
        "gpu_index": gpu_index,
        "iters": n,
        "elapsed_s": elapsed,
        "throughput_iters_s": (n / elapsed) if elapsed > 0 else 0.0,
        "iter_p50_ms": _pctl(iter_ms, 0.50),
        "iter_p95_ms": _pctl(iter_ms, 0.95),
    }


# ─────────────────────────── clock / throttle sampler ──────────────────────

class ClockSampler:
    """Samples SM clock + throttle reasons over a window via nvidia-smi.

    Kept separate from benchmarks/probes/gpu_sampler.py on purpose: that sampler
    is on the working single-model path and parses fixed columns, so folding
    clock fields in there risks a regression. The probe owns its own lightweight
    sampler; folding clock support into the shared sampler is a step-6 follow-up
    when the orchestrator needs per-window clocks in every result.
    """

    _QUERY = (
        "clocks.sm,"
        "clocks_throttle_reasons.sw_power_cap,"
        "clocks_throttle_reasons.hw_thermal_slowdown,"
        "clocks_throttle_reasons.sw_thermal_slowdown,"
        "clocks_throttle_reasons.hw_power_brake_slowdown"
    )

    def __init__(self, gpu_index: int = 0, interval_ms: int = 100) -> None:
        self.gpu_index = gpu_index
        self.interval_ms = interval_ms
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._stop = threading.Event()
        self.sm_clock_mhz: list[float] = []
        self.throttle_reasons: set[str] = set()

    def __enter__(self) -> "ClockSampler":
        if not shutil.which("nvidia-smi"):
            return self
        cmd = [
            "nvidia-smi", "-i", str(self.gpu_index),
            f"--query-gpu={self._QUERY}",
            "--format=csv,noheader,nounits",
            "-lms", str(self.interval_ms),
        ]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
            )
        except (OSError, ValueError):
            self._proc = None
            return self
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()
        return self

    def _drain(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        reasons = ("sw_power_cap", "hw_thermal_slowdown", "sw_thermal_slowdown", "hw_power_brake_slowdown")
        try:
            for line in self._proc.stdout:
                if self._stop.is_set():
                    break
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 5:
                    continue
                try:
                    self.sm_clock_mhz.append(float(parts[0]))
                except ValueError:
                    pass
                for name, tok in zip(reasons, parts[1:5]):
                    if tok.lower().startswith("active"):
                        self.throttle_reasons.add(name)
        except Exception:
            pass

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.send_signal(signal.SIGTERM)
                self._proc.wait(timeout=2.0)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    self._proc.kill()
                except ProcessLookupError:
                    pass
            finally:
                if self._proc.stdout is not None:
                    try:
                        self._proc.stdout.close()
                    except OSError:
                        pass
        if self._reader is not None:
            self._reader.join(timeout=2.0)

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "sm_clock_mhz_p50": _pctl(self.sm_clock_mhz, 0.50),
            "sm_clock_mhz_min": (min(self.sm_clock_mhz) if self.sm_clock_mhz else None),
            "throttle_reasons": sorted(self.throttle_reasons),
        }


# ─────────────────────────── orchestration ─────────────────────────────────

def _spawn_workers(
    n: int, worker_python: str, matrix: int, dtype: str, duration_s: float, gpu_index: int,
) -> list[dict[str, Any]]:
    """Launch `n` worker subprocesses as close to simultaneously as possible,
    wait for all, and return their parsed JSON results.

    Raises RuntimeError if any worker fails, so a broken environment surfaces
    loudly instead of silently reporting one-tenant numbers as if concurrent.
    """
    cmd = [
        worker_python, str(Path(__file__).resolve()),
        "--worker",
        "--matrix", str(matrix),
        "--dtype", dtype,
        "--duration-s", str(duration_s),
        "--gpu-index", str(gpu_index),
    ]
    procs = [
        subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for _ in range(n)
    ]
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for i, p in enumerate(procs):
        out, err = p.communicate()
        if p.returncode != 0:
            errors.append(f"worker {i} exit {p.returncode}: {err.strip()[-400:]}")
            continue
        try:
            results.append(json.loads(out.strip().splitlines()[-1]))
        except (json.JSONDecodeError, IndexError):
            errors.append(f"worker {i} bad output: {out.strip()[-200:]}")
    if errors:
        raise RuntimeError("; ".join(errors))
    return results


def _run_concurrent_rep(
    n: int, worker_python: str, matrix: int, dtype: str, duration_s: float, gpu_index: int,
) -> dict[str, Any]:
    """One concurrent rep: N tenants together, clocks sampled over the window."""
    with ClockSampler(gpu_index=gpu_index, interval_ms=100) as clk:
        workers = _spawn_workers(n, worker_python, matrix, dtype, duration_s, gpu_index)
    agg = sum(w["throughput_iters_s"] for w in workers)
    # Worst tenant's median latency — the one that felt the contention most.
    iter_p50_max = max(w["iter_p50_ms"] for w in workers)
    return {
        "aggregate_throughput_iters_s": agg,
        "per_worker_throughput_iters_s": [w["throughput_iters_s"] for w in workers],
        "iter_p50_ms_max": iter_p50_max,
        **clk.summary,
    }


# ─────────────────────────── analysis (pure, unit-tested) ───────────────────

def classify_overlap(overlap_ratio: float, latency_ratio: float, solo_util_pct: float | None) -> tuple[str, str]:
    """Return (verdict, reason). verdict ∈ {PASS, FAIL, INCONCLUSIVE}.

    PASS         genuine overlap — aggregate throughput scaled up, latency held.
    FAIL         serialising — aggregate flat, latency roughly doubled.
    INCONCLUSIVE kernel saturates the GPU solo (no headroom to observe overlap),
                 or the two signals disagree; shrink --matrix and re-run.
    """
    if solo_util_pct is not None and solo_util_pct >= SATURATION_UTIL_PCT:
        return "INCONCLUSIVE", (
            f"solo GPU util {solo_util_pct:.0f}% ≥ {SATURATION_UTIL_PCT:.0f}% — the probe "
            "kernel saturates the GPU, so overlap is not observable. Re-run with a smaller "
            "--matrix so a single tenant leaves SM headroom."
        )
    if overlap_ratio >= OVERLAP_PASS_RATIO and latency_ratio <= LATENCY_OK_RATIO:
        return "PASS", (
            f"aggregate throughput {overlap_ratio:.2f}x solo at {latency_ratio:.2f}x latency — "
            "tenants overlap on the SMs."
        )
    if overlap_ratio <= OVERLAP_FAIL_RATIO and latency_ratio > LATENCY_OK_RATIO:
        return "FAIL", (
            f"aggregate throughput only {overlap_ratio:.2f}x solo and latency {latency_ratio:.2f}x — "
            "tenants are serialising, not sharing. Enable MPS and retry; if still serialised, "
            "rescope the study to time-slice fairness."
        )
    return "INCONCLUSIVE", (
        f"mixed signal: throughput {overlap_ratio:.2f}x, latency {latency_ratio:.2f}x. "
        "Adjust --matrix / --duration-s and re-run."
    )


def recommend_reps(cov: float) -> tuple[int, str]:
    """Repetition policy from measured coefficient of variation (§4.5, §5).

    Low variance ⇒ one run suffices; high variance (near-OOM bimodality, thermal
    coupling) ⇒ repeat and report both modes.
    """
    if cov <= 0.05:
        return 1, f"CoV {cov:.1%} ≤ 5% — one run per scenario is enough."
    if cov <= 0.15:
        return 3, f"CoV {cov:.1%} in (5%, 15%] — repeat 3x and report mean ± std."
    return 5, f"CoV {cov:.1%} > 15% — high/bimodal variance; repeat 5x and report both modes."


def evaluate_gate(analysis: dict[str, Any]) -> dict[str, Any]:
    """Combine overlap + clock integrity into the Phase-0 gate verdict."""
    overlap, overlap_reason = classify_overlap(
        analysis["overlap_ratio"], analysis["latency_ratio"], analysis.get("solo_gpu_util_p50"),
    )
    fatal = [r for r in analysis.get("throttle_reasons_seen", []) if r in FATAL_THROTTLE_REASONS]
    clock = "FAIL" if fatal else "PASS"
    clock_reason = (
        f"throttle fired during the run: {', '.join(fatal)} — the slowdown is power/thermal, "
        "not contention. Pin the power limit, lock clocks at 60–80% of boost, and re-run."
        if fatal else "no fatal throttle reason fired."
    )
    gate = "PASS" if (overlap == "PASS" and clock == "PASS") else "FAIL"
    reasons = [f"overlap: {overlap_reason}", f"clock: {clock_reason}"]
    if overlap == "INCONCLUSIVE":
        gate = "FAIL"
    return {"overlap": overlap, "clock_integrity": clock, "gate": gate, "reasons": reasons}


# ─────────────────────────── helpers ───────────────────────────────────────

def _pctl(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    import math
    k = (len(s) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _cov(values: list[float]) -> float:
    """Coefficient of variation — std / mean. 0 if <2 samples or mean 0."""
    if len(values) < 2:
        return 0.0
    m = statistics.mean(values)
    if m == 0:
        return 0.0
    return statistics.pstdev(values) / m


def _detect_isolation() -> str:
    """Best-effort MPS detection. Not authoritative — the run records what it
    saw so a serialised result can be explained (was MPS actually on?)."""
    if os.environ.get("CUDA_MPS_PIPE_DIRECTORY"):
        return "mps"
    if shutil.which("nvidia-cuda-mps-control"):
        try:
            r = subprocess.run(
                ["pgrep", "-x", "nvidia-cuda-mps-control"],
                capture_output=True, text=True, timeout=3,
            )
            if r.returncode == 0:
                return "mps"
        except (OSError, subprocess.SubprocessError):
            pass
    return "none"


def _device_name(gpu_index: int) -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        r = subprocess.run(
            ["nvidia-smi", "-i", str(gpu_index), "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _gpu_util_p50(gpu_index: int, seconds: float) -> float | None:
    """Sample utilization.gpu for `seconds` and return the p50, to judge whether
    a single tenant leaves the SM headroom that makes overlap observable."""
    if not shutil.which("nvidia-smi"):
        return None
    samples: list[float] = []
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        try:
            r = subprocess.run(
                ["nvidia-smi", "-i", str(gpu_index),
                 "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            )
            samples.append(float(r.stdout.strip().splitlines()[0]))
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass
        time.sleep(0.1)
    return _pctl(samples, 0.50)


def _default_worker_python() -> str:
    cand = REPO_ROOT / ".venv-vllm" / "bin" / "python"
    return str(cand) if cand.exists() else sys.executable


# ─────────────────────────── main ──────────────────────────────────────────

def _run_probe(args: argparse.Namespace) -> dict[str, Any]:
    gi = args.gpu_index
    worker_python = args.worker_python or _default_worker_python()

    # 1. Solo baseline — one tenant alone. Sample its util concurrently so we
    #    know whether the kernel leaves headroom (else overlap is unobservable).
    util_box: dict[str, float | None] = {}
    util_thread = threading.Thread(
        target=lambda: util_box.__setitem__("p50", _gpu_util_p50(gi, args.duration_s)),
        daemon=True,
    )
    util_thread.start()
    solo = _spawn_workers(1, worker_python, args.matrix, args.dtype, args.duration_s, gi)[0]
    util_thread.join(timeout=2.0)
    solo_util = util_box.get("p50")

    # 2. Concurrent reps — N tenants together, repeated to measure variance.
    reps = [
        _run_concurrent_rep(args.n_tenants, worker_python, args.matrix, args.dtype, args.duration_s, gi)
        for _ in range(args.reps)
    ]

    # 3. Analysis.
    agg_series = [r["aggregate_throughput_iters_s"] for r in reps]
    agg_mean = statistics.mean(agg_series)
    lat_series = [r["iter_p50_ms_max"] for r in reps]
    lat_mean = statistics.mean(lat_series)
    throttles_seen = sorted({r for rep in reps for r in rep["throttle_reasons"]})
    cov = _cov(agg_series)
    rec_reps, rec_reason = recommend_reps(cov)

    analysis = {
        "overlap_ratio": (agg_mean / solo["throughput_iters_s"]) if solo["throughput_iters_s"] else 0.0,
        "latency_ratio": (lat_mean / solo["iter_p50_ms"]) if solo["iter_p50_ms"] else 0.0,
        "throughput_cov": cov,
        "recommended_reps": rec_reps,
        "recommended_reps_reason": rec_reason,
        "solo_gpu_util_p50": solo_util,
        "throttle_reasons_seen": throttles_seen,
    }
    verdict = evaluate_gate(analysis)

    return {
        "gpu": args.gpu,
        "gpu_index": gi,
        "device_name": _device_name(gi),
        "isolation_detected": _detect_isolation(),
        "probe": {
            "matrix": args.matrix, "dtype": args.dtype, "duration_s": args.duration_s,
            "n_tenants": args.n_tenants, "reps": args.reps, "worker_python": worker_python,
        },
        "solo": {**solo, "gpu_util_p50": solo_util},
        "concurrent": {
            "reps": reps,
            "aggregate_throughput_mean": agg_mean,
            "iter_p50_ms_max_mean": lat_mean,
        },
        "analysis": analysis,
        "verdict": verdict,
    }


def _print_human(report: dict[str, Any]) -> None:
    v, a = report["verdict"], report["analysis"]
    print(f"\n=== Phase-0 concurrency gate — {report['gpu']} "
          f"({report.get('device_name') or 'unknown device'}) ===")
    print(f"isolation detected : {report['isolation_detected']}")
    print(f"overlap ratio      : {a['overlap_ratio']:.2f}x  (throughput, want ≥ {OVERLAP_PASS_RATIO})")
    print(f"latency ratio      : {a['latency_ratio']:.2f}x  (per-iter, want ≤ {LATENCY_OK_RATIO})")
    util = a.get("solo_gpu_util_p50")
    print(f"solo GPU util p50  : {util if util is None else f'{util:.0f}%'}")
    print(f"throughput CoV     : {a['throughput_cov']:.1%}  → {a['recommended_reps_reason']}")
    if a["throttle_reasons_seen"]:
        print(f"throttle reasons   : {', '.join(a['throttle_reasons_seen'])}")
    print(f"\n  overlap         : {v['overlap']}")
    print(f"  clock integrity : {v['clock_integrity']}")
    print(f"  GATE            : {v['gate']}")
    for r in v["reasons"]:
        print(f"    - {r}")
    print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase-0 GPU concurrency gate for the contention study.")
    p.add_argument("--gpu", default="unknown", help="GPU profile label for the report (e.g. rtx_pro6000).")
    p.add_argument("--gpu-index", type=int, default=0, help="CUDA device index to probe.")
    p.add_argument("--matrix", type=int, default=1024, help="GEMM side length. Smaller = more SM headroom.")
    p.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    p.add_argument("--duration-s", type=float, default=5.0, help="Seconds each tenant runs per rep.")
    p.add_argument("--n-tenants", type=int, default=2, help="Concurrent tenants (2 = the overlap test).")
    p.add_argument("--reps", type=int, default=5, help="Concurrent reps, for the variance measurement.")
    p.add_argument("--worker-python", default=None, help="Torch-capable python for workers (default: .venv-vllm).")
    p.add_argument("--json", action="store_true", help="Print the full JSON report to stdout.")
    p.add_argument("--out", default=None, help="Also write the JSON report to this path.")
    # Internal per-tenant entrypoint; users never pass this.
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    if args.worker:
        result = run_worker(args.matrix, args.dtype, args.duration_s, args.gpu_index)
        print(json.dumps(result))
        return 0

    report = _run_probe(args)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_human(report)

    return 0 if report["verdict"]["gate"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
