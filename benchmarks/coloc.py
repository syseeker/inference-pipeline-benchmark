"""Contention orchestrator (step 6) — turn a Colocation into one timed window.

`scenario_config.iter_colocation()` resolves a `colocations:` entry into
`Colocation` objects (solo baselines first, then the co-resident windows). This
module *runs* one: launch each tenant's server, hold a single shared `t0`, drive
every tenant open-loop through its correct load generator, own exactly one GPU
sampler for the whole window, and merge the per-request traces with the GPU
trace into the coloc result layout.

Non-negotiables enforced here (skills/gpu-contention-benchmark/reference/
design-decisions.md):

  §4.1  open-loop only    — a closed-loop tenant throttles itself in proportion
                            to the slowdown we are measuring, so its degradation
                            ratio would describe the harness, not the GPU.
  §4.2  clock integrity   — a window where a fatal throttle fired is discarded,
                            not published; the slowdown was power/thermal.
  §4.3  one sampler        — N samplers means N dcgmi processes and every tenant
                            reporting the whole GPU's memory as its own.
  §4.4  shared wall clock  — time.time() for the alignment timeline across
                            processes; perf_counter() only for durations.

Two load generators, one wall clock (§1.13): AIPerf drives HTTP LLM/VLM tenants
(`--request-rate` + `--arrival-pattern`), perf_analyzer drives Triton CV tenants
(`--request-rate-range` + `--request-distribution`). AIPerf cannot drive Triton,
so the split is structural, not a preference.

`scripts/run_all_scenarios.sh` is deliberately untouched — its single-model
invariants (kill the server every round, refuse <30 GB free) are correct for the
serial sweep and wrong for co-residency. Contention gets this separate entry
point. Triton CV tenants' *server* side (model repo + config.pbtxt) is step 7;
this module builds their driver command but does not yet launch the Triton repo.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.probes.gpu_sampler import GpuSampler
from benchmarks.scenario_config import Colocation, Tenant

REPO_ROOT = Path(__file__).resolve().parents[1]

# vLLM defaults to gpu_memory_utilization=0.90 and will take the whole card,
# OOMing the second tenant at startup. A colocation tenant must cap itself.
VLLM_DEFAULT_GPU_FRACTION = 0.90

# Fatal throttle reasons — a clock drop from any of these is a power/thermal
# artifact, not contention (§4.2). Mirrors the Phase-0 probe's fatal set.
FATAL_THROTTLE_REASONS = (
    "sw_power_cap", "hw_thermal_slowdown", "sw_thermal_slowdown", "hw_power_brake_slowdown",
)


# ─────────────────────────── VRAM pre-flight ───────────────────────────────

def preflight_vram(tenants: list[Tenant]) -> list[str]:
    """Return a list of blocking issues (empty ⇒ OK) with the tenant VRAM plan.

    The rule (skill pre-flight #1): the sum of every HTTP tenant's
    `gpu_memory_utilization`, plus headroom for the CV tenants' footprint, must
    stay ≤ 1.0. An uncapped vLLM tenant is treated as claiming 0.90, because
    that is what it will actually take.

    The sum is per GPU, not per colocation: two tenants at 0.9 on different
    cards fit fine, and rejecting them would forbid the multi-GPU windows the
    schema now allows. A tensor-parallel tenant charges its fraction in FULL to
    every card it occupies — `gpu_memory_utilization` is a fraction of each
    card, not of the aggregate pool.
    """
    issues: list[str] = []
    per_gpu: dict[int, float] = {}
    uncapped: list[str] = []
    for t in tenants:
        if t.round.transport == "triton":
            # CV footprint is small and not expressed as a GPU fraction; it is
            # accounted as headroom, not summed here. Step 7 refines this with
            # the model's measured footprint.
            continue
        frac = t.gpu_memory_utilization
        if frac is None:
            uncapped.append(t.name)
            frac = VLLM_DEFAULT_GPU_FRACTION
        for dev in t.devices:
            per_gpu[dev] = per_gpu.get(dev, 0.0) + frac
    if uncapped:
        issues.append(
            f"tenants {uncapped} have no gpu_memory_utilization — each will claim "
            f"vLLM's default {VLLM_DEFAULT_GPU_FRACTION}, starving co-tenants. Set an explicit cap."
        )
    for dev in sorted(per_gpu):
        if per_gpu[dev] > 1.0:
            issues.append(
                f"GPU {dev}: sum of tenant GPU fractions is {per_gpu[dev]:.2f} > 1.0 — "
                "they will not co-reside. Lower the caps or move a tenant to another GPU."
            )
    return issues


# ─────────────────────────── command builders ──────────────────────────────

def venv_bin(backend: str, tool: str) -> str:
    """Resolve a backend tool to its per-venv binary, e.g. vllm →
    `.venv-vllm/bin/vllm`. The orchestrator's subprocesses don't inherit an
    activated venv, so a bare name would hit a `FileNotFoundError` (or the wrong
    binary). Falls back to the bare name for single-venv hosts, matching
    gpu_probe.sh's convention."""
    venvs = {"vllm": ".venv-vllm", "sglang": ".venv-sglang", "trtllm": ".venv-trtllm"}
    venv = venvs.get(backend)
    if venv:
        p = REPO_ROOT / venv / "bin" / tool
        if p.exists():
            return str(p)
    return tool


def build_server_cmd(
    tenant: Tenant, *, vllm_bin: str = "vllm", python_bin: str = "python",
    trtllm_bin: str = "trtllm-serve",
) -> list[str] | None:
    """Server launch command for an HTTP tenant, with its VRAM cap injected.

    Executables are injectable so the orchestrator can pass venv-resolved paths
    (the subprocess has no activated venv); the bare defaults keep unit tests
    hermetic. Returns None for Triton tenants — their server is a model repo,
    launched out of band in step 7, not a per-tenant process here.
    """
    r = tenant.round
    if r.transport == "triton":
        return None

    cap = tenant.gpu_memory_utilization
    if r.backend == "vllm":
        cmd = [vllm_bin, "serve", r.hf_id, "--port", str(r.port), *r.launch_args]
        if cap is not None:
            cmd = _override_flag(cmd, "--gpu-memory-utilization", str(cap))
        return cmd
    if r.backend == "sglang":
        cmd = [
            python_bin, "-m", "sglang.launch_server",
            "--model-path", r.hf_id, "--port", str(r.port), *r.launch_args,
        ]
        if cap is not None:
            cmd = _override_flag(cmd, "--mem-fraction-static", str(cap))
        return cmd
    if r.backend == "trtllm":
        trt_backend = r.trtllm_backend or "pytorch"
        trt_backend = "tensorrt" if trt_backend == "trtllm" else trt_backend
        return [
            trtllm_bin, r.hf_id, "--backend", trt_backend,
            "--port", str(r.port), *r.launch_args,
        ]
    raise ValueError(f"tenant {tenant.name!r}: no server launcher for backend {r.backend!r}")


def build_server_env(tenant: Tenant) -> dict[str, str]:
    """Environment overlay that pins a tenant to its GPU(s).

    Separate from build_server_cmd because placement is not a command-line
    concern for any of the three backends — they all read CUDA_VISIBLE_DEVICES —
    and because that builder's `list[str]` return is what the tests and the
    orchestrator already consume. Caller merges this over os.environ.

    A tensor-parallel tenant gets every index it occupies, in ascending order,
    so the backend's local device 0..N-1 map onto exactly those cards.
    """
    return {"CUDA_VISIBLE_DEVICES": ",".join(str(d) for d in tenant.devices)}


def build_aiperf_cmd(
    *, base_url: str, model: str, tenant: Tenant, duration_s: int, artifact_dir: Path,
    warmup: int = 3, seed: int = 0, endpoint_type: str = "chat", aiperf_bin: str = "aiperf",
) -> list[str]:
    """AIPerf command for an open-loop HTTP tenant.

    Open-loop is enforced by construction: we pass `--request-rate` +
    `--arrival-pattern`, never `--concurrency`. A tenant with no rate is a config
    error — that would be closed-loop and its ratio would be meaningless (§4.1).
    """
    load = tenant.load
    if not load.is_open_loop:
        raise ValueError(
            f"tenant {tenant.name!r} has no open-loop rate (pattern={load.pattern!r}, "
            f"rps={load.rps!r}); contention runs must be open-loop (design-decisions §4.1)."
        )
    root = base_url.rstrip("/")
    if root.endswith("/v1"):           # AIPerf --url is the server root, not /v1
        root = root[: -len("/v1")]
    cmd = [
        aiperf_bin, "profile",
        "--url", root,
        "--model", model,
        "--endpoint-type", endpoint_type,
        "--request-rate", str(load.rps),
        "--arrival-pattern", load.pattern,     # poisson | constant | gamma
        "--benchmark-duration", str(duration_s),
        "--warmup-request-count", str(warmup),
        "--random-seed", str(seed),
        "--output-artifact-dir", str(artifact_dir),
        "--ui", "none",
        # Stream so per-request TTFT + ITL are recorded — without it AIPerf
        # emits neither, and degradation_ratio_ttft_p95 cannot be computed.
        "--streaming",
    ]
    if load.output_tokens is not None:
        # Bound the decode length via the OpenAI request param. Without a cap,
        # requests don't finish inside --benchmark-duration and all cancel.
        cmd += ["--extra-inputs", f"max_tokens:{load.output_tokens}"]
    wl_input = _workload_input_file(tenant)
    if wl_input is not None:
        cmd += ["--input-file", str(wl_input), "--custom-dataset-type", "single_turn"]
    return cmd


def build_perf_analyzer_cmd(
    *, model: str, url: str, tenant: Tenant, duration_s: int,
    input_data: Path | None = None, output_csv: Path | None = None,
) -> list[str]:
    """perf_analyzer command for an open-loop Triton CV tenant.

    Open-loop via `--request-rate-range` (fixed low==high==rps) and
    `--request-distribution`. `--allow-client-shm=true` is a Triton *server*
    flag, not a perf_analyzer flag. `output_csv` writes aggregate stats to a
    file via `-f`; the orchestrator reads it with parse_perf_analyzer_records().
    """
    load = tenant.load
    if not load.is_open_loop:
        raise ValueError(f"tenant {tenant.name!r} (CV) needs an open-loop rps.")
    rps = int(load.rps)
    dist = "poisson" if load.pattern == "poisson" else "constant"
    # perf_analyzer -u expects hostname:port, not a full URL with scheme.
    clean_url = url.removeprefix("http://").removeprefix("https://")
    cmd = [
        "perf_analyzer",
        "-m", model,
        "--service-kind", "triton",
        "-u", clean_url,
        "--request-rate-range", f"{rps}:{rps}:1",
        "--request-distribution", dist,
        "--measurement-mode", "time_windows",
        "--measurement-interval", str(duration_s * 1000),
    ]
    if input_data is not None:
        cmd += ["--input-data", str(input_data)]
    if output_csv is not None:
        cmd += ["-f", str(output_csv)]
    return cmd


def driver_command(
    tenant: Tenant, *, base_url: str, model: str, duration_s: int, artifact_dir: Path,
    warmup: int = 3, seed: int = 0, endpoint_type: str = "chat", aiperf_bin: str = "aiperf",
) -> list[str]:
    """Dispatch a tenant to its load generator per the two-driver rule."""
    if tenant.driver == "aiperf":
        return build_aiperf_cmd(
            base_url=base_url, model=model, tenant=tenant, duration_s=duration_s,
            artifact_dir=artifact_dir, warmup=warmup, seed=seed, endpoint_type=endpoint_type,
            aiperf_bin=aiperf_bin,
        )
    if tenant.driver == "perf_analyzer":
        return build_perf_analyzer_cmd(
            model=model, url=base_url, tenant=tenant, duration_s=duration_s,
            input_data=_workload_input_file(tenant),
        )
    raise ValueError(
        f"tenant {tenant.name!r}: no driver for {tenant.driver!r} "
        "(expected aiperf | perf_analyzer; zmq_client/nitrogen is not a contention tenant)."
    )


# ─────────────────────────── trace parsing / alignment ─────────────────────

def parse_aiperf_records(artifact_dir: Path) -> list[dict[str, Any]]:
    """Per-request trace from an AIPerf artifact dir's `profile_export.jsonl`.

    Schema pinned against AIPerf v0.11 (live-validated): each line is
    `{"metadata": {...}, "metrics": {...}}` where metadata carries epoch-ns
    request_start_ns / request_end_ns + was_cancelled + benchmark_phase, and
    each metric is `{"value": x, "unit": "ms"|"tokens"}` (latencies already in
    ms, NOT ns). Warmup and cancelled records are dropped so only completed
    profiling requests reach the trace. Emits t_start_ms / t_end_ms on the
    shared epoch timeline. Returns [] if the file is absent.
    """
    path = artifact_dir / "profile_export.jsonl"
    if not path.exists():
        # Fall back to any *.jsonl export if the canonical name changes.
        alts = sorted(artifact_dir.glob("*export*.jsonl"))
        if not alts:
            return []
        path = alts[0]
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        rec = _map_request_record(obj)
        if rec is not None:
            records.append(rec)
    return records


def parse_perf_analyzer_records(artifact_dir: Path) -> list[dict[str, Any]]:
    """Aggregate summary from perf_analyzer's `-f` CSV output.

    perf_analyzer writes one header row + one data row per rate point. With
    `--request-rate-range R:R:1` there is exactly one data row. Latency columns
    are in microseconds; we convert to milliseconds to match the aiperf trace
    schema. `measured_rps` carries the Inferences/Second value so achieved_rps()
    can return it directly without per-request timestamps. Returns [] if the CSV
    is absent or unparseable.
    """
    path = artifact_dir / "perf_analyzer.csv"
    if not path.exists():
        return []
    try:
        text = path.read_text().strip()
    except OSError:
        return []
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    headers = [h.strip() for h in lines[0].split(",")]
    values = [v.strip() for v in lines[-1].split(",")]
    row: dict[str, str] = dict(zip(headers, values))

    def _float(key: str) -> float | None:
        v = row.get(key)
        try:
            return float(v) if v is not None else None
        except ValueError:
            return None

    def _us_to_ms(key: str) -> float | None:
        v = _float(key)
        return v / 1000.0 if v is not None else None

    return [{
        "measured_rps": _float("Inferences/Second"),
        "e2e_ms":       _us_to_ms("Avg latency"),
        "p50_ms":       _us_to_ms("p50 latency"),
        "p95_ms":       _us_to_ms("p95 latency"),
        "p99_ms":       _us_to_ms("p99 latency"),
        "ok": True,
    }]


def _metric_value(metrics: dict[str, Any], key: str) -> float | None:
    """Pull `metrics.<key>.value` (AIPerf wraps every metric as {value, unit})."""
    m = metrics.get(key)
    if isinstance(m, dict) and m.get("value") is not None:
        return float(m["value"])
    return None


def _map_request_record(obj: dict[str, Any]) -> dict[str, Any] | None:
    """Map one AIPerf `{metadata, metrics}` record to the coloc ndjson row.

    Drops warmup and cancelled records — a cancelled request finished only
    because the window closed, so counting it would inflate achieved_rps.
    """
    meta = obj.get("metadata") or {}
    metrics = obj.get("metrics") or {}
    if meta.get("was_cancelled"):
        return None
    if meta.get("benchmark_phase") not in (None, "profiling"):
        return None
    start_ns = meta.get("request_start_ns")
    end_ns = meta.get("request_end_ns")
    if start_ns is None:
        return None
    e2e_ms = _metric_value(metrics, "request_latency")               # already ms
    tokens = _metric_value(metrics, "output_sequence_length")
    if tokens is None:
        tokens = _metric_value(metrics, "output_token_count")
    return {
        "t_start_ms": float(start_ns) / 1e6,                         # epoch ns → ms
        "t_end_ms": (float(end_ns) / 1e6) if end_ns is not None else None,
        "ttft_ms": _metric_value(metrics, "time_to_first_token"),    # ms (streaming only)
        "itl_ms": _metric_value(metrics, "inter_token_latency"),     # ms (streaming only)
        "e2e_ms": e2e_ms,
        "output_tokens": int(tokens) if tokens is not None else None,
        "ok": True,
    }


def achieved_rps(records: list[dict[str, Any]]) -> float | None:
    """Completed requests per wall-clock second from the trace. Where this falls
    below the tenant's offered rps, the tenant is past its envelope (§4.1).

    CV tenants use perf_analyzer which reports aggregate throughput directly
    (`measured_rps`). LLM/VLM tenants compute from per-request timestamps.
    """
    if records and records[0].get("measured_rps") is not None:
        return records[0]["measured_rps"]
    stamps = [r["t_end_ms"] for r in records if r.get("t_end_ms") is not None]
    starts = [r["t_start_ms"] for r in records if r.get("t_start_ms") is not None]
    if len(stamps) < 2 or not starts:
        return None
    span_ms = max(stamps) - min(starts)
    if span_ms <= 0:
        return None
    return len(stamps) / (span_ms / 1000.0)


def union_window(traces: dict[str, list[dict[str, Any]]]) -> tuple[float, float] | None:
    """The [min start, max end] across all tenant traces — the span the single
    GPU sampler must cover so every request has GPU context."""
    starts, ends = [], []
    for recs in traces.values():
        starts += [r["t_start_ms"] for r in recs if r.get("t_start_ms") is not None]
        ends += [r["t_end_ms"] for r in recs if r.get("t_end_ms") is not None]
    if not starts or not ends:
        return None
    return min(starts), max(ends)


# ─────────────────────────── manifest / result layout ──────────────────────

@dataclass
class RunPaths:
    """Where one colocation run's artifacts land."""

    root: Path                          # benchmarks/results/<gpu>/coloc/<run_id>/

    def tenant_ndjson(self, tenant_name: str) -> Path:
        return self.root / f"{tenant_name}.ndjson"

    def tenant_artifact_dir(self, tenant_name: str) -> Path:
        return self.root / f"{tenant_name}.aiperf"

    @property
    def triton_repo_root(self) -> Path:
        # Stable across runs: benchmarks/results/<gpu>/triton_repo/
        # Two levels up from coloc/<run_id>/ lands at <gpu>/.
        return self.root.parent.parent / "triton_repo"

    @property
    def gpu_ndjson(self) -> Path:
        return self.root / "gpu.ndjson"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"


def build_manifest(
    coloc: Colocation, *, t0_epoch_ms: float, gpu: str,
    sampler_summary: dict[str, Any] | None = None,
    throttle_reasons: list[str] | None = None,
    achieved: dict[str, float | None] | None = None,
) -> dict[str, Any]:
    """The self-describing record for one colocation window.

    Carries the tenant set, load specs, isolation mode, the shared t0, and the
    whole-GPU sampler numbers — which attach to the *colocation*, not to any
    single tenant row (§4.3). `co_tenants` is sorted so the contention matrix in
    summary.py §10 is a groupby, not bespoke bookkeeping.
    """
    tenant_models = sorted(t.round.model_id for t in coloc.tenants)
    tenants_out = []
    for t in coloc.tenants:
        others = sorted(m for m in tenant_models if m != t.round.model_id)
        tenants_out.append({
            **t.to_dict(),
            "co_tenants": others,
            "offered_rps": t.load.rps,
            "achieved_rps": (achieved or {}).get(t.name),
        })
    return {
        "colocation_id": coloc.id,
        "phase": coloc.phase,
        "run_label": coloc.run_label,
        "is_solo": coloc.is_solo,
        "gpu": gpu,
        "isolation_mode": coloc.isolation,
        "duration_s": coloc.duration_s,
        "n_tenants": len(coloc.tenants),
        "t0_epoch_ms": t0_epoch_ms,
        "tenants": tenants_out,
        "gpu_sampler": sampler_summary or {},
        "throttle_reasons": sorted(throttle_reasons or []),
    }


# ─────────────────────────── solo-baseline caching ─────────────────────────

def _solo_key(tenant: Tenant) -> tuple:
    """Same identity scenario_config uses: a baseline is valid only at the SAME
    offered load, so load is part of the key. Lets a session skip re-running the
    ~40 duplicate baselines across a full study. Placement is part of it too —
    a GPU-0 baseline does not describe a tenant pinned elsewhere, nor a TP-2
    one. So is the VRAM cap: it sets the KV cache size, so the same model at
    two caps is two deployments, and sharing one baseline between them would
    compare one of them against a reference that never ran (§2b)."""
    t = tenant
    return (t.round.backend, t.round.model_id, t.workload, t.load.pattern, t.load.rps,
            tuple(t.devices), t.gpu_memory_utilization)


class SoloBaselineCache:
    """Tracks which solo baselines have run this session, so identical baselines
    shared across colocations execute once (iter_colocation dedups only within a
    colocation, by design)."""

    def __init__(self) -> None:
        self._seen: dict[tuple, str] = {}   # solo_key -> run_id

    def seen(self, coloc: Colocation) -> str | None:
        if not coloc.is_solo:
            return None
        return self._seen.get(_solo_key(coloc.tenants[0]))

    def record(self, coloc: Colocation, run_id: str) -> None:
        if coloc.is_solo:
            self._seen[_solo_key(coloc.tenants[0])] = run_id


# ─────────────────────────── planning ──────────────────────────────────────

def plan_runs(cfg: dict[str, Any], names: list[str], *, solo_only: bool = False) -> list[Colocation]:
    """Flatten iter_colocation across the named colocations, deduping solo
    baselines across the whole plan (not just within one colocation)."""
    from benchmarks.scenario_config import iter_colocation

    out: list[Colocation] = []
    seen: set[tuple] = set()
    for name in names:
        for coloc in iter_colocation(cfg, name):
            if coloc.is_solo:
                k = _solo_key(coloc.tenants[0])
                if k in seen:
                    continue
                seen.add(k)
            elif solo_only:
                continue
            out.append(coloc)
    return out


# ─────────────────────────── orchestration (live path) ─────────────────────

@dataclass
class ServerHandle:
    tenant: Tenant
    proc: subprocess.Popen | None
    reused: bool = False
    container_name: str | None = None   # set when we launched a Docker container


class ColocationOrchestrator:
    """Runs one Colocation end to end: launch servers, drive load, collect traces.

    HTTP tenants (vLLM / SGLang / TRT-LLM) are launched as subprocesses;
    Triton CV tenants are launched as a Docker container (step 7). Both paths
    share a single wall-clock anchor (t0) and a single GPU sampler (§4.3).
    """

    def __init__(self, gpu: str, *, gpu_index: int = 0, sampler_interval_ms: int = 50,
                 warmup: int = 3, seed: int = 0) -> None:
        self.gpu = gpu
        self.gpu_index = gpu_index
        self.sampler_interval_ms = sampler_interval_ms
        self.warmup = warmup
        self.seed = seed

    def run(self, coloc: Colocation, paths: RunPaths) -> dict[str, Any]:
        issues = preflight_vram(coloc.tenants)
        if issues:
            raise RuntimeError("VRAM pre-flight failed: " + "; ".join(issues))

        paths.root.mkdir(parents=True, exist_ok=True)

        # Pre-pass: write all CV model configs before launching the container so
        # Triton starts with a complete repo (avoids mid-startup model additions).
        triton_tenants = [t for t in coloc.tenants if t.round.transport == "triton"]
        if triton_tenants:
            self._build_triton_repos(triton_tenants, paths)

        servers = [self._ensure_server(t, paths) for t in coloc.tenants]
        try:
            for h in servers:
                self._wait_ready(h.tenant)

            # §4.4 — one shared wall-clock anchor for every tenant.
            t0_epoch_ms = time.time() * 1000.0

            # §4.3 — exactly one sampler for the whole window.
            traces: dict[str, list[dict[str, Any]]] = {}
            with GpuSampler(gpu_index=self.gpu_index, interval_ms=self.sampler_interval_ms) as gs:
                procs = self._launch_drivers(coloc, paths)
                for name, p in procs.items():
                    p.wait()
                for t in coloc.tenants:
                    if t.driver == "perf_analyzer":
                        traces[t.name] = parse_perf_analyzer_records(
                            paths.tenant_artifact_dir(t.name)
                        )
                    else:
                        traces[t.name] = parse_aiperf_records(paths.tenant_artifact_dir(t.name))
            sampler_summary = gs.summary

            achieved = {t.name: achieved_rps(traces.get(t.name, [])) for t in coloc.tenants}
            for t in coloc.tenants:
                self._write_trace(paths.tenant_ndjson(t.name), traces.get(t.name, []))

            manifest = build_manifest(
                coloc, t0_epoch_ms=t0_epoch_ms, gpu=self.gpu,
                sampler_summary=sampler_summary, achieved=achieved,
            )
            paths.manifest.write_text(json.dumps(manifest, indent=2))
            return manifest
        finally:
            for h in servers:
                self._stop_server(h)

    # ---- server lifecycle --------------------------------------------------

    def _build_triton_repos(self, triton_tenants: list[Tenant], paths: RunPaths) -> None:
        """Write config.pbtxt + create version dirs for all CV models in one pass.

        Weights (model.onnx / model.plan) must be pre-exported via
        scripts/build_triton_cv_repo.py — this only writes the Triton config.
        Called before _ensure_server so the container starts with a full repo.
        """
        from benchmarks.triton_cv import resolve_spec, write_model_repo
        for t in triton_tenants:
            spec = resolve_spec(t.round.model_id)
            bk = t.triton_backend or "tensorrt"
            write_model_repo(paths.triton_repo_root, spec, bk)

    def _ensure_server(self, tenant: Tenant, paths: RunPaths) -> ServerHandle:
        bk = tenant.round.backend
        cmd = build_server_cmd(
            tenant,
            vllm_bin=venv_bin(bk, "vllm"),
            python_bin=venv_bin(bk, "python"),
            trtllm_bin=venv_bin(bk, "trtllm-serve"),
        )
        if cmd is None:
            # Triton tenant — _build_triton_repos has written config.pbtxt.
            # One container serves all CV models in the colocation; use the
            # container name as a mutex so we don't launch a second one.
            container = "triton-cv"
            if self._triton_container_running(container) or self._triton_ready(tenant.round.port):
                return ServerHandle(tenant=tenant, proc=None, reused=True,
                                    container_name=None)
            from benchmarks.triton_cv import build_triton_serve_cmd
            docker_cmd = build_triton_serve_cmd(
                paths.triton_repo_root,
                http_port=tenant.round.port,
                grpc_port=tenant.round.port + 1,
                metrics_port=tenant.round.port + 2,
                container_name=container,
            )
            proc = subprocess.Popen(docker_cmd, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.STDOUT)
            proc.wait()  # docker run -d exits immediately after spawning
            return ServerHandle(tenant=tenant, proc=None, reused=False,
                                container_name=container)
        if self._port_serving(tenant):
            return ServerHandle(tenant=tenant, proc=None, reused=True)
        # Overlay, not replace: the backend still needs HF_HOME, CUDA paths and
        # the rest of the caller's environment.
        env = {**os.environ, **build_server_env(tenant)}
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                                env=env)
        return ServerHandle(tenant=tenant, proc=proc, reused=False)

    def _triton_ready(self, port: int) -> bool:
        from benchmarks.triton_cv import triton_ready_url
        try:
            with urllib.request.urlopen(triton_ready_url(port), timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    def _triton_container_running(self, name: str) -> bool:
        try:
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}", name],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0 and result.stdout.strip() == "true"
        except Exception:
            return False

    def _port_serving(self, tenant: Tenant) -> bool:
        try:
            with urllib.request.urlopen(self._models_url(tenant), timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    def _wait_ready(self, tenant: Tenant, timeout_s: int | None = None) -> None:
        if tenant.round.transport == "triton":
            from benchmarks.triton_cv import triton_ready_url
            deadline = time.time() + (timeout_s or 300)
            url = triton_ready_url(tenant.round.port)
            while time.time() < deadline:
                try:
                    with urllib.request.urlopen(url, timeout=5) as r:
                        if r.status == 200:
                            return
                except Exception:
                    time.sleep(2.0)
            raise RuntimeError(
                f"Triton server not ready within {timeout_s or 300}s ({url}). "
                "Ensure model weights are exported (scripts/build_triton_cv_repo.py) "
                "and the Docker daemon is running."
            )
        deadline = time.time() + (timeout_s or tenant.round.ready_timeout_s or 600)
        url = self._models_url(tenant)
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    if r.status == 200:
                        return
            except Exception:
                time.sleep(2.0)
        raise RuntimeError(f"tenant {tenant.name!r} server not ready within budget ({url})")

    def _launch_drivers(self, coloc: Colocation, paths: RunPaths) -> dict[str, subprocess.Popen]:
        from benchmarks.triton_cv import wrap_perf_analyzer_docker
        procs: dict[str, subprocess.Popen] = {}
        for t in coloc.tenants:
            art = paths.tenant_artifact_dir(t.name)
            art.mkdir(parents=True, exist_ok=True)
            if t.driver == "perf_analyzer":
                # perf_analyzer runs inside the SDK container; mount the
                # artifact dir at the same path so -f writes to the host.
                inner = build_perf_analyzer_cmd(
                    model=t.round.model_id,
                    url=t.round.base_url,
                    tenant=t,
                    duration_s=coloc.duration_s,
                    input_data=_workload_input_file(t),
                    output_csv=art / "perf_analyzer.csv",
                )
                cmd = wrap_perf_analyzer_docker(inner, mounts=[(str(art), str(art))])
            else:
                cmd = driver_command(
                    t, base_url=t.round.base_url, model=t.round.hf_id,
                    duration_s=coloc.duration_s, artifact_dir=art,
                    warmup=self.warmup, seed=self.seed,
                    aiperf_bin=venv_bin(t.round.backend, "aiperf"),
                )
            procs[t.name] = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                             stderr=subprocess.STDOUT)
        return procs

    def _stop_server(self, handle: ServerHandle) -> None:
        if handle.container_name and not handle.reused:
            subprocess.run(
                ["docker", "stop", handle.container_name],
                capture_output=True, timeout=30,
            )
            return
        if handle.proc is None:
            return
        handle.proc.terminate()
        try:
            handle.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            handle.proc.kill()

    def _models_url(self, tenant: Tenant) -> str:
        root = tenant.round.base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        return f"{root}/v1/models"

    @staticmethod
    def _write_trace(path: Path, records: list[dict[str, Any]]) -> None:
        path.write_text("".join(json.dumps(r) + "\n" for r in records))


# ─────────────────────────── helpers ───────────────────────────────────────

def _has_flag(cmd: list[str], flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in cmd)


def _override_flag(cmd: list[str], flag: str, value: str) -> list[str]:
    """Drop every occurrence of `flag` (both `--flag=v` and `--flag v`) and
    append it once with `value`.

    A tenant's VRAM cap has to win over whatever `launch_args` carried in.
    `backends.<b>.extra_args` holds GPU-shape DEFAULTS — on this GPU that
    includes `--gpu-memory-utilization=0.90` — and every tenant inherits
    them. Merely checking "is the flag already there?" let that default
    stand, so both tenants launched at 0.90 and the second OOMed, while the
    VRAM pre-flight (which reads the tenant cap, not the command) reported
    the plan as fine. A per-tenant cap is an override, not a fallback.
    """
    out: list[str] = []
    skip_next = False
    for a in cmd:
        if skip_next:
            skip_next = False
            continue
        if a == flag:
            skip_next = True           # `--flag value` — drop the value too
            continue
        if a.startswith(flag + "="):
            continue
        out.append(a)
    return [*out, flag, value]


def _workload_input_file(tenant: Tenant) -> Path | None:
    """Resolve the tenant's workload payload file (image/video) if any. The
    workload→file mapping lives in the yaml `workloads:` block; here we only
    surface a path the driver can pass through. Text-only workloads return None.
    Full workload resolution is wired with the live run; this keeps the builder
    honest about where the payload comes from."""
    # Populated when the orchestrator is handed the resolved workloads block;
    # kept None here so command builders stay pure and unit-testable.
    return getattr(tenant, "_workload_input_file", None)
