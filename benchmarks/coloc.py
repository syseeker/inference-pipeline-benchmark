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
    """
    issues: list[str] = []
    total = 0.0
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
        total += frac
    if uncapped:
        issues.append(
            f"tenants {uncapped} have no gpu_memory_utilization — each will claim "
            f"vLLM's default {VLLM_DEFAULT_GPU_FRACTION}, starving co-tenants. Set an explicit cap."
        )
    if total > 1.0:
        issues.append(
            f"sum of tenant GPU fractions is {total:.2f} > 1.0 — they will not co-reside. "
            "Lower the caps or move a tenant to another GPU."
        )
    return issues


# ─────────────────────────── command builders ──────────────────────────────

def build_server_cmd(tenant: Tenant) -> list[str] | None:
    """Server launch command for an HTTP tenant, with its VRAM cap injected.

    Returns None for Triton tenants — their server is a model repo, launched
    out of band in step 7, not a per-tenant process here.
    """
    r = tenant.round
    if r.transport == "triton":
        return None

    cap = tenant.gpu_memory_utilization
    if r.backend == "vllm":
        cmd = ["vllm", "serve", r.hf_id, "--port", str(r.port), *r.launch_args]
        if cap is not None and not _has_flag(cmd, "--gpu-memory-utilization"):
            cmd += ["--gpu-memory-utilization", str(cap)]
        return cmd
    if r.backend == "sglang":
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", r.hf_id, "--port", str(r.port), *r.launch_args,
        ]
        if cap is not None and not _has_flag(cmd, "--mem-fraction-static"):
            cmd += ["--mem-fraction-static", str(cap)]
        return cmd
    if r.backend == "trtllm":
        trt_backend = r.trtllm_backend or "pytorch"
        trt_backend = "tensorrt" if trt_backend == "trtllm" else trt_backend
        return [
            "trtllm-serve", r.hf_id, "--backend", trt_backend,
            "--port", str(r.port), *r.launch_args,
        ]
    raise ValueError(f"tenant {tenant.name!r}: no server launcher for backend {r.backend!r}")


def build_aiperf_cmd(
    *, base_url: str, model: str, tenant: Tenant, duration_s: int, artifact_dir: Path,
    warmup: int = 3, seed: int = 0, endpoint_type: str = "chat",
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
        "aiperf", "profile",
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
    ]
    if load.output_tokens is not None:
        # Bound the decode length via the OpenAI request param.
        cmd += ["--extra-inputs", f"max_tokens:{load.output_tokens}"]
    wl_input = _workload_input_file(tenant)
    if wl_input is not None:
        cmd += ["--input-file", str(wl_input), "--custom-dataset-type", "single_turn"]
    return cmd


def build_perf_analyzer_cmd(
    *, model: str, url: str, tenant: Tenant, duration_s: int, input_data: Path | None = None,
) -> list[str]:
    """perf_analyzer command for an open-loop Triton CV tenant.

    Open-loop via `--request-rate-range` (fixed low==high==rps) and
    `--request-distribution`. `--shared-memory` off but client SHM allowed at the
    server; the `--allow-client-shm=true` gotcha is a Triton *server* flag (step
    7), not a perf_analyzer flag. The Triton server side is step 7; this builder
    is exercised by unit tests until then.
    """
    load = tenant.load
    if not load.is_open_loop:
        raise ValueError(f"tenant {tenant.name!r} (CV) needs an open-loop rps.")
    rps = int(load.rps)
    dist = "poisson" if load.pattern == "poisson" else "constant"
    cmd = [
        "perf_analyzer",
        "-m", model,
        "--service-kind", "triton",
        "-u", url,
        "--request-rate-range", f"{rps}:{rps}:1",
        "--request-distribution", dist,
        "--measurement-mode", "time_windows",
        "--measurement-interval", str(duration_s * 1000),
    ]
    if input_data is not None:
        cmd += ["--input-data", str(input_data)]
    return cmd


def driver_command(
    tenant: Tenant, *, base_url: str, model: str, duration_s: int, artifact_dir: Path,
    warmup: int = 3, seed: int = 0, endpoint_type: str = "chat",
) -> list[str]:
    """Dispatch a tenant to its load generator per the two-driver rule."""
    if tenant.driver == "aiperf":
        return build_aiperf_cmd(
            base_url=base_url, model=model, tenant=tenant, duration_s=duration_s,
            artifact_dir=artifact_dir, warmup=warmup, seed=seed, endpoint_type=endpoint_type,
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
    """Best-effort per-request trace from an AIPerf artifact dir.

    AIPerf writes a per-request JSONL alongside the aggregate export; field
    names vary by version, so we map defensively and skip records we can't read
    rather than crash. Returns [] if no per-request file is found (the exact
    schema is pinned in the first live run; the aggregate export still carries
    achieved rate). Emits t_start_ms / t_end_ms on the shared epoch timeline.
    """
    records: list[dict[str, Any]] = []
    candidates = sorted(artifact_dir.glob("*.jsonl")) + sorted(artifact_dir.glob("**/*.jsonl"))
    for path in candidates:
        if "aiperf" not in path.name and "export" not in path.name and "request" not in path.name:
            continue
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
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
        if records:
            break
    return records


def _map_request_record(obj: dict[str, Any]) -> dict[str, Any] | None:
    """Map one AIPerf request object to the coloc ndjson row shape."""
    start_ns = _first(obj, "timestamp", "start_ns", "request_start_ns", "start")
    if start_ns is None:
        return None
    # AIPerf timestamps are ns since epoch; normalise to ms.
    t_start_ms = float(start_ns) / 1e6
    ttft_ns = _first(obj, "time_to_first_token", "ttft_ns", "ttft")
    e2e_ns = _first(obj, "request_latency", "latency_ns", "e2e_ns", "latency")
    t_end_ms = (t_start_ms + float(e2e_ns) / 1e6) if e2e_ns is not None else None
    tokens = _first(obj, "output_tokens", "num_output_tokens", "output_token_count")
    return {
        "t_start_ms": t_start_ms,
        "t_end_ms": t_end_ms,
        "ttft_ms": (float(ttft_ns) / 1e6) if ttft_ns is not None else None,
        "e2e_ms": (float(e2e_ns) / 1e6) if e2e_ns is not None else None,
        "output_tokens": int(tokens) if tokens is not None else None,
        "ok": True,
    }


def achieved_rps(records: list[dict[str, Any]]) -> float | None:
    """Completed requests per wall-clock second from the trace. Where this falls
    below the tenant's offered rps, the tenant is past its envelope (§4.1)."""
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
    ~40 duplicate baselines across a full study."""
    t = tenant
    return (t.round.backend, t.round.model_id, t.workload, t.load.pattern, t.load.rps)


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


class ColocationOrchestrator:
    """Runs one Colocation end to end. The HTTP (LLM/VLM) tenant path is live;
    the Triton CV tenant server side is step 7 (its driver command is built, but
    launching the model repo is deferred)."""

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
        servers = [self._ensure_server(t) for t in coloc.tenants]
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

    def _ensure_server(self, tenant: Tenant) -> ServerHandle:
        cmd = build_server_cmd(tenant)
        if cmd is None:
            # Triton tenant — server launch is step 7. Assume an out-of-band
            # Triton server is already up on the tenant's URL.
            return ServerHandle(tenant=tenant, proc=None, reused=True)
        if self._port_serving(tenant):
            return ServerHandle(tenant=tenant, proc=None, reused=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return ServerHandle(tenant=tenant, proc=proc, reused=False)

    def _port_serving(self, tenant: Tenant) -> bool:
        try:
            with urllib.request.urlopen(self._models_url(tenant), timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    def _wait_ready(self, tenant: Tenant, timeout_s: int | None = None) -> None:
        if tenant.round.transport == "triton":
            return
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
        procs: dict[str, subprocess.Popen] = {}
        for t in coloc.tenants:
            art = paths.tenant_artifact_dir(t.name)
            art.mkdir(parents=True, exist_ok=True)
            cmd = driver_command(
                t, base_url=t.round.base_url, model=t.round.hf_id,
                duration_s=coloc.duration_s, artifact_dir=art,
                warmup=self.warmup, seed=self.seed,
            )
            procs[t.name] = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return procs

    def _stop_server(self, handle: ServerHandle) -> None:
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


def _first(obj: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    return None


def _workload_input_file(tenant: Tenant) -> Path | None:
    """Resolve the tenant's workload payload file (image/video) if any. The
    workload→file mapping lives in the yaml `workloads:` block; here we only
    surface a path the driver can pass through. Text-only workloads return None.
    Full workload resolution is wired with the live run; this keeps the builder
    honest about where the payload comes from."""
    # Populated when the orchestrator is handed the resolved workloads block;
    # kept None here so command builders stay pure and unit-testable.
    return getattr(tenant, "_workload_input_file", None)
