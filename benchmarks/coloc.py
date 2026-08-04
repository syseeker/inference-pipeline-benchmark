"""Contention orchestrator (step 6) — turn a Colocation into one timed window.

`scenario_config.iter_colocation()` resolves a `colocations:` entry into
`Colocation` objects (solo baselines first, then the co-resident windows). This
module *runs* one: launch each tenant's server, hold a single shared `t0`, drive
every tenant open-loop through its correct load generator, own one GPU sampler
per card the colocation occupies, and merge the per-request traces with the GPU
trace into the coloc result layout.

Non-negotiables enforced here (skills/gpu-contention-benchmark/reference/
design-decisions.md):

  §4.1  open-loop only    — a closed-loop tenant throttles itself in proportion
                            to the slowdown we are measuring, so its degradation
                            ratio would describe the harness, not the GPU.
  §4.2  clock integrity   — a window where a fatal throttle fired is discarded,
                            not published; the slowdown was power/thermal.
  §4.3  one sampler        — per CARD the colocation occupies, never per tenant.
                            DCGM is device-scoped: two samplers on one card is
                            two dcgmi processes and both tenants reporting that
                            card's memory as their own, while a card with NO
                            sampler leaves its placement result unexplainable.
  §4.4  shared wall clock  — time.time() for the alignment timeline across
                            processes; perf_counter() only for durations.

Two load generators, one wall clock (§1.13): AIPerf drives HTTP LLM/VLM tenants
(`--request-rate` + `--arrival-pattern`), perf_analyzer drives Triton CV tenants
(`--request-rate-range` + `--request-distribution`). AIPerf cannot drive Triton,
so the split is structural, not a preference.

`scripts/run_all_scenarios.sh` is deliberately untouched — its single-model
invariants (kill the server every round, refuse <30 GB free) are correct for the
serial sweep and wrong for co-residency. Contention gets this separate entry
point.

Placement (Phase 5): HTTP tenants are pinned with CUDA_VISIBLE_DEVICES; Triton
tenants are pinned by giving each GPU its own container — see the "Triton
placement" section below for why the two mechanisms differ.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
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


# ─────────────────────────── placement ─────────────────────────────────────

def occupied_devices(tenants: list[Tenant]) -> list[int]:
    """Every distinct GPU index the colocation actually sits on, ascending.

    This is the sampler set (§4.3): telemetry is opened per card, so a card that
    appears here and nowhere else still gets its utilisation, power and VRAM
    recorded. HTTP tenants contribute `devices` (a tensor-parallel tenant
    contributes ALL of its cards — it is really running on each of them);
    Triton tenants contribute the single card resolve_triton_device pins them to.

    A colocation that never mentions `device:` yields exactly [0], so the
    single-GPU case is byte-for-byte what it was before placement existed.
    """
    devs: set[int] = set()
    for t in tenants:
        if t.round.transport == "triton":
            devs.add(triton_device_of(t))
        else:
            devs.update(t.devices)
    return sorted(devs)


# ─────────────────────────── VRAM pre-flight ───────────────────────────────

def preflight_vram(tenants: list[Tenant]) -> list[str]:
    """Return a list of blocking issues (empty ⇒ OK) with the tenant VRAM plan.

    The rule (skill pre-flight #1): the sum of every HTTP tenant's
    `gpu_memory_utilization`, plus headroom for the CV tenants' footprint, must
    stay ≤ 1.0. An uncapped vLLM tenant is treated as claiming 0.90, because
    that is what it will actually take.

    That sum is a budget check, NOT a statement about how vLLM reads the flag.
    `--gpu-memory-utilization` is a target for TOTAL device utilisation — vLLM
    sizes KV from `total * util - torch.cuda.mem_get_info()`, and mem_get_info
    counts every process on the card — so caps summing under 1.0 does not stop
    a tenant whose cap is lower than what its neighbours already hold from
    computing a negative KV budget and dying with "No available memory for the
    cache blocks". mix-full failed exactly that way while passing this check.
    The fix is elsewhere: `build_server_cmd` states the KV size absolutely via
    `--kv-cache-memory-bytes`, which vLLM honours without profiling and without
    reference to the cap. This function still bounds the total.

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


# ─────────────────────────── workload payloads ─────────────────────────────
#
# A workload's `prompts:` / `data:` are the experiment. Nothing read them until
# now, so every LLM/VLM tenant silently ran on aiperf's built-in synthetic
# prompts and the video clips were never sent — cross-vlm-prefill-vs-llm, whose
# whole premise is a 40-frame prefill burst, would have measured a text workload
# and produced plausible numbers for the wrong thing. Hence two pieces here: a
# pre-flight that refuses to start when a payload is missing, and a materialiser
# that pairs the prompts with the media at run time.

# aiperf's `single_turn` loader (aiperf/dataset/loader/single_turn.py, model
# `SingleTurn`) keys media by modality, not by extension — so the file type has
# to be mapped to the field name here. Singular fields: one prompt, at most one
# media file, per line.
MEDIA_FIELD_BY_SUFFIX = {
    ".mp4": "video", ".mov": "video", ".webm": "video",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image",
    ".wav": "audio", ".mp3": "audio", ".flac": "audio",
}


def _workload_files(spec: dict[str, Any], key: str) -> list[Path]:
    """Absolute paths for a workload's `prompts:` / `data:` list.

    The yaml writes them repo-relative ("workspace/contention/..."), but aiperf
    runs with its own working directory, so everything that leaves this module —
    the pre-flight message, and every media path inside the generated JSONL —
    is absolute.
    """
    raw = spec.get(key) or []
    if isinstance(raw, (str, Path)):
        raw = [raw]
    out = []
    for p in raw:
        path = Path(p)
        out.append(path if path.is_absolute() else (REPO_ROOT / path))
    return out


def preflight_workload_payloads(tenants: list[Tenant]) -> list[str]:
    """Return blocking issues (empty ⇒ OK) for missing prompt / data files.

    Runs before any server launches, because the failure mode it guards is
    silent: aiperf falls back to synthetic prompts when `--input-file` is absent,
    so a missing payload costs a full GPU window and yields numbers that look
    fine. A loud abort is strictly better.
    """
    issues: list[str] = []
    for t in tenants:
        spec = t.workload_spec or {}
        for key in ("prompts", "data"):
            for path in _workload_files(spec, key):
                if not path.exists():
                    issues.append(
                        f"tenant {t.name!r} workload {t.workload!r}: {key} file not found: "
                        f"{path} — regenerate prompts with "
                        "`python3 scripts/build_contention_prompts.py` (test data: "
                        "workspace/contention/test_data/prepare_data.py)."
                    )
    return issues


def _media_field(path: Path, *, workload: str | None) -> str:
    field_name = MEDIA_FIELD_BY_SUFFIX.get(path.suffix.lower())
    if field_name is None:
        raise ValueError(
            f"workload {workload!r}: no aiperf single_turn field for media type "
            f"{path.suffix!r} ({path}). Known: {sorted(set(MEDIA_FIELD_BY_SUFFIX))}."
        )
    return field_name


def _read_prompt_texts(path: Path, *, workload: str | None) -> list[str]:
    """The `text` of every line in a prompts .jsonl."""
    texts: list[str] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"workload {workload!r}: {path}:{lineno} is not JSON ({e}); "
                "regenerate with `python3 scripts/build_contention_prompts.py`."
            ) from None
        text = obj.get("text") if isinstance(obj, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"workload {workload!r}: {path}:{lineno} has no `text` field — "
                "aiperf's single_turn loader needs one prompt per line."
            )
        texts.append(text)
    if not texts:
        raise ValueError(f"workload {workload!r}: {path} is empty.")
    return texts


def materialise_workload_input(tenant: Tenant, artifact_dir: Path) -> Path | None:
    """Write this tenant's aiperf `--input-file` and hand it to the builders.

    The prompts file carries text only; the media file is named separately by
    the workload's `data:`. They are combined HERE, per run, rather than shipped
    as a static artifact, because `vlm_video_short` and `vlm_video_long` share
    one prompts file and differ only by the clip — a checked-in combined file
    could only be correct for one of them.

    Lands in the run's artifact dir, never in workspace/: it is run output, and
    two windows using the same workload must not race on one path.

    Returns None (and sets nothing) for a workload with no `prompts:` — a CV
    workload declares `data:` only and is driven by perf_analyzer, which takes a
    different input format entirely.
    """
    spec = tenant.workload_spec or {}
    prompt_files = _workload_files(spec, "prompts")
    if not prompt_files:
        return None

    texts: list[str] = []
    for pf in prompt_files:
        texts += _read_prompt_texts(pf, workload=tenant.workload)

    media = _workload_files(spec, "data")
    lines: list[str] = []
    for i, text in enumerate(texts):
        row: dict[str, Any] = {"text": text}
        if media:
            # Cycle when a workload names several files so each one is actually
            # sent; with the single file every contention workload declares,
            # this attaches it to every prompt.
            path = media[i % len(media)]
            row[_media_field(path, workload=tenant.workload)] = str(path.resolve())
        lines.append(json.dumps(row, ensure_ascii=False))

    artifact_dir.mkdir(parents=True, exist_ok=True)
    # Keyed by tenant AND workload so the two vlm_video_* windows never collide.
    out = artifact_dir / f"{tenant.name}.{tenant.workload or 'workload'}.input.jsonl"
    out.write_text("\n".join(lines) + "\n")
    tenant._workload_input_file = out          # what _workload_input_file reads
    return out


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
    kv_gb = tenant.kv_budget_gb
    if r.backend == "vllm":
        cmd = [vllm_bin, "serve", r.hf_id, "--port", str(r.port), *r.launch_args]
        if cap is not None:
            cmd = _override_flag(cmd, "--gpu-memory-utilization", str(cap))
        if kv_gb:
            # `--gpu-memory-utilization` is a fraction of the WHOLE CARD, not a
            # private slice: vLLM sizes the KV cache from
            # `total * util - torch.cuda.mem_get_info()`, and mem_get_info
            # counts every process on the device. So a tenant whose cap is
            # lower than what its neighbours already hold computes a negative
            # budget and dies with "No available memory for the cache blocks" —
            # which is exactly how mix-full's VLM failed behind a 37 GB LLM,
            # while the pre-flight said the plan was fine because the caps
            # summed to under 1.0.
            #
            # kv_budget_gb is already the absolute KV size the study means, so
            # state it directly and stop the cap being load-bearing.
            cmd = _override_flag(cmd, "--kv-cache-memory-bytes",
                                 str(int(kv_gb * 1024 ** 3)))
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


# ─────────────────────────── Triton placement ──────────────────────────────
#
# HTTP tenants are pinned with CUDA_VISIBLE_DEVICES (build_server_env). Triton
# tenants cannot be: they are not our process — they are served by a container,
# and several tenants can share one. So placement for them is expressed as "one
# container per GPU that has Triton tenants on it", and everything that
# addresses that container (name, ports, model repo, readiness URL, the
# perf_analyzer `-u`) has to be derived from the tenant's device.

def triton_device_of(tenant: Tenant) -> int:
    """The single GPU index a Triton tenant is served on.

    Raises if the tenant asks for several — see resolve_triton_device: the CV
    models are not tensor-parallel, so a device list has no honest meaning here
    and guessing the first index would silently mis-place the tenant.
    """
    from benchmarks.triton_cv import resolve_triton_device
    try:
        return resolve_triton_device(tenant.device)
    except ValueError as e:
        raise ValueError(f"tenant {tenant.name!r}: {e}") from None


def triton_tenant_url(tenant: Tenant) -> str:
    """`host:port` of the Triton container serving THIS tenant.

    The yaml gives one `base_url` for the whole `triton:` backend, so every
    Triton tenant inherits the same base port; the device offset is what
    separates them. Returned scheme-less because that is what perf_analyzer's
    `-u` wants.
    """
    from benchmarks.triton_cv import triton_ports
    host = tenant.round.base_url.removeprefix("http://").removeprefix("https://")
    host = host.split("/")[0].rsplit(":", 1)[0] or "localhost"
    http_port, _, _ = triton_ports(tenant.round.port, triton_device_of(tenant))
    return f"{host}:{http_port}"


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
    # NOT int(): perf_analyzer accepts fractional rates, and truncating meant a
    # sub-1 rps tenant was driven at `--request-rate-range 0:0:1` and sent
    # nothing at all — kosmos-2.5 at 0.2 rps recorded achieved_rps 0.00 in every
    # colocation while the manifest said ok. Formatted with %g so an integer
    # rate still renders as "50", not "50.0".
    rps = f"{load.rps:g}"
    dist = "poisson" if load.pattern == "poisson" else "constant"
    # perf_analyzer -u expects hostname:port, not a full URL with scheme.
    clean_url = url.removeprefix("http://").removeprefix("https://")
    # A fixed request count, not a stabilising measurement. perf_analyzer's
    # time-window mode repeats windows until the numbers settle, which at a low
    # rate never happens: kosmos-2.5 at 0.2 rps gets ~12 requests per 60s window
    # and loops forever, producing no CSV at all. A contention window has a known
    # length and every tenant must cover the same one, so the deterministic form
    # is also the correct one — send exactly what the offered rate implies for
    # the window, then stop.
    count = max(1, round(load.rps * duration_s))
    cmd = [
        "perf_analyzer",
        "-m", model,
        "--service-kind", "triton",
        "-u", clean_url,
        "--request-count", str(count),
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
    # A rejected request still gets a record, with an `error` block and no
    # latency metrics. Marking it ok would turn a run where every request
    # 400'd into a baseline with a plausible achieved_rps and null latencies —
    # which is exactly what happened to the first qwen2.5-vl-7b window
    # (178/178 rejected for exceeding max-model-len, recorded as clean).
    err = obj.get("error") or {}
    return {
        "t_start_ms": float(start_ns) / 1e6,                         # epoch ns → ms
        "t_end_ms": (float(end_ns) / 1e6) if end_ns is not None else None,
        "ttft_ms": _metric_value(metrics, "time_to_first_token"),    # ms (streaming only)
        "itl_ms": _metric_value(metrics, "inter_token_latency"),     # ms (streaming only)
        "e2e_ms": e2e_ms,
        "output_tokens": int(tokens) if tokens is not None else None,
        "ok": not err,
        "error_code": err.get("code"),
        "error_message": (err.get("message") or None),
    }


def achieved_rps(records: list[dict[str, Any]]) -> float | None:
    """Completed requests per wall-clock second from the trace. Where this falls
    below the tenant's offered rps, the tenant is past its envelope (§4.1).

    CV tenants use perf_analyzer which reports aggregate throughput directly
    (`measured_rps`). LLM/VLM tenants compute from per-request timestamps.
    """
    if records and records[0].get("measured_rps") is not None:
        return records[0]["measured_rps"]
    # Successful requests only: a rejected request is not throughput. Counting
    # errors here reported achieved ~= offered for a window that served nothing.
    records = [r for r in records if r.get("ok", True)]
    stamps = [r["t_end_ms"] for r in records if r.get("t_end_ms") is not None]
    starts = [r["t_start_ms"] for r in records if r.get("t_start_ms") is not None]
    if len(stamps) < 2 or not starts:
        return None
    span_ms = max(stamps) - min(starts)
    if span_ms <= 0:
        return None
    return len(stamps) / (span_ms / 1000.0)


def union_window(traces: dict[str, list[dict[str, Any]]]) -> tuple[float, float] | None:
    """The [min start, max end] across all tenant traces — the span every GPU
    sampler must cover so each request has GPU context on its own card."""
    starts, ends = [], []
    for recs in traces.values():
        starts += [r["t_start_ms"] for r in recs if r.get("t_start_ms") is not None]
        ends += [r["t_end_ms"] for r in recs if r.get("t_end_ms") is not None]
    if not starts or not ends:
        return None
    return min(starts), max(ends)


# ─────────────────────────── run environment ───────────────────────────────
#
# Two facts about the box were previously taken on faith from the yaml and never
# checked while a run was happening. Both change how the numbers must be read,
# so a result has to carry its own evidence rather than point at a config file.
# Every capture here is best-effort by construction: a missing nvidia-smi
# records "we could not tell", it never aborts a window.

# NVLink shows up in `nvidia-smi topo -m` as NV<n> in the peer matrix; PCIe
# shows as PIX/PXB/PHB/NODE/SYS. Nothing else in that output looks like NV\d+.
_NVLINK_CELL = re.compile(r"\bNV\d+\b")


def _run_text(cmd: list[str], *, timeout: float = 10.0) -> tuple[str | None, str | None]:
    """(stdout, error) for a short probe command. Never raises: a missing binary
    or a non-zero exit is a *finding* to record, not a reason to lose the run."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return None, f"{cmd[0]} not found"
    except Exception as e:                       # timeout, permissions, OSError…
        return None, f"{cmd[0]} failed: {e}"
    if r.returncode != 0:
        return None, f"{cmd[0]} exited {r.returncode}: {(r.stderr or '').strip()[:200]}"
    return r.stdout, None


def capture_interconnect() -> dict[str, Any]:
    """`nvidia-smi topo -m`, verbatim, plus whether any NVLink appears in it.

    A tensor-parallel or two-card result is dominated by whatever fabric the
    cards actually talk over, and `nvlink: false` in the GPU yaml is a claim
    nobody has confirmed on this hardware. Storing the raw matrix means a later
    reader can re-derive the answer even if our parse of it was too crude.
    """
    out, err = _run_text(["nvidia-smi", "topo", "-m"])
    if out is None:
        return {"available": False, "error": err, "topo_matrix": None, "nvlink_detected": None}
    return {
        "available": True,
        "error": None,
        "topo_matrix": out,
        "nvlink_detected": bool(_NVLINK_CELL.search(out)),
    }


DEFAULT_MPS_PIPE_DIR = "/tmp/nvidia-mps"


def mps_pipe_dir_for_containers() -> str | None:
    """The MPS pipe directory to share into a Triton container, or None.

    A containerised CUDA process only joins the host's MPS control if the pipe
    directory is bind-mounted in AND named by CUDA_MPS_PIPE_DIRECTORY. Without
    it the container creates its own context and *time-slices* against the LLM
    tenants — which still produces plausible-looking numbers, so the failure is
    silent. `capture_mps()` would not catch it either: it inspects the host
    daemon, which is genuinely running.

    Resolved rather than required, because the daemon is usually started with
    no CUDA_MPS_PIPE_DIRECTORY set and therefore lands on the documented
    default. Returning None when the directory does not exist matters: docker
    `-v` on a missing host path creates it as a root-owned empty directory, so
    a blind mount would leave litter on every no-MPS box and still not join.
    """
    d = os.environ.get("CUDA_MPS_PIPE_DIRECTORY") or DEFAULT_MPS_PIPE_DIR
    return d if Path(d).is_dir() else None


def capture_mps() -> dict[str, Any]:
    """Evidence that an MPS control daemon exists on this host — nothing more.

    Deliberately NOT "is device N covered by MPS": which devices a daemon serves
    is not reliably introspectable (it depends on how the daemon was started and
    on CUDA_VISIBLE_DEVICES at that moment), and a confident wrong answer here
    would be worse than none. So we record what we saw — the control process,
    the pipe directory — and let the warning below do the interpreting.
    """
    pipe_dir = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
    out, err = _run_text(["pgrep", "-f", "nvidia-cuda-mps-control"], timeout=5.0)
    if out is None:
        # pgrep exits 1 when nothing matched, which _run_text reports as an
        # error; that is a real "no daemon", not an inconclusive probe.
        daemon = False if err and "exited 1" in err else None
        probe_error = None if daemon is False else err
    else:
        daemon = bool(out.strip())
        probe_error = None
    return {
        "control_daemon_running": daemon,       # True | False | None (could not tell)
        "pipe_directory": pipe_dir,
        # What a Triton container was actually given. `pipe_directory` above can
        # be None while the daemon runs fine on the default path, so this is the
        # field that says whether a containerised tenant could join MPS at all —
        # the host daemon running tells you nothing about that.
        "container_pipe_directory": mps_pipe_dir_for_containers(),
        "probe_error": probe_error,
        "detected": bool(daemon) or bool(pipe_dir),
    }


def capture_environment() -> dict[str, Any]:
    """Both run-time facts, captured once per window before the drivers start."""
    return {"interconnect": capture_interconnect(), "mps": capture_mps()}


def environment_warnings(coloc: Colocation, environment: dict[str, Any] | None) -> list[str]:
    """Conditions that make the window's numbers un-interpretable, in plain text.

    The one that matters: with more than one tenant and no MPS, the tenants do
    not share the SMs — they time-slice the card, and the measurement stops being
    about contention at all. Phase 0 measured 0.28x aggregate throughput with MPS
    off, so a window run this way is not a milder version of the result, it is a
    different experiment. Warn on the result rather than refuse the run: a
    deliberate `isolation: none` window is exactly how that number was obtained.
    """
    warnings: list[str] = []
    if not environment:
        return warnings
    mps = environment.get("mps") or {}
    if len(coloc.tenants) > 1 and not mps.get("detected"):
        warnings.append(
            "no MPS control daemon detected while running "
            f"{len(coloc.tenants)} tenants (isolation={coloc.isolation!r}) — without MPS the "
            "tenants time-slice the GPU instead of sharing it, and the degradation ratios "
            "describe the scheduler, not contention (Phase 0: 0.28x aggregate throughput)."
        )
    interconnect = environment.get("interconnect") or {}
    if not interconnect.get("available") and len(occupied_devices(coloc.tenants)) > 1:
        warnings.append(
            "interconnect topology unknown (" + str(interconnect.get("error")) + ") — "
            "multi-GPU results cannot be attributed to NVLink vs PCIe."
        )
    return warnings


def trace_warnings(traces: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Per-tenant request failures, which no other check surfaces.

    A tenant whose requests are all rejected still produces a full trace, a
    manifest and a directory; before this existed it also produced an
    achieved_rps indistinguishable from a healthy run. The first
    qwen2.5-vl-7b baseline had 178/178 requests rejected with HTTP 400
    ("Input length (19184) exceeds model's maximum context length (16384)")
    and was recorded as a clean solo baseline — which every later window
    would then have computed its degradation ratios against.
    """
    out: list[str] = []
    for name, recs in sorted(traces.items()):
        if not recs:
            # An EMPTY trace is the worst case, not a benign one. Skipping it is
            # how dinov2-base and kosmos-2.5 both recorded achieved_rps 0.00
            # with warnings: [] — the driver wrote no output at all, so there
            # were no records in which to find a failure.
            out.append(
                f"tenant {name!r}: NO trace records — the driver produced no output, so this "
                f"tenant served nothing. Check <run_dir>/{name}.aiperf/ and "
                f"<run_dir>/{name}.server.log."
            )
            continue
        bad = [r for r in recs if not r.get("ok", True)]
        if not bad:
            continue
        codes = sorted({r.get("error_code") for r in bad if r.get("error_code")})
        msg = next((r.get("error_message") for r in bad if r.get("error_message")), "")
        detail = f" HTTP {'/'.join(str(c) for c in codes)}." if codes else ""
        frac = f"{len(bad)}/{len(recs)}"
        if len(bad) == len(recs):
            out.append(
                f"tenant {name!r}: ALL {frac} requests failed.{detail} This run measured "
                f"nothing and must not be used as a baseline or a window. {msg[:300]}"
            )
        else:
            out.append(
                f"tenant {name!r}: {frac} requests failed.{detail} achieved_rps counts "
                f"only the successful ones. {msg[:200]}"
            )
    return out


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
    def gpu_root(self) -> Path:
        """benchmarks/results/<gpu>/ — the anchor for everything shared between
        runs (today: the Triton staging repo).

        Found by walking up to the `coloc/` ancestor rather than counting
        directory levels. Level-counting was already wrong: the CLI has always
        nested runs one deeper than this class assumed
        (coloc/<colocation>/<run_id>), which put the staging repo at
        <gpu>/coloc/triton_repo while build_triton_cv_repo.py exports to
        <gpu>/triton_repo — so Triton mounted an empty repo and never went
        ready. A multi-colocation plan nests deeper still, so the depth has to
        stop being load-bearing.
        """
        for p in self.root.parents:
            if p.name == "coloc":
                return p.parent
        return self.root.parent.parent

    @property
    def triton_repo_root(self) -> Path:
        # Stable across runs: benchmarks/results/<gpu>/triton_repo/
        #
        # This is also the STAGING repo: scripts/build_triton_cv_repo.py exports
        # every model's weights here once, and it is GPU 0's serving repo, so a
        # config that never mentions `device:` sees exactly today's layout.
        return self.gpu_root / "triton_repo"

    def triton_repo_root_for(self, device: int | list[int] | None = None) -> Path:
        """The model repository the container on `device` serves.

        Each GPU gets its own repo holding ONLY the models placed on it — one
        shared repo would make every container load every CV model and burn VRAM
        on cards that do not need it, perturbing the contention we are measuring.
        GPU 0 keeps the historical path (it is also the staging repo, which is
        where the weights physically live); other devices get a sibling dir whose
        version dirs symlink back to those weights.
        """
        from benchmarks.triton_cv import resolve_triton_device
        dev = resolve_triton_device(device)
        if dev == 0:
            return self.triton_repo_root
        return self.gpu_root / f"triton_repo-gpu{dev}"

    def server_log(self, tenant_name: str) -> Path:
        """Where a tenant's server stdout+stderr lands, per run.

        Kept inside the run directory rather than the shared
        <gpu>/server-logs/ the single-model sweep uses: colocation tenants are
        concurrent, so one shared file per backend would interleave two servers,
        and `--resume` would leave a failed run's log overwritten by the next
        one. A log belongs to the window that produced it.
        """
        return self.root / f"{tenant_name}.server.log"

    @property
    def gpu_ndjson(self) -> Path:
        return self.root / "gpu.ndjson"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"


def rate_warnings(coloc: Colocation, achieved: dict[str, float | None]) -> list[str]:
    """A tenant that achieved nothing, or a small fraction of what was offered.

    Separate from trace_warnings because achieved_rps is the number every
    degradation ratio divides by: a zero here does not fail the run, it poisons
    the analysis quietly.
    """
    out: list[str] = []
    for t in coloc.tenants:
        offered = t.load.rps
        if not offered:
            continue
        got = achieved.get(t.name)
        if got is None or got <= 0:
            out.append(
                f"tenant {t.name!r}: achieved_rps is {got!r} against an offered {offered} — "
                "it served nothing. This run must not be used as a baseline or a window."
            )
        elif got < 0.5 * offered:
            out.append(
                f"tenant {t.name!r}: achieved_rps {got:.2f} is under half the offered "
                f"{offered}. Either the tenant is past its capacity or the driver failed."
            )
    return out


def build_manifest(
    coloc: Colocation, *, t0_epoch_ms: float, gpu: str,
    sampler_summaries: dict[Any, dict[str, Any]] | None = None,
    throttle_reasons: list[str] | None = None,
    achieved: dict[str, float | None] | None = None,
    environment: dict[str, Any] | None = None,
    traces: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """The self-describing record for one colocation window.

    Carries the tenant set, load specs, isolation mode, the shared t0, the cards
    the window occupied, and the per-card sampler numbers — which attach to the
    *colocation*, not to any single tenant row (§4.3). `co_tenants` is sorted so
    the contention matrix in summary.py §10 is a groupby, not bespoke
    bookkeeping.

    `gpu_sampler` is keyed by device index as a STRING ("0", "1"): JSON object
    keys are strings, and round-tripping a manifest must not turn the key into
    something a reader has to special-case. `devices` states the same set
    directly so nothing has to re-derive placement from the tenant rows.

    `environment` is the run-time capture (capture_environment) — kept as a
    parameter rather than taken here so this stays pure and unit-testable
    without a GPU.
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
        "devices": occupied_devices(coloc.tenants),
        "tenants": tenants_out,
        "gpu_sampler": {str(dev): summ for dev, summ in sorted(
            (sampler_summaries or {}).items(), key=lambda kv: int(kv[0]))},
        "environment": environment or {},
        "warnings": (environment_warnings(coloc, environment)
                     + trace_warnings(traces or {})
                     + rate_warnings(coloc, achieved or {})),
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
    compare one of them against a reference that never ran (§2b).

    `launch_args` belongs here for the same reason the cap does, and was
    missed: `--max-model-len` also sets the KV cache, and it is where a config
    fix lands. Without it, raising qwen2.5-vl-7b from 16384 to 32768 — the
    change that took its baseline from 178/178 rejected requests to a working
    one — produced an identical key, so `--resume` would have skipped the
    re-run and left every ratio dividing by a baseline that measured nothing.
    """
    t = tenant
    return (t.round.backend, t.round.model_id, t.workload, t.load.pattern, t.load.rps,
            tuple(t.devices), t.gpu_memory_utilization,
            tuple(t.round.launch_args or ()))


def solo_key_from_manifest(manifest: dict[str, Any]) -> tuple | None:
    """Reconstruct `_solo_key` from a manifest already on disk.

    The manifest records every field the key is built from, so a baseline can
    be recognised by what it IS rather than by what its directory is called.
    Returns None if the manifest is not a single-tenant solo run.
    """
    tenants = manifest.get("tenants") or []
    if len(tenants) != 1:
        return None
    t = tenants[0]
    rnd = t.get("round") or {}
    load = t.get("load") or {}
    try:
        return (rnd.get("backend"), rnd.get("model_id"), t.get("workload"),
                load.get("pattern"), float(load.get("rps")),
                tuple(t.get("devices") or []), t.get("gpu_memory_utilization"),
                tuple(rnd.get("launch_args") or ()))
    except (TypeError, ValueError):
        return None


def find_existing_baseline(baselines_dir: Path, coloc: Colocation) -> Path | None:
    """A baseline already on disk that IS this run, whatever it is named.

    `--resume` used to ask only whether this run's exact directory existed, so
    any change to how the directory is named invalidated every result on disk —
    including results the change had nothing to do with. Adding launch_args to
    the key rehashed all 72 baselines and re-ran the ones that had not changed
    at all. Matching on identity instead makes resume immune to that: the
    directory name is a label, and the manifest is the record.
    """
    if not coloc.is_solo or not coloc.tenants:
        return None
    want = _solo_key(coloc.tenants[0])
    if not baselines_dir.is_dir():
        return None
    for m in sorted(baselines_dir.glob("*/manifest.json")):
        try:
            got = solo_key_from_manifest(json.loads(m.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
        if got is not None and got == want:
            return m
    return None


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


# ─────────────────────────── run directory layout ──────────────────────────

SOLO_DIR = "_baselines"


def _slug(text: str) -> str:
    """Filesystem-safe, still readable. Model ids carry '/' and '.'."""
    keep = [ch if (ch.isalnum() or ch in "-.") else "-" for ch in str(text).strip().lower()]
    return "".join(keep).strip("-") or "x"


def _identity_hash(parts: Any) -> str:
    """8 hex chars over a run's identity. Short enough to type, wide enough
    that two distinct windows in a 163-run study will not collide."""
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]  # noqa: S324 — naming, not security


def run_dir_for(root: Path, coloc: Colocation) -> Path:
    """Where one run of `coloc` lands under the plan's run root.

        <root>/_baselines/solo-<tenant><rps>-<model>-<h8>/
        <root>/<colocation-id>/coloc-<tenant><rps>…[-r<N>]-<h8>/

    Three properties this has to hold, now that one plan spans 39 colocations
    and the run index is global:

    1. Findable. Contention windows sit under their colocation id, so
       `coloc/mix-llm-cv/` is still the answer to "where did mix-llm-cv go".
    2. Baselines are not duplicated on disk. plan_runs dedupes solo runs
       across the plan, but a baseline belongs to the STUDY, not to whichever
       colocation happened to name it first — filing it under that colocation
       would re-run it under a different path the next time a different phase
       is selected. `_baselines/` is shared, and the name is a pure function
       of `_solo_key`, so the same baseline always maps to the same directory
       and `--resume` can see it.
    3. No collisions. rps_sweep / vary windows differ only in tenant fields,
       and `repetitions:` emits the same window N times. The `-h8` identity
       hash separates the former, `-r<N>` the latter — and neither depends on
       the run's position in the plan, so `--phase 3` and `--all` agree on the
       path for the same run.
    """
    # `@` separates name from rate so `llm2` at 2 rps reads as `llm2@2`
    # rather than the unparseable `llm22`.
    load_tag = "-".join(f"{_slug(t.name)}@{t.load.rps:g}" for t in coloc.tenants)
    if coloc.is_solo:
        # Every part of a baseline's name comes from `_solo_key` — deliberately
        # NOT the tenant name. The same baseline is labelled `llm` in one
        # colocation and `llm-a` in another; naming the directory after the
        # label would give one baseline two paths depending on which phase you
        # selected, which defeats both the on-disk dedup and --resume.
        t = coloc.tenants[0]
        h = _identity_hash(_solo_key(t))
        name = f"solo-{_slug(t.round.backend)}-{_slug(t.round.model_id)}@{t.load.rps:g}-{h}"
        return root / SOLO_DIR / name
    # The FULL tenant spec, not _solo_key: `vary:` can move a field that no
    # baseline distinguishes (secondary-backend-cv varies the Triton backend at
    # identical model/load), and hashing only the baseline key made those two
    # windows overwrite each other.
    h = _identity_hash([coloc.id, coloc.isolation, coloc.duration_s,
                        [t.to_dict() for t in coloc.tenants]])
    rep = f"-r{coloc.repetition}" if coloc.repetition > 1 else ""
    return root / _slug(coloc.id) / f"coloc-{load_tag}{rep}-{h}"


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
    share a single wall-clock anchor (t0) and one GPU sampler per occupied card
    (§4.3).

    `gpu_index` is only the fallback for a colocation with no tenants to read a
    placement from; the sampler set comes from the tenants themselves, because
    a hard-coded index is exactly how GPU 1's telemetry went missing.
    """

    def __init__(self, gpu: str, *, gpu_index: int = 0, sampler_interval_ms: int = 50,
                 warmup: int = 3, seed: int = 0) -> None:
        self.gpu = gpu
        self.gpu_index = gpu_index
        self.sampler_interval_ms = sampler_interval_ms
        self.warmup = warmup
        self.seed = seed
        self._server_logs: list[Any] = []

    def run(self, coloc: Colocation, paths: RunPaths) -> dict[str, Any]:
        issues = preflight_vram(coloc.tenants)
        if issues:
            raise RuntimeError("VRAM pre-flight failed: " + "; ".join(issues))
        payload_issues = preflight_workload_payloads(coloc.tenants)
        if payload_issues:
            raise RuntimeError("workload payload pre-flight failed: " + "; ".join(payload_issues))

        paths.root.mkdir(parents=True, exist_ok=True)

        # Captured before anything is launched, so the manifest records the box
        # as it was when the window ran rather than what the yaml claims about
        # it. Both probes are soft — see capture_environment.
        environment = capture_environment()

        # Pre-pass: write all CV model configs before launching the container so
        # Triton starts with a complete repo (avoids mid-startup model additions).
        triton_tenants = [t for t in coloc.tenants if t.round.transport == "triton"]
        if triton_tenants:
            self._build_triton_repos(triton_tenants, paths)

        # The launch loop is INSIDE the try: with one container per GPU there is
        # more than one thing to leak, and a failure on the second must not
        # leave the first running to poison the next colocation's numbers.
        servers: list[ServerHandle] = []
        try:
            for t in coloc.tenants:
                servers.append(self._ensure_server(t, paths, triton_tenants=triton_tenants))
            for h in servers:
                self._wait_ready(h.tenant, paths=paths)

            # §4.4 — one shared wall-clock anchor for every tenant.
            t0_epoch_ms = time.time() * 1000.0

            # §4.3 — one sampler per card this colocation occupies. ExitStack so
            # every card's window is the SAME window: all samplers are live
            # before the first driver starts and none stops until the last one
            # finishes, which hand-nesting cannot guarantee for an N decided at
            # run time.
            devices = occupied_devices(coloc.tenants) or [self.gpu_index]
            traces: dict[str, list[dict[str, Any]]] = {}
            with contextlib.ExitStack() as stack:
                samplers = {
                    dev: stack.enter_context(
                        GpuSampler(gpu_index=dev, interval_ms=self.sampler_interval_ms)
                    )
                    for dev in devices
                }
                procs = self._launch_drivers(coloc, paths)
                # A driver that cannot finish must not stall the study. perf_analyzer
                # loops until its measurement stabilises, and at a low request rate it
                # may never get enough samples per window to do so — kosmos-2.5 at
                # 0.2 rps hung indefinitely, with `p.wait()` waiting for it forever.
                # The window has a known length, so anything past a generous multiple
                # of it is a hang, not a slow run.
                budget = max(120.0, (coloc.duration_s or 180) * 3.0)
                deadline = time.time() + budget
                for name, proc in procs.items():
                    try:
                        proc.wait(timeout=max(10.0, deadline - time.time()))
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        with contextlib.suppress(Exception):
                            proc.wait(timeout=15)
                        print(f"  [warn] driver for tenant {name!r} exceeded "
                              f"{budget:.0f}s and was killed; its trace will be "
                              "empty and the run flagged", file=sys.stderr)
                for t in coloc.tenants:
                    if t.driver == "perf_analyzer":
                        traces[t.name] = parse_perf_analyzer_records(
                            paths.tenant_artifact_dir(t.name)
                        )
                    else:
                        traces[t.name] = parse_aiperf_records(paths.tenant_artifact_dir(t.name))
            sampler_summaries = {dev: gs.summary for dev, gs in samplers.items()}

            achieved = {t.name: achieved_rps(traces.get(t.name, [])) for t in coloc.tenants}
            for t in coloc.tenants:
                self._write_trace(paths.tenant_ndjson(t.name), traces.get(t.name, []))

            manifest = build_manifest(
                coloc, t0_epoch_ms=t0_epoch_ms, gpu=self.gpu,
                sampler_summaries=sampler_summaries, achieved=achieved,
                environment=environment, traces=traces,
            )
            paths.manifest.write_text(json.dumps(manifest, indent=2))
            for w in manifest.get("warnings", []):
                print(f"  [warn] {w}", file=sys.stderr)
            return manifest
        finally:
            for h in servers:
                self._stop_server(h)
            self._close_server_logs()

    # ---- server lifecycle --------------------------------------------------

    def _build_triton_repos(self, triton_tenants: list[Tenant], paths: RunPaths) -> None:
        """Write one model repository PER DEVICE, each holding only its models.

        Weights (model.onnx / model.plan) must be pre-exported via
        scripts/build_triton_cv_repo.py into the staging repo
        (RunPaths.triton_repo_root); this writes the Triton config and, for the
        non-staging devices, links the staged weights in. Called before
        _ensure_server so every container starts with a complete repo.
        """
        from benchmarks.triton_cv import RepoLayout, resolve_spec, write_model_repo
        staging = paths.triton_repo_root
        for t in triton_tenants:
            spec = resolve_spec(t.round.model_id)
            bk = t.triton_backend or ("python" if spec.is_python_backend else "tensorrt")
            repo = paths.triton_repo_root_for(triton_device_of(t))
            layout = write_model_repo(repo, spec, bk, params=_python_model_params(t))
            # A python-backend model.py IS the weight file and write_model_repo
            # copies it; there is nothing staged to link.
            if repo != staging and bk != "python":
                src = RepoLayout(repo_root=staging, name=spec.name, triton_backend=bk).weight_file
                _link_staged_weight(src, layout.weight_file)

    def _ensure_server(self, tenant: Tenant, paths: RunPaths,
                       *, triton_tenants: list[Tenant] | None = None) -> ServerHandle:
        bk = tenant.round.backend
        cmd = build_server_cmd(
            tenant,
            vllm_bin=venv_bin(bk, "vllm"),
            python_bin=venv_bin(bk, "python"),
            trtllm_bin=venv_bin(bk, "trtllm-serve"),
        )
        if cmd is None:
            # Triton tenant — _build_triton_repos has written config.pbtxt.
            # One container per GPU serves every CV model placed on that GPU;
            # the per-device container NAME is the mutex, so two tenants on one
            # card reuse a container and two on different cards each get their
            # own. Keying the mutex on a fixed name (as this did before per-GPU
            # placement) would hand the second tenant a container on the wrong
            # card, silently invalidating the placement.
            from benchmarks.triton_cv import triton_container_name, triton_ports
            device = triton_device_of(tenant)
            container = triton_container_name(device)
            http_port, grpc_port, metrics_port = triton_ports(tenant.round.port, device)
            # Reuse only a container that is serving THIS tenant's model.
            # "the container exists" is not the same question: triton-cv started
            # for yolov8-l with --model-control-mode=explicit --load-model=yolov8-l
            # is up and ready, and has never heard of dinov2-base. Reusing it
            # sent perf_analyzer at a model the server does not have, which
            # produced no output and an achieved_rps of 0.00 that the manifest
            # reported as ok.
            if self._triton_model_ready(http_port, tenant.round.model_id):
                return ServerHandle(tenant=tenant, proc=None, reused=True,
                                    container_name=None)
            # A container that is up but serving something else has to go, or
            # the port is taken and the new one cannot bind.
            if self._triton_container_running(container):
                subprocess.run(["docker", "rm", "-f", container],
                               capture_output=True, timeout=30)
            from benchmarks.triton_cv import build_triton_serve_cmd
            docker_cmd = build_triton_serve_cmd(
                paths.triton_repo_root_for(device),
                device=device,
                http_port=http_port,
                grpc_port=grpc_port,
                metrics_port=metrics_port,
                container_name=container,
                models=_triton_models_on(triton_tenants or [tenant], device),
                mps_pipe_dir=mps_pipe_dir_for_containers(),
                extra_mounts=[m for t in (triton_tenants or [tenant])
                              for m in python_model_mounts(t)],
            )
            # `docker run -d` prints the container id on success and the
            # reason on failure; the latter is the only record of why a CV
            # tenant never came up.
            log_path = paths.server_log(tenant.name)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w") as fh:
                proc = subprocess.Popen(docker_cmd, stdout=fh, stderr=subprocess.STDOUT)
                proc.wait()  # docker run -d exits immediately after spawning
            return ServerHandle(tenant=tenant, proc=None, reused=False,
                                container_name=container)
        if self._port_serving(tenant):
            return ServerHandle(tenant=tenant, proc=None, reused=True)
        # Overlay, not replace: the backend still needs HF_HOME, CUDA paths and
        # the rest of the caller's environment.
        env = {**os.environ, **build_server_env(tenant)}
        # Captured, not discarded. When a tenant misbehaves the server log is
        # the first thing the skill's failure-recovery table tells you to read,
        # and it did not exist: the qwen2.5-vl-7b 400s had to be reconstructed
        # from aiperf's profile_export.jsonl because vLLM's own complaint went
        # to /dev/null.
        log_path = paths.server_log(tenant.name)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = log_path.open("w")
        self._server_logs.append(log_fh)
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT,
                                env=env)
        return ServerHandle(tenant=tenant, proc=proc, reused=False)

    def _triton_ready(self, port: int) -> bool:
        from benchmarks.triton_cv import triton_ready_url
        try:
            with urllib.request.urlopen(triton_ready_url(port), timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    def _triton_model_ready(self, port: int, model_id: str) -> bool:
        """Is THIS model loaded and ready on the container at `port`?

        Triton answers per model at /v2/models/<name>/ready, which is the only
        check that distinguishes "a Triton is running" from "the tenant I am
        about to drive is being served".
        """
        url = f"http://localhost:{port}/v2/models/{model_id}/ready"
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
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

    def _serves_this_tenant(self, tenant: Tenant, timeout: float = 3.0) -> bool:
        """Is the server on this port serving THIS tenant's model?

        Not "is anything answering". Colocation tenants share a backend's port
        — cross-vlm-prefill-vs-llm puts qwen2.5-vl-7b and gemma2-9b both on
        8000 — so a bare HTTP 200 means the previous tenant's server has not
        finished shutting down, and treating it as ready makes the run reuse
        it: no server launched, no log, and aiperf driving one model's
        workload against a different model's weights. That completes and
        writes a normal-looking manifest, so the baseline is silently
        attributed to the wrong model and every ratio computed against it is
        wrong. Compare the id before trusting the port.
        """
        try:
            with urllib.request.urlopen(self._models_url(tenant), timeout=timeout) as r:
                if r.status != 200:
                    return False
                body = json.loads(r.read().decode() or "{}")
        except Exception:
            return False
        served = {str(m.get("id")) for m in (body.get("data") or []) if isinstance(m, dict)}
        if not served:
            return False
        want = {tenant.round.hf_id, tenant.round.model_id}
        # vLLM echoes back whatever --served-model-name / model path it was
        # given, so accept an exact match either way round plus a basename
        # match for a path-style id.
        return bool(served & want) or any(
            s.rsplit("/", 1)[-1] == str(w).rsplit("/", 1)[-1] for s in served for w in want if w
        )

    def _port_serving(self, tenant: Tenant) -> bool:
        return self._serves_this_tenant(tenant)

    def _wait_ready(self, tenant: Tenant, timeout_s: int | None = None,
                    paths: RunPaths | None = None) -> None:
        if tenant.round.transport == "triton":
            from benchmarks.triton_cv import triton_ports, triton_ready_url
            deadline = time.time() + (timeout_s or 300)
            # Poll the container on THIS tenant's card; a ready GPU-0 container
            # says nothing about whether GPU 1's has loaded its models yet.
            from benchmarks.triton_cv import triton_container_name
            url = triton_ready_url(triton_ports(tenant.round.port, triton_device_of(tenant))[0])
            while time.time() < deadline:
                try:
                    with urllib.request.urlopen(url, timeout=5) as r:
                        if r.status == 200:
                            return
                except Exception:
                    time.sleep(2.0)
            # `docker run -d` succeeding says only that the container started;
            # a model that fails to load leaves it up and never ready. The
            # container's own log carries the reason (e.g. "unable to get number
            # of CUDA devices: MPS client failed to connect"), and it disappears
            # with the container, so capture it into the run before raising.
            tail = (self._capture_container_log(
                triton_container_name(triton_device_of(tenant)), paths, tenant.name)
                if paths is not None else "")
            raise RuntimeError(
                f"Triton server not ready within {timeout_s or 300}s ({url}). "
                "Ensure model weights are exported (scripts/build_triton_cv_repo.py) "
                "and the Docker daemon is running."
                + (f"\nContainer log ({paths.server_log(tenant.name)}):\n{tail}" if tail else "")
            )
        deadline = time.time() + (timeout_s or tenant.round.ready_timeout_s or 600)
        url = self._models_url(tenant)
        while time.time() < deadline:
            # The model id, not just a 200: a 200 from the PREVIOUS tenant's
            # server would otherwise pass here and the window would measure
            # the wrong model. See _serves_this_tenant.
            if self._serves_this_tenant(tenant, timeout=5.0):
                return
            time.sleep(2.0)
        tail = ""
        if paths is not None:
            with contextlib.suppress(OSError):
                log = paths.server_log(tenant.name)
                if log.exists():
                    tail = "\n".join(log.read_text().splitlines()[-15:])
        raise RuntimeError(
            f"tenant {tenant.name!r} server not ready within budget ({url}); "
            f"expected model {tenant.round.hf_id!r}"
            + (f"\nServer log ({paths.server_log(tenant.name)}):\n{tail}" if tail else "")
        )

    def _launch_drivers(self, coloc: Colocation, paths: RunPaths) -> dict[str, subprocess.Popen]:
        from benchmarks.triton_cv import wrap_perf_analyzer_docker
        procs: dict[str, subprocess.Popen] = {}
        for t in coloc.tenants:
            art = paths.tenant_artifact_dir(t.name)
            art.mkdir(parents=True, exist_ok=True)
            if t.driver == "aiperf":
                # Per window, not per plan: the media file is workload-specific
                # and the combined JSONL lives with this run's artifacts.
                materialise_workload_input(t, art)
            if t.driver == "perf_analyzer":
                # perf_analyzer runs inside the SDK container; mount the
                # artifact dir at the same path so -f writes to the host.
                inner = build_perf_analyzer_cmd(
                    model=t.round.model_id,
                    # Not t.round.base_url: that is the backend-wide base, which
                    # points every Triton tenant at GPU 0's container.
                    url=triton_tenant_url(t),
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

    def _capture_container_log(self, container: str, paths: RunPaths,
                               tenant_name: str) -> str:
        """Persist `docker logs <container>` into the run and return its tail.

        Written before the container is torn down: once `--rm` reaps it the
        only account of why a model never loaded is gone.
        """
        try:
            r = subprocess.run(["docker", "logs", "--tail", "200", container],
                               capture_output=True, text=True, timeout=30)
        except Exception:
            return ""
        body = (r.stdout or "") + (r.stderr or "")
        if not body.strip():
            return ""
        with contextlib.suppress(OSError):
            path = paths.server_log(tenant_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            # APPEND. `docker run -d` already wrote its own output here, and if
            # the container never started that is the ONLY record of why —
            # overwriting it with "No such container: triton-cv" destroys the
            # launch error and leaves a message that explains nothing.
            with path.open("a") as fh:
                fh.write("\n--- docker logs ---\n")
                fh.write(body)
        return "\n".join(body.splitlines()[-15:])

    def _close_server_logs(self) -> None:
        for fh in self._server_logs:
            with contextlib.suppress(Exception):
                fh.close()
        self._server_logs.clear()

    def _stop_server(self, handle: ServerHandle) -> None:
        if handle.container_name and not handle.reused:
            # Stop, then remove — the container is started without --rm so its
            # logs survive a startup failure long enough to be captured.
            subprocess.run(["docker", "stop", handle.container_name],
                           capture_output=True, timeout=30)
            subprocess.run(["docker", "rm", "-f", handle.container_name],
                           capture_output=True, timeout=30)
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

def _triton_models_on(tenants: list[Tenant], device: int) -> list[str]:
    """Model ids of the Triton tenants placed on `device`, sorted for a stable
    argv (the container command ends up in logs and in a manifest diff)."""
    return sorted({t.round.model_id for t in tenants if triton_device_of(t) == device})


def _link_staged_weight(src: Path, dest: Path) -> None:
    """Point a per-device repo's version dir at the weights staged once on disk.

    scripts/build_triton_cv_repo.py exports each model's weights ONE time, into
    the staging repo (= GPU 0's repo). A second card's repo carries only its own
    configs, so without this its version dir is empty and Triton refuses to load
    the model. Symlink rather than copy: a TensorRT plan is hundreds of MB and
    both containers only ever read it. Silent when the weights are not staged
    yet — _wait_ready is where that surfaces, with the export instructions.
    """
    if dest.exists() or not src.exists():
        return
    try:
        dest.symlink_to(src)
    except OSError:
        shutil.copyfile(src, dest)


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


def _python_model_params(tenant: Tenant) -> dict[str, str]:
    """config.pbtxt `parameters` for a python-backend tenant.

    Its model.py reads the workload's real document and prompts at load rather
    than taking them per-request, because perf_analyzer can only synthesise a
    tensor of the declared shape and random pix2struct patches are not a
    document. Passing the paths through the config keeps the workload the yaml
    declares as the one actually served.
    """
    spec_files = tenant.workload_spec or {}
    out: dict[str, str] = {"hf_id": tenant.round.hf_id or ""}
    data = _workload_files(spec_files, "data")
    prompts = _workload_files(spec_files, "prompts")
    if data:
        out["document_path"] = str(data[0])
    if prompts:
        out["prompt_path"] = str(prompts[0])
    tokens = (spec_files or {}).get("output_tokens")
    if tokens:
        out["output_tokens"] = str(tokens)
    return out


def python_model_mounts(tenant: Tenant) -> list[tuple[str, str]]:
    """Host paths a python-backend model.py must be able to open in-container.

    Mounted at the same absolute path, so the config.pbtxt parameters written
    on the host resolve unchanged inside the container.
    """
    mounts = []
    for key in ("data", "prompts"):
        for f in _workload_files(tenant.workload_spec or {}, key):
            mounts.append((str(f), str(f)))
    return mounts


def _workload_input_file(tenant: Tenant) -> Path | None:
    """The tenant's materialised driver input file, or None.

    Set by materialise_workload_input, which the orchestrator calls once per
    tenant before building the driver command. Kept as an attribute lookup so
    the command builders stay pure: they read a path, they never touch the
    filesystem. None means "no prompts declared" ⇒ no `--input-file` ⇒ aiperf's
    synthetic dataset, which is only ever correct for a workload that genuinely
    declares none (preflight_workload_payloads guards the other case)."""
    return getattr(tenant, "_workload_input_file", None)
