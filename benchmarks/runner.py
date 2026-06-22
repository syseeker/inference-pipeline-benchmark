"""Benchmark runner.

Runs every scenario under tests/smoke/scenarios end-to-end through the
real Pipeline against a chosen backend, then emits:

    benchmarks/results/<gpu>/<backend>/<scenario>__<run_id>.json   # per-scenario row
    benchmarks/results/<gpu>/<backend>-<model_id>-<run_id>.json    # aggregate BenchmarkResult

Single-round usage:
    python -m benchmarks.runner --backend vllm --gpu rtx_pro6000
    python -m benchmarks.runner --backend vllm --gpu rtx_pro6000 \
        --model qwen3.6-27b-fp8
    python -m benchmarks.runner --backend vllm --gpu rtx_pro6000 \
        --variant eager

Sweep usage:
    python -m benchmarks.runner --gpu rtx_pro6000 --sweep models

`--label` is a free-form tag stamped onto every row (defaults to the
variant name, or "baseline"). Use it to pair runs in summary.py.
"""

from __future__ import annotations

import json
import re
import socket
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import typer

from benchmarks.accuracy import GamepadAccuracy, aggregate_accuracy, compare_gamepad
from benchmarks.metrics import (
    BenchmarkResult,
    LatencySamples,
    compute_throughput,
    derive_itl,
    summarise_latencies,
    summarise_token_counts,
    utc_now_iso,
)
from benchmarks.nitrogen_exec import parse_exec_plan
from benchmarks.probes.gpu_sampler import GpuSampler
from benchmarks.probes.prom_poller import PromPoller
from benchmarks.probes.prom_scrape import ScrapeError, scrape
from benchmarks.scenario_config import (
    Round,
    iter_sweep,
    load_gpu_config,
    resolve_round,
)
from tests.smoke.scenarios.loader import load_all
from vlm_pipeline import Pipeline
from vlm_pipeline.config import PipelineConfig
from vlm_pipeline.reasoners.base import VlmReasoner

app = typer.Typer(add_completion=False)


def _make_reasoner(backend: str, cfg: PipelineConfig) -> VlmReasoner:
    if backend == "vllm":
        from vlm_pipeline.reasoners.vllm_backend import VllmReasoner

        return VllmReasoner(cfg.vllm)
    if backend == "sglang":
        from vlm_pipeline.reasoners.sglang_backend import SglangReasoner

        return SglangReasoner(cfg.sglang)
    if backend == "trtllm":
        from vlm_pipeline.reasoners.trtllm_backend import TrtLlmReasoner

        return TrtLlmReasoner(cfg.trtllm)
    if backend.startswith("nitrogen"):
        # nitrogen-eager / nitrogen-tensorrt / ... — engine backends that run the
        # NitroGen model. All share one reasoner; the engine is a server launch flag.
        from vlm_pipeline.reasoners.nitrogen_backend import NitrogenReasoner

        return NitrogenReasoner(cfg.nitrogen)
    raise typer.BadParameter(
        f"unknown backend: {backend} (expected vllm | sglang | trtllm | nitrogen-*)"
    )


def _framework_version(backend: str) -> str:
    try:
        if backend == "vllm":
            import vllm

            return getattr(vllm, "__version__", "unknown")
        if backend == "sglang":
            import sglang

            return getattr(sglang, "__version__", "unknown")
        if backend == "trtllm":
            import tensorrt_llm

            return getattr(tensorrt_llm, "__version__", "unknown")
        if backend.startswith("nitrogen"):
            import nitrogen

            return getattr(nitrogen, "__version__", "unknown")
    except ImportError:
        return "not-installed"
    return "unknown"


def _load_host_metadata(out_dir: Path) -> dict[str, Any]:
    """Read `<out_dir>/host_<hostname>.json` written by `scripts/gpu_probe.sh`.

    Returns the parsed dict, or `{}` if the file is missing/unreadable. The
    GPU YAML carries placeholder values like `driver: "unknown"` — host probe
    is the source of truth for what's actually running.
    """
    host_file = out_dir / f"host_{socket.gethostname()}.json"
    if not host_file.is_file():
        return {}
    try:
        return json.loads(host_file.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _apply_round_to_cfg(cfg: PipelineConfig, round_: Round) -> None:
    """Stamp the round's base_url and HF id onto the matching backend config."""
    if round_.backend == "vllm":
        cfg.vllm.base_url = round_.base_url
        cfg.vllm.model = round_.hf_id
    elif round_.backend == "sglang":
        cfg.sglang.base_url = round_.base_url
        cfg.sglang.model = round_.hf_id
    elif round_.backend == "trtllm":
        cfg.trtllm.base_url = round_.base_url
        cfg.trtllm.model = round_.hf_id
    elif round_.backend.startswith("nitrogen"):
        cfg.nitrogen.base_url = round_.base_url
        cfg.nitrogen.model_id = round_.ckpt or round_.hf_id
        # Pin the denoising seed from the round's launch args so accuracy deltas
        # across precisions reflect precision, not sampling noise.
        from benchmarks.nitrogen_exec import parse_exec_plan

        cfg.nitrogen.seed = parse_exec_plan(round_.launch_args).seed


# Framework defaults applied when no explicit launch flag overrides them.
# vllm and sglang both ship chunked_prefill on / cuda-graphs on (eager off).
# trtllm-serve has no equivalent CLI knobs at this layer; leave None.
_FW_FLAG_DEFAULTS: dict[str, dict[str, bool | None]] = {
    "vllm":   {"chunked_prefill": True,  "enforce_eager": False},
    "sglang": {"chunked_prefill": True,  "enforce_eager": False},
    "trtllm": {"chunked_prefill": None,  "enforce_eager": None},
}


def _detect_server_flags(
    backend: str, launch_args: list[str]
) -> tuple[bool | None, bool | None]:
    """Read chunked-prefill and enforce-eager state from launch flags.

    Returns (chunked_prefill, enforce_eager). When the launch args don't
    mention either flag, falls back to the framework default — both vllm
    and sglang ship with chunked-prefill on and cuda-graphs on, so a run
    with no overrides is "on, off" rather than unknown.

    Recognized flags:
      vllm:   --enable-chunked-prefill / --no-enable-chunked-prefill,
              --enforce-eager / --no-enforce-eager
      sglang: --chunked-prefill-size N (>0 → on, <=0 → off),
              --disable-cuda-graph (→ enforce_eager=True)
    """
    joined = " ".join(launch_args)
    defaults = _FW_FLAG_DEFAULTS.get(backend, {})
    chunked: bool | None = defaults.get("chunked_prefill")
    eager: bool | None = defaults.get("enforce_eager")

    if "--no-enable-chunked-prefill" in joined or "--disable-chunked-prefill" in joined:
        chunked = False
    elif "--enable-chunked-prefill" in joined:
        chunked = True

    if backend == "sglang":
        m = re.search(r"--chunked-prefill-size[= ]\s*(-?\d+)", joined)
        if m:
            chunked = int(m.group(1)) > 0

    if "--enforce-eager" in joined or "enforce_eager=true" in joined.lower():
        eager = True
    elif "--no-enforce-eager" in joined:
        eager = False

    if backend == "sglang" and "--disable-cuda-graph" in joined:
        eager = True

    return chunked, eager


def _run_one(pipe: Pipeline, sc, samples: LatencySamples) -> tuple[Any, bool, bool]:
    """Run one scenario, append samples, return (resp, is_valid, was_executed)."""
    resp = pipe.run(sc.pipeline_request())

    if resp.latency.total_ms is not None:
        samples.end_to_end.append(resp.latency.total_ms)
    if resp.latency.reasoner_ttft_ms is not None:
        samples.ttft.append(resp.latency.reasoner_ttft_ms)
    if resp.latency.vision_encoder_ms is not None:
        samples.vision_encoder.append(resp.latency.vision_encoder_ms)

    if resp.model_meta and resp.model_meta.extras:
        pt = resp.model_meta.extras.get("prompt_tokens")
        ct = resp.model_meta.extras.get("completion_tokens")
        if isinstance(pt, int):
            samples.prompt_tokens.append(pt)
        if isinstance(ct, int):
            samples.completion_tokens.append(ct)

    is_valid = resp.validation.schema_valid and resp.validation.safe
    return resp, is_valid, bool(resp.was_executed)


def _run_round(
    *,
    round_: Round,
    gpu: str,
    gpu_cfg: dict[str, Any],
    label: str,
    out_dir: Path,
    scenarios_dir: Path | None,
    warmup_requests: int,
    gpu_index: int,
    sampler_interval_ms: int,
) -> None:
    """Execute one (backend, model, variant) round end-to-end."""
    pipeline_cfg = PipelineConfig.from_env()
    _apply_round_to_cfg(pipeline_cfg, round_)
    reasoner = _make_reasoner(round_.backend, pipeline_cfg)
    pipe = Pipeline(reasoner=reasoner, config=pipeline_cfg)

    run_id = uuid.uuid4().hex[:12]
    started_at = utc_now_iso()

    per_scenario_dir = out_dir / gpu / round_.backend
    per_scenario_dir.mkdir(parents=True, exist_ok=True)

    scenarios = load_all(scenarios_dir)
    if not scenarios:
        where = scenarios_dir or "tests/smoke/scenarios/"
        typer.echo(f"no scenarios found under {where}", err=True)
        raise typer.Exit(2)

    if warmup_requests > 0:
        # Cycle through scenarios so each distinct input shape gets a
        # CUDA-graph capture / kernel-autotune pass before the timed loop.
        # Hitting only scenarios[0] N times only warms one shape, leaving
        # later scenarios paying first-touch costs against the clock.
        typer.echo(
            f">> warming up: {warmup_requests} request(s) "
            f"cycling through {len(scenarios)} scenario(s)"
        )
        for i in range(warmup_requests):
            sc = scenarios[i % len(scenarios)]
            try:
                pipe.run(sc.pipeline_request())
            except Exception as e:  # warm-up failures shouldn't kill the run
                typer.echo(f"   warm-up {i+1} ({sc.name}) failed: {e}", err=True)

    samples = LatencySamples()
    accuracies: list[GamepadAccuracy] = []  # policy backends only (NitroGen)
    n_valid = 0
    n_executed = 0
    n_completed = 0

    sampler_summary: dict[str, Any] = {}
    gs_ctx = (
        GpuSampler(gpu_index=gpu_index, interval_ms=sampler_interval_ms)
        if gpu_index >= 0
        else None
    )
    # Poll /metrics during the run so we capture peak gauges
    # (kv_cache_usage_pct, prefix_cache_hit_rate). The post-run scrape
    # below stays for cumulative histograms / counter ratios.
    poller_ctx = (
        PromPoller(base_url=round_.base_url, framework=round_.backend)
        if round_.backend in ("vllm", "sglang", "trtllm")
        else None
    )

    t_loop_start = time.perf_counter()
    if gs_ctx is not None:
        gs_ctx.__enter__()
    if poller_ctx is not None:
        poller_ctx.__enter__()
    try:
        for sc in scenarios:
            resp, is_valid, was_exec = _run_one(pipe, sc, samples)
            n_completed += 1
            n_valid += int(is_valid)
            n_executed += int(was_exec)

            extras = (resp.model_meta.extras if resp.model_meta else {}) or {}

            # Accuracy-vs-gold for policy backends: compare the raw predicted
            # gamepad (extras["gamepad"]) against the scenario's gold_action.json
            # sidecar. Skipped for text-VLM scenarios (no sidecar / no gamepad).
            scenario_acc = None
            if sc.gold_action is not None and isinstance(extras.get("gamepad"), dict):
                scenario_acc = compare_gamepad(extras["gamepad"], sc.gold_action)
                accuracies.append(scenario_acc)

            # Per-scenario row split into two top-level groups:
            #   - configs: fixture data + runner config (deterministic across runs)
            #   - results: this request's outputs and measurements
            row: dict[str, Any] = {
                "configs": {
                    "scenario": sc.name,
                    "framework": round_.backend,
                    "model": round_.model_id,
                    "hf_id": round_.hf_id,
                    "variant": round_.variant,
                    "run_label": label,
                    "instruction": sc.spec.instruction,
                    # Only present for VLM scenarios (those that ship expected.json).
                    # Policy scenarios grade via the gold_action.json sidecar.
                    "actions_gold": (
                        sc.expected.actions.model_dump() if sc.expected is not None else None
                    ),
                    "model_meta": {
                        "framework": resp.model_meta.framework if resp.model_meta else None,
                        "model_id": resp.model_meta.model_id if resp.model_meta else None,
                        "quantization": resp.model_meta.quantization if resp.model_meta else None,
                    },
                },
                "results": {
                    "run_id": run_id,
                    "latency_ms": resp.latency.model_dump(),
                    "prompt_tokens": extras.get("prompt_tokens"),
                    "completion_tokens": extras.get("completion_tokens"),
                    "actions_actual": resp.actions.model_dump() if resp.actions else None,
                    "validation": resp.validation.model_dump(),
                    "was_executed": resp.was_executed,
                    "error": resp.error,
                    "gamepad": extras.get("gamepad"),  # raw policy output (NitroGen)
                    "accuracy_vs_gold": scenario_acc.to_dict() if scenario_acc else None,
                },
            }
            (per_scenario_dir / f"{sc.name}__{run_id}.json").write_text(
                json.dumps(row, indent=2)
            )
            typer.echo(
                f"  {sc.name}: total_ms={resp.latency.total_ms} valid={is_valid} "
                f"prompt_toks={extras.get('prompt_tokens')} "
                f"completion_toks={extras.get('completion_tokens')}"
            )
    finally:
        if poller_ctx is not None:
            poller_ctx.__exit__(None, None, None)
        if gs_ctx is not None:
            gs_ctx.__exit__(None, None, None)
            sampler_summary = gs_ctx.summary
    wall_time_s = time.perf_counter() - t_loop_start

    n = len(scenarios)
    derive_itl(samples)
    pct = summarise_latencies(samples)
    toks = summarise_token_counts(samples)
    tput = compute_throughput(
        samples,
        n_completed=n_completed,
        n_valid=n_valid,
        wall_time_s=wall_time_s,
    )

    chunked_prefill, enforce_eager = _detect_server_flags(
        round_.backend, round_.launch_args
    )
    notes = [
        f"scenario-mode run over {n} scenarios from {scenarios_dir or 'tests/smoke/scenarios'}",
        f"warmup_requests={warmup_requests} (discarded before timing)",
        "command_success_rate counts DryRunExecutor accepts (≈ validity unless executor is wired)",
    ]

    # Phase 2: scrape /metrics once at the end (cumulative histograms +
    # counter ratios), then merge in the in-run gauge peaks captured by
    # PromPoller. For trtllm, queue_time + kv_cache usage only fill in if
    # the server was launched with `return_perf_metrics: true` in
    # --extra_llm_api_options.
    prom_fields: dict[str, Any] = {}
    if round_.backend in ("vllm", "sglang", "trtllm"):
        poller_peaks = poller_ctx.peaks if poller_ctx is not None else {}
        try:
            prom = scrape(round_.base_url, round_.backend)
            # For gauge fields, prefer the run-time peak over the post-run
            # snapshot; fall back to the final scrape if polling caught nothing.
            kv_peak = poller_peaks.get("kv_cache_usage_pct")
            pc_peak = poller_peaks.get("prefix_cache_hit_rate")
            prom_fields = {
                "prefix_cache_hit_rate": (
                    pc_peak if pc_peak is not None else prom.prefix_cache_hit_rate
                ),
                "kv_cache_usage_pct": (
                    kv_peak if kv_peak is not None else prom.kv_cache_usage_pct
                ),
                "prefill_time_p50_ms": prom.prefill_time_p50_ms,
                "prefill_time_p95_ms": prom.prefill_time_p95_ms,
                "prefill_time_p99_ms": prom.prefill_time_p99_ms,
                "decode_time_p50_ms": prom.decode_time_p50_ms,
                "decode_time_p95_ms": prom.decode_time_p95_ms,
                "decode_time_p99_ms": prom.decode_time_p99_ms,
                "queue_time_p50_ms": prom.queue_time_p50_ms,
                "queue_time_p95_ms": prom.queue_time_p95_ms,
                "queue_time_p99_ms": prom.queue_time_p99_ms,
            }
            if round_.backend == "trtllm" and prom.queue_time_p50_ms is None:
                notes.append(
                    "trtllm /prometheus/metrics absent — set "
                    "`return_perf_metrics: true` in --extra_llm_api_options "
                    "to capture queue_time histograms"
                )
        except ScrapeError as e:
            notes.append(f"prometheus scrape failed: {e}")
            # Even when the post-run scrape fails (server already gone),
            # the in-run peaks may still be valid.
            kv_peak = poller_peaks.get("kv_cache_usage_pct")
            pc_peak = poller_peaks.get("prefix_cache_hit_rate")
            if kv_peak is not None or pc_peak is not None:
                prom_fields = {
                    "prefix_cache_hit_rate": pc_peak,
                    "kv_cache_usage_pct": kv_peak,
                }
        if poller_ctx is not None and poller_ctx.n_errors > 0:
            notes.append(
                f"prom poller: {poller_ctx.n_samples} samples, "
                f"{poller_ctx.n_errors} errors (last: {poller_ctx.last_error})"
            )

    # Phase 3: GPU sampler aggregates → BenchmarkResult fields.
    gpu_fields: dict[str, Any] = {}
    if sampler_summary:
        gpu_fields = {
            "sampler_backend": sampler_summary.get("sampler_backend"),
            "sampler_n_samples": sampler_summary.get("n_samples"),
            "mem_bw_util_pct_p50": sampler_summary.get("mem_bw_util_pct_p50"),
            "mem_bw_util_pct_peak": sampler_summary.get("mem_bw_util_pct_peak"),
            "gpu_util_pct_p50": sampler_summary.get("gpu_util_pct_p50"),
            "gpu_util_pct_peak": sampler_summary.get("gpu_util_pct_peak"),
            "fb_used_peak_gb": sampler_summary.get("fb_used_peak_gb"),
            "power_avg_w": sampler_summary.get("power_avg_w"),
            "power_peak_w": sampler_summary.get("power_peak_w"),
        }
        if (sampler_summary.get("sampler_backend") or "none") == "none":
            notes.append(f"gpu sampler unavailable: {sampler_summary.get('sampler_backend')}")
        elif sampler_summary.get("sampler_backend") == "nvidia-smi":
            notes.append("mem_bw_util_pct n/a — DCGM not detected, fell back to nvidia-smi")
        p_avg = sampler_summary.get("power_avg_w")
        if p_avg is not None and wall_time_s and n_completed > 0:
            gpu_fields["energy_per_request_j"] = p_avg * wall_time_s / n_completed

    framework_knobs: dict[str, Any] = {
        "model_id": round_.model_id,
        "hf_id": round_.hf_id,
        "family": round_.family,
        "variant": round_.variant,
        "launch_args": round_.launch_args,
    }
    if round_.trtllm_backend is not None:
        framework_knobs["trtllm_backend"] = round_.trtllm_backend

    # Policy backend (NitroGen): accuracy-vs-gold + execution-plan knobs.
    accuracy_fields: dict[str, Any] = {}
    if round_.transport == "zmq":
        agg = aggregate_accuracy(accuracies)
        plan = parse_exec_plan(round_.launch_args)
        framework_knobs.update(plan.to_knobs())
        accuracy_fields = {
            "action_mse": agg["action_mse"],
            "button_agreement_rate": agg["button_agreement_rate"],
            "joystick_mae": agg["joystick_mae"],
            "denoise_steps": plan.steps,
        }
        if not accuracies:
            notes.append(
                "accuracy-vs-gold skipped: scenarios have no gold_action.json "
                "sidecar (run build_nitrogen_scenarios.py to generate them)"
            )

    aggregate = BenchmarkResult(
        run_id=run_id,
        started_at=started_at,
        framework=round_.backend,
        framework_version=_framework_version(round_.backend),
        gpu=gpu_cfg.get("display_name", gpu),
        driver=gpu_cfg.get("driver", "unknown"),
        cuda=gpu_cfg.get("cuda", "unknown"),
        model=round_.model_id,
        quantization=round_.quantization,
        tensor_parallel=int(gpu_cfg.get("tensor_parallel", 1)),
        concurrency=1,
        n_requests=n,
        framework_knobs=framework_knobs,
        run_label=label,
        warmup_requests=warmup_requests,
        chunked_prefill_enabled=chunked_prefill,
        enforce_eager=enforce_eager,
        wall_time_s=wall_time_s,
        e2e_p50_ms=pct["e2e_p50_ms"],
        e2e_p95_ms=pct["e2e_p95_ms"],
        e2e_p99_ms=pct["e2e_p99_ms"],
        command_success_rate=(n_executed / n) if n else None,
        grammar_validity_rate=(n_valid / n) if n else None,
        ttft_p50_ms=pct["ttft_p50_ms"],
        ttft_p95_ms=pct["ttft_p95_ms"],
        ttft_p99_ms=pct["ttft_p99_ms"],
        itl_p50_ms=pct["itl_p50_ms"],
        itl_p95_ms=pct["itl_p95_ms"],
        itl_p99_ms=pct["itl_p99_ms"],
        vision_encoder_p50_ms=pct["vision_encoder_p50_ms"],
        throughput_seq_per_s=tput["throughput_seq_per_s"],
        goodput_seq_per_s=tput["goodput_seq_per_s"],
        tokens_per_sec_decode=tput["tokens_per_sec_decode"],
        mean_prompt_tokens=toks["mean_prompt_tokens"],
        mean_completion_tokens=toks["mean_completion_tokens"],
        total_prompt_tokens=toks["total_prompt_tokens"],
        total_completion_tokens=toks["total_completion_tokens"],
        notes=notes,
        **prom_fields,
        **gpu_fields,
        **accuracy_fields,
    )

    out_path = out_dir / gpu / f"{round_.backend}-{round_.model_id}-{run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aggregate.to_dict(), indent=2))
    typer.echo(f"wrote {out_path}")


@app.command()
def main(
    gpu: str = typer.Option(..., help="GPU profile name; resolves benchmarks/configs/<gpu>.yaml"),
    backend: str = typer.Option(
        None, help="vllm | sglang | trtllm | nitrogen. Required unless --sweep is given."
    ),
    model: str = typer.Option(
        None,
        "--model",
        help="Model id from `models:`. Defaults to `default_model`.",
    ),
    variant: str = typer.Option(
        None,
        "--variant",
        help="Variant name from `backends.<backend>.variants` (e.g. eager, chunked_off, tp2).",
    ),
    sweep: str = typer.Option(
        None,
        "--sweep",
        help="Sweep name from the yaml's `sweeps:` block. Iterates rounds; one BenchmarkResult per round.",
    ),
    out_dir: Path = typer.Option(Path("benchmarks/results"), help="Where to write rows."),
    scenarios_dir: Path = typer.Option(
        None,
        help="Folder of scenario directories to run. Defaults to tests/smoke/scenarios/.",
    ),
    warmup_requests: int = typer.Option(
        3,
        min=0,
        help=(
            "Run this many requests before timing starts (results discarded). "
            "Cycles through the scenario list deterministically. Default 3 "
            "covers each shape once for the standard 3-scenario smoke set. "
            "Pick N based on the number of *distinct input shapes* in your "
            "dataset, not its size — even huge sets usually have <20 unique "
            "image-resolution / prompt-length combinations. Bump to ~6 for "
            "the trtllm pytorch backend (lazy CUDA-graph capture benefits "
            "from a second warmup pass)."
        ),
    ),
    label: str = typer.Option(
        "",
        help="Free-form tag stamped on every row. Defaults to the variant name (or 'baseline').",
    ),
    gpu_index: int = typer.Option(
        0, help="Which GPU to sample (passed to dcgmi/nvidia-smi). Set -1 to disable sampling."
    ),
    sampler_interval_ms: int = typer.Option(
        250, min=50, help="GPU sampler polling cadence in ms."
    ),
) -> None:
    try:
        gpu_cfg = load_gpu_config(gpu)
    except FileNotFoundError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2)

    # Merge host probe data into gpu_cfg. Host probe (driver, cuda kit
    # version) is the source of truth for what's actually running on this
    # box; the YAML carries "unknown" placeholders for portability.
    host_meta = _load_host_metadata(out_dir)
    if not host_meta:
        typer.echo(
            f">> WARNING: no host metadata found under {out_dir}/host_<hostname>.json. "
            f"Run `bash scripts/gpu_probe.sh` to populate driver/cuda fields.",
            err=True,
        )
    for key in ("driver", "cuda"):
        val = host_meta.get(key)
        if val and val not in ("unknown", "not-installed"):
            gpu_cfg[key] = val

    # Build the list of rounds to run.
    rounds: list[Round]
    if sweep is not None:
        try:
            rounds = list(iter_sweep(gpu_cfg, sweep))
        except ValueError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(2)
        if not rounds:
            typer.echo(f"sweep {sweep!r} has no rounds", err=True)
            raise typer.Exit(2)
        typer.echo(f">> sweep {sweep!r}: {len(rounds)} rounds")
    else:
        if not backend:
            typer.echo("--backend is required when --sweep is not given", err=True)
            raise typer.Exit(2)
        try:
            rounds = [resolve_round(gpu_cfg, backend=backend, model_id=model, variant=variant)]
        except ValueError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(2)

    for i, r in enumerate(rounds, 1):
        round_label = label or r.variant or "baseline"
        typer.echo(
            f"\n>> round {i}/{len(rounds)}: backend={r.backend} model={r.model_id} "
            f"variant={r.variant or '-'} label={round_label}"
        )
        _run_round(
            round_=r,
            gpu=gpu,
            gpu_cfg=gpu_cfg,
            label=round_label,
            out_dir=out_dir,
            scenarios_dir=scenarios_dir,
            warmup_requests=warmup_requests,
            gpu_index=gpu_index,
            sampler_interval_ms=sampler_interval_ms,
        )


if __name__ == "__main__":
    sys.exit(app() or 0)
