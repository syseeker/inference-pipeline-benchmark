"""Benchmark result types + percentile helpers.

The fields here mirror docs/metrics.md. Anything beyond this struct is
either a per-framework knob (goes into `framework_knobs`) or is captured
out-of-band (Nsight/DCGM trace alongside `BenchmarkResult.run_id`).

Phase-1 additions: per-request token counts, ITL (derived), goodput,
prefill/decode/queue percentiles (Prometheus-fed in Phase 2), and a
`run_label` so cross-run deltas (eager-vs-graph, bf16-vs-fp8, tp1-vs-tp2,
chunked-prefill-on-vs-off) can be paired in summary.py.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class LatencySamples:
    """All latencies in ms; token counts are absolute. Empty lists mean
    that stage was skipped or the framework didn't expose the value."""

    ttft: list[float] = field(default_factory=list)
    itl: list[float] = field(default_factory=list)            # derived per-request
    vision_encoder: list[float] = field(default_factory=list)
    end_to_end: list[float] = field(default_factory=list)

    # Per-request token counts (None entries dropped before aggregation).
    prompt_tokens: list[int] = field(default_factory=list)
    completion_tokens: list[int] = field(default_factory=list)


@dataclass
class PromMetrics:
    """Server-side metrics scraped from <base_url>/metrics once after the
    scenario loop. Fields are None when the framework doesn't expose
    them or scraping was skipped (currently: trtllm)."""

    prefix_cache_hit_rate: float | None = None      # 0-1
    kv_cache_usage_pct: float | None = None         # 0-100, peak/last sample
    prefill_time_p50_ms: float | None = None
    prefill_time_p95_ms: float | None = None
    prefill_time_p99_ms: float | None = None
    decode_time_p50_ms: float | None = None
    decode_time_p95_ms: float | None = None
    decode_time_p99_ms: float | None = None
    queue_time_p50_ms: float | None = None
    queue_time_p95_ms: float | None = None
    queue_time_p99_ms: float | None = None


@dataclass
class BenchmarkResult:
    run_id: str
    started_at: str  # ISO8601 UTC
    framework: str  # vllm | sglang | trtllm | modelopt | triton | nim
    framework_version: str
    gpu: str
    driver: str
    cuda: str
    model: str
    quantization: str | None
    tensor_parallel: int
    concurrency: int
    n_requests: int
    framework_knobs: dict[str, Any]

    # Run identity / shape
    run_label: str = "baseline"            # baseline | eager | chunked_off | fp8 | tp2 | ...
    warmup_requests: int = 0
    chunked_prefill_enabled: bool | None = None
    enforce_eager: bool | None = None
    wall_time_s: float | None = None       # full timed loop, used for throughput

    # Decision metrics
    e2e_p50_ms: float | None = None    # full Pipeline.run wall time, all completed reqs
    e2e_p95_ms: float | None = None
    e2e_p99_ms: float | None = None
    command_success_rate: float | None = None
    grammar_validity_rate: float | None = None

    # Latency diagnostics (client-side)
    ttft_p50_ms: float | None = None
    ttft_p95_ms: float | None = None
    ttft_p99_ms: float | None = None
    itl_p50_ms: float | None = None
    itl_p95_ms: float | None = None
    itl_p99_ms: float | None = None
    vision_encoder_p50_ms: float | None = None

    # Latency diagnostics (server-side, from /metrics)
    prefill_time_p50_ms: float | None = None
    prefill_time_p95_ms: float | None = None
    prefill_time_p99_ms: float | None = None
    decode_time_p50_ms: float | None = None
    decode_time_p95_ms: float | None = None
    decode_time_p99_ms: float | None = None
    queue_time_p50_ms: float | None = None
    queue_time_p95_ms: float | None = None
    queue_time_p99_ms: float | None = None

    # Throughput / token counts
    throughput_seq_per_s: float | None = None        # all completed requests
    goodput_seq_per_s: float | None = None           # validator-accepted only
    tokens_per_sec_decode: float | None = None       # sum(completion) / sum(e2e - ttft)
    mean_prompt_tokens: float | None = None          # for VLMs this includes vision tokens
    mean_completion_tokens: float | None = None
    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None

    # Cache / scheduling (from /metrics; n/a for trtllm)
    prefix_cache_hit_rate: float | None = None       # 0-1
    kv_cache_usage_pct: float | None = None          # 0-100

    # GPU resource usage (Phase 3 — populated by GpuSampler; n/a if backend is "none")
    sampler_backend: str | None = None               # "dcgm" | "nvidia-smi" | "none"
    sampler_n_samples: int | None = None
    mem_bw_util_pct_p50: float | None = None         # DCGM-only (DCGM_FI_PROF_DRAM_ACTIVE)
    mem_bw_util_pct_peak: float | None = None
    gpu_util_pct_p50: float | None = None
    gpu_util_pct_peak: float | None = None
    fb_used_peak_gb: float | None = None
    power_avg_w: float | None = None
    power_peak_w: float | None = None
    energy_per_request_j: float | None = None        # power_avg_w * wall_time_s / n_completed

    # Policy-backend accuracy-vs-gold (NitroGen). None for text-VLM backends.
    # Per-scenario from ModelMeta.extras["gamepad"] vs gold_action.json, averaged.
    action_mse: float | None = None                  # MSE over sticks + shared buttons
    button_agreement_rate: float | None = None       # 0-1, fraction of 17 buttons matching gold
    joystick_mae: float | None = None                # mean abs error over 4 stick axes
    denoise_steps: int | None = None                 # flow-matching iterations this run

    # Cross-run deltas (Phase 5 — computed by summary.py from paired runs, not the runner)
    cuda_graph_speedup: float | None = None          # latency(eager) / latency(graph)
    quant_accuracy_delta: float | None = None        # baseline_acc - quant_acc
    tp_efficiency: float | None = None               # latency(TP=1) / (2 * latency(TP=2))

    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize as `{"configs": {…}, "results": {…}}`.

        `configs` = environment + runner config (deterministic across runs
        with the same setup). `results` = everything measured this run.
        Anything not explicitly listed in `_CONFIG_FIELDS` is treated as a
        result, so new measurement fields land in the right bucket
        automatically.
        """
        flat = asdict(self)
        configs: dict[str, Any] = {}
        results: dict[str, Any] = {}
        for k, v in flat.items():
            (configs if k in _CONFIG_FIELDS else results)[k] = v
        return {"configs": configs, "results": results}


# Fields whose value is determined by the host environment or runner
# config rather than produced during the timed loop. Everything else on
# `BenchmarkResult` is treated as a result.
_CONFIG_FIELDS: frozenset[str] = frozenset({
    # environment
    "framework", "framework_version", "gpu", "driver", "cuda",
    # config / shape
    "model", "quantization", "tensor_parallel", "concurrency",
    "framework_knobs", "run_label", "warmup_requests",
    "chunked_prefill_enabled", "enforce_eager",
    "n_requests",
})


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def derive_itl(samples: LatencySamples) -> list[float]:
    """ITL per request = (e2e - ttft) / (completion_tokens - 1).

    Skips requests missing any of e2e / ttft / completion_tokens, or
    with < 2 completion tokens (ITL is undefined for single-token
    outputs). Mutates `samples.itl` in-place AND returns it for chaining.
    """
    itl: list[float] = []
    n = min(len(samples.end_to_end), len(samples.ttft), len(samples.completion_tokens))
    for i in range(n):
        e2e = samples.end_to_end[i]
        ttft = samples.ttft[i]
        ct = samples.completion_tokens[i]
        if ct is None or ct < 2 or e2e is None or ttft is None:
            continue
        decode_ms = e2e - ttft
        if decode_ms <= 0:
            continue
        itl.append(decode_ms / (ct - 1))
    samples.itl = itl
    return itl


def summarise_latencies(samples: LatencySamples) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    out["e2e_p50_ms"] = percentile(samples.end_to_end, 0.50)
    out["e2e_p95_ms"] = percentile(samples.end_to_end, 0.95)
    out["e2e_p99_ms"] = percentile(samples.end_to_end, 0.99)
    out["ttft_p50_ms"] = percentile(samples.ttft, 0.50)
    out["ttft_p95_ms"] = percentile(samples.ttft, 0.95)
    out["ttft_p99_ms"] = percentile(samples.ttft, 0.99)
    out["itl_p50_ms"] = percentile(samples.itl, 0.50)
    out["itl_p95_ms"] = percentile(samples.itl, 0.95)
    out["itl_p99_ms"] = percentile(samples.itl, 0.99)
    out["vision_encoder_p50_ms"] = percentile(samples.vision_encoder, 0.50)
    return out


def summarise_token_counts(samples: LatencySamples) -> dict[str, float | int | None]:
    pt = samples.prompt_tokens
    ct = samples.completion_tokens
    return {
        "mean_prompt_tokens": (sum(pt) / len(pt)) if pt else None,
        "mean_completion_tokens": (sum(ct) / len(ct)) if ct else None,
        "total_prompt_tokens": sum(pt) if pt else None,
        "total_completion_tokens": sum(ct) if ct else None,
    }


def compute_throughput(
    samples: LatencySamples,
    *,
    n_completed: int,
    n_valid: int,
    wall_time_s: float | None,
) -> dict[str, float | None]:
    """seq/s and tok/s_decode. tok/s_decode uses sum(decode_time_per_request)
    rather than wall time so it isn't deflated by warm-up/idle gaps."""

    out: dict[str, float | None] = {
        "throughput_seq_per_s": None,
        "goodput_seq_per_s": None,
        "tokens_per_sec_decode": None,
    }
    if wall_time_s and wall_time_s > 0:
        out["throughput_seq_per_s"] = n_completed / wall_time_s
        out["goodput_seq_per_s"] = n_valid / wall_time_s

    # Decode-only tokens/sec from per-request decode windows.
    decode_secs = 0.0
    decode_toks = 0
    n = min(len(samples.end_to_end), len(samples.ttft), len(samples.completion_tokens))
    for i in range(n):
        e2e = samples.end_to_end[i]
        ttft = samples.ttft[i]
        ct = samples.completion_tokens[i]
        if ct is None or ct < 2 or e2e is None or ttft is None:
            continue
        decode_ms = e2e - ttft
        if decode_ms <= 0:
            continue
        decode_secs += decode_ms / 1000.0
        decode_toks += ct - 1  # first token is TTFT, not decode
    if decode_secs > 0:
        out["tokens_per_sec_decode"] = decode_toks / decode_secs
    return out


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
