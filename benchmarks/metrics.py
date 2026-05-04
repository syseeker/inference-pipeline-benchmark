"""Benchmark result types + percentile helpers.

The fields here mirror docs/metrics.md. Anything beyond this struct is
either a per-framework knob (goes into `framework_knobs`) or is captured
out-of-band (Nsight/DCGM trace alongside `BenchmarkResult.run_id`).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class LatencySamples:
    """All latencies in ms. Empty lists are valid (skipped stage)."""

    ttft: list[float] = field(default_factory=list)
    itl: list[float] = field(default_factory=list)
    vision_encoder: list[float] = field(default_factory=list)
    end_to_end: list[float] = field(default_factory=list)


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

    # Decision metrics
    valid_e2e_p50_ms: float | None = None
    valid_e2e_p95_ms: float | None = None
    valid_e2e_p99_ms: float | None = None
    command_success_rate: float | None = None
    grammar_validity_rate: float | None = None

    # Diagnostics
    ttft_p50_ms: float | None = None
    ttft_p95_ms: float | None = None
    ttft_p99_ms: float | None = None
    itl_p50_ms: float | None = None
    itl_p95_ms: float | None = None
    itl_p99_ms: float | None = None
    vision_encoder_p50_ms: float | None = None
    throughput_seq_per_s: float | None = None
    mem_bw_util_pct: float | None = None  # peak sample
    kv_cache_hit_rate: float | None = None
    cuda_graph_speedup: float | None = None  # latency(eager) / latency(graph)
    quant_accuracy_delta: float | None = None  # baseline_acc - quant_acc

    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def summarise_latencies(samples: LatencySamples) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    out["valid_e2e_p50_ms"] = percentile(samples.end_to_end, 0.50)
    out["valid_e2e_p95_ms"] = percentile(samples.end_to_end, 0.95)
    out["valid_e2e_p99_ms"] = percentile(samples.end_to_end, 0.99)
    out["ttft_p50_ms"] = percentile(samples.ttft, 0.50)
    out["ttft_p95_ms"] = percentile(samples.ttft, 0.95)
    out["ttft_p99_ms"] = percentile(samples.ttft, 0.99)
    out["itl_p50_ms"] = percentile(samples.itl, 0.50)
    out["itl_p95_ms"] = percentile(samples.itl, 0.95)
    out["itl_p99_ms"] = percentile(samples.itl, 0.99)
    out["vision_encoder_p50_ms"] = percentile(samples.vision_encoder, 0.50)
    return out


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
