"""Server-side metrics scraper.

Single-shot pull from the model server's metrics endpoint(s), populate a
`PromMetrics`. Missing fields stay None — different framework versions
emit different subsets and we don't want a brittle dependency.

Histogram percentiles are interpolated from `_bucket{le=...}` cumulative
counts. This is approximate (resolution = bucket boundaries) but matches
how Grafana / promtool compute them. For our small N it's fine; if you
need sub-bucket precision, capture per-request server-side timings via a
trace endpoint instead.

Per-framework endpoint shapes:
  - vllm/sglang: `/metrics` is Prometheus text. Single GET.
  - trtllm: `/prometheus/metrics` is Prometheus text but only mounted
    when `return_perf_metrics: true` is in --extra_llm_api_options;
    `/metrics` is iteration-stats JSON (always available, gives kv cache
    usage). We hit both and merge.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable

from benchmarks.metrics import PromMetrics


# --------- text-format parser ---------------------------------------------------

_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


@dataclass
class _Series:
    """One line in Prometheus text format: metric{labels} value."""
    name: str
    labels: dict[str, str]
    value: float


def parse_prom_text(text: str) -> list[_Series]:
    """Parse Prometheus exposition format. Skips # comment lines.

    Tolerates the things real exporters do: timestamps, NaN, +Inf.
    """
    out: list[_Series] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # name{labels} value [timestamp]
        # name value [timestamp]
        if "{" in line:
            head, rest = line.split("{", 1)
            label_str, _, value_str = rest.partition("} ")
            name = head
            labels = dict(_LABEL_RE.findall(label_str))
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            name, value_str = parts[0], parts[1]
            labels = {}
        # Drop optional trailing timestamp.
        value_token = value_str.split()[0] if value_str else ""
        try:
            value = float(value_token)
        except ValueError:
            continue
        out.append(_Series(name=name, labels=labels, value=value))
    return out


# --------- helpers --------------------------------------------------------------

def _sum_total(series: Iterable[_Series], name: str) -> float | None:
    """Sum a counter across all label permutations (e.g. per-engine)."""
    vs = [s.value for s in series if s.name == name]
    return sum(vs) if vs else None


def _hist_percentile(series: list[_Series], name: str, p: float) -> float | None:
    """Linear-interpolate a percentile from `<name>_bucket{le=...}` counts.

    Returns the bucket boundary (in seconds, since vLLM/SGLang publish
    times in seconds). Caller multiplies by 1000 to get ms.
    """
    buckets: list[tuple[float, float]] = []  # (le, cumulative_count)
    for s in series:
        if s.name != f"{name}_bucket":
            continue
        le = s.labels.get("le")
        if le is None:
            continue
        try:
            le_f = float("inf") if le == "+Inf" else float(le)
        except ValueError:
            continue
        buckets.append((le_f, s.value))
    if not buckets:
        return None
    buckets.sort(key=lambda x: x[0])
    total = buckets[-1][1]
    if total <= 0:
        return None
    target = total * p
    prev_le, prev_count = 0.0, 0.0
    for le, count in buckets:
        if count >= target:
            if le == float("inf"):
                return prev_le if prev_count > 0 else None
            if count == prev_count:
                return le
            frac = (target - prev_count) / (count - prev_count)
            return prev_le + frac * (le - prev_le)
        prev_le, prev_count = le, count
    return buckets[-1][0]


def _last_value(series: list[_Series], name: str) -> float | None:
    """For gauges with a single label-set we just take the value; if
    multiple, take the max (matches `kv_cache_usage_perc` which is per-engine)."""
    vs = [s.value for s in series if s.name == name]
    return max(vs) if vs else None


# --------- per-framework field maps --------------------------------------------

# Map: BenchmarkResult-flavoured field -> (metric_name, kind)
# kind ∈ {"hist", "gauge_max", "ratio:hits/queries"}
_VLLM_FIELDS = {
    "prefill_time": ("vllm:request_prefill_time_seconds", "hist"),
    "decode_time": ("vllm:request_decode_time_seconds", "hist"),
    "queue_time": ("vllm:request_queue_time_seconds", "hist"),
    "kv_cache_usage_pct": ("vllm:gpu_cache_usage_perc", "gauge_max"),
    "prefix_cache_hit_rate": (
        ("vllm:gpu_prefix_cache_hits_total", "vllm:gpu_prefix_cache_queries_total"),
        "ratio",
    ),
}

_SGLANG_FIELDS = {
    # SGLang's metric names have shifted across versions; we try the most common
    # ones and silently skip the rest.
    "prefill_time": ("sglang:func_latency_seconds_prefill", "hist"),
    "decode_time": ("sglang:func_latency_seconds_decode", "hist"),
    "queue_time": ("sglang:request_queue_time_seconds", "hist"),
    "kv_cache_usage_pct": ("sglang:token_usage", "gauge_max"),
    "prefix_cache_hit_rate": ("sglang:cache_hit_rate", "gauge_max"),
}

# trtllm-serve exposes TTFT and TPOT histograms (no separate prefill/decode
# series the way vLLM does). We map only `queue_time` from the prometheus
# endpoint; kv_cache usage comes from the JSON iteration-stats endpoint
# (see _apply_trtllm_iteration_stats below). prefix_cache_hit_rate stays
# None for VLM runs — kv-block reuse must be disabled for multimodal.
_TRTLLM_FIELDS = {
    "queue_time": ("trtllm_request_queue_time_seconds", "hist"),
}


def _apply(series: list[_Series], fields: dict, out: PromMetrics) -> None:
    for key, (name, kind) in fields.items():
        if kind == "hist":
            p50 = _hist_percentile(series, name, 0.50)
            p95 = _hist_percentile(series, name, 0.95)
            p99 = _hist_percentile(series, name, 0.99)
            if p50 is not None:
                setattr(out, f"{key}_p50_ms", p50 * 1000.0)
            if p95 is not None:
                setattr(out, f"{key}_p95_ms", p95 * 1000.0)
            if p99 is not None:
                setattr(out, f"{key}_p99_ms", p99 * 1000.0)
        elif kind == "gauge_max":
            v = _last_value(series, name)
            if v is None:
                continue
            # vllm:gpu_cache_usage_perc is 0-1, scale to 0-100.
            # sglang token_usage is 0-1; sglang:cache_hit_rate already 0-1.
            if key == "kv_cache_usage_pct":
                setattr(out, key, v * 100.0 if v <= 1.0 else v)
            elif key == "prefix_cache_hit_rate":
                setattr(out, key, v if v <= 1.0 else v / 100.0)
            else:
                setattr(out, key, v)
        elif kind == "ratio":
            num_name, den_name = name  # type: ignore[misc]
            num = _sum_total(series, num_name)
            den = _sum_total(series, den_name)
            if num is not None and den and den > 0:
                setattr(out, key, num / den)


# --------- trtllm iteration-stats (JSON) ---------------------------------------

def _apply_trtllm_iteration_stats(payload: Any, out: PromMetrics) -> None:
    """Pull KV-cache usage from trtllm-serve's `/metrics` JSON.

    The endpoint returns a list of recent iteration snapshots. We use the
    last one as the steady-state read — this is identical to taking a
    `_last_value` from a Prometheus gauge.
    """
    if not isinstance(payload, list) or not payload:
        return
    last = payload[-1]
    if not isinstance(last, dict):
        return
    # Keys arrive in either snake_case or camelCase depending on TRT-LLM
    # version; accept both rather than guessing wrong.
    kv = last.get("kv_cache_stats") or last.get("kvCacheStats")
    if not isinstance(kv, dict):
        return
    used = kv.get("used_num_blocks", kv.get("usedNumBlocks"))
    total = kv.get("max_num_blocks", kv.get("maxNumBlocks"))
    if isinstance(used, (int, float)) and isinstance(total, (int, float)) and total > 0:
        out.kv_cache_usage_pct = (used / total) * 100.0
    hit_rate = kv.get("cache_hit_rate", kv.get("cacheHitRate"))
    if isinstance(hit_rate, (int, float)):
        out.prefix_cache_hit_rate = hit_rate if hit_rate <= 1.0 else hit_rate / 100.0


# --------- public API ----------------------------------------------------------

class ScrapeError(RuntimeError):
    pass


def _fetch_text(url: str, timeout_s: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout_s) as r:
        return r.read().decode("utf-8", errors="replace")


def scrape(base_url: str, framework: str, *, timeout_s: float = 5.0) -> PromMetrics:
    """Scrape the model server's metrics endpoint(s). `base_url` is the
    OpenAI-style `http://host:port/v1` — we strip the `/v1` to hit the
    metrics paths.

    Raises `ScrapeError` on the framework's *primary* metrics endpoint
    being unreachable. Missing individual fields stay None on the
    returned `PromMetrics`."""
    root = base_url.rstrip("/").removesuffix("/v1")
    out = PromMetrics()

    if framework in ("vllm", "sglang"):
        url = f"{root}/metrics"
        try:
            text = _fetch_text(url, timeout_s)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ScrapeError(f"GET {url}: {e}") from e
        series = parse_prom_text(text)
        if framework == "vllm":
            _apply(series, _VLLM_FIELDS, out)
        else:
            _apply(series, _SGLANG_FIELDS, out)
        return out

    if framework == "trtllm":
        # `/prometheus/metrics` is only mounted with `return_perf_metrics:
        # true` in --extra_llm_api_options. Treat its absence as
        # informational, not fatal.
        prom_url = f"{root}/prometheus/metrics"
        try:
            series = parse_prom_text(_fetch_text(prom_url, timeout_s))
            _apply(series, _TRTLLM_FIELDS, out)
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        # `/metrics` is iteration-stats JSON, always mounted on trtllm-serve.
        json_url = f"{root}/metrics"
        try:
            payload = json.loads(_fetch_text(json_url, timeout_s))
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ScrapeError(f"GET {json_url}: {e}") from e
        except json.JSONDecodeError:
            return out
        _apply_trtllm_iteration_stats(payload, out)
        return out

    # Other frameworks: nothing to scrape; return defaults.
    return out
