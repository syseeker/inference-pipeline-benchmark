"""Co-residency additions to metrics.py — trace rows, achieved rate, schema routing.

These have no pydantic/torch dependency, so they run anywhere.
"""

from __future__ import annotations

from benchmarks.metrics import (
    BenchmarkResult,
    LatencySamples,
    achieved_rps,
    request_rows,
)


def _result(**kw) -> BenchmarkResult:
    base = dict(
        run_id="r1", started_at="2026-07-31T00:00:00Z", framework="vllm",
        framework_version="0.1", gpu="rtx_pro6000", driver="580", cuda="13.0",
        model="qwen2.5-7b", quantization=None, tensor_parallel=1,
        concurrency=1, n_requests=3, framework_knobs={},
    )
    base.update(kw)
    return BenchmarkResult(**base)


# --------------------------------------------------------------------------- #
# request_rows
# --------------------------------------------------------------------------- #


def test_request_rows_does_not_borrow_latency_across_requests():
    """The lists on LatencySamples are not index-aligned: durations are only
    appended when the framework returned a value, epochs on every attempt.
    Zipping positionally would attribute a failed request's window to a
    successful request's latency — an alignment plot that looks convincing
    and is wrong."""
    s = LatencySamples()
    s.start_epoch_ms = [1000.0, 2000.0, 3000.0]
    s.end_epoch_ms = [1500.0, 2400.0, 3600.0]
    s.end_to_end = [500.0, 400.0]   # third attempt errored
    s.ttft = [100.0, 90.0]

    rows = request_rows(s)

    assert len(rows) == 3, "one row per attempt, including failures"
    assert rows[0]["e2e_ms"] == 500.0
    assert rows[1]["e2e_ms"] == 400.0
    assert rows[2]["e2e_ms"] is None, "must be an explicit gap, not a borrowed value"
    assert rows[2]["t_start_ms"] == 3000.0, "failed attempt still occupied the GPU"


def test_request_rows_empty_samples():
    assert request_rows(LatencySamples()) == []


def test_request_rows_carries_token_counts():
    s = LatencySamples()
    s.start_epoch_ms = [1000.0]
    s.end_epoch_ms = [1200.0]
    s.end_to_end = [200.0]
    s.prompt_tokens = [50]
    s.completion_tokens = [32]

    (row,) = request_rows(s)
    assert row["prompt_tokens"] == 50
    assert row["completion_tokens"] == 32


# --------------------------------------------------------------------------- #
# achieved_rps
# --------------------------------------------------------------------------- #


def test_achieved_rps_spans_first_start_to_last_end():
    s = LatencySamples()
    s.start_epoch_ms = [0.0, 1000.0, 2000.0]
    s.end_epoch_ms = [500.0, 1500.0, 4000.0]
    # 3 completions over a 4.0 s span
    assert achieved_rps(s) == 0.75


def test_achieved_rps_none_when_too_few_samples():
    s = LatencySamples()
    s.start_epoch_ms = [1000.0]
    s.end_epoch_ms = [1200.0]
    assert achieved_rps(s) is None, "a single request cannot define a rate"
    assert achieved_rps(LatencySamples()) is None


def test_achieved_rps_none_on_zero_span():
    """Guards the divide-by-zero when every stamp collapses to one instant."""
    s = LatencySamples()
    s.start_epoch_ms = [1000.0, 1000.0]
    s.end_epoch_ms = [1000.0, 1000.0]
    assert achieved_rps(s) is None


# --------------------------------------------------------------------------- #
# BenchmarkResult schema routing
# --------------------------------------------------------------------------- #


def test_offered_load_is_config_and_achieved_is_result():
    """offered_rps is what we asked for; achieved_rps is what happened. The
    whole safe-operating-envelope finding is the comparison between them, so
    they must not both land in the same bucket."""
    d = _result(offered_rps=4.0, achieved_rps=3.2).to_dict()

    assert d["configs"]["offered_rps"] == 4.0
    assert d["results"]["achieved_rps"] == 3.2
    assert "achieved_rps" not in d["configs"]
    assert "offered_rps" not in d["results"]


def test_cotenancy_identity_is_config():
    d = _result(
        colocation_id="mix-llm-cv",
        tenant_name="llm",
        co_tenants=["yolov8-l"],
        n_tenants=2,
        isolation_mode="mps",
        arrival_pattern="poisson",
    ).to_dict()

    cfg = d["configs"]
    assert cfg["colocation_id"] == "mix-llm-cv"
    assert cfg["co_tenants"] == ["yolov8-l"]
    assert cfg["n_tenants"] == 2
    assert cfg["isolation_mode"] == "mps"
    assert cfg["arrival_pattern"] == "poisson"


def test_degradation_ratios_default_to_none_in_results():
    """summary.py fills these from paired runs; the runner never sets them."""
    res = _result().to_dict()["results"]
    for f in (
        "degradation_ratio_e2e_p50",
        "degradation_ratio_e2e_p95",
        "degradation_ratio_ttft_p95",
        "throughput_retention",
        "solo_baseline_run_id",
    ):
        assert f in res and res[f] is None


def test_solo_defaults_are_backwards_compatible():
    """A single-model round must serialise exactly as before, aside from the
    new keys being present-and-None. n_tenants=1 marks a solo baseline."""
    r = _result()
    assert r.n_tenants == 1
    assert r.co_tenants == []
    assert r.colocation_id is None
    assert r.throttle_reasons == []


def test_mutable_defaults_are_not_shared():
    a, b = _result(), _result()
    a.co_tenants.append("leak")
    a.throttle_reasons.append("SwPowerCap")
    assert b.co_tenants == []
    assert b.throttle_reasons == []
