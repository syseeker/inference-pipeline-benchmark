"""Unit tests for §10 contention analysis (summary.py) and align_traces.py.

All pure logic: no GPU, no live servers, no filesystem beyond tmp_path.
"""

from __future__ import annotations

import json
import sys
import types

# summary.py imports typer and yaml; both are installed in the project venv.
# No stubs needed — import directly.

from benchmarks import summary as sm
from scripts import align_traces as at


# ── _compute_trace_stats ─────────────────────────────────────────────────────

def _aiperf_records(e2e_vals, ttft_vals=None):
    """Build synthetic aiperf records."""
    ttft_vals = ttft_vals or [0.0] * len(e2e_vals)
    return [
        {"e2e_ms": e, "ttft_ms": t, "t_start_ms": float(i * 100),
         "t_end_ms": float(i * 100 + e), "ok": True}
        for i, (e, t) in enumerate(zip(e2e_vals, ttft_vals))
    ]


def test_compute_trace_stats_aiperf():
    recs = _aiperf_records([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
                           [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
    s = sm._compute_trace_stats(recs)
    assert s["e2e_p50"] == 60.0   # int(10*50/100)=5 → s[5]=60
    assert s["e2e_p95"] == 100.0  # int(10*95/100)=9 → s[9]=100
    assert s["ttft_p95"] is not None
    assert s["achieved_rps"] is None   # not in aiperf records


def test_compute_trace_stats_empty():
    s = sm._compute_trace_stats([])
    assert all(v is None for v in s.values())


def test_compute_trace_stats_cv_perf_analyzer():
    recs = [{"measured_rps": 47.2, "e2e_ms": 5.5, "p50_ms": 5.2, "p95_ms": 7.0,
             "p99_ms": 8.0, "ok": True}]
    s = sm._compute_trace_stats(recs)
    assert s["e2e_p50"] == 5.2
    assert s["e2e_p95"] == 7.0
    assert s["ttft_p95"] is None      # no TTFT for CV
    assert s["achieved_rps"] == 47.2


# ── _load_coloc_runs ─────────────────────────────────────────────────────────

def _write_manifest(run_dir, coloc_id, is_solo, tenants, achieved=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    achieved = achieved or {}
    manifest = {
        "colocation_id": coloc_id,
        "is_solo": is_solo,
        "run_label": "solo" if is_solo else coloc_id,
        "gpu": "rtx_pro6000",
        "isolation_mode": "mps",
        "duration_s": 120,
        "n_tenants": len(tenants),
        "t0_epoch_ms": 1_700_000_000_000.0,
        "tenants": [
            {
                "name": t["name"],
                "round": {"backend": t["backend"], "model_id": t["model_id"],
                          "hf_id": f"org/{t['model_id']}", "family": "test",
                          "quantization": "fp8", "base_url": "http://localhost:8000/v1",
                          "port": 8000, "launch_args": [], "transport": t.get("transport", "http")},
                "load": {"pattern": "poisson", "rps": t["offered_rps"], "output_tokens": 32},
                "offered_rps": t["offered_rps"],
                "achieved_rps": achieved.get(t["name"]),
                "co_tenants": [x["model_id"] for x in tenants if x["name"] != t["name"]],
                "driver": t.get("driver", "aiperf"),
                "workload": t.get("workload", "llm_short"),
                "gpu_memory_utilization": t.get("cap", 0.45),
                "devices": t.get("devices", [0]),
                "triton_backend": None,
            }
            for t in tenants
        ],
        "devices": [0],
        "gpu_sampler": {"0": {"gpu_util_pct_p50": 72.0, "power_avg_w": 280.0,
                              "fb_used_peak_gb": 38.0}},
        "throttle_reasons": [],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def _write_ndjson(run_dir, tenant_name, records):
    (run_dir / f"{tenant_name}.ndjson").write_text(
        "\n".join(json.dumps(r) for r in records)
    )


def test_load_coloc_runs_empty_when_no_dir(tmp_path):
    assert sm._load_coloc_runs(tmp_path) == []


def test_load_coloc_runs_loads_solo_and_contention(tmp_path):
    gpu_dir = tmp_path
    coloc_dir = gpu_dir / "coloc" / "mix-llm-cv"

    llm_t = {"name": "llm", "backend": "vllm", "model_id": "qwen2.5-7b", "offered_rps": 4.0}
    cv_t = {"name": "cv", "backend": "triton", "model_id": "yolov8-l", "offered_rps": 50.0,
            "transport": "triton", "driver": "perf_analyzer"}

    # Solo LLM baseline
    solo_dir = coloc_dir / "solo-0"
    _write_manifest(solo_dir, "mix-llm-cv", True, [llm_t], achieved={"llm": 4.01})
    _write_ndjson(solo_dir, "llm", _aiperf_records([40.0, 50.0, 60.0, 55.0]))

    # Contention run
    contention_dir = coloc_dir / "mix-llm-cv-0"
    _write_manifest(contention_dir, "mix-llm-cv", False, [llm_t, cv_t],
                    achieved={"llm": 3.8, "cv": 46.0})
    _write_ndjson(contention_dir, "llm", _aiperf_records([55.0, 70.0, 80.0, 65.0]))
    _write_ndjson(contention_dir, "cv",
                  [{"measured_rps": 46.0, "p50_ms": 5.5, "p95_ms": 7.5, "e2e_ms": 6.0, "ok": True}])

    runs = sm._load_coloc_runs(gpu_dir)
    assert len(runs) == 2
    solo = next(r for r in runs if r["manifest"]["is_solo"])
    contention = next(r for r in runs if not r["manifest"]["is_solo"])

    assert solo["tenant_stats"]["llm"]["e2e_p50"] is not None
    assert contention["tenant_stats"]["llm"]["e2e_p95"] is not None
    assert contention["tenant_stats"]["cv"]["achieved_rps"] == 46.0


# ── _build_solo_index ────────────────────────────────────────────────────────

def test_build_solo_index_keyed_correctly(tmp_path):
    gpu_dir = tmp_path
    coloc_dir = gpu_dir / "coloc" / "mix-llm-cv"
    llm_t = {"name": "llm", "backend": "vllm", "model_id": "qwen2.5-7b", "offered_rps": 4.0}
    solo_dir = coloc_dir / "solo-0"
    _write_manifest(solo_dir, "mix-llm-cv", True, [llm_t], achieved={"llm": 4.01})
    _write_ndjson(solo_dir, "llm", _aiperf_records([40.0, 50.0]))
    runs = sm._load_coloc_runs(gpu_dir)
    idx = sm._build_solo_index(runs)
    assert ("vllm", "qwen2.5-7b", "llm_short", "poisson", 4.0, 0.45, (0,)) in idx


def test_solo_index_does_not_collapse_two_caps(tmp_path):
    """Same model, same rate, different VRAM cap — two distinct baselines.

    The cap sets the KV cache size, so these two runs are not interchangeable
    references. Keying without it would let the last one scanned overwrite the
    other, and one contention run would silently be rated against a baseline
    with a different KV cache (docs/contention.md §2b).
    """
    gpu_dir = tmp_path
    tight = {"name": "llm", "backend": "vllm", "model_id": "qwen2.5-7b",
             "offered_rps": 4.0, "cap": 0.35}
    roomy = {**tight, "cap": 0.45}

    for label, tenant in (("solo-tight", tight), ("solo-roomy", roomy)):
        d = gpu_dir / "coloc" / "cross-size-scaling" / label
        _write_manifest(d, "cross-size-scaling", True, [tenant], achieved={"llm": 4.0})
        _write_ndjson(d, "llm", _aiperf_records([40.0, 50.0]))

    idx = sm._build_solo_index(sm._load_coloc_runs(gpu_dir))
    assert len(idx) == 2, "baselines at different caps must not share an index entry"


def test_solo_index_separates_placements(tmp_path):
    """A GPU-0 baseline does not describe a tenant that ran tensor-parallel."""
    gpu_dir = tmp_path
    single = {"name": "llm", "backend": "vllm", "model_id": "qwen2.5-72b",
              "offered_rps": 4.0, "devices": [0]}
    tp2 = {**single, "devices": [0, 1]}

    for label, tenant in (("solo-tp1", single), ("solo-tp2", tp2)):
        d = gpu_dir / "coloc" / "scale-llm-cv" / label
        _write_manifest(d, "scale-llm-cv", True, [tenant], achieved={"llm": 4.0})
        _write_ndjson(d, "llm", _aiperf_records([40.0, 50.0]))

    idx = sm._build_solo_index(sm._load_coloc_runs(gpu_dir))
    assert len(idx) == 2, "baselines at different placements must not share an index entry"


# ── _ratio ───────────────────────────────────────────────────────────────────

def test_ratio_basic():
    assert sm._ratio(110.0, 100.0) == 1.1
    assert sm._ratio(90.0, 100.0) == 0.9


def test_ratio_none_when_missing():
    assert sm._ratio(None, 100.0) is None
    assert sm._ratio(100.0, None) is None
    assert sm._ratio(100.0, 0.0) is None


# ── _degradation_table ───────────────────────────────────────────────────────

def _make_runs(tmp_path, llm_solo_e2e, llm_contention_e2e,
               llm_solo_rps=4.0, llm_achieved_rps=3.8):
    gpu_dir = tmp_path
    coloc_dir = gpu_dir / "coloc" / "mix-llm-cv"
    llm_t = {"name": "llm", "backend": "vllm", "model_id": "qwen2.5-7b", "offered_rps": llm_solo_rps}

    solo_dir = coloc_dir / "solo-0"
    _write_manifest(solo_dir, "mix-llm-cv", True, [llm_t], achieved={"llm": llm_solo_rps})
    _write_ndjson(solo_dir, "llm", _aiperf_records(llm_solo_e2e))

    cv_t = {"name": "cv", "backend": "triton", "model_id": "yolov8-l", "offered_rps": 50.0,
            "transport": "triton", "driver": "perf_analyzer"}
    cont_dir = coloc_dir / "mix-llm-cv-0"
    _write_manifest(cont_dir, "mix-llm-cv", False, [llm_t, cv_t],
                    achieved={"llm": llm_achieved_rps, "cv": 49.0})
    _write_ndjson(cont_dir, "llm", _aiperf_records(llm_contention_e2e))
    _write_ndjson(cont_dir, "cv",
                  [{"measured_rps": 49.0, "p50_ms": 5.5, "p95_ms": 7.5, "e2e_ms": 6.0, "ok": True}])
    return sm._load_coloc_runs(gpu_dir)


def test_degradation_table_shows_degradation(tmp_path):
    # Solo LLM e2e ~ 50ms, contention ~ 75ms → ratio ~ 1.5×
    runs = _make_runs(tmp_path, [50.0] * 20, [75.0] * 20)
    solo_idx = sm._build_solo_index(runs)
    lines = sm._degradation_table(runs, solo_idx)
    table = "\n".join(lines)
    assert "▲" in table   # degradation marker
    assert "qwen2.5-7b" in table


def test_degradation_table_no_contention_message(tmp_path):
    gpu_dir = tmp_path
    coloc_dir = gpu_dir / "coloc" / "mix-llm-cv"
    llm_t = {"name": "llm", "backend": "vllm", "model_id": "qwen2.5-7b", "offered_rps": 4.0}
    solo_dir = coloc_dir / "solo-0"
    _write_manifest(solo_dir, "mix-llm-cv", True, [llm_t])
    _write_ndjson(solo_dir, "llm", _aiperf_records([50.0]))
    runs = sm._load_coloc_runs(gpu_dir)
    solo_idx = sm._build_solo_index(runs)
    lines = sm._degradation_table(runs, solo_idx)
    assert any("No contention" in l for l in lines)


# ── _envelope_section ────────────────────────────────────────────────────────

def test_envelope_detects_crossing(tmp_path):
    # achieved 3.5 vs offered 4.0 = 87.5% retention → crossing
    runs = _make_runs(tmp_path, [50.0] * 4, [75.0] * 4,
                      llm_solo_rps=4.0, llm_achieved_rps=3.5)
    lines = sm._envelope_section(runs)
    assert any("qwen2.5-7b" in l for l in lines)


def test_envelope_no_crossing_when_ok(tmp_path):
    # achieved 3.96 vs offered 4.0 = 99% retention → no crossing
    runs = _make_runs(tmp_path, [50.0] * 4, [52.0] * 4,
                      llm_solo_rps=4.0, llm_achieved_rps=3.96)
    lines = sm._envelope_section(runs)
    assert any("No envelope crossings" in l for l in lines)


# ── align_traces.py ──────────────────────────────────────────────────────────

def _write_align_manifest(run_dir, t0, duration_s, tenants, achieved,
                          devices=(0,), gpu_sampler=None, environment=None, warnings=()):
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "colocation_id": "mix-llm-cv",
        "is_solo": False,
        "gpu": "rtx_pro6000",
        "isolation_mode": "mps",
        "duration_s": duration_s,
        "t0_epoch_ms": t0,
        "tenants": [
            {"name": t["name"],
             "round": {"backend": t["backend"], "model_id": t["model_id"],
                       "transport": t.get("transport", "http")},
             "offered_rps": t["offered_rps"],
             "achieved_rps": achieved.get(t["name"]),
             "co_tenants": [], "driver": t.get("driver", "aiperf"),
             "load": {"pattern": "poisson", "rps": t["offered_rps"]},
             "workload": None, "gpu_memory_utilization": 0.45, "triton_backend": None}
            for t in tenants
        ],
        "devices": list(devices),
        "gpu_sampler": gpu_sampler if gpu_sampler is not None else {
            "0": {"gpu_util_pct_p50": 72.0, "power_avg_w": 280.0,
                  "fb_used_peak_gb": 38.0, "mem_bw_util_pct_p50": None},
        },
        "environment": environment or {},
        "warnings": list(warnings),
        "throttle_reasons": [],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))


def test_align_traces_analyse_text(tmp_path):
    t0 = 1_700_000_000_000.0
    llm_t = {"name": "llm", "backend": "vllm", "model_id": "qwen2.5-7b", "offered_rps": 4.0}
    cv_t = {"name": "cv", "backend": "triton", "model_id": "yolov8-l", "offered_rps": 50.0,
            "transport": "triton", "driver": "perf_analyzer"}
    _write_align_manifest(tmp_path, t0, 120, [llm_t, cv_t], {"llm": 3.99, "cv": 47.2})

    # LLM aiperf records with timestamps
    llm_records = [
        {"t_start_ms": t0 + i * 250, "t_end_ms": t0 + i * 250 + 50,
         "e2e_ms": 50.0, "ttft_ms": 12.0, "ok": True}
        for i in range(20)
    ]
    (tmp_path / "llm.ndjson").write_text("\n".join(json.dumps(r) for r in llm_records))

    # CV perf_analyzer aggregate
    cv_records = [{"measured_rps": 47.2, "p50_ms": 5.2, "p95_ms": 7.0, "e2e_ms": 5.5, "ok": True}]
    (tmp_path / "cv.ndjson").write_text(json.dumps(cv_records[0]))

    result = at.analyse(tmp_path)
    assert "mix-llm-cv" in result
    assert "llm" in result
    assert "cv" in result
    assert "Overlap window" in result


def test_align_traces_analyse_json(tmp_path):
    t0 = 1_700_000_000_000.0
    llm_t = {"name": "llm", "backend": "vllm", "model_id": "qwen2.5-7b", "offered_rps": 4.0}
    _write_align_manifest(tmp_path, t0, 120, [llm_t], {"llm": 3.99})
    llm_records = [
        {"t_start_ms": t0 + i * 250, "t_end_ms": t0 + i * 250 + 50,
         "e2e_ms": 50.0, "ttft_ms": 12.0, "ok": True}
        for i in range(10)
    ]
    (tmp_path / "llm.ndjson").write_text("\n".join(json.dumps(r) for r in llm_records))

    result = at.analyse(tmp_path, json_out=True)
    obj = json.loads(result)
    assert obj["colocation_id"] == "mix-llm-cv"
    assert "tenants" in obj
    assert "overlap_window" in obj


def test_align_traces_no_overlap_when_tenants_non_concurrent(tmp_path):
    t0 = 1_700_000_000_000.0
    duration_ms = 120_000.0
    llm_t = {"name": "llm", "backend": "vllm", "model_id": "qwen2.5-7b", "offered_rps": 4.0}
    cv_t = {"name": "cv", "backend": "vllm", "model_id": "dinov2-base", "offered_rps": 10.0}
    _write_align_manifest(tmp_path, t0, 120, [llm_t, cv_t], {})

    # llm active in first 60s, cv active in last 60s — no overlap
    (tmp_path / "llm.ndjson").write_text(json.dumps(
        {"t_start_ms": t0, "t_end_ms": t0 + 60_000, "e2e_ms": 50.0, "ttft_ms": 10.0, "ok": True}
    ))
    (tmp_path / "cv.ndjson").write_text(json.dumps(
        {"t_start_ms": t0 + 61_000, "t_end_ms": t0 + 120_000, "e2e_ms": 40.0, "ttft_ms": 8.0, "ok": True}
    ))

    result = at.analyse(tmp_path)
    assert "NONE" in result


def test_align_traces_missing_manifest_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        at.analyse(tmp_path)


def _one_tenant_run(tmp_path, t0, **manifest_kw):
    llm_t = {"name": "llm", "backend": "vllm", "model_id": "qwen2.5-7b", "offered_rps": 4.0}
    _write_align_manifest(tmp_path, t0, 120, [llm_t], {"llm": 3.99}, **manifest_kw)
    (tmp_path / "llm.ndjson").write_text("\n".join(json.dumps(
        {"t_start_ms": t0 + i * 250, "t_end_ms": t0 + i * 250 + 50,
         "e2e_ms": 50.0, "ttft_ms": 12.0, "ok": True}) for i in range(10)))


def test_align_traces_reports_each_card_separately(tmp_path):
    # A placement run's whole point is that the two cards differ; one averaged
    # GPU line would erase exactly the asymmetry the run exists to show.
    t0 = 1_700_000_000_000.0
    _one_tenant_run(tmp_path, t0, devices=(0, 1), gpu_sampler={
        "0": {"gpu_util_pct_p50": 91.0, "power_avg_w": 410.0, "fb_used_peak_gb": 60.0},
        "1": {"gpu_util_pct_p50": 12.0, "power_avg_w": 90.0, "fb_used_peak_gb": 4.0},
    })
    result = at.analyse(tmp_path)
    assert "GPU 0: util p50 91%" in result
    assert "GPU 1: util p50 12%" in result


def test_align_traces_json_carries_per_device_sampler(tmp_path):
    t0 = 1_700_000_000_000.0
    _one_tenant_run(tmp_path, t0, devices=(0, 1), gpu_sampler={
        "0": {"gpu_util_pct_p50": 91.0}, "1": {"gpu_util_pct_p50": 12.0},
    })
    obj = json.loads(at.analyse(tmp_path, json_out=True))
    assert obj["devices"] == [0, 1]
    assert obj["gpu_sampler"]["1"]["gpu_util_pct_p50"] == 12.0


def test_align_traces_surfaces_environment_and_warning(tmp_path):
    t0 = 1_700_000_000_000.0
    _one_tenant_run(
        tmp_path, t0,
        environment={"interconnect": {"available": True, "nvlink_detected": False},
                     "mps": {"control_daemon_running": False, "pipe_directory": None,
                             "detected": False}},
        warnings=["no MPS control daemon detected while running 2 tenants"],
    )
    result = at.analyse(tmp_path)
    assert "NVLink absent" in result
    assert "MPS control daemon: not running" in result
    assert "WARNING: no MPS control daemon" in result
    obj = json.loads(at.analyse(tmp_path, json_out=True))
    assert obj["warnings"] and obj["environment"]["interconnect"]["nvlink_detected"] is False


def test_align_traces_handles_missing_sampler_data(tmp_path):
    t0 = 1_700_000_000_000.0
    _one_tenant_run(tmp_path, t0, gpu_sampler={})
    assert "no sampler data" in at.analyse(tmp_path)


def test_align_traces_pct_helper():
    assert at._pct([10, 20, 30, 40, 50], 50) == 30
    assert at._pct([], 95) is None
    assert at._pct([100], 95) == 100
