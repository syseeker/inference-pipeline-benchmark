"""Contention orchestrator (benchmarks/coloc.py) — the pure logic.

The server-launch / aiperf-exec path needs live models; the command builders,
VRAM pre-flight, trace parsing, alignment, manifest and solo-cache are pure and
are where a wrong flag or mis-mapped field silently invalidates a run. Those are
tested here without a GPU or a server.

scenario_config imports typer at module scope; stub it if absent so these run in
a bare environment (mirrors test_colocation_config.py).
"""

from __future__ import annotations

import sys
import types

try:  # pragma: no cover
    import typer  # noqa: F401
except ImportError:  # pragma: no cover
    _t = types.ModuleType("typer")
    _t.Typer = lambda *a, **k: types.SimpleNamespace(command=lambda *a, **k: (lambda f: f))
    _t.Option = lambda *a, **k: None
    _t.Exit = SystemExit
    _t.echo = lambda *a, **k: None
    sys.modules["typer"] = _t

from benchmarks import coloc
from benchmarks.scenario_config import Colocation, LoadSpec, Round, Tenant


# ── fixtures ────────────────────────────────────────────────────────────────

def _round(backend="vllm", model="qwen2.5-7b", port=8001, transport="http", trtllm_backend=None):
    return Round(
        backend=backend, model_id=model, hf_id=f"org/{model}", family="qwen2.5",
        quantization="bf16", base_url=f"http://localhost:{port}/v1", port=port,
        launch_args=[], transport=transport, trtllm_backend=trtllm_backend,
    )


def _tenant(name="llm", backend="vllm", model="qwen2.5-7b", port=8001, rps=4.0,
            pattern="poisson", driver="aiperf", frac=0.45, workload="llm_short",
            transport="http", output_tokens=None):
    return Tenant(
        name=name, round=_round(backend, model, port, transport),
        driver=driver, load=LoadSpec(pattern=pattern, rps=rps, output_tokens=output_tokens),
        workload=workload, gpu_memory_utilization=frac,
    )


# ── VRAM pre-flight ─────────────────────────────────────────────────────────

def test_preflight_ok_when_fractions_fit():
    tenants = [_tenant("a", port=8001, frac=0.45), _tenant("b", port=8002, frac=0.45)]
    assert coloc.preflight_vram(tenants) == []


def test_preflight_flags_oversubscription():
    tenants = [_tenant("a", frac=0.6), _tenant("b", frac=0.6)]
    issues = coloc.preflight_vram(tenants)
    assert any("> 1.0" in i for i in issues)


def test_preflight_flags_uncapped_tenant():
    tenants = [_tenant("a", frac=None), _tenant("b", frac=0.45)]
    issues = coloc.preflight_vram(tenants)
    assert any("no gpu_memory_utilization" in i for i in issues)


def test_preflight_ignores_triton_fraction():
    # A Triton CV tenant carries no GPU fraction; it must not trip the sum.
    llm = _tenant("llm", frac=0.9)
    cv = _tenant("cv", backend="triton", transport="triton", driver="perf_analyzer", frac=None)
    assert coloc.preflight_vram([llm, cv]) == []


# ── server command builder ──────────────────────────────────────────────────

def test_server_cmd_vllm_injects_cap():
    cmd = coloc.build_server_cmd(_tenant(backend="vllm", frac=0.45))
    assert cmd[:2] == ["vllm", "serve"]
    assert "--gpu-memory-utilization" in cmd
    assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.45"


def test_server_cmd_vllm_respects_explicit_flag():
    t = _tenant(backend="vllm", frac=0.45)
    t.round.launch_args = ["--gpu-memory-utilization", "0.30"]
    cmd = coloc.build_server_cmd(t)
    # Explicit launch arg wins; we don't append a second one.
    assert cmd.count("--gpu-memory-utilization") == 1
    assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.30"


def test_server_cmd_sglang_uses_mem_fraction_static():
    cmd = coloc.build_server_cmd(_tenant(backend="sglang", frac=0.5))
    assert "sglang.launch_server" in cmd
    assert "--mem-fraction-static" in cmd


def test_server_cmd_triton_is_none():
    t = _tenant(backend="triton", transport="triton", driver="perf_analyzer", frac=None)
    assert coloc.build_server_cmd(t) is None


# ── aiperf command builder (open-loop enforcement) ─────────────────────────

def test_aiperf_is_open_loop(tmp_path):
    t = _tenant(rps=4.0, pattern="poisson")
    cmd = coloc.build_aiperf_cmd(
        base_url="http://localhost:8001/v1", model="org/qwen2.5-7b", tenant=t,
        duration_s=120, artifact_dir=tmp_path,
    )
    assert "--request-rate" in cmd and cmd[cmd.index("--request-rate") + 1] == "4.0"
    assert "--arrival-pattern" in cmd and cmd[cmd.index("--arrival-pattern") + 1] == "poisson"
    assert "--concurrency" not in cmd          # never closed-loop
    # --url must be the server root, /v1 stripped.
    assert cmd[cmd.index("--url") + 1] == "http://localhost:8001"


def test_aiperf_rejects_closed_loop(tmp_path):
    t = _tenant(rps=None, pattern="closed")
    try:
        coloc.build_aiperf_cmd(base_url="http://x/v1", model="m", tenant=t,
                               duration_s=60, artifact_dir=tmp_path)
        assert False, "expected ValueError for closed-loop tenant"
    except ValueError as e:
        assert "open-loop" in str(e)


def test_aiperf_passes_output_tokens(tmp_path):
    t = _tenant(rps=2.0, output_tokens=512)
    cmd = coloc.build_aiperf_cmd(base_url="http://x/v1", model="m", tenant=t,
                                 duration_s=60, artifact_dir=tmp_path)
    assert "--extra-inputs" in cmd
    assert "max_tokens:512" in cmd


# ── perf_analyzer command builder ───────────────────────────────────────────

def test_perf_analyzer_open_loop_rate_range():
    t = _tenant(name="cv", backend="triton", transport="triton", driver="perf_analyzer",
                rps=50.0, frac=None, workload="cv_detect_default")
    cmd = coloc.build_perf_analyzer_cmd(model="yolov8-l", url="localhost:8000", tenant=t, duration_s=120)
    assert cmd[0] == "perf_analyzer"
    assert "--request-rate-range" in cmd
    assert cmd[cmd.index("--request-rate-range") + 1] == "50:50:1"
    assert "--service-kind" in cmd and cmd[cmd.index("--service-kind") + 1] == "triton"


def test_driver_command_dispatch():
    llm = _tenant(driver="aiperf")
    cv = _tenant(name="cv", backend="triton", transport="triton", driver="perf_analyzer",
                 rps=50.0, frac=None)
    import pathlib
    a = coloc.driver_command(llm, base_url="http://x/v1", model="m", duration_s=60,
                             artifact_dir=pathlib.Path("/tmp/x"))
    assert a[0] == "aiperf"
    b = coloc.driver_command(cv, base_url="localhost:8000", model="yolov8-l", duration_s=60,
                             artifact_dir=pathlib.Path("/tmp/y"))
    assert b[0] == "perf_analyzer"


# ── trace parsing / alignment ───────────────────────────────────────────────

def test_parse_aiperf_records_maps_fields(tmp_path):
    art = tmp_path / "llm.aiperf"
    art.mkdir()
    # ns-since-epoch timestamps; latency in ns.
    rows = [
        {"timestamp": 1_000_000_000_000_000, "request_latency": 50_000_000,
         "time_to_first_token": 10_000_000, "output_tokens": 32},
        {"timestamp": 1_000_000_100_000_000, "request_latency": 60_000_000,
         "time_to_first_token": 12_000_000, "output_tokens": 30},
    ]
    (art / "profile_export_aiperf.jsonl").write_text("\n".join(__import__("json").dumps(r) for r in rows))
    recs = coloc.parse_aiperf_records(art)
    assert len(recs) == 2
    assert recs[0]["t_start_ms"] == 1_000_000_000.0        # 1e15 ns → 1e9 ms
    assert recs[0]["e2e_ms"] == 50.0
    assert recs[0]["ttft_ms"] == 10.0
    assert recs[0]["output_tokens"] == 32
    assert recs[0]["t_end_ms"] == 1_000_000_050.0


def test_parse_aiperf_records_empty_when_absent(tmp_path):
    assert coloc.parse_aiperf_records(tmp_path) == []


def test_achieved_rps_from_trace():
    # 3 requests completing across ~200 ms of wall clock ⇒ ~15 rps.
    recs = [
        {"t_start_ms": 0.0, "t_end_ms": 50.0},
        {"t_start_ms": 100.0, "t_end_ms": 150.0},
        {"t_start_ms": 150.0, "t_end_ms": 200.0},
    ]
    r = coloc.achieved_rps(recs)
    assert r is not None and 14.0 < r < 16.0


def test_achieved_rps_none_for_thin_trace():
    assert coloc.achieved_rps([{"t_start_ms": 0.0, "t_end_ms": 5.0}]) is None


def test_union_window_spans_all_tenants():
    traces = {
        "a": [{"t_start_ms": 10.0, "t_end_ms": 60.0}],
        "b": [{"t_start_ms": 5.0, "t_end_ms": 90.0}],
    }
    assert coloc.union_window(traces) == (5.0, 90.0)


def test_union_window_none_when_empty():
    assert coloc.union_window({"a": []}) is None


# ── manifest ────────────────────────────────────────────────────────────────

def _coloc(is_solo=False):
    tenants = [
        _tenant("llm", model="qwen2.5-7b", port=8001, rps=4.0),
        _tenant("cv", model="yolov8-l", port=8002, rps=50.0, backend="triton",
                transport="triton", driver="perf_analyzer", frac=None),
    ]
    if is_solo:
        tenants = tenants[:1]
    return Colocation(id="mix-llm-cv", tenants=tenants, duration_s=120,
                      isolation="mps", phase=3, is_solo=is_solo)


def test_manifest_sorts_co_tenants():
    c = _coloc()
    m = coloc.build_manifest(c, t0_epoch_ms=1234.0, gpu="rtx_pro6000",
                             achieved={"llm": 3.9, "cv": 47.0})
    assert m["colocation_id"] == "mix-llm-cv"
    assert m["n_tenants"] == 2
    llm_row = next(t for t in m["tenants"] if t["name"] == "llm")
    assert llm_row["co_tenants"] == ["yolov8-l"]       # sorted OTHER models
    assert llm_row["offered_rps"] == 4.0
    assert llm_row["achieved_rps"] == 3.9
    assert m["t0_epoch_ms"] == 1234.0
    assert m["isolation_mode"] == "mps"


def test_manifest_solo_has_no_co_tenants():
    m = coloc.build_manifest(_coloc(is_solo=True), t0_epoch_ms=1.0, gpu="g")
    assert m["is_solo"] is True
    assert m["run_label"] == "solo"
    assert m["tenants"][0]["co_tenants"] == []


# ── solo-baseline cache ─────────────────────────────────────────────────────

def test_solo_cache_dedupes_identical_baselines():
    cache = coloc.SoloBaselineCache()
    s1 = _coloc(is_solo=True)
    assert cache.seen(s1) is None
    cache.record(s1, "run-1")
    # Same backend/model/workload/load ⇒ already seen.
    s2 = _coloc(is_solo=True)
    assert cache.seen(s2) == "run-1"


def test_solo_cache_ignores_non_solo():
    cache = coloc.SoloBaselineCache()
    assert cache.seen(_coloc(is_solo=False)) is None
