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
            transport="http", output_tokens=None, device=None):
    return Tenant(
        name=name, round=_round(backend, model, port, transport),
        driver=driver, load=LoadSpec(pattern=pattern, rps=rps, output_tokens=output_tokens),
        workload=workload, gpu_memory_utilization=frac, device=device,
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


# ── VRAM pre-flight, per GPU ────────────────────────────────────────────────

def test_preflight_sums_per_gpu_not_per_colocation():
    """Two 0.9 tenants on DIFFERENT cards fit; the old whole-colocation sum
    would have rejected a perfectly valid multi-GPU window."""
    tenants = [_tenant("a", port=8001, frac=0.9, device=0),
               _tenant("b", port=8002, frac=0.9, device=1)]
    assert coloc.preflight_vram(tenants) == []


def test_preflight_flags_oversubscription_on_the_same_gpu():
    tenants = [_tenant("a", port=8001, frac=0.9, device=1),
               _tenant("b", port=8002, frac=0.9, device=1)]
    issues = coloc.preflight_vram(tenants)
    assert any("GPU 1:" in i and "1.80" in i for i in issues)
    assert not any("GPU 0:" in i for i in issues)


def test_preflight_charges_tensor_parallel_tenant_to_every_gpu():
    """gpu_memory_utilization is a fraction of EACH card, so a TP tenant on
    [0, 1] leaves only 0.4 on both — not 0.7 on one."""
    tp = _tenant("tp", port=8001, frac=0.6, device=[0, 1])
    tenants = [tp, _tenant("a", port=8002, frac=0.5, device=0),
               _tenant("b", port=8003, frac=0.5, device=1)]
    issues = coloc.preflight_vram(tenants)
    assert any("GPU 0:" in i and "1.10" in i for i in issues)
    assert any("GPU 1:" in i and "1.10" in i for i in issues)


def test_preflight_still_names_uncapped_tenants_per_gpu():
    """The uncapped warning is about the tenant, not the card, and survives
    the regrouping."""
    tenants = [_tenant("a", port=8001, frac=None, device=2),
               _tenant("b", port=8002, frac=0.45, device=3)]
    issues = coloc.preflight_vram(tenants)
    assert any("no gpu_memory_utilization" in i and "'a'" in i for i in issues)
    assert not any("GPU 2:" in i for i in issues), "0.90 alone still fits"


# ── device placement / CUDA_VISIBLE_DEVICES ─────────────────────────────────

def test_server_env_pins_a_single_device():
    assert coloc.build_server_env(_tenant(device=3)) == {"CUDA_VISIBLE_DEVICES": "3"}


def test_server_env_lists_every_tensor_parallel_device():
    env = coloc.build_server_env(_tenant(device=[2, 0]))
    assert env["CUDA_VISIBLE_DEVICES"] == "0,2", "ascending, so local 0..N-1 map predictably"


def test_server_env_defaults_to_gpu_zero():
    # Every colocation written before placement existed means GPU 0.
    assert coloc.build_server_env(_tenant(device=None)) == {"CUDA_VISIBLE_DEVICES": "0"}


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


def test_server_cmd_injectable_executable():
    # The orchestrator passes a venv-resolved path; the builder must use it.
    cmd = coloc.build_server_cmd(_tenant(backend="vllm", frac=0.45),
                                 vllm_bin="/repo/.venv-vllm/bin/vllm")
    assert cmd[0] == "/repo/.venv-vllm/bin/vllm"


def test_venv_bin_falls_back_to_bare_name():
    # Unknown backend or missing venv ⇒ bare tool name (single-venv hosts).
    assert coloc.venv_bin("nonexistent-backend", "vllm") == "vllm"


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
    assert "--streaming" in cmd                 # required for TTFT/ITL capture
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

def _aiperf_record(start_ns, end_ns, latency_ms, ttft_ms, tokens, *,
                   cancelled=False, phase="profiling"):
    """One AIPerf v0.11 {metadata, metrics} record, matching the live schema."""
    return {
        "metadata": {"request_start_ns": start_ns, "request_end_ns": end_ns,
                     "was_cancelled": cancelled, "benchmark_phase": phase},
        "metrics": {
            "request_latency": {"value": latency_ms, "unit": "ms"},
            "time_to_first_token": {"value": ttft_ms, "unit": "ms"},
            "inter_token_latency": {"value": 5.0, "unit": "ms"},
            "output_sequence_length": {"value": tokens, "unit": "tokens"},
        },
    }


def test_parse_aiperf_records_maps_nested_schema(tmp_path):
    art = tmp_path / "llm.aiperf"
    art.mkdir()
    rows = [
        _aiperf_record(1_000_000_000_000_000, 1_000_000_050_000_000, 50.0, 10.0, 32),
        _aiperf_record(1_000_000_100_000_000, 1_000_000_160_000_000, 60.0, 12.0, 30),
    ]
    (art / "profile_export.jsonl").write_text("\n".join(__import__("json").dumps(r) for r in rows))
    recs = coloc.parse_aiperf_records(art)
    assert len(recs) == 2
    assert recs[0]["t_start_ms"] == 1_000_000_000.0        # 1e15 ns → 1e9 ms
    assert recs[0]["t_end_ms"] == 1_000_000_050.0
    assert recs[0]["e2e_ms"] == 50.0                        # metrics.value already ms
    assert recs[0]["ttft_ms"] == 10.0
    assert recs[0]["itl_ms"] == 5.0
    assert recs[0]["output_tokens"] == 32


def test_parse_aiperf_drops_cancelled_and_warmup(tmp_path):
    art = tmp_path / "llm.aiperf"
    art.mkdir()
    rows = [
        _aiperf_record(1_000_000_000_000_000, 1_000_000_050_000_000, 50.0, 10.0, 32),
        _aiperf_record(1_000_000_100_000_000, 1_000_000_160_000_000, 60.0, 12.0, 30, cancelled=True),
        _aiperf_record(1_000_000_200_000_000, 1_000_000_260_000_000, 55.0, 11.0, 31, phase="warmup"),
    ]
    (art / "profile_export.jsonl").write_text("\n".join(__import__("json").dumps(r) for r in rows))
    recs = coloc.parse_aiperf_records(art)
    assert len(recs) == 1                                   # cancelled + warmup dropped
    assert recs[0]["e2e_ms"] == 50.0


def test_parse_aiperf_non_streaming_has_no_ttft(tmp_path):
    art = tmp_path / "llm.aiperf"
    art.mkdir()
    rec = {"metadata": {"request_start_ns": 1_000_000_000_000_000,
                        "request_end_ns": 1_000_000_050_000_000,
                        "was_cancelled": False, "benchmark_phase": "profiling"},
           "metrics": {"request_latency": {"value": 50.0, "unit": "ms"},
                       "output_sequence_length": {"value": 32, "unit": "tokens"}}}
    (art / "profile_export.jsonl").write_text(__import__("json").dumps(rec))
    recs = coloc.parse_aiperf_records(art)
    assert recs[0]["ttft_ms"] is None                       # no --streaming ⇒ no TTFT
    assert recs[0]["e2e_ms"] == 50.0


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


def test_solo_key_separates_tenants_on_different_devices():
    """A GPU-0 baseline is not a baseline for a tenant pinned to GPU 1."""
    assert coloc._solo_key(_tenant(device=0)) != coloc._solo_key(_tenant(device=1))
    assert coloc._solo_key(_tenant(device=None)) == coloc._solo_key(_tenant(device=0))


def test_solo_key_separates_tensor_parallel_widths():
    """TP-2 is a different deployment from TP-1, not the same run on a card."""
    assert coloc._solo_key(_tenant(device=[0, 1])) != coloc._solo_key(_tenant(device=0))


def test_solo_cache_ignores_non_solo():
    cache = coloc.SoloBaselineCache()
    assert cache.seen(_coloc(is_solo=False)) is None


# ── perf_analyzer CSV parsing ────────────────────────────────────────────────

_PA_HEADER = (
    "Inferences/Second,Client Send,Network+Server Send/Recv,Server Queue,"
    "Server Compute Input,Server Compute Infer,Server Compute Output,Client Recv,"
    "p50 latency,p90 latency,p95 latency,p99 latency,Avg latency,request_count"
)
_PA_ROW = "47.2,0,100,50,10,5000,10,0,5200,6500,7000,8000,5500,472"


def test_parse_perf_analyzer_records(tmp_path):
    art = tmp_path / "cv.aiperf"
    art.mkdir()
    (art / "perf_analyzer.csv").write_text(f"{_PA_HEADER}\n{_PA_ROW}\n")
    recs = coloc.parse_perf_analyzer_records(art)
    assert len(recs) == 1
    assert recs[0]["measured_rps"] == 47.2
    assert abs(recs[0]["e2e_ms"] - 5.5) < 0.01       # 5500 us → 5.5 ms
    assert abs(recs[0]["p50_ms"] - 5.2) < 0.01        # 5200 us → 5.2 ms
    assert abs(recs[0]["p95_ms"] - 7.0) < 0.01        # 7000 us → 7.0 ms
    assert abs(recs[0]["p99_ms"] - 8.0) < 0.01        # 8000 us → 8.0 ms
    assert recs[0]["ok"] is True


def test_parse_perf_analyzer_records_uses_last_data_row(tmp_path):
    # Some perf_analyzer versions emit a trailing status line; we take lines[-1].
    art = tmp_path / "cv.aiperf"
    art.mkdir()
    (art / "perf_analyzer.csv").write_text(
        f"{_PA_HEADER}\n47.2,0,100,50,10,5000,10,0,5200,6500,7000,8000,5500,472\n"
        f"50.0,0,100,50,10,5000,10,0,5300,6600,7100,8100,5600,500\n"
    )
    recs = coloc.parse_perf_analyzer_records(art)
    assert recs[0]["measured_rps"] == 50.0             # last row wins


def test_parse_perf_analyzer_records_empty_when_absent(tmp_path):
    assert coloc.parse_perf_analyzer_records(tmp_path) == []


def test_parse_perf_analyzer_records_empty_on_header_only(tmp_path):
    art = tmp_path / "cv.aiperf"
    art.mkdir()
    (art / "perf_analyzer.csv").write_text(_PA_HEADER + "\n")
    assert coloc.parse_perf_analyzer_records(art) == []


def test_achieved_rps_uses_measured_rps():
    recs = [{"measured_rps": 47.2, "e2e_ms": 5.5, "ok": True}]
    assert coloc.achieved_rps(recs) == 47.2


def test_achieved_rps_falls_back_to_timestamps_when_no_measured_rps():
    # Standard aiperf records have no measured_rps — existing path unchanged.
    recs = [
        {"t_start_ms": 0.0, "t_end_ms": 50.0},
        {"t_start_ms": 100.0, "t_end_ms": 150.0},
        {"t_start_ms": 150.0, "t_end_ms": 200.0},
    ]
    r = coloc.achieved_rps(recs)
    assert r is not None and 14.0 < r < 16.0


# ── RunPaths.triton_repo_root ─────────────────────────────────────────────────

def test_run_paths_triton_repo_root(tmp_path):
    paths = coloc.RunPaths(root=tmp_path / "coloc" / "run-1")
    # Two levels up from coloc/run-1/ → tmp_path/, then triton_repo/.
    assert paths.triton_repo_root == tmp_path / "triton_repo"


# ── build_perf_analyzer_cmd additions ────────────────────────────────────────

def _cv_tenant(**kw):
    defaults = dict(name="cv", backend="triton", transport="triton",
                    driver="perf_analyzer", rps=50.0, frac=None,
                    workload="cv_detect_default")
    defaults.update(kw)
    return _tenant(**defaults)


def test_build_perf_analyzer_writes_csv_flag(tmp_path):
    cmd = coloc.build_perf_analyzer_cmd(
        model="yolov8-l", url="localhost:8100", tenant=_cv_tenant(),
        duration_s=120, output_csv=tmp_path / "perf_analyzer.csv",
    )
    assert "-f" in cmd
    assert str(tmp_path / "perf_analyzer.csv") in cmd


def test_build_perf_analyzer_no_csv_flag_without_path():
    cmd = coloc.build_perf_analyzer_cmd(
        model="yolov8-l", url="localhost:8100", tenant=_cv_tenant(), duration_s=120,
    )
    assert "-f" not in cmd


def test_build_perf_analyzer_strips_http_scheme():
    cmd = coloc.build_perf_analyzer_cmd(
        model="yolov8-l", url="http://localhost:8100", tenant=_cv_tenant(), duration_s=120,
    )
    u = cmd[cmd.index("-u") + 1]
    assert u == "localhost:8100"


def test_build_perf_analyzer_strips_https_scheme():
    cmd = coloc.build_perf_analyzer_cmd(
        model="yolov8-l", url="https://localhost:8100", tenant=_cv_tenant(), duration_s=120,
    )
    assert cmd[cmd.index("-u") + 1] == "localhost:8100"


def test_build_perf_analyzer_passthrough_bare_url():
    cmd = coloc.build_perf_analyzer_cmd(
        model="yolov8-l", url="localhost:8100", tenant=_cv_tenant(), duration_s=120,
    )
    assert cmd[cmd.index("-u") + 1] == "localhost:8100"
