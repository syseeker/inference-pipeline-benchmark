"""Contention orchestrator (benchmarks/coloc.py) — the pure logic.

The server-launch / aiperf-exec path needs live models; the command builders,
VRAM pre-flight, trace parsing, alignment, manifest and solo-cache are pure and
are where a wrong flag or mis-mapped field silently invalidates a run. Those are
tested here without a GPU or a server.

scenario_config imports typer at module scope; stub it if absent so these run in
a bare environment (mirrors test_colocation_config.py).
"""

from __future__ import annotations

import json
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


def test_server_cmd_tenant_cap_overrides_inherited_space_form():
    t = _tenant(backend="vllm", frac=0.45)
    t.round.launch_args = ["--gpu-memory-utilization", "0.30"]
    cmd = coloc.build_server_cmd(t)
    # The tenant cap is an override, not a fallback — and the stale value's
    # orphaned argument must not survive as a positional.
    assert cmd.count("--gpu-memory-utilization") == 1
    assert "0.30" not in cmd
    assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.45"


def test_server_cmd_tenant_cap_beats_backend_extra_args_default():
    """Regression: `backends.vllm.extra_args` carries
    `--gpu-memory-utilization=0.90`, which every tenant inherits through
    launch_args. The builder used to see the flag was already present and
    skip the tenant's cap, so both tenants launched at 0.90 and the second
    OOMed — while preflight_vram, reading the caps rather than the command,
    called the plan fine."""
    t = _tenant(backend="vllm", frac=0.45)
    t.round.launch_args = ["--gpu-memory-utilization=0.90", "--max-num-seqs=32"]
    cmd = coloc.build_server_cmd(t)

    assert "--gpu-memory-utilization=0.90" not in cmd
    assert cmd.count("--gpu-memory-utilization") == 1
    assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.45"
    assert "--max-num-seqs=32" in cmd, "unrelated backend defaults are untouched"


def test_server_cmd_sglang_cap_beats_inherited_mem_fraction():
    t = _tenant(backend="sglang", frac=0.5)
    t.round.launch_args = ["--mem-fraction-static=0.90"]
    cmd = coloc.build_server_cmd(t)
    assert "--mem-fraction-static=0.90" not in cmd
    assert cmd[cmd.index("--mem-fraction-static") + 1] == "0.5"


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


def test_manifest_keys_sampler_by_device_and_records_devices():
    c = _coloc()
    c.tenants[1].device = 1                      # CV tenant moved to the second card
    m = coloc.build_manifest(
        c, t0_epoch_ms=1.0, gpu="rtx_pro6000",
        sampler_summaries={0: {"gpu_util_pct_p50": 90.0}, 1: {"gpu_util_pct_p50": 10.0}},
    )
    assert m["devices"] == [0, 1]
    assert set(m["gpu_sampler"]) == {"0", "1"}   # JSON keys are strings
    assert m["gpu_sampler"]["1"]["gpu_util_pct_p50"] == 10.0


def test_manifest_single_gpu_sampler_shape_unchanged():
    # The overwhelmingly common case: no `device:` anywhere ⇒ one entry, card 0.
    m = coloc.build_manifest(_coloc(), t0_epoch_ms=1.0, gpu="g",
                             sampler_summaries={0: {"gpu_util_pct_p50": 72.0}})
    assert m["devices"] == [0]
    assert list(m["gpu_sampler"]) == ["0"]


# ── occupied devices (the sampler set) ──────────────────────────────────────

def test_occupied_devices_defaults_to_single_card():
    assert coloc.occupied_devices(_coloc().tenants) == [0]


def test_occupied_devices_unions_http_and_triton_placement():
    tenants = [
        _tenant("llm", port=8001, device=1),
        _cv_tenant(name="cv", port=8100, device=0),
    ]
    assert coloc.occupied_devices(tenants) == [0, 1]


def test_occupied_devices_counts_every_card_of_a_tp_tenant():
    # A tensor-parallel tenant is really running on both cards, so both need
    # telemetry — its second card is not somebody else's spare.
    assert coloc.occupied_devices([_tenant("llm", device=[1, 0])]) == [0, 1]


def test_occupied_devices_dedupes_co_resident_tenants():
    # Two tenants on one card share ONE sampler — the rule that has not changed.
    tenants = [_tenant("a", port=8001, device=1), _tenant("b", port=8002, device=1)]
    assert coloc.occupied_devices(tenants) == [1]


# ── run-time environment capture ────────────────────────────────────────────

def _fake_run_text(monkeypatch, results):
    """Stub the probe subprocess: {argv[0]: (stdout, error)}. No nvidia-smi, no
    pgrep, no GPU — the point is the recorded finding, not the tool."""
    monkeypatch.setattr(coloc, "_run_text",
                        lambda cmd, **kw: results.get(cmd[0], (None, "unstubbed")))


def test_capture_interconnect_detects_nvlink(monkeypatch):
    matrix = "\tGPU0\tGPU1\nGPU0\t X \tNV18\nGPU1\tNV18\t X \n"
    _fake_run_text(monkeypatch, {"nvidia-smi": (matrix, None)})
    got = coloc.capture_interconnect()
    assert got["available"] is True
    assert got["nvlink_detected"] is True
    assert got["topo_matrix"] == matrix          # raw, so a reader can re-derive


def test_capture_interconnect_sees_pcie_only(monkeypatch):
    _fake_run_text(monkeypatch, {"nvidia-smi": ("\tGPU0\tGPU1\nGPU0\t X \tSYS\n", None)})
    assert coloc.capture_interconnect()["nvlink_detected"] is False


def test_capture_interconnect_degrades_when_nvidia_smi_missing(monkeypatch):
    _fake_run_text(monkeypatch, {"nvidia-smi": (None, "nvidia-smi not found")})
    got = coloc.capture_interconnect()
    assert got["available"] is False and got["nvlink_detected"] is None
    assert "not found" in got["error"]


def test_capture_mps_reads_daemon_and_pipe_dir(monkeypatch):
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
    _fake_run_text(monkeypatch, {"pgrep": ("4242\n", None)})
    got = coloc.capture_mps()
    assert got["control_daemon_running"] is True
    assert got["pipe_directory"] == "/tmp/nvidia-mps"
    assert got["detected"] is True


def test_capture_mps_reports_absent_daemon(monkeypatch):
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    # pgrep exits 1 when nothing matched — a real "no daemon", not a failed probe.
    _fake_run_text(monkeypatch, {"pgrep": (None, "pgrep exited 1: ")})
    got = coloc.capture_mps()
    assert got["control_daemon_running"] is False
    assert got["detected"] is False
    assert got["probe_error"] is None


def test_capture_mps_inconclusive_when_pgrep_missing(monkeypatch):
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    _fake_run_text(monkeypatch, {"pgrep": (None, "pgrep not found")})
    got = coloc.capture_mps()
    assert got["control_daemon_running"] is None   # unknown ≠ absent
    assert got["probe_error"] == "pgrep not found"


def _env(mps_detected=True, interconnect_available=True):
    return {
        "interconnect": {"available": interconnect_available, "error": "nvidia-smi not found",
                         "nvlink_detected": False if interconnect_available else None},
        "mps": {"control_daemon_running": mps_detected, "detected": mps_detected},
    }


def test_warns_when_multi_tenant_window_has_no_mps():
    w = coloc.environment_warnings(_coloc(), _env(mps_detected=False))
    assert any("MPS" in x and "time-slice" in x for x in w)


def test_no_mps_warning_for_a_solo_window():
    # A solo baseline has nobody to share with; MPS is irrelevant to it.
    assert coloc.environment_warnings(_coloc(is_solo=True), _env(mps_detected=False)) == []


def test_no_mps_warning_when_daemon_present():
    assert coloc.environment_warnings(_coloc(), _env(mps_detected=True)) == []


def test_warns_about_unknown_topology_only_when_multi_gpu():
    single = coloc.environment_warnings(_coloc(), _env(interconnect_available=False))
    assert single == []                       # one card: NVLink cannot explain anything
    c = _coloc()
    c.tenants[1].device = 1
    multi = coloc.environment_warnings(c, _env(interconnect_available=False))
    assert any("interconnect topology unknown" in x for x in multi)


def test_manifest_carries_environment_and_warning():
    m = coloc.build_manifest(_coloc(), t0_epoch_ms=1.0, gpu="g",
                             environment=_env(mps_detected=False))
    assert m["environment"]["mps"]["detected"] is False
    assert len(m["warnings"]) == 1


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


def test_solo_key_separates_tenants_at_different_vram_caps():
    """The cap sets the KV cache size (§2b). Cache the 0.45 baseline under a
    key that ignores it and the 0.70 window is scored against a run that
    never happened."""
    assert coloc._solo_key(_tenant(frac=0.45)) != coloc._solo_key(_tenant(frac=0.70))


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


# ── Triton multi-GPU placement ───────────────────────────────────────────────
#
# Triton tenants are the one class that cannot be pinned with
# CUDA_VISIBLE_DEVICES: their server is a shared container, so placement lives in
# the container's name, ports, repo and `--gpus` flag. These check that a
# tenant's card determines all four, and — the load-bearing one — that a config
# which never mentions `device:` still gets exactly the single-GPU setup it had
# before placement existed.

def _triton_paths(tmp_path):
    return coloc.RunPaths(root=tmp_path / "coloc" / "run-1")


def _fake_docker(monkeypatch, orch):
    """Record `docker run` argv and pretend the named container is then up.

    No Docker daemon involved: _ensure_server's only observable effects are the
    argv it builds and the name it registers as its launch mutex.
    """
    launched: list[list[str]] = []
    running: set[str] = set()

    def fake_popen(cmd, **kw):
        launched.append(list(cmd))
        if "--name" in cmd:
            running.add(cmd[cmd.index("--name") + 1])
        return types.SimpleNamespace(wait=lambda *a, **k: 0)

    monkeypatch.setattr(coloc.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(orch, "_triton_container_running", lambda name: name in running)
    monkeypatch.setattr(orch, "_triton_ready", lambda port: False)
    return launched


def _launched_named(launched, name):
    return [c for c in launched if "--name" in c and c[c.index("--name") + 1] == name]


def test_triton_device_of_defaults_to_gpu0():
    assert coloc.triton_device_of(_cv_tenant()) == 0
    assert coloc.triton_device_of(_cv_tenant(device=1)) == 1


def test_triton_device_of_rejects_tensor_parallel():
    import pytest
    with pytest.raises(ValueError, match="single-GPU"):
        coloc.triton_device_of(_cv_tenant(name="cv", device=[0, 1]))


def test_run_paths_triton_repo_root_per_device(tmp_path):
    paths = _triton_paths(tmp_path)
    # GPU 0 keeps the historical path — it is also the weight-staging repo.
    assert paths.triton_repo_root_for(0) == paths.triton_repo_root
    assert paths.triton_repo_root_for(None) == tmp_path / "triton_repo"
    assert paths.triton_repo_root_for(1) == tmp_path / "triton_repo-gpu1"
    assert paths.triton_repo_root_for(1) != paths.triton_repo_root_for(0)


def test_triton_tenant_url_follows_the_device_port():
    assert coloc.triton_tenant_url(_cv_tenant(port=8100)) == "localhost:8100"
    assert coloc.triton_tenant_url(_cv_tenant(port=8100, device=1)) == "localhost:8110"


def test_build_triton_repos_writes_only_each_devices_models(tmp_path):
    paths = _triton_paths(tmp_path)
    tenants = [
        _cv_tenant(name="cv", model="yolov8-l", port=8100),
        _cv_tenant(name="ilm", model="dinov2-base", port=8100, device=1),
    ]
    coloc.ColocationOrchestrator(gpu="rtx_pro6000")._build_triton_repos(tenants, paths)

    gpu0 = paths.triton_repo_root_for(0)
    gpu1 = paths.triton_repo_root_for(1)
    assert sorted(p.name for p in gpu0.iterdir()) == ["yolov8-l"]
    assert sorted(p.name for p in gpu1.iterdir()) == ["dinov2-base"]
    # Each repo carries a real config, not just an empty dir.
    assert "yolov8-l" in (gpu0 / "yolov8-l" / "config.pbtxt").read_text()
    assert "dinov2-base" in (gpu1 / "dinov2-base" / "config.pbtxt").read_text()


def test_build_triton_repos_single_gpu_is_the_old_single_repo(tmp_path):
    paths = _triton_paths(tmp_path)
    tenants = [
        _cv_tenant(name="cv", model="yolov8-l", port=8100),
        _cv_tenant(name="ilm", model="dinov2-base", port=8100),
    ]
    coloc.ColocationOrchestrator(gpu="rtx_pro6000")._build_triton_repos(tenants, paths)
    repo = paths.triton_repo_root
    assert sorted(p.name for p in repo.iterdir()) == ["dinov2-base", "yolov8-l"]
    assert not (tmp_path / "triton_repo-gpu1").exists()


def test_build_triton_repos_links_staged_weights_into_other_devices(tmp_path):
    """Weights are exported once into the staging repo; a second card's repo has
    to reach them or Triton will not load the model there."""
    paths = _triton_paths(tmp_path)
    staged = paths.triton_repo_root / "dinov2-base" / "1"
    staged.mkdir(parents=True)
    (staged / "model.plan").write_bytes(b"PLAN")

    t = _cv_tenant(name="ilm", model="dinov2-base", port=8100, device=1)
    coloc.ColocationOrchestrator(gpu="rtx_pro6000")._build_triton_repos([t], paths)

    linked = paths.triton_repo_root_for(1) / "dinov2-base" / "1" / "model.plan"
    assert linked.read_bytes() == b"PLAN"


def test_ensure_server_single_gpu_default_matches_pre_placement_behaviour(tmp_path, monkeypatch):
    """The compatibility assertion: nothing in the existing configs sets
    `device:`, so they must still get the `triton-cv` container on the base
    ports, serving the same repo path."""
    paths = _triton_paths(tmp_path)
    orch = coloc.ColocationOrchestrator(gpu="rtx_pro6000")
    launched = _fake_docker(monkeypatch, orch)

    t = _cv_tenant(name="cv", model="yolov8-l", port=8100)
    handle = orch._ensure_server(t, paths, triton_tenants=[t])

    assert handle.container_name == "triton-cv"
    cmd = launched[0]
    assert cmd[cmd.index("--name") + 1] == "triton-cv"
    assert "--http-port=8100" in cmd
    assert "--grpc-port=8101" in cmd
    assert "--metrics-port=8102" in cmd
    assert f"{paths.triton_repo_root.resolve()}:/models" in cmd
    assert cmd[cmd.index("--gpus") + 1] == "device=0"


def test_ensure_server_shares_one_container_per_device(tmp_path, monkeypatch):
    paths = _triton_paths(tmp_path)
    orch = coloc.ColocationOrchestrator(gpu="rtx_pro6000")
    launched = _fake_docker(monkeypatch, orch)

    tenants = [
        _cv_tenant(name="cv", model="yolov8-l", port=8100),
        _cv_tenant(name="ilm", model="dinov2-base", port=8100),
    ]
    handles = [orch._ensure_server(t, paths, triton_tenants=tenants) for t in tenants]

    assert len(launched) == 1                      # the name is the mutex
    assert handles[1].reused is True
    assert handles[1].container_name is None       # only the launcher stops it
    # One container, both models — and it is told to load exactly those two.
    assert "--load-model=yolov8-l" in launched[0]
    assert "--load-model=dinov2-base" in launched[0]


def test_ensure_server_launches_one_container_per_distinct_device(tmp_path, monkeypatch):
    paths = _triton_paths(tmp_path)
    orch = coloc.ColocationOrchestrator(gpu="rtx_pro6000")
    launched = _fake_docker(monkeypatch, orch)

    tenants = [
        _cv_tenant(name="cv", model="yolov8-l", port=8100),
        _cv_tenant(name="ilm", model="dinov2-base", port=8100, device=1),
    ]
    handles = [orch._ensure_server(t, paths, triton_tenants=tenants) for t in tenants]

    assert [h.container_name for h in handles] == ["triton-cv", "triton-cv-gpu1"]
    assert not any(h.reused for h in handles)
    assert len(launched) == 2

    gpu0, = _launched_named(launched, "triton-cv")
    gpu1, = _launched_named(launched, "triton-cv-gpu1")
    assert gpu0[gpu0.index("--gpus") + 1] == "device=0"
    assert gpu1[gpu1.index("--gpus") + 1] == "device=1"
    # Distinct ports — two containers cannot bind the same ones.
    assert "--http-port=8100" in gpu0 and "--http-port=8110" in gpu1
    assert "--grpc-port=8101" in gpu0 and "--grpc-port=8111" in gpu1
    assert "--metrics-port=8102" in gpu0 and "--metrics-port=8112" in gpu1
    # Distinct repos, each loading only its own model.
    assert f"{paths.triton_repo_root_for(0).resolve()}:/models" in gpu0
    assert f"{paths.triton_repo_root_for(1).resolve()}:/models" in gpu1
    assert "--load-model=yolov8-l" in gpu0 and "--load-model=dinov2-base" not in gpu0
    assert "--load-model=dinov2-base" in gpu1 and "--load-model=yolov8-l" not in gpu1


def test_launch_drivers_target_each_tenants_own_container(tmp_path, monkeypatch):
    paths = _triton_paths(tmp_path)
    orch = coloc.ColocationOrchestrator(gpu="rtx_pro6000")
    launched = _fake_docker(monkeypatch, orch)

    tenants = [
        _cv_tenant(name="cv", model="yolov8-l", port=8100),
        _cv_tenant(name="ilm", model="dinov2-base", port=8100, device=1),
    ]
    c = Colocation(id="mix-cv-ilm", tenants=tenants, duration_s=60,
                   isolation="mps", phase=5, is_solo=False)
    orch._launch_drivers(c, paths)

    urls = [cmd[cmd.index("-u") + 1] for cmd in launched if "-u" in cmd]
    assert urls == ["localhost:8100", "localhost:8110"]


# ── sampler set over a whole window ─────────────────────────────────────────
#
# The live path with every GPU-touching part replaced: no servers, no drivers,
# no dcgmi. What is checked is the thing that silently broke before — which
# cards got a sampler, and that all of them were live for the entire window.


class _FakeSampler:
    """Stands in for GpuSampler, logging its own lifecycle into a shared list."""

    def __init__(self, events, *, gpu_index=0, interval_ms=250):
        self._events = events
        self.gpu_index = gpu_index

    def __enter__(self):
        self._events.append(("start", self.gpu_index))
        return self

    def __exit__(self, *a):
        self._events.append(("stop", self.gpu_index))

    @property
    def summary(self):
        return {"sampler_backend": "fake", "gpu_util_pct_p50": 10.0 * self.gpu_index}


def _run_window(monkeypatch, tmp_path, tenants, environment=None):
    """Run one colocation with everything external stubbed. Returns (manifest,
    events) where events interleaves sampler starts/stops with the driver launch."""
    events: list[tuple] = []
    c = Colocation(id="w", tenants=tenants, duration_s=10, isolation="mps",
                   phase=5, is_solo=len(tenants) == 1)
    orch = coloc.ColocationOrchestrator(gpu="rtx_pro6000")

    monkeypatch.setattr(coloc, "GpuSampler",
                        lambda **kw: _FakeSampler(events, **kw))
    monkeypatch.setattr(coloc, "capture_environment",
                        lambda: environment if environment is not None else _env())
    monkeypatch.setattr(orch, "_build_triton_repos", lambda *a, **k: None)
    monkeypatch.setattr(orch, "_ensure_server",
                        lambda t, paths, **kw: coloc.ServerHandle(tenant=t, proc=None, reused=True))
    monkeypatch.setattr(orch, "_wait_ready", lambda t, *a, **k: None)

    def fake_launch(coloc_obj, paths):
        events.append(("drivers", None))
        return {t.name: types.SimpleNamespace(wait=lambda *a, **k: 0) for t in coloc_obj.tenants}

    monkeypatch.setattr(orch, "_launch_drivers", fake_launch)
    manifest = orch.run(c, coloc.RunPaths(root=tmp_path / "run-1"))
    return manifest, events


def test_run_opens_exactly_one_sampler_on_a_single_gpu_window(tmp_path, monkeypatch):
    # The case that runs most often: unchanged from before placement existed.
    _, events = _run_window(monkeypatch, tmp_path, [_tenant("llm", port=8001)])
    assert [e for e in events if e[0] == "start"] == [("start", 0)]


def test_run_opens_one_sampler_per_occupied_card(tmp_path, monkeypatch):
    tenants = [_tenant("llm", port=8001, device=0), _tenant("vlm", port=8002, device=1)]
    manifest, events = _run_window(monkeypatch, tmp_path, tenants)
    assert sorted(dev for kind, dev in events if kind == "start") == [0, 1]
    assert manifest["devices"] == [0, 1]
    assert set(manifest["gpu_sampler"]) == {"0", "1"}


def test_run_opens_one_sampler_for_two_tenants_on_one_card(tmp_path, monkeypatch):
    # One per CARD, never one per tenant — the original §4.3 rule, intact.
    tenants = [_tenant("a", port=8001, device=1), _tenant("b", port=8002, device=1)]
    _, events = _run_window(monkeypatch, tmp_path, tenants)
    assert [e for e in events if e[0] == "start"] == [("start", 1)]


def test_run_samplers_span_the_whole_window(tmp_path, monkeypatch):
    # Every card must be sampling before the first request and still sampling
    # after the last, or its telemetry describes a different window.
    tenants = [_tenant("llm", port=8001, device=0), _tenant("vlm", port=8002, device=1)]
    _, events = _run_window(monkeypatch, tmp_path, tenants)
    launch = events.index(("drivers", None))
    assert all(i < launch for i, e in enumerate(events) if e[0] == "start")
    assert all(i > launch for i, e in enumerate(events) if e[0] == "stop")


def test_run_writes_environment_and_mps_warning_to_manifest(tmp_path, monkeypatch):
    tenants = [_tenant("a", port=8001), _tenant("b", port=8002)]
    manifest, _ = _run_window(monkeypatch, tmp_path, tenants,
                              environment=_env(mps_detected=False))
    on_disk = json.loads((tmp_path / "run-1" / "manifest.json").read_text())
    assert on_disk["environment"]["mps"]["detected"] is False
    assert any("MPS" in w for w in on_disk["warnings"])
    assert manifest["warnings"] == on_disk["warnings"]
