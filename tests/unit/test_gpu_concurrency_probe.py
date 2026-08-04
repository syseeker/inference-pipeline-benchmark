"""Phase-0 concurrency-gate verdict logic — the pure functions.

The probe's GPU work needs hardware, but its decision logic (overlap
classification, repetition policy, gate combination) is pure and is where a
wrong threshold would silently pass a serialising GPU or publish a throttled
run. Those are unit-tested here without a GPU.

The script lives under scripts/, so we load it by path like the other
script-under-test unit tests (see test_nitrogen_exec.py).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PROBE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "gpu_concurrency_probe.py"
_spec = importlib.util.spec_from_file_location("gpu_concurrency_probe", _PROBE_PATH)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)  # type: ignore[union-attr]


# ── classify_overlap ────────────────────────────────────────────────────────

def test_genuine_overlap_passes():
    # ~2x throughput, ~1x latency, plenty of solo headroom → real overlap.
    verdict, reason = probe.classify_overlap(overlap_ratio=1.95, latency_ratio=1.05, solo_util_pct=55.0)
    assert verdict == "PASS"
    assert "overlap" in reason.lower()


def test_serialising_fails():
    # Flat throughput + doubled latency → tenants time-slicing, not sharing.
    verdict, reason = probe.classify_overlap(overlap_ratio=1.05, latency_ratio=1.95, solo_util_pct=55.0)
    assert verdict == "FAIL"
    assert "serialis" in reason.lower()


def test_saturated_kernel_is_inconclusive():
    # Even a serialised-looking result is inconclusive if solo already saturates:
    # there is no headroom to observe overlap, so the kernel must shrink.
    verdict, reason = probe.classify_overlap(overlap_ratio=1.0, latency_ratio=2.0, solo_util_pct=98.0)
    assert verdict == "INCONCLUSIVE"
    assert "--matrix" in reason


def test_mixed_signal_is_inconclusive():
    verdict, _ = probe.classify_overlap(overlap_ratio=1.35, latency_ratio=1.35, solo_util_pct=50.0)
    assert verdict == "INCONCLUSIVE"


def test_saturation_check_precedes_pass():
    # High throughput ratio but saturated solo util is still inconclusive —
    # saturation is checked first so a fluke can't mask missing headroom.
    verdict, _ = probe.classify_overlap(overlap_ratio=1.8, latency_ratio=1.0, solo_util_pct=95.0)
    assert verdict == "INCONCLUSIVE"


def test_missing_util_does_not_block_pass():
    # No util sample (nvidia-smi absent) shouldn't force INCONCLUSIVE.
    verdict, _ = probe.classify_overlap(overlap_ratio=1.9, latency_ratio=1.05, solo_util_pct=None)
    assert verdict == "PASS"


# ── recommend_reps ──────────────────────────────────────────────────────────

def test_reps_low_variance():
    n, _ = probe.recommend_reps(0.03)
    assert n == 1


def test_reps_moderate_variance():
    n, _ = probe.recommend_reps(0.12)
    assert n == 3


def test_reps_high_variance():
    n, _ = probe.recommend_reps(0.30)
    assert n == 5


def test_reps_boundaries():
    assert probe.recommend_reps(0.05)[0] == 1     # inclusive lower band
    assert probe.recommend_reps(0.15)[0] == 3     # inclusive middle band


# ── evaluate_gate ───────────────────────────────────────────────────────────

def _analysis(overlap_ratio, latency_ratio, util, throttles):
    return {
        "overlap_ratio": overlap_ratio,
        "latency_ratio": latency_ratio,
        "solo_gpu_util_p50": util,
        "throttle_reasons_seen": throttles,
    }


def test_gate_passes_when_overlap_and_clocks_clean():
    v = probe.evaluate_gate(_analysis(1.9, 1.05, 55.0, []))
    assert v["gate"] == "PASS"
    assert v["overlap"] == "PASS"
    assert v["clock_integrity"] == "PASS"


def test_gate_fails_on_throttle_even_with_overlap():
    # Perfect overlap but a power cap fired → the slowdown is power, not
    # contention; the window must be discarded.
    v = probe.evaluate_gate(_analysis(1.9, 1.05, 55.0, ["sw_power_cap"]))
    assert v["clock_integrity"] == "FAIL"
    assert v["gate"] == "FAIL"


def test_gate_fails_on_serialisation():
    v = probe.evaluate_gate(_analysis(1.05, 1.95, 55.0, []))
    assert v["overlap"] == "FAIL"
    assert v["gate"] == "FAIL"


def test_gate_fails_when_inconclusive():
    v = probe.evaluate_gate(_analysis(1.0, 2.0, 98.0, []))
    assert v["overlap"] == "INCONCLUSIVE"
    assert v["gate"] == "FAIL"


def test_non_fatal_throttle_does_not_fail_clock():
    # A reason outside the fatal set (e.g. gpu_idle) must not fail the run.
    v = probe.evaluate_gate(_analysis(1.9, 1.05, 55.0, ["gpu_idle"]))
    assert v["clock_integrity"] == "PASS"


# ── _cov ────────────────────────────────────────────────────────────────────

def test_cov_zero_for_constant_series():
    assert probe._cov([10.0, 10.0, 10.0]) == 0.0


def test_cov_zero_for_single_sample():
    assert probe._cov([42.0]) == 0.0


def test_cov_positive_for_spread():
    assert probe._cov([8.0, 10.0, 12.0]) > 0.0


# ── _detect_isolation ───────────────────────────────────────────────────────
#
# Regression: the fallback used `pgrep -x nvidia-cuda-mps-control`. `-x` matches
# against comm, which the kernel truncates to 15 chars (TASK_COMM_LEN), so the
# 23-char daemon name can never match exactly and a running daemon read as
# absent — labelling a good MPS run isolation="none", the single fact Phase 0
# exists to establish. Verified on a live PRO 6000: `pgrep -f` matched pid
# 33994 while `pgrep -x` exited 1 against the same process.

def test_detect_isolation_matches_full_command_line_not_truncated_comm(monkeypatch):
    seen: list[list[str]] = []

    def fake_run(cmd, **kw):
        seen.append(list(cmd))
        import types
        # Emulate the kernel truncation: an -x probe for the full name finds
        # nothing, because comm is "nvidia-cuda-mps".
        matched = "-f" in cmd
        return types.SimpleNamespace(returncode=0 if matched else 1, stdout="33994\n" if matched else "")

    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    monkeypatch.setattr(probe.shutil, "which", lambda _: "/usr/bin/nvidia-cuda-mps-control")
    monkeypatch.setattr(probe.subprocess, "run", fake_run)

    assert probe._detect_isolation() == "mps"
    assert seen and "-f" in seen[0], "must match the full argv, not the truncated comm"


def test_detect_isolation_reports_none_when_no_daemon(monkeypatch):
    import types
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    monkeypatch.setattr(probe.shutil, "which", lambda _: "/usr/bin/nvidia-cuda-mps-control")
    monkeypatch.setattr(probe.subprocess, "run",
                        lambda cmd, **kw: types.SimpleNamespace(returncode=1, stdout=""))
    assert probe._detect_isolation() == "none"


def test_detect_isolation_trusts_pipe_dir_env_without_probing(monkeypatch):
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
    monkeypatch.setattr(probe.shutil, "which", _boom)
    assert probe._detect_isolation() == "mps"


def _boom(*a, **k):
    raise AssertionError("must not probe when the pipe dir env is set")
