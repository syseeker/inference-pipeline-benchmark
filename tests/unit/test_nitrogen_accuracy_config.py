"""Phase 3 tests: accuracy-vs-gold scoring + nitrogen config resolution."""

from __future__ import annotations

import pytest

from benchmarks.accuracy import aggregate_accuracy, compare_gamepad
from benchmarks.scenario_config import iter_sweep, load_gpu_config, resolve_round

# ─── accuracy ──────────────────────────────────────────────────────────


def test_perfect_match_scores_zero_error_full_agreement():
    pred = {"buttons": {"south": 1.0}, "j_left": [0.5, -0.5], "j_right": [0.0, 0.0]}
    gold = {"buttons": {"south": 1.0}, "j_left": [0.5, -0.5], "j_right": [0.0, 0.0]}
    acc = compare_gamepad(pred, gold)
    assert acc.joystick_mae == 0.0
    assert acc.button_agreement_rate == 1.0
    assert acc.action_mse == 0.0
    assert acc.n_buttons_scored == 17


def test_button_disagreement_lowers_agreement():
    # gold presses south; pred presses north instead → 2 of 17 buttons differ
    pred = {"buttons": {"north": 1.0}, "j_left": [0, 0], "j_right": [0, 0]}
    gold = {"buttons": {"south": 1.0}, "j_left": [0, 0], "j_right": [0, 0]}
    acc = compare_gamepad(pred, gold)
    assert acc.button_agreement_rate == pytest.approx(15 / 17)
    assert acc.joystick_mae == 0.0


def test_joystick_error_is_mean_absolute_over_four_axes():
    pred = {"buttons": {}, "j_left": [1.0, 1.0], "j_right": [0.0, 0.0]}
    gold = {"buttons": {}, "j_left": [0.0, 0.0], "j_right": [0.0, 0.0]}
    acc = compare_gamepad(pred, gold)
    assert acc.joystick_mae == pytest.approx((1 + 1 + 0 + 0) / 4)


def test_threshold_applied_before_button_comparison():
    # pred 0.6 and gold 1.0 both clear 0.5 → agree
    acc = compare_gamepad({"buttons": {"start": 0.6}}, {"buttons": {"start": 1.0}})
    assert acc.button_agreement_rate == 1.0


def test_aggregate_means_and_empty():
    a = compare_gamepad({"buttons": {"south": 1.0}}, {"buttons": {"south": 1.0}})
    b = compare_gamepad({"buttons": {"north": 1.0}}, {"buttons": {"south": 1.0}})
    agg = aggregate_accuracy([a, b])
    assert agg["button_agreement_rate"] == pytest.approx((1.0 + 15 / 17) / 2)
    assert aggregate_accuracy([]) == {
        "joystick_mae": None,
        "button_agreement_rate": None,
        "action_mse": None,
    }


# ─── config ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("gpu", ["rtx_pro6000", "h200", "rtx5090"])
def test_nitrogen_backend_resolves_per_gpu(gpu):
    cfg = load_gpu_config(gpu)
    r = resolve_round(cfg, backend="nitrogen", model_id="nitrogen-500m-fp8", variant="trt")
    assert r.transport == "zmq"
    assert r.ckpt == "nvidia/NitroGen:ng.pt"
    assert r.base_url.startswith("tcp://")
    assert r.quantization == "fp8"
    # exec flag (variant) + precision/steps (model backend_args) + seed (extra_args)
    assert "--exec=tensorrt" in r.launch_args
    assert "--precision=fp8" in r.launch_args
    assert "--steps=16" in r.launch_args
    assert "--seed=0" in r.launch_args


@pytest.mark.parametrize("gpu", ["rtx_pro6000", "h200", "rtx5090"])
def test_nitrogen_sweep_iterates(gpu):
    cfg = load_gpu_config(gpu)
    rounds = list(iter_sweep(cfg, "nitrogen-backends"))
    assert len(rounds) >= 6
    assert all(r.backend == "nitrogen" and r.transport == "zmq" for r in rounds)
    # The exec plan parses cleanly for every round.
    from benchmarks.nitrogen_exec import parse_exec_plan

    for r in rounds:
        plan = parse_exec_plan(r.launch_args)
        assert plan.steps >= 1


def test_nvfp4_only_on_blackwell():
    # pro6000 + 5090 carry an nvfp4 model; h200 (Hopper) does not.
    assert "nitrogen-500m-nvfp4" in load_gpu_config("rtx_pro6000")["models"]
    assert "nitrogen-500m-nvfp4" in load_gpu_config("rtx5090")["models"]
    assert "nitrogen-500m-nvfp4" not in load_gpu_config("h200")["models"]
