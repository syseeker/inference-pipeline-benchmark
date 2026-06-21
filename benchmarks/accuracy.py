"""Accuracy-vs-gold for the NitroGen policy backend.

Latency/throughput/GPU metrics tell you how *fast* an execution backend is;
this tells you whether it still produces the *right* action. We compare the
model's predicted gamepad (stashed in ModelMeta.extras["gamepad"]) against the
dataset gold (the scenario's `gold_action.json` sidecar):

    joystick_mae         : mean |pred - gold| over j_left/j_right axes ([-1,1])
    button_agreement_rate: fraction of the 17 shared buttons whose 0/1 state matches
    action_mse           : MSE over the joined action vector (sticks + shared buttons)

Only the 17 dataset buttons are scored (the model's 4 extra right-stick-as-dpad
tokens have no gold counterpart). Pure / CPU-safe — no numpy required.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from vlm_pipeline.adapters import DATASET_BUTTON_COLUMNS

_PRESS_THRESHOLD = 0.5


@dataclass
class GamepadAccuracy:
    joystick_mae: float
    button_agreement_rate: float
    action_mse: float
    n_buttons_scored: int

    def to_dict(self) -> dict:
        return {
            "joystick_mae": self.joystick_mae,
            "button_agreement_rate": self.button_agreement_rate,
            "action_mse": self.action_mse,
            "n_buttons_scored": self.n_buttons_scored,
        }


def _axis(d: Mapping, key: str) -> tuple[float, float]:
    v = d.get(key) or [0.0, 0.0]
    return float(v[0]), float(v[1])


def compare_gamepad(pred: Mapping, gold: Mapping) -> GamepadAccuracy:
    """Score a single predicted gamepad against gold.

    `pred` and `gold` both look like {"buttons": {name: value}, "j_left": [x, y],
    "j_right": [x, y]}. Button names are matched by the dataset's 17 columns;
    missing names default to 0. Joystick error is mean-absolute over 4 axes.
    """
    pred_btn = pred.get("buttons") or {}
    gold_btn = gold.get("buttons") or {}

    abs_errs: list[float] = []
    sq_errs: list[float] = []

    # Joystick axes (continuous, [-1, 1]).
    for key in ("j_left", "j_right"):
        px, py = _axis(pred, key)
        gx, gy = _axis(gold, key)
        for p, g in ((px, gx), (py, gy)):
            abs_errs.append(abs(p - g))
            sq_errs.append((p - g) ** 2)

    # Buttons (0/1 after threshold) over the shared 17.
    matches = 0
    for name in DATASET_BUTTON_COLUMNS:
        p = float(pred_btn.get(name, 0.0))
        g = float(gold_btn.get(name, 0.0))
        p_bit = 1.0 if p > _PRESS_THRESHOLD else 0.0
        g_bit = 1.0 if g > _PRESS_THRESHOLD else 0.0
        matches += int(p_bit == g_bit)
        sq_errs.append((p_bit - g_bit) ** 2)

    n_btn = len(DATASET_BUTTON_COLUMNS)
    joystick_mae = sum(abs_errs) / len(abs_errs) if abs_errs else 0.0
    button_agreement_rate = matches / n_btn if n_btn else 0.0
    action_mse = sum(sq_errs) / len(sq_errs) if sq_errs else 0.0
    return GamepadAccuracy(
        joystick_mae=joystick_mae,
        button_agreement_rate=button_agreement_rate,
        action_mse=action_mse,
        n_buttons_scored=n_btn,
    )


def aggregate_accuracy(per_scenario: list[GamepadAccuracy]) -> dict[str, float | None]:
    """Mean each metric across scenarios. Empty input → all None."""
    if not per_scenario:
        return {
            "joystick_mae": None,
            "button_agreement_rate": None,
            "action_mse": None,
        }
    n = len(per_scenario)
    return {
        "joystick_mae": sum(a.joystick_mae for a in per_scenario) / n,
        "button_agreement_rate": sum(a.button_agreement_rate for a in per_scenario) / n,
        "action_mse": sum(a.action_mse for a in per_scenario) / n,
    }
