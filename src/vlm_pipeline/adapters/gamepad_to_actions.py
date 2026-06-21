"""Lossy adapter: NitroGen gamepad action -> pipeline ActionSequence.

NitroGen does not emit the pipeline's discrete command vocabulary. It emits a
*continuous* gamepad state per horizon step: two analog sticks in [-1, 1] and a
set of (near-)boolean buttons. To run NitroGen through the existing
decoder/validator/executor path we project that gamepad state onto the closed
`ActionType` set.

This projection is deliberately **lossy** and exists only to keep the
downstream pipeline shape intact (so latency/validity stages still apply):

    left stick  (beyond deadzone)  -> MOVE {dx, dy}
    right stick (beyond deadzone)  -> MOVE {dx, dy}   (optional; default dropped)
    each pressed button            -> KEYPRESS {key: <token>}
    nothing active                 -> NOOP

The faithful, full-precision gamepad action is preserved separately (the
`gold_action.json` sidecar written by the scenario converter) and is what the
accuracy-vs-gold metric compares against. Do **not** treat the ActionSequence
produced here as the source of truth for accuracy.

The 17 button names below are the dataset's `actions_processed.parquet`
columns. The *model* emits 21 buttons (`nitrogen.shared.BUTTON_ACTION_TOKENS`);
that wider space is handled at the accuracy-metric seam, not here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from vlm_pipeline.schemas import ActionCommand, ActionSequence, ActionType

# Dataset (gold) button columns, in the order documented on the dataset card.
DATASET_BUTTON_COLUMNS: tuple[str, ...] = (
    "dpad_down",
    "dpad_left",
    "dpad_right",
    "dpad_up",
    "left_shoulder",
    "left_thumb",
    "left_trigger",
    "right_shoulder",
    "right_thumb",
    "right_trigger",
    "south",
    "west",
    "east",
    "north",
    "back",
    "start",
    "guide",
)


@dataclass
class Gamepad:
    """One gamepad state (a single horizon step).

    Joystick axes are in [-1, 1]; button values are typically 0/1 for gold and
    in [0, 1] for raw model output (probabilities) before thresholding.
    """

    buttons: dict[str, float] = field(default_factory=dict)
    j_left: tuple[float, float] = (0.0, 0.0)
    j_right: tuple[float, float] = (0.0, 0.0)

    @classmethod
    def from_dataset_row(cls, row: Mapping[str, object]) -> Gamepad:
        """Build from an `actions_processed.parquet` row (a plain mapping).

        `j_left`/`j_right` may arrive as (x, y) sequences (parquet list columns)
        or as separate ``j_left_x``/``j_left_y`` scalar columns.
        """

        buttons = {col: float(bool(row.get(col, 0))) for col in DATASET_BUTTON_COLUMNS}
        return cls(
            buttons=buttons,
            j_left=_read_axis(row, "j_left"),
            j_right=_read_axis(row, "j_right"),
        )


def _read_axis(row: Mapping[str, object], name: str) -> tuple[float, float]:
    if name in row and isinstance(row[name], Sequence) and not isinstance(row[name], str):
        xy = list(row[name])  # type: ignore[arg-type]
        return float(xy[0]), float(xy[1])
    return float(row.get(f"{name}_x", 0.0)), float(row.get(f"{name}_y", 0.0))


def _stick_to_move(
    axis: tuple[float, float], move_scale: int, deadzone: float
) -> ActionCommand | None:
    x, y = axis
    if (x * x + y * y) ** 0.5 < deadzone:
        return None
    # Magnitude (capped at 1.0) doubles as a coarse confidence signal.
    confidence = min(1.0, (x * x + y * y) ** 0.5)
    return ActionCommand(
        type=ActionType.MOVE,
        args={"dx": round(x * move_scale), "dy": round(y * move_scale)},
        confidence=round(confidence, 3),
    )


def gamepad_to_action_sequence(
    pad: Gamepad,
    *,
    move_scale: int = 512,
    deadzone: float = 0.15,
    press_threshold: float = 0.5,
    include_right_stick: bool = False,
    rationale: str | None = None,
) -> ActionSequence:
    """Project a single gamepad state onto a (lossy) ActionSequence.

    Always returns a non-empty sequence (falls back to a single NOOP), so the
    result is always a valid `ActionSequence`.

    Args:
        move_scale: pixels per unit stick deflection for MOVE dx/dy.
        deadzone: stick magnitude below which a stick contributes no MOVE.
        press_threshold: button value strictly above which counts as pressed.
        include_right_stick: also emit a MOVE for the right stick. Off by
            default — with only one MOVE type, two MOVEs are ambiguous, so the
            right stick (camera/aim) is dropped from the lossy view.
        rationale: optional ActionSequence.rationale.
    """

    commands: list[ActionCommand] = []

    left = _stick_to_move(pad.j_left, move_scale, deadzone)
    if left is not None:
        commands.append(left)
    if include_right_stick:
        right = _stick_to_move(pad.j_right, move_scale, deadzone)
        if right is not None:
            commands.append(right)

    for name in DATASET_BUTTON_COLUMNS:
        value = float(pad.buttons.get(name, 0.0))
        if value > press_threshold:
            commands.append(
                ActionCommand(
                    type=ActionType.KEYPRESS,
                    args={"key": name},
                    confidence=round(min(1.0, value), 3),
                )
            )

    if not commands:
        commands.append(ActionCommand(type=ActionType.NOOP, args={}, confidence=1.0))

    return ActionSequence(commands=commands, rationale=rationale)
