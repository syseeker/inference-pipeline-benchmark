"""Unit tests for the NitroGen gamepad -> ActionSequence adapter and the
pure helpers of the scenario converter. All CPU-only (no GPU, no network)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from vlm_pipeline.adapters import (
    DATASET_BUTTON_COLUMNS,
    Gamepad,
    gamepad_to_action_sequence,
)
from vlm_pipeline.schemas import ActionType

# Load the converter script as a module (it lives under scripts/, not the package).
_CONV_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_nitrogen_scenarios.py"
_spec = importlib.util.spec_from_file_location("build_nitrogen_scenarios", _CONV_PATH)
conv = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = conv  # so dataclass field-type resolution can find the module
_spec.loader.exec_module(conv)  # type: ignore[union-attr]


# --------------------------------------------------------------------------- #
# Adapter                                                                     #
# --------------------------------------------------------------------------- #


def test_idle_gamepad_yields_single_noop():
    seq = gamepad_to_action_sequence(Gamepad())
    assert len(seq.commands) == 1
    assert seq.commands[0].type == ActionType.NOOP


def test_left_stick_maps_to_move_with_scaled_pixels():
    pad = Gamepad(j_left=(1.0, -0.5))
    seq = gamepad_to_action_sequence(pad, move_scale=512)
    moves = [c for c in seq.commands if c.type == ActionType.MOVE]
    assert len(moves) == 1
    assert moves[0].args == {"dx": 512, "dy": -256}


def test_stick_inside_deadzone_is_dropped():
    pad = Gamepad(j_left=(0.05, 0.05))  # magnitude < default deadzone 0.15
    seq = gamepad_to_action_sequence(pad)
    assert all(c.type != ActionType.MOVE for c in seq.commands)
    assert seq.commands[0].type == ActionType.NOOP


def test_right_stick_dropped_by_default_included_on_flag():
    pad = Gamepad(j_right=(0.9, 0.0))
    assert all(c.type != ActionType.MOVE for c in gamepad_to_action_sequence(pad).commands)
    seq = gamepad_to_action_sequence(pad, include_right_stick=True)
    assert any(c.type == ActionType.MOVE for c in seq.commands)


def test_pressed_buttons_map_to_keypress_in_column_order():
    pad = Gamepad(buttons={"south": 1.0, "left_trigger": 0.9, "north": 0.0})
    seq = gamepad_to_action_sequence(pad)
    keys = [c.args["key"] for c in seq.commands if c.type == ActionType.KEYPRESS]
    assert keys == ["left_trigger", "south"]  # parquet column order, not insertion order
    assert "north" not in keys


def test_button_threshold_is_strict():
    pad = Gamepad(buttons={"south": 0.5})  # exactly threshold -> not pressed
    seq = gamepad_to_action_sequence(pad)
    assert all(c.type != ActionType.KEYPRESS for c in seq.commands)


def test_combined_move_and_buttons_order():
    pad = Gamepad(j_left=(0.0, 1.0), buttons={"east": 1.0})
    seq = gamepad_to_action_sequence(pad)
    types = [c.type for c in seq.commands]
    assert types == [ActionType.MOVE, ActionType.KEYPRESS]  # MOVE precedes buttons


def test_from_dataset_row_list_and_scalar_axes():
    row_list = {"south": 1, "j_left": [0.5, -0.5], "j_right": [0.0, 0.0]}
    pad = Gamepad.from_dataset_row(row_list)
    assert pad.j_left == (0.5, -0.5)
    assert pad.buttons["south"] == 1.0

    row_scalar = {"j_left_x": 0.2, "j_left_y": 0.3}
    pad2 = Gamepad.from_dataset_row(row_scalar)
    assert pad2.j_left == (0.2, 0.3)
    # All 17 dataset columns present, defaulting to 0.
    assert set(pad2.buttons) == set(DATASET_BUTTON_COLUMNS)


# --------------------------------------------------------------------------- #
# Converter pure helpers                                                      #
# --------------------------------------------------------------------------- #


def test_parse_metadata_flat_and_nested_layouts():
    flat = {
        "url": "u", "game": "Celeste", "width": 1920, "height": 1080,
        "frame_indices": [10, 11, 12],
    }
    m = conv.parse_metadata(flat)
    assert (m.url, m.game, m.width, m.height) == ("u", "Celeste", 1920, 1080)
    assert m.sample_frame_index == 11  # midpoint

    nested = {
        "url": "u", "game": "g",
        "resolution": {"width": 640, "height": 480},
        "frames": {"start": 100, "end": 104},
    }
    m2 = conv.parse_metadata(nested)
    assert (m2.width, m2.height) == (640, 480)
    assert m2.frame_indices == [100, 101, 102, 103]


def test_resolve_game_id_identity_and_normalized_lookup():
    assert conv.resolve_game_id("Celeste", None) == "Celeste"
    assert conv.resolve_game_id("Celeste", {"Celeste": "7"}) == "7"
    assert conv.resolve_game_id("clash of clans", {"Clash_Of_Clans": "3"}) == "3"
    with pytest.raises(KeyError):
        conv.resolve_game_id("Unknown", {"Celeste": "7"})


def test_build_scenario_payloads_shape_and_gold_sidecar():
    pad = Gamepad(j_left=(1.0, 0.0), buttons={"south": 1.0})
    request, gold = conv.build_scenario_payloads(
        name="00_celeste_chunk_0000",
        description="desc",
        game_id="7",
        pad=pad,
        deadline_ms=1500,
        provenance={"chunk": "SHARD_0000/v/v_chunk_0000", "frame_index": 11},
    )
    assert request["game_id"] == "7"
    assert request["instruction"] == ""
    assert request["image_path"] == "screen.png"
    # The gold sidecar preserves the faithful gamepad action. The extractor no
    # longer emits a lossy expected.json — VLM grading needs a human-authored
    # instruction + expected list, layered on later.
    assert gold["j_left"] == [1.0, 0.0]
    assert gold["buttons"]["south"] == 1.0
    assert gold["provenance"]["frame_index"] == 11


def test_payloads_validate_against_on_disk_schema():
    """The emitted JSON must round-trip through the real ScenarioRequest pydantic model."""
    from tests.smoke.scenarios.schema import ScenarioRequest

    pad = Gamepad(buttons={"start": 1.0})
    request, _gold = conv.build_scenario_payloads(
        name="00_g_chunk_0000", description="d", game_id="1",
        pad=pad, deadline_ms=1500, provenance={},
    )
    sr = ScenarioRequest.model_validate(request)
    assert sr.game_id == "1"
    assert sr.instruction == ""
