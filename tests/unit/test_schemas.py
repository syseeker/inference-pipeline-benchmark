"""Schema sanity — fast, offline, no GPU."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vlm_pipeline.schemas import ActionCommand, ActionSequence, ActionType


def test_action_sequence_requires_at_least_one_command() -> None:
    with pytest.raises(ValidationError):
        ActionSequence(commands=[])


def test_action_command_round_trip() -> None:
    seq = ActionSequence(
        commands=[
            ActionCommand(type=ActionType.MOVE, args={"dx": 10, "dy": 0}),
            ActionCommand(type=ActionType.CLICK, args={"button": "left"}),
        ],
        rationale="hover then click",
    )
    payload = seq.model_dump()
    assert payload["commands"][0]["type"] == "move"
    assert ActionSequence.model_validate(payload) == seq
