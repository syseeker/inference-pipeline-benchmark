"""Unit-level checks for the decoder + validator. No backend required."""

from __future__ import annotations

import json

from vlm_pipeline.decoders.action_decoder import ActionDecoder
from vlm_pipeline.schemas import ActionSequence
from vlm_pipeline.validators.safety_validator import SafetyValidator


def test_decoder_parses_well_formed_json() -> None:
    raw = json.dumps(
        {
            "commands": [
                {"type": "move", "args": {"dx": 1, "dy": 2}},
                {"type": "click", "args": {"button": "left"}},
            ],
            "rationale": "approach the target then click",
        }
    )
    seq, err = ActionDecoder().decode(raw)
    assert err is None
    assert isinstance(seq, ActionSequence)
    assert len(seq.commands) == 2


def test_decoder_rejects_garbage() -> None:
    seq, err = ActionDecoder().decode("not json")
    assert seq is None and err is not None


def test_validator_flags_missing_args() -> None:
    raw = json.dumps({"commands": [{"type": "click", "args": {}}]})
    seq, _ = ActionDecoder().decode(raw)
    report = SafetyValidator().validate(seq)
    assert report.safe is False
    assert 0 in report.rejected_command_indices


def test_validator_flags_banned_keypress() -> None:
    raw = json.dumps({"commands": [{"type": "keypress", "args": {"key": "ctrl+alt+del"}}]})
    seq, _ = ActionDecoder().decode(raw)
    report = SafetyValidator().validate(seq)
    assert report.safe is False
