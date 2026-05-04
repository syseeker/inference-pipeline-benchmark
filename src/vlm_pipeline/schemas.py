"""Typed I/O for the VLM-to-action pipeline.

`ActionSequence` is the contract the VLM reasoner is asked to produce. It
is the choke point between free-form model text and the executor: every
stage downstream operates on this typed shape, never on raw model output.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ActionType(str, Enum):
    """Closed set of low-level commands the executor understands.

    Keep this list small. New commands require a validator update.
    """

    NOOP = "noop"
    MOVE = "move"
    CLICK = "click"
    KEYPRESS = "keypress"
    WAIT = "wait"
    SAY = "say"


class ActionCommand(BaseModel):
    """One low-level command in a sequence."""

    type: ActionType
    args: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ActionSequence(BaseModel):
    """The constrained output of the VLM-to-action pipeline."""

    commands: list[ActionCommand]
    rationale: str | None = None

    @field_validator("commands")
    @classmethod
    def _non_empty(cls, v: list[ActionCommand]) -> list[ActionCommand]:
        if not v:
            raise ValueError("ActionSequence.commands must contain at least one command")
        return v


class ContextTurn(BaseModel):
    """A single past turn in the short rolling history."""

    role: Literal["user", "assistant", "system"]
    text: str
    image_ref: str | None = None  # opaque pointer; pipeline does not re-encode by default


class LatencyBreakdown(BaseModel):
    """Per-stage timings (ms). Populated by the pipeline as it runs."""

    vision_encoder_ms: float | None = None
    reasoner_ttft_ms: float | None = None
    reasoner_total_ms: float | None = None
    decoder_ms: float | None = None
    validator_ms: float | None = None
    executor_ms: float | None = None
    total_ms: float | None = None


class ValidationReport(BaseModel):
    """What the safety validator decided about the produced sequence."""

    schema_valid: bool
    safe: bool
    rejected_command_indices: list[int] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ModelMeta(BaseModel):
    """Trace of which model + framework actually answered."""

    framework: str  # "nim", "vllm", "sglang", "trtllm"
    model_id: str
    quantization: str | None = None
    extras: dict[str, Any] = Field(default_factory=dict)
