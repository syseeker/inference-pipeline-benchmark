"""On-disk schema for smoke-test fixtures.

Each fixture directory under `tests/smoke/fixtures/<name>/` holds:

    input.json     # FixtureInput — the PipelineRequest we feed in
    image.jpg      # the visual the request references
    expected.json  # FixtureExpected — the golden ActionSequence + validation

`FixtureInput` is a serialisable mirror of `PipelineRequest` (the binary
image lives next to it on disk so the JSON stays diff-friendly).
`FixtureExpected` captures only the deterministic parts of
`PipelineResponse`: the gold action sequence and what the validator
should say about it. Latency, model_meta, and was_executed are runtime
properties and are not asserted.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from vlm_pipeline.schemas import ActionSequence, ContextTurn, ValidationReport


class FixtureInput(BaseModel):
    name: str
    description: str
    image_path: str = Field(..., description="Image path relative to the fixture dir.")
    instruction: str
    context_history: list[ContextTurn] = Field(default_factory=list)
    deadline_ms: int = 1500


class FixtureExpected(BaseModel):
    actions: ActionSequence
    validation: ValidationReport
    notes: str | None = None
