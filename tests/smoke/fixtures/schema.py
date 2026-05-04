"""On-disk schema for smoke-test fixtures.

A fixture is one (visual + short context history + high-level instruction)
→ (low-level action sequence) example. The on-disk shape is:

    tests/smoke/fixtures/<name>/
        request.json   # FixtureRequest  — what the pipeline receives
        screen.<ext>   # the visual the request points at
        expected.json  # FixtureExpected — gold ActionSequence + verdict

`FixtureRequest` mirrors `vlm_pipeline.pipeline.PipelineRequest`. The
binary image lives next to the JSON on disk so the request stays
diff-friendly. `FixtureExpected` captures only the **deterministic**
parts of `PipelineResponse`: the gold action sequence and what the
validator should say. Latency, model_meta, and was_executed are runtime
properties and are not asserted.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from vlm_pipeline.schemas import ActionSequence, ContextTurn, ValidationReport


class FixtureRequest(BaseModel):
    """Serialisable form of a PipelineRequest, with the image referenced by path."""

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
