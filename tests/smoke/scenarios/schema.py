"""On-disk schema for smoke-test scenarios.

A scenario is one (visual + short context history + high-level instruction)
→ (low-level action sequence) example. The on-disk shape is:

    tests/smoke/scenarios/<name>/
        request.json   # ScenarioRequest  — what the pipeline receives
        screen.<ext>   # the visual the request points at
        expected.json  # ScenarioExpected — gold ActionSequence + verdict

`ScenarioRequest` mirrors `vlm_pipeline.pipeline.PipelineRequest`. The
binary image lives next to the JSON on disk so the request stays
diff-friendly. `ScenarioExpected` captures only the **deterministic**
parts of `PipelineResponse`: the gold action sequence and what the
validator should say. Latency, model_meta, and was_executed are runtime
properties and are not asserted.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from vlm_pipeline.schemas import ActionSequence, ContextTurn, ValidationReport


class ScenarioRequest(BaseModel):
    """Serialisable form of a PipelineRequest, with the image referenced by path.

    Image scenarios set image_path + instruction.
    Video scenarios set video_path + prompt (image_path / instruction are None).
    """

    name: str
    description: str
    # Image scenario fields (required for image/policy scenarios)
    image_path: str | None = Field(default=None, description="Image path relative to the scenario dir.")
    instruction: str | None = None
    context_history: list[ContextTurn] = Field(default_factory=list)
    deadline_ms: int = 1500
    game_id: str | None = Field(
        default=None,
        description=(
            "Optional game conditioning for policy backends (e.g. NitroGen), which "
            "are conditioned on a game id rather than the text instruction. Ignored "
            "by text-driven VLM reasoners."
        ),
    )
    # Video scenario fields
    video_path: str | None = Field(default=None, description="Video file path relative to the scenario dir.")
    prompt: str | None = Field(default=None, description="Text question / instruction for video scenarios.")

    @property
    def is_video(self) -> bool:
        return self.video_path is not None or (self.image_path is None and self.prompt is not None)


class ScenarioExpected(BaseModel):
    """Gold output for image/action scenarios."""
    actions: ActionSequence
    validation: ValidationReport
    notes: str | None = None


class VideoScenarioExpected(BaseModel):
    """Gold output for video+text scenarios — key-phrase coverage check."""
    key_phrases: list[str]
    min_coverage: float = 0.6
