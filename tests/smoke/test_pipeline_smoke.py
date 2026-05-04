"""Pipeline smoke test (offline).

Drives the orchestrator with a stub reasoner that returns canned JSON.
Proves: encoder → reasoner → decoder → validator → executor wiring is
intact and `LatencyBreakdown` is populated.

Replace `_StubReasoner` with `NimQwenVlReasoner(NimConfig.from_env())` for
the live NIM smoke check (requires `NIM_API_KEY`).
"""

from __future__ import annotations

import json

import pytest

from vlm_pipeline import Pipeline, PipelineRequest
from vlm_pipeline.schemas import ContextTurn, ModelMeta


class _StubReasoner:
    def generate(
        self,
        *,
        image: bytes,
        instruction: str,
        history: list[ContextTurn],
        deadline_ms: int,
    ) -> tuple[str, ModelMeta, float | None]:
        raw = json.dumps(
            {
                "commands": [
                    {"type": "move", "args": {"dx": 5, "dy": -3}},
                    {"type": "click", "args": {"button": "left"}},
                ],
                "rationale": "stub",
            }
        )
        return raw, ModelMeta(framework="stub", model_id="stub"), 12.5


@pytest.mark.smoke
def test_pipeline_happy_path(fake_image_bytes: bytes) -> None:
    pipe = Pipeline(reasoner=_StubReasoner())
    resp = pipe.run(
        PipelineRequest(image=fake_image_bytes, instruction="hover and click the button")
    )
    assert resp.error is None
    assert resp.actions is not None
    assert len(resp.actions.commands) == 2
    assert resp.validation.schema_valid and resp.validation.safe
    assert resp.was_executed is True
    # Latency breakdown is populated.
    assert resp.latency.total_ms is not None
    assert resp.latency.reasoner_total_ms is not None
    assert resp.latency.decoder_ms is not None
    assert resp.latency.validator_ms is not None


@pytest.mark.smoke
def test_pipeline_rejects_unparseable_output(fake_image_bytes: bytes) -> None:
    class _BadReasoner:
        def generate(self, **_: object) -> tuple[str, ModelMeta, float | None]:
            return "this is not json", ModelMeta(framework="stub", model_id="stub"), None

    pipe = Pipeline(reasoner=_BadReasoner())
    resp = pipe.run(PipelineRequest(image=fake_image_bytes, instruction="x"))
    assert resp.actions is None
    assert resp.was_executed is False
    assert resp.validation.schema_valid is False
    assert resp.error is not None
