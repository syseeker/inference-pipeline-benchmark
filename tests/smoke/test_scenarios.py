"""Parametrised scenario smoke test (offline).

For every scenario under `tests/smoke/scenarios/<name>/`:

1. Drive the pipeline with a `_GoldReasoner` that returns the scenario's
   gold ActionSequence as JSON.
2. Assert the decoder reconstructs the gold sequence exactly.
3. Assert the validator's verdict matches the scenario's expected
   ValidationReport (schema_valid + safe + rejected_command_indices).
4. Assert the executor accepted the sequence (was_executed == safe).

This proves the scenario is internally consistent — gold sequence
parses, the validator agrees with it, and the wiring works end-to-end.
It does NOT call any model. The live-model variant lives in
`examples/run_scenario.py` and is opt-in.
"""

from __future__ import annotations

import pytest

from tests.smoke.scenarios.loader import LoadedScenario, list_scenarios, load_scenario
from tests.smoke.scenarios.schema import ScenarioExpected
from vlm_pipeline import Pipeline
from vlm_pipeline.schemas import ContextTurn, ModelMeta


class _GoldReasoner:
    """Returns the scenario's gold action sequence as JSON, verbatim."""

    def __init__(self, expected: ScenarioExpected) -> None:
        self._raw = expected.actions.model_dump_json()

    def generate(
        self,
        *,
        image: bytes,
        instruction: str,
        history: list[ContextTurn],
        deadline_ms: int,
        game_id: str | None = None,
    ) -> tuple[str, ModelMeta, float | None]:
        return self._raw, ModelMeta(framework="gold-stub", model_id="gold"), None


@pytest.mark.smoke
@pytest.mark.parametrize("scenario_name", list_scenarios())
def test_scenario_round_trip(scenario_name: str) -> None:
    sc: LoadedScenario = load_scenario(scenario_name)
    if sc.expected is None:
        pytest.skip("policy scenario (no expected.json) — VLM round-trip not applicable")
    pipe = Pipeline(reasoner=_GoldReasoner(sc.expected))

    resp = pipe.run(sc.pipeline_request())

    assert resp.error is None, f"unexpected pipeline error: {resp.error}"
    assert resp.actions == sc.expected.actions, "decoder did not reproduce the gold sequence"
    assert resp.validation.schema_valid == sc.expected.validation.schema_valid
    assert resp.validation.safe == sc.expected.validation.safe
    assert (
        resp.validation.rejected_command_indices
        == sc.expected.validation.rejected_command_indices
    )
    assert resp.was_executed == sc.expected.validation.safe
    assert resp.latency.total_ms is not None and resp.latency.total_ms >= 0


def test_scenario_set_is_non_empty() -> None:
    """Fail loud if someone deletes the scenario set."""

    assert list_scenarios(), "no smoke-test scenarios found under tests/smoke/scenarios/"
