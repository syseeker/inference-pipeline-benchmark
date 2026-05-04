"""Parametrised fixture smoke test (offline).

For every fixture under `tests/smoke/fixtures/<name>/`:

1. Drive the pipeline with a `_GoldReasoner` that returns the fixture's
   gold ActionSequence as JSON.
2. Assert the decoder reconstructs the gold sequence exactly.
3. Assert the validator's verdict matches the fixture's expected
   ValidationReport (schema_valid + safe + rejected_command_indices).
4. Assert the executor accepted the sequence (was_executed == safe).

This proves the fixture is internally consistent — gold sequence parses,
the validator agrees with it, and the wiring works end-to-end. It does
NOT call any model. The live-model variant lives in
`examples/run_fixture.py` and is opt-in.
"""

from __future__ import annotations

import pytest

from tests.smoke.fixtures.loader import LoadedFixture, list_fixtures, load_fixture
from tests.smoke.fixtures.schema import FixtureExpected
from vlm_pipeline import Pipeline
from vlm_pipeline.schemas import ContextTurn, ModelMeta


class _GoldReasoner:
    """Returns the fixture's gold action sequence as JSON, verbatim."""

    def __init__(self, expected: FixtureExpected) -> None:
        self._raw = expected.actions.model_dump_json()

    def generate(
        self,
        *,
        image: bytes,
        instruction: str,
        history: list[ContextTurn],
        deadline_ms: int,
    ) -> tuple[str, ModelMeta, float | None]:
        return self._raw, ModelMeta(framework="gold-stub", model_id="gold"), None


@pytest.mark.smoke
@pytest.mark.parametrize("fixture_name", list_fixtures())
def test_fixture_round_trip(fixture_name: str) -> None:
    fx: LoadedFixture = load_fixture(fixture_name)
    pipe = Pipeline(reasoner=_GoldReasoner(fx.expected))

    resp = pipe.run(fx.request)

    assert resp.error is None, f"unexpected pipeline error: {resp.error}"
    assert resp.actions == fx.expected.actions, "decoder did not reproduce the gold sequence"
    assert resp.validation.schema_valid == fx.expected.validation.schema_valid
    assert resp.validation.safe == fx.expected.validation.safe
    assert (
        resp.validation.rejected_command_indices
        == fx.expected.validation.rejected_command_indices
    )
    assert resp.was_executed == fx.expected.validation.safe
    assert resp.latency.total_ms is not None and resp.latency.total_ms >= 0


def test_fixture_set_is_non_empty() -> None:
    """Fail loud if someone deletes the fixture set."""

    assert list_fixtures(), "no smoke-test fixtures found under tests/smoke/fixtures/"
