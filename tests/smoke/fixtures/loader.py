"""Fixture loader.

    fx = load_fixture("01_click_start_button")
    pipe.run(fx.request)

`load_all()` returns every fixture in lexical order — the parametrised
smoke test uses this so adding a new fixture directory is enough to put
it under coverage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tests.smoke.fixtures.schema import FixtureExpected, FixtureInput
from vlm_pipeline.pipeline import PipelineRequest

_FIXTURES_DIR = Path(__file__).parent


@dataclass
class LoadedFixture:
    name: str
    dir: Path
    input: FixtureInput
    expected: FixtureExpected
    image_bytes: bytes

    @property
    def request(self) -> PipelineRequest:
        return PipelineRequest(
            image=self.image_bytes,
            instruction=self.input.instruction,
            context_history=self.input.context_history,
            request_id=self.name,
            deadline_ms=self.input.deadline_ms,
        )


def load_fixture(name: str) -> LoadedFixture:
    fx_dir = _FIXTURES_DIR / name
    if not fx_dir.is_dir():
        raise FileNotFoundError(f"fixture not found: {fx_dir}")
    fx_input = FixtureInput.model_validate_json((fx_dir / "input.json").read_text())
    fx_expected = FixtureExpected.model_validate_json((fx_dir / "expected.json").read_text())
    image_bytes = (fx_dir / fx_input.image_path).read_bytes()
    return LoadedFixture(
        name=name,
        dir=fx_dir,
        input=fx_input,
        expected=fx_expected,
        image_bytes=image_bytes,
    )


def list_fixtures() -> list[str]:
    return sorted(p.name for p in _FIXTURES_DIR.iterdir() if p.is_dir() and (p / "input.json").exists())


def load_all() -> list[LoadedFixture]:
    return [load_fixture(name) for name in list_fixtures()]
