"""Fixture loader.

    fx = load_fixture("01_clash_of_clans_start_attack")
    pipe.run(fx.pipeline_request())

`load_all()` returns every fixture in lexical order — the parametrised
smoke test uses this so adding a new fixture directory is enough to put
it under coverage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tests.smoke.fixtures.schema import FixtureExpected, FixtureRequest
from vlm_pipeline.pipeline import PipelineRequest

_FIXTURES_DIR = Path(__file__).parent


@dataclass
class LoadedFixture:
    name: str
    dir: Path
    spec: FixtureRequest      # on-disk request spec (image referenced by path)
    expected: FixtureExpected
    image_bytes: bytes

    def pipeline_request(self) -> PipelineRequest:
        """Materialise the on-disk spec into a runtime PipelineRequest."""

        return PipelineRequest(
            image=self.image_bytes,
            instruction=self.spec.instruction,
            context_history=self.spec.context_history,
            request_id=self.name,
            deadline_ms=self.spec.deadline_ms,
        )


def load_fixture(name: str) -> LoadedFixture:
    fx_dir = _FIXTURES_DIR / name
    if not fx_dir.is_dir():
        raise FileNotFoundError(f"fixture not found: {fx_dir}")
    spec = FixtureRequest.model_validate_json((fx_dir / "request.json").read_text())
    expected = FixtureExpected.model_validate_json((fx_dir / "expected.json").read_text())
    image_bytes = (fx_dir / spec.image_path).read_bytes()
    return LoadedFixture(
        name=name,
        dir=fx_dir,
        spec=spec,
        expected=expected,
        image_bytes=image_bytes,
    )


def list_fixtures() -> list[str]:
    return sorted(
        p.name
        for p in _FIXTURES_DIR.iterdir()
        if p.is_dir() and not p.name.startswith("__") and (p / "request.json").exists()
    )


def load_all() -> list[LoadedFixture]:
    return [load_fixture(name) for name in list_fixtures()]
