"""Scenario loader.

    sc = load_scenario("01_clash_of_clans_start_attack")
    pipe.run(sc.pipeline_request())

`load_all()` returns every scenario in lexical order — the parametrised
smoke test uses this so adding a new scenario directory is enough to
put it under coverage. Pass `scenarios_dir=` to load from somewhere
other than the bundled `tests/smoke/scenarios/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tests.smoke.scenarios.schema import ScenarioExpected, ScenarioRequest
from vlm_pipeline.pipeline import PipelineRequest

DEFAULT_SCENARIOS_DIR = Path(__file__).parent


@dataclass
class LoadedScenario:
    name: str
    dir: Path
    spec: ScenarioRequest      # on-disk request spec (image referenced by path)
    expected: ScenarioExpected
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


def load_scenario(name: str, scenarios_dir: Path | None = None) -> LoadedScenario:
    root = Path(scenarios_dir) if scenarios_dir is not None else DEFAULT_SCENARIOS_DIR
    sc_dir = root / name
    if not sc_dir.is_dir():
        raise FileNotFoundError(f"scenario not found: {sc_dir}")
    spec = ScenarioRequest.model_validate_json((sc_dir / "request.json").read_text())
    expected = ScenarioExpected.model_validate_json((sc_dir / "expected.json").read_text())
    image_bytes = (sc_dir / spec.image_path).read_bytes()
    return LoadedScenario(
        name=name,
        dir=sc_dir,
        spec=spec,
        expected=expected,
        image_bytes=image_bytes,
    )


def list_scenarios(scenarios_dir: Path | None = None) -> list[str]:
    root = Path(scenarios_dir) if scenarios_dir is not None else DEFAULT_SCENARIOS_DIR
    if not root.is_dir():
        raise FileNotFoundError(f"scenarios dir not found: {root}")
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and not p.name.startswith("__") and (p / "request.json").exists()
    )


def load_all(scenarios_dir: Path | None = None) -> list[LoadedScenario]:
    return [load_scenario(name, scenarios_dir) for name in list_scenarios(scenarios_dir)]
