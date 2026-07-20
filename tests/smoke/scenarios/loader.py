"""Scenario loader.

    sc = load_scenario("01_clash_of_clans_start_attack")
    pipe.run(sc.pipeline_request())

`load_all()` returns every scenario in lexical order — the parametrised
smoke test uses this so adding a new scenario directory is enough to
put it under coverage. Pass `scenarios_dir=` to load from somewhere
other than the bundled `tests/smoke/scenarios/`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tests.smoke.scenarios.schema import ScenarioExpected, ScenarioRequest, VideoScenarioExpected
from vlm_pipeline.pipeline import PipelineRequest

DEFAULT_SCENARIOS_DIR = Path(__file__).parent


@dataclass
class LoadedScenario:
    name: str
    dir: Path
    spec: ScenarioRequest      # on-disk request spec (image referenced by path)
    image_bytes: bytes | None  # None for video scenarios
    # Both ground-truth files are OPTIONAL — the grader dispatches on presence:
    #   - `expected.json`       → VLM grading: parsed ActionSequence vs gold
    #                             OR video grading: VideoScenarioExpected key-phrases
    #   - `gold_action.json`    → policy grading: gamepad vs gold via accuracy.py
    expected: ScenarioExpected | None = None
    video_expected: VideoScenarioExpected | None = None
    gold_action: dict[str, Any] | None = None

    def pipeline_request(self) -> PipelineRequest:
        """Materialise the on-disk spec into a runtime PipelineRequest (image scenarios only)."""
        if self.spec.is_video:
            raise ValueError(
                f"scenario {self.name!r} is a video scenario — "
                "use VideoTextReasoner.generate() directly, not pipeline_request()"
            )
        return PipelineRequest(
            image=self.image_bytes,
            instruction=self.spec.instruction,
            context_history=self.spec.context_history,
            request_id=self.name,
            deadline_ms=self.spec.deadline_ms,
            game_id=self.spec.game_id,
        )


def load_scenario(name: str, scenarios_dir: Path | None = None) -> LoadedScenario:
    root = Path(scenarios_dir) if scenarios_dir is not None else DEFAULT_SCENARIOS_DIR
    sc_dir = root / name
    if not sc_dir.is_dir():
        raise FileNotFoundError(f"scenario not found: {sc_dir}")
    spec = ScenarioRequest.model_validate_json((sc_dir / "request.json").read_text())
    image_bytes = (sc_dir / spec.image_path).read_bytes() if spec.image_path else None

    expected: ScenarioExpected | None = None
    video_expected: VideoScenarioExpected | None = None
    expected_path = sc_dir / "expected.json"
    if expected_path.exists():
        raw = json.loads(expected_path.read_text())
        if "key_phrases" in raw:
            video_expected = VideoScenarioExpected.model_validate(raw)
        else:
            expected = ScenarioExpected.model_validate(raw)

    gold_path = sc_dir / "gold_action.json"
    gold_action = json.loads(gold_path.read_text()) if gold_path.exists() else None
    return LoadedScenario(
        name=name,
        dir=sc_dir,
        spec=spec,
        image_bytes=image_bytes,
        expected=expected,
        video_expected=video_expected,
        gold_action=gold_action,
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
