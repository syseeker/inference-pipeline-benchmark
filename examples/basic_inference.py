"""Minimal end-to-end example.

Run:
    export NIM_API_KEY=...
    python -m examples.basic_inference path/to/image.jpg "hover and click the start button"

Falls back to a stub reasoner if NIM_API_KEY is unset, so the example is
runnable on a laptop with no credentials.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from rich import print as rprint

from vlm_pipeline import Pipeline, PipelineRequest
from vlm_pipeline.config import PipelineConfig
from vlm_pipeline.schemas import ModelMeta


class _StubReasoner:
    def generate(self, **_: object) -> tuple[str, ModelMeta, float | None]:
        raw = json.dumps(
            {
                "commands": [
                    {"type": "move", "args": {"dx": 100, "dy": 50}},
                    {"type": "click", "args": {"button": "left"}},
                ],
                "rationale": "stub: no NIM_API_KEY set",
            }
        )
        return raw, ModelMeta(framework="stub", model_id="stub"), None


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: python -m examples.basic_inference <image_path> <instruction>", file=sys.stderr)
        return 2
    image = Path(argv[1]).read_bytes()
    instruction = argv[2]

    cfg = PipelineConfig.from_env()
    if cfg.nim.api_key and os.getenv("USE_NIM", "1") == "1":
        from vlm_pipeline.reasoners.nim_qwen_vl import NimQwenVlReasoner

        reasoner = NimQwenVlReasoner(cfg.nim)
    else:
        rprint("[yellow]NIM_API_KEY not set — using stub reasoner[/yellow]")
        reasoner = _StubReasoner()

    pipe = Pipeline(reasoner=reasoner, config=cfg)
    resp = pipe.run(PipelineRequest(image=image, instruction=instruction))
    rprint(resp)
    return 0 if resp.error is None else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
