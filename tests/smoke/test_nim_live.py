"""Live NIM smoke — only runs when NIM_API_KEY is set.

This is the placeholder the brief asks for. CI skips it; engineers can
opt-in with:

    NIM_API_KEY=... pytest -m nim tests/smoke/test_nim_live.py
"""

from __future__ import annotations

import os

import pytest

from vlm_pipeline import Pipeline, PipelineRequest
from vlm_pipeline.config import PipelineConfig

_HAS_KEY = bool(os.getenv("NIM_API_KEY"))


@pytest.mark.nim
@pytest.mark.smoke
@pytest.mark.skipif(not _HAS_KEY, reason="NIM_API_KEY not set")
def test_nim_qwen_vl_round_trip(fake_image_bytes: bytes) -> None:
    from vlm_pipeline.reasoners.nim_qwen_vl import NimQwenVlReasoner

    cfg = PipelineConfig.from_env()
    pipe = Pipeline(reasoner=NimQwenVlReasoner(cfg.nim), config=cfg)
    resp = pipe.run(
        PipelineRequest(
            image=fake_image_bytes,
            instruction=(
                "You see a grey 224x224 placeholder image. Reply with a single safe "
                "noop command and a one-sentence rationale."
            ),
        )
    )
    # We don't assert success — model output may vary — but we DO want the
    # round trip to complete and the decoder to have an opinion.
    assert resp.latency.reasoner_total_ms is not None
    assert resp.model_meta is not None
    assert resp.model_meta.framework == "nim"
