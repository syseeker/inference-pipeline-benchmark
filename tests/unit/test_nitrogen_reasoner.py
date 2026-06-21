"""Unit tests for NitrogenReasoner — CPU-only, using a fake ZMQ client and a
small in-memory PNG. Exercises the reasoner directly and end-to-end through the
real Pipeline (decoder + validator + executor)."""

from __future__ import annotations

import io
import json

from vlm_pipeline import Pipeline, PipelineRequest
from vlm_pipeline.config import NitrogenConfig
from vlm_pipeline.reasoners.nitrogen_backend import (
    MODEL_BUTTON_TOKENS,
    NitrogenReasoner,
    gamepad_from_prediction,
)
from vlm_pipeline.schemas import ActionType


def _png_bytes(size: int = 8) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (size, size), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


class FakeClient:
    """Records the last predict() call and returns a canned prediction."""

    def __init__(self, pred: dict) -> None:
        self._pred = pred
        self.calls: list[dict] = []
        self.reset_count = 0

    def predict(self, image, *, game_id=None, seed=None) -> dict:
        shape = getattr(image, "shape", None)
        self.calls.append({"game_id": game_id, "seed": seed, "shape": shape})
        return self._pred

    def reset(self) -> None:
        self.reset_count += 1


def _pred_south_and_left_stick() -> dict:
    # 21-button vector with SOUTH (index 18) pressed; left stick pushed right.
    buttons = [0.0] * 21
    buttons[MODEL_BUTTON_TOKENS.index("south")] = 1.0
    return {"buttons": [buttons], "j_left": [[1.0, 0.0]], "j_right": [[0.0, 0.0]]}


def test_gamepad_from_prediction_unpacks_horizon_step0():
    pad = gamepad_from_prediction(_pred_south_and_left_stick())
    assert pad.buttons["south"] == 1.0
    assert pad.buttons["north"] == 0.0
    assert pad.j_left == (1.0, 0.0)
    assert set(["right_bottom", "right_up"]).issubset(pad.buttons)  # all 21 present


def test_gamepad_from_prediction_handles_flat_arrays():
    # horizon-collapsed (dim,) arrays rather than (horizon, dim)
    buttons = [0.0] * 21
    pad = gamepad_from_prediction({"buttons": buttons, "j_left": [0.5, -0.5], "j_right": [0, 0]})
    assert pad.j_left == (0.5, -0.5)


def test_reasoner_returns_lossy_json_and_raw_gamepad_in_extras():
    client = FakeClient(_pred_south_and_left_stick())
    reasoner = NitrogenReasoner(NitrogenConfig(seed=7, move_scale=512), client=client)

    raw, meta, ttft = reasoner.generate(
        image=_png_bytes(), instruction="", history=[], deadline_ms=1500, game_id="celeste"
    )

    assert ttft is None  # no token stream
    assert meta.framework == "nitrogen"
    # Raw is JSON parseable into the lossy ActionSequence.
    obj = json.loads(raw)
    types = [c["type"] for c in obj["commands"]]
    assert "move" in types and "keypress" in types
    # Faithful gamepad preserved for accuracy; game_id + seed forwarded.
    assert meta.extras["gamepad"]["buttons"]["south"] == 1.0
    assert meta.extras["game_id"] == "celeste"
    assert client.calls[0] == {"game_id": "celeste", "seed": 7, "shape": (8, 8, 3)}


def test_reasoner_runs_end_to_end_through_pipeline():
    client = FakeClient(_pred_south_and_left_stick())
    reasoner = NitrogenReasoner(NitrogenConfig(), client=client)
    pipe = Pipeline(reasoner=reasoner)

    resp = pipe.run(PipelineRequest(image=_png_bytes(), instruction="", game_id="celeste"))

    assert resp.actions is not None
    assert resp.validation.schema_valid and resp.validation.safe
    assert resp.was_executed  # DryRunExecutor accepts a valid sequence
    assert resp.latency.total_ms is not None
    assert resp.model_meta.extras["gamepad"]["j_left"] == [1.0, 0.0]


def test_idle_prediction_yields_noop_but_valid():
    client = FakeClient({"buttons": [[0.0] * 21], "j_left": [[0.0, 0.0]], "j_right": [[0.0, 0.0]]})
    reasoner = NitrogenReasoner(NitrogenConfig(), client=client)
    raw, _, _ = reasoner.generate(image=_png_bytes(), instruction="", history=[], deadline_ms=1500)
    obj = json.loads(raw)
    assert [c["type"] for c in obj["commands"]] == [ActionType.NOOP.value]


def test_parse_zmq_url():
    from vlm_pipeline.reasoners.nitrogen_backend import _parse_zmq_url

    assert _parse_zmq_url("tcp://localhost:5555") == ("localhost", 5555)
    assert _parse_zmq_url("tcp://10.0.0.5:6000") == ("10.0.0.5", 6000)
