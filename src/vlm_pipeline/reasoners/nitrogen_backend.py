"""NitroGen reasoner — diffusion-policy backend over ZMQ.

NitroGen is not an autoregressive VLM: it's a flow-matching visuomotor policy
served over ZMQ (NitroGen's `scripts/serve.py` / our `scripts/serve_nitrogen.py`),
not an OpenAI-compatible HTTP server. It takes a 256x256 frame + game id and
returns a continuous gamepad action (2 sticks + 21 buttons over a horizon).

This reasoner adapts that to the `VlmReasoner` contract so NitroGen runs through
the unchanged decoder/validator/executor/metrics path:

    image bytes -> 256x256 RGB ndarray -> client.predict(...) -> gamepad ->
        lossy ActionSequence JSON  (returned as raw_text; decoder parses it)

The faithful gamepad action (21 buttons + both sticks, horizon step 0) is also
stashed in `ModelMeta.extras["gamepad"]` so the runner can score accuracy-vs-gold
against the scenario's `gold_action.json` sidecar — the lossy ActionSequence is
NOT used for accuracy.

Transport is injected (`NitrogenClient` protocol) so the reasoner is fully
unit-testable on CPU with a fake client; the real `ZmqNitrogenClient` lazily
imports NitroGen + pyzmq only when constructed.
"""

from __future__ import annotations

import io
import json
import time
from typing import Any, Protocol

from vlm_pipeline.adapters import Gamepad, gamepad_to_action_sequence
from vlm_pipeline.config import NitrogenConfig
from vlm_pipeline.schemas import ContextTurn, ModelMeta

# Model output button order — NitroGen's `nitrogen.shared.BUTTON_ACTION_TOKENS`,
# lowercased. The model emits 21; the dataset (gold) has 17. The 4 model-only
# buttons (right_bottom/left/right/up) have no gold counterpart and are dropped
# from the lossy ActionSequence (still kept in extras for the raw record).
MODEL_BUTTON_TOKENS: tuple[str, ...] = (
    "back", "dpad_down", "dpad_left", "dpad_right", "dpad_up", "east", "guide",
    "left_shoulder", "left_thumb", "left_trigger", "north", "right_bottom",
    "right_left", "right_right", "right_shoulder", "right_thumb", "right_trigger",
    "right_up", "south", "start", "west",
)


class NitrogenClient(Protocol):
    """Minimal transport surface the reasoner needs (subset of NitroGen's ModelClient)."""

    def predict(self, image: Any, *, game_id: str | None = None, seed: int | None = None) -> dict:
        """Return {'j_left': ndarray, 'j_right': ndarray, 'buttons': ndarray}."""

    def reset(self) -> None: ...


def _first_step(value: Any) -> list[float]:
    """Coerce a model output array to the horizon's first step as a flat float list.

    Handles (horizon, dim), (dim,), and scalars uniformly without requiring numpy.
    """
    arr = value
    # numpy array → list; leave python sequences as-is.
    if hasattr(arr, "tolist"):
        arr = arr.tolist()
    if isinstance(arr, (int, float)):
        return [float(arr)]
    if arr and isinstance(arr[0], (list, tuple)):  # (horizon, dim) → first step
        arr = arr[0]
    return [float(x) for x in arr]


def gamepad_from_prediction(pred: dict) -> Gamepad:
    """Build a Gamepad from the model's prediction dict (horizon step 0)."""
    buttons_vec = _first_step(pred.get("buttons", []))
    j_left = _first_step(pred.get("j_left", [0.0, 0.0]))
    j_right = _first_step(pred.get("j_right", [0.0, 0.0]))

    buttons = {
        name: float(buttons_vec[i])
        for i, name in enumerate(MODEL_BUTTON_TOKENS)
        if i < len(buttons_vec)
    }
    return Gamepad(
        buttons=buttons,
        j_left=(j_left[0], j_left[1]) if len(j_left) >= 2 else (0.0, 0.0),
        j_right=(j_right[0], j_right[1]) if len(j_right) >= 2 else (0.0, 0.0),
    )


class NitrogenReasoner:
    def __init__(self, config: NitrogenConfig, *, client: NitrogenClient | None = None) -> None:
        self._cfg = config
        self._client = client if client is not None else _make_zmq_client(config)

    def generate(
        self,
        *,
        image: bytes,
        instruction: str,
        history: list[ContextTurn],
        deadline_ms: int,
        game_id: str | None = None,
    ) -> tuple[str, ModelMeta, float | None]:
        frame = _decode_frame(image)
        game = game_id or self._cfg.game_id

        t0 = time.perf_counter()
        pred = self._client.predict(frame, game_id=game, seed=self._cfg.seed)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        pad = gamepad_from_prediction(pred)
        seq = gamepad_to_action_sequence(
            pad,
            move_scale=self._cfg.move_scale,
            rationale="NitroGen flow-matching policy action (lossy view).",
        )
        raw = json.dumps(seq.model_dump(mode="json"))

        meta = ModelMeta(
            framework="nitrogen",
            model_id=self._cfg.model_id,
            extras={
                "game_id": game,
                "seed": self._cfg.seed,
                # Faithful action for accuracy-vs-gold (NOT the lossy ActionSequence).
                "gamepad": {
                    "buttons": pad.buttons,
                    "j_left": list(pad.j_left),
                    "j_right": list(pad.j_right),
                },
                "infer_ms": infer_ms,
                # Token fields are LLM-only; absent here so the runner records them as N/A.
            },
        )
        # ttft is undefined for a one-shot denoising policy (no token stream).
        return raw, meta, None

    def reset(self) -> None:
        self._client.reset()


def _decode_frame(image: bytes) -> Any:
    """Decode encoded image bytes to a (H, W, 3) RGB uint8 ndarray."""
    import numpy as np
    from PIL import Image

    img = Image.open(io.BytesIO(image)).convert("RGB")
    return np.asarray(img)


def _make_zmq_client(config: NitrogenConfig) -> NitrogenClient:
    """Construct the real ZMQ client wrapping NitroGen's ModelClient (lazy import)."""
    return ZmqNitrogenClient(config)


class ZmqNitrogenClient:
    """Adapter over NitroGen's `nitrogen.inference_client.ModelClient`.

    NitroGen's stock `predict` takes only an image and selects the game
    interactively at server startup. We send `game_id`/`seed` in the request
    dict too; that is forward-compatible with the non-interactive server patch
    documented in docs/findings/nitrogen-serve-noninteractive.md (the stock
    server ignores the extra keys). The endpoint is parsed from
    `tcp://host:port`.
    """

    def __init__(self, config: NitrogenConfig) -> None:
        try:
            from nitrogen.inference_client import ModelClient
        except ImportError as e:
            raise ImportError(
                "NitroGen not importable. Install it (pip install -e .) in the "
                "serving env; this client is only used on the GPU instance."
            ) from e

        host, port = _parse_zmq_url(config.base_url)
        self._client = ModelClient(host=host, port=port)
        self._client.timeout_ms = config.timeout_ms

    def predict(self, image: Any, *, game_id: str | None = None, seed: int | None = None) -> dict:
        # NitroGen's ModelClient.predict(image) only sends the image; richer
        # conditioning rides on the request via a thin override.
        import pickle

        request = {"type": "predict", "image": image, "game_id": game_id, "seed": seed}
        self._client.socket.send(pickle.dumps(request))
        response = pickle.loads(self._client.socket.recv())
        if response.get("status") != "ok":
            raise RuntimeError(f"NitroGen server error: {response.get('message')}")
        return response["pred"]

    def reset(self) -> None:
        self._client.reset()


def _parse_zmq_url(url: str) -> tuple[str, int]:
    """tcp://host:port -> (host, port)."""
    rest = url.split("://", 1)[-1]
    host, _, port = rest.partition(":")
    return host or "localhost", int(port or 5555)
