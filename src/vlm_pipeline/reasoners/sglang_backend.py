"""SGLang reasoner — server mode via OpenAI-compatible API.

Points at the SGLang server (default: http://localhost:30000/v1). The
model id is auto-discovered from /v1/models unless SGLANG_MODEL is set or
SglangConfig.model is overridden.

SGLang's structured-output features (JSON schema, regex, EBNF) are the
key reason it's in the roster. For the smoke test we use JSON-mode; the
full benchmark harness should pass the action schema via extra_body.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.request

from vlm_pipeline.config import SglangConfig
from vlm_pipeline.schemas import ContextTurn, ModelMeta

_SYSTEM_PROMPT = (
    "You are a low-level control planner. Given an image, a short history, "
    "and a user instruction, produce a short JSON object of the form "
    '{"commands":[{"type":"<noop|move|click|keypress|wait|say>","args":{...}}], '
    '"rationale":"..."}. '
    "Use only the listed command types. Keep `commands` short (<= 8). "
    "Do not include any text outside the JSON object."
)


def _discover_model(base_url: str, timeout_s: float = 5.0) -> str:
    url = base_url.rstrip("/") + "/models"
    with urllib.request.urlopen(url, timeout=timeout_s) as r:
        data = json.loads(r.read())
    models = data.get("data", [])
    if not models:
        raise RuntimeError(f"No models found at {url}")
    return models[0]["id"]


def _media_type(image: bytes) -> str:
    return "image/png" if image[:4] == b"\x89PNG" else "image/jpeg"


class SglangReasoner:
    def __init__(self, config: SglangConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "Install the 'sglang' extra: pip install -e '.[sglang,dev]'"
            ) from e

        self._cfg = config
        self._model = config.model or _discover_model(config.base_url)
        self._client = OpenAI(api_key="none", base_url=config.base_url)

    def generate(
        self,
        *,
        image: bytes,
        instruction: str,
        history: list[ContextTurn],
        deadline_ms: int,
    ) -> tuple[str, ModelMeta, float | None]:
        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for turn in history:
            messages.append({"role": turn.role, "content": turn.text})

        b64 = base64.b64encode(image).decode("ascii")
        mime = _media_type(image)
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }
        )

        t_start = time.perf_counter()
        ttft_ms: float | None = None
        chunks: list[str] = []

        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
            timeout=max(deadline_ms / 1000.0, 120.0),
            response_format={"type": "json_object"},
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t_start) * 1000.0
                chunks.append(delta)

        raw = "".join(chunks).strip()
        if not raw.startswith("{"):
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end != -1:
                raw = raw[start : end + 1]
        json.loads(raw)

        return raw, ModelMeta(framework="sglang", model_id=self._model), ttft_ms
