"""vLLM reasoner — server mode via OpenAI-compatible API.

Points at the vLLM server (default: http://localhost:8000/v1). The model
id is auto-discovered from /v1/models unless VLLM_MODEL is set or
VllmConfig.model is overridden.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.request

from vlm_pipeline.config import VllmConfig
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


class VllmReasoner:
    def __init__(self, config: VllmConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "Install the 'vllm' extra: pip install -e '.[vllm,dev]'"
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
        game_id: str | None = None,  # unused: text VLM is conditioned on the instruction
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
        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            timeout=max(deadline_ms / 1000.0, 120.0),
            response_format={"type": "json_object"},
        )
        for chunk in stream:
            # The final chunk under include_usage carries `.usage` and an empty `.choices`.
            if chunk.choices:
                delta = chunk.choices[0].delta.content if chunk.choices[0].delta else None
                if delta:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t_start) * 1000.0
                    chunks.append(delta)
            if getattr(chunk, "usage", None) is not None:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens

        raw = "".join(chunks).strip()
        if not raw.startswith("{"):
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end != -1:
                raw = raw[start : end + 1]

        meta = ModelMeta(
            framework="vllm",
            model_id=self._model,
            extras={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )
        return raw, meta, ttft_ms
