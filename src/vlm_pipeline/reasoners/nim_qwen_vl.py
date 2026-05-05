"""NIM-hosted Qwen-VL reasoner.

NIM exposes an OpenAI-compatible chat-completions API. We use the official
`openai` SDK with `base_url=NIM_BASE_URL` so the same code works against
`integrate.api.nvidia.com` and a self-hosted NIM container.

Streaming is on so we can record TTFT. The action grammar is enforced via
a system prompt + JSON-mode response format; the decoder will still
validate.
"""

from __future__ import annotations

import base64
import json
import time

from vlm_pipeline.config import NimConfig
from vlm_pipeline.schemas import ContextTurn, ModelMeta


class NimModelNotFoundError(RuntimeError):
    """Raised when the NIM endpoint returns 404 for the requested model.

    This almost always means `NIM_MODEL` doesn't match any model id served
    at `NIM_BASE_URL`. NIM cloud's catalogue rotates; verify with::

        curl -s -H "Authorization: Bearer $NIM_API_KEY" \
             "$NIM_BASE_URL/models" | jq '.data[].id'
    """

_SYSTEM_PROMPT = (
    "You are a low-level control planner. Given an image, a short history, "
    "and a user instruction, produce a short JSON object of the form "
    '{"commands":[{"type":"<noop|move|click|keypress|wait|say>","args":{...}}], '
    '"rationale":"..."}. '
    "Use only the listed command types. Keep `commands` short (<= 8). "
    "Do not include any text outside the JSON object."
)


class NimQwenVlReasoner:
    def __init__(self, config: NimConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "Install the 'nim' extra: pip install -e '.[nim]' (provides openai>=1.30)"
            ) from e

        if not config.api_key:
            raise RuntimeError("NIM_API_KEY is not set; cannot reach NIM endpoint.")

        self._cfg = config
        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url)

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
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        )

        t_start = time.perf_counter()
        ttft_ms: float | None = None
        chunks: list[str] = []

        try:
            stream = self._client.chat.completions.create(
                model=self._cfg.model,
                messages=messages,
                stream=True,
                timeout=self._cfg.timeout_s,
                response_format={"type": "json_object"},
            )
        except Exception as e:  # openai.NotFoundError, APIStatusError, etc.
            status = getattr(e, "status_code", None)
            if status == 404 or "404" in str(e) or "not found" in str(e).lower():
                raise NimModelNotFoundError(
                    f"NIM returned 404 for model '{self._cfg.model}' at "
                    f"{self._cfg.base_url}. The model id likely does not exist "
                    f"on this endpoint. List available ids with:\n"
                    f"  curl -s -H 'Authorization: Bearer $NIM_API_KEY' "
                    f"{self._cfg.base_url.rstrip('/')}/models | jq '.data[].id'\n"
                    f"Then export NIM_MODEL=<id> and retry."
                ) from e
            raise

        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t_start) * 1000.0
                chunks.append(delta)

        raw = "".join(chunks).strip()
        # Defensive: if the model wrapped the JSON in prose, try to slice it.
        if not raw.startswith("{"):
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                raw = raw[start : end + 1]
        # Validate it's at least JSON-shaped before returning. The decoder
        # does the strict typed parse.
        json.loads(raw)

        meta = ModelMeta(framework="nim", model_id=self._cfg.model)
        return raw, meta, ttft_ms
