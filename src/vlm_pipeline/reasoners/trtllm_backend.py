"""TensorRT-LLM reasoner — `trtllm-serve --backend pytorch` over HTTP.

`trtllm-serve` exposes the same OpenAI-compatible chat-completions API as
vLLM/SGLang. Default port is 8002 to avoid clashing with vLLM (8000) and
SGLang (30000) when all three are profiled on the same host.

JSON validity caveat: TRT-LLM 1.2.1's pytorch backend does not enforce
schema-guided decoding for Qwen3-VL. Enabling xgrammar
(`guided_decoding_backend: xgrammar`) triggers a startup crash because
`Qwen3VLModel` (the multimodal wrapper) doesn't expose `vocab_size_padded`,
which the guided-decoder constructor reads unconditionally
(see py_executor_creator.py:504). So we send `response_format=
{"type": "json_object"}` — accepted by the server, but treated as a
prompt-level convention rather than a hard constraint. Expect a lower
`valid=True` rate than vLLM/SGLang on this backend; that is the
measurement, not a bug in the harness.

Start the server before running tests (multimodal requires kv-cache reuse
off; do NOT enable xgrammar with Qwen3-VL on TRT-LLM 1.2.1):
    source .venv-trtllm/bin/activate
    cat > /tmp/trtllm-vlm.yml <<'EOF'
    kv_cache_config:
      enable_block_reuse: false
    EOF
    trtllm-serve Qwen/Qwen3-VL-8B-Instruct-FP8 \\
      --backend pytorch --port 8002 \\
      --extra_llm_api_options /tmp/trtllm-vlm.yml
"""

from __future__ import annotations

import base64
import json
import time
import urllib.request

from vlm_pipeline.config import TrtLlmConfig
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


class TrtLlmReasoner:
    def __init__(self, config: TrtLlmConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "Install the 'dev' extra: pip install -e '.[dev]' (provides openai)"
            ) from e

        self._cfg = config
        self._model = _discover_model(config.base_url)
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
            framework="trtllm",
            model_id=self._model,
            extras={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )
        return raw, meta, ttft_ms
