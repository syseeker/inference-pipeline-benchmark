"""Video+text reasoner — OpenAI-compatible endpoint (vLLM / SGLang).

Sends a video (local file or remote URL) + text prompt via the multimodal
chat completions API and returns the raw text response.

Unlike the image-based VLM reasoners, the output here is free-form text (not
an ActionSequence JSON blob) — the decoder stage is bypassed. Validation uses
key-phrase coverage against expected.json instead of schema checking.

num_frames controls how many frames the serving framework extracts from the
video before feeding the vision encoder. Set via VideoTextConfig.num_frames or
the VIDEO_NUM_FRAMES env var. Sweep over 4 / 8 / 16 to measure the
latency-vs-quality tradeoff — latency scales roughly linearly with frame count.

Local video files are read and base64-encoded into a data: URL — the same
pattern the image reasoners use for image_url. This avoids vLLM's
--allowed-local-media-path requirement for file:// URLs and works with the
default server config. For remote URLs (https://) the URL is passed directly.

Supported by all VLM backends that serve Qwen3-VL or Gemma-4:
  vLLM   → VIDEO_BASE_URL=http://localhost:8000/v1
  SGLang → VIDEO_BASE_URL=http://localhost:30000/v1
  TRT-LLM (not recommended on Blackwell — arch gaps remain)

Nemotron-Omni is vLLM-only on RTX PRO 6000 (SGLang SM_120 fused-MoE OOM);
the video path works identically — same OpenAI-compatible client.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.request
from pathlib import Path

from vlm_pipeline.config import VideoTextConfig
from vlm_pipeline.schemas import ModelMeta

_SYSTEM_PROMPT = (
    "You are a video analysis assistant. Watch the video carefully and answer "
    "the user's question in clear, factual prose. Be specific about what you "
    "observe — objects, actions, timing, spatial relationships."
)


def _discover_model(base_url: str, timeout_s: float = 5.0) -> str:
    url = base_url.rstrip("/") + "/models"
    with urllib.request.urlopen(url, timeout=timeout_s) as r:
        data = json.loads(r.read())
    models = data.get("data", [])
    if not models:
        raise RuntimeError(f"No models found at {url}")
    return models[0]["id"]


def _to_video_url(video_path: str | None, video_url: str | None, scenario_dir: Path | None) -> str:
    """Resolve to a URL the serving framework can accept.

    Remote URLs (http/https) are passed through directly. Local files are
    read and base64-encoded into a data: URL — identical to how the image
    reasoners handle image_url. This avoids vLLM's --allowed-local-media-path
    requirement for file:// URLs and works with the default server config.
    """
    if video_url:
        return video_url
    if video_path:
        p = Path(video_path)
        if not p.is_absolute() and scenario_dir:
            p = scenario_dir / p
        p = p.resolve()
        data = p.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        mime_suffix = p.suffix.lstrip(".").lower() or "mp4"
        return f"data:video/{mime_suffix};base64,{b64}"
    raise ValueError("scenario must set either video_path or video_url in request.json")


class VideoTextReasoner:
    """Multimodal video+text reasoner over an OpenAI-compatible HTTP endpoint."""

    def __init__(self, config: VideoTextConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "Install the 'vllm' or 'sglang' extra: pip install -e '.[vllm,dev]'"
            ) from e

        self._cfg = config
        self._model = config.model or _discover_model(config.base_url)
        self._client = OpenAI(api_key="none", base_url=config.base_url)

    def generate(
        self,
        *,
        video_path: str | None = None,
        video_url: str | None = None,
        scenario_dir: Path | None = None,
        prompt: str,
        deadline_ms: int,
        num_frames: int | None = None,
    ) -> tuple[str, ModelMeta, float | None]:
        """Run the model on a video + text prompt.

        Returns (raw_text, model_meta, ttft_ms). raw_text is the model's
        free-form response — no JSON decoding; caller does key-phrase coverage.
        """
        url = _to_video_url(video_path, video_url, scenario_dir)
        n_frames = num_frames if num_frames is not None else self._cfg.num_frames

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": url},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]

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
            # num_video_frames is the vLLM / SGLang extra_body key for
            # controlling how many frames are sampled from the video.
            # Qwen3-VL's default is determined by the model config (varies
            # by resolution and sequence-length cap); passing it explicitly
            # pins the sweep knob so results are reproducible.
            extra_body={"num_video_frames": n_frames},
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
        meta = ModelMeta(
            framework="video-text",
            model_id=self._model,
            extras={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "num_frames": n_frames,
                "video_url": url,
            },
        )
        return raw, meta, ttft_ms


def key_phrase_coverage(response: str, key_phrases: list[str]) -> float:
    """Fraction of key_phrases that appear (case-insensitive) in response."""
    if not key_phrases:
        return 1.0
    lower = response.lower()
    matched = sum(1 for kp in key_phrases if kp.lower() in lower)
    return matched / len(key_phrases)
