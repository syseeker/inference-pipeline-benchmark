"""VLM reasoner interface.

Every backend (NIM, vLLM, SGLang, TRT-LLM, Triton) implements this. The
pipeline only sees `VlmReasoner`; backend swapping is the only thing that
should change between framework benchmarks.
"""

from __future__ import annotations

from typing import Protocol

from vlm_pipeline.schemas import ContextTurn, ModelMeta


class VlmReasoner(Protocol):
    def generate(
        self,
        *,
        image: bytes,
        instruction: str,
        history: list[ContextTurn],
        deadline_ms: int,
        game_id: str | None = None,
    ) -> tuple[str, ModelMeta, float | None]:
        """Run the VLM and return `(raw_text, model_meta, ttft_ms)`.

        - `raw_text` is whatever the model produced; the decoder is
          responsible for parsing it into an `ActionSequence`.
        - `ttft_ms` is time-to-first-token if the backend supports
          streaming, else `None`. Total reasoner time is measured by the
          orchestrator, not by the backend.
        - `game_id` conditions policy backends (NitroGen) on a game; text-driven
          VLM reasoners ignore it.
        """
