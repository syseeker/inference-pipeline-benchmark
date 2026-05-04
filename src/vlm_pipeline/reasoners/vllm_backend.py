"""vLLM reasoner — placeholder.

Two ways to wire this up:
1. **Server mode** (preferred for benchmarks): hit the vLLM OpenAI-
   compatible server at VLLM_BASE_URL. Reuse most of the NIM client
   code; the only diff is `base_url`.
2. **In-process mode**: instantiate `vllm.LLM(...)` directly. Lower
   latency floor (no HTTP), heavier env requirements.

Implementation deferred until the harness actually needs it.
"""

from __future__ import annotations

from vlm_pipeline.config import VllmConfig
from vlm_pipeline.schemas import ContextTurn, ModelMeta


class VllmReasoner:
    def __init__(self, config: VllmConfig) -> None:
        self._cfg = config

    def generate(
        self,
        *,
        image: bytes,
        instruction: str,
        history: list[ContextTurn],
        deadline_ms: int,
    ) -> tuple[str, ModelMeta, float | None]:
        raise NotImplementedError("VllmReasoner is a placeholder for the benchmark harness.")
