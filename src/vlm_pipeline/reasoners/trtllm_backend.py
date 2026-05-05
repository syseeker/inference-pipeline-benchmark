"""TensorRT-LLM reasoner — placeholder.

TRT engine-compiled path. Requires:
- A TRT-LLM engine for the language tower (built per GPU + batch shape).
- A TRT engine for the vision tower (or kept fused if the model supports).
- ModelOpt-calibrated weights for the FP8/INT8 path.

Implementation deferred until the harness actually needs it.
"""

from __future__ import annotations

from vlm_pipeline.config import TrtLlmConfig
from vlm_pipeline.schemas import ContextTurn, ModelMeta


class TrtLlmReasoner:
    def __init__(self, config: TrtLlmConfig) -> None:
        self._cfg = config

    def generate(
        self,
        *,
        image: bytes,
        instruction: str,
        history: list[ContextTurn],
        deadline_ms: int,
    ) -> tuple[str, ModelMeta, float | None]:
        raise NotImplementedError("TrtLlmReasoner is a placeholder for the benchmark harness.")
