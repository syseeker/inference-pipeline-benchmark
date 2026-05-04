"""SGLang reasoner — placeholder.

SGLang's structured-output features (JSON schema, regex, EBNF) are the
key reason it's in the roster: we want strict grammar on the action
sequence, not just JSON-mode.

Implementation deferred until the harness actually needs it.
"""

from __future__ import annotations

from vlm_pipeline.config import SglangConfig
from vlm_pipeline.schemas import ContextTurn, ModelMeta


class SglangReasoner:
    def __init__(self, config: SglangConfig) -> None:
        self._cfg = config

    def generate(
        self,
        *,
        image: bytes,
        instruction: str,
        history: list[ContextTurn],
        deadline_ms: int,
    ) -> tuple[str, ModelMeta, float | None]:
        raise NotImplementedError("SglangReasoner is a placeholder for the benchmark harness.")
