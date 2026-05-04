"""Constrained action-command decoder.

v0: parses the JSON the reasoner produced. Returns `(ActionSequence, None)`
on success, `(None, error_message)` on parse/shape failure — never raises
for routine bad output, since malformed model output is a metric, not an
exception.

v1+: feeds an EBNF / JSON-schema grammar back into the reasoner so the
backend constrains decoding at sample time (SGLang `regex`/`ebnf`,
vLLM `guided_json`/`guided_grammar`, TRT-LLM logits processors).
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from vlm_pipeline.schemas import ActionSequence


class ActionDecoder:
    def decode(self, raw: str) -> tuple[ActionSequence | None, str | None]:
        if not raw:
            return None, "empty model output"
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            return None, f"json parse: {e}"
        try:
            return ActionSequence.model_validate(obj), None
        except ValidationError as e:
            return None, f"schema: {e.errors(include_url=False)[:2]}"
