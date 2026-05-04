"""SGLang benchmark adapter — placeholder.

Implementation plan:
- Talk to an SGLang server.
- Use `regex` / `ebnf` constrained decoding so the action grammar is
  enforced at sample time. Compare validity rate against vLLM's
  `guided_json` and against unconstrained.
- Pull `radix_cache_hit_rate` from SGLang metrics.
"""

from __future__ import annotations

from typing import Any

from benchmarks.frameworks.base import BenchmarkRequest, SingleCallResult


class SglangAdapter:
    name = "sglang"

    def __init__(self) -> None:
        self._knobs: dict[str, Any] = {}

    def setup(self, *, model: str, quantization: str | None, knobs: dict[str, Any]) -> None:
        self._model = model
        self._quant = quantization
        self._knobs = dict(knobs)

    def teardown(self) -> None:
        return

    def call(self, req: BenchmarkRequest) -> SingleCallResult:
        raise NotImplementedError("SglangAdapter.call: implement once SGLang server is wired up.")

    def framework_version(self) -> str:
        try:
            import sglang  # noqa: F401

            return getattr(__import__("sglang"), "__version__", "unknown")
        except ImportError:
            return "not-installed"
