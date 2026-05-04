"""vLLM benchmark adapter — placeholder.

Implementation plan:
- Talk to a vLLM OpenAI-compatible server (`VLLM_BASE_URL`).
- Toggle prefix-cache and CUDA-graph knobs via server flags; record what
  was active in `framework_knobs_observed`.
- Pull `prefix_cache_hits` / `gpu_cache_usage` from `/metrics` (Prometheus
  text format) at the end of the run.
"""

from __future__ import annotations

from typing import Any

from benchmarks.frameworks.base import BenchmarkRequest, SingleCallResult


class VllmAdapter:
    name = "vllm"

    def __init__(self) -> None:
        self._knobs: dict[str, Any] = {}

    def setup(self, *, model: str, quantization: str | None, knobs: dict[str, Any]) -> None:
        self._model = model
        self._quant = quantization
        self._knobs = dict(knobs)

    def teardown(self) -> None:
        return

    def call(self, req: BenchmarkRequest) -> SingleCallResult:
        raise NotImplementedError("VllmAdapter.call: implement once vLLM server is wired up.")

    def framework_version(self) -> str:
        try:
            import vllm  # noqa: F401

            return getattr(__import__("vllm"), "__version__", "unknown")
        except ImportError:
            return "not-installed"
