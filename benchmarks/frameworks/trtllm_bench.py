"""TensorRT-LLM benchmark adapter — placeholder.

Implementation plan:
- Build the LLM engine with `trtllm-build` per (gpu, batch shape, FP8/INT8).
- Build/load the vision tower as a separate TRT engine.
- Drive via `tensorrt_llm.LLM` (in-process) or via a Triton ensemble (the
  triton adapter handles that path).
- Toggle `paged_kv_cache`, `enable_block_reuse`, `cuda_graph` and record
  which were active.
"""

from __future__ import annotations

from typing import Any

from benchmarks.frameworks.base import BenchmarkRequest, SingleCallResult


class TrtLlmAdapter:
    name = "trtllm"

    def __init__(self) -> None:
        self._knobs: dict[str, Any] = {}

    def setup(self, *, model: str, quantization: str | None, knobs: dict[str, Any]) -> None:
        self._model = model
        self._quant = quantization
        self._knobs = dict(knobs)

    def teardown(self) -> None:
        return

    def call(self, req: BenchmarkRequest) -> SingleCallResult:
        raise NotImplementedError("TrtLlmAdapter.call: implement once the engine is built.")

    def framework_version(self) -> str:
        try:
            import tensorrt_llm  # noqa: F401

            return getattr(__import__("tensorrt_llm"), "__version__", "unknown")
        except ImportError:
            return "not-installed"
