"""TensorRT + Triton ensemble adapter — placeholder.

This is the "production composition" path:

    Triton ensemble: cv_encoder (TRT) → vlm_reasoner (TRT-LLM)
                     → decoder (Python BLS) → validator (Python BLS)

The adapter calls the ensemble through tritonclient gRPC and records
per-stage latency reported by the ensemble's `INFER_RESPONSE_COMPLETE`
trace. This is the only adapter where vision-encoder latency is
first-class — for the other adapters the encoder is fused or absent.
"""

from __future__ import annotations

from typing import Any

from benchmarks.frameworks.base import BenchmarkRequest, SingleCallResult


class TritonAdapter:
    name = "triton"

    def __init__(self) -> None:
        self._knobs: dict[str, Any] = {}

    def setup(self, *, model: str, quantization: str | None, knobs: dict[str, Any]) -> None:
        self._model = model
        self._quant = quantization
        self._knobs = dict(knobs)

    def teardown(self) -> None:
        return

    def call(self, req: BenchmarkRequest) -> SingleCallResult:
        raise NotImplementedError(
            "TritonAdapter.call: implement once the ensemble model_repository is built."
        )

    def framework_version(self) -> str:
        try:
            import tritonclient  # noqa: F401

            return getattr(__import__("tritonclient"), "__version__", "unknown")
        except ImportError:
            return "not-installed"
