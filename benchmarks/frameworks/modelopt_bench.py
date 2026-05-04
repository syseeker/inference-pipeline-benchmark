"""ModelOpt benchmark adapter — placeholder.

ModelOpt is not a serving framework; it's a quantisation/calibration
toolchain that *feeds* TRT-LLM (and others). This adapter therefore
records the **quant accuracy delta** workflow rather than per-call
latency:

    1. Load BF16 baseline checkpoint.
    2. Run the validator suite, record baseline accuracy.
    3. Apply ModelOpt FP8 / INT8 calibration.
    4. Run the validator suite again, record quant accuracy.
    5. Emit (baseline - quant) into `quant_accuracy_delta`.

Per-call latency is captured by whichever serving adapter consumes the
quantised checkpoint (typically TRT-LLM).
"""

from __future__ import annotations

from typing import Any

from benchmarks.frameworks.base import BenchmarkRequest, SingleCallResult


class ModelOptAdapter:
    name = "modelopt"

    def setup(self, *, model: str, quantization: str | None, knobs: dict[str, Any]) -> None:
        self._model = model
        self._quant = quantization
        self._knobs = dict(knobs)

    def teardown(self) -> None:
        return

    def call(self, req: BenchmarkRequest) -> SingleCallResult:
        raise NotImplementedError(
            "ModelOptAdapter.call is not a per-request adapter. "
            "Use the calibration entrypoint and read quant_accuracy_delta off the result."
        )

    def framework_version(self) -> str:
        try:
            import modelopt  # noqa: F401

            return getattr(__import__("modelopt"), "__version__", "unknown")
        except ImportError:
            return "not-installed"
