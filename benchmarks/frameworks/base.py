"""Framework adapter contract.

Each framework adapter wraps a backend so the runner can drive it the
same way. Adapters do NOT contain the timing loop — they only expose the
single-request call. The runner is responsible for concurrency, warm-up,
and percentile calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class BenchmarkRequest:
    image: bytes
    instruction: str
    history_text: list[str] = field(default_factory=list)
    deadline_ms: int = 1500


@dataclass
class SingleCallResult:
    raw_text: str
    ttft_ms: float | None
    e2e_ms: float
    framework_knobs_observed: dict[str, Any] = field(default_factory=dict)


class BenchmarkAdapter(Protocol):
    name: str  # "vllm" | "sglang" | "trtllm" | "modelopt" | "triton"

    def setup(self, *, model: str, quantization: str | None, knobs: dict[str, Any]) -> None:
        """Bring up the backend. Idempotent — calling twice with the same
        args is a no-op. Records framework version + active knobs."""

    def teardown(self) -> None:
        """Release the backend. Adapter must be safe to setup() again after."""

    def call(self, req: BenchmarkRequest) -> SingleCallResult:
        """Run one request end-to-end through the reasoner stage."""

    def framework_version(self) -> str:
        """Pinned framework version string for the result row."""
