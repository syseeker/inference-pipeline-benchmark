"""Vision encoder stage.

v0: passthrough. The VLM reasoner ingests pixels directly.
v1: TensorRT-optimised vision tower (or a separate detector/segmenter)
    that emits a compact representation the VLM consumes alongside text.
"""

from __future__ import annotations

from typing import Protocol


class VisionEncoder(Protocol):
    def encode(self, image: bytes) -> bytes:
        """Return either the original bytes or a serialised compact tensor.

        The reasoner negotiates the on-wire format with its encoder; today
        we keep this simple by treating the encoder as a no-op.
        """


class PassthroughVisionEncoder:
    def encode(self, image: bytes) -> bytes:
        return image


class TrtVisionEncoder:
    """Placeholder for the TRT-optimised CV stage.

    Will load a TRT engine for the vision tower and emit an embedding
    tensor the downstream reasoner can splice into its prompt.
    """

    def __init__(self, engine_path: str) -> None:
        self.engine_path = engine_path

    def encode(self, image: bytes) -> bytes:
        raise NotImplementedError(
            "TrtVisionEncoder is a placeholder; implement once the CV tower lands."
        )
