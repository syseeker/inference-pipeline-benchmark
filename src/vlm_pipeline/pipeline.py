"""End-to-end pipeline orchestrator.

    encoder → reasoner → decoder → validator → executor

The orchestrator is intentionally thin: each stage is a swappable object,
each stage records its own latency, and the orchestrator never touches raw
model bytes. v0 ships with the encoder + executor as passthroughs and the
reasoner pluggable across NIM / vLLM / SGLang / TRT-LLM.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from vlm_pipeline.config import PipelineConfig
from vlm_pipeline.decoders.action_decoder import ActionDecoder
from vlm_pipeline.encoders.vision_encoder import PassthroughVisionEncoder, VisionEncoder
from vlm_pipeline.executors.executor import DryRunExecutor, Executor
from vlm_pipeline.reasoners.base import VlmReasoner
from vlm_pipeline.schemas import (
    ActionSequence,
    ContextTurn,
    LatencyBreakdown,
    ModelMeta,
    ValidationReport,
)
from vlm_pipeline.validators.safety_validator import SafetyValidator


@dataclass
class PipelineRequest:
    image: bytes  # encoded (jpeg/png). Encoder is responsible for decoding.
    instruction: str
    context_history: list[ContextTurn] = field(default_factory=list)
    session_id: str | None = None
    request_id: str | None = None
    deadline_ms: int | None = None
    # Game conditioning for policy backends (e.g. NitroGen). Text-driven VLM
    # reasoners ignore it; carried here so it survives into reasoner.generate().
    game_id: str | None = None


@dataclass
class PipelineResponse:
    actions: ActionSequence | None
    validation: ValidationReport
    latency: LatencyBreakdown
    model_meta: ModelMeta | None
    was_executed: bool
    error: str | None = None


class Pipeline:
    def __init__(
        self,
        reasoner: VlmReasoner,
        *,
        config: PipelineConfig | None = None,
        encoder: VisionEncoder | None = None,
        decoder: ActionDecoder | None = None,
        validator: SafetyValidator | None = None,
        executor: Executor | None = None,
    ) -> None:
        self.config = config or PipelineConfig.from_env()
        self.encoder = encoder or PassthroughVisionEncoder()
        self.reasoner = reasoner
        self.decoder = decoder or ActionDecoder()
        self.validator = validator or SafetyValidator()
        self.executor = executor or DryRunExecutor()

    def run(self, req: PipelineRequest) -> PipelineResponse:
        latency = LatencyBreakdown()
        t_start = time.perf_counter()

        # 1. Vision encode (today: passthrough; tomorrow: TRT vision tower).
        t0 = time.perf_counter()
        encoded_image = self.encoder.encode(req.image)
        latency.vision_encoder_ms = _ms_since(t0)

        # 2. VLM reason — produces raw text (or grammar-constrained tokens).
        t0 = time.perf_counter()
        raw, meta, ttft_ms = self.reasoner.generate(
            image=encoded_image,
            instruction=req.instruction,
            history=req.context_history[-self.config.max_history_turns :],
            deadline_ms=req.deadline_ms or self.config.deadline_ms,
            game_id=req.game_id,
        )
        latency.reasoner_total_ms = _ms_since(t0)
        latency.reasoner_ttft_ms = ttft_ms

        # 3. Decode raw → ActionSequence. Returns (None, error) on parse failure.
        t0 = time.perf_counter()
        actions, decode_err = self.decoder.decode(raw)
        latency.decoder_ms = _ms_since(t0)

        if actions is None:
            latency.total_ms = _ms_since(t_start)
            return PipelineResponse(
                actions=None,
                validation=ValidationReport(
                    schema_valid=False,
                    safe=False,
                    notes=[f"decoder: {decode_err}"],
                ),
                latency=latency,
                model_meta=meta,
                was_executed=False,
                error=decode_err,
            )

        # 4. Validate.
        t0 = time.perf_counter()
        report = self.validator.validate(actions)
        latency.validator_ms = _ms_since(t0)

        # 5. Execute (dry-run by default).
        was_executed = False
        if report.schema_valid and report.safe:
            t0 = time.perf_counter()
            was_executed = self.executor.execute(actions)
            latency.executor_ms = _ms_since(t0)

        latency.total_ms = _ms_since(t_start)
        return PipelineResponse(
            actions=actions,
            validation=report,
            latency=latency,
            model_meta=meta,
            was_executed=was_executed,
        )


def _ms_since(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0
