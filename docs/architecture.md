# Architecture

## Today (v0): VLM-only

The first cut is a single VLM call against a NIM endpoint. The CV encoder
and the executor are passthrough; the decoder coerces output into the
`ActionSequence` schema; the validator does shape + safety checks.

```
┌─────────────────────────────────────────────────────────────────┐
│ PipelineRequest                                                 │
│   image (PIL or bytes), context_history (list[Turn]),           │
│   instruction (str), session_id, request_id, deadline_ms        │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
       ┌────────────┐
       │  Vision    │   v0: passthrough (returns the image bytes)
       │  encoder   │   v1: TRT-optimised CV/encoder front-end
       └────────────┘
              │
              ▼
       ┌────────────┐
       │  VLM       │   NimQwenVLReasoner | VllmReasoner |
       │  reasoner  │   SglangReasoner | TrtLlmReasoner
       └────────────┘
              │  raw text (json or grammar-constrained tokens)
              ▼
       ┌────────────┐
       │  Action    │   Parses raw output into ActionSequence
       │  decoder   │   (json-schema or EBNF-constrained at gen time)
       └────────────┘
              │
              ▼
       ┌────────────┐
       │  Safety    │   shape check + per-command policy + rate limit
       │  validator │   (rejects out-of-grammar / unsafe sequences)
       └────────────┘
              │
              ▼
       ┌────────────┐
       │  Executor  │   v0: dry-run that records the sequence
       │            │   v1: dispatches commands to the host
       └────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│ PipelineResponse                                                │
│   actions: ActionSequence, validations, latency_breakdown,      │
│   model_meta, was_executed                                      │
└─────────────────────────────────────────────────────────────────┘
```

## v1+: split CV ↔ VLM

Once a CV encoder lands (e.g. a small vision tower exported to TensorRT or
a dedicated detector/segmenter), the encoder pre-processes the image into a
compact representation (embedding tensor, structured scene graph, or both).
The VLM reasoner consumes that representation alongside text instead of
re-encoding pixels every frame. This is where Triton ensembles enter:

- **Triton ensemble graph**: `cv_encoder` (TRT) → `vlm_reasoner` (TRT-LLM)
  → `decoder` (Python BLS) → `validator` (Python BLS) → response.
- The CV encoder runs on the GPU's tensor cores; the VLM reasoner shares
  the same GPU but uses a separate instance group.
- KV-cache is owned by the VLM reasoner; the CV encoder is stateless.

## Why this shape

- **Replaceable backends.** The reasoner is the only stage that changes
  when we swap NIM → vLLM → SGLang → TRT-LLM. Everything else stays put.
- **Constrained decoding.** The decoder is the choke point that converts
  free-form text into a typed `ActionSequence`. If we keep output strictly
  schema-bound, the validator becomes a fast structural check rather than
  an LLM judge.
- **Safety after decoding, not before.** We never trust the raw model
  output; the validator is the last gate before execution.
- **Latency budget is per-stage.** Each stage records its own timing into
  `LatencyBreakdown`, which feeds the success metrics (see
  [metrics.md](metrics.md)).

## Open questions tracked in this scaffold

- Does the action grammar warrant **EBNF-constrained sampling** (SGLang /
  vLLM `guided_grammar`) or is **JSON-schema** enough?
- How aggressive can KV-cache reuse be across consecutive frames in the
  same session? `RadixAttention` (SGLang) vs. prefix caching (vLLM) is one
  of the things the benchmark must answer.
- Where does the CV encoder live — same GPU as the VLM (Triton ensemble),
  or a dedicated CPU/iGPU pre-step?
