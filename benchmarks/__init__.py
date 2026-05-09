"""Benchmark harness for the VLM-to-action pipeline.

`benchmarks.runner` drives the real `vlm_pipeline.Pipeline` over every
scenario under `tests/smoke/scenarios/` against a chosen backend
(vllm | sglang | trtllm) and writes a `BenchmarkResult` row plus
per-scenario JSONs. `benchmarks.summary` aggregates them into
`benchmarks/results/<gpu>/summary.md`.

The per-backend "adapter" lives in `vlm_pipeline.reasoners.*` — the same
abstraction the production pipeline uses. The harness does not
re-introduce a parallel one.
"""
