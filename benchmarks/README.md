# benchmarks/

Scenario-driven benchmark harness. `runner.py` drives the real
`vlm_pipeline.Pipeline` over every scenario under `tests/smoke/scenarios/`
through one backend at a time and writes a `BenchmarkResult` plus
per-scenario JSON rows. `summary.py` rolls them up into `summary.md`.

## Layout

```
benchmarks/
├── runner.py             # CLI: --backend {vllm|sglang|trtllm} --gpu <profile>
├── summary.py            # CLI: --gpu <profile>  → summary.md
├── metrics.py            # BenchmarkResult, LatencySamples, percentile helpers
├── scenario_config.py    # YAML dotted-path resolver used by run_all_scenarios.sh
├── configs/
│   ├── rtx5090.yaml
│   ├── rtx_pro6000.yaml
│   └── h200.yaml
└── results/              # per-GPU, per-run outputs
```

The per-backend "adapter" lives in `src/vlm_pipeline/reasoners/*` — the
same layer the production pipeline uses. Adding a backend means adding a
reasoner there and a `backends.<name>` block in the GPU YAMLs, then
wiring the name into `runner._make_reasoner`.

## Decision metrics, not tokens/sec

See [../docs/metrics.md](../docs/metrics.md). `BenchmarkResult` carries:

- valid command-sequence latency (e2e p50 / p95 / p99)
- command success rate (executor accepted)
- safety / grammar validity rate

Diagnostics (TTFT, ITL, vision-encoder latency, KV-cache hit rate,
CUDA-graph delta) are recorded for explanation, not pass/fail.
