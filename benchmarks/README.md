# benchmarks/

Framework adapters + a runner. Adapters are scaffolded; loops are
intentionally `NotImplementedError` so reviewers can audit the shape
before any heavy bring-up.

## Layout

```
benchmarks/
├── runner.py                  # CLI: pick framework + gpu + model, run, write results
├── metrics.py                 # BenchmarkResult, percentile helpers, validity scoring
├── frameworks/
│   ├── base.py                # BenchmarkAdapter protocol + shared timing helpers
│   ├── vllm_bench.py
│   ├── sglang_bench.py
│   ├── trtllm_bench.py
│   ├── modelopt_bench.py
│   └── triton_bench.py
├── configs/
│   ├── rtx5090.yaml
│   ├── rtx_pro6000.yaml
│   └── h200.yaml
└── results/                   # per-GPU, per-run outputs (raw is gitignored)
```

## How to add a new framework

1. Add a module under `frameworks/` that implements `BenchmarkAdapter`.
2. Make sure `setup()` records framework version + relevant knobs into the
   `BenchmarkResult.framework_knobs` field.
3. Wire the new name into `runner.py`'s adapter table.
4. Add framework-specific tunables (engine paths, prefix-cache size, etc.)
   into the `configs/<gpu>.yaml` files where they matter.

## Decision metrics, not tokens/sec

See [../docs/metrics.md](../docs/metrics.md). The runner computes:

- valid command-sequence latency (e2e)
- command success rate (executor accepted)
- safety / grammar validity rate
- p95 / p99 stability

Diagnostics (TTFT, ITL, mem-bw, KV-cache hit rate, CUDA-graph delta) are
recorded for explanation, not for pass/fail.
