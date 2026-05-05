# Metrics — what counts as success

Tokens/sec is **not** the decision metric for this use case.

## Decision metrics

| Metric | Definition | Target |
| --- | --- | --- |
| Valid command-sequence latency | t(image+instruction in) → t(schema-valid sequence out) | meet interactive budget (TBD per customer) |
| Command success rate | fraction of sequences the executor accepts and that achieve the intended outcome | high — exact threshold per task suite |
| Safety / grammar validity | fraction passing the validator on first try (no resampling) | high — close to 1.0 |
| p95 / p99 stability | tail latency of valid command-sequence latency under realistic concurrency | within budget |

A run with great tokens/sec but a low validity rate is **a failed run**.
The decoder must produce something the validator accepts.

## Diagnostic metrics

These help explain *why* a decision metric moves. Not pass/fail on their
own.

- **TTFT** — time to first generated token (after vision encoding).
- **Inter-token latency (ITL)** — average ms between tokens once
  generation has started.
- **Vision encoder latency** — when the CV stage is real, time to embed
  the image. Stays separate from LLM TTFT.
- **Memory bandwidth utilisation** — sampled via DCGM /
  `nvidia-smi dmon -s u` / Nsight Systems. Speaks to the bandwidth thesis.
- **KV-cache hit rate** — framework-reported (vLLM `prefix_cache_hits`,
  SGLang RadixAttention reuse, TRT-LLM KV reuse).
- **CUDA graph delta** — same workload with `enforce_eager=True` vs.
  CUDA-graph-on. Quantifies graph capture impact.
- **Quantisation accuracy loss** — task-level accuracy of FP8/INT8
  vs. BF16 baseline on the validator suite.
- **2× GPU TP efficiency** — `latency(TP=1) / (2 × latency(TP=2))`. > 1 is
  a win, ≤ 1 means PCIe overhead ate the parallelism (relevant for RTX
  5090 since it has no NVLink).

## Reporting shape

Each benchmark cell produces one `BenchmarkResult` row; rows are joined
into a per-GPU summary table that lives in
`benchmarks/results/<gpu>/summary.md`. Raw JSONL traces live in
`benchmarks/results/<gpu>/raw/<run-id>.jsonl` (gitignored — too large).

## Statistical hygiene

- Always report **p50, p95, p99**, not just averages.
- Warm up before timing (CUDA graph capture, KV warm).
- Run with realistic concurrency, not single-stream — concurrent
  pipelines are how the executor will actually use this.
- Hold seed and image set fixed across frameworks so comparisons are
  apples-to-apples.
- Record framework version, GPU, driver, and CUDA in every row.
