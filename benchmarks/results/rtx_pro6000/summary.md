# Benchmark summary — rtx_pro6000

Decision metrics drive go/no-go (see docs/metrics.md). Diagnostics explain *why* a decision metric moved. Cross-run deltas pair `run_label` variants against `baseline` for the same (framework, model).

---

## 1. Contention analysis

4 solo baseline(s), 1 contention window(s). Ratios are `contention / solo`; ▲ = degraded, ▼ = improved, ≈ = within 5%.

- **Offered rps** — the rate the load generator was told to send.
- **Achieved rps** — the rate it actually managed. Below offered means the tenant could not keep up; that is the safe-operating-envelope boundary, not a measurement error.
- **Throughput kept** — achieved-with-a-neighbour ÷ achieved-alone. `1.00×` = the neighbour cost nothing; `0.60×` = it cost 40% of throughput.
- **e2e / TTFT ratios** — latency with a neighbour ÷ latency alone. Above 1.00× is slower.

### 1a. Degradation table

| Colocation             | Phase | Tenant   | Model                    | Backend | Offered rps | Achieved rps | Throughput kept |  e2e p50 |  e2e p95 |  TTFT p95 |
|-----------------------:|------:|---------:|-------------------------:|--------:|------------:|-------------:|----------------:|---------:|---------:|----------:|
| mix-llm-cv             | 3     | cv       | yolov8-l                 | triton  |        50.0 |         50.8 |          ≈1.00× |   ≈0.99× |   ≈0.98× |       n/a |
| mix-llm-cv             | 3     | llm      | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ≈1.02× |   ≈1.03× |    ≈1.01× |

### 1b. Contention matrix

**e2e p95 degradation (victim × aggressors)**

| Victim model | Aggressors | e2e p95 ratio (mean) |
|---|---|---|
| qwen2.5-7b | yolov8-l | ≈1.03× |
| yolov8-l | qwen2.5-7b | ≈0.98× |

### 1c. Safe-operating envelope

_No envelope crossings detected (achieved_rps ≥ 0.95 × offered across all runs)._

