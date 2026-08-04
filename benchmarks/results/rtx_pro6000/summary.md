# Benchmark summary — rtx_pro6000

Decision metrics drive go/no-go (see docs/metrics.md). Diagnostics explain *why* a decision metric moved. Cross-run deltas pair `run_label` variants against `baseline` for the same (framework, model).

---

## 1. Contention analysis

8 solo baseline(s), 5 contention window(s). Ratios are `contention / solo`; ▲ = degraded, ▼ = improved, ≈ = within 5%.

- **Offered rps** — the rate the load generator was told to send.
- **Achieved rps** — the rate it actually managed. Below offered means the tenant could not keep up; that is the safe-operating-envelope boundary, not a measurement error.
- **Throughput kept** — achieved-with-a-neighbour ÷ achieved-alone. `1.00×` = the neighbour cost nothing; `0.60×` = it cost 40% of throughput.
- **e2e / TTFT ratios** — latency with a neighbour ÷ latency alone. Above 1.00× is slower.

### 1a. Degradation table

| Colocation             | Phase | Tenant   | Model                    | Backend | Offered rps | Achieved rps | Throughput kept |  e2e p50 |  e2e p95 |  TTFT p95 |
|-----------------------:|------:|---------:|-------------------------:|--------:|------------:|-------------:|----------------:|---------:|---------:|----------:|
| mix-full               | 3     | cv       | yolov8-l                 | triton  |        50.0 |         50.5 |          ≈1.00× |   ▲2.46× |   ▲2.88× |       n/a |
| mix-full               | 3     | ilm      | kosmos-2.5               | triton  |         0.1 |          0.1 |          ≈1.00× |   ▲1.75× |   ▲2.14× |       n/a |
| mix-full               | 3     | llm      | qwen2.5-7b               | vllm    |         4.0 |          3.8 |          ≈0.98× |   ▲1.40× |   ▲1.64× |    ▲1.51× |
| mix-full               | 3     | vlm      | qwen2.5-vl-7b            | vllm    |         1.0 |          1.0 |          ≈0.96× |   ▲2.07× |   ▲2.14× |    ▲1.27× |
| mix-ilm-cv             | 3     | cv       | yolov8-l                 | triton  |        50.0 |         50.5 |          ≈1.00× |   ≈1.00× |   ≈1.02× |       n/a |
| mix-ilm-cv             | 3     | ilm      | kosmos-2.5               | triton  |         0.1 |          0.1 |          ≈1.00× |   ≈1.05× |   ▲1.06× |       n/a |
| mix-llm-cv             | 3     | cv       | yolov8-l                 | triton  |        50.0 |         50.8 |          ≈1.00× |   ≈0.98× |   ≈0.97× |       n/a |
| mix-llm-cv             | 3     | llm      | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ≈1.02× |   ≈1.02× |    ≈1.01× |
| mix-vlm-cv             | 3     | cv       | dinov2-base              | triton  |        50.0 |         50.8 |          ≈1.01× |   ≈0.97× |   ≈0.95× |       n/a |
| mix-vlm-cv             | 3     | vlm      | qwen2.5-vl-7b            | vllm    |         1.0 |          1.0 |          ≈1.00× |   ≈1.03× |   ≈1.02× |    ▲1.13× |
| mix-vlm-ilm            | 3     | ilm      | kosmos-2.5               | triton  |         0.1 |          0.1 |          ≈1.00× |   ≈1.04× |   ≈1.04× |       n/a |
| mix-vlm-ilm            | 3     | vlm      | qwen2.5-vl-7b            | vllm    |         1.0 |          1.0 |          ≈0.96× |   ≈1.04× |   ▲1.11× |    ▲1.08× |

### 1b. Contention matrix

**e2e p95 degradation (victim × aggressors)**

| Victim model | Aggressors | e2e p95 ratio (mean) |
|---|---|---|
| dinov2-base | qwen2.5-vl-7b | ≈0.95× |
| kosmos-2.5 | qwen2.5-7b, qwen2.5-vl-7b, yolov8-l | ▲2.14× |
| kosmos-2.5 | qwen2.5-vl-7b | ≈1.04× |
| kosmos-2.5 | yolov8-l | ▲1.06× |
| qwen2.5-7b | kosmos-2.5, qwen2.5-vl-7b, yolov8-l | ▲1.64× |
| qwen2.5-7b | yolov8-l | ≈1.02× |
| qwen2.5-vl-7b | dinov2-base | ≈1.02× |
| qwen2.5-vl-7b | kosmos-2.5 | ▲1.11× |
| qwen2.5-vl-7b | kosmos-2.5, qwen2.5-7b, yolov8-l | ▲2.14× |
| yolov8-l | kosmos-2.5 | ≈1.02× |
| yolov8-l | kosmos-2.5, qwen2.5-7b, qwen2.5-vl-7b | ▲2.88× |
| yolov8-l | qwen2.5-7b | ≈0.97× |

### 1c. Safe-operating envelope

**Envelope crossings** — achieved_rps < 0.95 × offered_rps:

| Colocation | Tenant | Model | Offered | Achieved | Retention |
|---|---|---|---|---|---|
| mix-full | ilm | kosmos-2.5 | 0.1 | 0.1 | 0.83× |
| mix-vlm-ilm | ilm | kosmos-2.5 | 0.1 | 0.1 | 0.83× |
| mix-ilm-cv | ilm | kosmos-2.5 | 0.1 | 0.1 | 0.83× |
