# Benchmark summary — rtx_pro6000

Decision metrics drive go/no-go (see docs/metrics.md). Diagnostics explain *why* a decision metric moved. Cross-run deltas pair `run_label` variants against `baseline` for the same (framework, model).

---

## 1. Contention analysis

40 solo baseline(s), 34 contention window(s). Ratios are `contention / solo`; ▲ = degraded, ▼ = improved, ≈ = within 5%.

- **Offered rps** — the rate the load generator was told to send.
- **Achieved rps** — the rate it actually managed. Below offered means the tenant could not keep up; that is the safe-operating-envelope boundary, not a measurement error.
- **Throughput kept** — achieved-with-a-neighbour ÷ achieved-alone. `1.00×` = the neighbour cost nothing; `0.60×` = it cost 40% of throughput.
- **e2e / TTFT ratios** — latency with a neighbour ÷ latency alone. Above 1.00× is slower.

### 1a. Degradation table

| Colocation             | Phase | Tenant   | Model                    | Backend | Offered rps | Achieved rps | Throughput kept |  e2e p50 |  e2e p95 |  TTFT p95 |
|-----------------------:|------:|---------:|-------------------------:|--------:|------------:|-------------:|----------------:|---------:|---------:|----------:|
| same-cv                | 2     | cv_a     | yolov8-l                 | triton  |         1.0 |          0.8 |          ≈1.00× |   ≈1.00× |   ≈0.95× |       n/a |
| same-cv                | 2     | cv_a     | yolov8-l                 | triton  |        10.0 |          9.6 |          ≈1.00× |   ≈1.02× |   ▲1.24× |       n/a |
| same-cv                | 2     | cv_a     | yolov8-l                 | triton  |       200.0 |        201.6 |          ≈1.00× |   ▲1.25× |   ▲1.21× |       n/a |
| same-cv                | 2     | cv_a     | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈0.97× |   ≈1.05× |   ▲1.07× |       n/a |
| same-cv                | 2     | cv_b     | dinov2-base              | triton  |         1.0 |          0.8 |          ≈1.00× |   ≈1.02× |   ≈0.96× |       n/a |
| same-cv                | 2     | cv_b     | dinov2-base              | triton  |        10.0 |          9.6 |          ≈1.00× |   ▲1.55× |   ▲1.53× |       n/a |
| same-cv                | 2     | cv_b     | dinov2-base              | triton  |       200.0 |        201.7 |          ≈1.00× |   ▲1.70× |   ▲1.92× |       n/a |
| same-cv                | 2     | cv_b     | dinov2-base              | triton  |        50.0 |         49.2 |          ≈0.97× |   ▲1.05× |   ▲1.23× |       n/a |
| same-ilm               | 2     | ilm_a    | kosmos-2.5               | triton  |         0.0 |          0.0 |          ≈1.00× |   ≈0.98× |   ≈0.98× |       n/a |
| same-ilm               | 2     | ilm_a    | kosmos-2.5               | triton  |         0.1 |          0.0 |          ≈1.00× |   ≈1.01× |   ≈1.03× |       n/a |
| same-ilm               | 2     | ilm_a    | kosmos-2.5               | triton  |         0.1 |          0.1 |          ≈1.00× |   ≈0.99× |   ≈0.98× |       n/a |
| same-ilm               | 2     | ilm_a    | kosmos-2.5               | triton  |         0.2 |          0.2 |          ≈1.00× |   ≈1.04× |   ▲1.09× |       n/a |
| same-ilm               | 2     | ilm_b    | qwen2.5-vl-7b            | vllm    |         0.0 |          0.0 |          ≈1.00× |   ≈1.00× |   ≈1.00× |    ≈1.00× |
| same-ilm               | 2     | ilm_b    | qwen2.5-vl-7b            | vllm    |         0.1 |          0.1 |          ≈1.00× |   ≈1.00× |   ▲1.06× |    ≈1.05× |
| same-ilm               | 2     | ilm_b    | qwen2.5-vl-7b            | vllm    |         0.1 |          0.1 |          ≈1.00× |   ≈1.00× |   ▲1.15× |    ≈1.04× |
| same-ilm               | 2     | ilm_b    | qwen2.5-vl-7b            | vllm    |         0.2 |          0.2 |          ≈1.00× |   ≈1.02× |   ▲1.23× |    ≈1.04× |
| same-llm               | 2     | llm_a    | qwen2.5-7b               | vllm    |         1.0 |          1.0 |          ≈1.00× |   ▲1.51× |   ▲1.63× |    ▲1.59× |
| same-llm               | 2     | llm_a    | qwen2.5-7b               | vllm    |        16.0 |         15.5 |          ≈1.00× |   ▲1.72× |   ▲1.75× |    ▲1.65× |
| same-llm               | 2     | llm_a    | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ▲1.67× |   ▲1.67× |    ▲1.60× |
| same-llm               | 2     | llm_a    | qwen2.5-7b               | vllm    |        64.0 |         47.9 |          ▼0.77× |  ▲33.27× |  ▲64.71× |  ▲600.22× |
| same-llm               | 2     | llm_b    | gemma2-9b                | vllm    |         1.0 |          1.0 |          ≈1.00× |   ▲1.57× |   ▲1.92× |    ▲1.73× |
| same-llm               | 2     | llm_b    | gemma2-9b                | vllm    |        16.0 |         15.4 |          ≈1.00× |   ▲1.89× |   ▲1.88× |    ▲1.80× |
| same-llm               | 2     | llm_b    | gemma2-9b                | vllm    |         4.0 |          3.9 |          ≈1.00× |   ▲1.84× |   ▲1.85× |    ▲1.73× |
| same-llm               | 2     | llm_b    | gemma2-9b                | vllm    |        64.0 |         34.2 |          ▼0.55× |  ▲37.24× |  ▲36.71× |   ▲54.38× |
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
| cross-cv-vs-llm-rps    | 4     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈0.97× |   ▲1.11× |   ▲1.96× |       n/a |
| cross-cv-vs-llm-rps    | 4     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈0.97× |   ▲1.74× |   ▲1.93× |       n/a |
| cross-cv-vs-llm-rps    | 4     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈0.97× |   ▲1.79× |   ▲2.17× |       n/a |
| cross-cv-vs-llm-rps    | 4     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈0.97× |   ▲1.84× |   ▲2.09× |       n/a |
| cross-cv-vs-llm-rps    | 4     | llm      | qwen2.5-7b               | vllm    |         1.0 |          1.0 |          ≈1.00× |   ≈1.02× |   ≈1.02× |    ≈1.00× |
| cross-cv-vs-llm-rps    | 4     | llm      | qwen2.5-7b               | vllm    |        16.0 |         15.5 |          ≈1.00× |   ≈1.02× |   ≈1.03× |    ≈1.03× |
| cross-cv-vs-llm-rps    | 4     | llm      | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ≈1.01× |   ≈1.02× |    ≈1.02× |
| cross-cv-vs-llm-rps    | 4     | llm      | qwen2.5-7b               | vllm    |        64.0 |         62.4 |          ≈1.00× |   ≈1.02× |   ≈1.03× |    ▲1.15× |
| cross-ilm-vs-cv        | 4     | cv       | yolov8-l                 | triton  |         1.0 |          0.9 |          ▲1.15× |   ≈1.02× |   ≈1.02× |       n/a |
| cross-ilm-vs-cv        | 4     | cv       | yolov8-l                 | triton  |        10.0 |          9.9 |          ≈1.03× |   ≈0.98× |   ≈0.98× |       n/a |
| cross-ilm-vs-cv        | 4     | cv       | yolov8-l                 | triton  |       200.0 |          n/a |             n/a |      n/a |      n/a |       n/a |
| cross-ilm-vs-cv        | 4     | cv       | yolov8-l                 | triton  |        50.0 |         50.5 |          ≈1.00× |   ≈1.02× |   ≈1.04× |       n/a |
| cross-ilm-vs-cv        | 4     | ilm      | kosmos-2.5               | triton  |         0.1 |          0.1 |          ≈1.00× |   ≈0.97× |   ≈0.96× |       n/a |
| cross-ilm-vs-cv        | 4     | ilm      | kosmos-2.5               | triton  |         0.1 |          0.1 |          ≈1.00× |   ≈0.96× |   ≈0.96× |       n/a |
| cross-ilm-vs-cv        | 4     | ilm      | kosmos-2.5               | triton  |         0.1 |          n/a |             n/a |      n/a |      n/a |       n/a |
| cross-ilm-vs-cv        | 4     | ilm      | kosmos-2.5               | triton  |         0.1 |          0.1 |          ≈1.00× |   ≈0.98× |   ≈0.97× |       n/a |
| cross-llm-vs-cv-rps    | 4     | cv       | yolov8-l                 | triton  |         1.0 |          0.8 |          ≈1.00× |   ▲1.58× |   ▲1.89× |       n/a |
| cross-llm-vs-cv-rps    | 4     | cv       | yolov8-l                 | triton  |        10.0 |          9.6 |          ≈1.00× |   ▲1.70× |   ▲1.92× |       n/a |
| cross-llm-vs-cv-rps    | 4     | cv       | yolov8-l                 | triton  |       200.0 |        201.6 |          ≈1.00× |   ▲2.13× |   ▲1.93× |       n/a |
| cross-llm-vs-cv-rps    | 4     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈0.97× |   ▲1.80× |   ▲2.19× |       n/a |
| cross-llm-vs-cv-rps    | 4     | llm      | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ≈1.00× |   ≈0.99× |    ≈0.99× |
| cross-llm-vs-cv-rps    | 4     | llm      | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ≈1.00× |   ≈1.00× |    ≈1.00× |
| cross-llm-vs-cv-rps    | 4     | llm      | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ▲1.09× |   ▲1.09× |    ▲1.10× |
| cross-llm-vs-cv-rps    | 4     | llm      | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ≈1.01× |   ≈1.02× |    ≈1.03× |
| place-isolated         | 5     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈1.00× |   ≈1.01× |   ≈1.02× |       n/a |
| place-isolated         | 5     | llm      | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ≈1.00× |   ≈1.00× |    ≈1.01× |
| place-p1               | 5     | cv       | yolov8-l                 | triton  |        50.0 |         50.5 |          ≈1.03× |   ▲1.06× |   ▲1.05× |       n/a |
| place-p1               | 5     | ilm      | kosmos-2.5               | triton  |         0.1 |          0.1 |          ≈1.00× |   ▲1.08× |   ▲1.12× |       n/a |
| place-p1               | 5     | llm      | qwen2.5-7b               | vllm    |         4.0 |          3.8 |          ≈0.98× |   ▲1.31× |   ▲1.39× |    ▲1.34× |
| place-p1               | 5     | vlm      | qwen2.5-vl-7b            | vllm    |         1.0 |          1.0 |          ≈0.96× |   ▲1.91× |   ▲1.95× |    ▲1.24× |
| place-p2               | 5     | cv       | yolov8-l                 | triton  |        50.0 |         50.5 |          ≈1.03× |   ▲1.41× |   ▲1.46× |       n/a |
| place-p2               | 5     | ilm      | kosmos-2.5               | triton  |         0.1 |          0.1 |          ≈1.00× |   ▲1.20× |   ▲1.21× |       n/a |
| place-p2               | 5     | llm      | qwen2.5-7b               | vllm    |         4.0 |          3.8 |          ≈0.98× |   ≈1.00× |   ▲1.21× |    ▲1.08× |
| place-p2               | 5     | vlm      | qwen2.5-vl-7b            | vllm    |         1.0 |          1.0 |          ≈1.00× |   ≈1.02× |   ≈1.02× |    ≈1.03× |
| place-p3               | 5     | cv       | yolov8-l                 | triton  |        50.0 |         50.5 |          ≈1.00× |   ▲1.86× |   ▲2.19× |       n/a |
| place-p3               | 5     | ilm      | kosmos-2.5               | triton  |         0.1 |          0.1 |          ≈1.00× |   ▲1.05× |   ▲1.06× |       n/a |
| place-p3               | 5     | llm      | qwen2.5-7b               | vllm    |         4.0 |          3.8 |          ≈0.98× |   ≈1.02× |   ≈1.02× |    ≈1.02× |
| place-p3               | 5     | vlm      | qwen2.5-vl-7b            | vllm    |         1.0 |          1.0 |          ≈1.00× |   ≈1.02× |   ▲1.12× |    ≈1.03× |
| place-vlm-prefill-split | 5     | llm      | gemma2-9b                | vllm    |         4.0 |          3.9 |          ≈1.00× |   ≈1.00× |   ≈1.00× |    ≈1.01× |
| place-vlm-prefill-split | 5     | vlm      | qwen2.5-vl-7b            | vllm    |         1.0 |          1.0 |          ≈1.00× |   ≈1.00× |   ≈0.99× |    ≈1.01× |

### 1b. Contention matrix

**e2e p95 degradation (victim × aggressors)**

| Victim model | Aggressors | e2e p95 ratio (mean) |
|---|---|---|
| dinov2-base | qwen2.5-vl-7b | ≈0.95× |
| dinov2-base | yolov8-l | ▲1.41× |
| gemma2-9b | qwen2.5-7b | ▲10.59× |
| gemma2-9b | qwen2.5-vl-7b | ≈1.00× |
| kosmos-2.5 | qwen2.5-7b, qwen2.5-vl-7b, yolov8-l | ▲1.38× |
| kosmos-2.5 | qwen2.5-vl-7b | ≈1.03× |
| kosmos-2.5 | yolov8-l | ≈0.99× |
| qwen2.5-7b | gemma2-9b | ▲17.44× |
| qwen2.5-7b | kosmos-2.5, qwen2.5-vl-7b, yolov8-l | ▲1.31× |
| qwen2.5-7b | yolov8-l | ≈1.02× |
| qwen2.5-vl-7b | dinov2-base | ≈1.02× |
| qwen2.5-vl-7b | gemma2-9b | ≈0.99× |
| qwen2.5-vl-7b | kosmos-2.5 | ▲1.11× |
| qwen2.5-vl-7b | kosmos-2.5, qwen2.5-7b, yolov8-l | ▲1.56× |
| yolov8-l | dinov2-base | ▲1.12× |
| yolov8-l | kosmos-2.5 | ≈1.01× |
| yolov8-l | kosmos-2.5, qwen2.5-7b, qwen2.5-vl-7b | ▲1.90× |
| yolov8-l | qwen2.5-7b | ▲1.81× |

### 1c. Safe-operating envelope

**Envelope crossings** — achieved_rps < 0.95 × offered_rps:

| Colocation | Tenant | Model | Offered | Achieved | Retention |
|---|---|---|---|---|---|
| same-llm | llm_b | gemma2-9b | 64.0 | 34.2 | 0.53× |
| same-ilm | ilm_a | kosmos-2.5 | 0.0 | 0.0 | 0.61× |
| same-ilm | ilm_a | kosmos-2.5 | 0.1 | 0.0 | 0.71× |
| same-llm | llm_a | qwen2.5-7b | 64.0 | 47.9 | 0.75× |
| cross-llm-vs-cv-rps | cv | yolov8-l | 1.0 | 0.8 | 0.83× |
| same-cv | cv_a | yolov8-l | 1.0 | 0.8 | 0.83× |
| same-cv | cv_b | dinov2-base | 1.0 | 0.8 | 0.83× |
| place-p3 | ilm | kosmos-2.5 | 0.1 | 0.1 | 0.83× |
| cross-ilm-vs-cv | ilm | kosmos-2.5 | 0.1 | 0.1 | 0.83× |
| mix-full | ilm | kosmos-2.5 | 0.1 | 0.1 | 0.83× |
| mix-vlm-ilm | ilm | kosmos-2.5 | 0.1 | 0.1 | 0.83× |
| place-p1 | ilm | kosmos-2.5 | 0.1 | 0.1 | 0.83× |
| place-p2 | ilm | kosmos-2.5 | 0.1 | 0.1 | 0.83× |
| mix-ilm-cv | ilm | kosmos-2.5 | 0.1 | 0.1 | 0.83× |
| cross-ilm-vs-cv | ilm | kosmos-2.5 | 0.1 | 0.1 | 0.83× |
| same-ilm | ilm_a | kosmos-2.5 | 0.1 | 0.1 | 0.83× |
| cross-ilm-vs-cv | ilm | kosmos-2.5 | 0.1 | 0.1 | 0.83× |
| same-ilm | ilm_a | kosmos-2.5 | 0.2 | 0.2 | 0.83× |
| cross-ilm-vs-cv | cv | yolov8-l | 1.0 | 0.9 | 0.95× |
