# Benchmark summary — rtx_pro6000

Decision metrics drive go/no-go (see docs/metrics.md). Diagnostics explain *why* a decision metric moved. Cross-run deltas pair `run_label` variants against `baseline` for the same (framework, model).

---

## 1. Contention analysis

80 solo baseline(s), 90 contention window(s). Ratios are `contention / solo`; ▲ = degraded, ▼ = improved, ≈ = within 5%.

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
| same-vlm               | 2     | vlm_a    | qwen2.5-vl-7b            | vllm    |         0.5 |          0.5 |          ≈0.98× |   ▲5.70× |   ▲9.77× |    ▲3.64× |
| same-vlm               | 2     | vlm_a    | qwen2.5-vl-7b            | vllm    |         1.0 |          1.0 |          ≈0.98× |   ▲4.34× |   ▲9.95× |    ▲4.49× |
| same-vlm               | 2     | vlm_a    | qwen2.5-vl-7b            | vllm    |         2.0 |          1.9 |          ≈0.97× |   ▲3.14× |   ▲6.20× |    ▲3.19× |
| same-vlm               | 2     | vlm_a    | qwen2.5-vl-7b            | vllm    |         2.0 |          1.9 |          ≈0.98× |   ▲6.31× |  ▲10.38× |    ▲5.11× |
| same-vlm               | 2     | vlm_a    | qwen2.5-vl-7b            | vllm    |         4.0 |          3.5 |          ▼0.91× |  ▲11.05× |  ▲12.09× |   ▲38.01× |
| same-vlm               | 2     | vlm_b    | qwen3-vl-32b-fp8         | vllm    |         0.5 |          0.3 |          ▼0.92× |   ▲1.11× |   ▲1.20× |    ▲1.22× |
| same-vlm               | 2     | vlm_b    | qwen3-vl-32b-fp8         | vllm    |         1.0 |          0.3 |          ▼0.88× |   ▲1.10× |   ≈1.04× |    ≈1.04× |
| same-vlm               | 2     | vlm_b    | qwen3-vl-32b-fp8         | vllm    |         0.1 |          0.2 |          ≈1.00× |   ▲1.28× |   ▲1.31× |    ▲1.49× |
| same-vlm               | 2     | vlm_b    | qwen3-vl-32b-fp8         | vllm    |         2.0 |          0.3 |          ▼0.83× |   ≈1.02× |   ≈1.04× |    ≈1.03× |
| same-vlm               | 2     | vlm_b    | qwen3-vl-32b-fp8         | vllm    |         4.0 |          0.2 |          ▼0.73× |   ▲1.08× |   ≈1.04× |    ≈1.02× |
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
| cross-arch-validation  | 4     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈0.97× |   ▲1.85× |   ▲2.05× |       n/a |
| cross-arch-validation  | 4     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈0.97× |   ▲1.80× |   ▲2.20× |       n/a |
| cross-arch-validation  | 4     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈0.97× |   ▲1.84× |   ▲2.01× |       n/a |
| cross-arch-validation  | 4     | llm      | llama3.1-8b              | vllm    |         4.0 |          3.9 |          ≈1.00× |   ≈1.02× |   ≈1.02× |    ≈1.03× |
| cross-arch-validation  | 4     | llm      | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ≈1.01× |   ≈1.02× |    ≈1.02× |
| cross-arch-validation  | 4     | llm      | mistral-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ≈1.02× |   ≈1.02× |    ≈1.02× |
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
| cross-memory-pressure-kv13 | 4     | anchor   | qwen2.5-72b              | vllm    |         2.0 |          2.0 |          ≈1.00× |   ▲1.58× |   ▼0.31× |    ▲1.65× |
| cross-memory-pressure-kv13 | 4     | anchor   | qwen2.5-72b              | vllm    |         2.0 |          2.0 |          ≈1.00× |   ▲1.62× |   ▼0.31× |    ▲1.65× |
| cross-memory-pressure-kv13 | 4     | anchor   | qwen2.5-72b              | vllm    |         2.0 |          2.0 |          ≈1.00× |   ▲1.62× |   ▼0.31× |    ▲1.64× |
| cross-memory-pressure-kv13 | 4     | neighbour | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ▲1.84× |   ▲5.54× |    ▲1.81× |
| cross-memory-pressure-kv13 | 4     | neighbour | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ▲1.84× |   ▲1.84× |    ▲1.74× |
| cross-memory-pressure-kv13 | 4     | neighbour | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ▲1.84× |   ▲1.84× |    ▲1.74× |
| cross-memory-pressure-kv22 | 4     | anchor   | qwen2.5-72b              | vllm    |         2.0 |          2.0 |          ≈1.00× |   ▲1.57× |   ▼0.31× |    ▲1.65× |
| cross-memory-pressure-kv22 | 4     | anchor   | qwen2.5-72b              | vllm    |         2.0 |          2.0 |          ≈1.00× |   ▲1.61× |   ▼0.31× |    ▲1.65× |
| cross-memory-pressure-kv22 | 4     | anchor   | qwen2.5-72b              | vllm    |         2.0 |          2.0 |          ≈1.00× |   ▲1.61× |   ▼0.31× |    ▲1.66× |
| cross-memory-pressure-kv22 | 4     | neighbour | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ▲1.84× |   ▲5.52× |    ▲1.81× |
| cross-memory-pressure-kv22 | 4     | neighbour | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ▲1.84× |   ▲1.84× |    ▲1.75× |
| cross-memory-pressure-kv22 | 4     | neighbour | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ▲1.84× |   ▲1.84× |    ▲1.75× |
| cross-memory-pressure-kv29 | 4     | anchor   | qwen2.5-72b              | vllm    |         2.0 |          2.0 |          ≈1.00× |   ▲1.56× |   ▼0.31× |    ▲1.63× |
| cross-memory-pressure-kv29 | 4     | anchor   | qwen2.5-72b              | vllm    |         2.0 |          2.0 |          ≈1.00× |   ▲1.63× |   ▼0.31× |    ▲1.62× |
| cross-memory-pressure-kv29 | 4     | anchor   | qwen2.5-72b              | vllm    |         2.0 |          2.0 |          ≈1.00× |   ▲1.62× |   ▼0.31× |    ▲1.63× |
| cross-memory-pressure-kv29 | 4     | neighbour | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ▲1.84× |   ▲5.96× |    ▲1.83× |
| cross-memory-pressure-kv29 | 4     | neighbour | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ▲1.84× |   ▲1.84× |    ▲1.73× |
| cross-memory-pressure-kv29 | 4     | neighbour | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ▲1.83× |   ▲1.83× |    ▲1.73× |
| cross-size-scaling     | 4     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈0.97× |   ▲2.34× |   ▲2.82× |       n/a |
| cross-size-scaling     | 4     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈0.97× |   ▲1.82× |   ▲2.19× |       n/a |
| cross-size-scaling     | 4     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈0.97× |   ▲2.28× |   ▲2.55× |       n/a |
| cross-size-scaling     | 4     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈0.97× |   ▲1.91× |   ▲2.18× |       n/a |
| cross-size-scaling     | 4     | llm      | qwen2.5-32b              | vllm    |         4.0 |          3.9 |          ≈1.00× |   ≈1.02× |   ▼0.32× |    ▼0.07× |
| cross-size-scaling     | 4     | llm      | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ≈1.02× |   ≈1.02× |    ≈1.03× |
| cross-size-scaling     | 4     | llm      | qwen2.5-72b              | vllm    |         4.0 |          3.9 |          ≈1.00× |   ≈1.02× |   ▼0.31× |    ▼0.06× |
| cross-size-scaling     | 4     | llm      | qwen2.5-14b              | vllm    |         4.0 |          3.9 |          ≈1.00× |   ≈1.02× |   ▼0.32× |    ≈0.99× |
| cross-vlm-prefill-vs-llm | 4     | llm      | gemma2-9b                | vllm    |         4.0 |          3.9 |          ≈1.00× |   ▲1.39× |   ▲1.44× |    ▲1.39× |
| cross-vlm-prefill-vs-llm | 4     | vlm      | qwen2.5-vl-7b            | vllm    |         1.0 |          1.0 |          ≈1.00× |   ▲2.00× |   ▲1.96× |    ▲1.32× |
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
| place-vlm-prefill-split | 5     | llm      | gemma2-9b                | vllm    |         4.0 |          3.9 |          ≈1.00× |   ≈1.00× |   ≈1.00× |    ≈1.00× |
| place-vlm-prefill-split | 5     | vlm      | qwen2.5-vl-7b            | vllm    |         1.0 |          1.0 |          ≈1.00× |   ≈1.00× |   ≈0.99× |    ≈1.01× |
| place-vlm-prefill-split | 5     | vlm      | qwen2.5-vl-7b            | vllm    |         1.0 |          1.0 |          ≈1.00× |   ≈1.02× |   ≈1.01× |    ▲1.16× |
| mix-memory-bound       | 6     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈0.97× |   ▲3.26× |   ▲3.70× |       n/a |
| mix-memory-bound       | 6     | llm      | qwen2.5-72b              | vllm    |         2.0 |          2.0 |          ≈0.99× |   ▲1.88× |   ▼0.34× |    ▲1.86× |
| mix-memory-bound       | 6     | llm2     | qwen2.5-14b              | vllm    |         2.0 |          2.0 |          ≈1.00× |   ▲1.77× |   ▲1.79× |    ▲1.73× |
| secondary-arrival-a    | 6     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈0.97× |   ▲1.80× |   ▲2.17× |       n/a |
| secondary-arrival-a    | 6     | cv       | yolov8-l                 | triton  |        50.0 |         50.0 |          ≈1.00× |   ▲1.66× |   ▲2.15× |       n/a |
| secondary-arrival-a    | 6     | llm      | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ≈1.01× |   ≈1.02× |    ≈1.02× |
| secondary-arrival-a    | 6     | llm      | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ≈1.01× |   ≈1.02× |    ≈1.02× |
| secondary-arrival-b    | 6     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈0.97× |   ▲3.26× |   ▲3.71× |       n/a |
| secondary-arrival-b    | 6     | cv       | yolov8-l                 | triton  |        50.0 |         49.6 |          ≈0.99× |   ▲2.95× |   ▲3.26× |       n/a |
| secondary-arrival-b    | 6     | llm      | qwen2.5-72b              | vllm    |         2.0 |          2.0 |          ≈0.99× |   ▲1.90× |   ▼0.34× |    ▲1.82× |
| secondary-arrival-b    | 6     | llm      | qwen2.5-72b              | vllm    |         2.0 |          2.0 |          ≈0.99× |   ▲1.89× |   ▼0.34× |    ▲1.83× |
| secondary-arrival-b    | 6     | llm2     | qwen2.5-14b              | vllm    |         2.0 |          2.0 |          ≈1.00× |   ▲1.78× |   ▲1.79× |    ▲1.72× |
| secondary-arrival-b    | 6     | llm2     | qwen2.5-14b              | vllm    |         2.0 |          2.0 |          ≈1.00× |   ▲1.77× |   ▲1.78× |    ▲1.71× |
| secondary-asymmetry-a  | 6     | cv       | yolov8-l                 | triton  |        16.0 |         15.6 |          ≈1.00× |   ▲1.65× |   ▲1.91× |       n/a |
| secondary-asymmetry-a  | 6     | cv       | yolov8-l                 | triton  |         4.0 |          3.9 |          ≈1.00× |   ▲1.66× |   ▲1.97× |       n/a |
| secondary-asymmetry-a  | 6     | cv       | yolov8-l                 | triton  |        64.0 |         63.5 |          ≈1.00× |   ▲1.83× |   ▲2.16× |       n/a |
| secondary-asymmetry-a  | 6     | llm      | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ≈1.00× |   ≈1.00× |    ≈1.00× |
| secondary-asymmetry-a  | 6     | llm      | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ≈1.00× |   ≈1.00× |    ≈0.98× |
| secondary-asymmetry-a  | 6     | llm      | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ≈1.02× |   ≈1.03× |    ≈1.03× |
| secondary-asymmetry-b  | 6     | cv       | yolov8-l                 | triton  |        16.0 |         15.6 |          ≈1.00× |   ▲2.95× |   ▲3.79× |       n/a |
| secondary-asymmetry-b  | 6     | cv       | yolov8-l                 | triton  |         4.0 |          3.9 |          ≈1.00× |   ▲2.88× |   ▲3.03× |       n/a |
| secondary-asymmetry-b  | 6     | cv       | yolov8-l                 | triton  |        64.0 |         63.5 |          ≈1.00× |   ▲3.42× |   ▲3.65× |       n/a |
| secondary-asymmetry-b  | 6     | llm      | qwen2.5-72b              | vllm    |         2.0 |          2.0 |          ≈0.99× |   ▲1.87× |   ▼0.34× |    ▲1.78× |
| secondary-asymmetry-b  | 6     | llm      | qwen2.5-72b              | vllm    |         2.0 |          2.0 |          ≈0.99× |   ▲1.87× |   ▼0.34× |    ▲1.81× |
| secondary-asymmetry-b  | 6     | llm      | qwen2.5-72b              | vllm    |         2.0 |          2.0 |          ≈0.99× |   ▲1.90× |   ▼0.35× |    ▲1.85× |
| secondary-asymmetry-b  | 6     | llm2     | qwen2.5-14b              | vllm    |         2.0 |          2.0 |          ≈1.00× |   ▲1.73× |   ▲1.74× |    ▲1.65× |
| secondary-asymmetry-b  | 6     | llm2     | qwen2.5-14b              | vllm    |         2.0 |          2.0 |          ≈1.00× |   ▲1.72× |   ▲1.72× |    ▲1.65× |
| secondary-asymmetry-b  | 6     | llm2     | qwen2.5-14b              | vllm    |         2.0 |          2.0 |          ≈1.00× |   ▲1.80× |   ▲1.81× |    ▲1.75× |
| secondary-backend-cv-a | 6     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈0.97× |   ▲1.79× |   ▲2.18× |       n/a |
| secondary-backend-cv-a | 6     | cv       | yolov8-l                 | triton  |        50.0 |          n/a |             n/a |      n/a |      n/a |       n/a |
| secondary-backend-cv-a | 6     | llm      | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ≈1.01× |   ≈1.02× |    ≈1.02× |
| secondary-backend-cv-a | 6     | llm      | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ≈1.00× |   ≈1.01× |    ≈1.01× |
| secondary-backend-cv-b | 6     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈0.97× |   ▲3.27× |   ▲3.71× |       n/a |
| secondary-backend-cv-b | 6     | cv       | yolov8-l                 | triton  |        50.0 |          n/a |             n/a |      n/a |      n/a |       n/a |
| secondary-backend-cv-b | 6     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈0.97× |   ▲3.26× |   ▲3.72× |       n/a |
| secondary-backend-cv-b | 6     | cv       | yolov8-l                 | triton  |        50.0 |          n/a |             n/a |      n/a |      n/a |       n/a |
| secondary-backend-cv-b | 6     | llm      | qwen2.5-72b              | vllm    |         2.0 |          2.0 |          ≈0.99× |   ▲1.90× |   ▼0.34× |    ▲1.84× |
| secondary-backend-cv-b | 6     | llm      | qwen2.5-72b              | vllm    |         2.0 |          2.0 |          ≈0.99× |   ▲1.86× |   ▲1.90× |    ▲1.83× |
| secondary-backend-cv-b | 6     | llm      | qwen2.5-72b              | vllm    |         2.0 |          2.0 |          ≈0.99× |   ▲1.91× |   ▲1.94× |    ▲1.87× |
| secondary-backend-cv-b | 6     | llm      | qwen2.5-72b              | vllm    |         2.0 |          2.0 |          ≈0.99× |   ▲1.86× |   ▼0.34× |    ▲1.80× |
| secondary-backend-cv-b | 6     | llm2     | qwen2.5-14b              | vllm    |         2.0 |          2.0 |          ≈1.00× |   ▲1.78× |   ▲1.78× |    ▲1.70× |
| secondary-backend-cv-b | 6     | llm2     | qwen2.5-14b              | vllm    |         2.0 |          2.0 |          ≈1.00× |   ▲1.76× |   ▲1.77× |    ▲1.69× |
| secondary-backend-cv-b | 6     | llm2     | qwen2.5-14b              | vllm    |         2.0 |          2.0 |          ≈1.00× |   ▲1.78× |   ▲1.79× |    ▲1.72× |
| secondary-backend-cv-b | 6     | llm2     | qwen2.5-14b              | vllm    |         2.0 |          2.0 |          ≈1.00× |   ▲1.76× |   ▲1.77× |    ▲1.68× |
| secondary-backend-llm-a | 6     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈0.97× |   ▲1.91× |   ▲2.54× |       n/a |
| secondary-backend-llm-a | 6     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈0.97× |   ▲1.79× |   ▲2.18× |       n/a |
| secondary-backend-llm-a | 6     | llm      | qwen2.5-7b               | sglang  |         4.0 |          3.9 |          ≈1.00× |   ≈1.03× |   ≈1.02× |    ≈1.03× |
| secondary-backend-llm-a | 6     | llm      | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ≈1.01× |   ≈1.02× |    ≈1.02× |
| secondary-backend-llm-b | 6     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈0.97× |   ▲3.26× |   ▲3.73× |       n/a |
| secondary-backend-llm-b | 6     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈0.97× |   ▲3.25× |   ▲3.72× |       n/a |
| secondary-backend-llm-b | 6     | llm      | qwen2.5-72b              | vllm    |         2.0 |          2.0 |          ≈0.99× |   ▲1.88× |   ▼0.34× |    ▲1.83× |
| secondary-backend-llm-b | 6     | llm      | qwen2.5-72b              | vllm    |         2.0 |          2.0 |          ≈0.99× |   ▲1.91× |   ▲1.93× |    ▲1.87× |
| secondary-backend-llm-b | 6     | llm2     | qwen2.5-14b              | vllm    |         2.0 |          2.0 |          ≈1.00× |   ▲1.78× |   ▲1.79× |    ▲1.73× |
| secondary-backend-llm-b | 6     | llm2     | qwen2.5-14b              | vllm    |         2.0 |          2.0 |          ≈1.00× |   ▲1.78× |   ▲1.80× |    ▲1.73× |
| secondary-input-size-cv-a | 6     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈1.00× |   ▲1.78× |   ▲2.21× |       n/a |
| secondary-input-size-cv-a | 6     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈1.00× |   ▲1.79× |   ▲2.21× |       n/a |
| secondary-input-size-cv-a | 6     | llm      | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ≈1.01× |   ≈1.02× |    ≈1.01× |
| secondary-input-size-cv-a | 6     | llm      | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ≈1.01× |   ≈1.02× |    ≈1.02× |
| secondary-input-size-cv-b | 6     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈1.00× |   ▲3.27× |   ▲3.77× |       n/a |
| secondary-input-size-cv-b | 6     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈1.00× |   ▲3.21× |   ▲3.71× |       n/a |
| secondary-input-size-cv-b | 6     | llm      | qwen2.5-72b              | vllm    |         2.0 |          2.0 |          ≈0.99× |   ▲1.89× |   ▼0.34× |    ▲1.83× |
| secondary-input-size-cv-b | 6     | llm      | qwen2.5-72b              | vllm    |         2.0 |          2.0 |          ≈0.99× |   ▲1.90× |   ▼0.34× |    ▲1.84× |
| secondary-input-size-cv-b | 6     | llm2     | qwen2.5-14b              | vllm    |         2.0 |          2.0 |          ≈1.00× |   ▲1.78× |   ▲1.79× |    ▲1.72× |
| secondary-input-size-cv-b | 6     | llm2     | qwen2.5-14b              | vllm    |         2.0 |          2.0 |          ≈1.00× |   ▲1.77× |   ▲1.79× |    ▲1.71× |
| secondary-input-size-llm-a | 6     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈0.97× |   ▲1.79× |   ▲2.18× |       n/a |
| secondary-input-size-llm-a | 6     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈0.97× |   ▲1.79× |   ▲2.19× |       n/a |
| secondary-input-size-llm-a | 6     | llm      | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ≈1.01× |   ≈1.02× |    ≈1.02× |
| secondary-input-size-llm-a | 6     | llm      | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ≈1.02× |   ≈1.03× |    ≈1.03× |
| secondary-input-size-llm-b | 6     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈0.97× |   ▲3.33× |   ▲3.86× |       n/a |
| secondary-input-size-llm-b | 6     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈0.97× |   ▲3.26× |   ▲3.71× |       n/a |
| secondary-input-size-llm-b | 6     | llm      | qwen2.5-72b              | vllm    |         2.0 |          2.0 |          ≈0.99× |   ▲1.95× |   ▲1.96× |    ▲1.86× |
| secondary-input-size-llm-b | 6     | llm      | qwen2.5-72b              | vllm    |         2.0 |          2.0 |          ≈0.99× |   ▲1.90× |   ▼0.34× |    ▲1.84× |
| secondary-input-size-llm-b | 6     | llm2     | qwen2.5-14b              | vllm    |         2.0 |          2.0 |          ≈1.00× |   ▲1.80× |   ▲1.83× |    ▲1.75× |
| secondary-input-size-llm-b | 6     | llm2     | qwen2.5-14b              | vllm    |         2.0 |          2.0 |          ≈1.00× |   ▲1.78× |   ▲1.79× |    ▲1.71× |
| secondary-output-length-a | 6     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈0.97× |   ▲1.80× |   ▲2.17× |       n/a |
| secondary-output-length-a | 6     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈0.97× |   ▲1.83× |   ▲2.09× |       n/a |
| secondary-output-length-a | 6     | llm      | qwen2.5-7b               | vllm    |         4.0 |          3.9 |          ≈1.00× |   ≈1.01× |   ≈1.02× |    ≈1.02× |
| secondary-output-length-a | 6     | llm      | qwen2.5-7b               | vllm    |         4.0 |          3.7 |          ≈1.00× |   ≈1.02× |   ≈1.04× |    ▲1.36× |
| secondary-output-length-b | 6     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈0.97× |   ▲3.23× |   ▲3.69× |       n/a |
| secondary-output-length-b | 6     | cv       | yolov8-l                 | triton  |        50.0 |         49.2 |          ≈0.97× |   ▲3.53× |   ▲4.12× |       n/a |
| secondary-output-length-b | 6     | llm      | qwen2.5-72b              | vllm    |         2.0 |          2.0 |          ≈0.99× |   ▲1.90× |   ▼0.34× |    ▲1.83× |
| secondary-output-length-b | 6     | llm      | qwen2.5-72b              | vllm    |         2.0 |          0.9 |          ▼0.67× |   ▲1.97× |   ▲1.99× |    ▲3.47× |
| secondary-output-length-b | 6     | llm2     | qwen2.5-14b              | vllm    |         2.0 |          2.0 |          ≈1.00× |   ▲1.77× |   ▲1.79× |    ▲1.71× |
| secondary-output-length-b | 6     | llm2     | qwen2.5-14b              | vllm    |         2.0 |          2.0 |          ≈1.00× |   ▲1.89× |   ▲2.03× |    ▲1.86× |

### 1b. Contention matrix

**e2e p95 degradation (victim × aggressors)**

| Victim model | Aggressors | e2e p95 ratio (mean) |
|---|---|---|
| dinov2-base | qwen2.5-vl-7b | ≈0.95× |
| dinov2-base | yolov8-l | ▲1.41× |
| gemma2-9b | qwen2.5-7b | ▲10.59× |
| gemma2-9b | qwen2.5-vl-7b | ▲1.15× |
| kosmos-2.5 | qwen2.5-7b, qwen2.5-vl-7b, yolov8-l | ▲1.38× |
| kosmos-2.5 | qwen2.5-vl-7b | ≈1.03× |
| kosmos-2.5 | yolov8-l | ≈0.99× |
| llama3.1-8b | yolov8-l | ≈1.02× |
| mistral-7b | yolov8-l | ≈1.02× |
| qwen2.5-14b | qwen2.5-72b, yolov8-l | ▲1.80× |
| qwen2.5-14b | yolov8-l | ▼0.32× |
| qwen2.5-32b | yolov8-l | ▼0.32× |
| qwen2.5-72b | qwen2.5-14b, yolov8-l | ▼0.79× |
| qwen2.5-72b | qwen2.5-7b | ▼0.31× |
| qwen2.5-72b | yolov8-l | ▼0.31× |
| qwen2.5-7b | gemma2-9b | ▲17.44× |
| qwen2.5-7b | kosmos-2.5, qwen2.5-vl-7b, yolov8-l | ▲1.31× |
| qwen2.5-7b | qwen2.5-72b | ▲3.12× |
| qwen2.5-7b | yolov8-l | ≈1.02× |
| qwen2.5-vl-7b | dinov2-base | ≈1.02× |
| qwen2.5-vl-7b | gemma2-9b | ▲1.32× |
| qwen2.5-vl-7b | kosmos-2.5 | ▲1.11× |
| qwen2.5-vl-7b | kosmos-2.5, qwen2.5-7b, yolov8-l | ▲1.56× |
| qwen2.5-vl-7b | qwen3-vl-32b-fp8 | ▲9.68× |
| qwen3-vl-32b-fp8 | qwen2.5-vl-7b | ▲1.13× |
| yolov8-l | dinov2-base | ▲1.12× |
| yolov8-l | kosmos-2.5 | ≈1.01× |
| yolov8-l | kosmos-2.5, qwen2.5-7b, qwen2.5-vl-7b | ▲1.90× |
| yolov8-l | llama3.1-8b | ▲2.05× |
| yolov8-l | mistral-7b | ▲2.01× |
| yolov8-l | qwen2.5-14b | ▲2.18× |
| yolov8-l | qwen2.5-14b, qwen2.5-72b | ▲3.68× |
| yolov8-l | qwen2.5-32b | ▲2.82× |
| yolov8-l | qwen2.5-72b | ▲2.55× |
| yolov8-l | qwen2.5-7b | ▲2.03× |

### 1c. Safe-operating envelope

**Envelope crossings** — achieved_rps < 0.95 × offered_rps:

| Colocation | Tenant | Model | Offered | Achieved | Retention |
|---|---|---|---|---|---|
| same-vlm | vlm_b | qwen3-vl-32b-fp8 | 4.0 | 0.2 | 0.06× |
| same-vlm | vlm_b | qwen3-vl-32b-fp8 | 2.0 | 0.3 | 0.14× |
| same-vlm | vlm_b | qwen3-vl-32b-fp8 | 1.0 | 0.3 | 0.29× |
| secondary-output-length-b | llm | qwen2.5-72b | 2.0 | 0.9 | 0.46× |
| same-llm | llm_b | gemma2-9b | 64.0 | 34.2 | 0.53× |
| same-vlm | vlm_b | qwen3-vl-32b-fp8 | 0.5 | 0.3 | 0.60× |
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
| same-vlm | vlm_a | qwen2.5-vl-7b | 4.0 | 3.5 | 0.89× |
| secondary-output-length-a | llm | qwen2.5-7b | 4.0 | 3.7 | 0.93× |
| same-vlm | vlm_a | qwen2.5-vl-7b | 2.0 | 1.9 | 0.95× |
| cross-ilm-vs-cv | cv | yolov8-l | 1.0 | 0.9 | 0.95× |
