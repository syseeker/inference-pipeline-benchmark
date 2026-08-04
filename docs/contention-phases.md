# Contention benchmark — phases, configuration, and what to observe

The customer's design is `workspace/contention/experiment_design.md` and
`workspace/contention/experiment_config.json`. This file states, per phase, what
was built against it — with the configured values rather than the intended ones.
Every number in the tables is resolved from `benchmarks/configs/rtx_pro6000.yaml`
the way `bench coloc` resolves it, including caps the yaml does not spell out.

Phase 0 is the concurrency gate (`scripts/gpu_concurrency_probe.py`) and Phase 1
is the solo baselines, which every plan generates for itself. Neither is a
colocation you invoke. Phase 5 is excluded from `--all` and run separately.

## Reading the tables

| Column | Meaning |
|---|---|
| Offered rate | Requests per second the load generator is told to send. Open-loop: it does not wait for a response before sending the next request. |
| Arrival | How those requests are spaced. `poisson` is random arrivals at that mean rate; `constant` is evenly spaced. |
| GPU memory fraction | The tenant's `gpu_memory_utilization`. A target for **total device** utilisation, not a private reservation: vLLM sizes its cache from `total x fraction` minus memory already in use by any process on the card. Triton tenants do not take one. |
| KV cache budget | The colocation's `kv_budget_gb`, passed to vLLM as an absolute size in bytes. This, not the fraction, is what fixes the cache. |
| Weights | Checkpoint size, used by the memory pre-flight. `not set` means that tenant is invisible to the check. |
| Output tokens | Generated tokens per request, taken from the workload. |

---

## Phase 2 — Same-category pairs

**Customer's intent.** Fixed model mix per category, sweep concurrency to find saturation curves

**Question this answers.** Do two models of the same kind contend worse than a mixed pair?

### Implemented

#### `same-cv`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| cv_a | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 1 req/s | poisson | `cv_detect_default` | not set | n/a |
| cv_b | `dinov2-base` | Triton, tensorrt backend | fp16 | not set | 1 req/s | poisson | `cv_embed` | not set | n/a |

**Swept:** offered rate on every tenant simultaneously, across [1, 10, 50, 200] requests per second.

#### `same-ilm`

KV cache budget **20 GB**

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| ilm_a | `kosmos-2.5` | Triton, python backend | bf16 | 5.5 GB | 0.1 req/s | poisson | `ilm_document` | 256 | n/a |
| ilm_b | `qwen2.5-vl-7b` | vLLM | awq | 7 GB | 0.1 req/s | poisson | `ilm_document` | 256 | 0.3 |

**Swept:** offered rate on every tenant simultaneously, across [0.1, 0.2, 0.4, 0.8] requests per second.

#### `same-llm`

KV cache budget **20 GB**

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm_a | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 1 req/s | poisson | `llm_short` | 32 | 0.39 |
| llm_b | `gemma2-9b` | vLLM | bf16 | 18.4 GB | 1 req/s | poisson | `llm_short` | 32 | 0.42 |

**Swept:** offered rate on every tenant simultaneously, across [1, 4, 16, 64] requests per second.

#### `same-vlm`

KV cache budget **16 GB**

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| vlm_a | `qwen2.5-vl-7b` | vLLM | awq | 7 GB | 0.5 req/s | poisson | `vlm_video_long` | 128 | 0.26 |
| vlm_b | `gemma-4-31b-it-fp8` | vLLM | fp8 | not set | 0.5 req/s | poisson | `vlm_video_long` | 128 | 0.51 |

**Swept:** offered rate on every tenant simultaneously, across [0.5, 1, 2, 4] requests per second.

### Not implemented, and why

- The customer swept *concurrency* (c1 to c16). Built as an offered-rate sweep instead: a fixed concurrency is closed-loop, and a closed-loop client throttles itself in proportion to the slowdown it is meant to measure.
- `rfdetr-base` is not in the roster, so `same-cv` pairs yolov8-l with dinov2-base rather than three CV models.

### What to observe

- Throughput kept per tenant against its solo baseline. The sweep point where it falls below 1.0 is the pair's capacity.
- Whether the two tenants degrade symmetrically. They rarely do, and the asymmetry is the finding.
- Whether achieved rate still tracks offered rate. Where it stops, the pair is saturated rather than merely slower.

---

## Phase 3 — Mixed pairings at one fixed load

> **The ILM tenant is slow, and its window is long because of it.**
> `kosmos-2.5` sustains **0.133 requests per second** on this hardware — measured
> solo, with 2.0 s of compute per request. It runs at 0.1 (about 75% of that) so
> it has headroom to lose, in a **600 s** window so that 0.1 rps still yields 60
> requests: at 180 s it would yield 18, and a p95 over 18 samples is the
> second-worst value rather than a percentile. An earlier pass at 0.2 rps had it
> saturated in its own baseline, where a neighbour arriving only deepens an
> existing queue and the ratio returns 1.00 for the wrong reason.

**Customer's intent.** Fixed concurrency (c4/rps50), vary model composition across all 8 mix types

**Question this answers.** What does each realistic pairing cost, measured at one fixed load?

### Implemented

#### `mix-full`

KV cache budget **20 GB**

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 4 req/s | poisson | `llm_short` | 32 | 0.39 |
| vlm | `qwen2.5-vl-7b` | vLLM | awq | 7 GB | 1 req/s | poisson | `vlm_video_long` | 128 | 0.3 |
| ilm | `kosmos-2.5` | Triton, python backend | bf16 | 5.5 GB | 0.2 req/s | poisson | `ilm_document` | 256 | n/a |
| cv | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_detect_default` | not set | n/a |

#### `mix-ilm-cv`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| ilm | `kosmos-2.5` | Triton, python backend | bf16 | 5.5 GB | 0.2 req/s | poisson | `ilm_document` | 256 | n/a |
| cv | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_detect_default` | not set | n/a |

#### `mix-llm-cv`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 4 req/s | poisson | `llm_short` | 32 | 0.45 |
| cv | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_detect_default` | not set | n/a |

#### `mix-vlm-cv`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| vlm | `qwen2.5-vl-7b` | vLLM | awq | 7 GB | 1 req/s | poisson | `vlm_video_long` | 128 | 0.5 |
| cv | `dinov2-base` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_embed` | not set | n/a |

#### `mix-vlm-ilm`

KV cache budget **20 GB**

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| vlm | `qwen2.5-vl-7b` | vLLM | awq | 7 GB | 1 req/s | poisson | `vlm_video_long` | 128 | 0.3 |
| ilm | `kosmos-2.5` | Triton, python backend | bf16 | 5.5 GB | 0.2 req/s | poisson | `ilm_document` | 256 | n/a |

### Not implemented, and why

- The customer listed 8 mix types including llm-only and cv-only. Those are same-category pairs and belong to Phase 2, so Phase 3 carries the 5 genuinely mixed compositions.
- `llm+vlm` has no colocation of its own. `cross-vlm-prefill-vs-llm` in Phase 4 covers that pair with a sharper question — a video prefill burst against a decoding LLM.

### What to observe

- End-to-end p95 ratio per tenant. This is the headline number for each pairing.
- The contention matrix read in both directions. Victim by aggressor is not symmetric.
- GPU sampler memory and utilisation for the window, which explains a ratio the latencies alone do not.

---

## Phase 4 — Cross-type characterisation

**Customer's intent.** Characterize specific contention interactions — one model as subject, sweep neighbor load

**Question this answers.** At what neighbour load does degradation begin, and what drives it?

### Implemented

#### `cross-arch-validation`

inherits tenants from `mix-llm-cv`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 4 req/s | poisson | `llm_short` | 32 | 0.45 |
| cv | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_detect_default` | not set | n/a |

**Varied:** `model` on the `llm` tenant, across ['qwen2.5-7b', 'llama3.1-8b', 'mistral-7b'].

#### `cross-cv-vs-llm-rps`

inherits tenants from `mix-llm-cv`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 1 req/s | poisson | `llm_short` | 32 | 0.45 |
| cv | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_detect_default` | not set | n/a |

**Swept:** offered rate on the `llm` tenant only, across [1, 4, 16, 64] requests per second.

#### `cross-ilm-vs-cv`

inherits tenants from `mix-ilm-cv`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| ilm | `kosmos-2.5` | Triton, python backend | bf16 | 5.5 GB | 0.2 req/s | poisson | `ilm_document` | 256 | n/a |
| cv | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 1 req/s | poisson | `cv_detect_default` | not set | n/a |

**Swept:** offered rate on the `cv` tenant only, across [1, 10, 50, 200] requests per second.

#### `cross-llm-vs-cv-rps`

inherits tenants from `mix-llm-cv`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 4 req/s | poisson | `llm_short` | 32 | 0.45 |
| cv | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 1 req/s | poisson | `cv_detect_default` | not set | n/a |

**Swept:** offered rate on the `cv` tenant only, across [1, 10, 50, 200] requests per second.

#### `cross-memory-pressure-kv03`

repetitions **3**

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| anchor | `qwen2.5-72b` | vLLM | awq | 45 GB | 2 req/s | poisson | `llm_short` | 32 | 0.51 |
| neighbour | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 4 req/s | poisson | `llm_short` | 32 | 0.19 |

#### `cross-memory-pressure-kv13`

inherits tenants from `cross-memory-pressure-kv03`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| anchor | `qwen2.5-72b` | vLLM | awq | 45 GB | 2 req/s | poisson | `llm_short` | 32 | 0.58 |
| neighbour | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 4 req/s | poisson | `llm_short` | 32 | 0.22 |

#### `cross-memory-pressure-kv22`

inherits tenants from `cross-memory-pressure-kv03`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| anchor | `qwen2.5-72b` | vLLM | awq | 45 GB | 2 req/s | poisson | `llm_short` | 32 | 0.64 |
| neighbour | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 4 req/s | poisson | `llm_short` | 32 | 0.26 |

#### `cross-memory-pressure-kv29`

inherits tenants from `cross-memory-pressure-kv03`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| anchor | `qwen2.5-72b` | vLLM | awq | 45 GB | 2 req/s | poisson | `llm_short` | 32 | 0.69 |
| neighbour | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 4 req/s | poisson | `llm_short` | 32 | 0.28 |

#### `cross-size-scaling`

KV cache budget **16 GB** · inherits tenants from `mix-llm-cv`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 4 req/s | poisson | `llm_short` | 32 | 0.35 |
| cv | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_detect_default` | not set | n/a |

**Varied:** `model` on the `llm` tenant, across ['qwen2.5-7b', 'qwen2.5-14b', 'qwen2.5-32b', 'qwen2.5-72b'].

#### `cross-vlm-prefill-vs-llm`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| vlm | `qwen2.5-vl-7b` | vLLM | awq | 7 GB | 1 req/s | poisson | `vlm_video_long` | 128 | 0.45 |
| llm | `gemma2-9b` | vLLM | bf16 | 18.4 GB | 4 req/s | poisson | `llm_short` | 32 | 0.35 |

### Not implemented, and why

- Two of the four memory-pressure rungs are gone. `(72b, 32b)` is 110.5 GB of weights and will not load on a 96 GB card; `(vl-72b, 72b)` needs a checkpoint the roster does not carry. Rebuilt as a KV-budget sweep on one pair that fits, which isolates cache pressure instead of confounding it with model size.
- `qwen2.5-27b` does not exist as a HuggingFace repository. The size ladder runs 7B, 14B, 32B, 72B.

### What to observe

- The knee: the first swept rate at which the fixed tenant's p95 ratio leaves 1.0.
- For the size ladder, whether damage scales with parameter count or with KV cache footprint.
- For the KV sweep, whether behaviour turns bimodal near the memory ceiling. `cross-memory-pressure-kv29` runs 3 repetitions for exactly that reason.

---

## Phase 5 — Two GPUs (placement)

> **Run 2026-08-04 — 15/15 clean, and the prediction was wrong.** Worst-tenant
> end-to-end p95, each tenant against its own baseline *on the same card*:
>
> | Placement | GPU 0 | GPU 1 | worst p95 | mean p95 |
> |---|---|---|---|---|
> | one card (`mix-full`, Phase 3) | all four | — | 2.88× | 2.20× |
> | `place-p3` | llm + cv | vlm + ilm | 2.19× | 1.35× |
> | `place-p1` | llm + vlm | ilm + cv | 1.95× | 1.38× |
> | **`place-p2`** | **llm + ilm** | **vlm + cv** | **1.46×** | **1.23×** |
> | `place-vlm-prefill-split` | vlm | gemma2-9b | 1.00× | 0.99× |
> | `place-isolated` (null test) | llm | cv | 1.02× | 1.01× |
>
> Predicted P1 > P3 > P2; measured **P2 best**. Two rules the data supports:
> never co-locate the two vLLM tenants (the VLM pays 1.95× in P1, 1.02× once
> split), and then pair the CV tenant with the VLM rather than the LLM (1.46×
> versus 2.19×) — the LLM's steady decode leaves no gaps, the VLM's bursty
> prefill does. Full reasoning in [contention.md](contention.md) §3.
>
> Caveat: measured at a load point where the LLM was bandwidth-saturated (99%
> utilisation, 77% memory bandwidth, alone) and the second card near idle (20%,
> 1%). The ranking describes that asymmetry.

**Customer's intent.** Repeat key scenarios with 2 and 4 GPUs — does adding GPUs eliminate contention or just redistribute it?

**Question this answers.** Does a second GPU remove contention, or only move it?

### Implemented

#### `place-isolated`

inherits tenants from `mix-llm-cv`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm (GPU 0) | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 4 req/s | poisson | `llm_short` | 32 | 0.45 |
| cv (GPU 1) | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_detect_default` | not set | n/a |

#### `place-p1`

KV cache budget **20 GB** · inherits tenants from `mix-full`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm (GPU 0) | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 4 req/s | poisson | `llm_short` | 32 | 0.39 |
| vlm (GPU 0) | `qwen2.5-vl-7b` | vLLM | awq | 7 GB | 1 req/s | poisson | `vlm_video_long` | 128 | 0.3 |
| ilm (GPU 1) | `kosmos-2.5` | Triton, python backend | bf16 | 5.5 GB | 0.2 req/s | poisson | `ilm_document` | 256 | n/a |
| cv (GPU 1) | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_detect_default` | not set | n/a |

#### `place-p2`

KV cache budget **20 GB** · inherits tenants from `mix-full`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm (GPU 0) | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 4 req/s | poisson | `llm_short` | 32 | 0.39 |
| vlm (GPU 1) | `qwen2.5-vl-7b` | vLLM | awq | 7 GB | 1 req/s | poisson | `vlm_video_long` | 128 | 0.3 |
| ilm (GPU 0) | `kosmos-2.5` | Triton, python backend | bf16 | 5.5 GB | 0.2 req/s | poisson | `ilm_document` | 256 | n/a |
| cv (GPU 1) | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_detect_default` | not set | n/a |

#### `place-p3`

KV cache budget **20 GB** · inherits tenants from `mix-full`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm (GPU 0) | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 4 req/s | poisson | `llm_short` | 32 | 0.39 |
| vlm (GPU 1) | `qwen2.5-vl-7b` | vLLM | awq | 7 GB | 1 req/s | poisson | `vlm_video_long` | 128 | 0.3 |
| ilm (GPU 1) | `kosmos-2.5` | Triton, python backend | bf16 | 5.5 GB | 0.2 req/s | poisson | `ilm_document` | 256 | n/a |
| cv (GPU 0) | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_detect_default` | not set | n/a |

#### `place-vlm-prefill-split`

inherits tenants from `cross-vlm-prefill-vs-llm`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| vlm (GPU 0) | `qwen2.5-vl-7b` | vLLM | awq | 7 GB | 1 req/s | poisson | `vlm_video_long` | 128 | 0.45 |
| llm (GPU 1) | `gemma2-9b` | vLLM | bf16 | 18.4 GB | 4 req/s | poisson | `llm_short` | 32 | 0.35 |

### Not implemented, and why

- The customer swept 2 and 4 GPUs. The 4-GPU arm is dropped by decision — this machine has two cards. The `device:` schema accepts up to 8, so restoring it is a configuration change rather than a code change.
- No tensor-parallel placements are written. `nvidia-smi topo -m` reports PIX, so cross-GPU traffic is PCIe Gen5 and a tensor-parallel result would be dominated by the interconnect rather than by contention.

### What to observe

- `place-isolated` must return a ratio of about 1.0. Its two tenants share no streaming multiprocessors, no bandwidth and no memory. If it degrades, the harness is wrong and nothing else it reports can be trusted.
- The ranking of the three pairings. The prediction on record is P1 best, then P3, then P2.
- One GPU sampler block per occupied card. A placement result with telemetry for only one card cannot be explained.

---

## Phase 6 — Secondary dimensions

**Customer's intent.** Sweep secondary dimensions against TWO contrasting baselines — confirms whether dimension matters and whether its effect is context-dependent

**Question this answers.** Does a secondary setting change the answer, and does it change it differently under memory pressure?

### Implemented

#### `mix-memory-bound`

KV cache budget **6 GB**

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm | `qwen2.5-72b` | vLLM | awq | 45 GB | 2 req/s | poisson | `llm_short` | 32 | 0.55 |
| llm2 | `qwen2.5-14b` | vLLM | bf16 | 29.5 GB | 2 req/s | poisson | `llm_short` | 32 | 0.39 |
| cv | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_detect_default` | not set | n/a |

#### `secondary-arrival-a`

inherits tenants from `mix-llm-cv`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 4 req/s | poisson | `llm_short` | 32 | 0.45 |
| cv | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_detect_default` | not set | n/a |

**Varied:** `load` on the `cv` tenant, across [{'pattern': 'poisson', 'rps': 50}, {'pattern': 'constant', 'rps': 50}].

#### `secondary-arrival-b`

inherits tenants from `mix-memory-bound`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm | `qwen2.5-72b` | vLLM | awq | 45 GB | 2 req/s | poisson | `llm_short` | 32 | 0.55 |
| llm2 | `qwen2.5-14b` | vLLM | bf16 | 29.5 GB | 2 req/s | poisson | `llm_short` | 32 | 0.39 |
| cv | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_detect_default` | not set | n/a |

**Varied:** `load` on the `cv` tenant, across [{'pattern': 'poisson', 'rps': 50}, {'pattern': 'constant', 'rps': 50}].

#### `secondary-asymmetry-a`

inherits tenants from `mix-llm-cv`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 4 req/s | poisson | `llm_short` | 32 | 0.45 |
| cv | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 4 req/s | poisson | `cv_detect_default` | not set | n/a |

**Swept:** offered rate on the `cv` tenant only, across [4, 16, 64] requests per second.

#### `secondary-asymmetry-b`

inherits tenants from `mix-memory-bound`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm | `qwen2.5-72b` | vLLM | awq | 45 GB | 2 req/s | poisson | `llm_short` | 32 | 0.55 |
| llm2 | `qwen2.5-14b` | vLLM | bf16 | 29.5 GB | 2 req/s | poisson | `llm_short` | 32 | 0.39 |
| cv | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 4 req/s | poisson | `cv_detect_default` | not set | n/a |

**Swept:** offered rate on the `cv` tenant only, across [4, 16, 64] requests per second.

#### `secondary-backend-cv-a`

inherits tenants from `mix-llm-cv`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 4 req/s | poisson | `llm_short` | 32 | 0.45 |
| cv | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_detect_default` | not set | n/a |

**Varied:** `triton_backend` on the `cv` tenant, across ['tensorrt', 'onnx', 'python'].

#### `secondary-backend-cv-b`

inherits tenants from `mix-memory-bound`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm | `qwen2.5-72b` | vLLM | awq | 45 GB | 2 req/s | poisson | `llm_short` | 32 | 0.55 |
| llm2 | `qwen2.5-14b` | vLLM | bf16 | 29.5 GB | 2 req/s | poisson | `llm_short` | 32 | 0.39 |
| cv | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_detect_default` | not set | n/a |

**Varied:** `triton_backend` on the `cv` tenant, across ['tensorrt', 'onnx', 'python'].

#### `secondary-backend-llm-a`

inherits tenants from `mix-llm-cv`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 4 req/s | poisson | `llm_short` | 32 | 0.45 |
| cv | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_detect_default` | not set | n/a |

**Varied:** `backend` on the `llm` tenant, across ['vllm', 'sglang'].

#### `secondary-backend-llm-b`

inherits tenants from `mix-memory-bound`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm | `qwen2.5-72b` | vLLM | awq | 45 GB | 2 req/s | poisson | `llm_short` | 32 | 0.55 |
| llm2 | `qwen2.5-14b` | vLLM | bf16 | 29.5 GB | 2 req/s | poisson | `llm_short` | 32 | 0.39 |
| cv | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_detect_default` | not set | n/a |

**Varied:** `backend` on the `llm` tenant, across ['vllm', 'sglang'].

#### `secondary-input-size-cv-a`

inherits tenants from `mix-llm-cv`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 4 req/s | poisson | `llm_short` | 32 | 0.45 |
| cv | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_detect_small` | not set | n/a |

**Varied:** `workload` on the `cv` tenant, across ['cv_detect_small', 'cv_detect_large'].

#### `secondary-input-size-cv-b`

inherits tenants from `mix-memory-bound`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm | `qwen2.5-72b` | vLLM | awq | 45 GB | 2 req/s | poisson | `llm_short` | 32 | 0.55 |
| llm2 | `qwen2.5-14b` | vLLM | bf16 | 29.5 GB | 2 req/s | poisson | `llm_short` | 32 | 0.39 |
| cv | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_detect_small` | not set | n/a |

**Varied:** `workload` on the `cv` tenant, across ['cv_detect_small', 'cv_detect_large'].

#### `secondary-input-size-llm-a`

inherits tenants from `mix-llm-cv`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 4 req/s | poisson | `llm_short` | 32 | 0.45 |
| cv | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_detect_default` | not set | n/a |

**Varied:** `workload` on the `llm` tenant, across ['llm_short', 'llm_long_prompt'].

#### `secondary-input-size-llm-b`

inherits tenants from `mix-memory-bound`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm | `qwen2.5-72b` | vLLM | awq | 45 GB | 2 req/s | poisson | `llm_short` | 32 | 0.55 |
| llm2 | `qwen2.5-14b` | vLLM | bf16 | 29.5 GB | 2 req/s | poisson | `llm_short` | 32 | 0.39 |
| cv | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_detect_default` | not set | n/a |

**Varied:** `workload` on the `llm` tenant, across ['llm_short', 'llm_long_prompt'].

#### `secondary-output-length-a`

inherits tenants from `mix-llm-cv`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 4 req/s | poisson | `llm_short` | 32 | 0.45 |
| cv | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_detect_default` | not set | n/a |

**Varied:** `workload` on the `llm` tenant, across ['llm_short', 'llm_long'].

#### `secondary-output-length-b`

inherits tenants from `mix-memory-bound`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm | `qwen2.5-72b` | vLLM | awq | 45 GB | 2 req/s | poisson | `llm_short` | 32 | 0.55 |
| llm2 | `qwen2.5-14b` | vLLM | bf16 | 29.5 GB | 2 req/s | poisson | `llm_short` | 32 | 0.39 |
| cv | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_detect_default` | not set | n/a |

**Varied:** `workload` on the `llm` tenant, across ['llm_short', 'llm_long'].

### Not implemented, and why

- The `q4` quantization dimension is dropped. Q4_0 is a llama.cpp GGUF format and vLLM cannot load it; each model runs at the best format it has.
- The `llamacpp` backend is omitted. No Triton backend exists for it, and it adds no contention axis the other three do not already cover.

### What to observe

- Whether the setting moves the ratio at all.
- Whether it moves it by the same amount under the light and the heavy baseline. A difference is an interaction effect.
- For backend swaps, whether the change is the backend itself or the memory footprint that came with it.

---
