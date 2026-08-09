# Contention benchmark — phases, configuration, and what to observe

The customer's design is `workspace/contention/experiment_design.md` and
`workspace/contention/experiment_config.json`. This file states, per phase, what
was built against it — with the configured values rather than the intended ones.
Every number in the tables is resolved from `benchmarks/configs/rtx_pro6000.yaml`
the way `bench coloc` resolves it, including caps the yaml does not spell out.

Phase 0 is the concurrency gate (`scripts/gpu_concurrency_probe.py`) and Phase 1
is the solo baselines, which every plan generates for itself. Neither is a
colocation you invoke.

There are **41 colocations**. `--all` selects every one of them — 41
colocations, 154 runs (66 deduped solo baselines + 88 contention windows). Phase
5 needs two cards, so on a single-GPU box run the study as
`--phase 2 --phase 3 --phase 4 --phase 6` (36 colocations, 145 runs) and Phase 5
separately (`--phase 5`: 5 colocations, 14 runs).

## Reading the tables

| Column | Meaning |
|---|---|
| Offered rate | Requests per second the load generator is told to send. Open-loop: it does not wait for a response before sending the next request. |
| Arrival | How those requests are spaced. `poisson` is random arrivals at that mean rate; `constant` is evenly spaced. |
| GPU memory fraction | The tenant's `gpu_memory_utilization`. A target for **total device** utilisation, not a private reservation: vLLM sizes its cache from `total x fraction` minus memory already in use by any process on the card. Triton tenants do not take one. |
| KV cache budget | `kv_budget_gb` — stated per tenant, usually inherited from one colocation-wide value. Passed to vLLM as an absolute `--kv-cache-memory-bytes`, and to SGLang as `--max-total-tokens` plus a permissive `--mem-fraction-static`. This, not the fraction, is what fixes the cache. Where a tenant overrides it, the table says so per row. |
| Weights | Checkpoint size, used by the memory pre-flight. `not set` means that tenant is invisible to the check. |
| Output tokens | Generated tokens per request, taken from the workload. |

## What the phases build toward

No phase stands alone; each is only interpretable given the one before it.

```
Phase 0  ->  does co-residency even happen?          gate
Phase 1  ->  how fast is each model alone?           the reference
Phase 2  ->  do same-kind models fight?              screen, load swept
Phase 3  ->  what does each realistic pairing cost?  screen, load fixed
Phase 4  ->  how bad, and where is the knee?         characterise, curves
Phase 5  ->  does a second GPU fix it?               placement
Phase 6  ->  does the answer survive changing X?     robustness
```

Phases 2 and 3 **screen**: they find which combinations hurt, cheaply, at one
or a few load points. Phase 4 **characterises**: it takes a pairing that
screening flagged and sweeps it until something breaks, which is the only way
to find a knee. Phase 5 asks whether hardware fixes what scheduling could not,
and Phase 6 asks whether any of it still holds when you change the backend, the
prompt, or the arrival pattern.

A screening result with no curve behind it tells you a pairing costs *something*
at *one* load. A curve with no screening behind it tells you a great deal about
a pairing you had no reason to care about.

---

## Phase 2 — Same-category pairs

**Customer's intent.** Fixed model mix per category, sweep concurrency to find saturation curves

**Question this answers.** Do two models of the same kind contend worse than a mixed pair?

> **Measured 2026-08-04/05.** Same-category cost, worst tenant, end-to-end p95:
>
> | Pair | 1 -> 16 req/s each | at 64 |
> |---|---|---|
> | `same-cv` (yolov8-l + dinov2-base) | 1.0-1.9x | not swept that high |
> | `same-llm` (qwen2.5-7b + gemma2-9b) | **1.5-1.9x, flat** | **33-37x** |
> | `same-ilm` | ~1.0x | rates too low to resolve |
> | `same-vlm` | 3-11x, but superseded — see below | not swept |
>
> **The answer is yes, and the shape matters more than the size.** Two LLMs cost
> a stable, predictable tax across a 16x range of load, then fall off a cliff.
>
> The cliff's signature is the finding: at 64 req/s TTFT p95 degraded **600x**
> while end-to-end degraded 33x and inter-token latency stayed under 2x.
> Requests are queueing for admission, not computing slowly — once one starts it
> runs at nearly full speed. **GPU utilisation and token-latency monitoring give
> no warning**, because neither is what moved. Full write-up:
> [findings/same-llm-colocation-envelope.md](findings/same-llm-colocation-envelope.md).
>
> Achieved rate fell below offered at the same point (47.9 of 64, and 34.2 of
> 64), which is the safe-operating-envelope boundary rather than a measurement
> error.
>
> **`same-vlm`'s numbers are superseded, and the colocation has been rebuilt
> twice.** Its original second tenant was `gemma-4-31b-it-fp8`, which never
> loaded — transformers 4.57.6 has no `gemma4` architecture — and took all eight
> of its runs with it. Rebuilt on `qwen3-vl-32b-fp8`, it then ran, but the
> `[0.5, 1, 2, 4]` shared sweep was not measuring load: the two tenants' ceilings
> are **12x apart** (4 req/s against ~0.33 req/s), so three of the four rungs sat
> above the 32B's limit and collapsed into one load point measured four times
> (achieved 0.30 / 0.29 / 0.28 / 0.24) while the 7B was still nearly idle. The
> sweep is gone; each tenant now sits at about half its own measured ceiling, and
> the pair has not been re-measured at those rates.
>
> **A shared `"*"` sweep is only meaningful when the tenants have comparable
> ceilings.** That is a rule rather than a `same-vlm` quirk, and it is why
> `same-llm`'s shared sweep is sound and this one was not.

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
| ilm_a | `kosmos-2.5` | Triton, python backend | bf16 | 5.5 GB | 0.03 req/s | poisson | `ilm_document` | 256 | n/a |
| ilm_b | `qwen2.5-vl-7b` | vLLM | awq | 7 GB | 0.03 req/s | poisson | `ilm_document` | 256 | 0.3 |

**Swept:** offered rate on every tenant simultaneously, across [0.03, 0.06, 0.1, 0.2] requests per second.

Scaled to the ILM tenant's *measured* ceiling: `kosmos-2.5` spends ~2.2 s of
compute on a 256-token document request and sustains **0.133 req/s**, so earlier
versions of this sweep sat entirely above saturation and every rung would have
measured a queue rather than contention. This one walks from headroom through
the knee.

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
| vlm_a | `qwen2.5-vl-7b` | vLLM | awq | 7 GB | 2 req/s | poisson | `vlm_video_long` | 128 | 0.26 |
| vlm_b | `qwen3-vl-32b-fp8` | vLLM | fp8 | 35.5 GB | 0.15 req/s | poisson | `vlm_video_long` | 128 | 0.56 |

**Not swept.** Each tenant sits at about half its own measured ceiling — 4 req/s
for the 7B, ~0.33 req/s for the 32B — which is the same *relative* load for
both. A shared `"*"` sweep would move them to the same *absolute* rate, and with
ceilings 12x apart that asks a question neither tenant can answer.

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

> **Measured 2026-08-04/05 — 12 clean runs.** Worst tenant, end-to-end p95:
>
> | Colocation | Worst tenant |
> |---|---|
> | `mix-llm-cv` | 1.02x |
> | `mix-vlm-cv` | 1.03x |
> | `mix-ilm-cv` | 1.05x |
> | `mix-vlm-ilm` | 1.04x |
> | **`mix-full`** (all four) | **2.46x** |
>
> **Every pair is essentially free at this load; four tenants is not — and the
> four-way cost cannot be predicted from the pairwise ones.** That
> non-additivity is the phase's real result, and it is why `mix-full` exists
> rather than being inferred.
>
> Inside `mix-full`, who pays is as informative as how much:
>
> | Tenant | e2e p95 |
> |---|---|
> | cv (`yolov8-l`) | **2.46x** |
> | vlm | 2.07x |
> | ilm | 1.75x |
> | llm | 1.40x |
>
> The smallest, fastest tenant absorbs the most. The same absolute interference
> is a disaster for a 7 ms detector and a shrug for a 500 ms LLM, so a single
> "how contended is this card" number would hide the only part that matters.

### Implemented

#### `mix-full`

KV cache budget **20 GB**

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 4 req/s | poisson | `llm_short` | 32 | 0.39 |
| vlm | `qwen2.5-vl-7b` | vLLM | awq | 7 GB | 1 req/s | poisson | `vlm_video_long` | 128 | 0.3 |
| ilm | `kosmos-2.5` | Triton, python backend | bf16 | 5.5 GB | 0.1 req/s | poisson | `ilm_document` | 256 | n/a |
| cv | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_detect_default` | not set | n/a |

#### `mix-ilm-cv`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| ilm | `kosmos-2.5` | Triton, python backend | bf16 | 5.5 GB | 0.1 req/s | poisson | `ilm_document` | 256 | n/a |
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
| ilm | `kosmos-2.5` | Triton, python backend | bf16 | 5.5 GB | 0.1 req/s | poisson | `ilm_document` | 256 | n/a |

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

> **Measured 2026-08-04/05.** The deployment-cost question — what does adding a
> second model to an occupied card actually cost? Both vLLM under MPS,
> `llm_long`, 300 s windows, 12 contention runs, reproducible to two decimals:
>
> | | alone (whole card) | colocated | kept |
> |---|---|---|---|
> | `qwen2.5-72b` | 1.52 req/s | 0.88 | **58%** |
> | `qwen2.5-7b` | 3.81 req/s | 2.48 | **65%** |
>
> Aggregate rises to 3.36 req/s. **The card does more total work while both
> tenants get materially slower** — that is the co-location trade stated in one
> line, and whether it is worth taking is a business question rather than a GPU
> one. Treat 42% as an *optimistic* bound: it was measured with prefill
> essentially free, so realistic prompt diversity should cost more.
>
> **How you split the memory does not matter.** Four splits of the same 28 GiB
> leftover gave identical throughput across a 3.4x range of 72B cache and 5x of
> 7B cache (0.88 and 2.48 req/s at every split). The cost is compute contention;
> KV was not the binding constraint anywhere in that range.
>
> **The KV knee is 5-7 GiB for the 72B**, and the 7B is flat across a **110x**
> cache range — it needs under 1 GiB on this workload, and anything beyond that
> is wasted card. Why three successive designs of this experiment missed the
> knee: [findings/kv-cache-knee-and-prefix-caching.md](findings/kv-cache-knee-and-prefix-caching.md).

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

#### `cross-deploy-*` — the deployment-cost family

Six colocations meant to be read as one experiment: two whole-card references
(`-alone-72b`, `-alone-7b`) and four ways of dividing the leftover between the
same two models (`-split-s25/s50/s75/s85`). All six run `llm_long` (989-token
prompt, 512 output tokens) in **300 s** windows, both tenants on vLLM under MPS;
each split runs **`repetitions: 3`**.

**What it answers.** Every other family holds the KV cache constant so that the
neighbour is the only variable. That isolates compute contention at a chosen
cache size — a mechanism curve — but it is not what an operator experiences. In
a deployment nobody picks the cache: you have a card running one model with
everything to itself, you add a second, and each now gets a share of what is left
after both sets of weights. The cache shrinks *because of* the neighbour. So here
the baseline is the model **alone with the whole card**, and the contention arms
divide the leftover — the measured cost is compute contention **and** cache loss
together. The swept variable is the **split**, because that is the operator's
actual knob.

The arithmetic the splits rest on, measured on this card: weights 38.77 GiB
(72B AWQ) + 14.25 GiB (7B) + ~4 GiB overhead = 57.0 GiB; of 89.4 GiB usable that
leaves 32.4 GiB, of which **28.0 GiB** is divided (95%, for headroom). `s85` is
the split that holds the same number of *sequences* for both tenants — a 72B
token costs 320 KiB against the 7B's 56 KiB — which is the natural "fair" point
and nowhere near an equal-gigabyte split.

Every tenant here states its **own** `kv_budget_gb`; nothing is inherited.

| Colocation | Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | KV cache budget | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `cross-deploy-alone-72b` | llm | `qwen2.5-72b` | vLLM | awq | 41.6 GB | 2 req/s | poisson | `llm_long` | 512 | **46 GB** | 0.93 |
| `cross-deploy-alone-7b` | llm | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 4 req/s | poisson | `llm_long` | 512 | **70 GB** | 0.91 |
| `cross-deploy-split-s25` | anchor | `qwen2.5-72b` | vLLM | awq | 41.6 GB | 2 req/s | poisson | `llm_long` | 512 | **7 GB** | 0.53 |
| | neighbour | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 4 req/s | poisson | `llm_long` | 512 | **21 GB** | 0.4 |
| `cross-deploy-split-s50` | anchor | `qwen2.5-72b` | vLLM | awq | 41.6 GB | 2 req/s | poisson | `llm_long` | 512 | **14 GB** | 0.6 |
| | neighbour | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 4 req/s | poisson | `llm_long` | 512 | **14 GB** | 0.33 |
| `cross-deploy-split-s75` | anchor | `qwen2.5-72b` | vLLM | awq | 41.6 GB | 2 req/s | poisson | `llm_long` | 512 | **21 GB** | 0.67 |
| | neighbour | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 4 req/s | poisson | `llm_long` | 512 | **7 GB** | 0.25 |
| `cross-deploy-split-s85` | anchor | `qwen2.5-72b` | vLLM | awq | 41.6 GB | 2 req/s | poisson | `llm_long` | 512 | **23.8 GB** | 0.7 |
| | neighbour | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 4 req/s | poisson | `llm_long` | 512 | **4.2 GB** | 0.22 |

`s25`, `s75` and `s85` are written as `extends: cross-deploy-split-s50` and
override nothing but the two budgets, so the split is provably the only variable
between them. Offered rates never move across the whole family, so a change in
throughput is never a change in what was asked for.

**What to observe.**

- Throughput kept against the *whole-card* reference, per tenant, and the
  **aggregate** across both. Those two move in opposite directions and both are
  the result.
- Whether the four splits differ at all. Measured, they do not: 0.88 and 2.48
  req/s at every split, across a 3.4x range of 72B cache and 5x of 7B cache.
- `GPU KV cache usage` in each tenant's `server.log`. Single-digit percentages
  mean the split moved nothing real, which is exactly what prefix caching did
  here — see [findings/kv-cache-knee-and-prefix-caching.md](findings/kv-cache-knee-and-prefix-caching.md).
- The spread across the three repetitions. Away from a cliff it was nil (12
  contention runs identical to two decimals); near the eviction limit two
  identical reps gave 0.79 and 0.40, and there the spread *is* the finding.

#### `cross-ilm-vs-cv`

inherits tenants from `mix-ilm-cv`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| ilm | `kosmos-2.5` | Triton, python backend | bf16 | 5.5 GB | 0.1 req/s | poisson | `ilm_document` | 256 | n/a |
| cv | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 1 req/s | poisson | `cv_detect_default` | not set | n/a |

**Swept:** offered rate on the `cv` tenant only, across [1, 10, 50, 200] requests per second.

#### `cross-llm-vs-cv-rps`

inherits tenants from `mix-llm-cv`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 4 req/s | poisson | `llm_short` | 32 | 0.45 |
| cv | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 1 req/s | poisson | `cv_detect_default` | not set | n/a |

**Swept:** offered rate on the `cv` tenant only, across [1, 10, 50, 200] requests per second.

#### `cross-size-scaling`

KV cache budget **16 GB** · inherits tenants from `mix-llm-cv`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 4 req/s | poisson | `llm_short` | 32 | 0.35 |
| cv | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_detect_default` | not set | n/a |

**Varied:** `model` on the `llm` tenant, across ['qwen2.5-7b', 'qwen2.5-14b', 'qwen2.5-32b', 'qwen2.5-72b'].

#### `cross-vlm-prefill-vs-llm`

KV cache budget **20 GB**

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| vlm | `qwen2.5-vl-7b` | vLLM | awq | 7 GB | 1 req/s | poisson | `vlm_video_long` | 128 | 0.3 |
| llm | `gemma2-9b` | vLLM | bf16 | 18.4 GB | 4 req/s | poisson | `llm_short` | 32 | 0.42 |

Both fractions are derived from that budget. They used to be hand-written as
0.45 / 0.35, and with two vLLM tenants on one card that killed the run: the `llm`
tenant derived a negative cache and never became ready. KV is not what this
colocation varies, so it is stated once and each fraction absorbs only its own
weights.

### Not implemented, and why

- The customer's memory-pressure ladder swapped in a bigger neighbour at each rung — (72b, 7b), (72b, 14b), (72b, 32b), (vl-72b, 72b) — so every rung changed both how full the card was and which model was being measured. Two rungs are also unbuildable here: `(72b, 32b)` is 110.5 GB of weights and will not load on a 96 GB card, and `(vl-72b, 72b)` needs a checkpoint the roster does not carry. Rebuilt as `cross-deploy-*`, which fixes the models and moves only the memory split — so a throughput change has one possible cause, and the answer transfers to model pairs nobody benchmarked.
- `qwen2.5-27b` does not exist as a HuggingFace repository. The size ladder runs 7B, 14B, 32B, 72B.

### What to observe

- The knee: the first swept rate at which the fixed tenant's p95 ratio leaves 1.0.
- For the size ladder, whether damage scales with parameter count or with KV cache footprint.
- For `cross-deploy-*`, the four observations listed under that family — in particular `GPU KV cache usage` in the server log, without which a flat curve cannot be told apart from a cache that was never the constraint.

---

## Phase 5 — Two GPUs (placement)

> **Measured 2026-08-04 — 15/15 clean.** Worst-tenant
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
> **A second card helps, but only if you split the right pair.** Every
> placement beats one card; the best is 1.5× better than the worst. Two rules,
> in priority order:
>
> 1. **Never co-locate the two autoregressive tenants.** The LLM and the VLM are
>    both KV-hungry token generators competing for the same resource, not
>    different ones. The VLM pays 1.95× when they share a card and 1.02× once
>    split — this single decision dominates the ranking.
> 2. **Then pair the CV tenant with the VLM, not the LLM** (1.46× versus 2.19×).
>    Counter-intuitively the bursty neighbour is the kinder one: prefill bursts
>    leave gaps a 7 ms request can slip into, where steady decode never does.
>
> The intuition to discard is "separate the compute-heavy models" — grouping by
> how *much* work a tenant does is the wrong axis. Group by *which resource* it
> competes for. Full reasoning in [contention.md](contention.md) §3.
>
> Caveat: measured at a **low** load point. Phase 2's sweep shows `qwen2.5-7b`
> sustains 62 of 64 requests/second alone, so the configured `llm@4` is about
> 1/16th of capacity. `gpu_util_pct` reads 99% there, but it measures engine-active
> time rather than occupancy. The ranking holds for the lightly loaded regime —
> which `same-llm` shows is the safe one, flat at 1.6–1.9× to 16 rps — and says
> nothing about behaviour past the cliff at 64 rps.

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
| ilm (GPU 1) | `kosmos-2.5` | Triton, python backend | bf16 | 5.5 GB | 0.1 req/s | poisson | `ilm_document` | 256 | n/a |
| cv (GPU 1) | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_detect_default` | not set | n/a |

#### `place-p2`

KV cache budget **20 GB** · inherits tenants from `mix-full`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm (GPU 0) | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 4 req/s | poisson | `llm_short` | 32 | 0.39 |
| vlm (GPU 1) | `qwen2.5-vl-7b` | vLLM | awq | 7 GB | 1 req/s | poisson | `vlm_video_long` | 128 | 0.3 |
| ilm (GPU 0) | `kosmos-2.5` | Triton, python backend | bf16 | 5.5 GB | 0.1 req/s | poisson | `ilm_document` | 256 | n/a |
| cv (GPU 1) | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_detect_default` | not set | n/a |

#### `place-p3`

KV cache budget **20 GB** · inherits tenants from `mix-full`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm (GPU 0) | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 4 req/s | poisson | `llm_short` | 32 | 0.39 |
| vlm (GPU 1) | `qwen2.5-vl-7b` | vLLM | awq | 7 GB | 1 req/s | poisson | `vlm_video_long` | 128 | 0.3 |
| ilm (GPU 1) | `kosmos-2.5` | Triton, python backend | bf16 | 5.5 GB | 0.1 req/s | poisson | `ilm_document` | 256 | n/a |
| cv (GPU 0) | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_detect_default` | not set | n/a |

#### `place-vlm-prefill-split`

KV cache budget **20 GB** · inherits tenants from `cross-vlm-prefill-vs-llm`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| vlm (GPU 0) | `qwen2.5-vl-7b` | vLLM | awq | 7 GB | 1 req/s | poisson | `vlm_video_long` | 128 | 0.3 |
| llm (GPU 1) | `gemma2-9b` | vLLM | bf16 | 18.4 GB | 4 req/s | poisson | `llm_short` | 32 | 0.42 |

### Not implemented, and why

- The customer swept 2 and 4 GPUs. The 4-GPU arm is dropped by decision — this machine has two cards. The `device:` schema accepts up to 8, so restoring it is a configuration change rather than a code change.
- No tensor-parallel placements are written. `nvidia-smi topo -m` reports PIX, so cross-GPU traffic is PCIe Gen5 and a tensor-parallel result would be dominated by the interconnect rather than by contention.

### What to observe

- `place-isolated` must return a ratio of about 1.0. Its two tenants share no streaming multiprocessors, no bandwidth and no memory. If it degrades, the harness is wrong and nothing else it reports can be trusted.
- The ranking of the three pairings, and specifically whether the two autoregressive tenants share a card in the winner. If they do, the rule above does not hold at your load point and the reason is worth finding.
- One GPU sampler block per occupied card. A placement result with telemetry for only one card cannot be explained.

---

## Phase 6 — Secondary dimensions

**Customer's intent.** Sweep secondary dimensions against TWO contrasting baselines — confirms whether dimension matters and whether its effect is context-dependent

**Question this answers.** Does a secondary setting change the answer, and does it change it differently under memory pressure?

> **Measured 2026-08-04/05.** SGLang **1.98** vs vLLM **1.97** req/s on
> `qwen2.5-72b` under three-tenant contention — parity, *with matched caches*
> (19,660 vs 19,648 tokens).
>
> The matching is the result. Before it, SGLang was being handed a different
> cache size and the comparison measured memory allocation rather than the
> backend. **A secondary-dimension result is only a finding once everything the
> dimension does not name is held equal** — which is exactly what the two
> baselines exist to expose.

### Implemented

#### `mix-memory-bound`

KV cache budget **6 GB**

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm | `qwen2.5-72b` | vLLM | awq | 41.6 GB | 2 req/s | poisson | `llm_short` | 32 | 0.52 |
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
| llm | `qwen2.5-72b` | vLLM | awq | 41.6 GB | 2 req/s | poisson | `llm_short` | 32 | 0.52 |
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
| llm | `qwen2.5-72b` | vLLM | awq | 41.6 GB | 2 req/s | poisson | `llm_short` | 32 | 0.52 |
| llm2 | `qwen2.5-14b` | vLLM | bf16 | 29.5 GB | 2 req/s | poisson | `llm_short` | 32 | 0.39 |
| cv | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 4 req/s | poisson | `cv_detect_default` | not set | n/a |

**Swept:** offered rate on the `cv` tenant only, across [4, 16, 64] requests per second.

#### `secondary-backend-cv-a`

inherits tenants from `mix-llm-cv`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm | `qwen2.5-7b` | vLLM | bf16 | 15.2 GB | 4 req/s | poisson | `llm_short` | 32 | 0.45 |
| cv | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_detect_default` | not set | n/a |

**Varied:** `triton_backend` on the `cv` tenant, across ['tensorrt', 'onnx'].

#### `secondary-backend-cv-b`

inherits tenants from `mix-memory-bound`

| Tenant | Model | Served by | Precision | Weights | Offered rate | Arrival | Workload | Output tokens | GPU memory fraction |
|---|---|---|---|---|---|---|---|---|---|
| llm | `qwen2.5-72b` | vLLM | awq | 41.6 GB | 2 req/s | poisson | `llm_short` | 32 | 0.52 |
| llm2 | `qwen2.5-14b` | vLLM | bf16 | 29.5 GB | 2 req/s | poisson | `llm_short` | 32 | 0.39 |
| cv | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_detect_default` | not set | n/a |

**Varied:** `triton_backend` on the `cv` tenant, across ['tensorrt', 'onnx'].

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
| llm | `qwen2.5-72b` | vLLM | awq | 41.6 GB | 2 req/s | poisson | `llm_short` | 32 | 0.52 |
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
| llm | `qwen2.5-72b` | vLLM | awq | 41.6 GB | 2 req/s | poisson | `llm_short` | 32 | 0.52 |
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
| llm | `qwen2.5-72b` | vLLM | awq | 41.6 GB | 2 req/s | poisson | `llm_short` | 32 | 0.52 |
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
| llm | `qwen2.5-72b` | vLLM | awq | 41.6 GB | 2 req/s | poisson | `llm_short` | 32 | 0.52 |
| llm2 | `qwen2.5-14b` | vLLM | bf16 | 29.5 GB | 2 req/s | poisson | `llm_short` | 32 | 0.39 |
| cv | `yolov8-l` | Triton, tensorrt backend | fp16 | not set | 50 req/s | poisson | `cv_detect_default` | not set | n/a |

**Varied:** `workload` on the `llm` tenant, across ['llm_short', 'llm_long'].

### Not implemented, and why

- `python` is not a rung of the CV backend dimension. Only `kosmos-2.5` has a hand-authored `model.py`; `yolov8-l` fails at staging with "is a python-backend model but has no model.py". The question the dimension asks is what the Triton backend costs under contention, which tensorrt-vs-onnx answers — a Python-backend YOLO would be slower for reasons that have nothing to do with a neighbour.
- The `q4` quantization dimension is dropped. Q4_0 is a llama.cpp GGUF format and vLLM cannot load it; each model runs at the best format it has.
- The `llamacpp` backend is omitted. No Triton backend exists for it, and it adds no contention axis the other three do not already cover.

### What to observe

- Whether the setting moves the ratio at all.
- Whether it moves it by the same amount under the light and the heavy baseline. A difference is an interaction effect.
- For backend swaps, whether the change is the backend itself or the memory footprint that came with it.

---

---

## What the study establishes, across all phases

Four results that survive being read out of context:

**1. Pairs are nearly free; four tenants are not, and the cost is not additive.**
Every two-tenant mix in Phase 3 landed within 5% of its solo baseline. The same
four models together cost 2.46× on the worst tenant. You cannot sum pairwise
costs to plan a four-way deployment.

**2. The smallest, fastest tenant pays the most — always.** In `mix-full` the
7 ms detector took 2.46× while the LLM took 1.40×. Identical absolute
interference, opposite consequence. Any single "how contended is this card"
number averages away the only part that decides whether you can ship.

**3. The failure mode at the edge is queueing, not compute.** At the `same-llm`
cliff, TTFT p95 moved 600× while inter-token latency stayed under 2×. Requests
wait to start, then run at nearly full speed. **Utilisation and token-latency
dashboards show nothing** until achieved rate falls off offered rate — so the
envelope boundary, not a latency threshold, is the alert worth building.

**4. Co-location buys aggregate throughput and sells per-tenant speed.**
58% and 65% kept, aggregate up to 3.36 req/s. That is the whole trade. Whether
it is a good one depends on whether your service level is written per request or
per fleet, which is not a question the GPU can answer.

---

## Tuning for the next run

Measured on 2× RTX PRO 6000, 2026-08-04. Every rate in this config was chosen
before any hardware existed; these are the numbers to choose them from next
time. Nothing here invalidates the current results — it says what regime they
describe.

### The inference backends were never tuned, and one default set a headline number

Every contention tenant runs the backend **stock**. What the config does set was
chosen for **fit** (does the model load) and **experimental control** (is the
cache the same in both runs) — never for speed:

```
--gpu-memory-utilization=0.90   backend-wide; now a ceiling, overridden per tenant
--max-num-seqs=32               backend-wide, EVERY tenant, never tuned
--max-model-len=8192…32768      per model — sized to fit VRAM, not for throughput
--kv-cache-memory-bytes=<N>     experimental control (from kv_budget_gb)
```

The tuning knobs that exist in this yaml — the `eager` and `chunked_off`
variants — are referenced only by the single-model **sweeps**. No colocation
uses one, and `backends.sglang.variants` is empty.

**This is mostly the right call.** A degradation ratio is
`contention ÷ solo` with the identical config on both sides, so a shared
inefficiency largely cancels. Tuning per workload would risk the opposite: if
the solo and contention runs landed on different optimal settings, two things
would have changed and the ratio would mean nothing again.

**But it makes the absolute numbers a floor, not a ceiling.** "1.52 req/s for
the 72B" is *vLLM defaults on this box*, not what the hardware can do. It should
never be quoted as a capability figure.

**And "it cancels out" stops being true when a default limits the mechanism
under test.** `--max-num-seqs=32` is the proof, twice over:

- It **caps residency**, so raising the arrival rate cannot fill the KV cache.
  `llm_short` occupies at most 1,952 tokens — 6.8% of even the smallest rung —
  at *any* rate. That is part of why three generations of the memory-pressure
  experiment produced flat curves.
- It **sets where the `same-llm` cliff falls.** The mechanism there is admission
  queueing: requests waiting to *start*. What decides when they start waiting is
  the sequence limit. Raise it to 128 and the cliff moves right; drop it to 8
  and it moves left.

The cliff is real — 32 is a configuration someone could deploy — but it is the
envelope of *that configuration*, not of the hardware. **"Two 7B models collapse
at 64 req/s" is wrong; "collapse at 64 req/s with `max_num_seqs=32`" is right.**
State the config whenever the absolute is quoted.

#### The untouched surface

Ordered by how much each would move the findings above.

| Knob | Backend | What it would change |
|---|---|---|
| **`--max-num-seqs`** | vLLM | The admission limit. Sets the cliff location and caps cache occupancy — the single highest-value sweep |
| **`--max-num-batched-tokens`** | vLLM | Chunked-prefill budget per step. Directly governs how much a VLM prefill burst can starve a co-tenant — the mechanism `cross-vlm-prefill-vs-llm` exists to measure |
| **`--kv-cache-dtype=fp8`** | vLLM / SGLang | Halves KV footprint. Changes every cache-pressure conclusion, and is free capacity if accuracy holds |
| **prefix caching on/off** | vLLM `--no-enable-prefix-caching`, SGLang `--disable-radix-cache` | With 2–3 prompts it made prefill free and the cache unreachable. Required for any experiment where the cache must be the constraint |
| **`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`** | MPS, not a backend | **Apportions SM share between tenants.** The study established `gpu_memory_utilization` cannot divide memory; this is the knob that can divide *compute*, and it was left at default for every run |
| **`--enforce-eager` vs CUDA graphs** | vLLM | Launch overhead vs capture memory. A `variant` already exists and is unused in contention |
| **`--enable-chunked-prefill`** | vLLM | Whether prefill is sliced or monolithic — changes the shape of the burst, not just its size |
| **Attention backend** | vLLM `VLLM_ATTENTION_BACKEND`, SGLang `--attention-backend` | FlashAttention / FlashInfer / Triton differ in memory and kernel behaviour under concurrency |
| **`--scheduling-policy`** | vLLM (`fcfs`/`priority`), SGLang (`--schedule-policy lpm/fcfs/dfs-weight`) | Who waits when the queue forms. Directly relevant to the queueing cliff |
| **`--max-running-requests`, `--chunked-prefill-size`, `--schedule-conservativeness`** | SGLang | SGLang's equivalents of the above. **None are set; `variants` is empty**, so the vLLM-vs-SGLang comparison is stock-vs-stock |
| **`--block-size`, `--swap-space`, `--num-scheduler-steps`** | vLLM | KV granularity, CPU offload, multi-step scheduling |
| **Speculative decoding** | vLLM `--speculative-config` | Trades compute for latency — changes the compute/bandwidth balance a tenant presents to its neighbour |
| **`dynamic_batching`, `instance_group`** | Triton | Queue delay, preferred batch size, replicas per GPU. The CV tenants are the fragile ones; batching is exactly what would protect them |
| **TensorRT precision / workspace** | Triton | fp16 vs int8 for the CV models |
| **`--tensor-parallel-size`** | vLLM / SGLang | Untested here — `nvidia-smi topo -m` reports PIX, so PCIe would dominate |

#### If only one thing gets swept

`--max-num-seqs`, on `same-llm`. It is the admission limit behind the study's
most striking result, it is a single integer, and the current value was never a
decision — it arrived as a backend-wide default and silently became a finding.

`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` is the close second, for a different reason:
it is the only knob measured or otherwise that can *apportion compute* between
co-tenants, which is the resource this study found to be the binding constraint.

### The configured rates are far below capacity

| Tenant | Configured | Measured solo | GPU utilisation there |
|---|---|---|---|
| `qwen2.5-7b` | 4 req/s | sustains **62 of 64** | 99% "util", but see below |
| `yolov8-l` | 50 req/s | **201 of 200**, still climbing | 33% at 200 req/s, 8% at 50 |
| `dinov2-base` | 50 req/s | 50 of 50 | 7% |
| `kosmos-2.5` | 0.1 req/s | ~0.133 ceiling | — |

Only `kosmos-2.5` is anywhere near its limit. The LLM runs at about 1/16th of
capacity and the CV tenants at under a tenth.

**Do not read `gpu_util_pct` as saturation.** It is the fraction of time the
engine is *active*, not occupancy: `qwen2.5-7b` shows 99% at 4 req/s and still
absorbs 16× more load at the same memory bandwidth, because continuous batching
reads the weights once per decode step rather than once per request. Memory
bandwidth and achieved-vs-offered are the honest signals.

### `gemma-4-31b-it-fp8` cannot be served — half fixed, and the other half is silently inert

Its first-ever launch attempt failed:

```
The checkpoint you are trying to load has model type `gemma4` but Transformers
does not recognize this architecture.
```

`transformers` 4.57.6 in `.venv-vllm` has no `gemma4`. Do not upgrade it under a
running study — vLLM pins the version, and every other tenant in the roster
currently works against it.

**Done:** no colocation reaches this model any more. `same-vlm` now pairs
`qwen2.5-vl-7b` with `qwen3-vl-32b-fp8`, which is this roster's other
video-capable 32B and serves today, so the same-category VLM pair has a
measurement again rather than 8 failed runs.

**Not done, and it looks done.** The intended pin

```yaml
unsupported_backends:
  vllm: "transformers 4.57.6 has no gemma4 architecture; needs a version bump"
```

*is* written into the model's block — but the block declares
`unsupported_backends:` **twice**, and the second (trtllm-only) mapping silently
replaces the first when the YAML is parsed. The effective value names `trtllm`
alone, so **vLLM is not pinned out**. Nothing in the contention study hits it,
but the single-model sweeps still do: `models`, `video`, `video-4f`, `video-8f`,
`video-16f` and `full` all carry a `gemma-4-31b-it-fp8` round on vLLM, and each
will fail the same way. Merging the two mappings into one is the fix.

**The transferable part:** a duplicate mapping key in YAML does not raise — it
overwrites. A pin you can read in the file is not a pin the loader applied.

### Sweeps that never reach a knee

| Colocation | Phase | Sweep | Top rung reaches | Suggested |
|---|---|---|---|---|
| `same-cv` | 2 | `[1, 10, 50, 200]` | 33% utilisation | `[50, 200, 600, 1500]` |
| `cross-llm-vs-cv-rps` | 4 | `[1, 10, 50, 200]` | 33% | same |
| `cross-ilm-vs-cv` | 4 | `[1, 10, 50, 200]` | 33% | same |
| `secondary-asymmetry-a/b` | 6 | `[4, 16, 64]` | ~10% | scale to the CV ceiling |
| `same-llm` | 2 | `[1, 4, 16, 64]` | past the cliff | add a 32 rung |

`same-llm` is the one sweep that *does* cross its knee, and it shows why the
rungs matter: flat at 1.6–1.9× from 1 to 16 req/s, then 33–37× with 23%
throughput loss at 64. The knee is somewhere between, and `[1, 4, 16, 64]` is
too coarse to place it.

**Check the driver before raising a CV rate past ~600 req/s.** `perf_analyzer`
runs in a container and may become the bottleneck itself, which would measure
the client rather than the GPU. Run the tenant solo at the intended top rate and
confirm `achieved_rps` still tracks `offered_rps`.

### The cache was never the constraint, so the splits measured nothing they claimed

`cross-deploy-split-*` returned 0.88 and 2.48 req/s at **every** split, across a
3.4x range of 72B cache. Read alone that says "how you divide memory does not
matter", and within the range tested it is true — but the reason is not that the
tenants had enough cache. It is that the workload cannot *use* cache:

```
Prefix cache hit rate: median 97.4%
Running: 32 reqs                  <- at the --max-num-seqs cap
GPU KV cache usage: 8.7%          <- of a 150,720-token cache
```

`llm_long` contains **two distinct prompts** and `llm_short` three, so with
prefix caching on, 32 concurrent requests that *should* occupy 48,032 tokens
occupy ~13,100. Only generated tokens cost unique cache, and the working set is
~4.8 GiB whatever you allocate. Three successive generations of this experiment
produced flat curves for that one reason, not because their rungs were sized
badly.

Two changes, in order of cost:

1. **Scope a `prefix_off` variant** (`--no-enable-prefix-caching` for vLLM,
   `--disable-radix-cache` for SGLang) onto this family and re-run it, ~1.3 h at
   `repetitions: 1`. Ready to apply — see
   [next-run/config-changes.md](next-run/config-changes.md) §1e.
2. **Add prompt diversity, 20–50 per workload.** A 97% prefix hit rate is
   nothing like production traffic, so *every* prefill-sensitive number in this
   study — TTFT above all — is measured against an unrealistically cheap prompt.
   This reaches far beyond one family.

Two consequences to carry:

- **Treat the 58% / 65% deployment cost as an optimistic bound.** It was
  measured with prefill essentially free.
- **Verify `GPU KV cache usage` in the server log before trusting any
  cache-pressure result.** Single-digit percentages mean the rungs measured
  nothing, however clean the numbers look.

**Also missing: a fullness sweep.** All four splits divide the same 28 GiB and
sit at ~95% of the card, so the family varies *how* memory is divided and never
*how much* is left. The cost of a second tenant on a card that is 60% full is
not measured.

### `cross-size-scaling` varies size and load fraction together

All four rungs are driven at 4 req/s, and all four serve it:

| Tenant | Cap | Quantization | Achieved of 4 req/s |
|---|---|---|---|
| `qwen2.5-7b` | 0.35 | bf16 | 3.91 |
| `qwen2.5-14b` | 0.49 | bf16 | 3.90 |
| `qwen2.5-32b` | 0.87 | bf16 | 3.88 |
| `qwen2.5-72b` | 0.62 | awq | 3.90 |

(Caps as derived today, from a 16 GB budget and the corrected `weights_gb`; the
72B rung read 0.66 when it ran, against the old 45 GB estimate.)

The cap is not monotonic in parameter count — 72B is AWQ 4-bit (~40 GB) and
32B is bf16 (~65 GB), so the 32B rung is the tightest tenant in the study.

The confound: 4 req/s is about **6% of `qwen2.5-7b`'s measured 62 req/s
capacity**, but a much larger fraction of 72B's. The sweep therefore changes
model size and proximity-to-saturation at the same time. If the larger rungs
degrade more under contention, this design cannot say whether that is because
they are larger or because they were already closer to their limit.

To separate them, drive each rung at a fixed *fraction* of its own solo
capacity (measure that first, per rung) rather than at a fixed absolute rate.
Keep one absolute-rate arm if the customer-facing question is "what happens at
4 req/s", but do not read the current arm as isolating size.

### Do not read the cap sum as memory pressure

Once a tenant has a `kv_budget_gb`, its cache is pinned by an absolute
`--kv-cache-memory-bytes` and the derived cap is only a **ceiling**. What
actually occupies the card is `weights + pinned KV + overhead`, and that is
unaffected by the cap.

`mix-memory-bound` is the case to remember:

| | Before the weights fix | After |
|---|---|---|
| Cap sum ("reserved") | 0.94 — 90 GB | 0.91 — 87 GB |
| Weights + pinned KV ("occupied") | **~82 GiB** | **~82 GiB** |

Nothing about the run changed. `qwen2.5-72b` always took 38.77 GiB and its
cache was always pinned at 6.0 GiB; only the *estimate* of the weights moved,
and it moved toward the truth. The old 0.94 was reserving 3.4 GB that nothing
ever filled.

The trap: "the memory-bound baseline dropped from 0.94 to 0.91, restore it"
sounds conservative, but the only way to raise the cap is to raise
`kv_budget_gb` — which adds **real** cache. That would be the first run at that
cache size, breaking comparability with every prior result, and more KV means
more batching room and fewer evictions, so it would be *less* constrained while
looking more committed.

Two consequences worth carrying:

- **This baseline is less memory-bound than its name suggests** — 82 of 96 GiB,
  not 90. A genuinely near-ceiling baseline needs bigger models or deliberately
  more KV, and that is a new design decision, not a restoration.
- **Correcting a `weights_gb` re-derives every cap that depends on it.** The
  72B fix changed 27 resolved windows across 16 colocations, 11 of which were
  not the target, so `--resume` will re-run them. The KV budgets — the quantity
  these experiments hold constant — did not move, so the results stay
  comparable in substance.

### The `weights_gb` estimates are verified — from the server logs

The ground truth is on disk already: every vLLM tenant logs

```
Model loading took 38.77 GiB memory and 14.913061 seconds
Available KV cache memory: 6.69 GiB
```

into `<tenant>.server.log`, which the orchestrator captures — so any declared
value can be checked against runs already on disk.

Converting the yaml's `weights_gb` (GB) to the GiB vLLM reports, seven of eight
declared values are correct to within 0.10 GiB:

| Model | Declared (GiB) | Measured (GiB) | Delta |
|---|---|---|---|
| `llama3.1-8b` | 14.99 | 14.99 | −0.00 |
| `mistral-7b` | 13.50 | 13.51 | +0.01 |
| `qwen2.5-32b` | 61.00 | 61.04 | +0.04 |
| `qwen2.5-vl-7b` | 6.52 | 6.59 | +0.07 |
| `gemma2-9b` | 17.14 | 17.22 | +0.08 |
| `qwen2.5-7b` | 14.16 | 14.25 | +0.09 |
| `qwen2.5-14b` | 27.47 | 27.57 | +0.10 |
| **`qwen2.5-72b`** | **41.91** | **38.77** | **−3.14** |

Only `qwen2.5-72b` (AWQ) was materially wrong. It has since been corrected to
`weights_gb: 41.6` (= 38.74 GiB), which is within 0.03 GiB of the measurement;
every cap derived from it moved down accordingly, and the KV budgets — the
quantity the experiments hold constant — did not move.

**Note the unit.** The field is `weights_gb` and the values are GB, but vLLM
reports GiB. Comparing them directly makes every estimate look ~7% high. Either
rename the field or record the conversion beside it.

**Caveat, and it costs something.** Once `--kv-cache-memory-bytes` is set, vLLM
stops logging `Available KV cache memory` — so the cache ground truth is no
longer recoverable from the log for exactly the tenants whose cache matters
most. `GPU KV cache usage` in the steady-state log lines is what remains.

### RESOLVED — `gpu_memory_utilization` cannot apportion memory, and every colocation that needed a budget now has one

`cross-vlm-prefill-vs-llm` and `place-vlm-prefill-split` used to set no
`kv_budget_gb`, so no `--kv-cache-memory-bytes` was emitted and the tenants fell
back to `--gpu-memory-utilization`. Both now carry `kv_budget_gb: 20.0` and
derive their fractions from it. The measurement behind that change is worth
keeping, because it generalises well past those two entries.

**The failure.** `cross-vlm-prefill-vs-llm` died in `llm.server.log`:

```
ValueError: No available memory for the cache blocks.
Try increasing `gpu_memory_utilization` when initializing the engine.
```

The tenant was `gemma2-9b`, which serves without trouble in `same-llm`, so this
was the colocation's missing KV reservation and not the model. `mix-full` failed
the same way. The starkest case was a 7B behind a co-resident 72B:

```
Model loading took 14.25 GiB memory
Available KV cache memory: -37.31 GiB
```

Its allowance was 0.19 × 96 ≈ 17.0 GiB against 14.25 GiB of weights, which on an
*empty* card leaves the ~1 GiB intended. The ~38.3 GiB shortfall is precisely the
neighbour's resident weights.

**Confirmed quantitatively, so it is the mechanism and not a coincidence.**
Raising the fraction by 0.03 (0.19 → 0.22) moved the deficit by 0.03 × 96 GB:
predicted −34.63 GiB, measured **−34.46 GiB**, within 0.17 GiB.

**Caps are not impossible, they are unmaintainable.** A fraction high enough to
cover the neighbour's footprint *plus* its own loads fine — total-device
fractions need not sum to 1.0. The reason to reject that is not infeasibility:
the second tenant's fraction then encodes the *first* tenant's footprint and the
load order, so the number means nothing on its own and breaks silently whenever
the neighbour's size, quantization or cache changes. Swapping a 72B for a 32B
would mean recomputing a number that never mentions either.

`--kv-cache-memory-bytes` is absolute and per-process, so it composes across
tenants and states the intended quantity directly instead of inferring it from a
fraction.

**The distinction to carry forward:** `gpu_memory_utilization` is unusable for
apportioning memory between colocated tenants, whatever the intent. Where the
cache is meant to be constant across a comparison, state one colocation-wide
budget and derive the fractions; where the *split* is the variable, state a
budget per tenant, as `cross-deploy-split-*` does. Either way the knob is
`kv_budget_gb`, never the fraction.

`place-vlm-prefill-split` never showed a symptom, because its tenants sit on
separate cards and neither sees the other's memory — the same missing config,
invisible until they shared a GPU. **A multi-GPU window is not a test of the
memory plan.**

### `duration_s` propagates through `extends:`, and sweeps multiply it

`cross-ilm-vs-cv` inherits `duration_s: 600` from `mix-ilm-cv`. That window was
chosen for kosmos at 0.1 req/s (60 requests); applied to the same colocation's
200 req/s CV rung it asks for **120,000 requests**, five times what the CV solo
baseline sent. Both drivers produced no output at that rung and the run was lost.

A window length has to suit the *highest* rung of any sweep that inherits it,
not just the tenant it was chosen for. Either set `duration_s` explicitly on
colocations that sweep, or scale the window per rung.

### The driver timeout shares one deadline across tenants

`ColocationOrchestrator.run` computes one deadline for all drivers, so a hung
driver can consume the budget a healthy sibling was still waiting on. Give each
driver its own timeout. (Not established as the cause of the `cross-ilm-vs-cv`
loss above — both drivers produced nothing there — but it is a real weakness.)

### One repetition is not enough at low utilisation

Phase 0 measured run-to-run variance at 1.8% and set `recommended_reps: 1`. That
was on a *loaded* probe. At the utilisations these colocations actually run at,
the ratios are dominated by variance rather than contention:

`same-cv`'s `dinov2-base` reads **1.55× at 10 req/s and 1.05× at 50 req/s** —
non-monotonic, so at least one of those points is noise. At 1–8% GPU
utilisation a single repetition cannot separate 1.55× from 1.05×.

Two ways out, and the first is better: raise the rates until the signal exceeds
the variance (see the sweep table above), or raise `repetitions` on the low
rungs. Only the four `cross-deploy-split-*` colocations currently set
`repetitions: 3`, and they do so for a different reason — expected bimodality
near the memory ceiling.

**Measured, and it cuts both ways.** Away from a cliff, repetitions buy nothing:
12 `cross-deploy` contention runs returned identical values to two decimals.
Near the eviction limit, two identical repetitions gave **0.79 and 0.40** — there
the spread *is* the finding, and a mean of the two describes neither state.
`repetitions: 3` is only justified next to a cliff; everywhere else, spend the
time on rungs instead.

### A measurement artifact to know about

`perf_analyzer`-driven tenants report `achieved_rps` about **17–20% below
offered**, independent of load or model: `yolov8-l`, `dinov2-base` and
`kosmos-2.5` all read exactly 83% of offered at their lowest rung. It is the
`--request-count` accounting — N requests over an elapsed window that runs
longer than the configured one.

This **cancels in degradation ratios**, since baseline and contention carry the
same offset, so every throughput-retention number stands. It does bias the
absolute "% of offered" reading, and it is why a Triton tenant at 83% is normal
rather than a warning. aiperf-driven vLLM tenants do not show it — they read
96–98% at every rate.
