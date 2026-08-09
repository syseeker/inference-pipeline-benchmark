# GPU contention benchmarking — the concept, and the architecture

For anyone asking "what is this study actually measuring, and how is it laid out on the hardware?".

This is the conceptual document. The methodology decisions and their reasoning
live in
[skills/gpu-contention-benchmark/reference/design-decisions.md](../skills/gpu-contention-benchmark/reference/design-decisions.md);
the model list lives in
[reference/model-catalogue.md](../skills/gpu-contention-benchmark/reference/model-catalogue.md).
Read this one first.

## Contents

- [1. The question](#1-the-question)
- [2. The baseline is the hard part](#2-the-baseline-is-the-hard-part)
  - [2a. Same request rate](#2a-same-request-rate)
  - [2b. Same KV cache — the memory allocation trap](#2b-same-kv-cache--the-memory-allocation-trap)
  - [2c. Same clocks](#2c-same-clocks)
- [3. Why the four model types contend differently](#3-why-the-four-model-types-contend-differently)
  - [The consequence: contention is not symmetric](#the-consequence-contention-is-not-symmetric)
  - [Measured, 2026-08-04 — the rule the data supports](#measured-2026-08-04--the-rule-the-data-supports)
- [4. Architecture — one GPU](#4-architecture--one-gpu)
  - [The memory budget](#the-memory-budget)
- [5. Architecture — more than one GPU](#5-architecture--more-than-one-gpu)
  - [Mode A — separate GPUs (isolation)](#mode-a--separate-gpus-isolation)
  - [Mode B — tensor parallel](#mode-b--tensor-parallel)
  - [Mode C — packing](#mode-c--packing)
  - [What changes in the measurement](#what-changes-in-the-measurement)
- [6. How to read the output](#6-how-to-read-the-output)
  - [Sanity checks before believing any of it](#sanity-checks-before-believing-any-of-it)
- [7. Where this sits](#7-where-this-sits)
  - [The mix-up, with numbers](#the-mix-up-with-numbers)

---

## 1. The question

You have one GPU and four models you'd like to run on it: a text LLM, a
video-understanding VLM, a document/image model, and an object detector.
They fit in memory. Can you run them together?

"They fit" is a memory question, and it's the easy one. The hard question is
what happens to *speed*. Two models on one GPU are not two models on two
half-GPUs — they interleave, they queue behind each other, they evict each
other's caches. One of them may be almost unaffected while the other falls
off a cliff.

**A contention benchmark measures exactly that: how much slower each model
gets because of its neighbours.**

The unit of measurement is a ratio:

```
                  latency of model A when sharing the GPU
degradation  =    ───────────────────────────────────────
                  latency of model A when running alone
```

A ratio of 1.0 means the neighbour cost you nothing. 1.3 means 30% slower.
3.0 means you probably can't ship it.

Everything else in this study — the phases, the config schema, the two load
generators, the orchestrator — exists to make that one fraction trustworthy.

---



## 2. The baseline is the hard part

The denominator is where contention studies go wrong, so it's worth being
blunt about the rule:

> **Between the baseline run and the contention run, the neighbour must be
> the only thing that changed.**

If anything else differs — request rate, memory allocation, clock speed,
prompt length — then that difference lands in the ratio and gets reported as
contention. The number comes out looking authoritative and is simply wrong.

Three things break this in practice, and all three are easy to get wrong.

### 2a. Same request rate

Drive both runs at the same offered rate, and drive them **open-loop** — a
fixed requests-per-second, not a fixed number of requests in flight.

The difference is *who decides when the next request is sent*.

**Closed-loop — "keep N requests in flight."** The client sends a new request
only when an old one comes back, so the server's speed controls the client:

```
concurrency = 2, fast GPU (100 ms/req)

client  │ A ────────►│ C ────────►│ E ────────►│
        │ B ────────►│ D ────────►│ F ────────►│
        └────────────┴────────────┴────────────┘
        0           100          200          300 ms      6 requests sent

a neighbour arrives; the GPU is now 2x slower (200 ms/req)

client  │ A ──────────────────►│ C ──────────────────►│
        │ B ──────────────────►│ D ──────────────────►│
        └──────────────────────┴──────────────────────┘
        0                     200                    400 ms   4 requests sent
```

The client slowed down too. It offered *less work* because the GPU got
slower — not by anyone's choice, but structurally: with only 2 requests
allowed in flight, a slower server necessarily receives fewer of them.

**Open-loop — "send 20 requests per second."** The client sends on a clock and
never waits, so nothing the server does changes the arrival rate:

```
20 req/s = one every 50 ms, fast GPU (100 ms/req)

arrivals  ↓    ↓    ↓    ↓    ↓    ↓        every 50 ms
queue     ░░   ░░   ░░   ░░   ░░   ░░       stays shallow
          └──────────────────────────────
          0                          300 ms       6 requests sent

a neighbour arrives; the GPU is now 2x slower (200 ms/req)

arrivals  ↓    ↓    ↓    ↓    ↓    ↓        every 50 ms — UNCHANGED
queue     ░░   ▒▒▒  ▓▓▓▓ ████ █████ ██████  grows without bound
          └──────────────────────────────
          0                          300 ms       6 requests sent
```

The client offered exactly the same work, so the damage has nowhere to hide.
It shows up as a growing backlog.

**Why this decides whether the ratio means anything.** The rule is that the
neighbour must be the only thing that changed:

| | solo run | contention run | load held equal? |
|---|---|---|---|
| Closed-loop | client sends 20/s | client sends 10/s | ❌ no |
| Open-loop | client sends 20/s | client sends 20/s | ✅ yes |

Under closed-loop *two* things changed — the neighbour appeared **and** the
offered load halved — so the result cannot be attributed to either. The client
quietly protects the GPU from the overload you were trying to create.

**What that would have cost in practice.** In the measured `same-llm` cliff,
TTFT p95 degraded **600×** while inter-token latency stayed under 2×: a queue
forming in front of the server. A closed-loop client *cannot produce that
number*, because it never lets a queue form — only N requests exist at once, so
there is nothing to queue behind. You would have seen a gentle latency rise and
concluded the pairing was fine, right up until it wasn't.

Open-loop is also the honest model. A game engine sends a decision request
every 16 ms whether or not the last one came back; a camera produces 30 frames
a second regardless. Nothing in production waits for your GPU.

Full reasoning in design-decisions §1, *Open-loop load, not concurrency*.

### 2b. Same KV cache — the memory allocation trap

This one is the least obvious, and it's the reason this section exists.

The **KV cache** is the pool that holds attention state for in-flight requests.
A bigger cache means more requests can be in flight simultaneously, which means
higher throughput and less queueing. So the cache size is not a passive memory
limit — it is a performance setting, and if it differs between the two runs, the
ratio is partly measuring *it* rather than the neighbour.

Watch what happens if the solo run gets the card to itself while the contention
run gets a production share of it. A 7B model, 15 GB of weights, on a 96 GB
card:


|                                    | **KV cache** | requests it can keep in flight |
| ---------------------------------- | ------------ | ------------------------------ |
| Solo baseline, whole card to itself | **~71 GB**   | many                           |
| Contention run, sharing with CV     | **~28 GB**   | ~2.5× fewer                    |


The contention run is now slower for **two** reasons, thoroughly mixed:

1. The neighbour is stealing SM time and memory bandwidth ← *what we want*
2. Its own KV cache is 2.5× smaller ← *a self-inflicted artifact*

There is no way to separate them afterwards. You would report "the detector
slowed my LLM by 2.4×" when much of that was "I gave my LLM a third of the
memory."

**The fix:** give the solo baseline the *same cache size* the contention run
gets — 28 GB in both. The neighbour is then the only difference left.

**How the cache is pinned.** The knob is `kv_budget_gb`, which is per tenant and
states the cache in gigabytes. The orchestrator turns it into an *absolute* size
the backend honours directly — `--kv-cache-memory-bytes` for vLLM,
`--max-total-tokens` for SGLang. It is not a fraction of anything, so it means
the same number on an empty card and on a crowded one. The memory *fraction*
(`gpu_memory_utilization`) does not set the cache and cannot apportion one;
that mechanism, and the measurement that settled it, is section 4.

Because `kv_budget_gb` is part of a solo baseline's identity, the baseline
inherits the contention run's cache automatically rather than anyone having to
remember to copy it.

The part worth saying out loud to anyone reading the results:

> **The solo baseline is deliberately handicapped.** It does not answer "how
> fast can this model go on this GPU." It answers "how fast is this model at
> its production memory allocation, with nobody else on the card."

Those are different questions. The first one is a single-model benchmark — a
different study, run with the
[benchmark-gpu-inference](../skills/benchmark-gpu-inference/SKILL.md) skill.
Mixing the two up produces contention ratios that are inflated across the
board.

### 2c. Same clocks

Under co-residency, power draw rises. The GPU hits its power cap, drops its
SM clocks, and everything slows down. That's real, but it is **not
contention** — it's thermodynamics, and it would happen to a single model
under equivalent load.

So: pin the power limit and lock clocks at 60–80% of max boost, record the
achieved clocks with every result, and throw the run away if a throttle
reason fired. Detail in design-decisions §2, *Clock throttling is not contention*.

---



## 3. Why the four model types contend differently

A GPU isn't one resource, it's several — and each workload leans on a
different mix. This is the whole reason a *matrix* of results is needed
rather than one number.


| Resource             | What exhausts it                                    |
| -------------------- | --------------------------------------------------- |
| SM compute (FLOPs)   | Big matmuls — prefill, vision encoders              |
| Memory bandwidth     | Streaming weights from VRAM — token-by-token decode |
| VRAM capacity        | Weights + KV cache                                  |
| NVDEC (video decode) | Separate silicon — video tenants only               |


Now the four categories:

**Text LLM — bandwidth-bound, steady.**
Generating each token requires streaming the entire weight matrix out of
VRAM. That's very low arithmetic intensity: the SMs are mostly idle, waiting
on memory. An LLM in its decode phase is a *bandwidth* hog, not a compute
hog. Its KV cache also grows with context length, so it's the tenant most
sensitive to having its memory cap cut.

**VLM on video — the worst neighbour in the set.**
A 40-frame clip means: decode the video (NVDEC), run a vision encoder over
all 40 frames (a large compute burst), then prefill with several thousand
tokens because each frame expands into hundreds. It is a periodic *compute
tsunami* — quiet, then enormous, then quiet. Anything sharing the card sees a
latency spike shaped exactly like that burst. This is why `vlm_video_long`
exists as a workload.

**ILM / document models — heavy but steadier.**
High-resolution single images through a vision encoder. Less bursty than
video, considerably heavier than text. Two of these (`kosmos-2.5`,
`paddleocr`) have no vLLM implementation at all and run on Triton's Python
backend — which means they're also the slowest-served tenants in the study,
for reasons that have nothing to do with contention.

**CV detection / embedding — small, fast, high-rate, fragile.**
YOLOv8 or DINOv2 at 50–200 requests per second, each one a few milliseconds.
A stream of tiny kernels. And here's the asymmetry that matters: **a 5 ms
model pushed to 15 ms is a 3× regression, while a 500 ms LLM pushed to 600 ms
is barely noticeable.** Same absolute interference, wildly different
consequence.

### The consequence: contention is not symmetric

Pairing matters, and it matters in both directions:

- A bandwidth-bound LLM next to a compute-bound CV model can overlap
surprisingly well — they're competing for different things.
- Two bandwidth-bound models fight badly.
- A small fast model next to a bursty one gets wrecked, while the bursty one
hardly notices.

So "model A and model B contend" is not one number — it's two, and they're
often very different. That's why `summary.py` reports a **victim × aggressor
matrix** rather than a single score per pair.

### Measured, 2026-08-04 — the rule the data supports

The reasoning above says to pair tenants that stress *different* resources.
Phase 5 tested that on 2× RTX PRO 6000 by splitting `mix-full`'s four tenants
across two cards three ways. Worst-tenant end-to-end p95, against each tenant's
own solo baseline on the same card:

| Placement | GPU 0 | GPU 1 | worst p95 | mean p95 |
|---|---|---|---|---|
| all four on one card | llm, vlm, ilm, cv | — | 2.88× | 2.20× |
| P3 | llm + cv | vlm + ilm | 2.19× | 1.35× |
| P1 | llm + vlm | ilm + cv | 1.95× | 1.38× |
| **P2** | **llm + ilm** | **vlm + cv** | **1.46×** | **1.23×** |

**P2 wins on both worst-tenant and mean**, and every split beats one card. The
gap between best and worst placement is 1.5×, so which pair you separate matters
about as much as adding the second card at all.

Two rules, each with a clean contrast behind it:

**1. Never co-locate the two vLLM tenants.** In P1 the LLM and VLM share GPU 0
and the VLM pays **1.95×**. In P2 and P3, where they are split, the VLM sits at
**1.02×** and **1.12×**. Both are autoregressive and KV-hungry: they contend for
precisely the resource the other needs, and no amount of "different resource"
reasoning applies because they are the *same* resource profile.

**2. Given that, pair the CV tenant with the VLM rather than the LLM.** P2 puts
CV with the VLM → **1.46×**. P3 puts CV with the LLM → **2.19×**. The LLM's
steady stream of small decode kernels leaves no gaps; the VLM's bursty prefill
does, and a small fast tenant can use them. This is the "small fast model next
to a bursty one gets wrecked" intuition above coming out *backwards* — against a
steady neighbour it fares worse than against a bursty one.

That also explains P3's shape: best mean (1.35×), second-worst tail (2.19×). It
fixes the vLLM pairing and then concentrates all the remaining damage on one
victim.

Two null tests bound the result. Two tenants on separate cards with nothing
shared returned **1.02×** (`place-isolated`), and a 40-frame video prefill burst
on one card against an LLM decoding on the other returned **1.00×**
(`place-vlm-prefill-split`) — so on a PCIe box the interconnect is not a hidden
coupling channel, and the harness does not manufacture degradation.

**Caveat on generality — the load point is low.** Phase 2's `same-llm` sweep
measured what one of these tenants can actually do:

| offered | achieved | memory bandwidth | power |
|---|---|---|---|
| 1 rps | 0.96 | 0% | 158 W |
| 4 rps | 3.91 | 77% | 272 W |
| 16 rps | 15.50 | 80% | 315 W |
| 64 rps | 62.38 | 77% | 344 W |

`qwen2.5-7b` sustains **62 of 64 requests/second alone**, with memory bandwidth
flat from 4 rps upward — continuous batching reads the weights once per decode
step regardless of how many requests share it. The study's configured `llm@4`
is therefore about **1/16th of this tenant's capacity**, not a saturating load.
(Read `gpu_util_pct` carefully: it sat at 99% from 4 rps on, but it measures the
fraction of time the engine is *active*, not occupancy. It saturates long before
the card does.)

So the placement ranking above, and Phase 3's pairing costs, are measurements of
a **lightly loaded** GPU. That is a legitimate operating regime — and the
`same-llm` sweep shows it is the *safe* one — but it is not the whole picture:

| both LLMs at | e2e p50 ratio | throughput kept |
|---|---|---|
| 1 rps | 1.51× / 1.57× | 1.00× |
| 4 rps | 1.67× / 1.84× | 1.00× |
| 16 rps | 1.72× / 1.89× | 1.00× |
| **64 rps** | **33.3× / 37.2×** | **0.77×** |

Flat to 16 rps, then a cliff. Co-locating two LLMs costs a near-constant
1.6–1.9× inside the safe region and falls apart past it — 20× more latency and
the first throughput loss in the study. Two tenants at 64 rps each are asking
128 rps of a card that delivers ~62 for one, so the collapse is capacity, not
mystery; the useful part is that the tax is flat until it isn't. The knee sits
between 16 and 64 rps per tenant and this sweep is too coarse to place it.

---



## 4. Architecture — one GPU

Using `mix-llm-cv` as the worked example — a text LLM beside an object
detector, which is the simplest real colocation:

```
                     ┌──────────── one timed window ────────────┐
                     │                                          │
  aiperf ────────────┼──► vLLM :8000   (LLM, cap 0.45)          │
  (open-loop, rps)   │                                          │
                     │                                          │
  perf_analyzer ─────┼──► Triton :8001 (CV, TensorRT)           │
  (open-loop, rps)   │                                          │
                     │                                          │
                     └──────────────────────────────────────────┘
                                        │
                     one dcgmi sampler ─┘  (whole window, not per tenant)

              shared t0  ── every request stamped in epoch ms ──►  aligned traces
```

Four design points, each with a reason:

**Native servers for LLM/VLM, Triton for CV.**
vLLM, SGLang and TRT-LLM each run their own process with their own scheduler
— putting them behind Triton adds a hop and buys nothing. CV models genuinely
benefit from Triton (TensorRT backend, dynamic batching, and a Python backend
for the two models with no export path). Detail in
[reference/serving-topology.md](../skills/gpu-contention-benchmark/reference/serving-topology.md).

**Two load generators, one per tenant type.**
AIPerf cannot drive Triton — it dropped GenAI-Perf's `kserve` endpoint types.
So LLM/VLM tenants get `aiperf --request-rate`, CV tenants get
`perf_analyzer --request-rate-range`. Both support Poisson arrivals, so the
two remain comparable. The orchestrator's job is to start them against a
shared `t0` and merge their traces — never to generate load itself.

**MPS on, always.** Without MPS, kernels from different processes
time-slice instead of overlapping. Measured on the PRO 6000 during Phase 0:
MPS off gave **0.28×** aggregate throughput (worse than serial — the tenants
were fighting over context switches), MPS on gave **1.94×**. Without MPS you
aren't measuring contention, you're measuring the time-slice scheduler.

**Exactly one GPU sampler for the whole window.** DCGM is device-scoped and
has no per-process view, so N samplers would mean N subprocesses all
reporting the *whole card's* memory as each tenant's own. Whole-GPU numbers
attach to the colocation; per-tenant VRAM comes from
`nvidia-smi --query-compute-apps`.

### The memory budget

There are two knobs here and they do different jobs. Conflating them is the
mistake this study made, hit, and then measured.

**`gpu_memory_utilization` is a ceiling, not a share.** It is a target for
*total device* utilisation: vLLM sizes its cache from
`total × fraction − whatever every process on the card already holds`. So a
tenant whose cap is lower than its neighbour's resident footprint derives a
*negative* cache and dies before serving anything:

```
Model loading took 14.25 GiB memory
Available KV cache memory: -37.31 GiB
ValueError: No available memory for the cache blocks.
```

That is exactly how `mix-full`'s VLM failed behind a 37 GB LLM while the
pre-flight called the plan fine. It is a mechanism, not a fluke: across rungs
the deficit tracked the cap to within 0.17 GiB of prediction. **A cap cannot
express "my share" of a shared card** — SGLang's `--mem-fraction-static` has the
same property and fails the same way.

The pre-flight still checks a sum, per GPU:

```
sum(tenant gpu_memory_utilization on one card) + CV footprint  ≤  1.0
```

because vLLM's default of 0.90 is essentially the whole card and would starve
everyone else. But that is a *budget* check on what tenants may claim — passing
it does not mean the tenants will co-reside.

**`kv_budget_gb` is what sets the cache**, and it is stated **per tenant**. The
orchestrator turns it into an absolute size, which composes across processes
because it never refers to the device total:

| Backend | Flag emitted |
|---|---|
| vLLM | `--kv-cache-memory-bytes = kv_budget_gb × 1024³` |
| SGLang | `--max-total-tokens = kv_budget_gb × 1024³ ÷ kv_bytes_per_token`, **plus** a permissive `--mem-fraction-static=0.95` |

SGLang needs both: the fraction still gates the overall allocation, so once the
token count is doing the real work the fraction has to stop being a control or
the tenant dies of a limit it is no longer using.

A colocation normally states **one budget that every tenant inherits**, and that
is what makes a comparison valid — the cache is held constant while only the
weights move (section 2b). A tenant may override it where the *split itself* is
the variable: `cross-deploy-split-*` divides one fixed 28 GiB leftover four ways,
and there the per-tenant budgets are the experiment.

The cap is then **derived from the budget, not hand-written**:

```
cap = (weights_gb + kv_budget_gb + 2 GB overhead) / total VRAM,  rounded to 2 dp
```

so it sits above what the tenant will actually occupy. The cap comes out
*different* per model — heavier weights need a bigger ceiling — while the cache,
the thing that actually sets speed, stays *equal*. Sizing caps *proportional to
model size* is the obvious move and it would be wrong: KV need follows request
rate and context length, not parameter count.

You can still write an explicit cap when you mean to, and it wins over
derivation. `mix-llm-cv` (0.45) and `mix-vlm-cv` (0.50) do, because each has a
single vLLM tenant and nothing to hold constant against.

Two consequences worth carrying out of this section:

- **A cap sum is not memory pressure.** Once the cache is pinned, occupancy is
  `weights + pinned KV + overhead` and the cap is only a ceiling above it.
  `mix-memory-bound` "reserves" 0.91 of the card and occupies about **82 of
  96 GiB**. Raising the budget to restore a reservation percentage would add
  *real* cache and change the experiment rather than restore it.
- **The derived cap is rounded to 2 dp, so two different caches can produce the
  same cap.** That makes the cap useless as an identity, which is why
  `kv_budget_gb` — not the cap — is part of the solo baseline's key. Measured:
  two tenants at 0.64 GiB and 1.28 GiB of cache both derived 0.19, and before
  the key was fixed, `--resume` reused the smaller one's baseline for the
  larger, halving the reference under every ratio built on it.

---



## 5. Architecture — more than one GPU

*Does adding GPUs eliminate contention, or just move it?* The answer depends entirely on **placement**,
and there are three genuinely different modes. They answer different
questions and should not be blended.

### Mode A — separate GPUs (isolation)

```
GPU 0: [ LLM ]          GPU 1: [ CV ]
```

One tenant per card. No shared SMs, no shared bandwidth, no shared VRAM.
Contention should disappear and the degradation ratio should return to
**≈ 1.0**.

Two uses. First, it's the honest answer to "should I just buy another GPU?" —
it puts a number on what the extra card actually buys. Second, it's a
**hardware null test**: if a tenant still shows degradation when its
neighbour is on a *different card*, the harness has a bug. That's a
verification we get for free.

### Mode B — tensor parallel

```
GPU 0: [ LLM shard 0 | CV ]     GPU 1: [ LLM shard 1 | CV ]
```

One model split across both cards. This is the mode people assume will fix
contention, and it mostly doesn't: the LLM now touches *every* GPU, so it
contends with whatever else is on *every* GPU, plus it adds cross-GPU
synchronisation on each forward pass. What TP actually buys is capacity —
more VRAM for weights, more aggregate bandwidth — not isolation.

For a 72B VLM this isn't a choice — `qwen2.5-vl-72b` at BF16 doesn't fit on one
card at all, which is also why the roster does not carry it.

> **Measured, 2026-08-04.** `nvidia-smi topo -m` on this box reports **PIX**:
> no NVLink, so cross-GPU traffic goes over PCIe Gen5. TP is therefore notably
> more expensive here than on an NVLink'd pair and would dominate any result it
> appeared in, so **no tensor-parallel colocation is written** — Phase 5
> measures placement only.



### Mode C — packing

```
GPU 0: [ LLM | CV ]     GPU 1: [ VLM | ILM ]
```

Four tenants over two cards, two per card. This is the actual
capacity-planning question: *given N GPUs and M models, what's the best
grouping?* And section 3 above gives the heuristic — pair tenants that stress **different**
resources. A bandwidth-bound LLM next to a compute-bound detector is a better
pairing than two LLMs, even though the memory arithmetic looks identical.

### What changes in the measurement

Two things, and both are easy to get wrong:

**The memory rule becomes per-GPU.** `gpu_memory_utilization` is a fraction
of *each* card. Tenants on different GPUs don't compete for VRAM at all, so
the `sum ≤ 1.0` check has to be applied per device, not per colocation. Two
tenants on separate cards can each take 0.9.

**The baseline must match the placement, not just the cache.** The KV cache trap
in section 2b said the solo baseline must use the contention run's cache size. On multi-GPU it must
also use the contention run's *topology*: a tenant running TP=2 in the
contention window needs a TP=2 solo baseline. Compared against a TP=1
baseline, the ratio would fold in all of TP's cross-GPU overhead and report
it as contention.

---



## 6. How to read the output

`bench summary --gpu <gpu>` writes a *Contention analysis* section into
`summary.md`, containing three things:

**The degradation table** — one row per (colocation, tenant): throughput
retention, and p50/p95/TTFT ratios against the matched solo baseline. The
primary result.

**The contention matrix** — p95 ratio arranged by victim × aggressor. Read it
in both directions — as section 3 argued, contention is not symmetric, and the asymmetry is usually
the most actionable finding in the study.

**The safe-operating envelope** — the colocations where `achieved_rps` fell
below `offered_rps`. That's the point where the GPU stopped keeping up with
what was asked of it, which is the deployment limit. It comes free from
open-loop load; no extra experiment is needed to find it.

### Sanity checks before believing any of it

- A solo tenant "colocated" with nothing must give ratio ≈ 1.0.
- `achieved_rps ≈ offered_rps` at low load, or the load generator was the
bottleneck rather than the GPU.
- No published run had a throttle reason fire.
- Exactly one sampler process per run.
- Ratios ≈ 1.0 *everywhere* usually means the load was too low to contend, or
closed-loop crept back in.

---



## 7. Where this sits


There are two different GPU studies in this repo, and mixing them up is the
most common way to misread either one.

| Your question | Which study | Skill |
|---|---|---|
| How fast can this model go on this GPU? | Single-model | `benchmark-gpu-inference` |
| Which backend should I deploy on? | Single-model | `benchmark-gpu-inference` |
| How much do these models slow each other down? | **Contention** | `gpu-contention-benchmark` |
| Can I run these two models on one card? | **Contention** | `gpu-contention-benchmark` |

**They use different baselines, and that is the whole reason they are separate
studies.**

- The **single-model** study gives each model the *whole GPU* and asks how fast
  it can possibly go. Best-case numbers.
- The **contention** study gives each model its *production share* of the GPU
  and asks how much a neighbour costs it. The baseline here is deliberately
  handicapped — see section 2b.

### The mix-up, with numbers

Take one 7B model. Here are three measurements of it — all real, all correct,
all different (figures illustrative):

| | Setup | p95 latency |
|---|---|---|
| **A** | the whole GPU to itself | 200 ms |
| **B** | at its production share of memory, no neighbour | 400 ms |
| **C** | same memory share, object detector running alongside | 500 ms |

**A is the single-model study. B and C are the contention study** — B is the
baseline, C is the measurement. The honest answer is **C ÷ B = 1.25×**: the
neighbour costs you 25%.

Now the two ways to get it wrong.

**Mistake 1 — reporting C as your hardware's speed.** *"A 7B takes 500 ms on
this card."* It doesn't. Alone it takes 200 ms. You have quoted a number
measured under a deliberate memory handicap *and* under interference, and made
the GPU look 2.5× slower than it is. Plan capacity on that and you buy
hardware you don't need.

**Mistake 2 — using A as the baseline instead of B.** Now the ratio is
C ÷ A = 500 ÷ 200 = **2.5×**, so you report *"the neighbour costs 150%"* and
conclude co-location is hopeless. But most of that gap — the 200 ms to 400 ms
part — was the memory allocation **you** chose, not the neighbour. The neighbour only
ever cost 25%. You would kill a deployment that was fine.

B is the number that looks wrong and is right. It is the model running exactly
as it will in production, with the card to itself — which is the only fair
thing to compare C against.
