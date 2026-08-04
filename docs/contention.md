# GPU contention benchmarking — the concept, and the architecture

For anyone asking "what is this study actually measuring, and how is it laid out on the hardware?".

This is the conceptual document. The methodology decisions and their reasoning
live in
[skills/gpu-contention-benchmark/reference/design-decisions.md](../skills/gpu-contention-benchmark/reference/design-decisions.md);
the model list lives in
[reference/model-catalogue.md](../skills/gpu-contention-benchmark/reference/model-catalogue.md).
Read this one first.

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

A closed-loop client ("keep 4 requests in flight") slows itself down when the
GPU slows down, because it waits for replies before sending more. It hides
the exact damage you're trying to measure. Real workloads — game engines,
camera feeds, video pipelines — push at their own rate and don't wait for
you. Full reasoning in design-decisions §1, *Open-loop load, not concurrency*.

### 2b. Same memory allocation — the KV cache trap

This one is the least obvious, and it's the reason this section exists.

When vLLM starts it reserves `gpu_memory_utilization` × VRAM. Model weights
take a fixed slice of that, and **everything left over becomes the KV cache**
— the pool that holds attention state for in-flight requests. A bigger KV
cache means more requests can be in flight simultaneously, which means higher
throughput and less queueing.

So the cap is not a passive memory limit. It directly sets how fast the model
runs.

Now watch what happens if you leave it at the default for the solo run. A 7B
model on a 96 GB card:


|                                 | cap  | reserved | weights | **KV cache** |
| ------------------------------- | ---- | -------- | ------- | ------------ |
| Solo baseline at vLLM's default | 0.90 | 86 GB    | 15 GB   | **71 GB**    |
| Contention run, sharing with CV | 0.45 | 43 GB    | 15 GB   | **28 GB**    |


The contention run is now slower for **two** reasons, thoroughly mixed:

1. The neighbour is stealing SM time and memory bandwidth ← *what we want*
2. Its own KV cache is 2.5× smaller ← *a self-inflicted artifact*

There is no way to separate them afterwards. You would report "the detector
slowed my LLM by 2.4×" when much of that was "I gave my LLM a third of the
memory."

**The fix:** run the solo baseline at 0.45 too. Identical KV cache in both
runs, so the neighbour is the only difference left.

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

### Measured, 2026-08-04 — and the prediction was wrong

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

The prediction on record was **P1 best, P3 middle, P2 worst**. The measurement
is close to the reverse: **P2 is best on both worst-tenant and mean**, and P1 —
the predicted winner — is second worst by tail.

Two rules the data does support, each with a clean contrast behind it:

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

The rule on one GPU:

```
sum(tenant gpu_memory_utilization) + CV footprint  ≤  1.0
```

vLLM's default is 0.90 — essentially the whole card. Leave two tenants at the
default and the second one OOMs at startup, so each tenant needs its own cap.

Those caps are **derived, not hand-written**:

```
cap = (model weights + KV budget + overhead) / total VRAM
```

The KV budget is set once per colocation and is identical for every tenant in
it. So the cap comes out *different* per model — heavier weights need a bigger
slice — while the KV cache, the thing that actually sets speed, stays *equal*.

That is the same rule as the KV cache trap in section 2b: hold the cache
constant, and the neighbour is the only thing left that can explain a
difference. It also means the solo baseline inherits its cap from the
contention run automatically, rather than anyone having to remember to copy it.

Sizing caps *proportional to model size* is the obvious move and it would be
wrong — KV need follows request rate and context length, not parameter count.

You can still write an explicit cap when you mean to, and it wins. One
experiment does exactly that: the memory-pressure curve sweeps the cap on
purpose, because there the KV cache is the thing under test rather than the
thing held fixed.

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

For the 72B models this isn't a choice; `qwen2.5-vl-72b` at BF16 doesn't fit
on one card at all.

> **To verify on the hardware:** RTX PRO 6000 Blackwell has no NVLink, so
> cross-GPU traffic goes over PCIe. That makes TP notably more expensive than
> it would be on an NVLink'd H200 pair, and it's likely to dominate the
> multi-GPU results. Confirm the interconnect with `nvidia-smi topo -m`
> before drawing conclusions from Phase 5.



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

**The baseline must match the placement, not just the cap.** The KV cache trap
in section 2b said the solo baseline must use the contention run's memory cap. On multi-GPU it must
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
| **B** | capped at its production share, no neighbour | 400 ms |
| **C** | same cap, object detector running alongside | 500 ms |

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
part — was the memory cap **you** set, not the neighbour. The neighbour only
ever cost 25%. You would kill a deployment that was fine.

B is the number that looks wrong and is right. It is the model running exactly
as it will in production, with the card to itself — which is the only fair
thing to compare C against.