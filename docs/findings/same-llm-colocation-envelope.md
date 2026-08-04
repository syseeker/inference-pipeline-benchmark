# Two LLMs on one RTX PRO 6000: flat to 16 req/s, queue collapse at 64

**Date:** 2026-08-04
**GPU:** 1× RTX PRO 6000 Blackwell (96 GB), MPS on, `EXCLUSIVE_PROCESS`
**Colocation:** `same-llm` (phase 2) — `qwen2.5-7b` (bf16) + `gemma2-9b` (bf16), one vLLM server each, same card
**Sweep:** 1, 4, 16, 64 req/s applied to *both* tenants simultaneously
**Repetitions:** 1 (see "What this does not establish")
**Results:** `benchmarks/results/rtx_pro6000/coloc/same-llm/`

## TL;DR

Colocating two 7–9B LLMs on one card is **nearly free up to 16 req/s each** and
**catastrophic at 64**. The transition is not a gradual slope:

| Offered (each) | qwen2.5-7b e2e p50 | ratio vs solo | gemma2-9b e2e p50 | ratio vs solo |
|---|---|---|---|---|
| 1 | 560 ms | 1.51× | 749 ms | 1.57× |
| 4 | 635 ms | 1.68× | 880 ms | 1.83× |
| 16 | 647 ms | 1.72× | 914 ms | 1.89× |
| 64 | 13,179 ms | **33.3×** | 27,053 ms | **37.3×** |

Solo references: qwen2.5-7b 371–396 ms, gemma2-9b 478–726 ms across the same rates.

The safe envelope on this hardware is **≤16 req/s per tenant**, where the cost of
sharing is a stable ~1.7–1.9× on latency and no measurable throughput loss.

## The cliff is admission queueing, not compute

This is the part that matters for capacity planning. At 64 req/s the two tenants
are not computing more slowly — they are waiting to start:

| at 64 req/s | ITL p50 | TTFT p50 | TTFT as share of e2e |
|---|---|---|---|
| qwen2.5-7b solo | 11.7 ms | 36 ms | — |
| qwen2.5-7b colocated | 20.8 ms (**1.8×**) | 12,541 ms (**348×**) | 95% |
| gemma2-9b solo | 15.6 ms | 259 ms | — |
| gemma2-9b colocated | 28.9 ms (**1.9×**) | 26,226 ms (**101×**) | 97% |

Inter-token latency degrades by under 2× — the same factor seen at every rate
below the knee, and about what two models sharing SMs should cost. Nearly all of
the 33–37× end-to-end blowup is queue time before the first token.

Practical consequence: **the knee is a scheduling limit, not a FLOPs limit.**
Adding compute would not move it; admission control, a lower `--max-num-seqs`,
or rate limiting upstream would. A deployment that watches ITL or GPU
utilisation will see nothing wrong right up until the collapse — TTFT is the
signal that leads.

## Aggregate throughput rises while per-request latency collapses

At 64 req/s offered to each tenant:

| | achieved | vs its own solo |
|---|---|---|
| qwen2.5-7b | 47.95 req/s | 77% kept |
| gemma2-9b | 34.19 req/s | 55% kept |
| **combined** | **82.14 req/s** | vs **62.4** for one model alone on the card |

So colocation *increases* aggregate throughput on the card — 82 req/s against 62
for a single tenant — while making every individual request 33–37× slower. The
card is doing more total work; no client is getting a usable answer. Whether that
is a win depends entirely on whether the workload is latency-bound or
throughput-bound, and the two tenants do not share the pain evenly.

## The larger model is the larger victim

`gemma2-9b` loses roughly twice as much as `qwen2.5-7b` on both axes: 55% vs 77%
throughput kept, 37.3× vs 33.3× latency. It also starts from a worse solo
position (259 ms TTFT vs 36 ms at 64 req/s), so it enters the contended regime
with less headroom. This is the victim/aggressor asymmetry the matrix is meant to
surface, visible in the simplest possible pair.

## What this does not establish

- **One repetition per point.** Phase 0 measured 1.8% run-to-run variance and set
  `recommended_reps: 1`, but that was on a loaded probe. The 1→16 req/s ratios
  (1.51 → 1.89) are close enough together that their *ordering* is not
  established; the flat-then-cliff shape is, since 33× is far outside any
  plausible variance.
- **The knee is not located.** It is somewhere in (16, 64]. The sweep is too
  coarse to say where, and the gap spans a 4× range. A 32 req/s rung is the
  single most informative addition to the next run.
- **Both tenants were driven at the same rate.** This says nothing about an
  asymmetric pair — a saturated tenant beside an idle one is the more common
  production shape and is not measured here.
- **`gemma2-9b` at 64 req/s solo was already degrading** (726 ms p50 against
  478 ms at 1 req/s), so part of its contended penalty is its own approach to
  saturation rather than contention. `qwen2.5-7b` was still flat at 396 ms.

## How to improve the next run

1. Add a **32 req/s rung** to `same-llm` to bracket the knee.
2. Add an **asymmetric arm** (one tenant at 4 req/s, one at 64) to separate
   "the card is oversubscribed" from "my neighbour is oversubscribed".
3. Raise `repetitions` to 3 on the sub-knee rungs, where the effect is close to
   the noise floor.
4. **Record TTFT explicitly in the coloc summary.** It is the leading indicator
   here and currently has to be dug out of the aiperf exports by hand.
