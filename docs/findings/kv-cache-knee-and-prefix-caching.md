# The KV knee sits at ~5–7 GiB, and prefix caching is why three experiments missed it

**Date:** 2026-08-05
**GPU:** 1× RTX PRO 6000 Blackwell (96 GB), MPS, `EXCLUSIVE_PROCESS`
**Model:** `qwen2.5-72b` (AWQ, 38.77 GiB weights, 320 KiB of KV per token)
**Workload:** `llm_long` — 989-token prompt, 512-token output, 2 req/s offered, `--max-num-seqs=32`

## TL;DR

Two results, one of which invalidates a lot of earlier work.

**1. The KV knee is between 3.7 and 7 GiB.** Below it throughput falls ~17%;
above it, 7 / 14 / 46 GiB are indistinguishable.

| Cache | Achieved of 2 req/s | KV utilisation |
|---|---|---|
| 3.7 GiB | 1.27 | 99.9% |
| 6.0 GiB | 1.38 | 61.6% |
| **7.0 GiB** | **1.57** | 70.8% |
| 14.0 GiB | 1.52 | 35.1% |
| 46.0 GiB | 1.52 | 8.8% |

**2. The reason the cache never fills is prompt reuse, not rate or length.**
`llm_long` contains **two distinct prompts**. With prefix caching on, every
concurrent request shares the same 989-token prefix:

```
Prefix cache hit rate: median 97.4%
Running: 32 reqs            <- at the max-num-seqs cap
GPU KV cache usage: 8.7%    <- of a 150,720-token cache
```

32 concurrent × 1,501 tokens *should* occupy 48,032 tokens. Observed: ~13,100.
The prompt is free after the first request, so only generated tokens cost
unique cache — the effective working set is ~4.8 GiB no matter how much you
allocate.

## Why this matters

Three successive designs of the memory-pressure experiment tried to find the
eviction cliff by varying cache size, and all three produced flat curves:

| Generation | Approach | Result |
|---|---|---|
| `kv03…kv29` | caps + `llm_short` | 0.4% cache used, flat |
| `p25…p130` | `kv_budget_gb` + `llm_long` | reached 99.7% at p25 only |
| `cross-deploy-s25…s85` | split the leftover | **0.88 req/s at every split** |

The last is the sharpest evidence: halving the 72B's cache from 14 GiB to 7 GiB
changed throughput **not at all** (0.88 both times), and the 7B was equally
unmoved by 21 → 14 GiB.

Sizing the rungs was never the problem. **The workload cannot generate cache
pressure**, because 97% of its prompt tokens deduplicate.

## What to change

**Add prompt diversity, or disable prefix caching.** A scoped variant is the
smaller change and does not affect other colocations:

```yaml
backends:
  vllm:
    variants:
      prefix_off: ["--no-enable-prefix-caching"]
```

`--no-enable-prefix-caching` is already used by the `qwen3-vl` models in this
config, so it is known to work with this vLLM build.

The better long-term fix is more prompts: a 97% prefix hit rate is nothing like
production traffic, so *every* number in this study is measured against an
unrealistically cheap prefill. That affects far more than the memory-pressure
family.

## What this does NOT show

- **The knee is bracketed, not located.** It lies in (3.7, 7.0] GiB, and the
  7.0 GiB point (1.57) reads slightly *higher* than 14 and 46 (1.52) — within
  noise, but it means the plateau's exact start is unresolved.
- **These are solo numbers.** Under contention the same caches produced 0.88
  at both 7 and 14 GiB, so the knee under a neighbour was not measured.
- **It is specific to this workload.** With diverse prompts the working set
  would be ~48,000 tokens rather than ~13,100, and the knee would move up
  roughly proportionally — plausibly to ~15–20 GiB.
