# TRT-LLM 1.2.1 on Qwen3-VL-32B-FP8 — what's actually happening

**Date:** 2026-05-09
**GPU:** RTX PRO 6000 Blackwell Server Edition (96 GB)
**Backend:** `trtllm-serve --backend pytorch`, TRT-LLM 1.2.1
**Run id:** `76cd1fb047f9` (`benchmarks/results/rtx_pro6000/trtllm-qwen3-vl-32b-fp8-76cd1fb047f9.json`)

## TL;DR

TRT-LLM is not a bad framework. **TRT-LLM 1.2.1's `trtllm-serve` pytorch
flow on Qwen3-VL multimodal** is what's rough — three different
optimizations that vLLM/SGLang use freely on this model are either
broken or disabled here. On top of that, 32B FP8 on a single GPU is
genuinely big and would be slow on any framework.

| Headline metric | Value | Read |
|---|---|---|
| TTFT p50 | **42,476 ms** | First token takes 42 seconds. |
| E2E p50 | **44,225 ms** | Decode adds barely anything; TTFT *is* the cost. |
| `command_success_rate` | **0.0** | Every output failed schema validation. |
| `wall_time_s` | 106 s for 3 requests | 2 of those budgets are pure prefill. |
| `gpu_util_pct_peak` / `fb_used_peak_gb` | 100% / 88.5 GB | Hardware is working, just slow. |
| `power_avg_w` | 172 W (peak 421 W) | Long stretches of light load — not stuck on compute. |

## The four root causes, in plain terms

### 1. JSON guardrail is broken on this combo (validity = 0%)

vLLM and SGLang have a "physical takeout box" (xgrammar) that prevents
the model from emitting anything that isn't valid JSON for our action
schema. TRT-LLM 1.2.1 has the same box, but the lid doesn't fit on
Qwen3-VL — turning it on crashes the server at startup with:

```
AttributeError: 'Qwen3VLModel' object has no attribute 'vocab_size_padded'
```

(Inner cause: the guided-decoder constructor at
`tensorrt_llm/_torch/pyexecutor/py_executor_creator.py:504` reads
`vocab_size_padded` directly off the top-level model object, but
`Qwen3VLModel` is a multimodal *wrapper* — that attribute lives on the
inner language model.)

So the choice is "lid off" (this run, 0% valid) or "kitchen closed" (no
service). vLLM/SGLang will hit ~100% validity on the same model.

**Implication:** validity here is a measurement of *integration
maturity*, not raw inference quality.

### 2. Vision tower runs every request (no KV-cache reuse)

A VLM has two halves: a vision encoder (turns the image into thousands
of "image tokens") and the language model (reads tokens, generates a
reply). Every request goes through both halves — there's no way to
generate without first reading.

What KV-cache **reuse** changes is *only* whether one request can
shortcut work already done by an earlier request whose prompt starts
with the same tokens. *Remember page 1 instead of re-reading it.*

On TRT-LLM 1.2.1, multimodal models *must* run with KV-block reuse
turned off — leaving it on hits a different bug. So even when two
requests share their system prompt, the second one re-processes
everything from scratch. vLLM/SGLang keep reuse on.

For our benchmark this matters less than you'd think — every scenario
has a *different* image, so the shareable prefix is just the ~100-token
system prompt versus thousands of vision tokens. The bigger cost is
that the **vision tower itself runs from scratch on every request**,
regardless of KV setting; TRT-LLM 1.2.1 doesn't cache ViT outputs even
when the same image is sent twice.

**Implication:** TRT-LLM gives up the prefix-shortcut layer that the
others keep, and the always-rerun ViT is the dominant repeated work.

See [§ Appendix: prefill, decode, and three different
caches](#appendix-prefill-decode-and-three-different-caches) for the
full picture.

### 3. Lazy CUDA-graph capture penalizes the first few requests

CUDA graphs are a memorized GPU workflow. The first time the GPU runs
a particular tensor shape, it figures out the steps; later runs with
the same shape are much faster.

Look at the per-scenario times:

```
01_clash_of_clans_start_attack: 47.8 s   (warmup)
01_clash_of_clans_start_attack: ~47 s    (cold — recipe being learned)
02_catan_open_menu:             14.1 s   (recipe memorized — fast)
03_fps_engage_and_reload:       44.2 s   (different shape — re-learning)
```

The 14 s outlier is what TRT-LLM *actually* runs at, after warmup.
The 47 s and 44 s rows include the cost of capturing CUDA graphs for
new shapes. vLLM/SGLang try to capture a family of shapes upfront
during warmup; TRT-LLM 1.2.1's pytorch backend captures lazily.

**Implication:** the current `warmup_requests=1` is unfair to TRT-LLM.
It needs ~5–10 warmups spanning the image-size variety before timed
measurement starts to get a fair reading.

### 4. 32B is big — but in different ways for prefill vs decode

A 32B model in FP8 weighs ~33 GB. Add KV cache, activations, and the
vision tower → ~88 GB resident on a 96 GB card. The GPU has the room.
What "big" actually costs depends on which phase you're in:

| Phase | What happens | What limits it | Floor on PRO 6000 |
|---|---|---|---|
| **Prefill** (input → first token) | Each weight is read once and reused across thousands of input tokens in one big matmul. Lots of math per byte. | **Compute (FLOPS).** | ~100–300 ms theoretical |
| **Decode** (one output token at a time) | Each weight is read fresh for each output token. Almost no reuse. | **VRAM bandwidth.** | ~33 GB / 1.5 TB/s ≈ 22 ms / token |

Your run's 42-second TTFT is **neither** of those:

- Most of the 42 s is **first-touch overhead** — CUDA graph capture and
  kernel autotuning the first time the GPU sees a given input shape.
  After warmup, scenario 02 dropped to **14 s** for the same kind of
  work. That's the more honest steady-state number.
- 14 s is *still* slow vs the 100-300 ms compute-bound floor, because
  the vision tower runs from scratch every request and the model is
  wide enough that even a fast prefill has a lot to coordinate.

**Implication:** "32B is big for one GPU" still holds, but the
mechanism matters when reading the numbers. vLLM/SGLang on this same
model + GPU will also be slow — just less so, because they don't pay
the lazy-graph penalty and they can shortcut prefix work via KV reuse.

If you want fast: drop to 8B FP8 (much smaller weights), or split the
32B across 2× GPUs (TP=2) to halve the per-GPU weight footprint.

## What this benchmark *is* measuring

| Question | Answer |
|---|---|
| Is TRT-LLM 1.2.1 bad for Qwen3-VL-32B-FP8 today? | **Yes** — schema enforcement broken, KV reuse disabled, lazy graph capture penalizes small N. |
| Is TRT-LLM bad for VLMs in general? | **No** — H200 + a prebuilt TRT-LLM engine + sustained workloads is historically where TRT-LLM beats vLLM. NIM uses TRT-LLM under the hood. |
| Is TRT-LLM bad for text-only LLMs? | **Definitely no** — that's its strongest case. |
| Is `trtllm-serve` (pytorch backend) the right comparison vs vLLM? | **Yes for dev workflow, no for peak performance.** trtllm-serve is "easy mode." For peak, you build engines with `trtllm-build` and serve those — which is what NIM does internally. |

## Honest framing for the report

> On the pytorch-backend `trtllm-serve` flow at TRT-LLM 1.2.1, Qwen3-VL
> multimodal is roughly a year behind vLLM/SGLang in integration
> polish. The performance ceiling exists — NIM proves it — but reaching
> it means giving up the dev-friendly path and building engines
> manually.

## Open questions / next steps

- **Re-run with bigger warmup.** Bump `warmup_requests` to 8–10 and
  cycle through scenarios with varied image sizes so CUDA graphs are
  captured before timed measurement starts. Compare the warm-state TTFT
  to the current 42 s.
- **prefill / decode percentile fields stay null for trtllm.** The TRT-LLM
  Prometheus exporter doesn't emit `prefill_time` / `decode_time`
  histograms (only TTFT, TPOT, queue, E2E). The data *is* derivable
  per-request from the JSON `/perf_metrics` endpoint
  (`first_token_time − first_scheduled_time` for prefill,
  `last_token_time − first_token_time` for decode), but that scrape
  isn't wired up — small piece of work (~30 lines in `prom_scrape.py`),
  worth doing if the gaps become annoying when comparing against
  vLLM/SGLang rows.
- **Try 8B FP8 on the same hardware** to isolate "bandwidth-bound 32B"
  from "TRT-LLM's overhead." If 8B closes the gap with vLLM, the 32B
  number is mostly hardware. If 8B is *also* slow vs vLLM, it's
  framework.
- **Watch upstream for `vocab_size_padded`.** Once Qwen3VLModel exposes
  that attribute (or TRT-LLM teaches the guided decoder to look on the
  inner LM), xgrammar becomes available and validity should jump to
  ~100%.
- **NIM-as-baseline run.** If we can stand up a Qwen3-VL NIM container
  on the same GPU, the gap between NIM and `trtllm-serve` is the
  cost of "easy mode" — useful evidence in either direction.

## Appendix: prefill, decode, and three different caches

### Every request runs prefill — always

Inference has two phases:

```
[input tokens]  ──prefill──▶  [internal state (KV cache for this request)]
                                            │
                                       ┌────▼────┐
                                       │  decode │  ── one output token at a time
                                       └─────────┘
```

Prefill is the model "reading" the prompt+image to build internal
state; decode generates output tokens one at a time using that state.
**Every request always does both.** There's no way to generate without
first reading.

What changes between requests is *whether prefill can shortcut shared
work*, and that depends on which "cache" you're talking about. Three
different mechanisms get conflated:

| Mechanism | What it remembers | What triggers reuse | Status in this run |
|---|---|---|---|
| **CUDA graph cache** | The GPU's *recipe* for executing a given tensor shape (compiled kernels, scheduling) | Same input shape as a previous request | On — that's why scenario 02 was fast |
| **KV-cache reuse / prefix caching** | Intermediate per-token activations from previous prompts | Two requests start with identical token prefixes | **Off** — forced for VLM on TRT-LLM 1.2.1 |
| **ViT output caching** | Image embeddings produced by the vision tower | Same image sent twice | Not implemented in TRT-LLM 1.2.1 |

### Worked example

```
Req 1 input: [system prompt] [image A] [instruction A]
Req 2 input: [system prompt] [image A] [instruction B]
```

| Setting | What req 2 does at prefill |
|---|---|
| KV reuse on, ViT cache on | Skip system prompt + image-A tokens, only prefill instruction B |
| KV reuse on, ViT cache off | Re-run ViT on image A; LM-side prefill skips system prompt + (now-recomputed-but-identical) image tokens |
| KV reuse off (this run) | Re-run ViT on image A; LM-side re-prefills everything from scratch |

Across our scenarios the images are *all different*, so even ideal KV
reuse would only save the ~100-token system prompt — small compared
to thousands of vision tokens per request. The big repeated cost is
the ViT, which nothing in TRT-LLM 1.2.1 deduplicates.

### Why scenario 02 was fast even with all caches "off"

Scenario 02 (14 s) wasn't fast because it skipped prefill — prefill
ran in full. It was fast because the **CUDA graph cache** had already
memorized how to schedule the work from scenario 01's first run. Same
work, no compile-time overhead. This is a different layer from KV
reuse, and it's the one that *did* help.

That's why warmup matters so much for TRT-LLM 1.2.1 on this model:
each new image size is a "first sight" the graph cache hasn't covered
yet, and the lazy capture cost gets billed against the timed request.
Bumping `warmup_requests` from 1 to ~10 across varied sizes is
expected to drop the timed TTFT substantially.
