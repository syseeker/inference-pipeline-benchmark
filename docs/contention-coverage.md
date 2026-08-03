# Contention study — coverage against the original design

What the customer asked for in `workspace/contention/experiment_config.json`,
what this harness implements, and where the two deliberately differ.

Audited **2026-08-03** against the customer's seven phases. Build status for
the harness itself lives in the
[skill's handoff table](../skills/gpu-contention-benchmark/SKILL.md); this
document is about *experiment* coverage, not code.

For the concepts behind any of it, read [contention.md](contention.md) first.

**Legend** — ✅ implemented · ⚠️ implemented but differs from the request ·
❌ disqualified, reason given · ⬜ not yet built

---

## The short version

Of the customer's ~120 planned runs:

- **Most of the design is adopted unchanged.** The phase structure, the solo-baseline-first discipline, the dual-baseline idea in Phase 6 and the metrics-reporting policy are all theirs and all kept.
- **Four things are disqualified on technical grounds**, listed in §9. Three of them are the customer's own picks and need telling.
- **The largest single change is closed-loop → open-loop.** Every experiment specified as a *concurrency* sweep is reframed as a *request-rate* sweep. This is not a preference; a closed-loop client throttles itself in proportion to the slowdown being measured, so those runs would have described the harness rather than the GPU.

---

## Phase 0 — concurrency validation (gate)

| Their experiment | Status |
|---|---|
| `validate-concurrent-kernels` — LLM + CV SM overlap | ✅ `scripts/gpu_concurrency_probe.py`. Run on PRO 6000: MPS **off** → 0.28× aggregate throughput (serialising, gate FAIL); MPS **on** → 1.94× (PASS). Variance CoV 0.4% → 1 repetition per scenario |
| `validate-vlm-cv-concurrent` — VLM prefill + CV | ⬜ the probe supports it; this variant has not been run |
| `validate-standalone-vs-triton` — vLLM vs vLLM-inside-Triton | ❌ **We never serve an LLM through Triton.** LLM/VLM go to native servers, CV goes to Triton — so this overhead comparison has no bearing on any run we make. See [serving-topology.md](../skills/gpu-contention-benchmark/reference/serving-topology.md) |

The MPS result is worth dwelling on: **without MPS the gate fails outright.**
0.28× means the tenants were doing worse than running one after the other.
Any contention number collected without MPS would have been measuring the
time-slice scheduler.

---

## Phase 1 — solo baselines

| | Status |
|---|---|
| Per-model solo baselines | ✅ generated automatically by `solo_baselines: auto` for every tenant appearing in a colocation, deduped across the study |
| × output length {short, long} | ✅ `llm_short` / `llm_long` workloads |
| × quantization {q4, fp16} | ❌ see §9 |

**One departure worth understanding.** A solo baseline here runs at the *same
memory cap and same offered rate* as the contention run it will be compared
against — not at the model's best possible settings. That makes it a
deliberately handicapped reference, and it is the only kind of reference a
degradation ratio can be built on. Full explanation in
[contention.md §2](contention.md).

---

## Phase 2 + Phase 3 `*-only` — same-category load sweeps

These two phases overlap in the original design, and we implement them as one
family rather than two.

| Their experiment | Status |
|---|---|
| Ph2 `concurrency-llm` — 2 LLMs, c1→16 | ⚠️ reframed as an **rps** sweep |
| Ph2 `concurrency-cv` — 3 CV models, rps 1→200 | ⬜ already open-loop, needs building |
| Ph2 `concurrency-vlm` — 2 VLMs | ⚠️ reframed as rps |
| Ph2 `concurrency-ilm` — 2 ILMs | ⚠️ reframed as rps |
| Ph3 `mix-llm-only` — 3 LLMs at fixed c4 | ⚠️ folded into the Ph2 sweep |
| Ph3 `mix-cv-only` — 4 CV models at fixed rps 50 | ⚠️ folded into the Ph2 sweep |

**Why folded.** Phase 3's `*-only` entries are a single fixed-load point on
the curve Phase 2 already sweeps:

| | Models | Load |
|---|---|---|
| Ph2 `concurrency-llm` | qwen2.5-7b, gemma2-9b | swept c1→16 |
| Ph3 `mix-llm-only` | + qwen2.5-14b | fixed c4 |

Implementing them separately means running the same colocation twice to get a
point we already have. Instead there are four same-category colocations —
`same-llm`, `same-cv`, `same-vlm`, `same-ilm` — each with an rps sweep. Phase
2 is the whole curve; Phase 3's entry is the rps=50 row of it.

**Why rps and not concurrency.** Phase 2 is looking for the saturation point.
Under open-loop load that is exactly where `achieved_rps` falls below
`offered_rps` — the same answer the customer wanted, from ratios that are
still valid. It also comes free, with no extra experiment.

---

## Phase 3 — the eight mix types

| Their mix | Status |
|---|---|
| `mix-llm-cv` | ✅ |
| `mix-vlm-cv` | ✅ VLM substituted — see §9 (paligemma2 is image-only) |
| `mix-ilm-cv` | ✅ |
| `mix-llm-vlm` | ✅ covered by Phase 4's `cross-vlm-prefill-vs-llm`, which is the same composition with a designated subject |
| `mix-llm-only`, `mix-cv-only` | ⚠️ folded into Phase 2 above |
| `mix-vlm-ilm` | ⬜ needs a VLM substitution first |
| **`mix-full`** — all four categories | ⬜ **the significant gap.** It is the only 4-tenant combination, and it is what makes multi-GPU placement a real decision rather than a formality |

---

## Phase 4 — cross-type contention

| Their experiment | Status |
|---|---|
| `cross-llm-vs-cv-rps` 1/10/50/200 | ✅ exact match — already open-loop in their design |
| `cross-vlm-prefill-vs-llm` | ✅ the sharpest experiment in the study, see below |
| `cross-arch-validation` qwen/llama/mistral | ✅ |
| `cross-size-scaling` 7b/14b/32b/72b | ⚠️ present but was **broken** — inherited a fixed 0.45 cap (43 GB) while qwen2.5-72b AWQ needs ~45 GB, so the top rung could not load. Repaired by deriving caps from weights + a constant KV budget |
| `cross-ilm-vs-cv-rps` | ⬜ |
| `cross-cv-vs-llm-concurrency` | ❌ as written — sweeps LLM **concurrency**. Reframed to sweep LLM rps with CV as the subject ⬜ |
| `cross-memory-pressure` — 4-point VRAM curve | ⬜ **high value.** Walks utilization 47% → 86% to find where KV-cache eviction begins. `repetitions: 3`, because near-OOM behaviour is bimodal — the model either fits or thrashes, and the mean of those two states describes neither |

### What `cross-vlm-prefill-vs-llm` measures, and why it is the best one

Every transformer request has two phases with opposite resource profiles.
**Prefill** processes the whole input at once — a large parallel matmul,
compute-bound, saturating the SMs. **Decode** emits output tokens one at a
time, each requiring the full weight matrix streamed from VRAM —
bandwidth-bound, leaving the SMs largely idle.

For a text LLM with a short prompt, prefill is a few milliseconds. For a VLM
handed a 10-second clip it means decoding 40 frames, running the vision
encoder over all of them, then attending across several thousand tokens. That
is a sustained several-hundred-millisecond burst that pins the SMs.

So the experiment asks: **when that burst lands, what happens to the LLM
beside it?** The LLM is in bandwidth-bound decode, emitting tokens steadily,
until the VLM's prefill takes the SMs and the next token stalls.

The metric is **ITL P99 spike**, not mean latency — the effect is transient
and periodic, and an average erases it completely. It is also why per-request
epoch timestamps exist: the causal claim is that the LLM's ITL spike lands
*inside* the VLM's prefill window, and only a shared wall clock can show that.

---

## Phase 5 — GPU scaling (2 GPUs)

✅ **Built 2026-08-03** — five colocations, listed below.

⚠️ **Scoped to 2 GPUs. The customer's 4-GPU arm is dropped by decision**, not
by any technical limitation — `device:` supports up to 8 GPUs, so restoring it
is a config change rather than a code change. This halves the phase's run
count.

| Entry | Placement | Question |
|---|---|---|
| `place-p1` | GPU0[LLM+VLM] GPU1[ILM+CV] | Predicted best — does separating the two compute-heavy tenants win? |
| `place-p2` | GPU0[LLM+ILM] GPU1[VLM+CV] | Predicted worst — VLM burst landing on the 5 ms detector |
| `place-p3` | GPU0[LLM+CV] GPU1[VLM+ILM] | Predicted middle — two compute-heavy tenants together |
| `place-isolated` | `mix-llm-cv`, one tenant per card | What a second GPU actually buys — **and the hardware null test** |
| `place-vlm-prefill-split` | `cross-vlm-prefill-vs-llm` split | Does the LLM's ITL spike vanish when the burst is on another card? |

**The measurement rule that makes the three pairings comparable.** They are
compared against each other, so placement must be the only variable — but the
pairings are not naturally memory-equivalent. P1 puts both vLLM tenants on one
card while the two Triton tenants take almost no fraction; P2 and P3 give each
vLLM tenant a card to itself. Derived per-GPU, P1's tenants would get roughly
half the KV cache of P2's and P3's, and **P1 would look bad for a reason that
has nothing to do with its neighbours** — §2b's artifact, wearing a different
hat.

So all three carry an identical `kv_budget_gb: 20.0`, sized for P1 as the
tightest case, and it is the same budget `mix-full` uses — which makes the
1-GPU → 2-GPU comparison valid as well. Verified: every vLLM tenant resolves
to the same cap (llm 0.39, vlm 0.30) in all three pairings and in `mix-full`.

**No tensor-parallel entries.** See the interconnect note below; TP results
would likely be dominated by PCIe rather than by contention, so they are
deferred rather than collected and misread.

The customer's framing — *"does adding GPUs eliminate contention or just
redistribute it?"* — resolves into three genuinely different placements that
answer different questions:

| Mode | Shape | What it answers |
|---|---|---|
| **Separate** | one tenant per card | Should I just buy another GPU? Also a hardware **null test** — if a tenant still degrades with its neighbour on a different card, the harness has a bug |
| **Tensor parallel** | one model split across cards | Adds capacity, **not** isolation. The model now touches every GPU, so it contends on every GPU, plus cross-card sync each forward pass |
| **Packing** | 4 tenants over 2 cards | The real capacity-planning question — and the one that needs `mix-full` to exist |

**A prediction worth testing rather than assuming.** With four tenants over
two cards there are exactly three pairings. The resource-profile heuristic in
[contention.md §3](contention.md) predicts a ranking:

```
P1   [LLM+VLM] | [ILM+CV]     predicted best  — separates the two compute-heavy models
P3   [LLM+CV]  | [VLM+ILM]    predicted worse — VLM and ILM both compute-heavy
P2   [LLM+ILM] | [VLM+CV]     predicted worst — VLM's burst lands on the most fragile tenant
```

If the measured ranking matches, the customer gets a **placement rule they can
apply to models we never tested** — worth considerably more than three
benchmark numbers.

> **Blocker cleared (2026-08-03).** Per-GPU Triton containers landed: the
> orchestrator now launches one container per card that has Triton tenants on
> it, each with its own name (`triton-cv`, `triton-cv-gpu1`, …), its own port
> block (base + 10 × device, so GPU 0 keeps 8100/8101/8102), its own model
> repository holding only that card's models, and `--gpus device=<N>`. All
> three pairings are now expressible; P2 and P3 no longer need CV and ILM on
> the same card. A Triton tenant must still name exactly one device — the CV
> models are not tensor-parallel, and a `device:` list is rejected.

> **To verify on hardware.** The PRO 6000 config records `nvlink: false`, so
> cross-GPU traffic goes over PCIe Gen5. Tensor-parallel results will likely
> be dominated by interconnect rather than by contention. Confirm with
> `nvidia-smi topo -m` before drawing conclusions.

---

## Phase 6 — secondary dimensions

| Their dimension | Status |
|---|---|
| `output_length` {short, long} | ✅ |
| `backend_cv` {pytorch, onnx, tensorrt} | ✅ as {python, onnx, tensorrt} |
| `asymmetry` {equal, moderate, extreme} | ✅ as rps 4 / 16 / 64 — 1:1, 4:1, 16:1 |
| `arrival_pattern` {burst, uniform} | ✅ as {poisson, constant} — Poisson added at no cost, and it is the realistic middle the original was interpolating between |
| `backend_llm` {vllm, trt-llm, sglang, llamacpp} | ⚠️ {vllm, sglang} built; trt-llm ⬜; llamacpp ❌ §9 |
| `input_size_llm` — longer prompt → longer prefill | ⚠️ **`input_size_cv` was built instead.** A legitimate experiment, but not the one asked for; the LLM-prefill question is still open ⬜ |
| `quantization` | ❌ §9 |
| **Dual baseline A/B** | ❌ **structural gap** — every `secondary-*` currently extends the compute-bound baseline only |

**The dual-baseline gap is the most substantive omission in the harness.**
The customer's design runs each secondary dimension against *two* contrasting
baselines — compute-bound (small models, high load) and memory-bound (72B,
moderate load) — precisely so interaction effects surface. A finding like
*"backend choice matters 3× more under memory pressure"* is invisible with one
baseline. Our own design record singles this out as something the customer got
right; it then went unimplemented. Every `secondary-*` result today answers
half its question.

---

## §9 — Disqualified, with reasons

Four technical disqualifications. Three concern the customer's own picks and
should be raised with them directly.

| Item | Why | What we do instead |
|---|---|---|
| **Quantization dimension (`q4` = Q4_0)** | Q4_0 is a **llama.cpp GGUF** format. vLLM cannot load it — GGUF moved out-of-tree to `vllm-gguf-plugin` and is documented as highly experimental | Drop the dimension; pick the best-fitting format per model per GPU — official AWQ where one exists, else BF16/FP8 |
| **`llamacpp` backend** | No Triton backend exists, and it adds no contention axis that vLLM/SGLang/TRT-LLM do not already cover | Omitted |
| **`gemma-vlm-32b` → `paligemma2-28b-pt-896`** | It is 28B not 32B, a `-pt-` **base** checkpoint, and **image-only** — vLLM raises `"Only image modality is supported"`. **It cannot serve the video dimension it was chosen for** | Kept in config, serves image rounds, skips video rounds cleanly with a reason. Natural replacement from their own list: `qwen2.5-vl-7b` |
| **`qwen2.5-27b`** | The HF repo does not exist. The Qwen2.5 ladder is 0.5/1.5/3/7/14/**32**/72B — 27B is a Gemma 2 size | `qwen2.5-32b`, with the original entry recorded in a comment |

Two further picks are kept but degraded, and the customer should know:

- **`paddleocr`** — its TensorRT path pins TRT 8.6.1.6 + CUDA 11.8, which cannot coexist with Triton 26.07's stack. Served on the Triton **Python backend**, unoptimised, rather than not at all.
- **`kosmos-2.5`** — no vLLM or SGLang implementation exists. Also Triton Python backend.

Neither is a contention finding. If these two look slow in the results, that
is the serving path, not the neighbour.

---

## What is outstanding

**Construction is done.** As of 2026-08-03 everything on the original
outstanding list is built — VRAM cap sizing, the four `same-*` colocations,
the Phase 6 dual baseline, the memory-pressure curve, `mix-full`, per-GPU
Triton containers and the Phase 5 placement study. **39 colocations** resolve
with no VRAM pre-flight issues, under **221 unit tests**.

What remains is **validation, not construction** — and it needs hardware.
Nothing here has ever run on a GPU. The weight figures that set every derived
cap are estimates rather than measurements; the Docker flags, port scheme and
per-device repositories are asserted only as strings in unit tests; and the
placement ranking is a prediction on the record, not a result.

See **[contention-gpu-validation.md](contention-gpu-validation.md)**. It is
ordered by how much work each assumption invalidates if it is wrong — start at
the top, not at the interesting end.
