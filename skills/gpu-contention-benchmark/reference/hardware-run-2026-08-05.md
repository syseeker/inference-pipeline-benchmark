# Hardware run, 2026-08-04 → 05: results, root causes, and what to do next

33 hours on 2× RTX PRO 6000, 201 manifests, 38 commits. This is the record a
fresh session should read before touching the contention harness.

**Read this first if you are about to run or redesign the memory-pressure /
deployment family.** Three successive designs of that experiment produced
meaningless flat curves for a reason that has nothing to do with how the rungs
were sized (§3), and the same class of `--resume` bug bit three separate fields
in one day (§4).

---

## 1. Results worth quoting

### The deployment cost — the customer's phase-4 question

Adding a `qwen2.5-7b` to a card already running `qwen2.5-72b`, both vLLM under
MPS, `llm_long`, 300 s windows:

| | alone (whole card) | colocated | kept |
|---|---|---|---|
| `qwen2.5-72b` | 1.52 req/s | 0.88 | **58%** |
| `qwen2.5-7b` | 3.81 req/s | 2.48 | **65%** |

Aggregate rises to 3.36 req/s — the card does more total work while both
tenants get materially slower. Reproducible to **two decimals across 12
contention runs**.

**Treat 42% as an OPTIMISTIC bound.** It was measured with prefill essentially
free (§3); with realistic prompt diversity the cost should be higher.

### How you split the memory does not matter

Four splits of the same 28 GiB leftover, three reps each:

| split | 72B cache | 72B req/s | 7B cache | 7B req/s |
|---|---|---|---|---|
| s25 | 7.0 GiB | 0.88 | 21.0 | 2.48 |
| s50 | 14.0 | 0.88 | 14.0 | 2.48 |
| s75 | 21.0 | 0.88 | 7.0 | 2.48 |
| s85 | 23.8 | 0.88 | 4.2 | 2.48 |

Identical across a 3.4× range of 72B cache and 5× of 7B cache. **The cost is
compute contention; KV is not the binding constraint anywhere in that range** —
though see §3 for why that conclusion is conditional.

### The KV knee, and how little the small model needs

From solo baselines accumulated across all three experiment generations
(`llm_long`, 2 req/s offered):

| 72B cache | achieved | KV utilisation |
|---|---|---|
| 3.7 GiB | 1.27 | 99.9% |
| 6.0 | 1.38 | 61.6% |
| **7.0** | **1.57** | 70.8% |
| 14.0 | 1.52 | 35.1% |
| 21.0 | 1.52 | 23.4% |
| 46.0 | 1.52 | 8.8% |

The knee is in (3.7, 7.0] GiB; above it, 7 / 14 / 21 / 46 are indistinguishable.

The 7B is flat across a **110× range** — 3.81 req/s at every cache from 0.64 to
70 GiB, losing nothing even at 92.9% utilisation. **A 7B on this workload needs
under 1 GiB of KV.** Anything more is wasted card.

### same-llm: a flat tax, then a cliff

Two 7-9B LLMs on one card cost a stable **1.6-1.9×** latency from 1 to 16
req/s each, then **33-37×** at 64. The cliff is admission queueing, not
compute: ITL degrades under 2× while TTFT degrades 348×/101× and accounts for
95-97% of end-to-end latency. **Monitoring utilisation or token latency gives
no warning.** Full write-up:
[docs/findings/same-llm-colocation-envelope.md](../../../docs/findings/same-llm-colocation-envelope.md).

### Backend parity (with a caveat)

SGLang 1.98 vs vLLM 1.97 req/s on `qwen2.5-72b` under three-tenant contention,
**with matched caches** (19,660 vs 19,648 tokens). Before the fix in §5, SGLang
self-sized to 32,702 tokens — 1.7× vLLM's — so any earlier "the backends tie"
reading was not a fair test. Both sit at ~99% of offered rate, so this shows
parity *when neither is stressed*, not under saturation.

---

## 2. `weights_gb` is measurable, and mostly correct

vLLM logs `Model loading took X GiB memory` into the captured server log. Seven
of eight declared values are accurate to within 0.10 GiB once the yaml's GB is
converted to the GiB vLLM reports. Only `qwen2.5-72b` was materially wrong
(41.91 declared vs 38.77 measured) and is now corrected to 41.6.

**The field is `weights_gb` and its values are GB; vLLM reports GiB.** Compare
them directly and every estimate looks ~7% high.

**Caveat:** once `--kv-cache-memory-bytes` is set, vLLM stops logging
`Available KV cache memory`, so the cache ground truth is no longer recoverable
from the log for exactly the tenants whose cache matters most.

---

## 3. THE ROOT CAUSE: prefix caching makes cache pressure unreachable

Three generations of the memory-pressure experiment produced flat curves:

| generation | approach | outcome |
|---|---|---|
| `kv03…kv29` | caps + `llm_short` | 0.4% cache used, flat |
| `p25…p130` | `kv_budget_gb` + `llm_long` | 99.7% at p25 only |
| `cross-deploy-s25…s85` | split the leftover | 0.88 req/s at **every** split |

The cause is not rung sizing. It is that **`llm_long` has two distinct prompts
and `llm_short` has three**, so with prefix caching on every concurrent request
shares the same prefix:

```
Prefix cache hit rate: median 97.4%
Running: 32 reqs                  <- at the --max-num-seqs cap
GPU KV cache usage: 8.7%          <- of a 150,720-token cache
```

32 concurrent × 1,501 tokens *should* occupy 48,032 tokens. Observed ~13,100.
The prompt is free after the first request, so only generated tokens cost
unique cache and **the working set is ~4.8 GiB no matter what you allocate**.

Two further traps in the same area:

- **`--max-num-seqs=32` caps residency**, so raising the arrival rate cannot
  fill the cache either. `llm_short` (61 tokens) occupies at most 1,952 tokens
  — 6.8% of even the smallest rung — at *any* rate. Only a workload that holds
  tokens can make the cache the constraint.
- **A cap sum is not memory pressure.** Once `kv_budget_gb` pins the cache, the
  derived `gpu_memory_utilization` is only a ceiling; occupancy is
  `weights + pinned KV`. `mix-memory-bound` "reserves" 0.91 of the card but
  occupies ~82 GiB. Raising the budget to restore a reservation percentage adds
  *real* cache and changes the experiment.

**Fix, ready to apply** (`docs/next-run/config-changes.md` §1e, ~1.3 h at
`repetitions: 1`):

```yaml
backends:
  vllm:
    variants:
      prefix_off: ["--no-enable-prefix-caching"]
```

Scoped via `variant:` so no other colocation is affected. Already used by the
`qwen3-vl` models, so it works with this vLLM build.

**The deeper fix is more prompts.** A 97% prefix hit rate is nothing like
production traffic, so *every* prefill-sensitive number in this study — TTFT
above all — is measured against an unrealistically cheap prompt. This reaches
far beyond the memory-pressure family.

**Always verify with `GPU KV cache usage` in the server log before trusting any
cache-pressure result.** Single-digit percentages mean the rungs measured
nothing, however clean the numbers look.

---

## 4. `gpu_memory_utilization` cannot apportion a shared card

A cap is a **total-device** target. A colocated tenant subtracts its
neighbours' resident memory from its own allowance and derives a negative
cache:

```
Model loading took 14.25 GiB memory
Available KV cache memory: -37.31 GiB
ValueError: No available memory for the cache blocks.
```

Confirmed quantitatively — the deficit tracks the cap exactly (predicted
−34.63 GiB at the next rung, measured −34.46).

**Use `kv_budget_gb`**, which emits the absolute `--kv-cache-memory-bytes` and
composes across processes. Caps are not *impossible* (a high enough cap loads)
but they are unmaintainable: the number then encodes the other tenant's
footprint and the load order.

### The `--resume` identity bug — it bit THREE fields in one day

`_solo_key` decides whether an existing baseline can be reused. It lost, in
sequence: `kv_budget_gb`, then both SGLang flags. Each time, `--resume`
silently reused a baseline from a *different deployment* — e.g. p50's
neighbour (1.28 GiB) matching p25's baseline (0.64 GiB), so every p50 ratio
would have divided by a reference at half its cache.

Three gotchas, all real:

1. **There are TWO `_solo_key` functions** — one in `scenario_config.py`, one
   in `coloc.py`. `find_existing_baseline` uses the *coloc* one. Fixing only
   one leaves the bug live, and because both were 8-tuples whose last field
   differed, nothing raised.
2. **`solo_key_from_manifest` must match field-for-field.** Different lengths
   fail to match (harmless re-runs); the same length with different fields
   matches the WRONG baseline.
3. **Flags injected in `build_server_cmd` never reach the key** — that is how
   the SGLang flags went missing. `launch_args` from the yaml is in the key;
   anything the orchestrator adds is not.

**Recommended structural fix (not yet done):** key baselines on the *resolved
launch command* rather than an enumerated field list. Three fields have now
gone missing in one day; enumeration does not hold.

---

## 5. Backend gotchas

### SGLang needs BOTH fixes to colocate

- `import sglang` fails outright if `kernels` is installed alongside
  transformers 5.6.0 (`ValueError: Either a revision or a version must be
  specified`). `pip uninstall kernels`.
- FlashInfer JIT-compiles kernels and shells out to **`ninja`**, which lives in
  `.venv-sglang/bin` and is not on the orchestrator's PATH. The server dies
  15 s in with `FileNotFoundError: 'ninja'`, before loading weights.
  `build_server_env` now puts the tenant's venv `bin` on PATH.
- `--max-total-tokens` alone is **not** enough. `--mem-fraction-static` still
  gates the overall allocation and a fraction derived from the tenant's own
  budget is blind to neighbours (needed ≥0.94, was given 0.49). Once the token
  count is the real control, the fraction must be permissive —
  `SGLANG_PERMISSIVE_MEM_FRACTION = "0.95"`.
- `--max-total-tokens` is in **tokens**, so models carry `kv_bytes_per_token`
  (`layers × kv_heads × head_dim × 2 × 2`). Verified against vLLM's own
  arithmetic to within 0.06%.

### vLLM / model gotchas

- **`gemma-4-31b-it-fp8` cannot be served.** vLLM registers the arch but
  transformers 4.57.6 does not know `gemma4`. Pinned out. Do not bump
  transformers under a running study — vLLM pins it and every other tenant
  works against 4.57.6.
- **The customer's second VLM cannot do video.** `gemma-vlm-32b` is
  `paligemma2-28b`, image-only and gated, so it cannot serve `vlm_video_long`
  at all — the config already declares that unsupported. `same-vlm` now uses
  `qwen3-vl-32b-fp8`.
- **`triton_backend: python` is only valid for `kosmos-2.5`.** Only it has a
  hand-authored `model.py`; `yolov8-l` fails at staging.
- **vLLM refuses to start unless the cache holds one max-length sequence.** At
  8192 context that is 2.50 GiB for the 72B and 0.44 GiB for the 7B. A rung
  below that floor cannot load even alone on an empty card.

---

## 6. Operational lessons

- **`pgrep` matches your own shell.** Its command line contains the pattern you
  are searching for. A broad `pkill -f "bench coloc"` killed the session's own
  shell. Always list PIDs, exclude `$$`, and kill explicitly.
- **`VLLM::EngineCore` outlives its parent.** Killing the orchestrator and the
  `vllm` process can leave a child holding tens of GB. `pgrep` will not show it
  usefully; **`nvidia-smi --query-compute-apps=pid,used_memory,process_name`
  names it directly.** An orphan holding the `EXCLUSIVE_PROCESS` context makes
  the *next* run die with `CUDA-capable device(s) is/are busy or unavailable`.
  Always drain the GPUs before relaunching.
- **Do not run two orchestrators concurrently**, even on different GPUs. Ports
  do NOT stride by device (both resolve to 8000/8001/8100) and both write to
  the same `_baselines` directory. The 404 cross-talk this causes is already
  documented in `_assign_distinct_ports`.
- **Rate limits are per-tenant, not per-pair.** A shared `rps_sweep: "*"` only
  makes sense when both tenants have comparable ceilings. `same-vlm`'s were 12×
  apart, so three of its four rungs sat above the 32B's limit and collapsed
  into one load point measured four times.
- **`repetitions: 3` is only justified near a cliff.** Away from one, 12
  contention runs returned identical values to two decimals. Near the eviction
  limit, two identical reps gave 0.79 and 0.40 — there, the spread IS the
  finding.

---

## 7. Open items

| Item | Where |
|---|---|
| Apply `prefix_off` and re-run the deployment family (~1.3 h) | `docs/next-run/config-changes.md` §1e |
| Add prompt diversity (20-50 per workload) — affects the whole study | §3 above |
| Key baselines on the resolved launch command, not a field list | §4 above |
| `cross-deploy` has no *fullness* sweep — all splits sit at 95% | `docs/next-run/config-changes.md` |
| Cannot attribute loss between compute contention and cache shrinkage | needs a fixed-cache arm alongside the derived-cache one |
| Rates across the study are far below capacity (7B sustains 62 of 64 rps) | `docs/contention-phases.md` |
