# Contention harness — what to validate when the GPU comes back

The contention harness was built and unit-tested **without a GPU**. 200+ tests
cover the pure logic, but a passing test suite cannot tell you whether a
weight estimate is right, whether a Docker flag is spelled correctly, or
whether a cap that looks fine on paper actually loads.

This is the list of everything assumed rather than measured, ordered by **how
much work it invalidates if it turns out to be wrong**. Work down it; do not
start at the interesting end.

Written 2026-08-03, at the point where all seven customer phases had
colocations defined and nothing had ever run on hardware.

---

## Tier 1 — does anything run at all

Cheap, fast, and everything below depends on them.

### 1.1 Phase 0 gate, re-run

```bash
python3 scripts/gpu_concurrency_probe.py --gpu rtx_pro6000 --json
```

Already passed once (MPS off → 0.28×, MPS on → 1.94×, CoV 0.4%). Re-run it
because the machine may not be the same one. **If it fails, stop** — every
number below becomes a statement about the time-slice scheduler rather than
about contention.

Also run the variant that was never run: **VLM prefill + CV**. It is the
sharpest contention pair in the study and the one most likely to expose a
scheduling surprise.

### 1.2 A single colocation, end to end

```bash
bench coloc --gpu rtx_pro6000 --colocation mix-llm-cv
```

The only path with any live validation behind it is a *solo* HTTP run. A
two-tenant window with a real Triton container has never executed. Expect to
debug here rather than at the far end of a sweep.

### 1.3 The null test — the harness checking itself

```bash
bench coloc --gpu rtx_pro6000 --colocation place-isolated
```

Two tenants, one per card. No shared SMs, bandwidth or VRAM, so **the
degradation ratio must come back ≈ 1.0**. If a tenant degrades against a
neighbour on a *different* GPU, the harness has a bug and nothing else it
reports can be trusted. This is the single highest-value check in the list.

---

## Tier 2 — numbers that were estimated, and silently set everything else

These do not fail loudly. They produce plausible-looking results that are
wrong, which is worse.

### 2.1 `weights_gb` — the estimates that set every derived cap

Caps are derived as `(weights_gb + kv_budget_gb + overhead) / vram_gb`. The
weight figures in `benchmarks/configs/rtx_pro6000.yaml` are mostly
**param-count × 2** (BF16 arithmetic) or lifted from free-text `notes:`. None
was measured.

The runner already records **`vram_after_load_gb`** — that is the ground
truth. After the first runs, compare and correct:

| Model | `weights_gb` assumed |
|---|---|
| qwen2.5-7b | 15.2 |
| qwen2.5-14b | 29.5 |
| qwen2.5-32b | 65.5 |
| qwen2.5-72b | 45.0 (AWQ) |
| qwen2.5-vl-7b | 7.0 (AWQ + FP16 vision tower) |
| gemma2-9b | 18.4 |
| llama3.1-8b | 16.1 |
| mistral-7b | 14.5 |

If these are off, **every derived cap is off**, and with it every KV cache and
every ratio. Correcting them is a config edit, not a code change.

`gemma-4-31b-it-fp8` has no `weights_gb` entry at all — its cap (0.51) is
hand-written. Check it explicitly.

### 2.2 The three tightest caps

Ordered by likelihood of failing to load:

| Where | Cap | Reservation | Concern |
|---|---|---|---|
| `cross-memory-pressure-kv29` | 0.69 + 0.28 = **0.97** | ~93 of 96 GB | Deliberately near-OOM. It is *supposed* to be at the edge — but confirm it loads at all, else the top of the curve is missing |
| `cross-size-scaling` 32b rung | **0.87** | ~83.5 GB | The tightest single tenant. Leaves ~12.5 GB for the CV container. Fails if 65.5 GB is an underestimate |
| `mix-memory-bound` | 0.55 + 0.39 = **0.94** | ~90 GB | Leaves ~5.8 GB for Triton + driver. Is that enough for yolov8-l? |

`preflight_vram` does not count the Triton footprint — it is treated as
headroom. Once you have real numbers, check the headroom assumption holds.

### 2.3 Is 16 GB actually a sensible KV budget?

`cross-size-scaling` holds KV at 16 GB across the ladder, and the placement
pairings hold it constant too. That figure was chosen so the 32B rung would
fit, not from a measurement of what the workload needs. If 16 GB throttles
throughput at the offered rates, the whole ladder is measuring KV starvation
rather than model size. Check `achieved_rps ≈ offered_rps` on the solo
baselines before believing any ratio.

---

## Tier 3 — commands and syntax that have never executed

All of this is asserted in unit tests as *strings*. No test has run Docker.

| Assumption | Where | How to check |
|---|---|---|
| `--gpus device=N` pins one card | `triton_cv.build_triton_serve_cmd` | Launch two containers, confirm each sees one GPU |
| Ports `base + 10*device` (8100 / 8110) | `triton_cv.triton_ports` | Two live containers, no bind conflict |
| `--model-control-mode=explicit` + `--load-model=` | same | Each container loads *only* its own models |
| Symlinking staged weights into another device's repo | `coloc._link_staged_weight` | GPU 1's container loads a model exported once into GPU 0's repo |
| `perf_analyzer -u localhost:8110` | `coloc.triton_tenant_url` | Driver reaches the second container |
| `CUDA_VISIBLE_DEVICES` pins vLLM tenants | `coloc.build_server_env` | `nvidia-smi` shows the process on the intended card |
| The cap override actually reaches vLLM | `coloc._override_flag` | **Confirm the server logs the tenant's cap, not 0.90.** This was a live bug; the fix is unverified against a real vLLM |

That last one matters most. The old code silently dropped the per-tenant cap
so both tenants launched at 0.90 and the second OOMed — while the pre-flight
called the plan fine. The fix is correct in tests; confirm vLLM honours it.

---

## Tier 4 — hardware facts assumed from the config file

### 4.1 Interconnect

```bash
nvidia-smi topo -m
```

`rtx_pro6000.yaml` records `nvlink: false`. If true, cross-GPU traffic is
PCIe Gen5 and **tensor-parallel results would be dominated by interconnect
rather than by contention** — which is why no TP colocations were written.
Confirm before adding any. If NVLink is in fact present, TP becomes worth
measuring and that decision should be revisited.

### 4.2 MPS with several containers

Phase 0 validated MPS with one Triton container. The placement study runs
**two** containers plus two vLLM processes against one MPS control daemon.
Confirm the pipe is shared correctly and that Phase 0's 1.94× overlap still
holds under that shape.

### 4.3 Clocks

Pin power limit first, then `nvidia-smi -lgc` at 60–80% of max boost. Confirm
no `clocks_throttle_reasons.active` fires during a run — a throttled run is
thermodynamics, not contention, and must not be published as a finding.

---

## Tier 5 — experiments whose *outcome* is a prediction

Not correctness checks. These are the study's actual hypotheses, recorded
here so the prediction is on the record before the measurement.

### 5.1 The placement heuristic

`docs/contention.md` §3 argues you should pair tenants that stress *different*
resources. Applied to the three 2-GPU pairings, that predicts:

```
P1  [LLM+VLM] | [ILM+CV]    best   — separates the two compute-heavy models
P3  [LLM+CV]  | [VLM+ILM]   worse  — VLM and ILM both compute-heavy
P2  [LLM+ILM] | [VLM+CV]    worst  — VLM's burst lands on the most fragile tenant
```

If the measured ranking matches, the customer gets a placement rule that
generalises to models we never tested. **If it does not match, the resource
model in contention.md §3 is wrong and that doc needs revising** — which is a
more interesting result than confirmation.

### 5.2 Bimodality under memory pressure

`cross-memory-pressure-kv29` carries `repetitions: 3` on the theory that near-OOM
behaviour is bimodal — the model either fits or thrashes, so the mean of the
two describes neither. Check whether the three repeats actually separate into
modes. If they are tightly clustered, the repetition can be dropped and the
reasoning in the config is wrong.

### 5.3 Do same-category pairs really fight hardest?

The four `same-*` colocations test contention.md §3's claim that two models
stressing the same resource contend worse than a mixed pair. Compare
`same-llm` against `mix-llm-cv` at equivalent load. Never tested — every
colocation before these was cross-category.

---

## Tier 6 — design questions deliberately left open

Not bugs. Decisions that need information we do not have yet.

**Tensor-parallel width has two owners.** `device: [0,1]` says which cards a
tenant occupies, while `backends.<b>.variants` carries
`--tensor-parallel-size=2` and a top-level `tensor_parallel:` key exists too.
Nothing reconciles them, so they can silently disagree. Deferred rather than
guessed. Once TP is worth running (see 4.1), decide whether `device` owns the
width and the variant is validated against it, or the reverse.

**Per-tenant Triton ports.** The port-offset scheme assumes all Triton tenants
share one backend-wide base port. If a config ever gives them different bases,
two tenants on one card would disagree about the port and the first to launch
would win. Not a problem today; add a schema check if that changes.

**Phase 5 is 2 GPUs only.** The customer's design sweeps `gpus: [2, 4]`. The
4-GPU arm is dropped by decision. The `device:` schema still supports up to 8,
so restoring it is a config change, not a code change.

---

## Reporting back

Anything corrected here should land in the repo, not in a chat log:

- Wrong `weights_gb` → fix the yaml, note the measured value in the comment
- A failed assumption in Tier 3/4 → fix the code, add the regression test
- A hypothesis in Tier 5 resolved → write it up in `docs/findings/` (the
  summary generator auto-links from Core findings) and update
  `docs/findings/knowledge.yaml`
- A revised design decision → update
  `skills/gpu-contention-benchmark/reference/design-decisions.md`, which is
  the record of *why*, and say what changed
