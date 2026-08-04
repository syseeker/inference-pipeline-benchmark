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

## This document is meant to be deleted

It is a temporary record of an unusual situation — a fully built harness that
has never touched the hardware it was built for. Once that is no longer true,
it should not exist.

**How an item closes.** Not by ticking it. By landing its answer somewhere
durable:

| What you found | Where it goes |
|---|---|
| A wrong `weights_gb` | Fix `benchmarks/configs/rtx_pro6000.yaml`; put the measured number in the comment |
| A broken assumption in Tier 3/4 | Fix the code **and add the regression test** — a hardware bug with no test comes back |
| A hypothesis in Tier 5 resolved | Write it up in `docs/findings/`, add to `docs/findings/knowledge.yaml` (the summary generator reads it) |
| A design decision revised | Update [design-decisions.md](design-decisions.md), and say what changed and why |
| A Tier 6 question settled | Same — it is a decision, so it belongs with the decisions |

**Then delete this file and the pointer to it in
[SKILL.md](../SKILL.md)**, in the same commit that closes the last item.

**Do not delete it early to tidy up.** The value here is the corrections, not
the checklist. An item deleted before its answer is written down silently
reverts to being an unverified assumption — except now nobody knows it is one,
which is strictly worse than the situation this document was written to fix.

Everything here is one-time. Recurring per-run checks live in SKILL.md's
"Pre-flight checks" section and stay there permanently.

---

## Tier 1 — does anything run at all

Cheap, fast, and everything below depends on them.

### 1.1 Phase 0 gate, re-run — ✅ ANSWERED 2026-08-04

```bash
python3 scripts/gpu_concurrency_probe.py --gpu rtx_pro6000 --json
```

Re-run on a 2× PRO 6000 box (driver 595.71.05): **PASS. Overlap 2.07× at
0.95× latency, CoV 1.8% → `recommended_reps: 1`.** Better than the original
1.94×. Note `solo_gpu_util_p50` was 46%, so the probe had headroom — 2.07×
says the tenants genuinely overlap, not that real models will not contend.

The repetition policy for the study is therefore **1 rep**, except
`cross-memory-pressure-kv29`, which keeps `repetitions: 3` deliberately
(bimodality, see 5.2).

This run also exposed the bug in the probe's own MPS detection: it used
`pgrep -x`, which matches comm, truncated by the kernel to 15 chars, so the
23-char daemon name never matched and a good MPS run recorded
`isolation: "none"`. Fixed to `pgrep -f`, with a regression test.

Still not run: the **VLM prefill + CV** variant — the sharpest contention pair
in the study and the one most likely to expose a scheduling surprise.

### 1.2 A single colocation, end to end — ✅ ANSWERED 2026-08-04

```bash
bench coloc --gpu rtx_pro6000 --phase 3 --resume
```

"Expect to debug here rather than at the far end of a sweep" was right. Phase 3
took **four attempts and nine distinct bugs**, most invisible until an earlier
one was fixed — the Triton container-reuse bug hid a broken dinov2 engine,
`--rm` hid the error message that would have named it, `int(rps)` hid kosmos
serving nothing, and a backend-wide port default hid two vLLM tenants answering
each other's traffic with HTTP 404s.

All 12 runs now complete clean. What it established:

- `achieved_rps` 3.91 against 4.0 on the LLM — the driver is not the bottleneck.
- `gpu_memory_utilization: 0.45` in the manifest — **the per-tenant cap override
  does reach vLLM**, not silently 0.90.
- Pairs are cheap and four-way is not: every two-tenant pairing costs under 11%
  on end-to-end p95, while all four on one card costs **1.64× to 2.88×**.
  Contention is not additive in tenant count.
- The victim ranking inverts intuition: the smallest, fastest tenant
  (`yolov8-l`, 7 ms requests) suffers most at 2.88×, the largest least at 1.64×.
  A short request has no slack to absorb queueing.

Every bug found here has a regression test; see the commits on
`contention/first-hardware-run`.

### 1.3 The null test — ✅ ANSWERED 2026-08-04, TWICE

```bash
bench coloc --gpu rtx_pro6000 --colocation place-isolated
```

**`place-isolated` returned 1.02× worst-tenant, 1.01× mean.** Two tenants, one
per card, nothing shared — the harness does not manufacture degradation.

A second, sharper null test came free with the phase:
**`place-vlm-prefill-split` returned 1.00× and 0.99×** — a 40-frame video
prefill burst on GPU 0 has no measurable effect on an LLM decoding on GPU 1. On
a PCIe box the interconnect is not a hidden coupling channel either.

It failed on its first attempt, for a real reason worth recording: the second
card's repo symlinked its weights to the staging repo's absolute host path,
which dangles inside a container that mounts only its own repo. Hard-linked now
— see the Tier 3 table.

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

| Assumption | Where | State |
|---|---|---|
| The cap override actually reaches vLLM | `coloc._override_flag` | ✅ **confirmed.** Solo LLM baseline manifest records `gpu_memory_utilization: 0.45`, not 0.90 |
| MPS detection | `coloc.capture_mps` | ✅ **confirmed** — `environment.mps.detected: true`. But see 3.1: detection was never the hard part |
| Container joins MPS | `triton_cv.build_triton_serve_cmd` | ❌ **was broken, now fixed** — see 3.1 |
| `--model-control-mode=explicit` + `--load-model=` | same | ✅ one container loads only `yolov8-l`; `READY` in ~2 s |
| `--gpus device=N` pins one card | same | ✅ **confirmed** — `place-p2` ran two containers, one per card, on DIFFERENT images (derived for kosmos on GPU 0, stock for yolov8 on GPU 1) |
| One sampler per occupied card | `coloc.occupied_devices` | ✅ **confirmed** — every `place-*` manifest carries `gpu_sampler` keys `"0"` AND `"1"` |
| `nvidia-smi topo -m` capture | `coloc.capture_interconnect` | ✅ see 4.1 |
| Ports `base + 10*device` (8100 / 8110) | `triton_cv.triton_ports` | ✅ **confirmed** — both bound simultaneously, no conflict |
| Staging weights into another device's repo | `coloc._link_staged_weight` | ❌ **was broken, now fixed.** A symlink to the staging repo's absolute host path dangles inside a container that mounts only its own repo — "Failed to determine modification time for '/models/yolov8-l/1/model.plan'". Hard-linked now, siblings included (`model.onnx.data`) |
| `perf_analyzer -u localhost:8110` | `coloc.triton_tenant_url` | ✅ **confirmed** — drove the GPU 1 container |
| `CUDA_VISIBLE_DEVICES` pins vLLM tenants | `coloc.build_server_env` | ✅ **confirmed** — and the per-card baselines differ measurably (yolov8-l: 50.55 on GPU 0, 49.17 on GPU 1), which is why `_solo_key` carries the device |

### 3.1 The two MPS bugs, and why the tests could not see them

Recorded because the *shape* of them generalises: both were in the seam
between a correct function and its caller, which is exactly what a unit suite
of pure functions cannot reach.

**The CV tenant never joined MPS.** `build_triton_serve_cmd` had always
accepted `mps_pipe_dir`, and a unit test asserted it honoured it — but
`_ensure_server` never passed it. So every CV tenant ran on its own context
and time-sliced against the LLM. Nothing warned: `capture_mps()` inspects the
*host* daemon, which is genuinely running, so the manifest said
`mps.detected: true`. New field `environment.mps.container_pipe_directory`
now records what the container was actually given; `null` means it could not
have joined.

**Then passing the pipe made Triton fail outright.** The image runs as root,
MPS servers are per-UID, and a non-root control daemon cannot spawn one for a
different UID. Every model came back `UNAVAILABLE: unable to get number of
CUDA devices`, `cuInit` → 805. Fixed with `--user <uid>:<gid>`. `--ipc=host`
was tried and is **not** required. `--pid=host` does not help.

The lesson for the remaining ⬜ rows: a passing assertion about the *string*
a builder returns says nothing about whether the caller passes the argument,
and nothing about whether the flag is sufficient. Verify at the seam.

**`scripts/check_mps_clients.sh`** exists now for exactly this and should be
run during any window whose ratios matter. `nvidia-smi` cannot answer the
question: on Volta and later each MPS client keeps its own address space and
lists as its own process, so separate `vllm` and `tritonserver` entries are
what a *correctly shared* GPU looks like.

---

## Tier 4 — hardware facts assumed from the config file

### 4.1 Interconnect — ✅ ANSWERED 2026-08-04

```bash
nvidia-smi topo -m
```

**`PIX` between GPU0 and GPU1 — no NVLink.** The yaml's `nvlink: false` is
correct, and it is now measured rather than assumed. Cross-GPU traffic is
PCIe Gen5 x16, so tensor-parallel results here would be dominated by the
interconnect rather than by contention: **the decision to write no TP
colocations stands.** Revisit only on a box where `topo -m` shows `NV#`.

Also on the record for this host, since neither is reconstructable later and
both bear on published numbers:

- **ECC is enabled** on both cards. It costs some effective bandwidth. Fixed
  and consistent, so no ratio is distorted — but `dinov2-base` is the
  bandwidth aggressor and its absolute figures are ECC-on figures.
- **CUDA forward compatibility is active** in the Triton container (CUDA 13.3
  via the compat shim against a 595.71.05 kernel driver). Supported, and it
  works, but it applies to every CV tenant for the whole study.
  `capture_environment()` does not record either of these today.

### 4.2 MPS with several containers — ✅ ANSWERED 2026-08-04

`place-p2` and `place-p3` each ran **two Triton containers plus two vLLM
processes against one MPS control daemon**, under `EXCLUSIVE_PROCESS`, with the
two containers on different images. All four were MPS clients; no run recorded a
no-MPS warning and none failed to get a context. Phase 5 completed 15/15 with
zero errors and zero warnings.

### 4.3 Clocks

Pin power limit first, then `nvidia-smi -lgc` at 60–80% of max boost. Confirm
no `clocks_throttle_reasons.active` fires during a run — a throttled run is
thermodynamics, not contention, and must not be published as a finding.

**Not yet applied on this host.** The Phase 0 probe reported no throttle
reasons at stock clocks (600 W limit, 2430 MHz max boost, 27 °C idle), but
that is not the same as having pinned them.

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

**ANSWERED 2026-08-04 — it did not match, and the doc has been revised.**
Measured worst-tenant end-to-end p95:

```
P2  [LLM+ILM] | [VLM+CV]    1.46x   predicted WORST, measured BEST
P1  [LLM+VLM] | [ILM+CV]    1.95x   predicted best
P3  [LLM+CV]  | [VLM+ILM]   2.19x
```

The annotation above is where it went wrong: P1 is labelled "separates the two
compute-heavy models" but P1 *groups* the LLM and VLM. Those two are the pair
that must not share a card — both autoregressive, both KV-hungry, contending for
the same resource rather than different ones. The VLM pays 1.95x in P1 and 1.02x
once split.

The second-order rule is backwards from the intuition in §3 too: a small fast
tenant fares WORSE beside a steady neighbour than a bursty one. CV with the LLM
is 2.19x; CV with the VLM is 1.46x, because the VLM's prefill bursts leave gaps
the CV tenant can use.

Both rules and the numbers are now in `docs/contention.md` §3, with the caveat
that they come from a load point where the LLM was bandwidth-saturated and the
second card near idle.

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

See [This document is meant to be deleted](#this-document-is-meant-to-be-deleted)
at the top — each finding has a durable home, and this file goes away once they
have all reached it.
