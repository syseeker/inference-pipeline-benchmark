---
name: gpu-contention-benchmark
description: |
  Measure how co-resident models degrade each other on one GPU — text LLM,
  video VLM, image-LM and computer vision sharing the same card. Use when
  the user asks about noisy neighbours, co-location, multi-tenant GPU
  capacity planning, safe-operating envelopes, or "can I run these two
  models on one GPU". Covers the hybrid serving topology (Triton for CV,
  native vLLM/SGLang/TRT-LLM for LLM/VLM), the two-driver load generation
  rule, and the open-loop measurement discipline that makes degradation
  ratios meaningful.
---

# gpu-contention-benchmark

## When to invoke

- "What happens to my LLM's latency if I also run object detection on this GPU?"
- "How many models can I co-locate before p95 breaks?"
- "Which of these two models is the noisy neighbour?"
- "At what request rate does my CV pipeline start hurting the VLM?"
- Capacity planning for multi-tenant inference nodes.

**Do not** invoke for single-model benchmarking — that is
[benchmark-gpu-inference](../benchmark-gpu-inference/SKILL.md). This skill is
specifically about **two or more models resident at the same time**.

## Build status

This capability is under construction. **This table is the handoff record** —
read it first, update it as you complete a step, and commit the change.

| # | Step | State |
|---|---|---|
| 1 | Customer brief + test data staged (`workspace/contention/`) | ✅ done |
| 1b | Design decisions recorded (`reference/`) | ✅ done |
| 2 | Video clips → H.264 at spec | ✅ done (`clip_3s_224.mp4` 224²/3f, `clip_10s_720p.mp4` 720p/40f, both H.264) |
| 3 | Phase-0 concurrency probe + clock policy | ✅ built (`scripts/gpu_concurrency_probe.py` + unit tests). Validated on PRO 6000: MPS off → 0.28× (serialises, gate FAIL); MPS on → 1.94× (gate PASS); variance CoV 0.4% → 1 rep/scenario. Clock pinning itself stays a pre-flight step; the probe records clocks + fails on throttle |
| 4 | Co-tenancy result schema + per-request timestamps | ✅ done |
| 5 | `colocations:` config schema | ✅ done (rtx_pro6000; 5090/H200 pending) |
| 6 | `bench coloc` orchestrator | ✅ HTTP path live-validated (`benchmarks/coloc.py`, `bench coloc`, 27 unit tests). End-to-end solo run on PRO 6000 via config: orchestrator launched the server, held t0, ran one DCGM sampler, aiperf-streamed 58 reqs → `llm.ndjson` (epoch ts + TTFT + ITL), achieved_rps 4.02 vs offered 4.0, no throttle. Pinned the real aiperf `{metadata,metrics}` schema. **Note:** contention tenants need `--streaming` (done) and a vllm backend **variant without the `--gpu-memory-utilization=0.90` pin** in `extra_args`, else per-tenant caps can't take effect. Triton CV tenant server side is step 7 |
| 7 | Triton CV tenants | ✅ done (`benchmarks/triton_cv.py`: CVModelSpec registry, config.pbtxt builder, Triton model-repo layout, perf_analyzer wrapper; `scripts/build_triton_cv_repo.py`: ONNX + TRT export paths; `coloc.py`: Triton lifecycle, CSV parsing, docker-inspect mutex. 75 unit tests pass) |
| 8 | Contention analysis (summary §10, `align_traces.py`) | ✅ done (`scripts/align_traces.py`: aligns all tenant traces to t0, computes overlap window + per-tenant stats; `benchmarks/summary.py` §10: degradation table, contention matrix, safe-operating-envelope section. 92 unit tests pass) |

### Continuing on another machine

Everything needed to resume is committed to the repo — no local state is
required. On a fresh clone:

1. Read the table above to see where the build stopped.
2. Read [reference/build-plan.md](reference/build-plan.md) §8 for the full build
   sequence, and the section covering your step for the design detail.
3. Read [reference/design-decisions.md](reference/design-decisions.md) **before**
   writing measurement code — the open-loop rule and sampler ownership are easy
   to get wrong and invalidate results silently.
4. Test data is already in `workspace/contention/` (checked in, not ignored).

## Two rules that are non-negotiable

**1. Open-loop load, always.** Drive every tenant at a fixed *request rate*, never
at a fixed *concurrency*. A closed-loop client throttles itself in proportion to
the slowdown it is supposed to be measuring, so the degradation ratio ends up
describing the harness rather than the GPU. Record `offered_rps` and
`achieved_rps`; the point where they diverge is the safe-operating-envelope
boundary. Full reasoning in [reference/design-decisions.md](reference/design-decisions.md).

**2. One driver per tenant type.** AIPerf cannot drive Triton — it dropped
GenAI-Perf's `kserve` and `dynamic_grpc` endpoint types.

| Tenant | Driver | Open-loop flag | Arrival pattern flag |
|---|---|---|---|
| LLM / VLM (vLLM, SGLang, TRT-LLM) | `aiperf` | `--request-rate` | `--arrival-pattern {constant,poisson,gamma}` |
| CV (Triton) | `perf_analyzer` | `--request-rate-range` | `--request-distribution {constant,poisson}` |

Both support Poisson open-loop, so the two tenants remain comparable. The
orchestrator's job is to start them together against a shared `t0` and merge
their traces — not to generate load itself.

## Serving topology

Hybrid, and deliberately so — see [reference/serving-topology.md](reference/serving-topology.md).

- **CV models → Triton** (`26.07-py3`), TensorRT backend for the optimised path,
  ONNX Runtime for the portable baseline, Python backend for models with no
  export path (kosmos-2.5, PaddleOCR).
- **LLM / VLM → native servers** (vLLM, SGLang, TRT-LLM). They run their own
  process with their own scheduler; putting them behind Triton buys nothing.
- **Isolation is a fixed setting, not a swept dimension.** MPS on everywhere.
  MIG exists only on RTX PRO 6000 and is used for a single hardware-isolated
  reference run.

## Recipe

Phases follow the customer's own structure in
`workspace/contention/experiment_design.md`.

> **First session on real hardware?** The harness was built and unit-tested
> without a GPU. Work through
> [docs/contention-gpu-validation.md](../../docs/contention-gpu-validation.md)
> before trusting any number it produces — it lists every assumption made
> without hardware, ordered by how much work each one invalidates if wrong.
> The weight estimates that set every VRAM cap are top of that list.

```bash
# Phase 0 — GATE. Do this first and stop if it fails.
python scripts/gpu_concurrency_probe.py --gpu rtx_pro6000 --json
#   Confirms tenants genuinely overlap on the GPU rather than time-slicing.
#   Also measures run-to-run variance (5x) — that sets the repetition policy.

# Phase 1 — solo baselines, at the SAME offered rate as the contention runs
bench coloc --gpu rtx_pro6000 --colocation mix-llm-cv --solo-only

# Phases 2-6 — named colocations from the GPU yaml
bench coloc --gpu rtx_pro6000 --colocation cross-llm-vs-cv     # rate sweep
bench coloc --gpu rtx_pro6000 --colocation mix-full            # all 4 categories

bench summary --gpu rtx_pro6000                                # §10 = contention
python scripts/align_traces.py benchmarks/results/rtx_pro6000/coloc/<run_id>/
```

## Pre-flight checks (do not skip)

1. **VRAM budget.** `sum(tenant gpu_memory_utilization) + CV footprint <= 1.0`.
   vLLM defaults to `0.9` and will take the whole card, starving tenant 2.
2. **Clock policy applied.** Power limit first, then `nvidia-smi -lgc` at 60–80%
   of max boost. On GeForce the lock is *advisory* — verify, don't assume.
3. **No throttle reasons active.** Abort if `clocks_throttle_reasons.active`
   shows `SwPowerCap` or `HwThermalSlowdown`; a throttled run is not a
   contention measurement.
4. **Video clips are H.264.** `mp4v` (MPEG-4 Part 2) risks a CPU decode fallback,
   which turns a "GPU video tenant" into a partly-CPU tenant.
5. **Triton client shared memory enabled** — `--allow-client-shm=true`. Disabled
   by default since Triton 26.04; without it, large CV tensors read as a model
   regression when it is really serialization overhead.
6. **One GPU sampler for the whole window**, not one per tenant. N samplers means
   N `dcgmi dmon` processes and every tenant reporting whole-GPU memory as its own.

## Failure recovery

| Symptom | Cause | Action |
|---|---|---|
| Phase 0 shows ~1.0× aggregate throughput, ~2.0× latency | Tenants are **serialising**, not sharing | Stop. Enable MPS and retry. If still serialised, the study measures time-slice fairness — rescope and tell the user |
| Tenant 2 OOMs at startup | Tenant 1 took the whole card | Set explicit `gpu_memory_utilization` per tenant |
| `achieved_rps` << `offered_rps` even solo | The driver is the bottleneck, not the GPU | Lower the rate, or check the client host isn't CPU-saturated |
| Degradation ratio ≈ 1.0 everywhere | Load too low to contend, or closed-loop crept back in | Raise offered rate; confirm `--request-rate` is set, not `--concurrency` |
| Ratio varies wildly between repeats | Near-OOM KV-cache eviction is bimodal | Expected at the memory-pressure points — report both modes, not the mean |
| CV latency varies with image content | NMS is data-dependent | Expected for YOLOv8; hold the input fixed so the variance you measure is contention |

## Verification

- **Null test** — a "colocation" of one tenant alone must give ratio ≈ 1.0. If a
  tenant degrades against no neighbour, the harness is wrong, not the GPU.
- **Load fidelity** — `achieved_rps ≈ offered_rps` at low load.
- **Sampler sanity** — exactly one sampler process per run.
- **Clock integrity** — no published run had a throttle reason fire.

## Pinned references

- [docs/contention.md](../../docs/contention.md) — **start here**: what a degradation ratio means, why the solo baseline must be handicapped, how the four model types contend, and the 1/2/4-GPU topologies. Written for customers too
- [reference/design-decisions.md](reference/design-decisions.md) — the methodology and *why*: open-loop, clock policy, sampler ownership, timestamps, repetition policy
- [reference/model-catalogue.md](reference/model-catalogue.md) — verified model sources, per-GPU scoping, and which picks are broken
- [reference/serving-topology.md](reference/serving-topology.md) — vLLM vs Triton vs MPS vs MIG, explained from first principles
- `workspace/contention/experiment_design.md` — the customer's original brief
- `workspace/contention/experiment_config.json` — model catalogue, prompts, phase definitions
- [docs/metrics.md](../../docs/metrics.md) — per-request metric definitions, reused unchanged
