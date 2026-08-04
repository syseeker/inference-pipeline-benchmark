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

**Built; hardware validation started 2026-08-04 on 2× RTX PRO 6000.** All
eight build steps are complete: 39 colocations covering all seven phases, 357
unit tests, no VRAM pre-flight issues.

Validated so far: the Phase 0 gate (2.07× overlap, CoV 1.8% → 1 rep), the
solo LLM baseline, the per-tenant VRAM cap genuinely reaching vLLM (0.45, not
0.90), a Triton CV container loading and joining MPS, and `nvlink: false`.
**Not yet: any two-tenant contention window**, and therefore no degradation
ratio has ever been produced.

First hardware contact found eight bugs that the unit tests could not reach —
the CV tenant never joined MPS, and fixing that exposed a second failure where
it then could not initialise CUDA at all. Both were in the seam between a
correct function and its caller. Details and the remaining open items:
[reference/gpu-validation.md](reference/gpu-validation.md).

**This table is the handoff record** — read it first, update it as you complete
a step, and commit the change.

| # | Step | State |
|---|---|---|
| 1 | Customer brief + test data staged (`workspace/contention/`) | ✅ done |
| 1b | Design decisions recorded (`reference/`) | ✅ done |
| 2 | Video clips → H.264 at spec | ✅ done (`clip_3s_224.mp4` 224²/3f, `clip_10s_720p.mp4` 720p/40f, both H.264) |
| 3 | Phase-0 concurrency probe + clock policy | ✅ built + **re-validated on hardware 2026-08-04**: overlap 2.07× at 0.95× latency, CoV 1.8% → 1 rep/scenario, no throttle. (Earlier reference: MPS off → 0.28× gate FAIL; MPS on → 1.94×.) Fixed here: `pgrep -x` could never match the 23-char daemon name (comm truncates to 15), so a good MPS run recorded `isolation: "none"`. Clock pinning stays a pre-flight step and is **not yet applied on this host** |
| 4 | Co-tenancy result schema + per-request timestamps | ✅ done |
| 5 | `colocations:` config schema | ✅ done (rtx_pro6000; 5090/H200 pending) |
| 6 | `bench coloc` orchestrator | ✅ HTTP path live-validated (`benchmarks/coloc.py`, `bench coloc`, 27 unit tests). End-to-end solo run on PRO 6000 via config: orchestrator launched the server, held t0, ran one DCGM sampler, aiperf-streamed 58 reqs → `llm.ndjson` (epoch ts + TTFT + ITL), achieved_rps 4.02 vs offered 4.0, no throttle. Pinned the real aiperf `{metadata,metrics}` schema. **Note:** contention tenants need `--streaming` (done) and a vllm backend **variant without the `--gpu-memory-utilization=0.90` pin** in `extra_args`, else per-tenant caps can't take effect. Triton CV tenant server side is step 7 |
| 7 | Triton CV tenants | ✅ built, and **one container live-validated 2026-08-04**: `yolov8-l` READY in ~2 s, loaded via `--model-control-mode=explicit`, confirmed an MPS client. Needed two fixes invisible to the tests — `coloc` never passed `mps_pipe_dir` (CV ran outside MPS, silently), and the container must run `--user <uid>:<gid>` because MPS servers are per-UID and the image is root. CV plans also now build at the yaml's declared fp16; `trtexec --fp16` is gone in TRT v11 (strongly typed), so precision comes from the ONNX. **Two containers / placement still untested** |
| 8 | Contention analysis (summary contention section, `align_traces.py`) | ✅ done (`scripts/align_traces.py`: aligns all tenant traces to t0, computes overlap window + per-tenant stats; `benchmarks/summary.py` §10: degradation table, contention matrix, safe-operating-envelope section. 92 unit tests pass) |

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

Step-by-step walkthrough with agent prompts: [QUICKSTART_MULTIMODEL_CONTENTION.md](../../QUICKSTART_MULTIMODEL_CONTENTION.md).

Phases follow the structure in `workspace/contention/experiment_design.md`.

> **First session on real hardware?** The harness was built and unit-tested
> without a GPU. Work through
> [reference/gpu-validation.md](reference/gpu-validation.md)
> before trusting any number it produces — it lists every assumption made
> without hardware, ordered by how much work each one invalidates if wrong.
> The weight estimates that set every VRAM cap are top of that list.

### Prerequisites — once per machine, before any run

Skipping these does not fail early. The Triton one fails at the first CV
tenant, several minutes into a window you paid GPU time for.

```bash
# 0. The CLI itself. `bench` is not on the system path — it lives in a venv you
#    own, separate from the per-backend venvs below. Without this you get
#    "Command 'bench' not found".
python3 -m venv ~/venv && ~/venv/bin/pip install -e .
export PATH="$HOME/venv/bin:$PATH"

# 1. Backend venvs. Each backend gets its own; the orchestrator activates them.
bench setup --backend vllm
bench setup --backend sglang        # only if a colocation names sglang

# 1b. Model access. A gated HF repo does NOT fail at plan time — the server
#     401s on config.json, exits, and the run waits out its full 600s
#     readiness budget before saying "server not ready". Once per affected
#     colocation. On rtx_pro6000: gemma2-9b and llama3.1-8b, both Phase 4.
python3 scripts/check_model_access.py --gpu rtx_pro6000
#     If GATED: (1) `.venv-vllm/bin/hf auth login` with a read token from
#     https://huggingface.co/settings/tokens — `hf`, not `huggingface-cli`,
#     and it is in the venv, not on PATH. (2) Accept the licence on each
#     gated model's page as the same account:
#       https://huggingface.co/google/gemma-2-9b-it
#       https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct  (form, not instant)
#     (3) Re-run this check until it exits 0. 403 = logged in but licence not
#     accepted; 401 = not logged in.

# 2. CV export deps. Not in requirements.txt / pyproject.toml on purpose —
#    contention-only. `bench setup` does NOT install them. Needs a
#    torch-capable venv; system python gives ModuleNotFoundError: ultralytics.
.venv-vllm/bin/pip install -r requirements-contention.txt

# 3. CV models into a Triton model repo. NOTHING ELSE DOES THIS — the
#    orchestrator writes each model's config.pbtxt but cannot produce the
#    weights, so an unbuilt repo means every CV tenant fails to load.
#    One invocation per (model, backend) a colocation references.
.venv-vllm/bin/python scripts/build_triton_cv_repo.py --model yolov8-l \
       --triton-backend tensorrt --repo-root benchmarks/results/rtx_pro6000/triton_repo
.venv-vllm/bin/python scripts/build_triton_cv_repo.py --model dinov2-base \
       --triton-backend tensorrt --repo-root benchmarks/results/rtx_pro6000/triton_repo

#    With --triton-backend tensorrt that is only HALF the build: the script
#    exports model.onnx and PRINTS a trtexec command it deliberately does not
#    run (a host-built plan will not load in the container). Run the printed
#    command per model, or use --triton-backend onnx to get a one-step build.
#    config.pbtxt + model.onnx alone = Triton fails to load at the first window.

# 4. MPS. Without it tenants time-slice and the study measures the scheduler.
#    Do NOT set CUDA_VISIBLE_DEVICES — unset, the daemon serves every card,
#    which is what you want on 1, 4 or 8 GPUs. Setting it pins MPS to a fixed
#    list and silently excludes the rest.
#    Persist the pipe dir rather than exporting it once: it must be set in
#    whichever shell later runs `bench coloc`, because that is how the pipe
#    reaches the Triton containers, which cannot join MPS without it.
echo 'export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps' >> ~/.bashrc
source ~/.bashrc
nvidia-cuda-mps-control -d
echo get_default_active_thread_percentage | nvidia-cuda-mps-control   # expect 100.0
#    Verify with the control socket, NOT pgrep: the daemon's name is 23 chars
#    and comm truncates to 15, so `pgrep -x nvidia-cuda-mps-control` never
#    matches a running daemon. `pgrep -f` does.
#    Export the pipe dir in the SAME shell as `bench coloc` — that is how it
#    reaches the Triton containers, which must bind-mount it to join MPS.

# 5. Workload payloads. The .jsonl are committed, so this only verifies them —
#    but a missing one silently drops --input-file and every LLM/VLM tenant
#    runs on synthetic prompts instead of yours.
python3 scripts/build_contention_prompts.py --check
```

### The run

```bash
# Phase 0 — GATE. Do this first and stop if it fails.
python3 scripts/gpu_concurrency_probe.py --gpu rtx_pro6000 --json
#   Confirms tenants genuinely overlap on the GPU rather than time-slicing.
#   Also measures run-to-run variance (5x) — that sets the repetition policy.

# Resolve a plan and run every pre-flight WITHOUT launching anything.
# Non-zero exit + "PRE-FLIGHT WOULD BLOCK THIS RUN" means fix it first.
# NEEDS NO GPU — worth running on any machine before you rent one. All 39
# colocations were dry-run clean as of 2026-08-03.
bench coloc --gpu rtx_pro6000 --all --dry-run

# The whole study as ONE plan: 163 runs, not 237 — selecting many colocations
# dedupes solo baselines across all of them (~74 runs / ~3h of GPU saved).
# --resume skips runs that already have a manifest.json; --continue-on-error
# records a failure and keeps going, exiting non-zero at the end.
bench coloc --gpu rtx_pro6000 --all --continue-on-error --resume

# Phase 1 — solo baselines, at the SAME offered rate as the contention runs
bench coloc --gpu rtx_pro6000 --colocation mix-llm-cv --solo-only

# Phases 2-6 — by phase, or by name (both flags repeatable)
bench coloc --gpu rtx_pro6000 --phase 3 --phase 4
bench coloc --gpu rtx_pro6000 --colocation cross-llm-vs-cv-rps     # rate sweep
bench coloc --gpu rtx_pro6000 --colocation mix-full            # all 4 categories

# Phase 5 — placement, 2 GPUs
bench coloc --gpu rtx_pro6000 --colocation place-isolated      # null test: ratio ~1.0
bench coloc --gpu rtx_pro6000 --colocation place-p1

bench summary --gpu rtx_pro6000                                # contention section
# Runs land in coloc/<colocation>/coloc-<tenant>@<rps>[-r<rep>]-<hash>/, shared
# solo baselines in coloc/_baselines/solo-<backend>-<model>@<rps>-<hash>/.
python3 scripts/align_traces.py benchmarks/results/rtx_pro6000/coloc/<colocation>/<run_dir>/
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
6. **One GPU sampler per occupied card** for the whole window, never one per
   tenant. Two samplers on one card means two `dcgmi dmon` processes and both
   tenants reporting that card's memory as their own; a card with no sampler
   means a placement result with no telemetry to explain it.
7. **Workload payloads present.** `bench coloc` refuses to launch if a
   workload's `prompts:` or `data:` file is missing — but only because that
   check exists. Without it aiperf silently falls back to synthetic prompts and
   never sends the video, so `cross-vlm-prefill-vs-llm` would run as a
   text-only workload and still produce plausible numbers.
8. **Confirm the video is actually being sent.** Run
   `cross-vlm-prefill-vs-llm` solo for ~30s and check the VLM tenant's
   `input_sequence_length` is in the thousands, not ~30. That single number
   separates "40-frame prefill burst" from "silently text".
9. **Confirm every tenant joined MPS.** `bench coloc` holds its terminal, so
   run this from a **second terminal** during a *contention* window (solo
   baselines run first and have only one tenant). Every tenant PID must appear
   in an MPS client list:
   ```bash
   scripts/check_mps_clients.sh      # want: PASS: all 2 GPU process(es) are MPS clients
   ```
   It prints `IDLE` between windows, a 1-tenant `PASS` during a solo baseline,
   and `FAIL` naming any GPU process that is not an MPS client.
   **Do not judge this from `nvidia-smi` alone**: on
   Volta and later each MPS client keeps its own address space and lists as its
   own process, so separate `tritonserver` and `vllm` entries are what a
   correctly shared GPU looks like. A host daemon running is also not evidence —
   a container only joins if `CUDA_MPS_PIPE_DIRECTORY` is bind-mounted in, so
   `environment.mps.detected` can be `true` while the CV tenant is outside MPS
   entirely. `environment.mps.container_pipe_directory` is the manifest field
   that answers it; `null` means it could not have joined.

## Failure recovery

| Symptom | Cause | Action |
|---|---|---|
| Phase 0 shows ~1.0× aggregate throughput, ~2.0× latency | Tenants are **serialising**, not sharing | Stop. Enable MPS and retry. If still serialised, the study measures time-slice fairness — rescope and tell the user |
| Tenant 2 OOMs at startup | Tenant 1 took the whole card | Set explicit `gpu_memory_utilization` per tenant |
| A tenant produced no numbers, or the run warns `ALL n/n requests failed` | The server rejected every request | Read `<run_dir>/<tenant>.server.log` — written per run, per tenant. The warning quotes the upstream error; a 400 on a video tenant is usually `--max-model-len` smaller than the clip |
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

- [reference/gpu-validation.md](reference/gpu-validation.md) — **every assumption made without a GPU**, ordered by what each invalidates. Work through it on first hardware contact; delete it once its answers are absorbed
- [docs/contention.md](../../docs/contention.md) — **start here**: what a degradation ratio means, why the solo baseline must be handicapped, how the four model types contend, and the single- and multi-GPU topologies. Written for customers too
- [reference/design-decisions.md](reference/design-decisions.md) — the methodology and *why*: open-loop, clock policy, sampler ownership, timestamps, repetition policy
- [reference/model-catalogue.md](reference/model-catalogue.md) — verified model sources, per-GPU scoping, and which picks are broken
- [reference/serving-topology.md](reference/serving-topology.md) — vLLM vs Triton vs MPS vs MIG, explained from first principles
- `workspace/contention/experiment_design.md` — the customer's original brief
- `workspace/contention/experiment_config.json` — model catalogue, prompts, phase definitions
- [docs/metrics.md](../../docs/metrics.md) — per-request metric definitions, reused unchanged
