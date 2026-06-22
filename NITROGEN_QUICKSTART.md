# NitroGen end-to-end quickstart

Drive the full NitroGen FP8 sweep on one GPU by **prompting your agent**
— no flag memorisation. Read this once and either:

- **(verify a fresh deploy)** copy each prompt block in order. Each step
  shows what the agent should do under the hood + what to expect on disk.
- **(hand to a customer)** they paste the prompts. They get summary.md
  and the headline number without learning the CLI.

> New here? Skim [docs/why-this-matters.md](docs/why-this-matters.md) first.
> 10 minutes; tells you why the 5-minute walkthrough below matters.

---

## Prerequisites

- 1× NVIDIA GPU on a recent driver. Tested on RTX PRO 6000 Blackwell (SM_120). Should work on H200 + 5090 with minor yaml tweaks.
- ~10 GB free disk (`ng.pt` ~2 GB · one actions shard ~1.7 GB · FP8 ONNX artifacts ~1.5 GB · TRT plans ~180 MB).
- `git`, `python>=3.10`, `pip`, a HuggingFace account.
- One of: Claude Code / Codex / Cursor (any agent that follows skills/AGENTS.md/.cursor-rules).

---

## Step 0 — Clone + agent-skill install (one-time per workstation)

```bash
git clone https://github.com/syseeker/inference-pipeline-benchmark
cd inference-pipeline-benchmark
pip install -e .                                # installs the `bench` CLI
bench install-skill --agent auto --json         # symlinks SKILL.md into your agent's expected location
```

That's all the setup outside the agent itself.

---

## Step 1 — Probe the GPU

**Prompt:**
> "What GPU does this box have and what backends does it support?"

**What the agent does:**
1. `bench probe --json` → writes `benchmarks/results/host_<hostname>.json` and reports the GPU model, driver, CUDA toolkit, and per-venv backend versions.
2. Reads `benchmarks/configs/<gpu>.yaml`'s `unsupported_backends:` matrix and surfaces the pinned-out combos.

**What to expect:**
- `gpu`, `driver`, `cuda` fields populated.
- For PRO 6000 today: `trtllm` pinned out on Qwen3-VL-30B-A3B-FP8 (SM_120 fused-MoE), Nemotron-Omni (sglang shmem), Qwen3.6 (transformers registry), Gemma 4 (same reason).

---

## Step 2 — Install the NitroGen-FP8 backend

**Prompt:**
> "Set up the nitrogen-quant backend so I can run FP8 rounds."

**What the agent does:**
1. `bench setup --backend nitrogen-quant --json` → creates `.venv-nitrogen` with `[nitrogen,nitrogen-quant,dataset,dev]` extras (pulls torch + transformers<5 pin + modelopt + onnxruntime-gpu + tensorrt).
2. Surfaces the manual follow-ups it can't automate: `pip install -e ../NitroGen` (clone https://github.com/MineDojo/NitroGen first) + `hf download nvidia/NitroGen ng.pt`.

**What to expect:**
- `.venv-nitrogen/` directory exists.
- `ng.pt` is cached under `~/.cache/huggingface/hub/models--nvidia--NitroGen/`.
- (`which nsys` failing here is fine — we install that in Step 7 only if you ask for it.)

---

## Step 3 — Pull the dataset (3 scenarios)

**Prompt:**
> "Pull 3 NitroGen scenarios so we have something to benchmark on."

**What the agent does:**
1. `hf download nvidia/NitroGen --repo-type dataset --include "actions/SHARD_0000.tar.gz" --local-dir /your/path` (any free disk; this is the smallest shard).
2. `tar -xf SHARD_0000.tar.gz` if needed.
3. `bench scenarios build --source nitrogen --n 3 --actions-root /your/path/actions --synthetic-frames --json`.

**What to expect:**
- `tests/smoke/scenarios_nitrogen/` with 3 subdirs.
- Each subdir has: `screen.png` + `request.json` + `gold_action.json`. **No** `expected.json` — that's correct for a policy scenario (see [docs/scenarios.md](docs/scenarios.md) §2).
- `--synthetic-frames` writes a placeholder image; real video frames need YouTube cookies (cloud IPs get bot-blocked). Gold gamepad action is still real.

---

## Step 4 — Smoke one round (bf16, sanity check)

**Prompt:**
> "Smoke `nitrogen-eager` at bf16 to confirm the stack works."

**What the agent does:**
1. `bench smoke --gpu rtx_pro6000 --backend nitrogen-eager --model nitrogen-500m-bf16 --scenarios-dir tests/smoke/scenarios_nitrogen --nitrogen-ckpt-path <ng.pt>`.
2. Single round, 3 scenarios.

**What to expect:**
- `ok` status.
- p50 latency around **107 ms** on PRO 6000 BF16.
- One aggregate result JSON written under `benchmarks/results/<gpu>/`.
- Server log: `benchmarks/results/<gpu>/server-logs/nitrogen-eager.log`.

If this fails, **stop and fix the env** — the sweep won't recover.

---

## Step 5 — Run the full nitrogen-backends sweep

**Prompt:**
> "Run the `nitrogen-backends` sweep and tell me the winning backend + why."

**What the agent does:**
1. `bench sweep --gpu rtx_pro6000 --sweep nitrogen-backends --backends "nitrogen-eager nitrogen-compile nitrogen-cudagraph nitrogen-onnx nitrogen-tensorrt" --scenarios-dir tests/smoke/scenarios_nitrogen --nitrogen-ckpt-path <ng.pt>`.
2. Per-round:
   - First FP8 round triggers `ensure_artifact()` → downloads the calibrated ONNX from <https://huggingface.co/syseeker-at-nv/nitrogen-quant> (~488 MB, one-time per precision).
   - First TRT round triggers `_build_trt_plan_from_onnx()` → compiles a per-GPU `.plan` in ~10s, cached.
   - `nitrogen-tensorrt + nvfp4` row skipped: pinned out via `unsupported_backends` (TRT 10.16 missing FP4 plugin).
3. Six rows populated; `bench summary --gpu rtx_pro6000 --json` regenerates `summary.md`.

**What to expect — the headline:**

| Engine | Precision | p50 | seq/s | J/req |
|---|---|---|---|---|
| nitrogen-eager | bf16 | 107 ms | 9.4 | 11.2 |
| nitrogen-compile | bf16 | 108 ms | 9.4 | 11.1 |
| nitrogen-cudagraph | bf16 | 106 ms | 9.4 | 10.7 |
| nitrogen-onnx | **fp8** | **59 ms** | **16.4** | **5.7** ← **winner** |
| nitrogen-tensorrt | fp8 | 60 ms | 16.4 | 5.7 |
| nitrogen-tensorrt | fp8-4step | 61 ms | 16.4 | 5.8 |

Winner: `nitrogen-500m-fp8` on `nitrogen-onnx`. **45% lower latency at 2× throughput vs the bf16 baseline.** Energy halves.

---

## Step 6 — Interpret the result

**Prompt:**
> "Read summary.md and explain the winner and the surprises."

**What the agent does:**
1. Reads `benchmarks/results/<gpu>/summary.md` — specifically the Core findings, sections 1 (Decision metrics), and 5 (GPU resource usage).
2. Cross-references `docs/findings/knowledge.yaml` for known-cause explanations.
3. Applies the house style ([interpret-benchmark-summary skill](skills/interpret-benchmark-summary/SKILL.md)): winner first; under-performers get both "why" and "how to improve."

**What to expect in the answer:**
- Winner = FP8 because: weight bytes ↓50%, GEMM kernel selection shifts to FP8 paths on Blackwell.
- BF16 engines cluster within 1 ms because: workload is **launch-bound** (`DRAM_ACTIVE` ~0.4%, far below the 70% bandwidth-bound threshold), so CUDA-graph capture's per-launch-overhead saving is marginal at batch=1.
- FP8 4-step vs 16-step: nearly identical at batch=1 (61 vs 60 ms) because per-step launch cost dominates over iteration count.
- NVFP4 row absent: TRT 10.16 lacks the FP4 plugin; revisit on TRT bump.

That's the level of explanation you'd want in a deployment review.

---

## Step 7 — (Optional) Prove a hypothesis with a profile

When summary.md says "this is launch-bound" and someone asks for proof,
the agent shouldn't argue from numbers — it should produce a timeline.

**Prompt:**
> "Prove that `nitrogen-eager` is launch-bound."

**What the agent does:**
1. `bench setup --backend profile` (first time only — `sudo apt-get install` the latest `nsight-systems-YYYY.X.Y`, then chmod + symlink so `nsys` is on PATH).
2. `bench profile --tool nsys --gpu rtx_pro6000 --backend nitrogen-eager --model nitrogen-500m-bf16 ...`.
3. Reads the auto-emitted `<run>.summary.md` (`nsys stats` output) and quotes the NVTX-region timing.

**What to expect:**
- `<run>.nsys-rep` — open in Nsight Systems UI to see GPU-idle gaps between the DiT step's 16 kernel launches. That's the launch-bound smoking gun.
- `<run>.summary.md` — text narrative: top NVTX regions, GPU active vs idle percentages. The agent can quote from this directly in chat.

---

## Step 8 — (Optional) Concurrency curves (HTTP backends only)

NitroGen doesn't do this — its ZMQ server is single-flight (PR #6 + PR #8
detail). But for **VLM backends** the question becomes important: how
many parallel sims per GPU?

**Prompt:**
> "How does Qwen3-VL-32B-FP8 scale on vLLM under load on this GPU?"

**What the agent does:**
1. `bench setup --backend vllm` (one-time; pulls AIPerf alongside).
2. Start a vLLM server (`bench smoke --backend vllm --model qwen3-vl-32b-fp8 ...` or your own launch).
3. `bench load-test --gpu rtx_pro6000 --backend vllm --model Qwen/Qwen3-VL-32B-Instruct-FP8 --concurrency "1,4,16,32" --json`.
4. `bench summary --gpu rtx_pro6000 --json` → section 9 (Concurrency profile) populates.

**What to expect:**
- AIPerf writes `profile_export_aiperf.json` under `benchmarks/results/<gpu>/aiperf/<run>/`.
- Summary §9 shows TTFT p50/p99 + req/s + tok/s at each concurrency level — the curve tells you where throughput saturates and TTFT degrades.

---

## Cheat sheet — when something looks wrong

| Symptom | Most-likely cause | Where the agent looks |
|---|---|---|
| `bench setup` exit 4 | missing manual follow-up (NitroGen package, ng.pt, NVIDIA-index wheel) | the `next_action` field of the JSON status |
| `bench sweep` exit 2 (unsupported) | yaml's `unsupported_backends:` matched | swap to a supported backend in `--backends` |
| `bench sweep` exit 3 (runtime) | server crashed; OOM; kernel incompat | `benchmarks/results/<gpu>/server-logs/<backend>.log` |
| `bench profile --tool ncu` exits with `ERR_NVGPUCTRPERM` | driver-level perf-counter restriction | set `NVreg_RestrictProfilingToAdminUsers=0` and reload, or run as root |
| Section 9 (concurrency) missing from summary | no `bench load-test` runs yet (or HTTP server wasn't running when invoked) | go back to Step 8 |
| FP8 win is smaller than expected | calibration mismatch with your production frame distribution | `NITROGEN_FORCE_RECALIBRATE=1 bench sweep …` to bypass the pre-built artifact and re-calibrate locally |

---

## Clean re-run from scratch

```bash
rm -rf .venv-nitrogen .claude/skills .cursor/rules AGENTS.md
rm -rf benchmarks/results tests/smoke/scenarios_nitrogen
rm -rf /ephemeral/cache/nitrogen-engines    # cached TRT plans + ONNX
# Keep ~/.cache/huggingface — that's the deduplicated HF cache; redownloading
# nvidia/NitroGen wastes ~2 GB if you delete it.
```

Then go back to **Step 0**.

---

## What's verified vs deferred

This walkthrough was verified end-to-end on a real PRO 6000 Blackwell as part of PRs #5/5.0.5/5.1/6/7/7.1 (June 2026). What's not verified yet:

- AIPerf concurrency curves (Step 8) — requires a running HTTP backend; the wiring is verified but real numbers weren't on this box.
- NVFP4 row — TRT 10.16 missing the FP4 plugin; pinned out, revisit on TRT bump.
- Multi-GPU (TP=N, replicate-per-GPU) — runtime supports it via yaml; verified config-driven; first real customer with multi-GPU is the first to populate those numbers.

If your first run trips on something outside this list, that's a bug we want to know about — note it on the PR thread.
