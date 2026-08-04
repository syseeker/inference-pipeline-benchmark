# Multi-model contention end-to-end quickstart

You'll measure how much co-resident models slow each other down on one GPU —
or across two — by **prompting an agent**, with the CLI shown alongside every
step. Paste each step's prompt into your agent (Claude Code / Codex / Cursor);
the walkthrough says what the agent should do and what to verify on disk.

First useful number in about 20 minutes. The full study is hours.

> **Read [docs/contention.md](docs/contention.md) first.** Not optional here.
> A degradation ratio is easy to produce and easy to produce *wrongly*, and
> that doc explains the one rule everything rests on: between the baseline and
> the contention run, the neighbour must be the only thing that changed.

> **Hardware validation is partway through.** Setup, the Phase 0 gate and the
> solo baselines are proven on a PRO 6000; no contention window has completed,
> so no degradation ratio exists yet. Several hardware assumptions are still
> unverified — work through
> [the validation record](skills/gpu-contention-benchmark/reference/gpu-validation.md)
> as you go; it is ordered by how much work each one invalidates.

---

## Prerequisites

- 1× or 2× NVIDIA GPUs. Written against RTX PRO 6000 Blackwell (96 GB, SM_120).
- **MPS available and startable.** Without it the tenants time-slice instead of
  overlapping and the whole study measures the scheduler. Step 4 sets it up and
  Phase 0 (Step 6) is the gate that proves it.
- Docker, for the Triton CV tenants — and for the `trtexec` plan build in Step 3.
- ~150 GB free disk for the model checkpoints in the roster.
- `git`, `python>=3.10`, `pip`.
- One of: Claude Code / Codex / Cursor.

---

## Step 0 — Install the CLI and the skill (one-time per machine)

`bench` is **not** on the system path. It lives in a venv you own — separate
from the per-backend venvs it later creates. Skipping this gives you
`Command 'bench' not found`.

```bash
git clone https://github.com/syseeker/inference-pipeline-benchmark
cd inference-pipeline-benchmark
python3 -m venv ~/venv && ~/venv/bin/pip install -e .
export PATH="$HOME/venv/bin:$PATH"

bench install-skill --agent auto --json     # so your agent knows the workflow
```

---

## Step 1 — Probe the box

**Prompt:**
> "What GPUs does this box have, and is MPS running?"

**What the agent does:**
1. `bench probe --json` → GPU model, driver, CUDA, per-venv backend versions.
2. `nvidia-smi topo -m` → the interconnect, which decides whether tensor
   parallelism is even worth measuring.

**CLI:**
```bash
bench probe --json
nvidia-smi topo -m
```

**What to expect:** on a 2-GPU PRO 6000 box, `topo -m` showing **PIX/PHB, not
NV#**. The config records `nvlink: false`; this is where you confirm it. If
NVLink *is* present, say so — tensor-parallel colocations were deliberately
left unwritten on the assumption it is not.

---

## Step 2 — Install the serving backends

**Prompt:**
> "Set up the vLLM backend for the contention study."

**CLI:**
```bash
bench setup --backend vllm
bench setup --backend sglang      # only if you'll run secondary-backend-llm-*
```

Each backend gets its own venv (`.venv-vllm`, …) which the orchestrator
activates for you.

---

## Step 3 — Build the CV models and check the payloads

Nothing else builds the CV weights. An unbuilt repo means every CV tenant fails
to load, minutes into a window you are paying for.

**Prompt:**
> "Build the Triton CV model repo for yolov8-l and dinov2-base, and verify the
> contention workload payloads are present."

**CLI** — four commands per model, in order:

```bash
REPO=benchmarks/results/rtx_pro6000/triton_repo

# 1. CV export deps. `bench setup` does not install these.
.venv-vllm/bin/pip install -r requirements-contention.txt

# 2. Export the ONNX. Must be a torch-capable venv, not system python.
.venv-vllm/bin/python scripts/build_triton_cv_repo.py --model yolov8-l \
       --triton-backend tensorrt --repo-root "$REPO"
.venv-vllm/bin/python scripts/build_triton_cv_repo.py --model dinov2-base \
       --triton-backend tensorrt --repo-root "$REPO"

# 3. Build the plan. The script PRINTS this command per model and does not run
#    it — a plan built outside the container will not load inside it. Copy the
#    printed command; the shapes differ per model.
docker run --rm --gpus all -v "$PWD/$REPO/yolov8-l/1":/w \
  nvcr.io/nvidia/tritonserver:26.07-py3 \
  trtexec --onnx=/w/model.onnx --saveEngine=/w/model.plan \
    --minShapes=images:1x3x640x640 --optShapes=images:8x3x640x640 \
    --maxShapes=images:8x3x640x640

# 4. Docker wrote model.plan as root; Triton must be able to read it.
sudo chown -R "$(id -u):$(id -g)" "$REPO"

python scripts/build_contention_prompts.py --check
```

**What to expect:** `config.pbtxt` **and** `1/model.plan` under
`triton_repo/<model>/`, and `prompts in sync` from the payload check.

`--triton-backend onnx` skips the `trtexec` step, at the cost of the optimised
path.

---

## Step 4 — Turn on MPS

Without MPS the tenants time-slice the card instead of sharing it, and every
number describes the scheduler. Step 6 is the gate that proves it.

**Prompt:**
> "Start MPS and confirm the daemon is up."

**CLI:**
```bash
# 1. Persist the pipe dir. Every later shell needs it — it is how the pipe
#    reaches the Triton containers, which cannot join MPS without it.
cat >> ~/.bashrc <<'EOF'
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps
EOF
source ~/.bashrc

# 2. Start the daemon.
sudo mkdir -p "$CUDA_MPS_LOG_DIRECTORY" && sudo chown "$(id -u):$(id -g)" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d

# 3. Verify. Use the control socket, not pgrep.
echo get_default_active_thread_percentage | nvidia-cuda-mps-control   # expect 100.0
echo "$CUDA_MPS_PIPE_DIRECTORY"                                       # expect /tmp/nvidia-mps
```

**Do not set `CUDA_VISIBLE_DEVICES`.** Unset, the daemon serves every card —
correct on 1, 4 or 8 GPUs. Setting it pins MPS to a fixed list and silently
excludes the rest. Per-tenant placement is the yaml's `device:` key.

The daemon does not survive a reboot; the `~/.bashrc` exports do.

**Rollback:** `echo quit | nvidia-cuda-mps-control`

---

## Step 5 — Dry-run the plan (no GPU needed)

Before spending a minute of GPU time, resolve every run and execute every
pre-flight without launching anything.

**Prompt:**
> "Dry-run every contention colocation and tell me if any would be blocked."

**CLI:**
```bash
bench coloc --gpu rtx_pro6000 --all --dry-run          # the whole study
bench coloc --gpu rtx_pro6000 --colocation mix-llm-cv --dry-run   # just one
```

**What to expect:**

```
[ok] coloc: plan: 39 colocation(s) → 163 run(s) (72 solo baseline(s) + 91 contention window(s))
```

`mix-llm-cv` alone gives `1 colocation(s) → 3 run(s)`. Add `--json` for the
per-run plan and the directory each run will land in.

A non-zero exit with `PRE-FLIGHT WOULD BLOCK THIS RUN` means a VRAM budget that
cannot fit or a missing payload. Fix it before proceeding.

---

## Step 6 — Phase 0, the gate

**Stop here if it fails.** Everything downstream becomes a statement about the
time-slice scheduler rather than about GPU contention.

**Prompt:**
> "Run the Phase 0 concurrency gate and tell me whether the tenants genuinely
> overlap."

**CLI:**
```bash
python scripts/gpu_concurrency_probe.py --gpu rtx_pro6000 --json
```

**What to expect:** aggregate throughput **> 1.0×** with MPS on. The reference
measurement on a PRO 6000 was **1.94× with MPS on, 0.28× with MPS off** — that
0.28 is worse than running the two models one after the other, because they
fight over context switches. If you see a number like that, MPS is not active.

The probe also measures run-to-run variance across 5 repeats, which sets the
repetition policy for the rest of the study.

---

## Step 7 — One colocation end-to-end

**Prompt:**
> "Run the mix-llm-cv colocation and show me the degradation ratios."

**CLI:**
```bash
bench coloc --gpu rtx_pro6000 --colocation mix-llm-cv
```

**What runs:** three windows, baselines first.

| | Tenants |
|---|---|
| solo | qwen2.5-7b @4 rps, alone |
| solo | yolov8-l @50 rps, alone |
| contention | both together |

**Open a second terminal.** The command above holds this one until all three
windows finish. Then check that the tenants are really sharing the GPU:

```bash
scripts/check_mps_clients.sh
```

It prints one of four verdicts:

| | Meaning |
|---|---|
| `IDLE` | Nothing on the GPU. Between windows — re-run in a moment |
| `PASS` … `Only one tenant` | A solo baseline. Correct, but not the check you want |
| `PASS: all 2 GPU process(es) are MPS clients` | **This is the one.** The contention window is sharing properly |
| `FAIL` | A tenant is outside MPS and time-slicing — this window's ratios are void. Re-check `CUDA_MPS_PIPE_DIRECTORY` from Step 4 |

Keep re-running it until you see the 2-tenant `PASS`. The two solo baselines
run first, so early on `IDLE` and 1-tenant `PASS` are both expected.

**What to expect on disk**, under `benchmarks/results/rtx_pro6000/coloc/`:

```
_baselines/solo-vllm-qwen2.5-7b@4-43d07607/   manifest.json  llm.ndjson
_baselines/solo-triton-yolov8-l@50-d2b34069/  manifest.json  cv.ndjson
mix-llm-cv/coloc-llm@4-cv@50-8a1c39f2/        manifest.json  llm.ndjson  cv.ndjson
```

**Check in each `manifest.json`:**

- `achieved_rps` close to `offered_rps` — if not, the load generator was the
  bottleneck, not the GPU
- `gpu_sampler` has a `"0"` key
- `environment.mps.container_pipe_directory` is not `null`

### Then lock the compute mode

Now that a CV tenant is confirmed inside MPS, `EXCLUSIVE_PROCESS` makes any
process that cannot reach MPS fail loudly instead of joining as an unmeasured
tenant.

```bash
echo quit | nvidia-cuda-mps-control
sudo nvidia-smi -c EXCLUSIVE_PROCESS    # add -i N to scope to one card
nvidia-cuda-mps-control -d
```

Revert with `sudo nvidia-smi -c DEFAULT`. Neither the mode nor the daemon
survives a reboot.

---

## Step 8 — Confirm the video is actually being sent

Do this before the VLM experiments, not after.

**Prompt:**
> "Run cross-vlm-prefill-vs-llm solo for a short window and tell me the VLM
> tenant's input sequence length."

**CLI:**
```bash
bench coloc --gpu rtx_pro6000 --colocation cross-vlm-prefill-vs-llm --solo-only
```

**What to expect:** the VLM tenant's `input_sequence_length` **in the
thousands** — a 40-frame clip expands to several thousand tokens.

**If it is ~30, the video is not being sent** and you are measuring a
text-only workload. That means aiperf's `video` field is not reaching vLLM the
way we assumed — most likely vLLM needs `--allowed-local-media-path`, which
nothing in the yaml currently sets. This is the single most important check in
the walkthrough: the whole prefill-burst premise depends on it, and it fails
silently rather than loudly.

---

## Step 9 — Run the study

**Do not start this until Steps 5-8 have passed.** It is hours of GPU time,
and each of those steps fails in a way that would otherwise waste all of it.

**Prompt:**
> "Work through the contention study phase by phase and summarise each."

**CLI:**
```bash
# Check the whole study first — no GPU needed, prints every pre-flight issue.
bench coloc --gpu rtx_pro6000 --all --dry-run

# The whole study, one command. ~163 runs; budget several hours.
bench coloc --gpu rtx_pro6000 --all --continue-on-error --resume

# Or phase by phase (--phase is repeatable, and composes):
bench coloc --gpu rtx_pro6000 --phase 2 --phase 3 --resume
bench coloc --gpu rtx_pro6000 --phase 4 --resume

# Or name them explicitly (--colocation is repeatable):
bench coloc --gpu rtx_pro6000 --colocation mix-full --colocation cross-size-scaling
```

Selecting several colocations builds **one** plan, and solo baselines are
deduped across the whole plan: all 39 colocations emit 163 runs (72 baselines +
91 contention windows) instead of the 237 you get by running them as 39
separate commands. Selection order is fixed (phase, then yaml order), so a
re-run is comparable to the previous one.

`--resume` skips any run whose `manifest.json` already exists, so a job that
dies at hour 5 restarts where it stopped. `--continue-on-error` records a
failed run and carries on, then exits non-zero with the failure list — one bad
colocation at run 50 must not cost you the other 113.

---

## Step 10 — Two GPUs

Only worth doing once the single-GPU numbers exist — the whole point is the
comparison.

**Prompt:**
> "Run the placement study on both GPUs and tell me which pairing wins."

**CLI:**
```bash
# The null test FIRST — it validates the harness, not the GPU.
bench coloc --gpu rtx_pro6000 --colocation place-isolated

# The three pairings of mix-full's four tenants across two cards
bench coloc --gpu rtx_pro6000 --colocation place-p1    # [LLM+VLM] | [ILM+CV]
bench coloc --gpu rtx_pro6000 --colocation place-p2    # [LLM+ILM] | [VLM+CV]
bench coloc --gpu rtx_pro6000 --colocation place-p3    # [LLM+CV]  | [VLM+ILM]
```

**`place-isolated` must return a ratio ≈ 1.0.** Its two tenants are on
different cards with no shared SMs, bandwidth or VRAM. If a tenant still
degrades there, the harness has a bug and nothing else it reports can be
trusted.

**The prediction on the record** (from [docs/contention.md](docs/contention.md)
§3, pair tenants that stress *different* resources): **P1 best, P3 worse, P2
worst**. If the ranking holds you have a placement rule that generalises to
models never tested. If it does not, the resource model in that doc is wrong —
which is the more interesting outcome.

Placement comes from a `device:` key on each tenant. The field accepts indices
up to 7; this study is scoped to 2 GPUs.

---

## Step 11 — Interpret

**Prompt:**
> "Summarise the contention results and tell me which pairings are deployable."

**CLI:**
```bash
bench summary --gpu rtx_pro6000
python scripts/align_traces.py benchmarks/results/rtx_pro6000/coloc/<run_dir>/
```

`summary.md` §10 gives three things:

- **Degradation table** — throughput retention and p50/p95/TTFT ratios per
  tenant against its matched solo baseline. The primary result.
- **Contention matrix** — p95 ratio by victim × aggressor. **Read it both
  ways**; contention is not symmetric, and the asymmetry is usually the most
  actionable finding.
- **Safe-operating envelope** — where `achieved_rps` fell below `offered_rps`.
  That is the deployment limit, and it comes free from open-loop load.

`align_traces.py` is the diagnostic view of a single run: every tenant's
requests on one wall clock, so you can show that the LLM's p99 spike lands
*inside* the VLM's prefill window.

---

## How the yaml drives a run

One command names one colocation. Everything else is resolved from
`benchmarks/configs/rtx_pro6000.yaml`:

```
bench coloc --gpu rtx_pro6000 --colocation cross-size-scaling
        │
        ├─ load the yaml
        │
        ├─ colocations.cross-size-scaling
        │     extends: mix-llm-cv ......... merged BY TENANT NAME
        │     kv_budget_gb: 16.0 .......... the constant KV budget
        │     vary: {tenant, field, values} expands to N windows
        │     rps_sweep / repetitions ..... expand the same way
        │
        └─ per tenant, three more blocks are pulled in:
              backends.<backend>   base_url, port, extra_args, variants
              models.<model>       hf_id, weights_gb, backend_args,
                                   unsupported_backends
              workloads.<workload> prompts, data, output_tokens
                                        │
                                        ▼
                     cap = (weights_gb + kv_budget_gb + overhead) / vram_gb
```

That yields an ordered list of runs — **solo baselines first**, then the
contention windows — and each run then does:

1. **Pre-flight.** VRAM sums per GPU; every prompt and media file exists.
2. **Capture the environment.** `nvidia-smi topo -m`, MPS state.
3. **Launch a server per HTTP tenant** (`vllm serve …` with
   `CUDA_VISIBLE_DEVICES` set from `device:`) and **one Triton container per
   occupied card**.
4. **Hold a shared `t0`**, so every tenant's requests share one wall clock.
5. **Open one GPU sampler per occupied card** — not one per tenant, which
   would have every tenant reporting the whole card's memory as its own.
6. **Start one driver per tenant**: `aiperf` for LLM/VLM, `perf_analyzer` for
   Triton. Both open-loop, at the configured rate.
7. **Write** `manifest.json` plus one `.ndjson` of per-request records per
   tenant.

`bench summary` then reads every manifest under `<gpu>/coloc/` and builds §10.

**`scripts/run_all_scenarios.sh` is not used here.** It is the single-model
sweep driver and its invariants — one model, kill the server every round —
are wrong for co-residency. Contention has its own entry point, `bench coloc`.
The two do not interact.

---

## Cheat sheet — when something looks wrong

| Symptom | Cause | Action |
|---|---|---|
| `Command 'bench' not found` | CLI not installed | Step 0 |
| `ModuleNotFoundError: ultralytics` | CV export deps missing, or wrong interpreter | `.venv-vllm/bin/pip install -r requirements-contention.txt`, and run the exporter with `.venv-vllm/bin/python` (Step 3) |
| Phase 0 gives ~0.28× | MPS not active | Step 4. Do not proceed |
| Phase 0 reports `isolation: "none"` with MPS clearly running | `CUDA_MPS_PIPE_DIRECTORY` not exported in that shell | Step 4 — export it, re-run |
| CV tenant fails to load | Triton model repo not built, or built to `model.onnx` while `config.pbtxt` says `backend: "tensorrt"` | Step 3 — run the `trtexec` step, or rebuild with `--triton-backend onnx` |
| Ratios plausible but `nvidia-smi` shows separate `tritonserver` + `vllm` processes | CV container never joined MPS | Step 4 — export `CUDA_MPS_PIPE_DIRECTORY` in the shell running `bench coloc`; check `mps.container_pipe_directory` in the manifest |
| VLM `input_sequence_length` ~30 | Video not being sent | Step 8; check `--allowed-local-media-path` |
| Tenant 2 OOMs at startup | VRAM caps don't fit | `--dry-run`; check the derived caps in the manifest |
| `achieved_rps` << `offered_rps` even solo | Load generator is the bottleneck | Lower the rate, or check the client host isn't CPU-saturated |
| Ratios ≈ 1.0 everywhere | Load too low to contend | Raise the offered rate |
| Ratios wildly variable between repeats | Near-OOM KV eviction is bimodal | Expected at the memory-pressure points — report both modes, not the mean |
| A placement run has telemetry for only one card | Sampler didn't cover both | Check `gpu_sampler` has a key per device in the manifest |

---

## What this walkthrough does and doesn't cover

**Does:** the contention study end to end — one GPU and two, all seven phases,
39 colocations, from install to interpreted `summary.md`.

**Doesn't:**

- **Single-model benchmarking.** Different question, different baseline, and
  mixing the two is the most common way to misread either. See
  [QUICKSTART_VLM.md](QUICKSTART_VLM.md) and
  [docs/contention.md](docs/contention.md) §7.
- **Tensor parallelism.** No TP colocations are written, pending the
  interconnect check in Step 1.
- **4 GPUs.** The `device:` field supports up to 8; this study is scoped to 2.
- **Adding your own models or colocations.** See
  [docs/contention-coverage.md](docs/contention-coverage.md) for what exists
  and why, and the `extend-benchmark-config` skill for the schema.
