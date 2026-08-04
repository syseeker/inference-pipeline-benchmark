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
- **A HuggingFace account**, with the licence accepted for every gated model.
  Step 2 checks it.
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

# Can this box actually download every model? One HEAD request each, no GPU.
python3 scripts/check_model_access.py --gpu rtx_pro6000
```

Each backend gets its own venv (`.venv-vllm`, …) which the orchestrator
activates for you.

**If any model comes back `GATED`:**

1. Log in with a read token from <https://huggingface.co/settings/tokens>:

   ```bash
   .venv-vllm/bin/hf auth login    # `hf`, not `huggingface-cli`; in the venv, not on PATH
   ```

2. Open each gated model's page **while signed in as that same account** and
   click *Agree and access repository*. On `rtx_pro6000` there are two:

   - <https://huggingface.co/google/gemma-2-9b-it>
   - <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct> — a form, and
     approval is not instant

3. Re-check until it exits 0:

   ```bash
   python3 scripts/check_model_access.py --gpu rtx_pro6000
   ```

`403 — logged in, licence not accepted` means step 2 is outstanding; `401 —
not logged in` means step 1 is.

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

# 5. Only if a colocation uses a python-backend CV model (kosmos-2.5 does, in
#    same-ilm / mix-ilm-cv / mix-vlm-ilm / mix-full). It runs inside the Triton
#    container and imports torch + transformers, which the stock image does not
#    ship. ~10 min, 11 GB, once per machine.
docker build -f docker/Dockerfile.triton-python \
  -t inference-bench/tritonserver:26.07-py3-transformers .

python3 scripts/build_contention_prompts.py --check
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
python3 scripts/gpu_concurrency_probe.py --gpu rtx_pro6000 --json
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

**Check `http_req_data_sent`, not `input_sequence_length`:**

```bash
D=benchmarks/results/rtx_pro6000/coloc/_baselines/solo-vllm-qwen2.5-vl-7b@1-*/
grep -o '"http_req_data_sent": {"value": [0-9.]*' $D/vlm.aiperf/profile_export.jsonl | head -3
```

**Want ~2,000 KB per request.** A text tenant sends ~0.1 KB, so the clip is
four orders of magnitude larger and there is no ambiguity. TTFT corroborates
it: 152 ms here against 40 ms for a text tenant on the same box.

**`input_sequence_length` cannot answer this question.** aiperf tokenizes the
text only and never counts multimodal expansion, so it reports ~29 for a video
request whose true length is 19,184 tokens — measured, from vLLM rejecting
these exact requests at `--max-model-len=16384`. Reading ~29 as "the video is
not being sent" is the wrong conclusion from a correct run.

If `http_req_data_sent` really is ~0.1 KB, the media is not reaching aiperf:
check the workload's `data:` file exists (Step 3's payload check) rather than
reaching for vLLM's `--allowed-local-media-path`, which is not how this
harness delivers video — it inlines the clip as a `data:` URL.

---

## Step 9 — Run the study

**Do not start this until Steps 5-8 have passed.** It is hours of GPU time,
and each of those steps fails in a way that would otherwise waste all of it.

**Prompt:**
> "Work through the contention study phase by phase and summarise each."

**What there is to run.** `--all` is the **single-GPU** study — it already
excludes Phase 5, which is Step 10. Phase 0 is the gate (Step 6) and Phase 1
is the solo baselines, which every plan generates for itself (72 of the 163
runs); neither is a command you issue.

| Phase | What it answers | Colocations | Runs |
|---|---|---|---|
| **2** — same-category | Do two models of the *same* type contend worst? | `same-llm` `same-cv` `same-vlm` `same-ilm` | 48 |
| **3** — mixed pairs | The headline pairings, cheapest phase | `mix-llm-cv` `mix-vlm-cv` `mix-ilm-cv` `mix-vlm-ilm` `mix-full` | 12 |
| **4** — cross-type | Where the load-vs-degradation answers are | `cross-llm-vs-cv-rps` `cross-cv-vs-llm-rps` `cross-ilm-vs-cv` `cross-vlm-prefill-vs-llm` `cross-size-scaling` `cross-arch-validation` `cross-memory-pressure-kv03/kv13/kv22/kv29` | 57 |
| **6** — secondary | One dimension at a time, `-a` light / `-b` heavy | `mix-memory-bound` `secondary-backend-llm-a/b` `secondary-backend-cv-a/b` `secondary-output-length-a/b` `secondary-input-size-cv-a/b` `secondary-input-size-llm-a/b` `secondary-asymmetry-a/b` `secondary-arrival-a/b` | 49 |
| **5** — placement | 2 GPUs — **Step 10**, not here | `place-isolated` `place-p1` `place-p2` `place-p3` `place-vlm-prefill-split` | 15 |

Per-phase counts are for that phase alone; they exceed 163 because `--all`
dedupes baselines shared across phases.

**CLI:**
```bash
# The whole single-GPU study, one command. 163 runs; budget several hours.
bench coloc --gpu rtx_pro6000 --all --continue-on-error --resume

# Or phase by phase (--phase is repeatable, and composes):
bench coloc --gpu rtx_pro6000 --phase 3 --resume                  # start here: 12 runs
bench coloc --gpu rtx_pro6000 --phase 4 --resume                  # the analytical core
bench coloc --gpu rtx_pro6000 --phase 2 --phase 6 --resume

# Or name them explicitly (--colocation is repeatable):
bench coloc --gpu rtx_pro6000 --colocation mix-full --colocation cross-size-scaling
```

**If the ratios from Step 7 were all ≈1.00×, find the load point first.**
A rate sweep costs 9 runs and tells you where degradation actually starts;
without it the whole study can return ≈1.00× everywhere, which reads as "co-location
is free on this GPU" when it means "nothing was driven hard enough":

```bash
bench coloc --gpu rtx_pro6000 --colocation cross-cv-vs-llm-rps --resume
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

**This is Phase 5, and `--all` in Step 9 did not run it** — the single-GPU
study stops at 39 colocations / 163 runs and holds these back. 5 colocations,
15 runs.

Only worth doing once the single-GPU numbers exist — the whole point is the
comparison. Nothing else may be running: both cards are occupied.

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
python3 scripts/align_traces.py benchmarks/results/rtx_pro6000/coloc/<run_dir>/
```

The **Contention analysis** section of `summary.md` gives three things (numbered §10 alongside a single-model sweep, §1 in a contention-only results dir):

- **Degradation table** — throughput retention and p50/p95/TTFT ratios per
  tenant against its matched solo baseline. The primary result.
- **Contention matrix** — p95 ratio by victim × aggressor. **Read it both
  ways**; contention is not symmetric, and the asymmetry is usually the most
  actionable finding.

  This is why the `cross-*` pairs are not duplicates. `cross-llm-vs-cv-rps` and
  `cross-cv-vs-llm-rps` share the same two tenants and differ only in which
  one's rate is swept — the names read `victim-vs-aggressor`, so the aggressor
  is the one that moves:

  | Colocation | Sweeps | Held fixed | Answers |
  |---|---|---|---|
  | `cross-llm-vs-cv-rps` | cv `1 → 200` rps | llm @4 | how much CV load costs the LLM |
  | `cross-cv-vs-llm-rps` | llm `1 → 64` rps | cv @50 | how much LLM load costs the CV |
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

`bench summary` then reads every manifest under `<gpu>/coloc/` and builds the contention section.

**`scripts/run_all_scenarios.sh` is not used here.** It is the single-model
sweep driver and its invariants — one model, kill the server every round —
are wrong for co-residency. Contention has its own entry point, `bench coloc`.
The two do not interact.

---

## Cheat sheet — when something looks wrong

| Symptom | Cause | Action |
|---|---|---|
| `Command 'bench' not found` | CLI not installed | Step 0 |
| `server not ready within budget` after ~600s, server log shows `GatedRepoError` / 401 | Model licence not accepted | `python3 scripts/check_model_access.py --gpu rtx_pro6000`, then Step 2 |
| `huggingface-cli: command not found` | Renamed to `hf`, and it is in the venv | `.venv-vllm/bin/hf auth login` |
| `ModuleNotFoundError: ultralytics` | CV export deps missing, or wrong interpreter | `.venv-vllm/bin/pip install -r requirements-contention.txt`, and run the exporter with `.venv-vllm/bin/python` (Step 3) |
| Phase 0 gives ~0.28× | MPS not active | Step 4. Do not proceed |
| Phase 0 reports `isolation: "none"` with MPS clearly running | `CUDA_MPS_PIPE_DIRECTORY` not exported in that shell | Step 4 — export it, re-run |
| CV tenant fails to load | Triton model repo not built, or built to `model.onnx` while `config.pbtxt` says `backend: "tensorrt"` | Step 3 — run the `trtexec` step, or rebuild with `--triton-backend onnx` |
| Ratios plausible but `nvidia-smi` shows separate `tritonserver` + `vllm` processes | CV container never joined MPS | Step 4 — export `CUDA_MPS_PIPE_DIRECTORY` in the shell running `bench coloc`; check `mps.container_pipe_directory` in the manifest |
| VLM `http_req_data_sent` ~0.1 KB instead of ~2000 KB | Video not reaching aiperf | Step 8 — check the workload's `data:` file exists. `input_sequence_length` is text-only and reads ~29 even when the video IS sent; it is not the check |
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
