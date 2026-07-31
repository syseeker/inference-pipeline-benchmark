# VLM inference benchmark quickstart

Benchmark VLMs across **vLLM / SGLang / TRT-LLM** on image+text and
video+text scenarios. You drive it by prompting an agent — no flag
memorisation. Paste each step's prompt into Claude Code / Codex / Cursor
and the walkthrough tells you what to expect on disk.

> New here? Read [docs/why-this-matters.md](docs/why-this-matters.md)
> first — it explains the four budgets your model has to fit inside
> and why backend choice matters.

---

## Two scenario types, same commands

| Scenario type | Input | Output | Pre-built? | Directory |
|---|---|---|---|---|
| Image+text | Screenshot + instruction | Schema-valid action sequence | Yes — 3 game scenarios | `tests/smoke/scenarios/` |
| Video+text | Short video + question | Free-form analysis | You provide the videos | `tests/smoke/scenarios_video/` |

Both use the same `bench setup → bench smoke → bench sweep → bench summary`
flow. The only difference is `--scenarios-dir`. You can run one or both.

---

## Models in scope (RTX PRO 6000)

| Model key | HF id | Backends | Notes |
|---|---|---|---|
| `qwen3-vl-32b-fp8` | `Qwen/Qwen3-VL-32B-Instruct-FP8` | vLLM, SGLang | Default; native 3D-RoPE video tower |
| `qwen3-vl-30b-a3b-fp8` | `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` | vLLM, SGLang | MoE; bandwidth stress-test |
| `gemma-4-31b-it-fp8` | `RedHatAI/gemma-4-31b-it-FP8-dynamic` | vLLM, SGLang | Cross-vendor; handles image + video |
| `nemotron-omni-fp8` | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` | vLLM only | SGLang blocked on SM_120 fused-MoE |

TRT-LLM is pinned out for most multimodal models on this GPU today — see
`benchmarks/configs/rtx_pro6000.yaml` `unsupported_backends:` for the
per-model reasons. Swap `rtx_pro6000` → `rtx5090` or `h200` for other GPUs.

---

## Prerequisites

- 1× NVIDIA GPU. Tested on RTX PRO 6000 Blackwell (SM_120). Works on H200
  and RTX 5090 with the appropriate `--gpu` flag.
- ~50 GB free disk (Qwen3-VL-32B-FP8 ≈ 16 GB; models auto-download from HF
  on first `vllm serve`).
- `git`, `python>=3.10`, `pip`, a HuggingFace account (`huggingface-cli login`).
- One of: Claude Code / Codex / Cursor.

---

## Step 0 — Clone + agent-skill install (one-time per workstation)

```bash
git clone https://github.com/syseeker/inference-pipeline-benchmark
cd inference-pipeline-benchmark
pip install -e .                                # installs the `bench` CLI
bench install-skill --agent auto --json         # symlinks SKILL.md into your agent's expected location
```

---

## Step 1 — Probe the GPU

**Prompt:**
> "What GPU does this box have and what backends does it support?"

**What the agent does:**
1. `bench probe --json` → writes `benchmarks/results/host_<hostname>.json`.
2. Reads `benchmarks/configs/<gpu>.yaml`'s `unsupported_backends:` matrix
   and surfaces which model+backend combos are pinned out.

**What to expect:**
- `gpu`, `driver`, `cuda` fields populated.
- For PRO 6000: TRT-LLM pinned out on all multimodal models (MoE fused-kernel
  or arch-registry gap). The sweep will skip those rows automatically.

---

## Step 2 — Set up a VLM backend venv

**Prompt:**
> "Set up the vLLM backend so I can run the VLM sweep."

**What the agent does:**
1. `bench setup --backend vllm --json` → creates `.venv-vllm` with
   `[vllm, aiperf, dev]` extras.
2. For SGLang: `bench setup --backend sglang --json` → `.venv-sglang`.
3. For TRT-LLM (only if needed): `bench setup --backend trtllm --json` →
   `.venv-trtllm`, then manually:
   ```bash
   source .venv-trtllm/bin/activate
   pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com
   deactivate
   ```

**What to expect:**
- `.venv-vllm/` (and optionally `.venv-sglang/`, `.venv-trtllm/`) exist.
- Models are **not** downloaded at setup time — they pull from HF on first
  `vllm serve`. Allow 10–20 minutes on a fresh machine for Qwen3-VL-32B-FP8.

---

## Step 3 — Prepare your scenarios

### 3A — Image+text (pre-built, nothing to do)

Three game scenarios are already committed:

```
tests/smoke/scenarios/
  01_clash_of_clans_start_attack/    screen.png + request.json + expected.json
  02_catan_open_menu/
  03_fps_engage_and_reload/
```

Verify with:
```bash
bench scenarios list --scenarios-dir tests/smoke/scenarios/
```

Skip to Step 4.

### 3B — Video+text (bring your own videos)

The three scenario folders are already created with placeholder `request.json`
and `expected.json`. Replace them with your content:

```
tests/smoke/scenarios_video/
  01_customer_scene_1/    ← upload your video here
  02_customer_scene_2/
  03_customer_scene_3/
```

**Upload videos** (from your local machine):
```bash
scp /local/path/to/scene1.mp4 <host>:~/inference-pipeline-benchmark/tests/smoke/scenarios_video/01_customer_scene_1/video.mp4
scp /local/path/to/scene2.mp4 <host>:~/inference-pipeline-benchmark/tests/smoke/scenarios_video/02_customer_scene_2/video.mp4
scp /local/path/to/scene3.mp4 <host>:~/inference-pipeline-benchmark/tests/smoke/scenarios_video/03_customer_scene_3/video.mp4
```

**Edit `request.json`** for each scenario:
```json
{
  "name": "01_customer_scene_1",
  "description": "Warehouse floor — forklift crossing a pedestrian zone.",
  "video_path": "video.mp4",
  "prompt": "What safety risks are visible in this video?",
  "deadline_ms": 30000
}
```

**Edit `expected.json`**:
```json
{
  "key_phrases": ["forklift", "pedestrian", "collision", "proximity"],
  "min_coverage": 0.6
}
```
- `key_phrases` — words you expect in a correct model response
- `min_coverage` — fraction that must appear (0.6 = pass if ≥60% present)

Validate after filling in:
```bash
bench scenarios list --scenarios-dir tests/smoke/scenarios_video/
```

---

## Step 4 — Smoke one backend

Validate end-to-end before the full sweep. For image:

**Prompt:**
> "Smoke `vllm` with `qwen3-vl-32b-fp8` on the image scenarios to confirm the stack works."

**What the agent does:**
```bash
bench smoke --gpu rtx_pro6000 --backend vllm --model qwen3-vl-32b-fp8
```

For video scenarios, add `--scenarios-dir tests/smoke/scenarios_video/`:
```bash
bench smoke --gpu rtx_pro6000 --backend vllm --model qwen3-vl-32b-fp8 \
    --scenarios-dir tests/smoke/scenarios_video/
```

**What to expect:**
- First run: model downloads from HF (~16 GB for 32B-FP8). Subsequent runs
  use the HF cache — no re-download.
- Server starts in the background, scenarios run, server stops.
- `ok` status + one aggregate result JSON under `benchmarks/results/rtx_pro6000/`.
- Latency is in **seconds** (not ms) for large VLMs at batch=1 on image+text;
  video adds vision-encoder time on top.

If smoke fails, stop and fix before running the full sweep.

---

## Step 5 — Run the sweep

### Image sweep

**Prompt:**
> "Run the VLM image sweep and tell me the winning backend."

**What the agent does:**
```bash
bench sweep --gpu rtx_pro6000
```

Runs all VLM models × all supported backends. TRT-LLM rows are skipped
automatically where `unsupported_backends:` applies.

To pin a specific model or backend subset:
```bash
bench sweep --gpu rtx_pro6000 --backends "vllm sglang" --model qwen3-vl-32b-fp8
```

### Video sweep

```bash
# Full matrix — all models × all backends × 3 frame counts (4 / 8 / 16)
bench sweep --gpu rtx_pro6000 --sweep video \
    --scenarios-dir tests/smoke/scenarios_video/

# Targeted — pick a frame count
bench sweep --gpu rtx_pro6000 --sweep video-4f --scenarios-dir tests/smoke/scenarios_video/
bench sweep --gpu rtx_pro6000 --sweep video-8f --scenarios-dir tests/smoke/scenarios_video/
bench sweep --gpu rtx_pro6000 --sweep video-16f --scenarios-dir tests/smoke/scenarios_video/
```

Frame count tradeoff: more frames = higher temporal coverage, higher latency.
The sweep produces one result row per (model, backend, frame-count) combination
so you can read the tradeoff directly from `summary.md`.

---

## Step 6 — Interpret the result

**Prompt:**
> "Read summary.md and explain the winner and the surprises."

**What the agent does:**
1. `bench summary --gpu rtx_pro6000 --json` → regenerates
   `benchmarks/results/rtx_pro6000/summary.md`.
2. Reads Core findings, §1 (Decision metrics), §5 (GPU resource).
3. Applies the house style: winner first; under-performers get "why" + "how to improve."

**For image scenarios**, the decision metrics are:
- Valid command-sequence latency (e2e ms) — did it meet the interactive budget?
- Command success rate — fraction accepted by the schema validator.
- Grammar validity — fraction passing on first try.

**For video scenarios**, the decision metrics are:
- e2e latency (seconds) across frame counts.
- Key-phrase coverage rate — quality proxy for whether the model answered on-target.

---

## Step 7 — (Optional) Load test

> **Note on video scenarios**: `bench load-test` wraps AIPerf, which sends
> synthetic text-only prompts. For video inference this skips the vision
> encoder — the numbers measure the serving stack, not your actual workload.
> Use `bench load-test` for image scenarios only.

For image backends with a running server:

**Prompt:**
> "How does Qwen3-VL-32B-FP8 scale on vLLM under load?"

**What the agent does:**
1. Starts vLLM (or confirms it is already running).
2. `bench load-test --gpu rtx_pro6000 --backend vllm --model Qwen/Qwen3-VL-32B-Instruct-FP8 --concurrency "1,4,16,32" --json`.
3. `bench summary --gpu rtx_pro6000 --json` → §9 (Concurrency profile) populates.

**What to expect:**
- AIPerf writes `profile_export_aiperf.json` under
  `benchmarks/results/rtx_pro6000/aiperf/<run>/`.
- Summary §9 shows TTFT p50/p99 + req/s + tok/s at each concurrency level.
- The curve tells you where throughput saturates and TTFT degrades.

---

## Step 8 — (Optional) Profile bottlenecks (Nsight Systems)

When `summary.md` flags high latency and you want to confirm which GPU phase
is responsible, escalate to a Nsight timeline.

> **Unlike `bench load-test`**, `bench profile` manages the server lifecycle
> itself — do **not** start vLLM manually first or you will get a port conflict.

**Prompt:**
> "Profile `vllm` with `qwen3-vl-32b-fp8` on the image scenarios — I want to see where time goes."

**What the agent does:**
1. `bench setup --backend profile` (first time only — installs `nsys`).
2. `bench profile --tool nsys --gpu rtx_pro6000 --backend vllm --model qwen3-vl-32b-fp8`

For video scenarios:
```bash
bench profile --tool nsys --gpu rtx_pro6000 --backend vllm \
    --model qwen3-vl-32b-fp8 \
    --scenarios-dir tests/smoke/scenarios_video/
```

**What to look for:**

For image inference, look for the prefill phase dominating TTFT on long
prompts, and decode dominating on long outputs.

For video inference, three distinct GPU phases appear:

| Phase | NVTX label | What it means |
|---|---|---|
| Vision encoder | `vllm.vision_encoder` | Processing N video frames through the ViT — scales with `num_frames` |
| Prefill | `vllm.prefill` | Attention over image tokens + prompt (TTFT ends here) |
| Decode | `vllm.decode` | Autoregressive token generation |

The vision encoder dominates when `num_frames` is high. If TTFT is high
relative to decode, the encoder or prefill is the bottleneck. If total
latency is high but TTFT is low, decode (output length) is the bottleneck.

---

## Cheat sheet — when something looks wrong

| Symptom | Most-likely cause | Where the agent looks |
|---|---|---|
| `bench setup` exit 4 | pip install failed (network, missing system libs) | stderr of the failed step |
| `bench smoke` exit 3 (runtime) | Server OOM or model download failed | `benchmarks/results/<gpu>/server-logs/<backend>.log` |
| `bench sweep` skips a row with `[skip]` | `unsupported_backends:` matched for this model+backend | The skip message names the reason + `benchmarks/configs/<gpu>.yaml` |
| Model download stalls | HF rate-limit or missing login | `huggingface-cli login` then retry |
| Section 9 (concurrency) missing from summary | No `bench load-test` runs yet | Go back to Step 7 |
| Video smoke passes but key-phrase coverage = 0 | `expected.json` key_phrases too specific or model answered off-target | Raw response in `benchmarks/results/<gpu>/vllm/<scenario>__<run_id>.json` |
| `bench profile` port conflict | vLLM already running on that port | Kill it: `pkill -f "vllm serve"` then retry |

---

## Clean re-run from scratch

```bash
rm -rf .venv-vllm .venv-sglang .venv-trtllm
rm -rf .claude/skills .cursor/rules AGENTS.md
rm -rf benchmarks/results
# HF model cache lives at ~/.cache/huggingface — keep it to avoid re-downloading.
```

Then go back to **Step 0**.

---

## What this walkthrough doesn't cover

Three things deliberately out of scope — flag them so you know expected vs bug:

- **TRT-LLM VLM serving** — most models are pinned out on PRO 6000 today
  (arch-registry gaps, MoE JIT issues). Check `unsupported_backends:` in the
  YAML for the per-model reason. Revisit after a TRT-LLM bump.
- **Video concurrency with real payloads** — `bench load-test` (AIPerf) sends
  synthetic text prompts, which skips the vision encoder. True video concurrency
  measurement with real frames is tracked as a future enhancement.
- **Multi-GPU (tensor-parallel)** — supported via the GPU yaml
  (`tensor_parallel: N`), not tested on single-GPU setups. See
  [docs/gpu-strategy.md](docs/gpu-strategy.md).
