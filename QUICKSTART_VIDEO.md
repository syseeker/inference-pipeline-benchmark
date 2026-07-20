# QUICKSTART — Video+Text Inference Benchmark

**Use case**: given a short video clip and a text question, benchmark how fast each VLM can answer it across serving backends, and find the optimal frame count / backend / model combination for your latency budget.

**GPU**: RTX PRO 6000 Blackwell (96 GB). The steps below work verbatim on RTX 5090 (swap `rtx_pro6000` → `rtx5090`) and H200 (swap → `h200`).

---

## Models in this sweep

| Model | Architecture | Backends | Notes |
|---|---|---|---|
| `qwen3-vl-32b-fp8` | Native 3D-RoPE video tower | vLLM, SGLang | Primary target; best temporal reasoning |
| `qwen3-vl-30b-a3b-fp8` | MoE, native video tower | vLLM, SGLang | Bandwidth stress-test |
| `gemma-4-31b-it-fp8` | Frame-extraction (SigLIP) | vLLM, SGLang | Cross-vendor comparison |
| `nemotron-omni-fp8` | Dedicated video encoder | vLLM only | SGLang blocked on SM_120 fused-MoE |

---

## Step 1: Upload your videos

Your 3 customer videos belong here — one per scenario folder:

```
tests/smoke/scenarios_video/
  01_customer_scene_1/
    video.mp4          ← upload YOUR video here (any name, update video_path below)
    request.json       ← fill in description + prompt
    expected.json      ← fill in key_phrases
  02_customer_scene_2/
    video.mp4
    request.json
    expected.json
  03_customer_scene_3/
    video.mp4
    request.json
    expected.json
```

**How to upload** (from your local machine):

```bash
# scp (replace <host> with the instance IP or hostname)
scp /local/path/to/scene1.mp4 <host>:~/qwenvl-inference-pipeline-benchmark/tests/smoke/scenarios_video/01_customer_scene_1/video.mp4
scp /local/path/to/scene2.mp4 <host>:~/qwenvl-inference-pipeline-benchmark/tests/smoke/scenarios_video/02_customer_scene_2/video.mp4
scp /local/path/to/scene3.mp4 <host>:~/qwenvl-inference-pipeline-benchmark/tests/smoke/scenarios_video/03_customer_scene_3/video.mp4

# Or rsync if you want a progress bar
rsync -avP /local/path/to/scene1.mp4 <host>:~/qwenvl-inference-pipeline-benchmark/tests/smoke/scenarios_video/01_customer_scene_1/video.mp4
```

> Videos are git-ignored (too large to commit). They stay local to the instance.

---

## Step 2: Fill in request.json and expected.json

Open each scenario folder and edit the two files. Example for scene 1:

**`tests/smoke/scenarios_video/01_customer_scene_1/request.json`**
```json
{
  "name": "01_customer_scene_1",
  "description": "Warehouse floor footage — forklift crossing a pedestrian zone.",
  "video_path": "video.mp4",
  "prompt": "What safety risks are visible in this video? Be specific about the objects and their positions.",
  "deadline_ms": 30000
}
```

Fields:
- `video_path` — filename of the video in this folder (default `video.mp4`; change if your file has a different name)
- `prompt` — your text question or instruction for the model (one per scenario)
- `deadline_ms` — per-request timeout in ms; 30000 (30 s) is safe for video inference

**`tests/smoke/scenarios_video/01_customer_scene_1/expected.json`**
```json
{
  "key_phrases": ["forklift", "pedestrian", "collision", "safety zone", "proximity"],
  "min_coverage": 0.6
}
```

Fields:
- `key_phrases` — list of words or short phrases you expect to appear in a correct response
- `min_coverage` — fraction of key_phrases that must appear; 0.6 = pass if ≥ 60% present

Repeat for `02_customer_scene_2/` and `03_customer_scene_3/`.

---

## Step 3: Activate a venv and validate your scenarios are ready

`bench` is a Python entry point — it must be run from a venv where the package
is installed. Activate whichever backend venv you set up (or create a minimal one):

```bash
cd ~/qwenvl-inference-pipeline-benchmark

# Option A — use an existing backend venv (vllm is the most common)
source .venv-vllm/bin/activate && pip install -e . --quiet

# Option B — create a lightweight base venv (no backend, just the CLI)
python3 -m venv .venv && source .venv/bin/activate && pip install -e .
```

Then validate your scenarios:

```bash
bench scenarios build --source video-text --out tests/smoke/scenarios_video/
```

Expected output:
```
[ok] scenarios.build: built 3/3 scenarios
```

If you see a `Video files missing` error, go back to Step 1 and upload the .mp4 files.

---

## Step 4: Start a backend and run a smoke test

Pick one backend to validate the pipeline end-to-end before running the full sweep.

```bash
# Terminal 1 — start vLLM with the default model (Qwen3-VL-32B-FP8)
source .venv-vllm/bin/activate
vllm serve Qwen/Qwen3-VL-32B-Instruct-FP8 \
    --max-model-len 32768 \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 32

# Terminal 2 — smoke test (single scenario, single backend)
bench smoke --gpu rtx_pro6000 --backend vllm --model qwen3-vl-32b-fp8 \
    --scenarios-dir tests/smoke/scenarios_video/
```

A passing smoke test writes a result JSON under `benchmarks/results/rtx_pro6000/`.

---

## Step 5: Run the frame-count sweep

The key optimization lever for video inference is `num_frames` — how many frames
the model sees. More frames = better temporal coverage, higher latency.

Run the full matrix in one shot — all models × all frame counts:

```bash
bench sweep --gpu rtx_pro6000 --sweep video \
    --scenarios-dir tests/smoke/scenarios_video/
```

This runs 4 models × 2–3 backends × 3 frame counts (4 / 8 / 16) = 21 rounds total.
Results let you compare latency and key-phrase coverage across every combination.

If you already know your frame budget and want a faster targeted run:

```bash
# 4 frames — fastest; suited for short clips or obvious single-moment events
bench sweep --gpu rtx_pro6000 --sweep video-4f --scenarios-dir tests/smoke/scenarios_video/

# 8 frames — balanced default
bench sweep --gpu rtx_pro6000 --sweep video-8f --scenarios-dir tests/smoke/scenarios_video/

# 16 frames — highest temporal coverage; latency ~2× vs 8f
bench sweep --gpu rtx_pro6000 --sweep video-16f --scenarios-dir tests/smoke/scenarios_video/
```

Nemotron-Omni runs vLLM-only in all sweeps (SGLang blocked on this GPU — see rtx_pro6000.yaml).

---

## Step 6: Concurrency measurement (NOT AIPerf)

> **Why not `bench load-test` / AIPerf?**
> AIPerf sends synthetic text-only prompts. For video inference that skips the
> most expensive parts of the pipeline — video decode, frame tokenization, and
> the vision encoder. The numbers would measure the serving stack, not the
> workload you actually care about.

The right concurrency measurement for video inference is to run the sweep at
increasing concurrency levels using real video payloads. Use the per-scenario
JSONs already written by `bench sweep` to compare p50/p95 across backends and
frame counts — that IS your concurrency signal at batch=1.

For multi-request concurrency with real video payloads, run multiple parallel
`bench sweep` invocations against a single live server (each in its own terminal)
or use the `--request-count` flag on a custom script that sends your actual
video scenarios. This is tracked as a future enhancement.

---

## Step 7: Profile the bottleneck (Nsight Systems)

If p95 latency from the sweep is unexpectedly high, escalate to Nsight.

> **Unlike load-test**, `bench profile` manages the server lifecycle itself —
> do NOT start vLLM manually first or you will get a port conflict.

```bash
# One-time: install nsys
bench setup --backend profile

# Profile one round against your video scenarios
bench profile --tool nsys \
    --gpu rtx_pro6000 \
    --backend vllm \
    --model qwen3-vl-32b-fp8 \
    --scenarios-dir tests/smoke/scenarios_video/
```

This starts vLLM, runs all 3 video scenarios under Nsight, stops vLLM, and
writes a `.nsys-rep` + `.summary.md` under `benchmarks/results/rtx_pro6000/profiles/`.

**What to look for in the Nsight Systems UI for video inference:**

The timeline will show vLLM startup (~82s) followed by 3 inference runs. Zoom
into a single inference call. You will see three distinct GPU phases:

| Phase | NVTX label | What it means |
|---|---|---|
| Vision encoder | `vllm.vision_encoder` | Processing the N video frames through the ViT — scales with `num_frames` |
| Prefill | `vllm.prefill` | Attention over all image tokens + text prompt tokens (TTFT ends here) |
| Decode | `vllm.decode` | Autoregressive token generation |

For video inference, **the vision encoder dominates** when `num_frames` is high.
If TTFT is high relative to decode, the encoder or prefill is the bottleneck.
If total latency is high but TTFT is low, decode (output length) is the bottleneck.

> **Note**: Nsight is valid for video inference — unlike AIPerf, it wraps real
> video requests (the same base64-encoded frames used by the sweep) so the
> encoder and prefill phases are faithfully captured.

---

## Step 8: Read the summary

```bash
bench summary --gpu rtx_pro6000
```

Reads `benchmarks/results/rtx_pro6000/summary.md` — the p50/p95 table, winner,
and per-backend findings. The key-phrase coverage rate per scenario is included
so you can see quality alongside latency.

---

## Changing the text prompt or video path after setup

All edits live in the scenario `request.json` files — no code change needed:

| What you want to change | Where to edit |
|---|---|
| The text question for scene 1 | `tests/smoke/scenarios_video/01_customer_scene_1/request.json` → `"prompt"` field |
| The video file for scene 2 | Upload a new file, then update `"video_path"` in `02_customer_scene_2/request.json` |
| Expected answer keywords | `expected.json` → `"key_phrases"` list |
| Pass/fail threshold | `expected.json` → `"min_coverage"` (0.0–1.0) |
| Per-request timeout | `request.json` → `"deadline_ms"` |

---

## FAQ

**Q: Can I use a remote video URL instead of uploading the file?**
Set `"video_url": "https://..."` in request.json and omit `"video_path"`. The
serving framework fetches it directly. Only works if the instance has outbound
internet access and the URL is publicly reachable.

**Q: How do I add a 4th video scenario?**
Create `tests/smoke/scenarios_video/04_my_scene/`, drop in `video.mp4`,
`request.json`, and `expected.json` following the pattern above. No code change.

**Q: What if the model response doesn't contain any key phrases?**
Either (a) the model genuinely answered off-target — check the raw response in
the per-scenario JSON under `benchmarks/results/rtx_pro6000/vllm/`; or (b) your
key_phrases are too specific. Try shorter, more common words.

**Q: Can I run just one model to iterate faster?**
Yes — use `--model` to pin a single model instead of running the full sweep:
```bash
bench sweep --gpu rtx_pro6000 --sweep video-8f --model qwen3-vl-32b-fp8 \
    --scenarios-dir tests/smoke/scenarios_video/
```
