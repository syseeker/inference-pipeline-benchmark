# Multi-model GPU contention benchmark — audit and build plan

> **Approved 2026-07-31**, after two review rounds (38 comments). The execution
> plan follows the customer's own Phase 0–6 structure. Model substitutions are
> *suggestions only* — the customer's picks are honoured, scoped per GPU, or
> made to fail safely with a reason. Nothing is silently swapped or deleted.
>
> Durable knowledge from this plan is split out into the sibling references:
> [design-decisions.md](design-decisions.md), [model-catalogue.md](model-catalogue.md),
> [serving-topology.md](serving-topology.md). This file is the *build sequence*;
> those are the *conclusions*.

## Context

A customer wants to know how heterogeneous AI workloads (text LLM, video-VLM,
image-LM, computer vision) degrade each other when sharing a GPU. They supplied
`experiment_design.md`, `experiment_config.json`, `prepare_data.py` and a staged
test-data tree.

**The deployment target is consumer / prosumer hardware** — RTX GeForce, RTX PRO
6000, DGX Spark — so models must stay small enough that **1 CV + 1 LLM/VLM fit
together**. This is not a "best latest model" study; it is a co-residency study.
The framework must let the customer swap in newer models themselves later.

This repo benchmarks **one model at a time** — `run_all_scenarios.sh` actively
prevents co-residency (`preflight_gpu` refuses <30 GB free; `cleanup` kills the
server every round). Contention is genuinely new capability.

### Decisions taken

| Decision | Choice |
|---|---|
| Serving topology | **Hybrid** — Triton for CV, native vLLM/SGLang/TRT-LLM for LLM/VLM |
| llama.cpp | **Dropped** |
| GPUs | RTX 5090 (32 GB) · RTX PRO 6000 (96 GB) · H200 (141 GB), at 1/2/4 |
| Broken models | Keep in config, scope per GPU, fail safely with a reason; suggest substitutes, don't impose |
| Isolation (MPS/MIG) | **Not a dimension.** Pick the best available config per GPU and hold it fixed |
| Quantization dimension | **Removed.** Pick the best-fitting weight format per model per GPU |
| Load shape | **Open-loop fixed-rate** everywhere (see §4.1) |
| Docs / learnings | Live in **repo `.claude/`**, authored in [agentskills.io](https://agentskills.io/home) skill format |

---

## 1. Answers to review questions

### 1.1 Two video-generation prompts

> **Status: clips generated, pending upload.** Drop them into
> `workspace/contention/test_data/source/`. Build step 2 then verifies the codec
> and transcodes each to its target spec — it does not regenerate them.
>
> ```bash
> # A → short-clip: 224², 3 s, 1 fps, 3 frames
> ffmpeg -i source/<clip_a>.mp4 -vf "scale=224:224,fps=1" -t 3 \
>        -c:v libx264 -pix_fmt yuv420p -an vlm/clip_3s_224.mp4
> # B → long-clip: 720p, 10 s, 4 fps, 40 frames
> ffmpeg -i source/<clip_b>.mp4 -vf "scale=1280:720,fps=4" -t 10 \
>        -c:v libx264 -pix_fmt yuv420p -an vlm/clip_10s_720p.mp4
> ```
> `-c:v libx264` is the part that matters — it replaces the `mp4v` encoding that
> would otherwise fall back to CPU decode and contaminate the measurement.

The two clips have different jobs, so they are generated as two sources rather
than derived from one. Target ≥12 s at ≥1080p so downscaling has headroom;
**H.264/MP4 output**.

**Clip A — "short-clip" source** (becomes 224×224, 3 s, 1 fps, 3 frames). Needs a
single legible action, because at 1 fps the model sees only 3 stills.

> A fixed, locked-off camera shot of a modern office desk. A person's hands enter
> frame from the right, pick up a red ceramic coffee mug, and place it down on a
> stack of white papers next to a silver laptop. Even, bright daylight from a
> window on the left. Sharp focus, no camera movement, no zoom, no cuts. 12
> seconds, 1080p, photorealistic.

**Clip B — "long-clip" source** (becomes 1280×720, 10 s, 4 fps, 40 frames). Needs
sustained motion and several distinct objects, so temporal reasoning and the
vision encoder both do real work.

> A busy warehouse aisle seen from a fixed elevated camera. A yellow forklift
> drives slowly from the left of frame to the right, carrying a wooden pallet
> stacked with brown cardboard boxes. Two workers in orange high-visibility vests
> and white hard hats walk in the opposite direction along a painted yellow floor
> line, one of them pausing to check a clipboard. Tall metal shelving racks filled
> with inventory line both sides. Bright industrial overhead lighting. Continuous
> single take, no cuts, no camera movement. 15 seconds, 1080p, photorealistic.

Clip B intentionally matches the `key_phrases` style already used in
`tests/smoke/scenarios_video/` — forklift, pedestrian, proximity — so the same
coverage-scoring path works unchanged.

### 1.2 What "4-channel decode" means — and no, don't delete the PNG

**Keep `document.png`.** The note was about *how it is loaded*, not the file.

A normal photo has 3 channels — Red, Green, Blue. A PNG can carry a 4th, **alpha**
(transparency). `source/document.png` is RGBA, so 33% more bytes move through
decode and preprocessing for a channel every vision model then discards. Two
practical consequences: it is slightly slower, and many preprocessing pipelines
assert exactly 3 channels and **raise an error** on a 4-channel array.

Fix is one call at load time — `Image.open(p).convert("RGB")` — not a change to
the asset. The file stays exactly as it is.

### 1.3 `prepare_data.py` — you're right, not important

It is the customer's own generator: it derives every `cv/` and `vlm/` file from
the three `source/` assets. **The outputs already exist and are spec-correct**, so
we never need to run it. It only matters if you want to *regenerate* on the GPU
box — and then it needs `opencv-python`, which isn't installed here.

Handled by adding a contention requirements file (§1.4) rather than by changing
their script.

### 1.4 Dependencies to install on the GPU instance

New `requirements-contention.txt`, installed before a run:

```
opencv-python>=4.10      # prepare_data.py transcode path
pillow>=10.2             # already present
tritonclient[grpc,http]  # Triton CV tenants
ultralytics>=8.4         # YOLO
onnx, onnxruntime-gpu    # CV export + portable baseline
aiperf>=0.11             # LLM/VLM load generation
```

Plus two **system** packages, not pip-installable: `ffmpeg` (video transcode) and
`perf_analyzer` (ships in Triton's SDK container — see §1.10). These follow the
repo's existing convention for system tools, documented in the README.

### 1.5 Where the research file is → moving into the repo

Currently `/home/boonpingl/.claude/plans/analyse-the-attached-3-toasty-quokka-agent-abdf1f83b848d3c28.md`
(model-availability research) and this plan, both on the laptop.

**Per your instruction, all of it moves into repo `.claude/`**, authored in
agentskills.io format:

```
.claude/skills/gpu-contention-benchmark/
  SKILL.md              # the how-to: run a contention sweep end to end
  reference/
    experiment-design.md      # customer's original brief (verbatim)
    design-decisions.md       # §4 methodology + WHY, in plain language
    model-catalogue.md        # verified model audit + sources
    serving-topology.md       # Triton vs vLLM vs MPS/MIG explained
  assets/
    experiment_config.json    # customer's original config
    colocations.example.yaml  # the full combination matrix
```

This is also the answer to *"state this design decision somewhere the user can
read and understand"* — `reference/design-decisions.md` is that document, and it
travels with the repo instead of living in a chat log.

### 1.6 NMS and "NMS-free", plainly

An object detector scores **thousands** of candidate boxes across a dense grid,
so a single dog produces ~40 overlapping "dog" boxes. **Non-Maximum Suppression**
is the cleanup pass: sort by confidence, keep the best box, delete everything
overlapping it beyond a threshold, repeat until none remain.

NMS is **sequential and data-dependent** — each decision depends on the previous
one, and the amount of work depends on the picture. A crowded frame costs more
than an empty one.

**NMS-free** (YOLO26's default) trains the model to emit one box per object
directly, so the output is a fixed `[B, 300, 6]` tensor you simply threshold.
Why that matters *for a contention benchmark specifically*: NMS makes the CV
tenant's own latency vary with scene content. You want the CV side's intrinsic
variance near zero, so that the variance you measure is **contention** and not
"this frame happened to be busy."

### 1.7 YOLOv8 vs YOLO26 — the numbers

Both re-benchmarked by Ultralytics on identical hardware (T4 / TensorRT10, 640px, COCO val):

| Model | mAP50-95 | Params | FLOPs | T4 TRT (ms) |
|---|---|---|---|---|
| YOLOv8n | 37.3 | 3.2 M | 8.7 B | **1.47** |
| YOLO26n | **40.9** | 2.4 M | 5.4 B | 1.7 |
| YOLOv8l | 52.9 | 43.7 M | 165.2 B | 9.06 |
| YOLO26l | **55.0** | 24.8 M | 86.4 B | **6.2** |

**At nano scale YOLO26 is actually *slower* on GPU** (1.47 → 1.7 ms) — you are
launch-overhead-bound there, so +3.6 mAP costs a little GPU time. At large scale
YOLO26 wins on both (1.46×).

**YOLOv8 is still fully supported** — current `ultralytics` 8.4.113 shipped
2026-07-30. **Recommendation: keep the customer's YOLOv8n + YOLOv8l as specified.**
They work, they are the requested pipeline shape, and at nano scale v8 is the
faster GPU tenant anyway. Note YOLO26 to the customer as an option, don't impose it.

### 1.8 PaddleOCR — where that config pin comes from

The TensorRT 8.6.1.6 + CUDA 11.8 pin is **PaddleOCR's own documented requirement**,
not anything to do with the PRO 6000. The problem is that it collides with the
stack Triton 26.07 ships (CUDA 13.3, TensorRT 11.0) — you cannot run both in one
environment.

**Two honest options, customer's call:**
- **Follow the request** — run PaddleOCR via Triton's **Python backend** (plain
  Paddle inference, no TensorRT). Works on the current stack, just not the
  TRT-optimised path.
- **Drop it** from the matrix and note why.

Either way we standardise on **one** TensorRT + CUDA environment for all CV
models. PP-OCRv6 is mentioned as a substitute; not imposed.

### 1.9 RF-DETR — flag to client, likely swap

`RFDETRBase` was deprecated 2025-07-23 in favour of `RFDETRMedium`. Medium is
close in size to the old Base, so this is a low-risk swap that keeps the
customer's intent intact. **Recommendation: use `RFDETRMedium`, tell the customer
why.** If they prefer, pin the old `rfdetr` release and keep Base.

### 1.10 Did the customer name PaliGemma2? — **Yes, they did**

Straight from their `experiment_config.json`:

```json
"gemma-vlm-32b": { "hf_id": "google/paligemma2-28b-pt-896", ... }
```

So it is their pick, not my suggestion. The problems are still real, and worth
telling them: it is **28B not 32B**; it is a `-pt-` **base** checkpoint that
Google says to use only after fine-tuning; and it is **image-only** — vLLM's
`paligemma.py` raises `ValueError("Only image modality is supported")`. It cannot
serve the video dimension it was chosen for.

Since the design needs a *video* VLM and the target is small models, the natural
replacement inside their own list is **Qwen2.5-VL-7B**, which does video and has
an official AWQ checkpoint.

### 1.11 Kosmos-2.5 → Triton Python backend

Confirmed. It has no vLLM or SGLang implementation — not in vLLM's model
registry, no `kosmos*.py` in either tree. We serve it through **Triton's Python
backend** alongside the CV models, and the YAML comment simply reads *"vLLM/SGLang
support not available — served via Triton Python backend."* No editorialising.

### 1.12 Can AIPerf do open-loop and arrival patterns? — **Yes. This is the good news.**

AIPerf v0.11.0 already has everything the design needs on the LLM/VLM side:

| Need | Flag |
|---|---|
| Open-loop fixed rate | `--request-rate <rps>` |
| Arrival pattern | `--arrival-pattern {constant,poisson,gamma}` (Poisson is default) |
| Burstiness control | `--arrival-smoothness` (<1 bursty, 1 Poisson, >1 smooth) |
| Rate + concurrency ceiling | both flags together |
| Real image/video payloads | `--input-file` + `--custom-dataset-type single_turn` |
| Reproducibility | `--random-seed` |

**So we do not build a load generator.** That removes an entire phase of work.

### 1.13 ⚠️ But AIPerf **cannot** drive Triton — the one real blocker

AIPerf dropped GenAI-Perf's `kserve` and `dynamic_grpc` endpoint types. Its
`--endpoint-type` list is all OpenAI/NIM/HF-shaped HTTP. It references Triton
**only** as a Prometheus scrape target (`--server-metrics`), never as a load target.

**Resolution: two load generators, one wall clock.**

| Tenant | Driver | Why |
|---|---|---|
| LLM / VLM (vLLM, SGLang, TRT-LLM) | **AIPerf** | OpenAI-compatible; Poisson + rate built in |
| CV (Triton) | **`perf_analyzer`** | Still maintained, not deprecated. Has `--request-rate-range`, `--request-distribution {constant,poisson}`, `--input-data <real images>`, `--service-kind triton` |

Both support open-loop Poisson, so the two tenants are directly comparable. All
we build is the **orchestrator** that starts them together, holds a common
`t0`, and merges their outputs — not the load generation itself.

### 1.14 Best server for CV models today

**Triton is not deprecated.** v2.71.0 shipped 2026-07-29; monthly cadence
unbroken; ONNX Runtime, TensorRT and Python backends all had commits this month.
"NVIDIA Dynamo-Triton" is a **rebrand**, not a retirement.

The deprecation I mentioned was narrower than it sounded, and there is a name
collision worth being careful about:

| "TensorRT backend" | Status |
|---|---|
| **Triton's `tensorrt_backend`** — runs your YOLO/DINOv2 `.plan` files | **Alive**, actively developed (gained multi-GPU in 26.07) |
| **TRT-LLM's internal engine-build backend** — LLM only | Removed in TRT-LLM 1.2 |

**Impact on CV serving: none.** A CV pipeline never imports `tensorrt_llm`.

**Recommendation: Dynamo-Triton `26.07-py3`, TensorRT backend (optimised) plus
ONNX Runtime backend (portable baseline), driven by `perf_analyzer`.** It gives a
real per-request boundary with server-side decomposition (queue / compute-input /
compute-infer / compute-output) — exactly the attribution a contention study needs.

⚠️ **Gotcha:** client shared memory is **disabled by default since Triton 26.04**.
Pass `--allow-client-shm=true`, or large CV tensors will look like a model
regression when it is really serialization overhead.

### 1.15 Clock pinning — all three GPUs, with caveats

Not PRO 6000 only. `nvidia-smi -lgc` is documented Volta+ and works on all three.
**All clock/power commands need root.**

| | RTX 5090 | RTX PRO 6000 | H200 |
|---|---|---|---|
| `-lgc` (lock graphics clock) | Yes, but **advisory** on GeForce | Yes | Yes |
| `-lmc` (lock memory clock) | Effectively a no-op (GDDR7 fixed in P0) | Yes | **Unsupported** — needs `--lock-memory-clocks-deferred` |
| `-pl` (power limit) | Yes (575 W default) | Yes (600 W, 300 W Max-Q) | Yes (700 W SXM) |

**Chosen policy — pick the best config rather than adding a dimension:** lock
clocks at **60–80% of max boost** (locking at max under a power cap silently
throttles anyway), pin power limit first, then clocks, then warm up and verify.

Because the lock is advisory on GeForce, we do both belt and braces: **record the
achieved clock distribution with every result**, and **assert on
`clocks_throttle_reasons.active`** — fail the run if `SwPowerCap` or
`HwThermalSlowdown` fired, rather than publishing a throttled number as if it
were contention.

### 1.16 MPS / MIG — not a dimension, just the best setting

Agreed. Only PRO 6000 has MIG (5090 is consumer — no MIG; Spark likewise). So per
GPU we fix one isolation config and hold it:

| GPU | Isolation used | Note |
|---|---|---|
| RTX 5090 | MPS on | No MIG available |
| RTX PRO 6000 | MPS on | MIG available; recorded as a **one-off reference run**, not a swept dimension |
| H200 | MPS on | |

The single PRO 6000 MIG run is worth keeping because it is the *hardware-isolated
1.0× reference* — the ceiling everything else is measured against. It costs one
run, not a dimension.

---

## 2. Model catalogue

**Governing principle from review:** the customer chose these for a reason —
small models that co-reside on consumer GPUs. Substitutions are **suggestions to
raise with them**, never silent swaps. Where a pick is factually broken, comment
it out with a one-line reason.

### 2.1 Keep — verified, and they fit

| ID | HF source | 4-bit | Fits 32 GB w/ CV? |
|---|---|---|---|
| `qwen2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` | `…-AWQ` | ✓ |
| `qwen2.5-14b` | `Qwen/Qwen2.5-14B-Instruct` | `…-AWQ` | AWQ only |
| `gemma2-9b` | `google/gemma-2-9b-it` (gated) | — | marginal at BF16 |
| `llama3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` (gated) | — | ✓ |
| `mistral-7b` | `mistralai/Mistral-7B-Instruct-v0.3` | — | ✓ |
| `qwen2.5-vl-7b` | `Qwen/Qwen2.5-VL-7B-Instruct` | **`…-AWQ` official** | ✓ · **video ✓** |
| `yolov8-n` / `yolov8-l` | `ultralytics` 8.4.x | — | ✓ |
| `dinov2-base` / `-large` | `facebook/dinov2-{base,large}` | — | ✓ |

### 2.2 Nothing is dropped — models are scoped per GPU, or skipped with a reason

**Rule from review: never comment out a model that still runs on one of our three
test GPUs.** The customer's deployment target is consumer hardware, but our test
bed is 5090 / PRO 6000 / H200 — so a model that doesn't fit 32 GB still belongs
in the PRO 6000 and H200 configs.

The repo already has exactly the right mechanisms for this, and we use both
rather than inventing anything:

1. **Per-GPU YAMLs** (`benchmarks/configs/{rtx5090,rtx_pro6000,h200}.yaml`) — a
   model appears only in the tiers where it fits. No comment-out needed.
2. **`unsupported_backends:`** — a per-model `{backend: reason}` map. The sweep
   **skips the row cleanly with the reason printed and exit code 2**
   (`scenario_config.py:232`, `run_all_scenarios.sh:348`). This is the
   "throw an error and safely return" behaviour, already built.

| Model | 5090 (32 GB) | PRO 6000 (96 GB) | H200 (141 GB) | Note |
|---|---|---|---|---|
| `qwen2.5-7b` | ✓ | ✓ | ✓ | |
| `qwen2.5-14b` | ✓ AWQ | ✓ | ✓ | |
| `qwen2.5-32b` | — | ✓ | ✓ | ~22 GB AWQ leaves no CV headroom on 32 GB |
| `qwen2.5-72b` | — | ✓ AWQ | ✓ AWQ | 45 GB AWQ; BF16 needs ≥2 GPU |
| `gemma2-9b` | ✓ | ✓ | ✓ | |
| `llama3.1-8b` | ✓ | ✓ | ✓ | |
| `mistral-7b` | ✓ | ✓ | ✓ | |
| `qwen2.5-vl-7b` | ✓ AWQ | ✓ | ✓ | official AWQ · video ✓ |
| `qwen2.5-vl-72b` | — | ✓ AWQ | ✓ AWQ | |
| `kosmos-2.5` | ✓ | ✓ | ✓ | **Triton Python backend** — vLLM/SGLang unavailable |
| `paddleocr` | ✓ | ✓ | ✓ | **Triton Python backend** — its TRT path pins TensorRT 8.6.1.6 + CUDA 11.8, incompatible with the Triton 26.07 stack |
| `rfdetr-medium` | ✓ | ✓ | ✓ | `RFDETRBase` deprecated 2025-07-23 → **Medium**, similar size |
| `yolov8-n` / `-l` | ✓ | ✓ | ✓ | as specified |
| `dinov2-base` / `-large` | ✓ | ✓ | ✓ | |

**Two special cases:**

- **`gemma-vlm-32b` (`paligemma2-28b-pt-896`) — kept in config, fails safely.**
  It loads and serves images fine, so it stays. But it is image-only, so any
  **video** round gets an `unsupported_backends`-style entry: the run skips with
  `"paligemma2 is image-only — vLLM raises 'Only image modality is supported'; a
  video VLM is required for this dimension"` and returns exit code 2 rather than
  crashing mid-sweep. The design doc records that this pick cannot serve the
  video dimension it was chosen for.

- **`qwen2.5-27b` — the one genuine correction.** The HF repo does not exist, so
  there is nothing to run or skip. The Qwen2.5 ladder is 0.5/1.5/3/7/14/**32**/72B.
  Config uses `qwen2.5-32b` with a comment recording the original entry.

### 2.3 Suggestions to raise with the customer — not applied

Newer small models exist that fit the same brief and would age better: Qwen3-VL
2B/4B/8B (official FP8 at every size), Qwen3.5 4B/9B (natively multimodal, and a
~4.5× smaller KV cache — which matters a lot under co-residency), Gemma 4 12B.
**Not adopted.** The framework lets the customer swap these in themselves later,
which was the stated goal.

---

## 3. Dimensions

| # | Dimension | Verdict |
|---|---|---|
| 1 | Model identity | ⚠ See §2 — scoped per GPU; two picks fail safely; one genuine correction |
| 2 | Model type mix | ✓ 8 mixes retained; ILM slot = kosmos-2.5 on Triton |
| 3 | Concurrency | ⚠ **Open-loop rate everywhere** — see §4.1 |
| 4 | GPU count | ✓ 1/2/4 |
| 5 | Output length (32 / 512) | ✓ |
| 6 | Input size | ✓ after the video re-generation (§1.1) |
| 7 | Quantization | ✗ **Removed.** "Q4_0" is a llama.cpp GGUF format vLLM cannot load. Pick the best-fitting format per model per GPU instead |
| 8 | Inference backend | ✓ LLM: vllm / sglang / trtllm. CV: tensorrt / onnx / python (Triton's own backends) |
| 9 | Load asymmetry | ✓ free — two independent rate-controlled generators |
| 10 | Arrival pattern | ✓ free — `--arrival-pattern` / `--request-distribution` |

No dimensions added. MPS/MIG and clock policy are **fixed best-settings**, §1.15–1.16.

---

## 4. Methodology — the decisions, and why

> Goes verbatim into `.claude/skills/gpu-contention-benchmark/reference/design-decisions.md`
> so it is readable outside this plan.

### 4.1 Why "concurrency" doesn't mean what the design assumes — a game example

The design drives LLMs by **concurrency** ("keep 4 requests in flight") and CV by
**request rate** ("50 images per second"). Those are different experiments, and
mixing them breaks the comparison.

**Think about a game.** The world sends the AI a decision request every 100 ms —
whether or not the last one has come back. That is **open-loop**: the world does
not wait. If inference slows down, requests pile up and the character visibly lags.

Now the closed-loop version: "always keep exactly 4 requests in flight." When the
GPU slows down, the client **automatically sends more slowly** — it is waiting for
replies before issuing new ones. Latency per request looks almost unchanged. The
system appears healthy.

That is the trap. **A closed-loop baseline hides the damage**, because the client
throttles itself in exact proportion to the slowdown. The degradation ratio then
measures the test harness, not the GPU. And no real workload behaves that way —
game engines, video pipelines and camera feeds all push at their own rate.

**Decision: open-loop fixed rate for every tenant, solo and co-resident.** Record
`offered_rps` and `achieved_rps`. When `achieved < offered`, the tenant can no
longer keep up — and **that point is the safe-operating-envelope boundary**, which
the design wants and which this gives us for free.

### 4.2 Clock throttling is not contention

Under co-residency power rises, the GPU hits its cap, clocks drop, everything
slows. Real, but not contention — and a contention matrix that reports it is
wrong. **Decision:** pin clocks per §1.15, record achieved clocks with every
result, and fail runs where a throttle reason fired.

### 4.3 One GPU sampler per window, not one per tenant

`GpuSampler` is currently created per round (`runner.py:462`). With N tenants
that means N `dcgmi dmon` processes on one GPU, N different windows, and every
tenant reporting the **whole GPU's** memory as its own. **Decision:** the
orchestrator owns one sampler spanning the union window; whole-GPU numbers attach
to the *colocation*, not to each tenant row. **Amended in Phase 5** (see
design-decisions §3): one sampler per *card the colocation occupies*, keyed by
device in the manifest — one per tenant is still forbidden.

### 4.4 Per-request timestamps

Only durations are stored today, so two tenants' request streams cannot be
aligned on a shared timeline — and without that you can never show that tenant
A's p99 spike lands inside tenant B's prefill window. **Decision:** add
`start_epoch_ms` / `end_epoch_ms` to `LatencySamples` (`metrics.py:23`), populated
in `runner.py:_run_one`. `time.time()` for the shared timeline, `perf_counter()`
for durations — never conflated.

### 4.5 Repetition policy from data, not assumption

The design assumes <5% variance and repeats only memory-pressure scenarios. That
figure is for **solo** inference; co-residency adds scheduler non-determinism and
allocator ordering. **Decision:** run one colocation 5× in Phase 0, report the
spread, and set the policy from the measurement.

### 4.6 Sampling interval

250 ms default vs a 200–500 ms prefill burst is 1–2 samples per event.
**Decision:** 50 ms for alignment runs (already supported, `runner.py:770`).

---

## 5. Metrics — how this differs from `docs/metrics.md`

Review asked exactly this. Short answer: **the per-request metrics are
unchanged**; contention adds three things on top.

| Layer | Source | Status |
|---|---|---|
| Per-request LLM/VLM — TTFT, ITL, e2e, tok/s | `docs/metrics.md` §1–3, `metrics.py:23` | **Reused as-is** |
| Per-request CV — inference, preprocess, postprocess, batch latency | *(new)* — Triton reports these natively as queue / compute-input / compute-infer / compute-output | **New, but free from Triton** |
| GPU telemetry — util, mem-bw, power, VRAM | `docs/metrics.md` §5, `gpu_sampler.py` | **Reused**, + clock fields (§4.2) |

Three additions, and only three:

1. **Co-tenancy identity** — which tenants shared the GPU, so rows can be paired
   with their solo baseline. Without this a result is uninterpretable.
2. **Degradation ratios** — `contention / solo` per metric. The repo already has
   this shape (`cuda_graph_speedup`, `tp_efficiency` at `metrics.py:138-141`) —
   computed in `summary.py`, not the runner. Same precedent, same place.
3. **Shared timeline** — per-request epoch timestamps (§4.4) so tenant traces can
   be aligned against each other and against the GPU trace.

Plus one carried over from their `fixed_methodology`: **model load measurement**
(`time_to_first_ready_s`, `vram_after_load_gb`), captured passively when each
tenant starts. Cold-start cost never shows up in inference metrics but decides
whether a co-residency plan is deployable.

Against the customer's `experiment_design.md` "Metrics Collected" section:
everything they list is covered. Their **Degradation** table (`throughput_ratio`,
`latency_ratio`, `ttft_ratio`, `itl_ratio`, `p99_latency_ratio`) maps 1:1 onto
addition 2. Their **Metrics Reporting** policy (P50/P95/max always; P99 and
mean±std only for repeated scenarios) is adopted unchanged — it is sound, and the
existing `percentile()` at `metrics.py:176` already computes it.

---

## 6. Execution plan — the customer's Phase 0–6

Structure preserved. Changes are to *scale* and *load shape*, not sequence.

### Phase 0 — Concurrency validation (gate)

**Unchanged in intent.** Validate that co-resident models genuinely overlap on
the GPU rather than serialising. If they serialise, the study becomes "measure
time-slice fairness" — a different result — so this stays a hard gate.

Adjustments: test **MPS on** (the chosen config) rather than bare co-residency;
add the **variance measurement** (§4.5, one colocation 5×); verify the
**H.264 clips** decode on NVDEC; confirm per-tenant VRAM caps hold (vLLM defaults
to `gpu_memory_utilization=0.9` and will otherwise take the whole card).

Pass criteria unchanged: overlapping SM activity from both tenants in the same
window. **~6 runs, 0.5 h.**

### Phase 1 — Solo baselines

**Unchanged in intent**, with one critical fix: baselines must run at the **same
offered rate** as the contention runs (§4.1), or every ratio is an artifact.

Scale drops from 41 runs because the quantization dimension is gone (§3) and the
32B/72B tier is out of the consumer-GPU scope. ~8 LLM/VLM + 5 CV = **~13 runs, 1.5 h.**

### Phase 2 — Concurrency sweep

**Reinterpreted as a rate sweep** — same curve, correct instrument. Sweep offered
RPS per category and find where `achieved_rps` departs from `offered_rps`. That
knee is the saturation point the design is looking for. **~12 runs, 1.5 h.**

### Phase 3 — Model type mix

**Unchanged** — all 8 mix types at a fixed rate. The ILM slot is kosmos-2.5 on
Triton. **8 runs, 2 h.**

### Phase 4 — Cross-type contention

**Unchanged in intent** — one subject, sweep the neighbour's load. Keeps the
LLM-vs-CV, VLM-prefill-vs-LLM, ILM-vs-CV and CV-vs-LLM sweeps, the size-scaling
ladder (7B → 14B, within the consumer-GPU tier) and the cross-architecture check
(Qwen vs Llama vs Mistral).

**Memory-pressure curve moves to PRO 6000 / H200** — it is anchored on 72B, which
does not fit 32 GB. On the 5090 the equivalent pressure point is 14B + a CV
tenant. Repetitions set by Phase 0's variance measurement rather than assumed.
**~16 runs, 2.5 h.**

### Phase 5 — GPU scaling

**Unchanged**, at 1/2/4 GPUs (8 dropped — not available). Key scenarios from
Phases 3–4 repeated. **~12 runs, 2 h.**

### Phase 6 — Secondary dimension confirmation

**Unchanged in structure** — the dual-baseline design (compute-bound vs
memory-bound) is good and stays; it is what surfaces interaction effects.

Dimensions swept drop from 7 to 5: quantization removed (§3), llama.cpp removed,
arrival pattern and asymmetry now free via the generators. Remaining: output
length, input size, LLM backend, CV backend, load asymmetry. **~16 runs, 2 h.**

### Cross-check against `experiment_config.json` → `phases`

Read their phase definitions for anything still relevant that rev 1 dropped.
Four things picked up, the rest already covered:

| Picked up | From | Why it stays |
|---|---|---|
| **`model_load_measurement`** — `time_to_first_ready_s`, `vram_after_load_gb` | `fixed_methodology` | Cold-start cost is invisible to inference metrics but decides deployment. Captured passively at tenant startup; costs nothing. |
| **`validate-standalone-vs-triton`** — the <5% wrapper-overhead check | `phase0`, exp 3 | Its original form is moot (we no longer put LLMs in Triton), but the *intent* is right and still needed: confirm the **isolation config itself doesn't distort solo baselines**. Re-purposed as a solo tenant measured with MPS off vs on. |
| **Reduced rate at the extreme memory point** — "runs at c2 to avoid immediate OOM" | `phase4`, `cross-memory-pressure` | Sensible guard. The near-OOM point runs at half rate so the tenant set loads at all. |
| **`metrics_reporting`** — P50/P95/max always; P99 and mean±std only for repeated runs | `fixed_methodology` | Statistically honest given sample counts. Adopted unchanged (§5). |

Already covered and unchanged: the Phase 3 mix rosters, Phase 4's four
neighbour-sweeps plus size-scaling and cross-architecture validation, Phase 6's
dual-baseline (compute-bound `baseline_A` / memory-bound `baseline_B`) design,
warmup 3 / measurement 10 rounds, and the 120 s request timeout.

Deliberately not carried over: `serving_platform: triton` as a *global* fixed
value (we use it for CV only — §1.13), the `quantization` dimension (§3), and
`llamacpp` from `inference_backend_llm`.

### Total

| Phase | Runs | Hours |
|---|---|---|
| 0 Gate + variance | 6 | 0.5 |
| 1 Solo baselines | 13 | 1.5 |
| 2 Rate sweep | 12 | 1.5 |
| 3 Model type mix | 8 | 2.0 |
| 4 Cross-type contention | 16 | 2.5 |
| 5 GPU scaling | 12 | 2.0 |
| 6 Secondary dimensions | 16 | 2.0 |
| **Total** | **~83** | **~12 h** |

Down from 155 runs / 22–34 h, mostly by removing the quantization dimension,
dropping models that cannot fit the consumer-GPU target, and getting arrival
patterns and asymmetry free from AIPerf instead of building them.

---

## 7. The combination matrix — `colocations:` schema

> Review liked this section and wants every possible experiment combination to be
> obvious from one file. This is that file.

### 7.1 Workload inputs — the customer's prompts and test data

`experiment_config.json` already defines both the prompt sets and a
`prompt_to_data_mapping` that pairs each workload with its files. We adopt both
verbatim as a `workloads:` block rather than inventing new ones — this is what
each tenant's `workload:` key resolves to.

```yaml
workloads:
  llm_short:     # 3 prompts, ~50 tok   — "What is CUDA?" …
    prompts: [workspace/contention/prompts/llm_short.jsonl]
    data: null                                    # text only
    output_tokens: 32
  llm_long:      # 2 prompts, ~1000 tok — GPU memory hierarchy; distributed inference spec
    prompts: [workspace/contention/prompts/llm_long.jsonl]
    data: null
    output_tokens: 512
  vlm_video_short:   # 2 prompts × short clip
    prompts: [workspace/contention/prompts/vlm_video.jsonl]
    data: [workspace/contention/test_data/vlm/clip_3s_224.mp4]      # 224², 3 s, 1 fps, 3 frames
  vlm_video_long:
    prompts: [workspace/contention/prompts/vlm_video.jsonl]
    data: [workspace/contention/test_data/vlm/clip_10s_720p.mp4]    # 720p, 10 s, 4 fps, 40 frames
  ilm_document:      # 2 prompts — extract fields / identify doc type
    prompts: [workspace/contention/prompts/ilm_document.jsonl]
    data: [workspace/contention/test_data/cv/sample_document.png]
  cv_detect_small:   # yolov8 / rfdetr
    data: [workspace/contention/test_data/cv/sample_320x320.jpg]
  cv_detect_large:
    data: [workspace/contention/test_data/cv/sample_1280x1280.jpg]
  cv_embed:          # dinov2 — fixed 224 input
    data: [workspace/contention/test_data/cv/sample_224x224.jpg]
  cv_ocr:            # paddleocr
    data: [workspace/contention/test_data/cv/sample_document.png]
```

Which workload each dimension exercises:

| Dimension / phase | Workload used |
|---|---|
| D5 output length | `llm_short` (32 tok) vs `llm_long` (512 tok) |
| D6 input size — LLM | `llm_short` (~50 tok) vs `llm_long` (~1000 tok prefill) |
| D6 input size — VLM | `vlm_video_short` (3 frames) vs `vlm_video_long` (40 frames) |
| D6 input size — CV | `cv_detect_small` (320²) vs `cv_detect_large` (1280²) |
| Phase 4 VLM-prefill-vs-LLM | `vlm_video_long` — the 40-frame encode is the sustained burst under test |
| Phase 3 ILM slots | `ilm_document` on kosmos-2.5 |

Note `sample_640x640.jpg` is the YOLO-native size and serves as the **default**
CV input where the size dimension isn't being swept. `dinov2` is pinned to
`cv_embed` (224²) because its patch grid is fixed — sweeping input size there
would change the model's token count, not just its pixel load.

Prompt sets are extracted from `experiment_config.json` into JSONL at build time
so both drivers can consume them: AIPerf takes `--input-file` +
`--custom-dataset-type single_turn`, and `perf_analyzer` takes `--input-data`.

### 7.2 The schema

Sits parallel to `sweeps:` in `benchmarks/configs/<gpu>.yaml`. Each tenant
resolves through the **existing** `resolve_round()` (`scenario_config.py:129`), so
`backend_args` fan-out via `family:` and `unsupported_backends` skipping work
unchanged.

```yaml
colocations:
  # ---- Phase 3: model type mix -------------------------------------------
  mix-llm-cv:
    phase: 3
    duration_s: 120
    isolation: mps                 # fixed per GPU, not swept (§1.16)
    solo_baselines: auto           # one solo run per tenant at the SAME rate
    tenants:
      - name: llm
        backend: vllm
        model: qwen2.5-7b
        workload: llm_short          # → §7.1 (prompts + output_tokens)
        gpu_memory_utilization: 0.45
        load: {pattern: poisson, rps: 4}
        driver: aiperf
      - name: cv
        backend: triton
        model: yolov8-l
        triton_backend: tensorrt     # tensorrt | onnx | python
        workload: cv_detect_small    # → §7.1 (image file)
        load: {pattern: poisson, rps: 50}
        driver: perf_analyzer

  # ---- Phase 4: sweep the neighbour's load --------------------------------
  cross-llm-vs-cv:
    phase: 4
    extends: mix-llm-cv
    rps_sweep: {tenant: cv, values: [1, 10, 50, 200]}

  # ---- Phase 6: secondary dimensions, dual baseline -----------------------
  secondary-backend-llm-a:
    phase: 6
    extends: mix-llm-cv          # baseline A — compute-bound
    vary: {tenant: llm, field: backend, values: [vllm, sglang, trtllm]}

  secondary-backend-llm-b:
    phase: 6
    extends: mix-memory-bound    # baseline B — memory-bound
    vary: {tenant: llm, field: backend, values: [vllm, sglang, trtllm]}
```

Three composition rules keep the whole matrix expressible without repetition:
`extends` (inherit a base colocation), `rps_sweep` (expand one tenant's rate into
N runs), `vary` (expand any tenant field into N runs). Phase 0–6 all fall out of
these.

New `iter_colocation(cfg, name)` beside `iter_sweep` (`scenario_config.py:217`);
`--emit-colocations` beside `--emit-rounds` (`:286`).

### Result schema additions (`metrics.py`)

`to_dict()` (`:145`) auto-routes unknown fields into `results`, so only
config-shaped ones need adding to `_CONFIG_FIELDS` (`:165`).

```
identity   → colocation_id, tenant_name, co_tenants[], n_tenants,
             isolation_mode, arrival_pattern, offered_rps
measured   → achieved_rps, t0_epoch_ms, request_trace_path,
             sm_clock_mhz_p50, throttle_reasons
ratios     → solo_baseline_run_id, degradation_ratio_{e2e_p50,e2e_p95,ttft_p95},
             throughput_retention        # computed in summary.py, §5 addition 2
```

Sorted `co_tenants` + `n_tenants` makes the contention matrix a groupby rather
than bespoke bookkeeping.

### Output layout

```
benchmarks/results/<gpu>/coloc/<run_id>/
  <tenant>.ndjson   # per request: t_start_ms, t_end_ms, ttft_ms, e2e_ms, tokens, ok
  gpu.ndjson        # timestamped sampler rows (50 ms for alignment runs)
  manifest.json     # tenant set, load specs, isolation mode, clock policy
```

---

## 8. Build sequence

| Step | Work | Files |
|---|---|---|
| 1 | Stage inputs + docs into repo | `workspace/contention/`, `.gitignore`, `.claude/skills/gpu-contention-benchmark/` (agentskills.io format) |
| 2 | **Customer-generated clips** — verify codec/spec, transcode to the two targets | upload to `workspace/contention/test_data/source/`; output to `…/test_data/vlm/` |
| 3 | Phase-0 probe + clock policy + variance | **new** `scripts/gpu_concurrency_probe.py`; `gpu_sampler.py` clock fields |
| 4 | Result schema + timestamps | `metrics.py`, `runner.py` |
| 5 | `colocations:` schema | `scenario_config.py`, `benchmarks/configs/*.yaml` |
| 6 | Orchestrator — start tenants, hold common `t0`, merge outputs | **new** `benchmarks/coloc.py`, **new** `bench coloc` in `cli.py` |
| 7 | CV tenants on Triton | model repo + `config.pbtxt`; `--allow-client-shm=true` |
| 8 | Analysis | `summary.py` §10 (degradation table, contention matrix, envelope); **new** `scripts/align_traces.py` |

`scripts/run_all_scenarios.sh` is **not touched** — its single-model invariants
are correct for the serial sweep. Contention gets its own entry point.

---

## 9. Verification

- **Phase 0 gate** — probe shows overlapping SM activity; findings recorded to
  `docs/findings/knowledge.yaml` (already read by `summary.py:300`).
- **Null test** — a "colocation" of one tenant against nothing must give ratio
  ≈ 1.0. If a tenant degrades against no neighbour, the harness is wrong.
- **Load fidelity** — `achieved_rps ≈ offered_rps` at low load, else the generator
  is the bottleneck, not the GPU.
- **Clock integrity** — no run published where `clocks_throttle_reasons.active`
  fired (§4.2).
- **Sampler sanity** — exactly one sampler process per run; tenant VRAM caps sum
  to ≤ observed peak.
- **Repeatability** — 5× repeat, spread reported, policy set from it (§4.5).
- **Regression** — `bench sweep` / `bench smoke` unchanged, `pytest tests/` green,
  `run_all_scenarios.sh` untouched.

---

## 10. Resolved in review — recorded here for the customer-facing writeup

All four earlier open items are settled. They still need **telling** to the
customer, since three concern their own picks:

| Item | Resolution |
|---|---|
| **PaddleOCR** | Kept, served via **Triton Python backend**. Its TensorRT path pins TRT 8.6.1.6 + CUDA 11.8, which cannot coexist with the Triton 26.07 stack — so it runs unoptimised rather than not at all. |
| **RF-DETR** | Use **`RFDETRMedium`**. `RFDETRBase` was deprecated 2025-07-23; Medium is comparable in size, so intent is preserved. |
| **PaliGemma2** | **Kept in config, fails safely.** Serves images; any video round skips with a clear reason and exit code 2. Tell the customer this pick cannot serve the video dimension it was chosen for — it is 28B not 32B, a base `-pt-` checkpoint, and image-only. |
| **32B / 72B tier** | **Kept**, scoped to PRO 6000 and H200 configs. Absent from the 5090 config on VRAM grounds only. |

Nothing blocks the build. The one dependency outside our control is the **two
generated video clips** (§1.1) — everything else is staged and verified.
