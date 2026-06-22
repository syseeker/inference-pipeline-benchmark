# NitroGen — diffusion-policy backend

NitroGen is a second *kind* of model in this harness. The vLLM / SGLang /
TRT-LLM backends serve **vision-language models** that read pixels + a text
instruction and emit a stream of language tokens. NitroGen reads a single game
frame + a game id and emits a **continuous gamepad action** (two analog sticks +
21 buttons, over a 16-step horizon). It is a 500M-parameter generalist gaming
policy, behavior-cloned on internet gameplay video.

We integrate it so a native control policy can be measured on the same harness,
latency budget, and metric set as the VLM backends — and so NitroGen itself
becomes the workload for an **inference-optimization study** across execution
backends, precisions, and GPUs.

> Setup, config, and run commands live in [README](../README.md) and the GPU
> YAMLs. This doc explains *what NitroGen is* and *how it relates to the other
> NVIDIA model families* so the benchmark numbers are read in the right context.

## How it produces an action

```
frame (256×256) + game_id
        │
        ▼
 SigLIP vision encoder ──► vision tokens
        │
        ▼
 Diffusion Transformer (DiT) — flow matching:
   start from random noise, run the DiT N times (N = 1..16 denoise steps),
   each pass refining the SAME action vector toward the final action
        │
        ▼
 action: j_left[2], j_right[2], buttons[21]  (per horizon step)
```

Two properties matter for benchmarking:

- **The whole action is produced together and then refined**, over a fixed,
  configurable number of denoise steps (`--steps`). There is no token stream, so
  there is no time-to-first-token or inter-token latency to measure — the
  decision metric is the **full inference time** and, with classifier-free
  guidance, the DiT runs twice per step.
- **Conditioning is a game id, not a sentence.** NitroGen is a reactive
  ("system-1") policy over the last frame; it does not consume a free-text
  instruction. In this harness each scenario carries an optional `game_id`
  ([ScenarioRequest](../tests/smoke/scenarios/schema.py)); the text instruction
  is recorded but unused by this backend.

## NitroGen vs a vision-language model (the other backends)

Both read pixels, but they are built to do different jobs and run differently:

| | VLM (Qwen3-VL, Nemotron-Omni, Gemma 4) | NitroGen |
| --- | --- | --- |
| **Output** | language tokens (then parsed to an `ActionSequence`) | continuous gamepad controls directly |
| **How output is formed** | one token at a time, each conditioned on the last | one action vector, refined over N denoise steps |
| **Conditioning** | text instruction + image | game id + frame |
| **Serving** | OpenAI-compatible HTTP (vLLM/SGLang/TRT-LLM) | ZMQ, `scripts/serve_nitrogen.py` |
| **Latency knobs** | KV cache, prefix cache, chunked prefill, CUDA graphs | denoise steps, CFG, CUDA graphs, compile/TRT/ONNX |
| **What "faster" means** | tokens/sec, TTFT, ITL | full-action latency; steps × per-DiT-pass cost |

Because the generation process and the I/O type differ, NitroGen does **not**
ride the LLM-serving stack (vLLM/SGLang/TRT-LLM expect token-generating
transformer architectures behind an OpenAI API). Its "different backends" are
the general deep-learning execution stack — eager PyTorch, `torch.compile`,
CUDA graphs, **TensorRT** (the compiler, distinct from TensorRT-LLM), and ONNX
Runtime — combined with precision (BF16 / FP8 / NVFP4) and denoise-step count.
See [frameworks.md](frameworks.md#nitrogen-execution-backends) for the
per-backend notes and [metrics.md](metrics.md) for what gets recorded.

## NitroGen vs NVIDIA Cosmos 3 and GR00T N1

NitroGen sits next to two NVIDIA model families, and both have moved fast. The
honest placement: NitroGen is a small, single-purpose **policy** (frame →
gamepad action) that overlaps the *action-generation* slice of NVIDIA's frontier
**Cosmos 3** omnimodel and shares an **architecture recipe** with **GR00T N1**
(vision backbone → DiT flow-matching action head).

| | **NitroGen** | **Cosmos 3** (frontier; ← Cosmos Predict 2.5) | **GR00T N1 / N1.5 / N1.7** |
| --- | --- | --- | --- |
| **Category** | game-playing **policy** (frame → action) | open **omnimodel** for Physical AI — vision reasoning + world generation + **action/policy** | humanoid robot **VLA policy** (obs+language → action) |
| **What it outputs** | gamepad controls (sticks + buttons) | text, image, video, ambient sound, **action trajectories** (Predict 2.5: video only) | robot joint/EEF actions |
| **Core architecture** | SigLIP encoder → DiT flow-matching action head | **mixture-of-transformers** — a reasoning transformer + an expert generation transformer (Predict 2.5: flow-based video diffusion) | dual-system: VLM backbone (System 2) + DiT flow-matching policy (System 1) |
| **Inference pattern** | iterative denoising of an action (1–16 steps) | reason-then-generate; the generation expert denoises video and/or action (Predict 2.5: up to 30 s video) | denoising of an action chunk, gated by a VLM plan |
| **Conditioning** | game id + single frame | text / image / video / audio prompt | image + **language instruction** |
| **Scale / tiers** | ~500M | frontier; tiers: **Super** (max physics accuracy), **Nano** ("video and action reasoning in fractions of a second"), **Edge** (real-time, coming soon) | VLM backbone + DiT (N1.7 backbone: Cosmos-Reason2-2B / Qwen3-VL) |
| **Trained on** | internet gameplay video (behavior cloning) | billions of multimodal Physical-AI samples (text/image/video/sound/**action trajectories**) | robot trajectories + human video + synthetic |
| **Real-time control?** | yes — single action chunk, low latency | Nano/Edge tiers target real-time action; Super prioritizes fidelity | yes — System 1 runs at high frequency |

What this means for the study:

- **vs Cosmos 3 / Cosmos Predict** — Cosmos 3 (launched June 1 2026, GTC Taipei)
  is the current frontier: a fully-open omnimodel whose **mixture-of-transformers**
  pairs "a reasoning transformer with an expert generation transformer," and which
  now natively generates **actions** alongside video/audio, explicitly for
  "physical AI policy model development." So the overlap with NitroGen is no longer
  just inference-pattern — Cosmos 3 *does* action generation. The difference is
  scope and scale: Cosmos 3 is a frontier multi-tier omnimodel (Super/Nano/Edge)
  for robotics, AV, and synthetic data; NitroGen is a 500M single-game-frame policy.
  The **Cosmos 3 Nano** goal — "action reasoning in fractions of a second" — is the
  closest cousin to NitroGen's real-time niche. Either way the *optimization
  toolbox* is shared with the diffusion/generation side (reduce denoise steps,
  capture the fixed-shape step in a CUDA graph, TensorRT export, FP8/NVFP4) — at a
  size small enough to iterate on quickly. (Lineage note: NVIDIA also shipped
  **Cosmos Policy**, a robot policy obtained by post-training the Cosmos-Predict
  video world model — the "world model → policy" direction, now subsumed by Cosmos
  3's native action generation.)
- **vs GR00T N1** — this is the closest *architectural* cousin. GR00T N1 is an
  open humanoid-robot foundation model with a dual-system design: a
  vision-language backbone (System 2) that interprets the scene + language, and a
  **DiT-based flow-matching policy (System 1)** that emits high-frequency
  actions. NitroGen is essentially that System-1 action expert, specialized for
  games: single frame, game-id conditioned, gamepad output, and no System-2
  language planner on top. GR00T N1.5 / N1.7 add architecture/data improvements
  (N1.7 early access pairs a Cosmos-Reason2-2B / Qwen3-VL backbone with 20K hours
  of human video) and target multi-embodiment robot control rather than games.

The practical takeaway: optimizing NitroGen's inference exercises the **same
techniques** you would use on the Cosmos generation transformer's diffusion loop
and GR00T's flow-matching action head — TensorRT/ONNX export, FP8/NVFP4
quantization, CUDA
graphs over the fixed-shape DiT, and denoise-step reduction — at a model size
small enough to iterate quickly and run head-to-head against the VLM backends.

## The inference-optimization study

The NitroGen **model** runs are swept across the execution axes:

- **execution engine = backend** — `nitrogen-eager` · `nitrogen-compile` ·
  `nitrogen-cudagraph` · `nitrogen-tensorrt` · `nitrogen-onnx`. One per run, never
  a combo (just as a VLM runs on one of vLLM/SGLang/TRT-LLM per run).
- **precision** — BF16 (reference) · FP8 · NVFP4 (Blackwell only)
- **denoise steps** — e.g. 16 vs 4 (latency↔quality knob)
- **GPU** — RTX PRO 6000 · H200 · RTX 5090

So a run reads: *run model `nitrogen-500m-fp8` on backend `nitrogen-tensorrt`.*
Every run records the full metric set — latency p50/p95/p99, throughput, GPU
util / power / energy — **plus accuracy-vs-gold** (does the optimized engine
still produce the same action as BF16?). See
[metrics.md](metrics.md#policy-accuracy-vs-gold-nitrogen). Denoising noise is
**seed-pinned** so FP8-vs-BF16 deltas reflect precision, not sampling.

Run it with (the `nitrogen-backends` sweep pairs each engine backend with the
right model automatically):

```bash
scripts/run_all_scenarios.sh --gpu rtx_pro6000 --sweep nitrogen-backends \
    --backends "nitrogen-eager nitrogen-compile nitrogen-cudagraph nitrogen-tensorrt nitrogen-onnx" \
    --scenarios-dir tests/smoke/scenarios_nitrogen
```

The NitroGen scenarios are built from the `nvidia/NitroGen` dataset (action
annotations + source-video frames) by
[`scripts/build_nitrogen_scenarios.py`](../scripts/build_nitrogen_scenarios.py).
For the field-by-field mapping (chunk → scenario) and the reasoning behind
converting to a common shape — plus how to plug in your own dataset
without forking — see [docs/scenarios.md](scenarios.md).

## Sources

- GR00T N1 — [arXiv:2503.14734](https://arxiv.org/abs/2503.14734),
  [NVIDIA newsroom](https://nvidianews.nvidia.com/news/nvidia-isaac-gr00t-n1-open-humanoid-robot-foundation-model-simulation-frameworks)
- GR00T N1.5 — [NVIDIA GEAR research](https://research.nvidia.com/labs/gear/gr00t-n1_5/);
  GR00T N1.7 / Isaac-GR00T — [github.com/NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)
- **Cosmos 3** (frontier omnimodel, launched June 1 2026 @ GTC Taipei) —
  [NVIDIA newsroom](https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-3-the-open-frontier-foundation-model-for-physical-ai),
  [NVIDIA blog](https://blogs.nvidia.com/blog/cosmos-3-physical-ai-open-world-foundation-model/),
  [technical report](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf),
  [nvidia.com/ai/cosmos](https://www.nvidia.com/en-us/ai/cosmos/)
- Cosmos Predict 2.5 (prior video WFM) —
  [research.nvidia.com](https://research.nvidia.com/labs/cosmos-lab/cosmos-predict2.5/),
  [github](https://github.com/nvidia-cosmos/cosmos-predict2.5)
- Cosmos Policy — [NVIDIA blog](https://huggingface.co/blog/nvidia/cosmos-policy-for-robot-control)
