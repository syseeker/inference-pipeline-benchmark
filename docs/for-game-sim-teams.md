# Using this benchmark for a game-AI / game-simulation project

This guide is for you if you're training a model to work across different game
genres and want to know how this inference-optimization harness helps. Short
version: it doesn't train your model — it tells you, for any checkpoint you
produce, **how fast it runs, on which GPU, how cheaply, and whether optimizing
it breaks the gameplay** — and it settles the "are we bandwidth-bound?" question
with a measurement instead of a guess.

## First: which kind of model are you building?

"Game simulation" means two very different things, and they need different base
models. Get this right before anything else:

| You want to… | You're building a… | Base model to post-train |
| --- | --- | --- |
| **Simulate a player** — an agent that *plays* across genres (frame → action) | **policy** | **NitroGen** |
| **Simulate the world** — predict/generate the *next frames* of gameplay (a learned game engine, dream rollouts) | **world model** | **Cosmos Predict / Cosmos 3** |

NitroGen does **not** predict the next frame. It predicts the next *action*
(which button/stick), given the current frame — it's a player, not a world. If
your project is "given the current gameplay, predict what happens next," that's a
world model, and Cosmos Predict is the right base to post-train, not NitroGen.

Both paths are visuomotor diffusion models, so this harness measures both the
same way — but the bottleneck profile differs sharply (see Bandwidth, below).

## What this harness measures, and the decision each number gives you

Every run emits one row with the full metric set (details in
[metrics.md](metrics.md)). The ones that matter for you:

| Metric | The decision it gives you |
| --- | --- |
| **e2e action latency** p50/p95/p99 | Can the model hit your frame budget? 60 fps ≈ 16 ms/frame, 30 fps ≈ 33 ms. **p99 matters** — one slow frame is a visible stutter. |
| **mem-bw util vs SM util, power** | **Settles the bandwidth question** — are you bandwidth-bound, compute-bound, or launch/latency-bound (see below)? |
| **throughput (actions/sec), goodput** | How many game instances/agents per GPU → sizing a **simulation farm** (many parallel environments for data-gen or RL). |
| **energy / request** | Your cost at scale. Thousands of sim instances is an energy bill, not just a latency number. |
| **accuracy-vs-gold across precision** | Does FP8 / NVFP4 still produce the right actions? Quantization that tanks gameplay quality is a false economy. |
| **denoise-step sweep (16 → 4)** | Your quality↔latency dial: how few refinement steps can you afford before the agent plays worse? |
| **cross-GPU (RTX 5090 / PRO 6000 / H200)** | Which hardware to deploy or scale on. The *same* run repeats per GPU — apples-to-apples. |

## The bandwidth question — measured, not assumed

If your instinct is "bandwidth is our concern," this harness will tell you
whether that's actually true for your model. Bandwidth-bound means you move a lot
of bytes per unit of compute. There are three regimes, and we sample
memory-bandwidth utilisation (DCGM `DRAM_ACTIVE`) next to SM/compute util and
power, so you can see which one you're in:

- **Launch / latency-bound** — both utilisations low but end-to-end latency high.
  *This is the common case for a small policy like NitroGen (~500M):* its 16-step
  denoise loop is many tiny sequential kernels with the GPU idle between them.
  The fix is **CUDA graphs / `torch.compile` / TensorRT fusion**, not bandwidth.
- **Bandwidth-bound** — mem-bw ≥ ~70%, SM lower. Happens with **big batch** (many
  sim instances at once) or **large models with long activations** — e.g. a
  Cosmos Predict 2B–14B video world model generating long frame latents. The fix
  is quantization (FP8/NVFP4 → fewer bytes) and batching strategy.
- **Compute-bound** — SM ≥ ~80%. Fix: smaller model, lower precision, fewer steps.

> **A 500M model is rarely a bandwidth problem.** 500M weights are ~1 GB at BF16,
> ~0.5 GB at FP8 — tiny. At interactive batch you are almost certainly
> launch/latency-bound, and CUDA graphs / compile / TensorRT win far more than
> chasing bandwidth. Bandwidth becomes the real bottleneck when you (a) batch many
> simulation instances, or (b) move to a **large video world model** (the Cosmos
> path). The harness shows you which case you're actually in — so you optimize the
> right thing instead of the assumed thing.

## How it helps when your goal is "works across genres"

Generalization across genres is won in **data + training** (NitroGen's recipe: a
huge multi-genre gameplay dataset + behavior cloning). This harness can't train
that for you — but it can **evaluate it per genre**:

1. Build a scenario set per genre with
   [`scripts/build_nitrogen_scenarios.py`](../scripts/build_nitrogen_scenarios.py)
   (or your own gameplay clips), each carrying a gold action.
2. Run the sweep. `accuracy-vs-gold` is reported **per scenario**, so you can
   see: does one model hold up across genres, or collapse on some? And does
   **FP8/NVFP4 degrade specific genres more than others** (a fast-twitch shooter
   often tolerates quant error worse than a turn-based strategy game)?

That per-genre **accuracy × latency × precision** matrix is the artifact that
tells you "this single checkpoint ships fast enough across genres, and here's the
precision that keeps every genre acceptable."

## Where the harness sits in your workflow

```
   Prepare multi-genre gameplay data ─► post-train your base model
        (NitroGen if a PLAYER; Cosmos Predict if a WORLD model)
                                   │  checkpoint
                                   ▼
   THIS harness:  latency · throughput · bandwidth-vs-compute · energy ·
                  per-genre accuracy-vs-gold · across RTX 5090 / PRO 6000 / H200
                                   │
                                   ▼
   Decision:  ships in real time?  on which GPU?  at which precision/steps?
              which genres regress?  how many sim instances per GPU?
```

It de-risks deployment **before** you over-invest in training: the latency and
throughput ceilings tell you the largest model and step count you can afford in
real time, which constrains your architecture up front. Better to learn "a 16B
model can't hit 60 fps on a 5090" from a five-minute benchmark than after a
training run. And it doubles as a **regression harness** — every new checkpoint,
same sweep, comparable speed/cost/per-genre-accuracy on the GPUs you'll ship on.

## The boundary, stated plainly

- This is **inference / serving optimization**, not training, and not a
  generalization recipe.
- It works on whichever model you choose — but pick the right *kind* first
  (player vs world model), because that decides whether you post-train NitroGen
  or Cosmos Predict, and it decides whether bandwidth is even your problem.

See [nitrogen.md](nitrogen.md) for how NitroGen relates to Cosmos 3 and GR00T N1,
and [metrics.md](metrics.md) for exact field definitions.
