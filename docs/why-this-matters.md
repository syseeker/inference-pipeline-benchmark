# Why this project matters — for game-AI teams

A friendly tour for the bug-simulation / gameplay-simulation folks who
asked: "I see your README, but **why** is all this machinery here?"

You write the model. We measure whether it can ship. That's the whole
thing. The five sections below walk you through the five budgets your
work has to fit inside — and how this harness makes each budget visible.

---

## 1. The 16 ms problem (latency)

A game running at 60 frames per second spends ~16 ms per frame. Inside
that 16 ms, your AI has to (a) see the frame, (b) decide what to do,
(c) hand the action back to the game loop. If inference takes 50 ms, your
player sees the AI react **three frames late** — and that's perceptible.
At 100 ms it feels broken.

**Your work, your latency.**

- **Bug-sim:** "Does the QA bot catch this collision glitch the same
  frame it appears?" If the bot takes 50 ms to emit "anomaly here",
  the frame-perfect repro you wanted is already gone.
- **NitroGen / gameplay AI:** "Does the player feel like they're up
  against an opponent or a slideshow?" Below ~30 ms it feels real-time;
  above ~100 ms it feels like reading off a sheet.

So latency isn't a "performance number." It's whether you can ship at all.

```
   Game tick (60 fps)
   ┌────────────────────────────┐
   │  16 ms                    │
   │  ┌──┐ ┌──────┐ ┌──┐       │
   │  │  │ │ AI?  │ │  │       │  <- inference fits, ship it
   │  └──┘ └──────┘ └──┘       │
   └────────────────────────────┘

   ┌────────────────────────────┐
   │  16 ms                    │
   │  ┌──┐  ┌────────────────┐ │
   │  │  │  │ AI? ...not yet │ │  <- inference misses, ship is broken
   │  └──┘  └────────────────┘ │
   └────────────────────────────┘
```

This harness measures **e2e p50/p95/p99** — the full
"frame in → action out" pipeline, not just the LLM call. That's the
number your QA team will look at when they say "feels laggy." We
report p99 explicitly because **the worst frame is what users feel**;
average means nothing if 1% of frames stutter.

---

## 2. The N-instances problem (throughput)

You want to A/B-test policy v1 vs v2 across **1,000** parallel game
instances overnight. How many fit on one GPU?

The honest answer: depends on the engine, the model size, the precision,
and the GPU's memory bandwidth — none of which you can guess from a spec
sheet. Same model, same GPU, different inference engine, **3× different
throughput is normal**.

- 1 sim per GPU → you need 1,000 GPUs. That's not a project, that's a
  data-center procurement.
- 100 sims per GPU → 10 GPUs. Doable.
- 1,000 sims per GPU → 1 GPU. Cheap iteration.

The harness reports **throughput (seq/s)** and **memory-bandwidth
utilisation (DRAM_ACTIVE)** so you can tell which regime you're in,
plus a follow-up tool (`bench load-test` — see §5) that measures the
full concurrency curve, not just the point.

---

## 3. The kWh problem (energy / cost at scale)

A 30B model takes ~500 W during inference. 100 instances × 24/7 ×
$0.20/kWh × 365 days = **about $100K/year** of pure electricity, before
you've paid for the GPUs themselves. Cut energy per request 30% and
you've saved an engineer's salary.

- "Run on cheaper hardware" usually means "fit on a smaller GPU at
  lower power." This is real money for any sim farm.
- "Use FP8 instead of BF16" usually means "30–50% less energy per
  request" because you've cut memory bandwidth almost in half.

The harness logs `power_avg × wall_time / n_completed` as **J/req**.
Multiply by your fleet size and electricity rate; that's your bill.

---

## 4. The "smaller GPU" problem (precision)

A 30B model in BF16 weighs ~60 GB. On a 96 GB GPU (PRO 6000) that
leaves ~36 GB for the KV cache — and KV cache is what determines how
many concurrent sims fit. Cramped.

Same model in FP8 weighs ~30 GB. Suddenly you have ~66 GB for KV.
You can run **3–4× more concurrent sims on the same GPU**, or fit
on a 48 GB card you couldn't before.

Sound too good? **It's not free.** Quantizing weights to FP8 changes
the numbers slightly. Done correctly, your model behaves the same.
Done wrong, your policy starts misclicking on edge cases that BF16
handled. The harness exposes **accuracy-vs-gold** per scenario, so you
*see* the regression instead of finding it in production.

For NitroGen specifically (see [docs/nitrogen.md](nitrogen.md)), FP8
gives **45% lower latency at 2× throughput, half the energy per
request** vs the BF16 baseline. Same gameplay quality. Measured on
a real RTX PRO 6000 Blackwell:

| Precision | p50 e2e | seq/s | J/req |
|---|---|---|---|
| BF16 (baseline) | 107 ms | 9.4 | 11.2 |
| **FP8 (winner)** | **59 ms** | **16.4** | **5.7** |

---

## 5. The "which engine?" problem (backends)

You'd think "I'll just use vLLM" or "I'll use TensorRT-LLM, that's the
NVIDIA one." Reality:

- **vLLM** — easiest to deploy; PagedAttention scheduler is excellent;
  ships fast against new architectures (Qwen3-VL on day 1).
- **SGLang** — RadixAttention; usually wins at high-batch text
  workloads; quirkier on multimodal.
- **TensorRT-LLM** — fastest *in theory*, painful in practice. Engine
  compile times measured in minutes per model. Lags behind on new
  archs (Qwen3.6, Gemma 4 today: not yet loadable).
- **NitroGen ZMQ** — single-flight policy server, not OpenAI-shaped.

On a single Blackwell PRO 6000, vLLM comes out 1.6–1.8× faster than
SGLang on dense models and **18× faster than TRT-LLM** on Qwen3-VL-30B,
because TRT-LLM 1.2.1's fused-MoE FP8 kernel doesn't yet target
SM_120 (Blackwell pro/workstation). You can't predict that from docs.
You measure it.

So **engine choice is a sweep, not a vibe.** This harness runs the same
scenarios through all three and prints the table.

---

## 6. Why this harness (instead of running each tool yourself)

You *could* run vLLM, then SGLang, then TRT-LLM, then NitroGen by hand,
each with their own CLI flags, output formats, and metric definitions.
Then write a spreadsheet. Then re-run when a model bumps. Then realize
the bench in run #3 used a different scenario set than run #2.

That spreadsheet is what this project replaces.

We orchestrate three tools NVIDIA provides — each excellent at its job,
none of which you should have to learn:

| Tool | What it does | When it runs in this harness |
|---|---|---|
| [**modelopt**](https://github.com/NVIDIA/TensorRT-Model-Optimizer) | FP8/NVFP4 PTQ calibration + ONNX export | Once on our hardware. We ship the calibrated artifact to [`syseeker-at-nv/nitrogen-quant`](https://huggingface.co/syseeker-at-nv/nitrogen-quant). Customers `hf download`, they don't recalibrate. |
| [**AIPerf**](https://github.com/ai-dynamo/aiperf) | Client-side load generator — TTFT, ITL, throughput across concurrency | `bench load-test` wraps it. Produces the **concurrency profile** in summary.md §9. Answers "how many sims per GPU." |
| [**Nsight Systems**](https://developer.nvidia.com/nsight-systems) | GPU-side timeline profiler — kernel launches, GPU idle gaps | `bench profile --tool nsys` wraps it. Used only when summary.md flags a row that needs explanation. Answers "WHY is this kernel slow." |

You drive everything via the `bench` CLI (one command surface, JSON
status, agent-friendly). The three tools above are invoked under the
hood. You don't need to know modelopt's recipe schema or AIPerf's flag
soup unless you're doing custom work — the skill docs ([skills/](../skills/))
tell agents the right invocation per question.

---

## 7. Two real questions, two real answers

### Bug-sim team

> "Will Qwen3-VL-32B-FP8 detect glitch X at 60 fps on an RTX 5090?"

What we do:
1. `bench sweep --gpu rtx5090 --model qwen3-vl-32b-fp8 --backends "vllm sglang"`
2. Read summary.md §1 — **e2e p99**. If it's <16 ms → ship it. If 16–33 ms → 30 fps acceptable. If >33 ms → reduce model, reduce context, or pick a smaller GPU.
3. `bench load-test ... --concurrency "1,4,16"` to see if you can multi-batch glitch frames.

The answer is **a number with units**, not "should be fast enough."

### NitroGen / gameplay-policy team

> "Is the FP8 policy fast enough to feel like a real opponent?"

Your workflow:
1. `bench sweep --gpu rtx_pro6000 --sweep nitrogen-backends`
2. summary.md tells you: **nitrogen-onnx + FP8 → 59 ms p50** (vs **107 ms** for the BF16 baseline — so 45% faster, but still 59 ms in absolute terms).
3. Translate to a polling rate: 1000 ms ÷ 59 ms ≈ **17 AI decisions per second**. That's the rate at which the policy can respond to fresh frames.
4. Match against your game's needs:
   - **Twitch shooter / fighting game (60 fps target, every frame matters)**: 17 Hz is too slow — the AI lags the player by ~4 frames at 60 fps. Not shippable. Either the model gets smaller, NVFP4 cuts another ~10–15% (Blackwell-only — track [docs/findings/](findings/) for when that lands), batch-mode multi-instance amortises the launch cost (`bench load-test` will tell you), or you accept a 30 fps game target.
   - **Strategy / RPG / sim (LoL-class — 10–30 Hz AI update is normal)**: 17 Hz is **fine, ship it**. The 45% headroom you gained over BF16 is exactly the safety margin you wanted.
5. Why the FP8 win is smaller in absolute ms than the headline percentage suggests: the workload is launch-bound (`DRAM_ACTIVE` ~0.4%), so per-step kernel-launch overhead dominates. AIPerf concurrency sweeps + NVFP4 (once your TRT version ships the plugin) are the next levers — both are one `bench` command away when you need them.

Concrete decisions, with numbers, from one sweep. **The benchmark didn't make the decision for you — it gave you the units to make it in.**

---

## 8. Where to go next

- **Drive your first sweep**: [NITROGEN_QUICKSTART.md](../NITROGEN_QUICKSTART.md) — an agent-prompt walkthrough. Read it once, paste the prompts into Claude Code / Codex / Cursor, you'll have summary.md in 10 minutes.
- **What this harness measures (in detail)**: [docs/metrics.md](metrics.md).
- **Player vs world model — which kind of model do you actually need?**: [docs/for-game-sim-teams.md](for-game-sim-teams.md).
- **Capacity math (which model fits which GPU)**: [docs/capacity.md](capacity.md).
- **Per-GPU model picks**: [docs/models.md](models.md).
- **The NitroGen-specific deep-dive**: [docs/nitrogen.md](nitrogen.md).

**Your job** is the model. **This harness's job** is to tell you whether
it'll work at the speed, scale, and cost you need. Treat us as the
measurement bench: bring the artifact, get the numbers, ship with
confidence.
