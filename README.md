# inference-pipeline-benchmark

A benchmark harness for **inference pipelines on NVIDIA GPUs**. Bring your
models and a target GPU; get apples-to-apples latency, throughput, energy and
accuracy numbers across **vLLM**, **SGLang**, **TensorRT-LLM**, **Triton** and
the **NitroGen** execution engines — from one command, one CLI, or one agent
prompt.

The output is a per-GPU `summary.md` that answers a deployment question:
*which backend, which precision, which model, at what cost?*

```markdown
## Core findings
- nitrogen-tensorrt @ fp8 wins: 43 ms p50, 20.4 seq/s, 4.3 J/req.
- FP8 halves latency vs BF16 — weight bytes ↓50%, GEMM dispatches to Blackwell FP8 paths.
- BF16 engines (eager / compile / cudagraph) cluster within 1 ms: launch overhead
  is marginal at batch=1 for a 500M model.

## 1. Decision metrics
| backend            | precision | e2e p50 | e2e p95 | seq/s | J/req | validity |
|--------------------|-----------|---------|---------|-------|-------|----------|
| nitrogen-tensorrt  | fp8       | 43 ms   | 46 ms   | 20.4  | 4.3   | 100%     |
| nitrogen-onnx      | fp8       | 43 ms   | 47 ms   | 20.3  | 4.4   | 100%     |
| nitrogen-eager     | bf16      | 69 ms   | 73 ms   | 9.3   | 10.4  | 100%     |
```

*(Real output, from the NitroGen engine sweep.)* Ten sections in total —
decision metrics, latency diagnostics, throughput, cache & scheduling, GPU
resource usage, cross-run deltas, per-scenario detail, environment,
concurrency profile, and contention. See [docs/metrics.md](docs/metrics.md).

---

## Three benchmarks, deliberately separate

They answer different questions, use different inputs, and are **not**
interchangeable. Pick the one that matches your question.

| | Question | Input | Start here |
|---|---|---|---|
| **1. Single-model** | How fast is this model on this GPU, and on which backend? | scenarios | [QUICKSTART_VLM.md](QUICKSTART_VLM.md) |
| **2. Multi-model contention** | How much do these models slow each other down when they share a card? | workloads | [QUICKSTART_MULTIMODEL_CONTENTION.md](QUICKSTART_MULTIMODEL_CONTENTION.md) |
| **3. NitroGen policy** | Which execution engine and precision for a diffusion policy model? | scenarios | [QUICKSTART_NITROGEN.md](QUICKSTART_NITROGEN.md) |

### 1. Single-model

One model, one backend, one round at a time — swept across backends,
precisions and models. This is the core of the harness.

| Family | Input → output | Backends |
|---|---|---|
| **VLM** — image+text | screenshot + instruction → schema-valid action sequence | vLLM · SGLang · TRT-LLM · NIM |
| **VLM** — video+text | short video + question → free-form analysis | vLLM · SGLang · TRT-LLM |

Driven by **scenarios** — a directory of folders, one per test case. Three
image scenarios ship ready to run; for video you supply the clips. Point any
run at a different folder with `--scenarios-dir`. Format and how to add your
own: [docs/scenarios.md](docs/scenarios.md).

```bash
bench sweep --gpu rtx_pro6000 --sweep vlm-backends
bench summary --gpu rtx_pro6000
```

### 2. Multi-model contention

Several models resident on one GPU at the same time, serving interleaved
traffic — measuring how much each one degrades the others. A text LLM, a video
VLM, an image-language model and a CV detector, in any combination, on one card
or split across two.

The unit of measurement is a **degradation ratio**: how much slower a model
gets because of its neighbours. Getting that ratio to mean anything is most of
the work — [docs/contention.md](docs/contention.md) explains why, and is worth
reading before you run this.

Driven by **workloads and colocations** defined in the GPU yaml, not by
scenarios — a colocation names its tenants, their models, their VRAM budget and
the request rate each is driven at.

```bash
python3 scripts/gpu_concurrency_probe.py --gpu rtx_pro6000   # gate: do tenants really overlap?
bench coloc --gpu rtx_pro6000 --colocation mix-llm-cv       # one colocation
bench coloc --gpu rtx_pro6000 --all --resume      # the whole study, one command
bench summary --gpu rtx_pro6000                             # contention section
```

> **Hardware validation in progress.** 39 colocations covering all seven
> phases, under 357 unit tests. The Phase 0 gate, the solo baselines and a
> Triton CV tenant now run on a PRO 6000 — but **no two-tenant contention
> window has completed**, so no degradation ratio exists yet. Open items:
> [the validation record](skills/gpu-contention-benchmark/reference/gpu-validation.md).

### 3. NitroGen policy

The outlier, and separate on purpose. NitroGen is a **diffusion policy** — it
emits a continuous gamepad action directly, with no token stream — so it is
served by ZMQ execution engines rather than an HTTP inference server, and
graded against gold gamepad vectors rather than schema validity.

| Model emits | Served by |
|---|---|
| a continuous gamepad action | `nitrogen-eager` / `-compile` / `-cudagraph` / `-tensorrt` / `-onnx` |

It shares the scenario format and the `summary.md` columns with the VLM
benchmark, so one run *can* compare them on the same input — but if you are not
specifically here for policy models, you can skip it entirely.
Everything about it: **[QUICKSTART_NITROGEN.md](QUICKSTART_NITROGEN.md)** and
[docs/nitrogen.md](docs/nitrogen.md).

---

## Start here

| What you want to do | Go to |
|---|---|
| Benchmark a **VLM** on image or video scenarios | **[QUICKSTART_VLM.md](QUICKSTART_VLM.md)** |
| Measure **models contending** on one or two GPUs | **[QUICKSTART_MULTIMODEL_CONTENTION.md](QUICKSTART_MULTIMODEL_CONTENTION.md)** |
| Benchmark the **NitroGen policy model** across execution engines | **[QUICKSTART_NITROGEN.md](QUICKSTART_NITROGEN.md)** |
| Understand *why* any of this matters before running it | **[docs/why-this-matters.md](docs/why-this-matters.md)** |
| Evaluate this for a game-AI or simulation team | **[docs/for-game-sim-teams.md](docs/for-game-sim-teams.md)** |
| Look up a flag, a yaml field, or an output path | **[BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md)** |

---

## Three ways to drive it

Every benchmark can be run three ways. They are the same code path — the CLI
wraps the scripts, and the agent skills wrap the CLI. Pick whichever suits you.

1. **Agent prompt** *(recommended)* — "Run the nitrogen sweep and tell me the
   winner." Six skills ship in [skills/](skills/); install with
   `bench install-skill`. Works with Claude Code, Codex and Cursor. No flags to
   memorise, and the agent reads `summary.md` back to you in plain language.
2. **`bench` CLI** — `bench {probe,setup,scenarios,smoke,sweep,coloc,summary,load-test,profile}`.
   Every command takes `--json` and returns a structured status with an exit
   code you can branch on. Full surface in [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md).
3. **Shell scripts** — `scripts/run_all_scenarios.sh` and friends, if you would
   rather not install the CLI. This is what the other two layers call.

Both quickstarts show all three for each step.

> **Two layers of Python environment — don't mix them up.** The `bench` CLI
> lives in a venv you own (`~/venv`); each backend gets its own venv inside the
> repo (`.venv-vllm`, `.venv-nitrogen`, …), created by `bench setup` and
> activated automatically by the sweep. All five NitroGen execution engines
> share **one** venv, `.venv-nitrogen`.

---

## What the harness reports

Tokens/sec is **not** the decision metric. The pipeline succeeds when:

- **Valid command-sequence latency** — time from input in to a schema-valid
  command list out — meets the interactive budget.
- **Command success rate** — fraction of generated sequences the executor
  accepts and that achieve the intended outcome.
- **Safety / grammar validity** — fraction passing the validator on first try.
- **p95 / p99 stability** — tail latency under realistic concurrency.
- **Energy per request** — J/req, the number that decides fleet cost.

For video scenarios, key-phrase coverage replaces schema validity as the
quality signal. For NitroGen, accuracy is measured against the dataset's gold
gamepad action. For contention, the decision metric is the **degradation
ratio** and the **safe-operating envelope** — the load at which achieved
throughput stops tracking what was asked for.

Token-level metrics (TTFT, ITL, vision-encoder latency, mem-bw util, KV-cache
hit rate, CUDA-graph delta, quant accuracy loss, TP efficiency) are tracked as
**diagnostics**. Full per-field definitions in [docs/metrics.md](docs/metrics.md).

---

## What this harness orchestrates

We wrap four NVIDIA tools so you don't have to learn them first:

| Tool | What | When |
|---|---|---|
| [**modelopt**](https://github.com/NVIDIA/TensorRT-Model-Optimizer) | FP8 / NVFP4 PTQ calibration + ONNX export | Run once on a known-good box; the calibrated artifact ships via [`syseeker-at-nv/nitrogen-quant`](https://huggingface.co/syseeker-at-nv/nitrogen-quant) and auto-downloads on the first FP8 round. |
| [**AIPerf**](https://github.com/ai-dynamo/aiperf) | Client-side load generator (OpenAI-compatible) | `bench load-test` wraps it for concurrency curves; `bench coloc` uses it to drive LLM/VLM tenants open-loop. |
| [**Triton**](https://github.com/triton-inference-server/server) | Inference server for CV models | Serves the CV tenants in the contention study. `perf_analyzer` drives them, since AIPerf cannot. |
| [**Nsight Systems**](https://developer.nvidia.com/nsight-systems) | GPU timeline profiler | `bench profile --tool nsys` wraps it. Escalation tool for when `summary.md` flags a row that needs proof, not inference. |

These are wrappers, not abstractions — call any of the tools directly whenever
you need to.

> **On system tools (nsys, modelopt, tensorrt).** These are system binaries and
> NVIDIA-index wheels, not ordinary Python packages, so they are deliberately
> **not** in `requirements.txt` or `pyproject.toml` — don't go hunting there.
> `bench setup --backend <name>` handles the system-level installs (apt for
> nsys, the NVIDIA wheel index for tensorrt and modelopt, post-install chmod
> and symlinks). You see `sudo` once at install time and then forget about it.

---

## GPUs and models

Three GPU profiles ship, each with a curated model list that fits its VRAM:

| GPU | VRAM | Default VLM | Why |
|---|---|---|---|
| RTX 5090 | 32 GB GDDR7 | `qwen3-vl-8b-fp8` | Customer-relevant device; 8B-FP8 is the largest Qwen3-VL that fits with KV headroom |
| RTX PRO 6000 | 96 GB GDDR7 | `qwen3-vl-32b-fp8` | Server workflow; 32B-FP8 leaves comfortable KV room |
| H200 | 141 GB HBM3e | `qwen3-vl-32b-bf16` | Bandwidth ceiling; HBM3e at 4.8 TB/s + BF16 = cleanest accuracy baseline |

Model families under test: **Qwen3-VL** (headline VLM), **Qwen3.5 / Qwen3.6**
(dense text, TRT-LLM engine candidate), **Gemma 4** (cross-vendor, video-capable),
**Nemotron-3-Nano-Omni** (NV multimodal MoE), and **NitroGen 500M** (policy).
The contention study adds a Qwen2.5 size ladder and CV models (YOLOv8, DINOv2,
kosmos-2.5, PaddleOCR).

Not every model runs on every backend. Each GPU yaml carries an
`unsupported_backends:` field with a one-line reason per pinned-out combination
— the sweep skips those rows and tells you why. Picks and rationale in
[docs/models.md](docs/models.md); memory math in [docs/capacity.md](docs/capacity.md).

Tensor parallelism on consumer cards is **an experiment, not the default** —
see [docs/gpu-strategy.md](docs/gpu-strategy.md).

---

## Doc map

**Getting started**

| File | What's in it |
|---|---|
| [QUICKSTART_VLM.md](QUICKSTART_VLM.md) | VLM benchmarks end-to-end — image+text and video+text, across vLLM / SGLang / TRT-LLM |
| [QUICKSTART_NITROGEN.md](QUICKSTART_NITROGEN.md) | NitroGen policy sweep end-to-end — 5 execution engines × precision × denoise steps |
| [QUICKSTART_MULTIMODEL_CONTENTION.md](QUICKSTART_MULTIMODEL_CONTENTION.md) | Contention study end-to-end — install, the Phase 0 gate, all seven phases, one GPU and two |
| [docs/why-this-matters.md](docs/why-this-matters.md) | Engineer-friendly intro: the four budgets (latency / throughput / energy / precision) and why backend choice isn't free |
| [docs/for-game-sim-teams.md](docs/for-game-sim-teams.md) | For game-AI teams: player-vs-world-model choice, the bandwidth reality, per-genre accuracy workflow |

**Operational reference**

| File | What's in it |
|---|---|
| [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md) | The full reference: `bench` command surface, yaml schema, sweep design, output structure, troubleshooting |
| [INFERENCE_BACKENDS.md](INFERENCE_BACKENDS.md) | Install vLLM / SGLang / TRT-LLM venvs; three operational modes (NIM cloud, local server, NIM container) |
| [SMOKE_TESTS.md](SMOKE_TESTS.md) | Per-backend "is the server alive" check before benchmarking |
| [tests/smoke/scenarios/README.md](tests/smoke/scenarios/README.md) | Scenario file format + how to add your own |

**Concepts and rationale**

| File | What's in it |
|---|---|
| [docs/metrics.md](docs/metrics.md) | Every metric defined — decision metrics vs diagnostics, mapped to `summary.md` sections |
| [docs/scenarios.md](docs/scenarios.md) | The scenario input format — what drives the single-model and NitroGen benchmarks, and how to register your own dataset source |
| [docs/models.md](docs/models.md) | Per-GPU model picks, the rationale, and hypothesis-vs-measured-reality |
| [docs/capacity.md](docs/capacity.md) | Memory math — which checkpoint fits which GPU at BF16 / FP8 / NVFP4 |
| [docs/nitrogen.md](docs/nitrogen.md) | The NitroGen diffusion policy: how it works, vs Cosmos 3 / GR00T N1 / VLMs, and the execution-backend study |
| [docs/frameworks.md](docs/frameworks.md) | Per-framework one-pager (vLLM, SGLang, TRT-LLM PyTorch backend, ModelOpt, Triton) |
| [docs/gpu-strategy.md](docs/gpu-strategy.md) | Tensor parallelism vs replicas; PCIe-vs-NVLink considerations |
| [docs/architecture.md](docs/architecture.md) | Pipeline shape — today VLM-only; v1+ splits CV ↔ VLM ↔ decoder ↔ validator |
| [docs/findings/](docs/findings/) | Per-(gpu, framework, model) postmortems that the summary generator auto-links from Core findings |

**Multi-model contention**

| File | What's in it |
|---|---|
| [docs/contention.md](docs/contention.md) | **Start here.** What a degradation ratio means, why the baseline is the hard part, how the four model types contend, single- and multi-GPU topologies |
| [docs/contention-coverage.md](docs/contention-coverage.md) | Coverage against the customer's original design — what's built, what's reframed, what's disqualified and why |
| [skills/gpu-contention-benchmark/](skills/gpu-contention-benchmark/) | The skill, its design record, and the GPU validation checklist to work through on first hardware contact |

**Agent integration**

| File | What's in it |
|---|---|
| [skills/](skills/) | Six Claude Code / Codex / Cursor skills — `benchmark-gpu-inference`, `gpu-contention-benchmark`, `prepare-nitrogen-dataset`, `setup-inference-backend`, `interpret-benchmark-summary`, `extend-benchmark-config`. Install via `bench install-skill`. |
| [AGENTS.md](AGENTS.md) | Rendered from `skills/` for Codex. Auto-loaded; do not edit by hand. |

---

## Status

**Running today.** The five-stage pipeline (encoder → reasoner → decoder →
validator → executor); all VLM reasoner backends (NIM, vLLM, SGLang, TRT-LLM
over HTTP) plus the NitroGen ZMQ reasoner; video+text scenarios; the benchmark
runner, orchestrator script, metrics and summary writer; the NitroGen
quantization chain (modelopt calibration → ONNX export → TRT plan compile);
GPU sampling via DCGM with an nvidia-smi fallback; AIPerf concurrency sweeps;
and Nsight profiling.

**Hardware validation in progress.** The multi-model contention benchmark —
orchestrator, per-GPU Triton containers, VRAM cap derivation, contention
analysis, and 39 colocations covering all seven phases of the study. 357 unit
tests pass. As of 2026-08-04 the Phase 0 gate gives 2.07x overlap on a PRO
6000, the solo baselines run, and a Triton CV container loads inside MPS —
but no contention window has completed, so **no degradation ratio has been
produced yet**. First hardware contact found eight bugs the unit tests could
not reach; details and what is still open are in
[the validation record](skills/gpu-contention-benchmark/reference/gpu-validation.md).

**Placeholders.** The vision encoder is a passthrough and the executor is
dry-run, so `command_success_rate` currently tracks `grammar_validity_rate`.
ModelOpt for VLMs and the TensorRT+Triton CV-encoder ensemble are documented
but not wired. NVFP4 NitroGen rows are pinned out until TensorRT ships the FP4
plugin.
