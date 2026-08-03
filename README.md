# inference-pipeline-benchmark

A benchmark harness for **inference pipelines on NVIDIA GPUs**. Bring your
scenarios and a target GPU; get apples-to-apples latency, throughput, energy
and accuracy numbers across **vLLM**, **SGLang**, **TensorRT-LLM** and the
**NitroGen** execution engines — from one command, one CLI, or one agent prompt.

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

Nine sections in total — decision metrics, latency diagnostics, throughput,
cache & scheduling, GPU resource usage, cross-run deltas, per-scenario detail,
environment, and concurrency profile. See [docs/metrics.md](docs/metrics.md).

---

## What you can benchmark

Two axes: **how many models are loaded**, and **what kind of model** it is.

### Single model (today)

One model, one backend, one round at a time. Two model families:

| Family | Input → output | Backends |
|---|---|---|
| **VLM** — image+text | screenshot + instruction → schema-valid action sequence | vLLM · SGLang · TRT-LLM · NIM |
| **VLM** — video+text | short video + question → free-form analysis | vLLM · SGLang · TRT-LLM |
| **Policy** — NitroGen | game frame → continuous gamepad action | 5 execution engines (see below) |

**Bring your own scenarios.** Each type reads a directory of scenario folders.
Three image scenarios ship ready to run; for video and policy you supply the
inputs. Point any run at a different folder with `--scenarios-dir`.

| Type | Default directory | What ships | To add your own |
|---|---|---|---|
| image+text | `tests/smoke/scenarios/` | 3 game scenarios, runnable as-is | Copy a folder as a template: `screen.png` + `request.json` (instruction, context history, deadline) + `expected.json` (gold action sequence) |
| video+text | `tests/smoke/scenarios_video/` | 3 folders with placeholder `request.json` / `expected.json` | Drop your `.mp4` in, then fill in `prompt` and the `key_phrases` you expect in a correct answer |
| policy | `tests/smoke/scenarios_nitrogen/` | nothing — generated on demand | `bench scenarios build --source nitrogen --n 3` pulls frames + gold gamepad actions from the NitroGen dataset |

Format and Pydantic models: [docs/scenarios.md](docs/scenarios.md) and
[tests/smoke/scenarios/README.md](tests/smoke/scenarios/README.md). Registering
a custom dataset source as an entry-point is covered in
[docs/scenarios.md](docs/scenarios.md).

### Multiple models concurrently (planned)

Several models resident on one GPU at the same time, serving interleaved
traffic — measuring how memory pressure, scheduler contention and KV-cache
sharing change the per-model numbers. Not yet implemented; see
[Status](#status).

---

## The two model families (don't conflate them)

Within single-model benchmarking, the load-bearing distinction is what the
model emits and therefore how it is served:

| Family | Model emits | Served by |
|---|---|---|
| **VLM** (Qwen3-VL / Gemma 4 / Nemotron-Omni) | language tokens, parsed to an `ActionSequence` JSON or read as free text | **vLLM**, **SGLang**, **TRT-LLM**, **NIM** — OpenAI-compatible HTTP servers |
| **Policy** (NitroGen 500M) | a continuous gamepad action directly — no token stream | **nitrogen-eager** / **-compile** / **-cudagraph** / **-tensorrt** / **-onnx** — ZMQ execution engines |

Both families share the same scenario format, the same `bench` command surface,
and the same `summary.md` — so one run can compare them on the same input.
Depth: [docs/nitrogen.md](docs/nitrogen.md) for the policy-vs-VLM distinction,
[docs/scenarios.md](docs/scenarios.md) for the on-disk shape that drives both.

---

## Start here

| What you want to do | Go to |
|---|---|
| Benchmark a **VLM** on image or video scenarios | **[QUICKSTART_VLM.md](QUICKSTART_VLM.md)** |
| Benchmark the **NitroGen policy model** across execution engines | **[QUICKSTART_NITROGEN.md](QUICKSTART_NITROGEN.md)** |
| Understand *why* any of this matters before running it | **[docs/why-this-matters.md](docs/why-this-matters.md)** |
| Evaluate this for a game-AI or simulation team | **[docs/for-game-sim-teams.md](docs/for-game-sim-teams.md)** |
| Look up a flag, a yaml field, or an output path | **[BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md)** |

---

## Three ways to drive it

Every benchmark can be run three ways. They are the same code path — the CLI
wraps the scripts, and the agent skills wrap the CLI. Pick whichever suits you.

1. **Agent prompt** *(recommended)* — "Run the nitrogen sweep and tell me the
   winner." Five skills ship in [skills/](skills/); install with
   `bench install-skill`. Works with Claude Code, Codex and Cursor. No flags to
   memorise, and the agent reads `summary.md` back to you in plain language.
2. **`bench` CLI** — `bench {probe,setup,scenarios,smoke,sweep,summary,load-test,profile}`.
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
gamepad action.

Token-level metrics (TTFT, ITL, vision-encoder latency, mem-bw util, KV-cache
hit rate, CUDA-graph delta, quant accuracy loss, TP efficiency) are tracked as
**diagnostics**. Full per-field definitions in [docs/metrics.md](docs/metrics.md).

---

## What this harness orchestrates

We wrap three NVIDIA tools so you don't have to learn them first:

| Tool | What | When |
|---|---|---|
| [**modelopt**](https://github.com/NVIDIA/TensorRT-Model-Optimizer) | FP8 / NVFP4 PTQ calibration + ONNX export | Run once on a known-good box; the calibrated artifact ships via [`syseeker-at-nv/nitrogen-quant`](https://huggingface.co/syseeker-at-nv/nitrogen-quant) and auto-downloads on the first FP8 round. |
| [**AIPerf**](https://github.com/ai-dynamo/aiperf) | Client-side load generator (OpenAI-compatible) | `bench load-test` wraps it; fills `summary.md` §9 with concurrency curves for HTTP backends. |
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
| [docs/contention.md](docs/contention.md) | Multi-model GPU sharing — what a degradation ratio means, why the baseline is the hard part, and the 1/2/4-GPU topologies |
| [docs/models.md](docs/models.md) | Per-GPU model picks, the rationale, and hypothesis-vs-measured-reality |
| [docs/capacity.md](docs/capacity.md) | Memory math — which checkpoint fits which GPU at BF16 / FP8 / NVFP4 |
| [docs/nitrogen.md](docs/nitrogen.md) | The NitroGen diffusion policy: how it works, vs Cosmos 3 / GR00T N1 / VLMs, and the execution-backend study |
| [docs/scenarios.md](docs/scenarios.md) | Scenario shape, the NitroGen-chunk → scenario mapping, and how to register your own dataset source |
| [docs/frameworks.md](docs/frameworks.md) | Per-framework one-pager (vLLM, SGLang, TRT-LLM PyTorch backend, ModelOpt, Triton) |
| [docs/gpu-strategy.md](docs/gpu-strategy.md) | Tensor parallelism vs replicas; PCIe-vs-NVLink considerations |
| [docs/architecture.md](docs/architecture.md) | Pipeline shape — today VLM-only; v1+ splits CV ↔ VLM ↔ decoder ↔ validator |
| [docs/findings/](docs/findings/) | Per-(gpu, framework, model) postmortems that the summary generator auto-links from Core findings |

**Agent integration**

| File | What's in it |
|---|---|
| [skills/](skills/) | Five Claude Code / Codex / Cursor skills — `benchmark-gpu-inference`, `prepare-nitrogen-dataset`, `setup-inference-backend`, `interpret-benchmark-summary`, `extend-benchmark-config`. Install via `bench install-skill`. |
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

**Placeholders.** The vision encoder is a passthrough and the executor is
dry-run, so `command_success_rate` currently tracks `grammar_validity_rate`.
ModelOpt for VLMs and the TensorRT+Triton CV-encoder ensemble are documented
but not wired. NVFP4 NitroGen rows are pinned out until TensorRT ships the FP4
plugin.

**Planned.** Multi-model concurrent benchmarking — several models resident on
one GPU serving interleaved traffic, to measure memory-pressure and
scheduler-contention effects that single-model rounds cannot surface.
