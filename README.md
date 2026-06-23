# inference-pipeline-benchmark

A benchmark harness for **VLM-to-action** inference pipelines on NVIDIA
GPUs. Bring your visual scenarios + a target GPU, get apples-to-apples
numbers across **vLLM**, **SGLang**, **TensorRT-LLM** and **Nitrogen** with one
command.

> **Pipeline**: image + short context history + high-level instruction
> → schema-validated low-level command sequence (move / click / keypress
> / say) **OR** continuous gamepad action vector. Pipeline scaffolding
> lives in [src/vlm_pipeline/](src/vlm_pipeline/); the encoder/executor
> are passthrough today, the reasoner is the stage wired to backends.

## Two backend families (don't conflate them)

This harness benchmarks **two kinds of model**, served through **two
kinds of backend**. Knowing which is which is the load-bearing concept:

| Family | Model emits | Served by |
|---|---|---|
| **VLM** (Qwen3-VL / Gemma 4 / Nemotron-Omni) | language tokens, parsed to an `ActionSequence` JSON | **vLLM**, **SGLang**, **TRT-LLM**, **NIM** — OpenAI-compatible HTTP servers |
| **Policy** (NitroGen 500M) | continuous gamepad action directly (no token stream) | **nitrogen-eager** / **nitrogen-compile** / **nitrogen-cudagraph** / **nitrogen-tensorrt** / **nitrogen-onnx** — ZMQ execution engines |

Both families share the same scenario format, the same
`bench {probe,setup,scenarios,smoke,sweep,summary}` surface, and the
same `summary.md` output — so a single run can compare them apples-to-
apples on the same input. See [docs/nitrogen.md](docs/nitrogen.md) for
the policy-vs-VLM distinction in depth and [docs/scenarios.md](docs/scenarios.md)
for the on-disk shape that lets one harness drive both.

---

## Driving it from CLI and an agent

One CLI, one JSON status contract, one prompt per step:

```bash
pip install -e .                                # installs the `bench` console script
bench install-skill --agent auto --json         # wires Claude Code / Codex / Cursor skills
bench probe --json                              # GPU + driver + per-backend versions
bench setup --backend nitrogen-quant --json     # idempotent per-backend venv install
bench scenarios build --source nitrogen --n 3 --synthetic-frames --json
bench smoke --gpu rtx_pro6000 --backend nitrogen-eager --model nitrogen-500m-bf16 --json
bench sweep --gpu rtx_pro6000 --sweep nitrogen-backends --json
bench summary --gpu rtx_pro6000 --json
bench load-test --gpu rtx_pro6000 --backend vllm --model … --concurrency "1,4,16,32" --json
bench profile --tool nsys --gpu rtx_pro6000 --backend nitrogen-eager --model nitrogen-500m-bf16 --json
```

User walkthrough with **natural-language prompts** via agent(recommended, no flag memorisation):
**→ [NITROGEN_QUICKSTART.md](NITROGEN_QUICKSTART.md)** ←

Don't yet know why we built this? Start with
**→ [docs/why-this-matters.md](docs/why-this-matters.md)** — an
engineer-friendly tour of the four budgets your model has to fit
inside (latency, throughput, energy, precision) and how the harness
exposes each one.

## What this harness orchestrates

Under the hood we wrap three NVIDIA tools so customers don't need to learn them:

| Tool | What | When |
|---|---|---|
| [**modelopt**](https://github.com/NVIDIA/TensorRT-Model-Optimizer) | FP8 / NVFP4 PTQ calibration + ONNX export | We run it once on a known-good box; ship the calibrated artifact via [`syseeker-at-nv/nitrogen-quant`](https://huggingface.co/syseeker-at-nv/nitrogen-quant). Customers `hf download`. |
| [**AIPerf**](https://github.com/ai-dynamo/aiperf) | Client-side load generator (OpenAI-compatible) | `bench load-test` wraps it; produces summary.md §9 (concurrency curves for HTTP backends). |
| [**Nsight Systems**](https://developer.nvidia.com/nsight-systems) | GPU timeline profiler | `bench profile --tool nsys` wraps it; escalation tool when summary.md flags a row that needs explanation (auto-installer in `bench setup --backend profile`). |

You can still call each tool directly if you need to — these are
wrappers, not abstractions over.

> **Note on system tools (nsys, modelopt, tensorrt, etc.).** These are
> system binaries / NVIDIA-index wheels, not regular Python packages —
> they're **not in `requirements.txt` or `pyproject.toml`** by design
> (you can't `pip install nsight-systems-cli`). Don't go hunting there.
> Run `bench setup --backend <name>` for each backend you want; it
> handles the system-level installs (apt-get for nsys, NVIDIA wheel
> index for tensorrt + modelopt, post-install chmod and symlinks). You
> only see `sudo` once at install time, then forget about it.

---

## How to run a benchmark — five steps (low-level)

If you'd rather drive the scripts directly (no `bench` CLI), here are
the underlying entry points. Each step links to the doc with the full
detail.

### 1. Pick a GPU profile and a model

The benchmark ships with three GPU profiles — `rtx5090`, `rtx_pro6000`,
`h200` — each with a curated list of candidate models that fit. Skim
[docs/models.md](docs/models.md) to see the picks per GPU and
[docs/capacity.md](docs/capacity.md) for the memory math.

The defaults are sensible (`rtx_pro6000` → `Qwen3-VL-32B-Instruct-FP8`).
You only need this step if you want to change the model.

### 2. Set up the inference server venv

One venv per backend family — VLM serving (vLLM/SGLang/TRT-LLM) goes
in its own venv per server; NitroGen's five execution engines share
**one** venv (`.venv-nitrogen`) because they're all swap-in runtimes
for the same `serve_nitrogen.py`. See [INFERENCE_BACKENDS.md](INFERENCE_BACKENDS.md)
for the full install commands.

```bash
# VLM serving venvs
python3 -m venv .venv-vllm && source .venv-vllm/bin/activate
pip install -e ".[vllm,dev]" && deactivate

python3 -m venv .venv-sglang && source .venv-sglang/bin/activate
pip install -e ".[sglang,dev]" && deactivate

# TRT-LLM (NVIDIA's wheel index — only needed if you want the trtllm leg)
python3 -m venv .venv-trtllm && source .venv-trtllm/bin/activate
pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com
pip install -e ".[dev]" && deactivate

# NitroGen policy venv (covers nitrogen-eager / -compile / -cudagraph
# / -tensorrt / -onnx — five engines, one venv).
python3 -m venv .venv-nitrogen && source .venv-nitrogen/bin/activate
pip install -e ".[nitrogen,nitrogen-quant,dataset,dev]"
pip install -e ../NitroGen                    # clone https://github.com/MineDojo/NitroGen first
hf download nvidia/NitroGen ng.pt             # checkpoint
deactivate
```

### 3. Smoke-test that one server actually works

Before running the full benchmark, bring up one backend and confirm a
single scenario goes through end-to-end. See
[SMOKE_TESTS.md](SMOKE_TESTS.md) for the per-backend launch command and
the smoke-test pytest invocation.

**VLM backend (HTTP):**

```bash
# Shell A: start a server
source .venv-vllm/bin/activate
vllm serve Qwen/Qwen3-VL-32B-Instruct-FP8 --port 8000

# Shell B: run one scenario through it
source .venv-vllm/bin/activate
python -m examples.run_scenario 01_clash_of_clans_start_attack --backend vllm
```

**NitroGen policy backend (ZMQ):**

```bash
# Shell A: start the ZMQ policy server
source .venv-nitrogen/bin/activate
python scripts/serve_nitrogen.py ~/.cache/huggingface/hub/models--nvidia--NitroGen/snapshots/*/ng.pt \
    --port 5560 --exec eager --precision bf16 --steps 16

# Shell B: drive a synthetic scenario via the NitrogenReasoner client
# (or via the runner — see Step 4).
```

If that prints actual-vs-gold action sequences (VLM) or a Gamepad dict
with non-zero latency (NitroGen), your stack is wired correctly.

### 4. Run the benchmark

The orchestrator script starts each backend's server, runs every
scenario through the real `Pipeline`, writes per-scenario + aggregate
JSON, and regenerates the per-GPU `summary.md`. Full operational detail
in [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md).

```bash
# Default VLM model on vllm/sglang/trtllm (rtx_pro6000)
scripts/run_all_scenarios.sh

# A different GPU profile
scripts/run_all_scenarios.sh --gpu h200

# A different VLM model (must be defined in the GPU yaml's `models:` block)
scripts/run_all_scenarios.sh --model qwen3.6-27b-fp8

# A backend-flag A/B comparison (vllm-only knobs in this case)
scripts/run_all_scenarios.sh --backends vllm --variants "baseline eager"

# Auto-run every VLM model in the yaml, on every VLM backend
scripts/run_all_scenarios.sh --sweep models

# NitroGen — all 5 execution engines × precision × denoise-step sweep.
# (Pre-built FP8/NVFP4 ONNX artifacts auto-download from
#  syseeker-at-nv/nitrogen-quant on the first FP8 round; the per-GPU
#  TRT plan is compiled on first use, ~10 s.)
NITROGEN_CKPT_PATH=~/.cache/huggingface/hub/models--nvidia--NitroGen/snapshots/*/ng.pt \
    scripts/run_all_scenarios.sh --gpu rtx_pro6000 --sweep nitrogen-backends \
    --backends "nitrogen-eager nitrogen-compile nitrogen-cudagraph nitrogen-onnx nitrogen-tensorrt" \
    --scenarios-dir tests/smoke/scenarios_nitrogen
```

Outputs land under `benchmarks/results/<gpu>/`:
- `summary.md` — per-GPU aggregated table (regenerated each run)
- `<backend>-<model>-<run_id>.json` — one aggregate `BenchmarkResult` per round
- `<backend>/<scenario>__<run_id>.json` — per-scenario detail

### 5. Run with your own scenarios

Every scenario is a directory with three files — point the runner at a
folder of them and it iterates. The format is documented in
[tests/smoke/scenarios/README.md](tests/smoke/scenarios/README.md):

```
my_scenarios/
├── 01_my_scene/
│   ├── request.json     # ScenarioRequest: instruction, context_history, image_path, deadline_ms
│   ├── screen.png       # the image referenced by request.json (PNG or JPEG)
│   └── expected.json    # ScenarioExpected: gold ActionSequence + ValidationReport
├── 02_another_scene/
│   └── ...
```

Then:

```bash
scripts/run_all_scenarios.sh --scenarios-dir /path/to/my_scenarios
```

The Pydantic models for both files are in
[tests/smoke/scenarios/schema.py](tests/smoke/scenarios/schema.py).
Three reference scenarios live in [tests/smoke/scenarios/](tests/smoke/scenarios/)
— copy one as a template.

---

## Doc map

| File | What's in it |
| --- | --- |
| **README.md** | This file — the journey above, project overview below |
| [INFERENCE_BACKENDS.md](INFERENCE_BACKENDS.md) | Install vLLM / SGLang / TRT-LLM venvs, three operational modes (NIM cloud, local server, NIM container) |
| [SMOKE_TESTS.md](SMOKE_TESTS.md) | Per-backend "is the server alive" check before benchmarking |
| [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md) | Full benchmark operational reference: yaml schema, sweep design, output structure, troubleshooting |
| [docs/models.md](docs/models.md) | Per-GPU model picks (Qwen3-VL, Qwen3.5/3.6, Nemotron-3-Nano-Omni) and the rationale |
| [docs/capacity.md](docs/capacity.md) | Memory math — which checkpoint fits which GPU at BF16 / FP8 / NVFP4 |
| [docs/metrics.md](docs/metrics.md) | What each metric means and why it's tracked (decision metrics vs diagnostics) |
| [docs/frameworks.md](docs/frameworks.md) | Per-framework one-pager (vLLM, SGLang, TRT-LLM PyTorch backend, ModelOpt, Triton) |
| [docs/gpu-strategy.md](docs/gpu-strategy.md) | When to do tensor parallelism vs replicas; PCIe-vs-NVLink considerations |
| [docs/architecture.md](docs/architecture.md) | Pipeline shape (today: VLM-only; v1+: split CV ↔ VLM ↔ decoder ↔ validator) |
| [docs/nitrogen.md](docs/nitrogen.md) | NitroGen diffusion-policy backend: how it works, vs Cosmos 3 / GR00T N1 / VLMs, and the execution-backend optimization study |
| [docs/scenarios.md](docs/scenarios.md) | Scenario shape (request + screen + optional `expected` / `gold_action`), the NitroGen-chunk → scenario mapping, why we convert, and how to add your own dataset source |
| [docs/for-game-sim-teams.md](docs/for-game-sim-teams.md) | For game-AI teams: what this measures for you, player-vs-world-model choice, the bandwidth reality, and per-genre accuracy workflow |
| [docs/why-this-matters.md](docs/why-this-matters.md) | **Andrew-Ng-style intro** for engineers new to inference benchmarking. The four budgets (latency / throughput / energy / precision), why backend choice isn't free, two real questions answered. |
| [NITROGEN_QUICKSTART.md](NITROGEN_QUICKSTART.md) | **Agent-prompt walkthrough** to take the NitroGen sweep end-to-end. Each step shows the prompt + what the agent does + what to expect on disk. The retest doc. |
| [skills/](skills/) | Five Claude Code / Codex / Cursor skills — `benchmark-gpu-inference`, `prepare-nitrogen-dataset`, `setup-inference-backend`, `interpret-benchmark-summary`, `extend-benchmark-config`. Install via `bench install-skill`. |
| [docs/findings/](docs/findings/) | Per-(gpu, framework, model) postmortems referenced by the summary generator (Core findings auto-link). |
| [tests/smoke/scenarios/README.md](tests/smoke/scenarios/README.md) | Scenario file format + how to add your own |

---

## What you're benchmarking

Three families per GPU so the cross-backend comparison covers a VLM
headline, a dense-text TRT-engine win-case, and an NV-tuned multimodal
MoE:

- **Qwen3-VL** (`Qwen/Qwen3-VL-*-Instruct[-FP8]`) — headline VLM.
- **Qwen3.5 / Qwen3.6** (`Qwen/Qwen3.5-9B`, `Qwen/Qwen3.6-27B-FP8`,
  `Qwen/Qwen3.6-35B-A3B-FP8`) — dense text, TRT-LLM trt-engine candidate.
- **Nemotron-3-Nano-Omni** (`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`)
  — NV multimodal MoE (NVFP4 on 5090, FP8 on PRO 6000, BF16 on H200).

| GPU | VRAM | Default model | Why |
| --- | --- | --- | --- |
| RTX 5090 | 32 GB GDDR7 | `qwen3-vl-8b-fp8` | Customer-relevant device; 8B-FP8 is the only Qwen3-VL size that fits with KV headroom |
| RTX PRO 6000 | 96 GB GDDR7 | `qwen3-vl-32b-fp8` | Server workflow; 32B-FP8 leaves comfortable KV room |
| H200 | 141 GB HBM3e | `qwen3-vl-32b-bf16` | Bandwidth ceiling; HBM3e at 4.8 TB/s + BF16 = cleanest accuracy baseline |

Tensor parallelism on consumer cards is **an experiment, not the
default**. See [docs/gpu-strategy.md](docs/gpu-strategy.md) for the
staged proposal.

---

## What the harness reports

Tokens/sec is **not** the decision metric. The pipeline succeeds when:

- **Valid command-sequence latency** — time from image+instruction in
  to a schema-valid command list out — meets the interactive budget.
- **Command success rate** — fraction of generated sequences the
  executor accepts and that achieve the intended outcome.
- **Safety / grammar validity** — fraction passing the validator on
  first try.
- **p95 / p99 stability** — tail latency under realistic concurrency.

Token-level metrics (TTFT, ITL, vision-encoder latency, mem-bw util,
KV-cache hit rate, CUDA-graph delta, quant accuracy loss, TP efficiency)
are tracked as **diagnostics**. See [docs/metrics.md](docs/metrics.md)
for the full list and per-field definitions.

---

## Frameworks under test

| Framework | Role |
| --- | --- |
| vLLM | Baseline (already familiar to the customer) |
| SGLang | Low-latency challenger; RadixAttention + structured output |
| TensorRT-LLM | PyTorch backend via `trtllm-serve` (HTTP, OpenAI-shape — mirrors vLLM/SGLang) |
| ModelOpt | FP8 / NVFP4 / W8A8 quant + calibration (placeholder; not wired yet) |
| TensorRT + Triton | CV encoder + LLM decoder ensemble for end-to-end serving (placeholder) |
| NitroGen | Diffusion **policy** model (not a VLM). Run on execution-engine backends — `nitrogen-eager` / `-compile` / `-cudagraph` / `-tensorrt` / `-onnx` — × precision × denoise steps. See [docs/nitrogen.md](docs/nitrogen.md). |

See [docs/frameworks.md](docs/frameworks.md).

---

## Status

The pipeline (encoder → reasoner → decoder → validator → executor),
all four reasoner backends (NIM, vLLM, SGLang, TRT-LLM HTTP), the
scenario benchmark runner, the orchestrator script, metrics, and the
summary writer all run today. The vision encoder is currently a
passthrough and the executor is dry-run. The TRT-LLM reasoner targets
`trtllm-serve --backend pytorch` — see
[INFERENCE_BACKENDS.md](INFERENCE_BACKENDS.md).
