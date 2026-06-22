# inference-pipeline-benchmark

A benchmark harness for **VLM-to-action** inference pipelines on NVIDIA
GPUs. Bring your visual scenarios + a target GPU, get apples-to-apples
numbers across **vLLM**, **SGLang**, and **TensorRT-LLM** with one
command.

> **Pipeline**: image + short context history + high-level instruction
> → schema-validated low-level command sequence (move / click / keypress
> / say). Pipeline scaffolding lives in [src/vlm_pipeline/](src/vlm_pipeline/);
> the encoder/executor are passthrough today, the reasoner is the stage
> wired to vLLM/SGLang/TRT-LLM.

---

## How to run a benchmark — five steps

If you're a new user reading this for the first time, follow these in
order. Each step links to the doc with the full detail.

### 1. Pick a GPU profile and a model

The benchmark ships with three GPU profiles — `rtx5090`, `rtx_pro6000`,
`h200` — each with a curated list of candidate models that fit. Skim
[docs/models.md](docs/models.md) to see the picks per GPU and
[docs/capacity.md](docs/capacity.md) for the memory math.

The defaults are sensible (`rtx_pro6000` → `Qwen3-VL-32B-Instruct-FP8`).
You only need this step if you want to change the model.

### 2. Set up the inference server venv

One venv per backend, isolated to avoid dependency clashes. See
[INFERENCE_BACKENDS.md](INFERENCE_BACKENDS.md) for the install commands.

```bash
# vLLM
python3 -m venv .venv-vllm && source .venv-vllm/bin/activate
pip install -e ".[vllm,dev]" && deactivate

# SGLang
python3 -m venv .venv-sglang && source .venv-sglang/bin/activate
pip install -e ".[sglang,dev]" && deactivate

# TRT-LLM (NVIDIA's wheel index — only needed if you want the trtllm leg)
python3 -m venv .venv-trtllm && source .venv-trtllm/bin/activate
pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com
pip install -e ".[dev]" && deactivate
```

### 3. Smoke-test that one server actually works

Before running the full benchmark, bring up one backend and confirm a
single scenario goes through end-to-end. See
[SMOKE_TESTS.md](SMOKE_TESTS.md) for the per-backend launch command and
the smoke-test pytest invocation.

```bash
# Shell A: start a server
source .venv-vllm/bin/activate
vllm serve Qwen/Qwen3-VL-32B-Instruct-FP8 --port 8000

# Shell B: run one scenario through it
source .venv-vllm/bin/activate
python -m examples.run_scenario 01_clash_of_clans_start_attack --backend vllm
```

If that prints actual-vs-gold action sequences with non-zero latency,
your stack is wired correctly.

### 4. Run the benchmark

The orchestrator script starts each backend's server, runs every
scenario through the real `Pipeline`, writes per-scenario + aggregate
JSON, and regenerates the per-GPU `summary.md`. Full operational detail
in [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md).

```bash
# Default model on all three backends (rtx_pro6000)
scripts/run_all_scenarios.sh

# A different GPU profile
scripts/run_all_scenarios.sh --gpu h200

# A different model (must be defined in the GPU yaml's `models:` block)
scripts/run_all_scenarios.sh --model qwen3.6-27b-fp8

# A backend-flag A/B comparison (vllm-only knobs in this case)
scripts/run_all_scenarios.sh --backends vllm --variants "baseline eager"

# Auto-run every model in the yaml, on every backend
scripts/run_all_scenarios.sh --sweep models
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
| [docs/for-game-sim-teams.md](docs/for-game-sim-teams.md) | For game-AI teams: what this measures for you, player-vs-world-model choice, the bandwidth reality, and per-genre accuracy workflow |
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
