# Inference backend setup

## One-shot via `bench setup`

For a recent install (PRs #2/#5.0.5/#7.1), there's a wrapper that does
the venv + extras + license/auth checks in one command per backend:

| Backend | Command | What you also need (the wrapper tells you in `next_action`) |
|---|---|---|
| `vllm`   | `bench setup --backend vllm --json` | nothing — pip pulls everything |
| `sglang` | `bench setup --backend sglang --json` | nothing |
| `trtllm` | `bench setup --backend trtllm --json` | `pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com` (NVIDIA index — not on PyPI) |
| `nim`    | `bench setup --backend nim --json` | `export NIM_API_KEY=nvapi-…` |
| `nitrogen` | `bench setup --backend nitrogen --json` | clone `MineDojo/NitroGen` + `pip install -e ../NitroGen` + `hf download nvidia/NitroGen ng.pt` |
| `nitrogen-quant` | `bench setup --backend nitrogen-quant --json` | same NitroGen prep AS above. Adds [`nvidia-modelopt`](https://github.com/NVIDIA/TensorRT-Model-Optimizer) + `onnxruntime-gpu` + `tensorrt`. **Pre-built FP8/NVFP4 ONNX artifacts download from [`syseeker-at-nv/nitrogen-quant`](https://huggingface.co/syseeker-at-nv/nitrogen-quant) on first sweep — customers don't recalibrate.** |
| `profile` | `bench setup --backend profile --json` | nothing — auto-`apt-get install` of the latest `nsight-systems-YYYY.X.Y`, post-install chmod + `/usr/local/bin/nsys` symlink. Falls back to a tarball-install hint when apt isn't available. |

All of these are idempotent (skip when the venv/tool already exists, pass `--force` to rebuild). All emit JSON status, exit-code 0/2/3/4 (see [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md) §CLI wrapper).

> **About system tools vs Python deps.** `requirements.txt` and
> `pyproject.toml` only carry pip-installable packages. The `trtllm`,
> `nitrogen-quant`, and `profile` rows install **system binaries** or
> NVIDIA-index wheels under the hood (apt-get / NVIDIA pypi index),
> which is why you won't find `nsight-systems-cli` or `tensorrt-llm`
> in those files. `bench setup --backend <name>` handles each
> category: `pip install -e ".[X]"` for Python extras, `apt-get` for
> nsys, and the right `--extra-index-url` for the NVIDIA wheels. You
> get prompted for `sudo` once per category, never again.

The per-backend recipes below describe **what `bench setup` does under the hood**. Read them when you're debugging or doing custom work; for the standard flow `bench setup --backend <name>` is the only command you need.

## Three operational modes — pick by what you actually want to measure

| Mode | What it answers | GPU? | Model location | Where in this doc |
| --- | --- | --- | --- | --- |
| **A. Pipeline smoke (NIM cloud)** | "Does the encoder→reasoner→decoder→validator chain wire up and produce schema-valid commands?" | no | NIM cloud or stub | [Mode A](#mode-a--pipeline-smoke-no-gpu) |
| **B. Framework benchmark** | "How does (framework × GPU × model × quant) compare on TTFT, p95 latency, validity rate?" | **yes** | **local server on the target GPU** | [Mode B](#mode-b--framework-benchmark-on-a-local-gpu) |
| **C. Production rehearsal (local NIM container)** | "What does the optimised, NIM-packaged stack feel like end-to-end?" | yes | local NIM container on the target GPU | [Appendix A2](#a2--local-nim-container-mode-c--production-rehearsal) |

Mode A runs on a laptop. Modes B and C **must** run on the actual GPU
you want to claim numbers for — there is no way to benchmark a
framework remotely. The TensorRT+Triton ensemble adapter (CV encoder +
VLM split) lives in [Appendix A1](#a1--tensorrt--triton-ensemble-cv-encoder--llm).

---

## Mode A — pipeline smoke (no GPU)

Useful when you're touching the decoder, validator, or schema and just
want to know nothing broke.

```bash
# 1. clone + install
git clone git@github.com:syseeker/inference-pipeline-benchmark.git
cd inference-pipeline-benchmark
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,nim]"

# 2. offline (uses gold stub — no credentials needed)
pytest -m "not nim" -q

# 3. run a scenario through the offline stub for eyeballing
python -m examples.run_scenario 01_clash_of_clans_start_attack

# 4. (optional) live NIM round-trip — needs an NVIDIA NIM API key
export NIM_API_KEY=nvapi-...
export NIM_BASE_URL=https://integrate.api.nvidia.com/v1

# Find a model id your key can reach (the catalogue rotates):
bash scripts/list_nim_models.sh qwen
# Today (2026-05) the only multimodal Qwen NIM is qwen/qwen3.5-397b-a17b.
# Qwen2.5-VL and Qwen3-VL are NOT on NIM cloud — for those, self-host (Mode B).

export NIM_MODEL=qwen/qwen3.5-397b-a17b
python -m examples.run_scenario 01_clash_of_clans_start_attack --backend nim
pytest -m nim tests/smoke/test_nim_live.py -q
```

> ⚠️ **Common 404.** If you see `NIM returned 404 for model '<id>'`,
> your `NIM_MODEL` doesn't match anything served at `NIM_BASE_URL`. Run
> `bash scripts/list_nim_models.sh` to see what's available.

This tells you **nothing about performance** — NIM cloud sits behind a
shared queue and your local network. It's purely a correctness check.

---

## Mode B — framework benchmark on a local GPU

The pattern is always:
1. Bring up a local model server on the target GPU.
2. Point the runner at `http://localhost:<port>/v1`.
3. Run the timing loop.

Each backend section below has **Install** (one-time per host) and
**Run / Test** (per benchmark session).

### Prerequisites — driver, CUDA, Docker

Required once per host before any backend install.

```bash
# NVIDIA driver matching your GPU (570+ for RTX 5090 / RTX PRO 6000 Blackwell).
nvidia-smi

# CUDA toolkit (12.8+ required for Blackwell; 12.4+ sufficient for Hopper).
nvcc --version

# Docker + the NVIDIA container toolkit (for NIM containers and TRT-LLM images).
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

All three commands must succeed before continuing. Run from the repo
root for everything below:

```bash
cd inference-pipeline-benchmark
```

Capture host-level metadata once (driver, CUDA, GPU model, topology) so
result rows are self-describing:

```bash
bash scripts/gpu_probe.sh
# Output: benchmarks/results/host_<hostname>.json
```

---

### B.1 — vLLM (the customer's existing baseline)

#### Install

```bash
python3 -m venv .venv-vllm
source .venv-vllm/bin/activate
pip install -e ".[vllm,dev]"

# Confirm:
python -c "import vllm; print(vllm.__version__)"   # → 0.20.1
deactivate
```

#### Run / Test

```bash
# Shell 1 — start the server.
# (Or just run `scripts/run_all_scenarios.sh --backends vllm --gpu rtx_pro6000`
#  which reads the launch flags from benchmarks/configs/rtx_pro6000.yaml and
#  starts/stops the server for you.)
#
# Do NOT add --enable-prefix-caching: it crashes Qwen3-VL with chunked
# prefill on cache hits. See BENCHMARK_GUIDE.md troubleshooting.
source .venv-vllm/bin/activate

# Example uses the rtx_pro6000 default. Substitute the right HF id for
# your GPU — per-GPU defaults in docs/models.md and benchmarks/configs/<gpu>.yaml.
vllm serve Qwen/Qwen3-VL-32B-Instruct-FP8 \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 32

# Shell 2 — run the harness.
source .venv-vllm/bin/activate
export VLLM_BASE_URL=http://localhost:8000/v1
python -m benchmarks.runner --backend vllm --gpu rtx_pro6000
```

Per-GPU headline picks: `Qwen3-VL-8B-Instruct-FP8` on 5090,
`Qwen3-VL-32B-Instruct-FP8` on PRO 6000, `Qwen3-VL-32B-Instruct` BF16
on H200. For Qwen3.6 / Nemotron Omni and other candidates see
[docs/models.md](docs/models.md) and the per-GPU YAML.

For the smoke-test pytest invocation (one scenario through the live
server), see [SMOKE_TESTS.md](SMOKE_TESTS.md) §B.1.

---

### B.2 — SGLang (low-latency challenger; structured output)

#### Install

```bash
python3 -m venv .venv-sglang
source .venv-sglang/bin/activate
pip install -e ".[sglang,dev]"

# Confirm:
python -c "import sglang; print(sglang.__version__)"   # → 0.5.11
deactivate
```

#### Run / Test

```bash
# Shell 1 — start the server.
source .venv-sglang/bin/activate
# Example uses the rtx_pro6000 default; substitute for your GPU.
sglang serve \
  --model-path Qwen/Qwen3-VL-32B-Instruct-FP8 \
  --port 30000

# Shell 2 — run the harness.
source .venv-sglang/bin/activate
export SGLANG_BASE_URL=http://localhost:30000/v1
python -m benchmarks.runner --backend sglang --gpu rtx_pro6000
```

The point of running SGLang in the matrix is its structured-output
support (regex / EBNF / JSON-schema-constrained sampling). The action
grammar from `vlm_pipeline/schemas.py` should be expressed as a JSON
schema and passed through to SGLang at sample time.

For the smoke-test pytest invocation, see [SMOKE_TESTS.md](SMOKE_TESTS.md) §B.2.

---

### B.3 — TensorRT-LLM (PyTorch backend)

The headline picks (Qwen3-VL, Qwen3.5/3.6, Nemotron-3-Nano-Omni) all run
through TRT-LLM's **PyTorch backend** via `trtllm-serve`. Same TRT-LLM
runtime infrastructure (paged KV cache, inflight batching, CUDA graphs,
custom kernels) — just no AOT engine compile, so model coverage tracks
upstream day-by-day. This is the apples-to-apples comparison with vLLM
and SGLang.

#### Install

TRT-LLM wheels are GPU/driver-specific and published on NVIDIA's own
index — not standard PyPI.

> **Python version:** TRT-LLM 1.x ships wheels for `cp310` and `cp312`
> — not `cp311`. If the install fails, check `python3 --version` and
> use `python3.10` or `python3.12` explicitly.
>
> **modelopt warning:** you may see a `UserWarning` about `transformers`
> being incompatible with `nvidia-modelopt`. This is harmless for
> inference — TRT-LLM still works correctly.

```bash
python3 -m venv .venv-trtllm
source .venv-trtllm/bin/activate
# Pinned to match the version recorded in pyproject.toml comments.
pip install tensorrt-llm==1.2.1 --extra-index-url https://pypi.nvidia.com
pip install -e ".[dev]"

# Confirm:
python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"   # → 1.2.1
deactivate
```

#### Run / Test

> **VLM constraint.** TRT-LLM multimodal is incompatible with KV-cache
> reuse. The launch command below disables it via the override file —
> create the file once on each host.
>
> **No schema-guided decoding for Qwen3-VL on 1.2.1.** Unlike vLLM/SGLang,
> TRT-LLM 1.2.1 cannot enforce `json_schema` / `json_object`
> response_format on Qwen3-VL. Enabling
> `guided_decoding_backend: xgrammar` in the YAML triggers a startup
> crash:
> ```
> AttributeError: 'Qwen3VLModel' object has no attribute 'vocab_size_padded'
> ```
> The guided-decoder constructor reads `vocab_size_padded` directly off
> the top-level model
> ([py_executor_creator.py:504](https://github.com/NVIDIA/TensorRT-LLM)),
> but Qwen3-VL is a multimodal wrapper — that attribute lives on the
> inner language model. Until upstream fixes this, **leave xgrammar
> off** and accept that TRT-LLM's `valid=True` rate on Qwen3-VL will
> trail vLLM/SGLang. That's a real cross-backend finding worth
> reporting, not a harness bug.

The trtllm-vlm extra-options YAML is committed at
[`benchmarks/configs/trtllm-vlm.yml`](benchmarks/configs/trtllm-vlm.yml)
— previously written into `/tmp/`, but `/tmp/` is wiped on instance
restart. Run from the repo root so the relative path resolves.

```bash
source .venv-trtllm/bin/activate
cd "$(git rev-parse --show-toplevel)"   # cwd must be repo root

# Shell 1 — start the server. Pick the HF id for your GPU from docs/models.md;
# example below is the PRO 6000 headline (Qwen3-VL-32B-FP8).
trtllm-serve Qwen/Qwen3-VL-32B-Instruct-FP8 \
  --backend pytorch \
  --port 8002 \
  --extra_llm_api_options benchmarks/configs/trtllm-vlm.yml

# Shell 2 — run the harness.
source .venv-trtllm/bin/activate
export TRTLLM_BASE_URL=http://localhost:8002/v1
python -m benchmarks.runner --backend trtllm --gpu rtx_pro6000
```

Multimodal models are chat-API only (need a `chat_template`); the
harness talks to `/v1/chat/completions`. PyTorch-backend metrics are
still beta in TRT-LLM 1.2.x — some detailed perf counters are thinner
than vLLM/SGLang expose.

For the smoke-test pytest invocation, see [SMOKE_TESTS.md](SMOKE_TESTS.md) §B.3.

---

## Appendix

Optional adapters — not part of the headline cross-backend matrix.

### A1 — TensorRT + Triton ensemble (CV encoder + LLM)

The only adapter that exercises a real CV-encoder + VLM split. Build a
Triton model repository where:

- `cv_encoder` is a TRT engine for the vision tower (or a no-op for v0)
- `vlm_reasoner` is the TRT-LLM engine from B.3
- `decoder` and `validator` are Python BLS models wrapping
  `vlm_pipeline.decoders.action_decoder` and
  `vlm_pipeline.validators.safety_validator`
- `vlm_pipeline_ensemble` chains them all

```bash
docker run --gpus all -p 8000-8002:8000-8002 \
  -v $(pwd)/triton_model_repo:/models \
  nvcr.io/nvidia/tritonserver:<tag>-py3 \
  tritonserver --model-repository=/models

export TRITON_GRPC_URL=localhost:8001
python -m benchmarks.runner --backend triton --gpu rtx_pro6000
```

### A2 — Local NIM container (Mode C — production rehearsal)

NIM is published as Docker containers per model; running one locally
gives you NVIDIA's optimised stack as a single benchmark target you can
pit against vLLM / SGLang / TRT-LLM.

```bash
# Pull the container (model id and tag come from build.nvidia.com).
docker login nvcr.io
docker pull nvcr.io/nim/qwen/qwen2.5-vl-7b-instruct:<tag>

# Run on the target GPU. Cache mounted so weights aren't redownloaded.
docker run --rm --gpus all \
  --shm-size=16g -p 8001:8000 \
  -v $HOME/.cache/nim:/opt/nim/.cache \
  -e NGC_API_KEY=$NGC_API_KEY \
  nvcr.io/nim/qwen/qwen2.5-vl-7b-instruct:<tag>

# Point the runner at the local NIM endpoint.
export NIM_BASE_URL=http://localhost:8001/v1
export NIM_API_KEY=local      # local NIM accepts any non-empty bearer
python -m examples.run_scenario 01_clash_of_clans_start_attack --backend nim
```

The NIM container exposes the same OpenAI-compatible API as cloud NIM,
so the existing `NimQwenVlReasoner` works unchanged. Re-verify the exact
container tag and model id from `build.nvidia.com/qwen` at run time.
