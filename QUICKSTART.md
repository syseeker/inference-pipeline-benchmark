# Quickstart

Three ways to use this repo, picked by what you actually want to measure.

| Mode | What it answers | GPU? | Model location |
| --- | --- | --- | --- |
| **A. Pipeline smoke** | "Does the encoder→reasoner→decoder→validator chain wire up and produce schema-valid commands?" | no | NIM cloud or stub |
| **B. Framework benchmark** | "How does (framework × GPU × model × quant) compare on TTFT, p95 latency, validity rate?" | **yes** | **local server on the target GPU** |
| **C. Production rehearsal** | "What does the optimised, NIM-packaged stack feel like end-to-end?" | yes | local NIM container on the target GPU |

You can do A on a laptop. You **must** do B and C on the actual GPU you
want to claim numbers for — there is no way to benchmark a framework
remotely.

---

## Mode A — pipeline smoke (no GPU)

Useful when you're touching the decoder, validator, or schema and just
want to know nothing broke.

```bash
# 1. clone + install
git clone git@github.com:syseeker/qwenvl-inference-pipeline-benchmark.git
cd qwenvl-inference-pipeline-benchmark
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
# Qwen2.5-VL and Qwen3-VL are NOT on NIM cloud — for those, self-host (Mode B/C).

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

This is the real benchmark. The pattern is always:

1. Bring up a local model server on the target GPU.
2. Point the pipeline / runner at `http://localhost:<port>/v1`.
3. Run the timing loop.
4. Capture `gpu_probe` metadata so the result row is self-describing.

### Prerequisites (one-time per host)

```bash
# NVIDIA driver matching your GPU (570+ for RTX 5090 / RTX PRO 6000 Blackwell).
nvidia-smi

# CUDA toolkit (12.4+ recommended for Blackwell).
nvcc --version

# Docker + the NVIDIA container toolkit (for NIM containers and TRT-LLM images).
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# Python venv per framework — vLLM/SGLang/TRT-LLM have conflicting deps,
# so install each in its own venv.
python3 -m venv .venv-vllm   && source .venv-vllm/bin/activate   && pip install -e ".[vllm,dev]"
deactivate
python3 -m venv .venv-sglang && source .venv-sglang/bin/activate && pip install -e ".[sglang,dev]"
deactivate
# TRT-LLM is GPU/driver-specific — install its wheel from NVIDIA's index per docs.
```

Run `scripts/gpu_probe.sh` once per host to capture driver/CUDA/framework
versions into `benchmarks/results/host_<hostname>.json`.

### B.1 — vLLM (the customer's existing baseline)

```bash
source .venv-vllm/bin/activate

# Start the OpenAI-compatible server. Knobs come from the per-GPU yaml.
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --port 8000 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 32

# In another shell, point the harness at it and run.
export VLLM_BASE_URL=http://localhost:8000/v1
python -m benchmarks.runner \
  --framework vllm \
  --gpu rtx_pro6000 \
  --model qwen3-vl-8b \
  --quantization bf16 \
  --concurrency 8 \
  --n-requests 256
```

For the FP8 path, swap the model id to `Qwen/Qwen3-VL-8B-Instruct-FP8`
and pass `--quantization fp8`.

### B.2 — SGLang (low-latency challenger; structured output)

```bash
source .venv-sglang/bin/activate

python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --port 30000 \
  --enable-radix-cache

export SGLANG_BASE_URL=http://localhost:30000/v1
python -m benchmarks.runner --framework sglang --gpu rtx_pro6000 --model qwen3-vl-8b
```

The point of running SGLang in the matrix is its structured-output
support (regex / EBNF / JSON-schema-constrained sampling). The action
grammar from `vlm_pipeline/schemas.py` should be expressed as a JSON
schema and passed through to SGLang at sample time.

### B.3 — TensorRT-LLM (production target)

TRT-LLM needs an engine built per (model, GPU, batch shape). Two-step:

```bash
# 1. Calibrate + quantise via ModelOpt (FP8 example).
python -m modelopt.torch.quantization.calibrate \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --quant_format FP8 \
  --output_dir ./checkpoints/qwen3-vl-8b-fp8

# 2. Build the TRT-LLM engine from the calibrated checkpoint.
trtllm-build \
  --checkpoint_dir ./checkpoints/qwen3-vl-8b-fp8 \
  --output_dir trt_engines/qwen3-vl-8b-pro6000-fp8 \
  --gemm_plugin auto \
  --use_paged_context_fmha enable \
  --max_batch_size 16

# 3. Serve via Triton (see B.4) or trtllm-serve, then run the benchmark.
python -m benchmarks.runner --framework trtllm --gpu rtx_pro6000 --model qwen3-vl-8b --quantization fp8
```

### B.4 — TensorRT + Triton (CV encoder + LLM ensemble)

This is the only adapter that exercises a real CV-encoder + VLM split.
Build a Triton model repository where:

- `cv_encoder` is a TRT engine for the vision tower (or a no-op for v0)
- `vlm_reasoner` is the TRT-LLM engine from B.3
- `decoder` and `validator` are Python BLS models wrapping
  `vlm_pipeline.decoders.action_decoder` and `vlm_pipeline.validators.safety_validator`
- `vlm_pipeline_ensemble` chains them all

```bash
docker run --gpus all -p 8000-8002:8000-8002 \
  -v $(pwd)/triton_model_repo:/models \
  nvcr.io/nvidia/tritonserver:<tag>-py3 \
  tritonserver --model-repository=/models

export TRITON_GRPC_URL=localhost:8001
python -m benchmarks.runner --framework triton --gpu rtx_pro6000 --model qwen3-vl-8b
```

---

## Mode C — production rehearsal (local NIM container)

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

# Point the pipeline / runner at the local NIM endpoint.
export NIM_BASE_URL=http://localhost:8001/v1
export NIM_API_KEY=local      # local NIM accepts any non-empty bearer
python -m examples.run_scenario 01_clash_of_clans_start_attack --backend nim
```

The NIM container exposes the same OpenAI-compatible API as cloud NIM,
so the existing `NimQwenVlReasoner` works unchanged. Re-verify the exact
container tag and model id from `build.nvidia.com/qwen` at run time.

---

## Where do results land?

`benchmarks/runner.py` writes one JSON per (framework, gpu, model,
run-id) under:

```
benchmarks/results/<gpu>/<framework>-<model>-<run_id>.json
```

Raw per-call traces (gitignored) are written next to the summary as
`raw/<run_id>.jsonl`. After a sweep, summarise by hand into
`benchmarks/results/<gpu>/summary.md` so the per-GPU table is reviewable
without re-running.

---

## What this repo is **not**

- Not an autotuner — knob choices live in `benchmarks/configs/<gpu>.yaml`
  and are deliberate.
- Not a kernel / engine project — we lean on vLLM/SGLang/TRT-LLM and
  measure them honestly.
- Not a model trainer — quantisation only (via ModelOpt). Training is
  out of scope.

## Decision metric reminder

Tokens/sec is **not** the success criterion. The pipeline succeeds when
**valid command-sequence latency**, **command success rate**,
**safety/grammar validity**, and **p95/p99 stability** are inside
budget. See [docs/metrics.md](docs/metrics.md).
