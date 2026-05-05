# Quickstart

Three ways to use this repo, picked by what you actually want to measure.


| Mode                        | What it answers                                                                                | GPU?    | Model location                        |
| --------------------------- | ---------------------------------------------------------------------------------------------- | ------- | ------------------------------------- |
| **A. Pipeline smoke**       | "Does the encoder→reasoner→decoder→validator chain wire up and produce schema-valid commands?" | no      | NIM cloud or stub                     |
| **B. Framework benchmark**  | "How does (framework × GPU × model × quant) compare on TTFT, p95 latency, validity rate?"      | **yes** | **local server on the target GPU**    |
| **C. Production rehearsal** | "What does the optimised, NIM-packaged stack feel like end-to-end?"                            | yes     | local NIM container on the target GPU |


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

#### General — driver, CUDA, Docker

```bash
# NVIDIA driver matching your GPU (570+ for RTX 5090 / RTX PRO 6000 Blackwell).
nvidia-smi

# CUDA toolkit (12.8+ required for Blackwell; 12.4+ sufficient for Hopper).
nvcc --version

# Docker + the NVIDIA container toolkit (for NIM containers and TRT-LLM images).
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

All three commands must succeed before continuing. Run from the repo root for all steps below:

```bash
cd qwenvl-inference-pipeline-benchmark
```

#### vLLM

```bash
python3 -m venv .venv-vllm
source .venv-vllm/bin/activate
pip install -e ".[vllm,dev]"

# Confirm:
python -c "import vllm; print(vllm.__version__)"
deactivate
```

#### SGLang

```bash
python3 -m venv .venv-sglang
source .venv-sglang/bin/activate
pip install -e ".[sglang,dev]"

# Confirm:
python -c "import sglang; print(sglang.__version__)"
deactivate
```

#### TRT-LLM

TRT-LLM wheels are GPU/driver-specific and published on NVIDIA's own index — not standard PyPI.

> **Python version:** TRT-LLM 1.x ships wheels for `cp310` and `cp312` — not `cp311`. If the
> install fails, check `python3 --version` and use `python3.10` or `python3.12` explicitly.
>
> **modelopt warning:** you may see a `UserWarning` about `transformers` being incompatible with
> `nvidia-modelopt`. This is harmless for inference — TRT-LLM still works correctly.

```bash
python3 -m venv .venv-trtllm
source .venv-trtllm/bin/activate
pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com
pip install nvidia-modelopt qwen-vl-utils
pip install -e ".[dev]"

# Confirm:
python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
deactivate
```

Run `scripts/gpu_probe.sh` once per host — can be called from any directory:

```bash
bash scripts/gpu_probe.sh
```

Output: `benchmarks/results/host_<hostname>.json`.

---

### B.1 — vLLM (the customer's existing baseline)

```bash
# Shell 1 — start the server.
source .venv-vllm/bin/activate
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --port 8000 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 32

# Shell 2 — run the harness.
source .venv-vllm/bin/activate
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

---

### B.2 — SGLang (low-latency challenger; structured output)

```bash
# Shell 1 — start the server.
source .venv-sglang/bin/activate
sglang serve \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --port 30000

# Shell 2 — run the harness.
source .venv-sglang/bin/activate
export SGLANG_BASE_URL=http://localhost:30000/v1
python -m benchmarks.runner --framework sglang --gpu rtx_pro6000 --model qwen3-vl-8b
```

The point of running SGLang in the matrix is its structured-output
support (regex / EBNF / JSON-schema-constrained sampling). The action
grammar from `vlm_pipeline/schemas.py` should be expressed as a JSON
schema and passed through to SGLang at sample time.

---

### B.3 — TensorRT-LLM (TRT engine-compiled path)

TRT-LLM needs an engine built per (model, GPU, batch shape). VLMs require two engines: one
for the LLM decoder and one for the vision encoder.

> **Why clone TRT-LLM?** `pip install tensorrt-llm` ships the runtime and `trtllm-build` CLI
> but does NOT include the example scripts (`quantize.py`, `build_multimodal_engine.py`).
> NVIDIA only distributes these via GitHub. Clone at the tag matching your installed version
> to avoid import mismatches — no install step needed:
> ```bash
> TRTLLM_VER=$(.venv-trtllm/bin/python -c "import tensorrt_llm; print(tensorrt_llm.__version__)" 2>/dev/null | grep -oP '^\d+\.\d+\.\d+$')
> git clone https://github.com/NVIDIA/TensorRT-LLM.git --depth 1 --branch "v${TRTLLM_VER}"
> ```

TRT-LLM engines are compiled for a specific GPU's compute capability and cannot be shared
across GPU architectures. Set `GPU` to match your config name before running.

```bash
source .venv-trtllm/bin/activate

# Set to match your GPU config: h200 | rtx_pro6000 | rtx5090
GPU=rtx_pro6000

# 0. Download the model locally — convert_checkpoint.py requires a local path, not a hub ID.
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct --local-dir ./hf_models/qwen2-vl-7b

# 1. Convert checkpoint to TRT-LLM format (BF16).
python TensorRT-LLM/examples/models/core/qwen/convert_checkpoint.py \
  --model_dir ./hf_models/qwen2-vl-7b \
  --output_dir ./checkpoints/qwen2-vl-7b-bf16 \
  --dtype bfloat16

# 2a. Build the LLM engine.
trtllm-build \
  --checkpoint_dir ./checkpoints/qwen2-vl-7b-bf16 \
  --output_dir trt_engines/qwen2-vl-7b-${GPU}-bf16/llm \
  --gemm_plugin auto \
  --max_batch_size 16 \
  --max_input_len 2048 \
  --max_seq_len 3072 \
  --max_multimodal_len 1296

# 2b. Build the vision encoder engine.
# If the process is killed during ONNX export (OOM), add swap and reduce --max_num_tiles:
#   sudo fallocate -l 16G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
python TensorRT-LLM/examples/models/core/multimodal/build_multimodal_engine.py \
  --model_type qwen2_vl \
  --model_path ./hf_models/qwen2-vl-7b \
  --output_dir trt_engines/qwen2-vl-7b-${GPU}-bf16/vision \
  --max_batch_size 1 \
  --max_hw_dims 896

# 3. Run the benchmark.
python -m benchmarks.runner --framework trtllm --gpu ${GPU} --model qwen2-vl-7b --quantization bf16
```

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

---

## Appendix

### A1 — TRT-LLM FP8 quantization (blocked upstream)

FP8 quantization for Qwen-VL via `quantize.py` fails in TRT-LLM 1.2.1 with
`AssertionError: The model is not supported` — the vision encoder makes the
checkpoint incompatible with the FP8 export path. Revisit once upstream adds
Qwen-VL multimodal FP8 export support. The commands below are preserved for
reference when that support lands.

```bash
source .venv-trtllm/bin/activate

# Ensure the cloned repo matches the installed version (see B.3 note).
TRTLLM_VER=$(.venv-trtllm/bin/python -c "import tensorrt_llm; print(tensorrt_llm.__version__)" 2>/dev/null | grep -oP '^\d+\.\d+\.\d+$')
git clone https://github.com/NVIDIA/TensorRT-LLM.git --depth 1 --branch "v${TRTLLM_VER}"

huggingface-cli download Qwen/Qwen2-VL-7B-Instruct --local-dir ./hf_models/qwen2-vl-7b

python TensorRT-LLM/examples/quantization/quantize.py \
  --model_dir ./hf_models/qwen2-vl-7b \
  --dtype float16 \
  --qformat fp8 \
  --kv_cache_dtype fp8 \
  --output_dir ./checkpoints/qwen2-vl-7b-fp8 \
  --calib_size 512

trtllm-build \
  --checkpoint_dir ./checkpoints/qwen2-vl-7b-fp8 \
  --output_dir trt_engines/qwen2-vl-7b-pro6000-fp8/llm \
  --gemm_plugin auto \
  --max_batch_size 16 \
  --max_input_len 2048 \
  --max_seq_len 3072 \
  --max_multimodal_len 1296

python TensorRT-LLM/examples/models/core/multimodal/build_multimodal_engine.py \
  --model_type qwen2_vl \
  --model_path ./hf_models/qwen2-vl-7b \
  --output_dir trt_engines/qwen2-vl-7b-pro6000-fp8/vision
```

---

### A2 — TensorRT + Triton ensemble (CV encoder + LLM)

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

### A3 — Mode C: production rehearsal (local NIM container)

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
