# Inference frameworks

We benchmark **many** and aim production at **one**. The decision is
data-driven: each framework gets a fair single-GPU run before we declare a
winner.

## Roster

### vLLM — baseline

- Already familiar to the customer.
- Strong story: PagedAttention, continuous batching, prefix caching, CUDA
  graphs, multiple quant formats, streaming, structured outputs, and
  multimodal models including Qwen-VL / LLaVA / Pixtral.
- Adapter: `benchmarks/frameworks/vllm_bench.py`.
- Reasoner: `src/vlm_pipeline/reasoners/vllm_backend.py` (talks to a vLLM
  OpenAI-compatible server via `VLLM_BASE_URL`).

### SGLang — low-latency challenger

- Designed for low-latency / high-throughput serving.
- RadixAttention + prefix caching is well-suited to short-history
  agentic loops where most of the prompt repeats every turn.
- **Structured output** (JSON schema, regex, EBNF) is directly useful for
  the action-decoder stage — we want strict grammar on the command list.
- Adapter: `benchmarks/frameworks/sglang_bench.py`.
- Reasoner: `src/vlm_pipeline/reasoners/sglang_backend.py`.

### TensorRT-LLM — TRT engine-compiled path

- The NVIDIA path: optimised engines, FP8 / INT8, KV-cache reuse, CUDA
  graphs, paged attention, multi-GPU paths, in-flight batching.
- Build is heavier (engine compilation per (model, GPU, batch shape)).
- Adapter: `benchmarks/frameworks/trtllm_bench.py`.
- Reasoner: `src/vlm_pipeline/reasoners/trtllm_backend.py`.

### ModelOpt — quantisation / compression layer

- Produces FP8 / INT8 / W8A8 calibrated checkpoints, plus pruning,
  distillation, speculative decoding, sparsity.
- Feeds **into** TensorRT-LLM (and where compatible, vLLM/SGLang).
- Adapter: `benchmarks/frameworks/modelopt_bench.py` runs the calibration
  → quant accuracy delta workflow and emits a small report.

### TensorRT + Triton — end-to-end serving composition

- TensorRT for the CV encoder / vision tower.
- TensorRT-LLM for the LLM/VLM decoder.
- Triton **ensembles** glue them into a single dataflow with no extra
  client-side hops between vision and language.
- Adapter: `benchmarks/frameworks/triton_bench.py` exercises the ensemble
  end-to-end.

## What each adapter must report

Every framework adapter implements `BenchmarkAdapter` and emits a
`BenchmarkResult` (see `benchmarks/metrics.py`):

- TTFT, ITL, end-to-end, vision-encoder latency (p50/p95/p99)
- Throughput (sequences/sec, tokens/sec — diagnostic only)
- Framework-specific knobs that were active (cuda_graph, prefix_cache,
  radix_attention, paged_kv_block_size, fp8, etc.)
- Validation result on each emitted sequence (schema-valid yes/no)
- Memory bandwidth utilisation sample (when DCGM available)

## Pinning

Pin `vllm`, `sglang`, `tensorrt-llm`, `nvidia-modelopt`, `tritonclient`
versions in the GPU-config yamls under `benchmarks/configs/` rather than
in `pyproject.toml`. The exact wheel choice depends on driver / CUDA
version on the host.

## Out of scope (for now)

- llama.cpp / GGUF runtimes — useful as a CPU/edge fallback comparison
  but not in the NVIDIA POC ask.
- TGI — not on the customer's roadmap.
- Custom kernels — we lean on the frameworks before writing kernels.
