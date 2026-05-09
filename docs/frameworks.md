# Inference frameworks

We benchmark **many** and aim production at **one**. The decision is
data-driven: each framework gets a fair single-GPU run before we declare a
winner.

## Roster

### vLLM — baseline

- Already familiar to the customer.
- Strong story: PagedAttention, continuous batching, prefix caching, CUDA
  graphs, multiple quant formats, streaming, structured outputs, and
  multimodal models including Qwen3-VL / Nemotron-Omni / LLaVA / Pixtral.
- Reasoner: `src/vlm_pipeline/reasoners/vllm_backend.py` (talks to a vLLM
  OpenAI-compatible server via `VLLM_BASE_URL`). Driven by
  `benchmarks/runner.py`.

### SGLang — low-latency challenger

- Designed for low-latency / high-throughput serving.
- RadixAttention + prefix caching is well-suited to short-history
  agentic loops where most of the prompt repeats every turn.
- **Structured output** (JSON schema, regex, EBNF) is directly useful for
  the action-decoder stage — we want strict grammar on the command list.
- Reasoner: `src/vlm_pipeline/reasoners/sglang_backend.py`. Driven by
  `benchmarks/runner.py`.

### TensorRT-LLM — PyTorch backend via `trtllm-serve`

TRT-LLM's PyTorch backend loads HF weights directly and runs through
TRT-LLM's PyTorch dispatch with custom CUDA kernels, paged KV cache,
inflight batching, and CUDA graphs — same runtime infrastructure as the
AOT-compiled engine path, just without the build step. Model coverage
tracks upstream day-by-day (`@register_auto_model` in
`tensorrt_llm/_torch/models/`), which is why this is the path that
serves the headline picks (Qwen3-VL, Qwen3.5/3.6, Nemotron-3-Nano-Omni).

Served via `trtllm-serve <hf_id> --backend pytorch` over HTTP — same
OpenAI-compatible chat-completions surface as vLLM and SGLang.

Multimodal models require `kv_cache_config.enable_block_reuse: false`
(TRT-LLM limitation).

Reasoner: `src/vlm_pipeline/reasoners/trtllm_backend.py`. Driven by
`benchmarks/runner.py`.

### ModelOpt — quantisation / compression layer

- Produces FP8 / INT8 / W8A8 calibrated checkpoints, plus pruning,
  distillation, speculative decoding, sparsity.
- Feeds **into** TensorRT-LLM (and where compatible, vLLM/SGLang).
- Not a serving framework — it produces calibrated checkpoints that the
  TRT-LLM reasoner then consumes. The `quant_accuracy_delta` field on
  `BenchmarkResult` is where its output surfaces.

### TensorRT + Triton — end-to-end serving composition

- TensorRT for the CV encoder / vision tower.
- TensorRT-LLM for the LLM/VLM decoder.
- Triton **ensembles** glue them into a single dataflow with no extra
  client-side hops between vision and language.
- Future work — no reasoner exists yet. When wired, it will plug in as
  another `vlm_pipeline/reasoners/*_backend.py` that calls the ensemble
  via tritonclient gRPC.

## What every run reports

Every run emits a `BenchmarkResult` (see `benchmarks/metrics.py`)
written by `benchmarks/runner.py`:

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
