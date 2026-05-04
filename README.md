# qwenvl-inference-pipeline-benchmark

Real-time multimodal **VLM-to-action** inference pipeline + benchmark harness.

> **Pipeline goal.** Visual input + short context history + high-level user
> instruction → a short, validated, low-level control-command sequence ready
> for interactive execution.

This repo is the **POC + benchmark scaffolding** for a Razer-facing study
NVIDIA is running. It establishes:

1. A pluggable inference pipeline whose first stage is a NIM-hosted Qwen3-VL
   reasoner, and whose later stages (CV encoder, action decoder, safety
   validator, executor) are scaffolded as placeholders today and will be
   filled in as the architecture lands.
2. A framework-agnostic benchmark harness with placeholders for **vLLM**,
   **SGLang**, **TensorRT-LLM**, **ModelOpt**, and **TensorRT + Triton**.
3. GPU-specific configs for **RTX 5090**, **RTX PRO 6000 Blackwell**, and
   **H200**, so single-GPU baselines can be compared before any
   tensor-parallel experiments.
4. Decision metrics that go **beyond tokens/sec** — valid command-sequence
   latency, command success rate, safety/grammar validity, p95/p99 stability.

## Architecture (target)

```
   ┌────────────┐    ┌───────────────┐    ┌────────────────┐    ┌────────────┐    ┌──────────┐
   │  CV /      │ →  │  Compact VLM  │ →  │  Constrained   │ →  │ Safety /   │ →  │ Executor │
   │  visual    │    │  reasoning    │    │  action-cmd    │    │ command    │    │          │
   │  encoder   │    │  model        │    │  decoder       │    │ validator  │    │          │
   └────────────┘    └───────────────┘    └────────────────┘    └────────────┘    └──────────┘
        ↑                  ↑                      ↑                    ↑
        │                  │                      │                    │
       (TRT)           (NIM Qwen3-VL,         (grammar /            (rules + LLM
                        vLLM, SGLang,          JSON-schema-          guardrail)
                        TRT-LLM)               constrained)
```

Today the CV encoder is a passthrough; the VLM reasoner is the first thing
wired up. See [docs/architecture.md](docs/architecture.md).

## Repo layout

```
.
├── docs/                       # architecture, model list, GPU plan, frameworks, metrics
├── src/vlm_pipeline/           # pipeline stages (encoder/reasoner/decoder/validator/executor)
├── benchmarks/                 # framework-specific benchmark adapters + metrics + GPU configs
├── tests/smoke/                # smoke-test placeholders (golden-path only)
├── examples/                   # minimal end-to-end example scripts
└── scripts/                    # one-shot shell helpers (install, env probes, etc.)
```

## Quickstart

```bash
# 1. Install (once a Python toolchain is decided)
pip install -e ".[dev]"

# 2. Smoke-test the pipeline against a NIM endpoint
export NIM_API_KEY=...                       # NVIDIA NIM key
export NIM_BASE_URL=https://integrate.api.nvidia.com/v1
python -m examples.basic_inference

# 3. Run a benchmark stub on the local GPU
python -m benchmarks.runner --framework vllm --gpu rtx_pro6000 --model qwen3-vl-8b
```

(Most of the above are placeholders — see the
[benchmarks README](benchmarks/README.md) for what is implemented vs. stubbed.)

## Models

Starting model is **Qwen3-VL** (4B/8B for the small targets, larger MoE
variants as a reference). Full curated list in [docs/models.md](docs/models.md).

## GPU plan

| Stage | GPU | Why |
| --- | --- | --- |
| Consumer-target baseline | 1× RTX 5090 | What Razer ships against; 32 GB GDDR7, no NVLink |
| Server POC | 1× RTX PRO 6000 Blackwell | 96 GB GDDR7, MIG, server workflow |
| Bandwidth ceiling | 1× H200 | 141 GB HBM3e @ 4.8 TB/s — clean memory-bw benchmark |

Tensor parallelism on consumer cards is **an experiment, not the default**.
See [docs/gpu-strategy.md](docs/gpu-strategy.md).

## Frameworks

| Framework | Role |
| --- | --- |
| vLLM | Baseline (already familiar to the customer) |
| SGLang | Low-latency challenger; RadixAttention + structured output |
| TensorRT-LLM | Production target on NVIDIA GPUs |
| ModelOpt | FP8 / INT8 / W8A8 quant + calibration |
| TensorRT + Triton | CV encoder + LLM decoder ensemble for end-to-end serving |

See [docs/frameworks.md](docs/frameworks.md).

## Success metrics

Tokens/sec is **not** the decision metric. The pipeline succeeds when:

- **Valid command-sequence latency** — time from image+instruction in to a
  schema-valid command list out — meets the interactive budget.
- **Command success rate** — fraction of generated sequences the executor
  accepts and that achieve the intended outcome.
- **Safety / grammar validity** — fraction passing the validator on first try.
- **p95 / p99 stability** — tail latency under realistic concurrency.

Token-level metrics (TTFT, ITL, vision-encoder latency, mem-bw util,
KV-cache hit rate, CUDA-graph delta, quant accuracy loss, 2× TP efficiency)
are tracked as **diagnostics**. See [docs/metrics.md](docs/metrics.md).

## Status

This commit lays out the scaffolding only. Backends, benchmark loops, and
validators are intentionally stubs that raise `NotImplementedError` so the
shape of the project is reviewable before any heavy implementation lands.
