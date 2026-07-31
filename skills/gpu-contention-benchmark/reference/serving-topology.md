# Serving topology — vLLM, Triton, MPS and MIG

Why the contention study uses a hybrid topology, explained from first principles.
Verified 2026-07-31.

---

## The three layers people conflate

**vLLM / SGLang / TRT-LLM are inference engines.** They take one model and run it
fast — continuous batching, paged KV cache, fused kernels. Each runs as **its own
OS process with its own CUDA context** and serves HTTP. They are built around one
model per process.

**Triton Inference Server is a serving platform.** It does not execute models
itself; it hosts *backends*. One Triton process can hold **many models at once in
a single CUDA context**, overlapping them on CUDA streams. It provides one
endpoint, model lifecycle management, and dynamic batching.

**MPS (Multi-Process Service) is an NVIDIA daemon**, not a server. Without it, two
processes sharing a GPU are **time-sliced** by the driver — only one runs at a
time, alternating quickly. With MPS, kernels from *different processes* genuinely
execute concurrently on different SMs. It is a daemon plus an environment
variable; no code changes.

**MIG (Multi-Instance GPU) is hardware partitioning.** The card is split into
instances with dedicated SMs, cache and memory bandwidth. Isolation is physical
rather than scheduled.

## How models can share one GPU

| Topology | Mechanism | Build cost | Realism |
|---|---|---|---|
| Separate processes, no MPS | Driver time-slices CUDA contexts | none | What most container deployments do today |
| Separate processes **+ MPS** | True kernel concurrency across processes | daemon + env var | Common on multi-tenant inference nodes |
| One Triton process, many models | CUDA streams inside one context | high — model repo, `config.pbtxt`, per-backend export | Standard for CV fleets |
| **MIG** | Hardware partition | config only, PRO 6000 only | The isolation ceiling |

---

## Why hybrid, and not all-Triton

The customer's design fixes Triton as *the* serving platform for everything. That
does not survive contact with the details:

1. **Triton has no SGLang or llama.cpp backend.** The same design lists
   `[vllm, trt-llm, sglang, llamacpp]` as a swappable LLM-backend dimension. Both
   statements cannot hold — two of those four cannot live inside Triton.
2. **An LLM served by vLLM is already its own process.** So the moment the tenant
   set contains an LLM, you are multi-process regardless, and the "single serving
   platform" justification evaporates.
3. **Triton would understate contention for the LLM side**, because intra-process
   stream concurrency removes the context-switch and scheduler-arbitration costs
   that dominate real multi-tenant co-residency.

**Where each option is genuinely the right tool:**

- **CV models → Triton.** They export cleanly to ONNX/TensorRT, Triton's dynamic
  batching is real value, and its per-request server-side decomposition (queue /
  compute-input / compute-infer / compute-output) is exactly the attribution a
  contention study needs.
- **LLM / VLM → native servers.** That is how they are deployed in practice.

---

## Triton status, and a name collision worth guarding against

**Triton is not deprecated.** v2.71.0 shipped 2026-07-29; the release cadence is
monthly and unbroken. ONNX Runtime, TensorRT and Python backends all had commits
that month. "NVIDIA Dynamo-Triton" is a **rebrand**, not a retirement — Dynamo
itself is a separate distributed orchestrator layered over vLLM/SGLang/TRT-LLM,
not a Triton replacement.

Two different things are called "the TensorRT backend":

| | Status |
|---|---|
| **Triton's `tensorrt_backend`** — runs your YOLO/DINOv2 `.plan` files | **Alive**, gained multi-GPU in 26.07 |
| **TRT-LLM's internal engine-build backend** — LLM only | Removed in TRT-LLM 1.2 |

**Impact of that removal on CV serving: none.** A CV pipeline never imports
`tensorrt_llm`.

### Triton gotchas that will cost you a day

- **Client shared memory is disabled by default since 26.04.** Pass
  `--allow-client-shm=true`. Without it, large CV tensors show inflated latency
  that reads as a model regression but is really serialization overhead.
- **Windows support was removed in 26.06.** Linux only.
- **Jetson**: 26.02 was the last GitHub release.

---

## Load generation — the constraint that shapes everything

**AIPerf cannot drive Triton.** It dropped GenAI-Perf's `kserve` and
`dynamic_grpc` endpoint types; its `--endpoint-type` values are all
OpenAI/NIM/HF-shaped HTTP. AIPerf references Triton *only* as a Prometheus
metrics-scrape target (`--server-metrics`), never as a load target.

So: **two drivers, one wall clock.**

| Tenant | Driver | Open-loop | Arrival pattern | Real payloads |
|---|---|---|---|---|
| LLM / VLM | `aiperf` ≥0.11 | `--request-rate` | `--arrival-pattern {constant,poisson,gamma}` | `--input-file` + `--custom-dataset-type single_turn` |
| CV (Triton) | `perf_analyzer` | `--request-rate-range` | `--request-distribution {constant,poisson}` | `--input-data <path>` |

`perf_analyzer` is still maintained and is **not** deprecated — only `genai-perf`
was superseded by AIPerf. Both tools support open-loop Poisson, which keeps the
two tenants comparable.

What we build is the orchestrator that starts them against a shared `t0` and
merges their traces. Not the load generation itself.

---

## Isolation per GPU

| GPU | MPS | MIG | Study setting |
|---|---|---|---|
| RTX 5090 (32 GB) | ✅ | ❌ consumer part | MPS on |
| RTX PRO 6000 (96 GB) | ✅ | ✅ up to 4 instances | MPS on; **one MIG reference run** |
| H200 (141 GB) | ✅ | ✅ | MPS on |

MPS works on GeForce — the "Tesla and Quadro only" phrasing that circulates is
2013-era legacy prose. Current requirements are just Linux/QNX, compute
capability ≥3.5, 64-bit and UVA. Both Blackwell cards are CC 12.0.

**RTX PRO 6000 Blackwell is the better contention testbed** — it is the only card
in the set where soft sharing (MPS) and hardware partitioning (MIG) can be
compared on identical silicon. MIG there needs driver R575+, vBIOS ≥
98.02.6A.00.03, the Display Mode Selector tool, and open kernel modules.

The single MIG run is worth its cost because it is the **hardware-isolated 1.0×
reference** — the ceiling every soft-shared number is measured against.

---

## A note on DGX Spark

The customer names DGX Spark as a deployment target. It is a different regime and
should be reported separately, not folded into the same matrix.

GB10 Grace Blackwell, **128 GB unified LPDDR5x at 273 GB/s**. Compute-rich but
bandwidth-starved: measured batch-1 Llama 3.1 8B FP8 gives ~7,991 tok/s prefill
against **20.5 tok/s decode** — a ~400:1 ratio that is the signature of a
bandwidth-bound part.

**Why that matters here:** CV and LLM tenants contend for the *same* 273 GB/s
pool, with no HBM headroom to absorb a co-resident stream. Unified memory is a
genuine win for capacity and for zero-copy vision-encoder→LLM handoff (no PCIe
hop), but Spark is a capacity and development box, not a latency box. Report
tok/s/W and maximum co-resident footprint there — not p99 latency.
