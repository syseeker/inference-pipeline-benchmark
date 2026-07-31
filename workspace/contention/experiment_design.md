# Contention Experiment Design

## Problem Statement

Measure how heterogeneous AI model workloads (text generation, vision-language, image-language, computer vision) degrade each other's performance when sharing GPU resources. Identify contention thresholds, scaling behavior, and workload isolation boundaries.

## Hardware Target (to be advised)

- NVIDIA RTX PRO 6000 Blackwell Server Edition (98 GB VRAM)
- Scaling to multi-GPU clusters (2, 4 GPUs) when available

---

## Dimensions Explored

### Dimension 1: Model Identity

Specific model under test. Each has a unique compute/memory profile.

**Text Generation**

| ID | Model | Params | VRAM (Q4) | VRAM (FP16) | Role |
|----|-------|--------|-----------|-------------|------|
| qwen2.5-7b | Qwen 2.5 7B | 7B | ~4.5 GB | ~14 GB | Size-scaling anchor (smallest) |
| qwen2.5-14b | Qwen 2.5 14B | 14B | ~8.5 GB | ~28 GB | Size-scaling |
| qwen2.5-27b | Qwen 2.5 27B | 27B | ~16 GB | ~54 GB | Size-scaling |
| qwen2.5-72b | Qwen 2.5 72B | 72B | ~42 GB | ~145 GB | Size-scaling + memory-pressure |
| gemma2-9b | Gemma 2 9B | 9B | ~5.5 GB | ~18 GB | Cross-architecture validation |
| llama3.1-8b | Llama 3.1 8B | 8B | ~5 GB | ~16 GB | Cross-architecture validation |
| mistral-7b | Mistral 7B v0.3 | 7B | ~4.5 GB | ~14 GB | Cross-architecture validation |

Architecture diversity rationale: Qwen 2.5 provides the only 7B→72B same-architecture family needed for size-scaling (Phase 4). Llama 3.1 and Mistral at the same ~7-9B size class validate that contention behavior generalizes beyond Qwen's specific attention implementation (GQA head count, RoPE variant, etc.). If all three architectures show similar degradation ratios under identical neighbor load, the size-scaling conclusions from Qwen-only are generalizable.

Note: Qwen 2.5 72B at Q4 (~42 GB) enables memory-pressure contention scenarios on a single 98 GB GPU. Pairing with Qwen 27B (Q4: 16 GB) = 58 GB total, leaving limited headroom for KV cache growth and co-resident models.

**Vision-Language (Video)**

| ID | Model | Params | VRAM (FP16) | Input |
|----|-------|--------|-------------|-------|
| gemma-vlm-32b | Gemma 32B VLM | 32B | ~64 GB | Video + text |
| qwen2.5-vl-7b | Qwen 2.5 VL 7B | 7B | ~16 GB | Video + image + text |
| qwen2.5-vl-72b | Qwen 2.5 VL 72B | 72B | ~145 GB (FP16) / ~42 GB (Q4) | Video + image + text |

**Image-Language (Document/Image Understanding)**

| ID | Model | Params | VRAM (FP16) | Input |
|----|-------|--------|-------------|-------|
| kosmos-2.5 | Kosmos-2.5 | 1.3B | ~3 GB | Image + text |
| qwen2.5-vl-7b | Qwen 2.5 VL 7B | 7B | ~16 GB | Video + image + text |
| qwen2.5-vl-72b | Qwen 2.5 VL 72B | 72B | ~145 GB (FP16) / ~42 GB (Q4) | Video + image + text |

Note: Qwen2.5-VL models appear in both vision_language and image_language categories as they natively support both video and image inputs. Qwen2.5-VL-72B requires multi-GPU or Q4 quantization on a single 98GB GPU.

**Computer Vision**

| ID | Model | Params | VRAM |
|----|-------|--------|------|
| paddleocr | PaddleOCR v4 | ~12M | ~200 MB |
| yolov8-n | YOLOv8 Nano | 3M | ~50 MB |
| yolov8-l | YOLOv8 Large | 44M | ~300 MB |
| rfdetr-base | RF-DETR Base | ~40M | ~400 MB |
| dinov2-base | DINOv2 ViT-B/14 | 86M | ~350 MB |
| dinov2-large | DINOv2 ViT-L/14 | 300M | ~1.2 GB |

### Dimension 2: Model Type Mix

Composition of models sharing the GPU simultaneously.

| ID | Composition | Example Configuration |
|----|-------------|----------------------|
| llm-only | 2-5 text LLMs | qwen2.5-7b + gemma2-9b |
| cv-only | 2-5 CV models | yolov8 + rfdetr + dinov2 |
| vlm+ilm | Video VLM + Image LM | gemma-vlm-32b + kosmos-2.5 |
| llm+cv | 1-2 LLMs + 1-3 CV | qwen2.5-14b + yolov8 + paddleocr |
| vlm+cv | 1 Video VLM + 1-3 CV | gemma-vlm-32b + rfdetr + dinov2 |
| llm+vlm | 1-2 LLMs + 1 Video VLM | gemma2-9b + gemma-vlm-32b |
| ilm+cv | 1 Image LM + 1-3 CV | kosmos-2.5 + yolov8 + paddleocr |
| full-mix | LLM + VLM + ILM + CV | qwen2.5-7b + gemma-vlm-32b + kosmos-2.5 + yolov8 |

### Dimension 3: Concurrency Level

Rate of requests sent to each model type.

**LLM/VLM (concurrent in-flight requests per model):**

| Level | Value |
|-------|-------|
| c1 | 1 |
| c2 | 2 |
| c4 | 4 |
| c8 | 8 |
| c16 | 16 |

**CV (sustained requests per second):**

| Level | Value |
|-------|-------|
| rps1 | 1 |
| rps10 | 10 |
| rps50 | 50 |
| rps200 | 200 |

### Dimension 4: GPU Count

Number of GPUs available to the workload.

| Level | GPUs | Contention Regime |
|-------|------|-------------------|
| 1gpu | 1 | Full sharing — compute, memory bandwidth, VRAM |
| 2gpu | 2 | Partial isolation or tensor parallel |
| 4gpu | 4 | Most workloads can be isolated |

### Dimension 5: Output Generation Length (LLM/VLM only)

Controls decode duration and KV cache growth.

| Level | Max Tokens | Represents |
|-------|-----------|------------|
| short | 32 | Action/classification token |
| long | 512 | Paragraph generation |

### Dimension 6: Input Size

Controls prefill compute burst.

**LLM (prompt tokens):**

| Level | Tokens |
|-------|--------|
| short-input | 50 |
| long-input | 1000 |

**VLM (video input):**

| Level | Resolution | Duration | FPS | Frames | Approx Tokens |
|-------|-----------|----------|-----|--------|---------------|
| short-clip | 224x224 | 3s | 1 fps | 3 | ~768 |
| long-clip | 1280x720 | 10s | 4 fps | 40 | ~40K+ |

Video introduces sustained prefill bursts (multi-second GPU compute for frame encoding) unlike single-image input which is a brief spike.

**CV (input image size):**

| Level | Resolution |
|-------|-----------|
| cv-small | 320x320 |
| cv-large | 1280x1280 |

### Dimension 7: Quantization (LLM/VLM only)

Controls VRAM footprint — tests whether models fit in VRAM together or trigger swapping.

| Level | Format | VRAM vs FP16 |
|-------|--------|--------------|
| q4 | 4-bit (Q4_0) | ~0.30x |
| fp16 | Float16 | 1.0x (baseline) |

### Dimension 8: Inference Backend

The execution engine that runs the model's forward pass. Different backends produce different GPU kernel patterns — long fused bursts vs many small kernels — which directly affects how much GPU time is available to co-resident models.

**LLM Backend** (CV held fixed):

| Level | Backend | Kernel Pattern | Contention Signature |
|-------|---------|----------------|---------------------|
| vllm | vLLM | Many small PagedAttention kernels, non-contiguous KV blocks | Frequent gaps → neighbors can interleave |
| trt-llm | TensorRT-LLM | Fused large kernels, compiled graph | Long GPU bursts → starves neighbors |
| sglang | SGLang | RadixAttention, prefix sharing reduces total work | Less compute → potentially less contention |
| llamacpp | llama.cpp | Layer-by-layer, frequent CPU-GPU sync | Many sync points → lots of interleaving |

**CV Backend** (LLM held fixed):

| Level | Backend | Kernel Pattern | Contention Signature |
|-------|---------|----------------|---------------------|
| pytorch | PyTorch eager | Many small kernel launches, frequent sync | More scheduling gaps for LLM |
| onnx | ONNX Runtime (CUDA) | Some graph optimization, fewer launches | Moderate |
| tensorrt | TensorRT | Few large fused kernels | Long bursts, less interleaving |

### Dimension 9: Load Asymmetry

Unequal load distribution across models — noisy neighbor test.

| Level | Ratio | Description |
|-------|-------|-------------|
| equal | 1:1 | Symmetric load |
| moderate | 4:1 | One model gets 4x the requests |
| extreme | 16:1 | One model dominates GPU time |

### Dimension 10: Arrival Pattern

How requests arrive over time.

| Level | Pattern | Description |
|-------|---------|-------------|
| burst | All at once | Simultaneous spike, worst-case contention |
| uniform | Fixed interval | Steady-state, evenly spaced requests |

---

## Dimensions Not Explored (and Why)

| Dimension | Reason for Exclusion |
|-----------|---------------------|
| **Warmup policy** | Methodology, not a contention variable. Fixed at 3 warmup rounds excluded from measurement. TTFT metric already captures cold-load penalty independently. |
| **Duration/rounds** | Statistical confidence choice, not a contention variable. Fixed at 10 rounds for all experiments. Longer runs would only be used for final validation of specific findings. |
| **CV batch size** | Redundant with CV RPS (concurrency dimension). Batch size controls GPU compute consumed per unit time, which RPS already captures from the contention perspective. Fixed at model-optimal default (e.g., batch=8 for YOLO). |
| **Placement strategy (multi-GPU)** | Only applies to multi-GPU scenarios. Frameworks have sensible defaults. Defer until multi-GPU results indicate placement matters — add as follow-up experiment if contention persists despite having enough GPUs. |
| **Intermediate quantization levels (Q8, Q5_K_M, Q2_K)** | The key question is binary: does the model set fit in VRAM together or not? Q4 (fits) vs FP16 (may not fit) tests both sides of the swap threshold. Intermediate levels interpolate predictably between these extremes. |
| **Intermediate output lengths (128, 1024, 2048)** | Degradation ratio is expected to remain constant across output lengths — both solo and contention latency scale linearly with decode steps. Two points (32, 512) confirm or refute this assumption. If non-linear effects appear, expand to full sweep. |
| **Intermediate input lengths (200, 500, 4000)** | Prefill is a one-time cost per request. Once in decode phase, contention behavior is independent of how long the input was. Two points (short, long) test whether the prefill burst itself causes neighbor disruption. |
| **Serving framework as a dimension** | The serving framework (Triton, TorchServe, etc.) is an infrastructure choice, not a contention variable. It determines *whether* models share the GPU, not *how* they contend. Replaced by inference backend dimension which directly varies GPU kernel patterns. Serving fixed to Triton (enables true multi-model concurrent GPU execution). |
| **Additional LLM backends (DeepSpeed-MII, ExLlamaV2, LMDeploy)** | Four backends (vLLM, TRT-LLM, SGLang, llama.cpp) cover the fundamental kernel pattern spectrum: paged small kernels, fused large kernels, prefix-shared, and layer-by-layer with sync points. Additional backends are architectural variants of these four. |
| **Arrival patterns (Poisson, staggered, wave)** | Poisson behavior falls between burst and uniform — can be inferred from the two extremes. Staggered and wave are parametric variants of burst with time offset. If burst vs uniform shows significant difference, Poisson can be added as a follow-up. |
| **Load asymmetry ratios (2:1, 8:1)** | Three points (1:1, 4:1, 16:1) define the shape of the fairness degradation curve. Intermediate ratios are interpolatable. If the curve shows unexpected non-linearity, add intermediate points. |
| **Thermal state / clock speed** | Requires hardware-level control (nvidia-smi -lgc) that may not persist across cloud instances. Confounded with duration — longer experiments naturally reach thermal steady state. Observable via power draw metrics without explicit control. |
| **PCIe bandwidth / system RAM** | Fixed by hardware. Cannot vary on a single machine. Would require different instance types to test — out of scope for single-system benchmarking. |

---

## Fixed Methodology (Constant Across All Experiments)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Warmup rounds | 3 | Excluded from measurement. Ensures models loaded, KV cache initialized, GPU thermally primed. |
| Measurement rounds | 10 | GPU inference on dedicated hardware is near-deterministic (<5% run-to-run variance). Contention effects measured here are large (2-10x degradation), not subtle statistical signals. 10 rounds × 2-3 prompts = 20-30 samples — sufficient to establish P50 and identify worst-case spikes. |
| GPU sampling interval | 100ms | Captures sub-second contention spikes (e.g., VLM prefill bursts lasting 200-500ms that disrupt LLM decode). |
| Temperature | 0.7 | Fixed to avoid output length variance confounding latency. |
| Timeout per request | 120s | Captures extreme tail latency without hanging indefinitely. |
| CV batch size | Model-optimal default | YOLO: 8, DETR: 4, DINOv2: 16, PaddleOCR: 8 |
| Serving platform | Triton Inference Server | Enables true multi-model concurrent GPU execution via CUDA streams. Not varied — fixed for all experiments. |
| Placement | Framework default | Defer to scheduler unless multi-GPU results warrant override. |
| Inter-round delay | 0s | No artificial cooldown between rounds (measures sustained behavior). |
| Model load measurement | time_to_first_ready_s, vram_after_load_gb | Cold start penalty is invisible to inference metrics but critical for deployment decisions. Captured passively during model setup. |

### Repetition Policy

| Scenario Class | Repetitions | Rationale |
|----------------|-------------|-----------|
| Standard (all phases) | 1 | GPU inference is deterministic on dedicated hardware with fixed inputs. No production traffic confounders, no sampling noise. A single run with 10 measurement rounds produces stable results. |
| Memory-pressure (Phase 4 `cross-memory-pressure-*`) | 3 | The ONE exception. Near-OOM scenarios exhibit non-deterministic KV cache eviction — the memory allocator's exact allocation order varies between runs, producing bimodal behavior (either fits comfortably or thrashes). This is not measurement noise but genuine runtime variance that single runs cannot capture. 3 repetitions reveal whether a result is stable or bimodal, and provide mean ± std for reporting. |

### Metrics Reporting

| Percentile | Reported | Rationale |
|------------|----------|-----------|
| P50 | Always | Primary metric — median latency under contention. |
| P95 | Always | With 20-30 samples, P95 = 2nd-worst observation. Indicative of tail behavior. |
| Max | Always | Captures worst-case spike (e.g., during VLM prefill burst). |
| P99 | Memory-pressure only | Only meaningful with 3 reps × 30 samples = 90 data points. Insufficient samples for standard runs. |
| Mean ± std (across reps) | Memory-pressure only | Quantifies run-to-run variance caused by non-deterministic KV cache behavior. |

---

## Metrics Collected

### Per-Request (LLM/VLM)

| Metric | Unit | Description |
|--------|------|-------------|
| time_to_first_token | ms | Request send → first streamed token |
| total_latency | ms | End-to-end request duration |
| tokens_per_second | tok/s | Generation throughput |
| inter_token_latency | ms | Average gap between consecutive tokens |
| output_tokens | count | Actual tokens generated |

### Per-Request (CV)

| Metric | Unit | Description |
|--------|------|-------------|
| inference_latency | ms | Input → output duration |
| preprocessing_time | ms | Image resize/normalize time |
| postprocessing_time | ms | NMS/decode time |
| batch_latency | ms | Full batch completion time |

### Per-Experiment (GPU)

| Metric | Unit | Sampling |
|--------|------|----------|
| gpu_util_pct | % | Every 100ms |
| memory_used_mb | MB | Every 100ms |
| memory_bandwidth_util | % | Every 100ms (if available) |
| power_draw_w | W | Every 500ms |
| gpu_temperature_c | C | Every 500ms |
| model_swap_count | count | Detected from memory drop events |

### Aggregate (Per Model Per Experiment)

All per-request metrics reported as: min, max, avg, p50, p90, p99, stddev

### Degradation (Vs Solo Baseline)

| Metric | Computation |
|--------|-------------|
| throughput_ratio | contention_avg / solo_avg (1.0 = no degradation) |
| latency_ratio | contention_avg / solo_avg (1.0 = no degradation) |
| ttft_ratio | contention_avg / solo_avg |
| itl_ratio | contention_avg / solo_avg |
| p99_latency_ratio | contention_p99 / solo_p99 |

---

## Experiment Execution Plan

### Phase 0: Concurrency Validation (Gate — Must Pass)

Validate that Triton achieves true multi-model concurrent GPU execution, not serialization.

- 3 validation runs with nsight/DCGM profiling at 1ms sample rate
- Test LLM+CV, VLM+CV, and standalone-vs-Triton overhead
- **Pass criteria**: overlapping SM activity from both models in same time window (not alternating blocks)
- **If serialized**: pivot experiment to measure time-slicing overhead instead of GPU contention
- Estimated: 3 runs (~20 min total)

### Phase 1: Solo Baselines (Required First)

Run each model in isolation to establish the 1.0x reference for degradation ratios.

- **LLM**: 7 models × 2 quant × 2 output = 28, minus 2 (72B fp16 exceeds 98GB) = 26 runs
- **VLM**: 3 models × 2 input sizes (short-clip, long-clip) = 6 runs
- **ILM**: 3 models × 1 (document image) = 3 runs
- **CV**: 6 models × 1 = 6 runs
- All at 1 GPU, concurrency 1
- Estimated: 41 runs

### Phase 2: Concurrency Sweep (Single Model Type)

Hold model mix constant per category, sweep concurrency to find saturation curves.

- **LLM**: qwen2.5-7b + gemma2-9b at c1, c2, c4, c8, c16 (5 runs)
- **CV**: yolov8-l + rfdetr-base + dinov2-base at rps1, rps10, rps50, rps200 (4 runs)
- **VLM**: gemma-vlm-32b + qwen2.5-vl-7b at c1, c2, c4, c8 (4 runs)
- **ILM**: kosmos-2.5 + qwen2.5-vl-7b at c1, c2, c4, c8 (4 runs)
- Estimated: 17 runs

### Phase 3: Model Type Mix (Fixed Concurrency)

Hold concurrency at c4/rps50, vary composition across all 8 mix types.

- llm-only, cv-only, vlm+ilm, llm+cv, vlm+cv, llm+vlm, ilm+cv, full-mix
- Fixed at 1 GPU, Q4, short output, burst arrival
- Estimated: 8 runs

### Phase 4: Cross-Type Contention Characterization

One model as subject, sweep neighbor load. Includes memory-pressure and size-scaling.

- **LLM vs CV**: qwen2.5-7b at c4, sweep CV RPS 1→200 (4 runs)
- **VLM prefill vs LLM**: gemma-vlm-32b encoding 10s video while gemma2-9b decodes (1 run)
- **ILM vs CV**: kosmos-2.5 at c4, sweep CV RPS 1→200 (4 runs)
- **CV vs LLM**: yolov8-l at rps50, sweep LLM concurrency 1→16 (4 runs)
- **Memory-pressure curve** (3 repetitions each): 4 VRAM utilization points using qwen2.5-72b as anchor:
  - 72B + 7B = 46.5GB (47% util) — comfortable
  - 72B + 14B = 50.5GB (52% util) — moderate
  - 72B + 27B = 58GB (59% util) — high
  - VL-72B + 72B = 84GB (86% util) — extreme, near OOM
- **Size-scaling**: qwen2.5 at 7B/14B/27B/72B as subject under fixed CV load (4 runs)
- **Cross-architecture validation**: qwen2.5-7b / llama3.1-8b / mistral-7b under identical CV load (3 runs) — confirms contention behavior generalizes beyond Qwen
- Estimated: 24 unique runs (32 executions including 3x memory-pressure repetitions)

### Phase 5: GPU Scaling

Repeat key scenarios with 2 and 4 GPUs — does adding GPUs eliminate contention or redistribute it?

- 10 scenarios from Phase 3/4 × 2 GPU configs (2, 4) = 20 runs
- Concurrency re-sweep: qwen2.5-7b + gemma2-9b at c4/c8/c16 × 2/4 GPU = 6 runs
- Estimated: 26 runs

### Phase 6: Secondary Dimension Confirmation

Sweep one secondary dimension at a time against TWO contrasting baselines to detect interaction effects.

**Baseline A** (compute-bound): qwen2.5-7b + gemma2-9b + yolov8-l, c8, rps50
**Baseline B** (memory-bound): qwen2.5-72b + qwen2.5-14b + yolov8-l, c4, rps50

- Output length: short vs long × 2 baselines (4 runs)
- Input size: short-input vs long-input × 2 baselines (4 runs)
- Quantization: q4 vs fp16 × 2 baselines (4 runs)
- LLM backend: vllm/trt-llm/sglang/llamacpp × 2 baselines (8 runs)
- CV backend: pytorch/onnx/tensorrt × 2 baselines (6 runs)
- Load asymmetry: 1:1/4:1/16:1 × 2 baselines (6 runs)
- Arrival pattern: burst/uniform × 2 baselines (4 runs)
- Estimated: 36 runs

Dual-baseline rationale: if a dimension shows similar effect under both compute-bound and memory-bound conditions, it's universally applicable. If it differs, there's an interaction effect (e.g., "backend choice matters 3x more under memory pressure").

### Total Estimated Runs

| Phase | Runs | Executions | Hours |
|-------|------|------------|-------|
| Phase 0: Concurrency validation | 3 | 3 | 0.3 |
| Phase 1: Baselines | 41 | 41 | 4.7 |
| Phase 2: Concurrency sweep | 17 | 17 | 2.4 |
| Phase 3: Model type mix | 8 | 8 | 2.0 |
| Phase 4: Cross-type contention | 24 | 32 | 6.4 |
| Phase 5: GPU scaling | 26 | 26 | 6.5 |
| Phase 6: Secondary dimensions | 36 | 36 | 6.0 |
| **Total** | **155** | **163** | **22-34 hours** |

Notes:
- Phase 4 memory-pressure runs execute 3x each (8 extra executions) — only scenarios with genuine run-to-run variance
- Phase 4 includes 3 cross-architecture validation runs (Qwen vs Llama vs Mistral at same size under identical load)
- Optimistic (22h): warm model cache, fast loads, models grouped to minimize reloading
- Pessimistic (34h): cold starts, 72B load times at 5min, contention runs take 3x solo time

---

## Analysis Plan

### 1. Saturation Curves (Phase 1 + 2)

**Question**: At what concurrency/RPS does each model type saturate the GPU?

**Method**: Plot throughput (normalized to solo baseline) vs concurrency level. One curve per category (LLM, VLM, ILM, CV). Identify the "knee" where throughput plateaus or declines.

**Deliverable**: Per-category saturation point (e.g., "LLMs saturate at c8 on this GPU, CV models sustain linear throughput to rps200").

### 2. Contention Matrix (Phase 1 + 3)

**Question**: Which model type combinations are most/least disruptive to each other?

**Method**: For each Phase 3 mix scenario, compute degradation ratio per model (contention throughput / solo baseline). Assemble into a 4×4 category-level heatmap: rows = victim category, columns = neighbor category, cell = median degradation ratio.

**Deliverable**: Category-level compatibility table. Identifies "safe" pairings (degradation <10%) vs "hostile" pairings (>50% degradation).

**Scope limitation**: This is category-level (4×4), not model-level (17×17). Phase 4 size-scaling and cross-architecture results inform whether model-level detail would differ.

### 3. Disruption Threshold Identification (Phase 4 sweeps)

**Question**: At what neighbor load does a model's performance cross an SLO boundary?

**Method**: For each sweep (e.g., LLM subject + CV neighbor at rps 1/10/50/200), plot subject's P50 latency and throughput vs neighbor load. Annotate the point where degradation exceeds 10%, 25%, 50%.

**Deliverable**: "Safe operating envelopes" — maximum neighbor load that keeps the subject within a given degradation budget. E.g., "qwen2.5-7b at c4 tolerates up to rps50 CV neighbors before TTFT doubles."

### 4. Memory Pressure Curve (Phase 4 memory scenarios)

**Question**: At what VRAM utilization does performance collapse?

**Method**: Plot throughput degradation ratio vs VRAM utilization % across 4 data points (47%, 52%, 59%, 86%). Report mean ± std from 3 repetitions. Identify whether degradation is linear, exponential, or step-function (cliff).

**Deliverable**: VRAM utilization threshold beyond which KV cache eviction begins. Practical guidance: "keep total model weights below X% of VRAM to avoid non-linear degradation."

**Statistical note**: 3 repetitions per point reveal bimodal behavior (stable vs thrashing). If std is high, the scenario is operating at the boundary — report both modes, not the average.

### 5. Model Size Vulnerability (Phase 4 size-scaling)

**Question**: Does model size affect susceptibility to contention?

**Method**: Plot degradation ratio (under fixed CV neighbor load) vs model parameter count (7B, 14B, 27B, 72B) — all same architecture (Qwen 2.5). Determine relationship: linear, flat, or exponential.

**Deliverable**: Guidance on whether to deploy one 72B or multiple smaller models for better aggregate throughput under contention. E.g., "degradation is flat across sizes → pick the largest model that fits" or "degradation scales with size → two 27Bs outperform one 72B under shared GPU."

### 6. Cross-Architecture Generalizability (Phase 4 validation)

**Question**: Do contention results from Qwen generalize to other architectures?

**Method**: Compare degradation ratios of qwen2.5-7b, llama3.1-8b, and mistral-7b under identical neighbor conditions. If all three agree within 20%, Qwen size-scaling conclusions generalize.

**Deliverable**: Confidence statement on generalizability. If architectures diverge: identify which architectural feature (GQA head count, attention implementation, memory access pattern) explains the difference.

### 7. GPU Scaling Efficiency (Phase 5 vs Phase 3/4)

**Question**: Does adding GPUs eliminate contention, and is it cost-efficient?

**Method**: For each scenario repeated at 1/2/4 GPUs, compute:
- Throughput scaling factor (2-GPU throughput / 1-GPU throughput)
- Degradation elimination (does the degradation ratio return to 1.0 at 4 GPUs?)

**Deliverable**: 
- Scaling efficiency table: "scenario X gets 1.8x throughput from 2x GPUs" (ideal = 2.0x)
- Cost-efficiency recommendation: "for LLM+CV workloads, 2 GPUs eliminates 90% of contention; 4 GPUs shows diminishing returns"
- Identification of scenarios where contention persists despite isolation (e.g., shared memory bandwidth on multi-GPU NVLINK)

### 8. Secondary Dimension Sensitivity (Phase 6)

**Question**: Which tuning knobs matter for contention, and which can be ignored?

**Method**: For each secondary dimension, compute the delta in degradation ratio between its extreme values. Rank dimensions by impact. Compare results across both baselines (compute-bound vs memory-bound) to identify interaction effects.

**Deliverable**: Tornado/sensitivity chart showing impact of each dimension. Dimensions grouped into:
- **High impact** (>20% delta): must be specified in deployment recommendations
- **Moderate impact** (5-20% delta): situationally relevant
- **Negligible** (<5% delta): can be ignored in capacity planning

**Interaction effects**: If a dimension shows high impact on one baseline but negligible on the other, document the interaction (e.g., "inference backend matters 3x more under memory pressure than under compute saturation").

### 9. Backend Selection Guide (Phase 6 backend sweeps)

**Question**: Which inference backend is best for multi-tenant GPU deployments?

**Method**: For each LLM backend (vllm/trt-llm/sglang/llamacpp), compare:
- Self-performance (subject's own throughput under contention)
- Neighbor impact (how much the subject's backend hurts co-resident models)

**Deliverable**: 2×2 characterization per backend:
- "Good neighbor" (low impact on others) vs "noisy neighbor" (starves co-residents)
- "Robust" (maintains own performance) vs "fragile" (degrades easily)

Practical recommendation: "use backend X if you prioritize fairness, backend Y if you prioritize single-model throughput."

### 10. Fairness Analysis (Phase 6 asymmetry)

**Question**: Under unequal load, does the GPU scheduler fairly distribute compute?

**Method**: At load ratios 1:1, 4:1, 16:1 — compute per-model throughput as fraction of its solo baseline. Plot both models' degradation vs load ratio.

**Deliverable**: Fairness characterization:
- Proportional (heavy user gets proportionally more, light user proportionally less)
- Winner-take-all (heavy user barely degrades, light user collapses)
- Equalized (both degrade equally regardless of their load ratio)

Scheduling policy recommendation based on findings.

### Composite Deliverables

By combining analyses 2 + 3 + 4 + 7 + 8, produce:

| Deliverable | Description |
|-------------|-------------|
| **Co-location compatibility table** | "These category pairs are safe at load X; these require isolation" |
| **Capacity planning formula** | "For N models of type X at concurrency Y, provision Z GPUs" |
| **SLO budget calculator** | "If LLM P95 ITL budget is 50ms, max co-located CV RPS is {threshold}" |
| **Deployment topology advisor** | "Given models A, B, C: optimal placement is [GPU0: A+C], [GPU1: B alone]" |
| **Backend selection matrix** | "For multi-tenant: use vLLM (fair). For latency-critical single model: use TRT-LLM (fast but noisy)" |

---

## Test Data Requirements

### Minimum Source Data

3 source assets, all other inputs derived via resize/transcode:

| Source | Type | Purpose | Suggested Origin |
|--------|------|---------|-----------------|
| Natural scene image | JPEG | CV models (YOLO, RF-DETR, DINOv2) + VLM frame source | COCO val2017 (e.g., 000000039769.jpg) |
| Document page | PNG | OCR (PaddleOCR, Kosmos-2.5) | FUNSD or SROIE dataset sample |
| Video clip | MP4 | VLM video input | Kinetics-400 sample or screen recording |

### Generated Test Files

```
llm-bench/contention/test_data/
├── source/
│   ├── scene.jpg                  # Natural image with objects (COCO)
│   ├── document.png               # Document with text/tables (FUNSD/SROIE)
│   └── clip_source.mp4            # 10s+ source video (Kinetics-400 or screen recording)
│
├── cv/
│   ├── sample_320x320.jpg         # Resized from scene.jpg — YOLO/DETR default
│   ├── sample_640x640.jpg         # Resized from scene.jpg — YOLO standard
│   ├── sample_1280x1280.jpg       # Resized from scene.jpg — CV large input
│   ├── sample_224x224.jpg         # Resized from scene.jpg — DINOv2 input
│   └── sample_document.png        # Resized from document.png — PaddleOCR (1920x1080)
│
└── vlm/
    ├── clip_3s_224.mp4            # 224x224, 3s, 1fps — VLM short-clip
    └── clip_10s_720p.mp4          # 1280x720, 10s, 4fps — VLM long-clip
```

### Text Prompts (No External Data Needed)

| Set | Count | Token Length | Purpose |
|-----|-------|-------------|---------|
| LLM short prompts | 3 | ~50 tokens | Default baseline input |
| LLM long prompts | 2 | ~1000 tokens | Prefill burst testing |
| VLM video prompts | 2 | ~20 tokens | Paired with video clips |

### Data Preparation

A single prep script generates all derived files from the 3 sources:

1. Download source assets (COCO image, FUNSD doc, Kinetics clip) — or accept user-provided files
2. Resize images to all required CV resolutions via PIL/Pillow
3. Transcode video to required specs via ffmpeg:
   - `ffmpeg -i source.mp4 -vf "scale=224:224,fps=1" -t 3 clip_3s_224.mp4`
   - `ffmpeg -i source.mp4 -vf "scale=1280:720,fps=4" -t 10 clip_10s_720p.mp4`
4. Validate all files exist and meet spec

### VLM Video vs Image — Why Video

VLM inputs are video (not static images) because:

- Video introduces **sustained prefill bursts** — encoding 40 frames at 720p is a multi-second GPU compute event, not a millisecond spike
- Video encoding **grows VRAM** with frame count (frame tokens stored for cross-attention)
- Video prefill **overlaps temporally** with LLM decode, creating measurable interference patterns
- This is the realistic workload — production VLMs process video streams, not single frames
- The contention signature of video VLM (sustained burst) vs image VLM (brief spike) is fundamentally different