# Model catalogue — verified sources and per-GPU scoping

Every model in the customer's `workspace/contention/experiment_config.json`,
verified against HuggingFace and upstream repos on **2026-07-31**.

## Governing rule

The customer chose these models for a reason: the deployment target is consumer
and prosumer hardware (RTX GeForce, RTX PRO 6000, DGX Spark), so models must stay
small enough that **1 CV + 1 LLM/VLM fit together**. This is a co-residency
study, not a best-latest-model study.

Therefore:

- **Never drop a model that still runs on one of the three test GPUs** (5090 /
  PRO 6000 / H200). Scope it to the tiers where it fits instead.
- **Substitutions are suggestions to raise with the customer**, never silent swaps.
- Where a pick is factually broken, keep it in config and **fail safely** with a
  reason — do not delete it.

The framework exists so the customer can swap in newer models themselves later.

---

## Verified — text LLM

| ID | HF source | Params | 4-bit source |
|---|---|---|---|
| `qwen2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` | 7.61 B | `…-AWQ` |
| `qwen2.5-14b` | `Qwen/Qwen2.5-14B-Instruct` | 14.7 B | `…-AWQ` |
| `qwen2.5-32b` | `Qwen/Qwen2.5-32B-Instruct` | 32.5 B | `…-AWQ` |
| `qwen2.5-72b` | `Qwen/Qwen2.5-72B-Instruct` | 72.7 B | `…-AWQ` |
| `gemma2-9b` | `google/gemma-2-9b-it` *(gated)* | 9 B | — |
| `llama3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` *(gated)* | 8 B | — |
| `mistral-7b` | `mistralai/Mistral-7B-Instruct-v0.3` | 7 B | — |

## Verified — vision-language

| ID | HF source | Video? | 4-bit |
|---|---|---|---|
| `qwen2.5-vl-7b` | `Qwen/Qwen2.5-VL-7B-Instruct` | ✅ | **official AWQ** |
| `qwen2.5-vl-72b` | `Qwen/Qwen2.5-VL-72B-Instruct` | ✅ | official AWQ |

Qwen2.5-VL is the only Qwen-org family with official 4-bit checkpoints at every
size. No official GPTQ exists — `…-GPTQ-Int4` returns 401.

## Verified — computer vision

| ID | Source | Triton backend |
|---|---|---|
| `yolov8-n` / `yolov8-l` | `ultralytics` 8.4.x | TensorRT / ONNX |
| `dinov2-base` / `-large` | `facebook/dinov2-{base,large}` (Apache-2.0) | TensorRT / ONNX |
| `rfdetr-medium` | `rfdetr` 1.9.0 | ONNX → trtexec |
| `paddleocr` | `paddleocr` 3.x | **Python** (see below) |
| `kosmos-2.5` | `microsoft/kosmos-2.5` | **Python** (see below) |

---

## Per-GPU scoping

A model appears only in the GPU YAMLs where it fits. This is scoping, not
removal — the mechanism is the repo's existing per-GPU config files.

| Model | 5090 (32 GB) | PRO 6000 (96 GB) | H200 (141 GB) |
|---|---|---|---|
| `qwen2.5-7b` | ✅ | ✅ | ✅ |
| `qwen2.5-14b` | ✅ AWQ | ✅ | ✅ |
| `qwen2.5-32b` | — | ✅ | ✅ |
| `qwen2.5-72b` | — | ✅ AWQ | ✅ AWQ |
| `gemma2-9b` | ✅ | ✅ | ✅ |
| `llama3.1-8b` | ✅ | ✅ | ✅ |
| `mistral-7b` | ✅ | ✅ | ✅ |
| `qwen2.5-vl-7b` | ✅ AWQ | ✅ | ✅ |
| `qwen2.5-vl-72b` | — | ✅ AWQ | ✅ AWQ |
| all CV models | ✅ | ✅ | ✅ |

**VRAM arithmetic.** AWQ ≈ 0.3× BF16. On a 32 GB card, budget ~25 GB after
driver overhead and a CV tenant, and cap `gpu_memory_utilization` at 0.70–0.75 or
the two processes fight. 32B at AWQ (~22 GB) leaves no CV headroom; 72B at AWQ
(~45 GB) does not fit at all. On H200, 72B BF16 (145 GB) still exceeds 141 GB —
it needs AWQ or ≥2 GPUs.

---

## Broken picks — kept, but they fail safely

### `gemma-vlm-32b` → `google/paligemma2-28b-pt-896`

**This is the customer's own entry**, straight from their config. Three problems:

1. It is **28B, not 32B** (Gemma 2 27B + SigLIP-400M).
2. It is a `-pt-` **base checkpoint** — Google documents it as "recommended to
   use after fine tuning."
3. It is **image-only**. vLLM's `paligemma.py` declares
   `get_supported_mm_limits() -> {"image": 1}` and raises
   `ValueError("Only image modality is supported")`.

**It cannot serve the video dimension it was selected for.**

**Handling:** kept in config, serves image rounds normally. Video rounds get an
`unsupported_backends` entry and skip cleanly with exit code 2 and the reason
printed — no mid-sweep crash. Tell the customer; the natural replacement inside
their own list is `qwen2.5-vl-7b`, which does video and has official AWQ.

### `kosmos-2.5`

No vLLM or SGLang implementation — absent from vLLM's model registry, and
`kosmos*.py` 404s in both trees. The "vLLM supported" note on its HF page is the
generic deploy widget, not real support.

**Handling:** served via **Triton Python backend**, alongside the CV models. YAML
comment reads: *"vLLM/SGLang support not available — served via Triton Python
backend."*

### `paddleocr`

PaddleOCR's TensorRT path pins **TensorRT 8.6.1.6 + CUDA 11.8**, which cannot
coexist with the stack Triton 26.07 ships (CUDA 13.3, TensorRT 11.0). This is
PaddleOCR's own requirement, not anything specific to the PRO 6000.

**Handling:** served via **Triton Python backend** — plain Paddle inference, no
TensorRT. Runs unoptimised rather than not at all. PP-OCRv6 is the current branch
if the customer wants to move; not imposed.

### `rfdetr-base`

`RFDETRBase` was deprecated 2025-07-23. **`RFDETRMedium`** is comparable in size,
so the swap preserves intent. Flag it to the customer.

### `qwen2.5-27b` — the one genuine correction

**The HF repo does not exist.** The Qwen2.5 ladder is 0.5/1.5/3/7/14/**32**/72B —
there is no 27B. (27B is a Gemma 2 size.) Since there is nothing to run *or*
skip, config uses `qwen2.5-32b` with a comment recording the original entry.

---

## Quantization — "Q4_0" is not a vLLM format

The design specifies `Q4_0`, which is a **llama.cpp GGUF** format. vLLM cannot
load it: GGUF support moved out-of-tree to `vllm-gguf-plugin` and is documented
as "highly experimental and under-optimized."

vLLM does support: `awq`, `gptq`, `gptq_marlin`, `awq_marlin`,
`compressed-tensors`, `bitsandbytes`, `fp8`, `modelopt_fp4` (NVFP4),
`nvfp4_per_token`, `mxfp4`, `torchao`, `quark`. SGLang supports the same set plus
in-tree `gguf`.

**Decision:** the quantization *dimension* is dropped. Instead, pick the
best-fitting format per model per GPU — AWQ where an official checkpoint exists,
otherwise BF16 or FP8.

---

## Blackwell SM_120 notes

H200 (SM_90) is clean. SM_120 — both the 5090 and PRO 6000 — is rougher, and
vLLM's own quantization hardware matrix still has no Blackwell column:

- NVFP4 MoE backend selection historically checked only SM_90/SM_10x and hard-failed SM_120.
- NVFP4 can silently fall back to Marlin W4A16.
- DeepGEMM must not be enabled (`arch_major 12` unsupported).
- SGLang has blockwise-FP8 gaps and auto-selects the SM_100-only `trtllm_mha`
  backend on the 5090.

This repo's `docs/findings/knowledge.yaml` corroborates independently: the Triton
fused-MoE kernel wants 147 KB shared memory against SM_120's ~100 KB cap, giving
`OutOfResources` at JIT.

---

## Newer alternatives — raised, not applied

For the record, and for whenever the customer wants to move:

| Their pick | Current equivalent |
|---|---|
| Qwen2.5 7B/14B/32B/72B | `Qwen/Qwen3.5-{4B,9B,27B}` — natively multimodal, and **~4.5× smaller KV cache**, which matters a lot under co-residency |
| Qwen2.5-VL-7B | `Qwen/Qwen3-VL-{2B,4B,8B}-Instruct-FP8` — official FP8 at every size |
| Gemma-2-9B | `google/gemma-4-12B-it` |
| Mistral-7B-v0.3 | `mistralai/Ministral-3-8B-Instruct-2512` |
| YOLOv8 | YOLO26 — NMS-free; but v8 is **faster on GPU at nano scale** (1.47 vs 1.7 ms on T4) and still fully supported in ultralytics 8.4.113 |

**Not adopted.** These are for the customer to choose, using this framework.
