# Model catalogue

Curated list of starting models. The benchmark harness must be able to
target each row at least once before we draw conclusions.

## Qwen3-VL on Hugging Face (`Qwen/...`)

Captured 2026-05-04 from the official Qwen3-VL collection.

### Dense models

| Size | HF id (Instruct) | HF id (Thinking) | FP8 variants | GGUF |
| --- | --- | --- | --- | --- |
| 2B | `Qwen/Qwen3-VL-2B-Instruct` | `Qwen/Qwen3-VL-2B-Thinking` | `…-FP8` for both | yes |
| 4B | `Qwen/Qwen3-VL-4B-Instruct` | `Qwen/Qwen3-VL-4B-Thinking` | `…-FP8` for both | yes |
| 8B | `Qwen/Qwen3-VL-8B-Instruct` | `Qwen/Qwen3-VL-8B-Thinking` | `…-FP8` for both | yes |
| 32B | `Qwen/Qwen3-VL-32B-Instruct` | `Qwen/Qwen3-VL-32B-Thinking` | `…-FP8` for both | yes |

### MoE models

| Size | HF id (Instruct) | HF id (Thinking) | FP8 variants | GGUF |
| --- | --- | --- | --- | --- |
| 30B-A3B | `Qwen/Qwen3-VL-30B-A3B-Instruct` | `Qwen/Qwen3-VL-30B-A3B-Thinking` | `…-FP8` for both | yes |
| 235B-A22B | `Qwen/Qwen3-VL-235B-A22B-Instruct` | `Qwen/Qwen3-VL-235B-A22B-Thinking` | `…-FP8` for both | yes |

### POC selection

- **Primary:** `Qwen3-VL-4B-Instruct` and `Qwen3-VL-8B-Instruct` for
  consumer-card targeting (RTX 5090 / RTX PRO 6000).
- **Reasoning quality reference:** `Qwen3-VL-32B-Instruct` (and `-Thinking`
  for command-planning quality experiments).
- **Capacity reference:** `Qwen3-VL-30B-A3B-Instruct` (MoE — interesting
  for the bandwidth-bound thesis: only a fraction of params active per
  token).
- **FP8 path:** the `…-FP8` HF checkpoints feed directly into vLLM/SGLang
  FP8 paths; for TRT-LLM we instead calibrate via ModelOpt.

## NVIDIA NIM (Qwen VL family)

The NIM catalogue rotates frequently and the quick fetch we ran for this
scaffold did not enumerate the Qwen3-VL endpoints cleanly. **Re-verify at
run time** with:

```bash
curl -s -H "Authorization: Bearer $NIM_API_KEY" \
     https://integrate.api.nvidia.com/v1/models | jq '.data[].id' | grep -i qwen
```

Endpoints to look for (subject to availability):

- `qwen/qwen2.5-vl-7b-instruct` — proven-stable Qwen2.5-VL VLM endpoint
  (good fallback if Qwen3-VL NIM is not yet published).
- `qwen/qwen3-vl-*-instruct` — Qwen3-VL endpoints as they roll out.

For a self-hosted NIM container, consult
`build.nvidia.com/qwen` and the per-model "Deploy" tab for the container
image, the required driver version, and the GPU compatibility matrix.

## Why these picks

- 4B/8B are the dense models with the best **latency/quality tradeoff** on
  a single consumer Blackwell card.
- 30B-A3B is the **bandwidth-thesis stress test**: small active-param
  count per token, so memory bandwidth and KV-cache locality dominate.
- 32B dense is the **quality ceiling** for what fits in 96 GB on a single
  RTX PRO 6000 without aggressive quant.
- Larger 235B-A22B is **out of scope for the consumer baseline** but worth
  one H200 reference run.

## Constraints we expect to hit

- vLLM Qwen3-VL support landed in 0.6+ — pin accordingly.
- SGLang Qwen-VL support is multimodal-image only; video frames must be
  decoded host-side first.
- TRT-LLM Qwen-VL needs the vision tower exported separately (TRT) and the
  language tower built as a TRT-LLM engine. ModelOpt provides the FP8
  calibration recipe.
