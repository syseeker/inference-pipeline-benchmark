# Model catalogue

The benchmark targets three model families per GPU so the cross-backend
comparison covers a VLM headline, a dense-text TRT-engine win-case, and an
NVIDIA-tuned multimodal MoE.

## Families

| Family | Why in the benchmark |
| --- | --- |
| **Qwen3-VL** (`Qwen/Qwen3-VL-*`) | Headline VLM. The model the benchmark was originally scoped against. |
| **Qwen3.5 / Qwen3.6** (`Qwen/Qwen3.5-*`, `Qwen/Qwen3.6-*`) | Dense text — best chance of a clear TRT-LLM win at high batch on Hopper/Blackwell. |
| **Nemotron-3-Nano-Omni** (`nvidia/Nemotron-3-Nano-Omni-*`) | NVIDIA-tuned multimodal MoE — NV silicon + NV runtime should give TRT-LLM its strongest shot at beating vLLM/SGLang on a multimodal workload. |

All three families run through TRT-LLM's PyTorch backend via
`trtllm-serve --backend pytorch` (HTTP, OpenAI-shape — same client
surface as vLLM and SGLang).

The point of running all three is to map *when* TRT-LLM wins versus when
it merely matches vLLM/SGLang. Picks are tuned per GPU so each comparison
is apples-to-apples (same model fits all three runtimes with KV headroom).

## Per-GPU picks

| GPU (VRAM) | Qwen3-VL | Qwen3.5 / 3.6 | Nemotron Nano Omni |
| --- | --- | --- | --- |
| **RTX 5090** (32 GB GDDR7, Blackwell) | `Qwen/Qwen3-VL-8B-Instruct-FP8` (~9 GB) | `Qwen/Qwen3.5-9B` FP8 (~5 GB) | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` (~21 GB) |
| **RTX PRO 6000 Blackwell Server** (96 GB GDDR7) | `Qwen/Qwen3-VL-32B-Instruct-FP8` (~33 GB) | `Qwen/Qwen3.6-27B-FP8` (~27 GB) | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8` (~33 GB) |
| **H200** (141 GB HBM3e, Hopper) | `Qwen/Qwen3-VL-32B-Instruct` BF16 (~66 GB) | `Qwen/Qwen3.6-35B-A3B-FP8` (~35 GB) + `Qwen/Qwen3.6-27B-FP8` (~27 GB) | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` (~60 GB) |

### Why these variants

- **5090: small + tight.** 22 GB usable for weights → only the 8B-FP8 Qwen3-VL fits with real KV+vision headroom; Qwen3.5-9B is the smallest dense Qwen3.x with a TRT-engine shot; Nemotron Omni only fits at NVFP4 (Blackwell-native quant).
- **PRO 6000: 96 GB room to scale.** 32B-FP8 Qwen3-VL fits with comfortable KV/ctx room. Qwen3.6-27B-FP8 is dense + Blackwell FP8 + stable shapes — the TRT-engine path's home turf. Nemotron Omni at FP8 is the sweet spot on Blackwell.
- **H200: BF16 + flagship MoE.** 141 GB HBM3e is the only place we can afford the BF16 accuracy baseline for 32B-class models. Qwen3.6-35B-A3B-FP8 on Hopper FP8 at high batch is *the* TRT-engine flagship case. Nemotron Omni at BF16 anchors accuracy for any quant comparison.

### Quants and why

| Quant | Where used | Notes |
| --- | --- | --- |
| BF16 | accuracy baseline; only on GPUs that can afford it (PRO 6000 32B, H200 32B+) | The reference for any quant-accuracy delta. Slow but correct. |
| FP8 | the production-style default everywhere | E4M3 weights + FP8 activations; supported by all three serving frameworks on Hopper / Blackwell tensor cores. ~1 B/param. |
| NVFP4 | RTX 5090 only | Blackwell tensor-core-native 4-bit format (~0.6 B/param). The only quant that fits Nemotron-3-Nano-Omni-30B in 32 GB with KV. Hopper does not support NVFP4 natively — do not enable on H200. |

W8A8 (INT8) and INT4/AWQ are intentionally **out of scope** for headline
runs. They're not apples-to-apples across the three serving frameworks
and rarely show up in production VLM stacks.

## Hub IDs and source

All checkpoints live on Hugging Face Hub. vLLM, SGLang, and TRT-LLM
(PyTorch backend via `trtllm-serve`) all pull on first launch into
`$HF_HOME` (default `~/.cache/huggingface/hub/`).

| HF id | Variants | Notes |
| --- | --- | --- |
| `Qwen/Qwen3-VL-8B-Instruct-FP8` | FP8 only | Pre-quantised by Qwen team. Direct load on all three runtimes. |
| `Qwen/Qwen3-VL-8B-Instruct` | BF16 | Reference precision. |
| `Qwen/Qwen3-VL-32B-Instruct-FP8` | FP8 only | Pre-quantised. |
| `Qwen/Qwen3-VL-32B-Instruct` | BF16 | Reference precision; H200 baseline. |
| `Qwen/Qwen3.5-9B` | BF16 (FP8 via runtime quant or `…-FP8` if Qwen ships one) | Smallest dense Qwen3.5 on the headline list. |
| `Qwen/Qwen3.6-27B-FP8` | FP8 only | Pre-quantised; dense; April 2026 release. |
| `Qwen/Qwen3.6-35B-A3B-FP8` | FP8 only | Pre-quantised; MoE (35B total / 3B active). |
| `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` | BF16 reference checkpoint. NVIDIA also publishes calibrated `…-FP8` and `…-NVFP4` siblings — load those directly on PRO 6000 / 5090 respectively (no runtime quant needed). | NV multimodal MoE; 30B total / 3B active; 131K context. |

## Out of scope

- **`Qwen/Qwen3-VL-2B`/`-4B`** — too small to stress any of the three GPUs in this benchmark.
- **`Qwen/Qwen3-VL-30B-A3B-Instruct`** — overlaps with Nemotron Omni's role (multimodal MoE) but without NV-tuned kernels; pick Nemotron for cleaner TRT-LLM signal.
- **`Qwen/Qwen3-VL-235B-A22B`** — doesn't fit any of the three target GPUs at any precision. B300 / multi-GPU only.
- **Qwen2-VL / Qwen2.5-VL** — superseded by Qwen3-VL.
- **Nemotron Super (120B-A12B), Nemotron Ultra (~500B-A50B)** — single-GPU sizing target is Nano Omni; Super/Ultra are multi-GPU.

## Constraints to verify before each run

- **vLLM Qwen3-VL deepstack + chunked prefill** crashes when prefix caching is on. Don't pass `--enable-prefix-caching` to `vllm serve`. See [SMOKE_TESTS.md](../SMOKE_TESTS.md).
- **SGLang Qwen-VL** is multimodal-image only; video frames must be decoded host-side first.
- **TRT-LLM multimodal** is incompatible with `kv_cache_reuse` — the YAML override `kv_cache_config.enable_block_reuse: false` is required when launching `trtllm-serve` for any of the picks above.
- **NVFP4** is Blackwell-only. Do not enable on H200 (Hopper).
