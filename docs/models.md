# Model catalogue

The benchmark targets four model families per GPU so the cross-backend
comparison covers a VLM headline, a dense-text TRT-engine win-case, an
NVIDIA-tuned multimodal MoE, and a cross-vendor video-capable VLM.

## Families

| Family | Why in the benchmark |
| --- | --- |
| **Qwen3-VL** (`Qwen/Qwen3-VL-*`) | Headline VLM. The model the benchmark was originally scoped against. |
| **Qwen3.5 / Qwen3.6** (`Qwen/Qwen3.5-*`, `Qwen/Qwen3.6-*`) | *Hypothesis:* dense text — best chance of a clear TRT-LLM win at high batch on Hopper/Blackwell. *Measured (2026-05):* TRT-LLM 1.2.1 cannot load any qwen3_5-arch checkpoint on any GPU — see [Measured reality](#measured-reality-2026-05-rtx-pro-6000-only). |
| **Nemotron-3-Nano-Omni** (`nvidia/Nemotron-3-Nano-Omni-*`) | *Hypothesis:* NV silicon + NV runtime → TRT-LLM's strongest shot at beating vLLM/SGLang on a multimodal workload. *Measured (2026-05):* TRT-LLM 1.2.1 doesn't register the `NemotronH` arch and refuses to load on every GPU. |
| **Gemma 4** (`google/gemma-4-*`, FP8 via `RedHatAI/gemma-4-*-FP8-block`) | Cross-vendor twin to Qwen3-VL: Google's flagship open VLM. Dense 31B variant brings the first video-capable modality into the matrix (image+video+text, 256K ctx). *Hypothesis (PRO 6000):* a fresh April-2026 day-0 release — vLLM/SGLang load it; TRT-LLM 1.2.1 expected to fail on arch registry (same shape as qwen3_5 / NemotronH) until bumped. |

All three families run through TRT-LLM's PyTorch backend via
`trtllm-serve --backend pytorch` (HTTP, OpenAI-shape — same client
surface as vLLM and SGLang).

The point of running all three is to map *when* TRT-LLM wins versus when
it merely matches vLLM/SGLang. Picks are tuned per GPU so each comparison
is apples-to-apples (same model fits all three runtimes with KV headroom).

## Per-GPU picks

| GPU (VRAM) | Qwen3-VL | Qwen3.5 / 3.6 | Nemotron Nano Omni | Gemma 4 |
| --- | --- | --- | --- | --- |
| **RTX 5090** (32 GB GDDR7, Blackwell) | `Qwen/Qwen3-VL-8B-Instruct-FP8` (~9 GB) | `Qwen/Qwen3.5-9B` FP8 (~5 GB) | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` (~21 GB) | TBD — likely `RedHatAI/gemma-4-31B-it-NVFP4` (~21 GB) on follow-up |
| **RTX PRO 6000 Blackwell Server** (96 GB GDDR7) | `Qwen/Qwen3-VL-32B-Instruct-FP8` (~33 GB) | `Qwen/Qwen3.6-27B-FP8` (~27 GB) | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8` (~33 GB) | `RedHatAI/gemma-4-31B-it-FP8-block` (~31 GB) |
| **H200** (141 GB HBM3e, Hopper) | `Qwen/Qwen3-VL-32B-Instruct` BF16 (~66 GB) | `Qwen/Qwen3.6-35B-A3B-FP8` (~35 GB) + `Qwen/Qwen3.6-27B-FP8` (~27 GB) | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` (~60 GB) | TBD — `google/gemma-4-31B-it` BF16 (~61 GB) for accuracy baseline |

### Why these variants (hypothesis)

The picks were chosen so each GPU has *at least one* model where TRT-LLM's
kernel and runtime advantages should plausibly show — i.e. so a TRT-LLM win
or loss is attributable to the framework, not to a wrong-sized model.

- **5090: small + tight.** 22 GB usable for weights → only the 8B-FP8 Qwen3-VL fits with real KV+vision headroom; Qwen3.5-9B is the smallest dense Qwen3.x with a TRT-engine shot; Nemotron Omni only fits at NVFP4 (Blackwell-native quant).
- **PRO 6000: 96 GB room to scale.** 32B-FP8 Qwen3-VL fits with comfortable KV/ctx room. Qwen3.6-27B-FP8 is dense + Blackwell FP8 + stable shapes — the TRT-engine path's home turf. Nemotron Omni at FP8 is the sweet spot on Blackwell. Gemma 4 31B-FP8 is the cross-vendor twin to Qwen3-VL-32B-FP8 — different lineage, video-capable tower, ~31 GB FP8 → identical KV budget on this GPU.
- **H200: BF16 + flagship MoE.** 141 GB HBM3e is the only place we can afford the BF16 accuracy baseline for 32B-class models. Qwen3.6-35B-A3B-FP8 on Hopper FP8 at high batch is *the* TRT-engine flagship case. Nemotron Omni at BF16 anchors accuracy for any quant comparison.

### Measured reality (2026-05, RTX PRO 6000 only)

The first PRO 6000 sweep contradicted the hypothesis on every model
where TRT-LLM was supposed to shine. The headline finding: **TRT-LLM
1.2.1's pytorch backend can load exactly *one* of the four PRO 6000
picks — and it loses badly on that one too.** Detailed write-ups live
in [docs/findings/](findings/); this table is the index.

| Model on PRO 6000 | Hypothesis (this doc) | Measured reality | Status |
|---|---|---|---|
| **`Qwen/Qwen3-VL-32B-Instruct-FP8`** | Headline VLM, dense, fits with KV room. Apples-to-apples comparison ground. | TRT-LLM loads but **TTFT 42 s, validity 0%**. Lazy CUDA-graph capture penalises every cold shape; xgrammar (`response_format`) crashes startup on the multimodal wrapper. **vLLM serves the same model in 1922 ms E2E.** | ⚠ TRT-LLM works, but loses on the very model it's expected to dominate. [findings](findings/trtllm-1.2.1-qwen3-vl-32b-fp8.md) |
| **`Qwen/Qwen3-VL-30B-A3B-Instruct-FP8`** (MoE) | "Bandwidth-thesis stress test" — fits at FP8, MoE active params ≈ small dense. | TRT-LLM **cannot start**: default `MoeConfig.backend = CUTLASS` dispatches into a DeepGEMM JIT path that hard-checks SM_90; PRO 6000 is SM_120 (`fused_moe_cutlass.py:441 → DeepGEMM`). Pinned in [rtx_pro6000.yaml](../benchmarks/configs/rtx_pro6000.yaml) under `models.qwen3-vl-30b-a3b-fp8.unsupported_backends`. | ✗ TRT-LLM blocked. |
| **`Qwen/Qwen3.6-27B-FP8`** | "TRT-engine path's home turf — dense + Blackwell FP8 + stable shapes." | TRT-LLM **cannot start**: model_type `qwen3_5` (hybrid attention + Mamba/SSM) isn't in TRT-LLM's bundled transformers registry — `KeyError: 'qwen3_5'`. **vLLM is the fastest backend in the matrix on this model** (44.8 tok/s decode, 22 ms ITL). | ✗ TRT-LLM blocked; the predicted home turf went to vLLM. [findings](findings/qwen3.6-on-trtllm.md) |
| **`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8`** | "NV silicon + NV runtime → TRT-LLM's strongest shot at beating vLLM/SGLang on a multimodal workload." | TRT-LLM **cannot start**: PyTorch backend rejects `NemotronH_Nano_Omni_Reasoning_V3` arch. SGLang **also** can't start (Triton fused-MoE asks 147 KB shmem; SM_120 has ~100 KB). Only vLLM works. | ✗ TRT-LLM blocked; SGLang also blocked. [findings](findings/nemotron-omni-on-trtllm.md) |
| **`RedHatAI/gemma-4-31B-it-FP8-block`** | Cross-vendor VLM headline (Google) — dense, FP8, video-capable, fits at ~31 GB with KV room. Expected: vLLM/SGLang load it day-0; TRT-LLM 1.2.1 fails on arch registry until bumped. | *Pending first sweep* (added 2026-05). TRT-LLM pinned out via `unsupported_backends` in [rtx_pro6000.yaml](../benchmarks/configs/rtx_pro6000.yaml); vLLM and SGLang sweep results to be filed here. | ⏳ Pending. |

**The pattern is integration maturity, not silicon.** TRT-LLM 1.2.1's
pytorch backend ships against an older `transformers` and a Hopper-tuned
fused-MoE path; it lags vLLM/SGLang by a release cycle on new arches
(qwen3_5, NemotronH) and on Blackwell-tensor-core kernels. Same model
with `trtllm-build`-compiled engines on H200 (where DeepGEMM is happy
and qwen3_5 is older) is expected to behave differently — but on PRO
6000 + day-0 multimodal checkpoints, today, vLLM wins on every model
both backends can load.

5090 and H200 sweeps haven't been run yet, but the **arch-not-registered**
blockers (`qwen3_5`, `NemotronH`) are GPU-independent and will reproduce
on both. The **SM_90-only DeepGEMM** blocker is SM_120-specific and
won't bite H200; status on 5090 (also SM_120) is unverified for the
NVFP4 path. Validate before claiming TRT-LLM wins on either GPU.

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
| `google/gemma-4-31B-it` | BF16 reference. Google ships BF16 only — FP8 / NVFP4 come from RedHatAI (`RedHatAI/gemma-4-31B-it-FP8-block`, `…-FP8-Dynamic`, `…-NVFP4`). | Dense 30.7 B VLM; image + video + text; 256K context; Apache 2.0. April 2026 release. |

## NitroGen — diffusion policy (separate model class)

A second kind of model also runs in this harness: **NitroGen**, a 500M
diffusion-policy that reads a frame + `game_id` and emits gamepad actions. It is
not a VLM and not served by vLLM/SGLang/TRT-LLM — full background in
[nitrogen.md](nitrogen.md). The "models" below are the **same checkpoint** run at
different precision / denoise-step settings; the execution backend (eager /
compile / TensorRT / ONNX) is the *variant*, not the model.

| Policy "model" | Precision | Steps | Where |
| --- | --- | --- | --- |
| `nitrogen-500m-bf16` | BF16 | 16 | all GPUs — accuracy-vs-gold reference |
| `nitrogen-500m-fp8` | FP8 | 16 | all GPUs |
| `nitrogen-500m-fp8-4step` | FP8 | 4 | all GPUs — latency floor |
| `nitrogen-500m-nvfp4` | NVFP4 | 16 | **Blackwell only** (RTX PRO 6000 / 5090; not H200) |

At 500M params NitroGen fits trivially on every target GPU, so this is a pure
latency / throughput / accuracy study, not a fit study. Checkpoint:
`nvidia/NitroGen` (`hf download nvidia/NitroGen ng.pt`).

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
