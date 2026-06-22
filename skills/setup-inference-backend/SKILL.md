---
name: setup-inference-backend
description: |
  Idempotent installer for the per-backend Python venv + dependencies
  (.venv-vllm / .venv-sglang / .venv-trtllm / .venv-nitrogen / .venv-nim).
  Knows about the install gaps (torchvision for nitrogen, transformers<5
  pin, NVIDIA-index wheels for TRT-LLM, JS runtime for yt-dlp) and the
  per-GPU `unsupported_backends` matrix in the GPU YAMLs.
---

# setup-inference-backend

## When to invoke

- "install vLLM on this machine"
- "set up the NitroGen serving stack"
- "what backends can this GPU actually run"
- "the venv is broken / rebuild it"

## Recipe

```bash
bench setup --backend nitrogen --json     # or vllm / sglang / trtllm / nim
```

Idempotent: skipped if `.venv-<backend>` already exists. Pass `--force`
to recreate.

## Per-backend extras (what's installed)

| Backend | venv | Extras (`[name]`) | Manual follow-up |
|---|---|---|---|
| `vllm` | `.venv-vllm` | `vllm,dev` | none |
| `sglang` | `.venv-sglang` | `sglang,dev` | none |
| `trtllm` | `.venv-trtllm` | `dev` | `pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com` (NVIDIA wheel, not on PyPI) |
| `nitrogen` | `.venv-nitrogen` | `nitrogen,dataset,dev` | `pip install -e ../NitroGen` then `hf download nvidia/NitroGen ng.pt` |
| `nim` | `.venv-nim` | `nim,dev` | `export NIM_API_KEY=...` |
| `nitrogen-quant` | `.venv-nitrogen` | `nitrogen,nitrogen-quant,dataset,dev` | Same NitroGen prep AS `nitrogen` (`pip install -e ../NitroGen` + `hf download nvidia/NitroGen ng.pt`). Required only when the sweep includes FP8/NVFP4 rounds. Customers do NOT recalibrate — pre-built artifacts download automatically from `syseeker-at-nv/nitrogen-quant` on first FP8/NVFP4 round. |

`bench setup` emits the follow-up step in `data.next_action` of its
JSON output. Surface it to the user; don't silently skip.

## Pre-flight checks

1. **GPU and driver** — `bench probe --json`. If `driver` is `unknown`,
   the host has no NVIDIA GPU or the driver is unloaded; refuse setup
   for everything except `nitrogen` CPU-mode dev work.
2. **HF token** — `hf whoami`. Required for gated models
   (Qwen3-VL, Nemotron Omni). The skill should detect 401s in `setup`
   downstream installs and tell the user to `huggingface-cli login`.
3. **JS runtime** — only matters for the `dataset` extra in
   `nitrogen`. If real-frame extraction is the goal and neither `node`
   nor `deno` is on PATH, surface the issue here (prepare-nitrogen-dataset
   will hit it later otherwise).

## Per-GPU unsupported backends

Each GPU yaml has a `models.<id>.unsupported_backends:` field — read it
before running. Example today on `rtx_pro6000.yaml`:

| Model | Backend | Reason |
|---|---|---|
| `qwen3-vl-30b-a3b-fp8` | `trtllm` | TRT-LLM 1.2.1 fused-MoE checks SM_90; PRO 6000 is SM_120 |
| `qwen3.6-27b-fp8` | `trtllm` | transformers registry missing `qwen3_5` arch |
| `gemma-4-31b-it-fp8` | `trtllm` | expected `gemma4` arch missing pre-bump |
| `nemotron-omni-fp8` | `sglang` | triton fused-MoE wants 147 KB shmem; SM_120 ≈ 100 KB |
| `qwen3-vl-30b-a3b-fp8` | `trtllm` | (Qwen3-VL Vision multimodal + SM_120 fused-MoE) |
| Various nitrogen-fp8/-nvfp4 | `nitrogen-tensorrt`, `nitrogen-onnx` | quant + export not implemented; **PR #5** |

If the user asks to run something on the wrong backend, **don't try** —
quote the reason from the yaml and propose the supported alternative.

## Failure recovery

| Symptom | Cause | Fix |
|---|---|---|
| `pip` 404 on `tensorrt-llm` | wrong index | re-run with `--extra-index-url https://pypi.nvidia.com` |
| `ImportError: torchvision` on `serve_nitrogen.py` | PR #1 already added torchvision to the extras; if you hit this, the venv is stale → `bench setup --backend nitrogen --force` |
| `vision_model` AttributeError loading siglip | `transformers>=5` slipped in. The `[nitrogen]` extra pins `<5`. If a follow-up pip install bumped it, reinstall the extra: `pip install -e ".[nitrogen]"` |
| `Address already in use 5555` at server bind | Old orphan listener. PR #1 spread to per-engine ports 5560-5564; if you're on a stale yaml, sync to latest |

## Pinned references

- Install detail per backend: [INFERENCE_BACKENDS.md](../../INFERENCE_BACKENDS.md)
- Smoke tests per backend: [SMOKE_TESTS.md](../../SMOKE_TESTS.md)
- GPU YAMLs: [benchmarks/configs/](../../benchmarks/configs/)
