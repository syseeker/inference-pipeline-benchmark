# Nemotron-3-Nano-Omni on TRT-LLM: cookbook-only path on 1.3.0rc13

**Date:** 2026-05-09
**GPU(s):** RTX PRO 6000 Blackwell (96 GB), RTX 5090 (32 GB), H200 (141 GB)
**Backend that fails today:** `trtllm-serve --backend pytorch`, TRT-LLM 1.2.1
**Backends that succeed today:** vLLM (all three GPUs), SGLang (H200 only — sglang fails on SM_120 due to a separate fused-MoE shmem bug)
**Affected models:**
- `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8` (RTX PRO 6000)
- `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` (RTX 5090)
- `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` (H200)
**Failed run id (no aggregate JSON written):** see `benchmarks/results/rtx_pro6000/server-logs/trtllm.log` (NemotronH arch error mirrors the qwen3_5 failure documented in [qwen3.6-on-trtllm.md](qwen3.6-on-trtllm.md))
**Cookbook reference:** <https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Nano-Omni/trtllm_cookbook.ipynb>

## TL;DR

`trtllm-serve` on TRT-LLM 1.2.1 **cannot load** Nemotron-3-Nano-Omni. The
`NemotronH_Nano_Omni_Reasoning_V3` arch isn't registered in the bundled
PyTorch model registry, so executor init raises
`ValueError: Unknown architecture for AutoModelForCausalLM` before any
GPU work. Same class of integration gap as `qwen3_5`.

NVIDIA's HF model card directs users to the **NeMo cookbook**, which
expects:

1. **TRT-LLM 1.3.0rc13** (container `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13`).
2. **CUDA 13.0+**.
3. A model-specific `extra_llm_api_options` YAML with
   `mamba_ssm_cache_dtype: float32` (the model is a hybrid attn + Mamba/SSM,
   like qwen3_5; the SSM cache dtype is mandatory).
4. Three new `trtllm-serve` flags not used elsewhere in our matrix:
   `--reasoning_parser nano-v3`, `--tool_parser qwen3_coder`,
   `--trust_remote_code`.
5. `PYTORCH_ALLOC_CONF=expandable_segments:True` in the launch env.

Net effect: lighting up trtllm for this model is **not a config-tweak
patch** — it's a TRT-LLM upgrade plus a per-model launcher branch.
The harness today pins it away from trtllm in all three GPU sweeps.

## Failure mode (current 1.2.1 install)

`tensorrt_llm.__version__ == "1.2.1"` (from `.venv-trtllm`). On all
three GPUs the failure is identical and GPU-independent:

```
ValueError: Unknown architecture for AutoModelForCausalLM:
NemotronH_Nano_Omni_Reasoning_V3
```

Crash site is `tensorrt_llm/_torch/pyexecutor/proxy.py` during executor
init — before any kernel runs. A secondary
`AttributeError: 'PyTorchModelEngine' object has no attribute
'cuda_graph_runner'` shows up on `__del__` and is teardown noise.

This is structurally the same as the `KeyError: 'qwen3_5'` issue
documented in [qwen3.6-on-trtllm.md](qwen3.6-on-trtllm.md): the
PyTorch backend's model-loader registry doesn't have an entry for the
new arch, and `--trust_remote_code` alone can't bridge it on 1.2.1.

## What the cookbook actually mandates

Quoting the cookbook's "Load the FP8 quantized version" cell verbatim:

```shell
PYTORCH_ALLOC_CONF=expandable_segments:True \
trtllm-serve serve "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8" \
--host 0.0.0.0 \
--port 8000 \
--trust_remote_code \
--reasoning_parser nano-v3 \
--tool_parser qwen3_coder \
--extra_llm_api_options nano_v3.yaml
```

And the YAML it points to:

```yaml
kv_cache_config:
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.80
  mamba_ssm_cache_dtype: float32
max_batch_size: 128
```

Container the cookbook recommends:
`nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13`.

For DGX Spark (memory-tight) the cookbook adds
`moe_config: backend: CUTLASS`,
`cuda_graph_config: { enable_padding: true, max_batch_size: 1 }`,
and `max_batch_size: 1`. For NVFP4 on B200 specifically, it suggests
`moe_config: backend: TRTLLM` for better perf. RTX PRO 6000 / RTX 5090
are SM_120 Blackwell — neither is B200 nor DGX Spark; default backend
is the right starting point and we'd tune from there.

NVFP4 path requires Blackwell, which RTX PRO 6000 (FP8 variant we
picked) and RTX 5090 (NVFP4 variant) both satisfy. H200 is Hopper and
must use BF16.

The audio-from-video toggles (`PyAV` + `TRTLLM_ENABLE_PYAV=1` +
`--media_io_kwargs '{"video": {"extract_audio": true}}'`) are **not
relevant** to this benchmark — our pipeline is image+text. Skip them;
they also carry a "first request must be video" caveat and a PyAV
redistribution restriction.

## Why this is more than a flag flip

Three reasons it isn't a simple `backend_args.trtllm` line in the YAML:

1. **TRT-LLM upgrade.** 1.2.1 → 1.3.0rc13 is a major bump. Two install
   vehicles, both with cost:
   - **A. Docker container** (cookbook-native). Adds a docker
     dependency to a harness that today is venv-only ([scripts/run_all_scenarios.sh:280](../../scripts/run_all_scenarios.sh#L280)
     calls `trtllm-serve` directly inside `.venv-trtllm`). Doesn't
     disturb the working 1.2.1 Qwen3-VL setup.
   - **B. New venv with the 1.3.0rc13 wheel.** Cleaner from a harness
     perspective, but RC wheels are fragile, the cookbook ships via
     container for a reason, and it forces a CUDA 13 system upgrade
     if the host is on CUDA 12.
   - Recommendation if/when we light this up: **A**, and gate the
     trtllm round through a per-model `runtime:` selector so Qwen3-VL
     stays on the existing venv.
2. **Per-model launcher branch.** The current launcher passes a single
   global `--extra_llm_api_options=benchmarks/configs/trtllm-vlm.yml` for
   all trtllm rounds (see `backends.trtllm.extra_args` in each GPU yaml).
   Nemotron needs its own YAML (`mamba_ssm_cache_dtype` is required
   and meaningless for Qwen3-VL). That means either a model-aware
   YAML emitter in the bash launcher, or a `backend_args.trtllm`
   override on the model entry that *replaces* (not adds to) the
   backend-level extras.
3. **CUDA 13.0+ floor.** `scripts/gpu_probe.sh` reports the host CUDA;
   if any host is on CUDA 12, the container path becomes mandatory
   (driver permitting), regardless of preference.

## Code changes that would be required (if/when we do this)

Documented for memory; **not implemented**.

- **3 GPU yamls** — drop the `nemotron-omni-*` `backends:` skip pin in
  the sweeps and add `backend_args.trtllm: ["--reasoning_parser=nano-v3",
  "--tool_parser=qwen3_coder", "--trust_remote_code",
  "--extra_llm_api_options=/tmp/trtllm-configs/nano_v3.yaml"]`. Also
  rewrite the long incompatibility comment blocks
  ([rtx_pro6000.yaml:111-128](../../benchmarks/configs/rtx_pro6000.yaml#L111-L128),
  [rtx5090.yaml:106-110](../../benchmarks/configs/rtx5090.yaml#L106-L110),
  [h200.yaml:111-112](../../benchmarks/configs/h200.yaml#L111-L112)) to
  point here instead of declaring the model unsupported.
- **[scripts/run_all_scenarios.sh](../../scripts/run_all_scenarios.sh)** — add
  a docker branch alongside the existing `trtllm)` venv branch
  (line 273), keyed by a new per-model `runtime:` field. Mount
  `~/.cache/huggingface` so the 30B weights don't re-download inside
  the container. Add `docker stop` to the cleanup `pkill` block at
  line 93. Set `PYTORCH_ALLOC_CONF=expandable_segments:True` on the
  trtllm process env.
- **[src/vlm_pipeline/reasoners/trtllm_backend.py](../../src/vlm_pipeline/reasoners/trtllm_backend.py)** —
  *probably no change*. With `--reasoning_parser nano-v3`, the model's
  `<think>...</think>` is split into `message.reasoning_content`; only
  the post-think payload reaches `delta.content` (line 117), which is
  what we already consume. The leading-`{` strip at lines 127-130 still
  works. **Verify with a smoke test**, especially the interaction with
  `response_format={"type":"json_object"}` (line 113) — the cookbook
  doesn't show that combo. Optional follow-up: capture
  `reasoning_tokens` from the final usage chunk and stash on
  `ModelMeta.extras` to track reasoning-trace length for this model.

## Workaround in the benchmark today

The three GPU YAMLs pin nemotron-omni rounds away from trtllm, so the
harness skips it cleanly:

- [rtx_pro6000.yaml](../../benchmarks/configs/rtx_pro6000.yaml) —
  `nemotron-omni-fp8`, `backends: [vllm]` (sglang also fails here on
  SM_120 fused-MoE shmem; trtllm fails on the arch).
- [rtx5090.yaml](../../benchmarks/configs/rtx5090.yaml) —
  `nemotron-omni-nvfp4`, `backends: [vllm]` (same SM_120 sglang issue).
- [h200.yaml](../../benchmarks/configs/h200.yaml) —
  `nemotron-omni-bf16`, `backends: [vllm, sglang]`.

This avoids the same three problems documented in
[qwen3.6-on-trtllm.md](qwen3.6-on-trtllm.md): VRAM-leaking crash mid-startup,
fail-fast halting the whole sweep, and ~10 min of weight-download
churn before the loader fails.

## Honest framing for the report

> Nemotron-3-Nano-Omni is a hybrid attention + Mamba/SSM multimodal MoE
> with a custom HF arch (`NemotronH_Nano_Omni_Reasoning_V3`). vLLM (and
> SGLang on Hopper) load it day-one via `--trust-remote-code`; TRT-LLM
> 1.2.1's PyTorch backend does not register the arch, so executor init
> raises `Unknown architecture` before any GPU work. **NVIDIA's
> documented path is the NeMo cookbook on TRT-LLM 1.3.0rc13** — a
> different binary release with a model-specific YAML
> (`mamba_ssm_cache_dtype: float32`) and three nemotron-only
> `trtllm-serve` flags (`--reasoning_parser nano-v3`,
> `--tool_parser qwen3_coder`, `--trust_remote_code`). This is a
> TRT-LLM integration / packaging gap, not a hardware or kernel issue.
> Lighting it up in the harness requires a TRT-LLM upgrade plus a
> per-model launcher branch, not a flag flip.

## Open questions / next steps

- **Decide install vehicle (A vs B above)** before any code change.
  Container is recommended; needs NGC pull rights and host-disk
  headroom for the multi-GB image plus 30B weights cache.
- **Confirm host CUDA** via `scripts/gpu_probe.sh` on each GPU host. If
  any is on CUDA 12, container path is mandatory (driver permitting).
- **Smoke-test 1.3.0rc13 with Qwen3-VL** once the container is pulled.
  If 1.3 also fixes the `vocab_size_padded` / xgrammar gap noted in
  [trtllm-1.2.1-qwen3-vl-32b-fp8.md](trtllm-1.2.1-qwen3-vl-32b-fp8.md),
  consider migrating Qwen3-VL too — but as a separate decision, not
  bundled with the nemotron rollout.
- **Watch upstream TRT-LLM release notes** for the
  `NemotronH_Nano_Omni_Reasoning_V3` arch landing in a stable release
  (post-1.3.0rc13). When a stable wheel ships, option B becomes more
  attractive and the docker dependency can be dropped.
- **Re-evaluate `response_format={"type":"json_object"}` interaction
  with `--reasoning_parser nano-v3`** before flipping the sweep. If
  the parser strips `<think>` but the JSON schema hint still works, we
  get *better* validity than the current xgrammar gap; if it conflicts,
  drop the hint for nemotron and accept the same lower validity rate
  documented for Qwen3-VL on 1.2.1.
