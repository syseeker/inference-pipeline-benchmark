# Qwen3.5 / 3.6 cannot run on TRT-LLM 1.2.1

**Date:** 2026-05-09
**GPU:** RTX PRO 6000 Blackwell Server Edition (96 GB)
**Backend that fails:** `trtllm-serve --backend pytorch`, TRT-LLM 1.2.1
**Backends that succeed:** vLLM 0.20.1, SGLang 0.5.11
**Affected models:** any `Qwen/Qwen3.5-*`, `Qwen/Qwen3.6-*` checkpoint
**Failed run id (no aggregate JSON written):** see `benchmarks/results/rtx_pro6000/server-logs/trtllm.log`

## TL;DR

`trtllm-serve` on TRT-LLM 1.2.1 **cannot load** Qwen3.5/3.6 checkpoints
at all — startup fails before any request is served. The problem is not
performance: the bundled `transformers` build doesn't register the
`qwen3_5` model_type used by the new hybrid attention + Mamba/SSM
architecture, so config loading raises `KeyError: 'qwen3_5'`.

vLLM and SGLang have their own model implementations that explicitly
handle this arch. On the same hardware vLLM is the *fastest* backend
in the matrix on this model (44.8 tok/s decode, 22 ms ITL p50) — so
TRT-LLM is missing out on what would otherwise be a strong run for it.

| Backend | Loaded? | TTFT p50 | E2E p50 | ITL p50 | tok/s decode |
|---|---|---:|---:|---:|---:|
| vLLM 0.20.1 | ✓ | 537 ms | 1922 ms | 22.3 ms | **44.8** |
| SGLang 0.5.11 | ✓ | 578 ms | 4573 ms | 38.2 ms | 26.2 |
| **TRT-LLM 1.2.1** | ✗ — `KeyError: 'qwen3_5'` | n/a | n/a | n/a | n/a |

(All three numbers are baseline runs over 3 scenarios with
warmup_requests=1 — see [§ caveats](#caveats-on-the-vllmsglang-numbers).)

## Failure mode

Server log: `benchmarks/results/rtx_pro6000/server-logs/trtllm.log`,
line 278.

```
[TRT-LLM] [E] Failed to initialize executor on rank 0:
The checkpoint you are trying to load has model type `qwen3_5` but
Transformers does not recognize this architecture. This could be
because of an issue with the checkpoint, or because your version of
Transformers is out of date.
```

Stack trace (truncated):

```
File ".../tensorrt_llm/_torch/models/checkpoints/hf/config_loader.py", line 11
    return ModelConfig.from_pretrained(checkpoint_dir, **kwargs)
File ".../transformers/models/auto/configuration_auto.py", line 1360
    config_class = CONFIG_MAPPING[config_dict["model_type"]]
File ".../transformers/models/auto/configuration_auto.py", line 1048
    raise KeyError(key)
KeyError: 'qwen3_5'
```

The failure happens inside TRT-LLM's checkpoint loader before any GPU
work begins — TRT-LLM does not run with `trust_remote_code=True` and
relies entirely on the `transformers` `CONFIG_MAPPING` registry to
resolve `model_type` → config class. `qwen3_5` is not registered in
the build of `transformers` shipped with `tensorrt_llm==1.2.1`.

## Why vLLM and SGLang don't fail

Qwen3.5/3.6 isn't a vanilla decoder. The architecture interleaves
regular attention with Mamba/SSM (linear-attention) blocks. vLLM and
SGLang both ship native implementations and explicit registry entries
for it.

Evidence in [sglang.log](../../benchmarks/results/rtx_pro6000/server-logs/sglang.log):

```
Load weight end. ... type=Qwen3_5ForConditionalGeneration, quant=fp8, fmt=e4m3, ...
Mamba Cache is allocated. max_mamba_cache_size: 154,
   conv_state size: 0.43GB, ssm_state size: 21.80GB
KV Cache is allocated. #tokens: 403904, K size: 12.33 GB, V size: 12.33 GB
```

Notice the SSM state cache (21.8 GB) is bigger than the KV cache
(12.3 GB+12.3 GB) — most of the working memory on this model is SSM,
not attention. That's the architecture TRT-LLM doesn't yet have a
loader for.

vLLM's logs show the same arch ID:

```
config: ... model='Qwen/Qwen3.6-27B-FP8', ...
```

Both backends recognize it via their own (non-`transformers`) registry
and load with no trust-remote-code escape hatch.

## What we measured (vLLM and SGLang)

Aggregate rows in `benchmarks/results/rtx_pro6000/`:

| Metric | vLLM | SGLang |
|---|---:|---:|
| TTFT p50 | 537 ms | 578 ms |
| TTFT p95 | 731 ms | 770 ms |
| E2E p50 | 1922 ms | 4573 ms |
| E2E p95 | 3595 ms | 6959 ms |
| ITL p50 | 22.3 ms | 38.2 ms |
| tok/s decode | **44.8** | 26.2 |
| Mean prompt tokens | 1470 | 1469 |
| Mean completion tokens | 89.3 | 113.0 |
| seq/s | 0.4 | 0.2 |
| GPU util peak | 100% | 97% |
| mem-bw util peak | 76.9% | 45.2% |
| FB peak | 84.9 GB | 79.0 GB |
| Power avg | 318 W | 295 W |
| Energy / req | 803 J | 1441 J |

**vLLM wins decode throughput** on this model — 44.8 tok/s vs SGLang's
26.2. Two things stand out vs. the Qwen3-VL-32B-FP8 baseline:

- vLLM's 44.8 tok/s on Qwen3.6-27B is **higher** than its 36.6 tok/s
  on Qwen3-VL-32B. Smaller weights (27B vs 32B) plus the SSM mixer
  shift the cost profile in vLLM's favour.
- SGLang's E2E p50 is 2.4× longer than vLLM's despite a similar TTFT.
  Likely related to the SSM state cache management; worth a deeper
  look once we have more rounds.

## Caveats on the vLLM/SGLang numbers

These are 3-scenario smoke runs with `warmup_requests=1`, so the first
request of each round still pays first-touch cost. Compare *trends*,
not absolute milliseconds. Re-run with `--warmup-requests 3+` once
qwen3.6 is part of a larger sweep.

`grammar_validity_rate` is 0% on SGLang and 33% on vLLM for this
model. **This is not a Qwen3.6-specific finding** — it's the same
schema-vs-action-args mismatch we see across the whole matrix
(model emits `{"button": "left_mouse_button"}` instead of
`{"button": "left"}`). Tracked separately; see
[trtllm-1.2.1-qwen3-vl-32b-fp8.md](trtllm-1.2.1-qwen3-vl-32b-fp8.md).

## Workaround in the benchmark

The three GPU YAMLs now pin Qwen3.5/3.6 sweep rounds to
`backends: [vllm, sglang]` so the harness skips trtllm cleanly:

- [rtx_pro6000.yaml](../../benchmarks/configs/rtx_pro6000.yaml) — `qwen3.6-27b-fp8`
- [rtx5090.yaml](../../benchmarks/configs/rtx5090.yaml) — `qwen3.5-9b`
- [h200.yaml](../../benchmarks/configs/h200.yaml) — `qwen3.6-35b-a3b-fp8`, `qwen3.6-27b-fp8`

Each affected model also carries an inline `backend_args` comment that
points readers here.

This avoids three problems that bit us before adding the workaround:

1. The trtllm round would crash mid-startup, leaking VRAM and a
   half-spawned MPI worker, blocking the next backend in line.
2. With fail-fast on, the whole sweep would halt at this round, even
   though vLLM and SGLang are healthy on the same model.
3. Each crash spent ~30 seconds writing a stack trace and ~10 minutes
   downloading 27 GB of weights only to fail on the first config read.

## Honest framing for the report

> Qwen3.5/3.6 is a hybrid attention + Mamba/SSM architecture. vLLM and
> SGLang have explicit native support and load the model day-one;
> TRT-LLM 1.2.1's pytorch backend does not — its bundled `transformers`
> registry is missing the `qwen3_5` model_type entry, so the loader
> raises `KeyError` before any GPU work begins. **This is a TRT-LLM
> integration gap, not a hardware or kernel issue.** Affected
> checkpoints will run on TRT-LLM as soon as upstream registers the
> arch (or ships a native pytorch model implementation with Mamba
> kernels).

## Open questions / next steps

- **Watch the upstream TRT-LLM release notes** for `qwen3_5` /
  `Qwen3_5ForConditionalGeneration` arch registration, or for a native
  pytorch backend implementation that includes the SSM/Mamba kernels.
  When it lands, drop the `backends: [vllm, sglang]` pins from the
  three YAMLs and re-run.
- **Try `pip install -U transformers` inside `.venv-trtllm`** as an
  experiment. If newer `transformers` registers `qwen3_5` AND
  TRT-LLM's pytorch-backend model loader has a fallback path that
  works without a TRT-LLM-side registration, the failure could be
  resolved without waiting for a TRT-LLM release. Don't deploy this
  to the benchmark venv until verified — TRT-LLM 1.2.1 pins
  `transformers==4.57.3` for a reason.
- **Re-run with bigger warmup** (`--warmup-requests 3+`) before
  citing the vLLM/SGLang absolute latencies. Current numbers include
  first-touch cost on the first scenario.
- **Investigate SGLang's 2.4× E2E gap** vs vLLM on this model. SSM
  state cache management is the most likely culprit; not
  characterised yet.
