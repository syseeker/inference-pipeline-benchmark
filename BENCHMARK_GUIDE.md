# Benchmark guide

Operational reference for `scripts/run_all_scenarios.sh` and
`benchmarks.runner` — the yaml schema, the run modes, what gets written
where, and how to debug when something's off.

For the metric definitions see [docs/metrics.md](docs/metrics.md).
For setup see [INFERENCE_BACKENDS.md](INFERENCE_BACKENDS.md).
For the server-alive check before benchmarking see [SMOKE_TESTS.md](SMOKE_TESTS.md).

## CLI wrapper — `bench`

In practice, you (and your agents) drive everything through the `bench`
console script. It wraps the scripts in this guide with a stable JSON
status contract:

| Command | What it wraps | When |
|---|---|---|
| `bench probe` | `scripts/gpu_probe.sh` | GPU + driver + per-backend versions; agent should run before everything else |
| `bench setup --backend X` | per-backend venv + pip extras | idempotent install. `X` ∈ `vllm`/`sglang`/`trtllm`/`nim`/`nitrogen`/`nitrogen-quant`/`profile` |
| `bench scenarios build --source nitrogen` | `scripts/build_nitrogen_scenarios.py` | dataset → scenarios; respects the `pipeline_bench.scenario_sources` entry-point group for customer-supplied sources |
| `bench smoke --gpu G --backend B --model M` | `scripts/run_all_scenarios.sh` (single round) | confirm `(backend, model)` works before paying for a sweep |
| `bench sweep --gpu G --sweep S` | `scripts/run_all_scenarios.sh` (multi-round) | the full benchmarking pass; produces summary.md |
| `bench summary --gpu G` | `python -m benchmarks.summary` | regenerates summary.md from existing result JSONs |
| `bench load-test --gpu G --backend B --model M --concurrency …` | [AIPerf](https://github.com/ai-dynamo/aiperf) `profile` | **HTTP backends only** (vLLM/SGLang/TRT-LLM/NIM); produces the concurrency curve in summary.md §9 |
| `bench profile --tool nsys --gpu G --backend B --model M` | `nsys profile` (or `ncu`) around one round | escalation: prove a hypothesis with a real timeline. Outputs `.nsys-rep` + auto `.summary.md` (from `nsys stats`) |
| `bench install-skill --agent auto` | symlink-or-copy `skills/<name>/` into the agent's load path | Claude Code, Cursor, Codex; one-time per workstation |

Exit codes the agent can branch on: `0` ok | `1` generic | `2` unsupported combo (per yaml `unsupported_backends:`) | `3` runtime fail | `4` missing dep. Every command supports `--json` so structured tooling has one thing to parse.

Each command in this guide that calls `scripts/run_all_scenarios.sh` directly is still correct — `bench *` is a wrapper, not a replacement. Both paths produce identical artifacts.

## What `bench load-test` and `bench profile` add (the new measurement axes)

The base `bench sweep` measures e2e latency at concurrency=1 — that's the customer-experience number. Two follow-up tools each open a different axis:

- **`bench load-test`** — wraps [AIPerf](https://github.com/ai-dynamo/aiperf) (NVIDIA's client-side load generator). Comma-separated `--concurrency` becomes a sweep: 1 / 4 / 16 / 32 / … . Reports TTFT/ITL/RPS/TPS per level. Populates summary.md **§9 — Concurrency profile**. Answers *"how many parallel sims per GPU"* and *"where does TTFT degrade under load."* HTTP backends only (NitroGen ZMQ is single-flight by design).
- **`bench profile`** — wraps [Nsight Systems](https://developer.nvidia.com/nsight-systems) (default, `--tool nsys`) or [Nsight Compute](https://developer.nvidia.com/nsight-compute) (`--tool ncu`). Wraps one round under the profiler and writes the binary report. The `nsys` path also auto-emits a `<run>.summary.md` from `nsys stats` (top NVTX regions + GPU activity) so agents can quote from it without opening the GUI. **Escalation only** — overhead is real (~5–10% for nsys, 10×+ for ncu).

First-time nsys install: `bench setup --backend profile` — `apt-get` discovery of the latest `nsight-systems-YYYY.X.Y` package, post-install `chmod -R o+rX` of NVIDIA's root-only `/opt/nvidia/nsight-systems/`, and a `/usr/local/bin/nsys` symlink so plain `bench profile` finds it. Falls back to a tarball-download hint if apt isn't available.

## What the harness does

```
                 benchmarks/configs/<gpu>.yaml
                         (single source of truth)
                                  │
              ┌───────────────────┼────────────────────┐
              ▼                   ▼                    ▼
       start vllm server   start sglang server   start trtllm server
              │                   │                    │
              ▼                   ▼                    ▼
   benchmarks.runner — runs every scenario in
   tests/smoke/scenarios/ (or --scenarios-dir) through Pipeline
   (encoder → reasoner → decoder → validator → executor) and records
   latency + validation.
              │                   │                    │
              ▼                   ▼                    ▼
        per-scenario json   per-scenario json    per-scenario json
        + aggregate row     + aggregate row      + aggregate row
              └───────────────────┼────────────────────┘
                                  ▼
                       benchmarks.summary
                                  ▼
                  benchmarks/results/<gpu>/summary.md
```

Each "run" is one `(backend, model, variant)` round. `variant` is
optional — it composes additional launch flags onto the backend's base
config.

## Run modes

### Single round (one backend, default model, no variant)

```bash
scripts/run_all_scenarios.sh                            # all 3 backends, rtx_pro6000
scripts/run_all_scenarios.sh --gpu h200                 # different GPU profile
scripts/run_all_scenarios.sh --backends "vllm sglang"   # subset of backends
```

### Override the model for one run

```bash
scripts/run_all_scenarios.sh --model qwen3.6-27b-fp8
```

`--model` accepts any id from the GPU yaml's `models:` block.

### Backend-flag A/B comparison (variants)

```bash
scripts/run_all_scenarios.sh --backends vllm --variants "baseline eager"
```

Runs vLLM twice — once with the yaml's base flags (`baseline`), once
with `--enforce-eager` appended (`eager`). The cross-run section of
`summary.md` then surfaces `cuda_graph_speedup`. Variants are defined
per backend in the yaml; running a variant on a backend that doesn't
define it is silently skipped.

### Auto-multi-round (yaml sweep)

```bash
scripts/run_all_scenarios.sh --sweep models
```

Reads `sweeps.<name>` from the GPU yaml and iterates each round on the
backends listed for that round. The shipped yaml defines:

| Sweep | What it does |
| --- | --- |
| `models` | Each candidate model × all 3 backends — apples-to-apples model comparison |
| `vllm_knobs` | `baseline / eager / chunked_off` on vllm only — backend-flag A/B |
| `tp2` (h200 only) | TP=2 variant on vllm + sglang — needs a 2× HGX H200 host |
| `full` | Mixed matrix — the union of the above |

### Custom scenarios

```bash
scripts/run_all_scenarios.sh --scenarios-dir /path/to/my_scenarios
```

See [tests/smoke/scenarios/README.md](tests/smoke/scenarios/README.md)
for the directory layout and the Pydantic schema.

## YAML schema (one GPU = one yaml)

Every per-backend launch knob lives in `benchmarks/configs/<gpu>.yaml`.
The shell script reads it via `benchmarks/scenario_config.py`; the
Python runner reads the same file. Nothing is hardcoded.

```yaml
# GPU metadata — used by summary.md and capacity-fit decisions.
display_name: "RTX PRO 6000 Blackwell Server Edition"
arch: blackwell
vram_gb: 96
peak_bandwidth_tbps: 1.6
nvlink: false
tensor_parallel: 1

# Models: one block per candidate. Keys are stable ids referenced by
# `default_model:` and by sweep rounds.
models:
  qwen3-vl-32b-fp8:
    hf_id: "Qwen/Qwen3-VL-32B-Instruct-FP8"
    family: "qwen3-vl"
    role: "vlm-headline"
    quantization: "fp8"
    notes: "..."
  nemotron-omni-fp8:
    hf_id: "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
    family: "nemotron"
    quantization: "fp8"
    # Optional: launch flags this model needs on a specific backend
    # (e.g. asking vllm to load BF16 weights as FP8 at runtime).
    backend_args:
      vllm:   ["--quantization=fp8"]
      sglang: []
      trtllm: []

# Default model loaded by every backend unless --model / sweep overrides.
default_model: qwen3-vl-32b-fp8

# Backends: launcher params only, model-agnostic.
backends:
  vllm:
    base_url: "http://localhost:8000/v1"
    port: 8000
    extra_args:
      - "--gpu-memory-utilization=0.90"
      - "--max-num-seqs=32"
    variants:                              # backend-flag A/B groups
      eager:       ["--enforce-eager"]
      chunked_off: ["--no-enable-chunked-prefill"]

  sglang:
    base_url: "http://localhost:30000/v1"
    port: 30000
    extra_args: []
    variants: {}

  trtllm:
    base_url: "http://localhost:8002/v1"
    port: 8002
    backend: pytorch                       # pytorch | trtllm | _autodeploy
    extra_args:
      - "--extra_llm_api_options=/tmp/trtllm-vlm.yml"
    variants: {}

# Sweeps: named ordered lists of rounds. Each round can override `model`
# and/or `variant`. Optional per-round `backends:` filter.
sweeps:
  models:
    rounds:
      - model: qwen3-vl-32b-fp8
      - model: qwen3-vl-32b-bf16
      - model: nemotron-omni-fp8
  vllm_knobs:
    backends: [vllm]
    rounds:
      - {}                                 # baseline
      - variant: eager
      - variant: chunked_off
```

### Resolution rules

- **HF id** comes from `models.<id>.hf_id` (resolved id = `--model` flag,
  sweep round's `model:`, or `default_model`).
- **Launch args** = `backends.<bk>.extra_args` + `variants.<variant>` (if
  any) + `models.<id>.backend_args.<bk>` (if any).
- **`bench.model_label`** in the result JSON = the model id (yaml key).
- **`bench.quantization`** = `models.<id>.quantization`.

For `trtllm`, `backends.trtllm.backend` selects the trtllm-serve
execution path. Allowed values:

| Value | Maps to | When |
| --- | --- | --- |
| `pytorch` | `--backend pytorch` | **Default for all headline picks.** Day-0 model coverage; only path that supports Qwen3-VL / Qwen3.6 / Nemotron Omni today. |
| `trtllm` | `--backend tensorrt` | AOT-compiled TRT engine. Strongest on dense text + stable shapes; multimodal coverage limited. Check [TRT-LLM supported-models](https://nvidia.github.io/TensorRT-LLM/models/supported-models.html). |
| `_autodeploy` | `--backend _autodeploy` | Experimental auto-routing path; not for headline runs. |

> **TRT-LLM multimodal constraint.** TRT-LLM is incompatible with KV-cache
> reuse for any multimodal model. The yaml's trtllm `extra_args` carries
> `--extra_llm_api_options=/tmp/trtllm-vlm.yml`; create the file once with:
>
> ```bash
> cat > /tmp/trtllm-vlm.yml <<'EOF'
> kv_cache_config:
>   enable_block_reuse: false
> EOF
> ```

### Adding a model / variant / sweep

- **New model**: add a key under `models:` with `hf_id`, `family`,
  `quantization`, optional `backend_args`. Reference it from
  `default_model:` or in a sweep round.
- **New backend variant**: add a key under
  `backends.<backend>.variants:` whose value is a list of additional
  flags. Run with `--variants "baseline <name>"` for the A/B.
- **New sweep**: add a key under `sweeps:` with `rounds:` (and optional
  `backends:` filter). Run with `--sweep <name>`.

### Adding a GPU profile

1. Copy `benchmarks/configs/rtx_pro6000.yaml` → `benchmarks/configs/<your-gpu>.yaml`.
2. Update `display_name`, `arch`, `vram_gb`, `peak_bandwidth_tbps`, `nvlink`.
3. Curate the `models:` block to fit your VRAM (see [docs/capacity.md](docs/capacity.md)).
4. Run with `--gpu <your-gpu>`.

There's no GPU registry — the script resolves the file by name, so the
filename is the profile name.

## Direct CLI (when you've already started a server)

The shell script wraps `python -m benchmarks.runner`, which you can call
directly when a server is already running on the expected port:

```bash
python -m benchmarks.runner --gpu rtx_pro6000 --backend vllm
python -m benchmarks.runner --gpu rtx_pro6000 --backend vllm --model qwen3.6-27b-fp8
python -m benchmarks.runner --gpu rtx_pro6000 --sweep models
```

| Flag | Default | When to change |
| --- | --- | --- |
| `--gpu` | required | always |
| `--backend` | required (unless `--sweep`) | always for single-round mode |
| `--model` | yaml `default_model` | running a non-default candidate |
| `--variant` | none | applying a backend-flag variant |
| `--sweep` | none | iterating a yaml sweep block |
| `--label` | variant name or `baseline` | tagging rows for cross-run pairing |
| `--scenarios-dir` | `tests/smoke/scenarios/` | custom dataset |
| `--warmup-requests 1` | 1 | first-request slowness was skewing p50 |
| `--gpu-index 0` | 0 | multi-GPU host; `-1` to skip the GPU sampler |

## Outputs

```
benchmarks/results/rtx_pro6000/
├── summary.md                                  # per-GPU table (regenerated each run)
├── vllm-qwen3-vl-32b-fp8-<run_id>.json         # aggregate BenchmarkResult, one per round
├── sglang-qwen3-vl-32b-fp8-<run_id>.json
├── trtllm-qwen3-vl-32b-fp8-<run_id>.json
├── vllm/
│   ├── 01_clash_of_clans_start_attack__<run_id>.json   # per-scenario detail
│   └── ...
├── sglang/...
├── trtllm/...
└── server-logs/
    ├── vllm.log
    ├── sglang.log
    └── trtllm.log
```

### Per-scenario JSON

One file per (backend × scenario × run). Includes:
- `latency_ms` — full `LatencyBreakdown` from the pipeline.
- `validation` — `ValidationReport` (`schema_valid`, `safe`, rejected
  command indices, notes).
- `actions_actual` — what the model produced.
- `actions_gold` — the scenario's expected actions.
- `was_executed`, `error`, `model_meta`, `variant`.

This is the row to inspect when debugging a single failure.

### Aggregate JSON

One file per round. Same shape as `benchmarks.metrics.BenchmarkResult` —
documented in [docs/metrics.md](docs/metrics.md). Carries:
- p50 / p95 / p99 of `valid_e2e` and `ttft`
- `grammar_validity_rate`, `command_success_rate`
- `framework_knobs` — what was actually passed to the server
  (`model_id`, `hf_id`, `family`, `variant`, `launch_args`,
  `trtllm_backend` for trtllm)
- `framework_version`, `gpu`, `driver`, `cuda` for reproducibility

### `summary.md`

`benchmarks/results/<gpu>/summary.md` is generated last by
`benchmarks.summary`. Three sections:

1. **Decision metrics** — one row per aggregate JSON.
2. **Diagnostic metrics** — TTFT / vision / KV-hit / ITL.
3. **Per-scenario detail** — every per-scenario row, grouped by backend.

Regenerate without re-running benchmarks:

```bash
python -m benchmarks.summary --gpu rtx_pro6000
```

## Prerequisites

Set up once per backend per GPU host. Detailed in
[INFERENCE_BACKENDS.md](INFERENCE_BACKENDS.md); the short version:

| Backend | Venv             | Driver                                  | Model on disk                                  |
| ------- | ---------------- | --------------------------------------- | ---------------------------------------------- |
| vllm    | `.venv-vllm`     | `vllm serve` (HTTP)                     | HF id (per-GPU pick); HF-cached on first launch |
| sglang  | `.venv-sglang`   | `python -m sglang.launch_server` (HTTP) | HF id (per-GPU pick); HF-cached on first launch |
| trtllm  | `.venv-trtllm`   | `trtllm-serve --backend pytorch` (HTTP) | HF id (per-GPU pick); HF-cached on first launch |

The sweep skips any backend whose `.venv-<backend>` is missing rather
than failing.

## Troubleshooting

### `vllm` crashes mid-run with `deepstack tokens > buffer`

Known interaction between prefix caching + chunked prefill + Qwen3-VL.
The shipped yaml keeps prefix caching off via the explicit `extra_args`.
Don't add `--enable-prefix-caching` until upstream patches it.

### Server didn't become ready in time

Default wait is 600s, set by `READY_TIMEOUT_S`. First-time vLLM startup
includes weight download + CUDA graph capture (~3 minutes on this host).
Bump for larger models:

```bash
READY_TIMEOUT_S=1200 scripts/run_all_scenarios.sh --backends vllm
```

Server logs land in `benchmarks/results/<gpu>/server-logs/<backend>.log`.

### Port already in use

Another server you started manually is still running. The script does
not adopt foreign servers. Either stop the old one or change
`backends.<bk>.port` in the yaml.

### `command_success_rate` ≈ `grammar_validity_rate`

Expected today — `DryRunExecutor` accepts anything the validator passes.
The two diverge once a real executor that can fail is wired up.

### TRT-LLM server fails on launch — `kv_cache_reuse` error

The yaml's `extra_args` carries `--extra_llm_api_options=/tmp/trtllm-vlm.yml`
to disable KV-cache reuse (required for multimodal models). Create the
file once — see the schema section above.

### TRT-LLM model not found in the registered architectures

The PyTorch backend reads the HF `config.json`'s `architectures` field
and dispatches via `@register_auto_model`. Verify the model is listed at
[supported-models.html](https://nvidia.github.io/TensorRT-LLM/models/supported-models.html);
if missing, the run is blocked until upstream registration lands in
your installed `tensorrt-llm` version.

### Variant silently skipped

That backend doesn't define the variant in its `variants:` block. Add
one or pick a variant that's defined for the backend you're running.

### `summary.md` shows dashes for some fields

First run on a new branch may have rows that pre-date a metric (e.g.
GPU sampler, prefix-cache hit rate). Re-run produces fully-populated
rows.

### `mem_bw_util_pct = n/a`

DCGM not installed or `nv-hostengine` not running. Either install
(`apt install datacenter-gpu-manager`) or accept the nvidia-smi
fallback (no mem-bw figure).
