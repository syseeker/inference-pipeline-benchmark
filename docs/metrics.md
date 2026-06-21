# Metrics — what counts as success

Tokens/sec is **not** the decision metric for this use case.

## End-to-end (e2e) latency — the headline number

When summary.md says `e2e p50 / p95 / p99`, it means **`LatencyBreakdown.total_ms`** —
the full `Pipeline.run()` wall time from request-in to executor-decision-out.
This covers **every stage the user pays for**, not just the LLM call:

```
e2e (total_ms) = vision_encoder_ms      # 1. encode image (passthrough today)
               + reasoner_total_ms      # 2. VLM generate (TTFT + all decode tokens)
               + decoder_ms             # 3. parse raw text → ActionSequence
               + validator_ms           # 4. schema + safety validator
               + executor_ms            # 5. dry-run executor (or real, when wired)
```

If decoder/validator rejects the response, executor is skipped but `total_ms`
still includes everything up to that rejection — so a 0% validity run still
emits a meaningful e2e number. **e2e is the user-experienced latency**;
TTFT / ITL / prefill / decode breakdowns under §2 of summary.md exist to
*explain* a regression in e2e, not to replace it.

`reasoner_total_ms` is the LLM's slice (TTFT + all decode tokens until stop).
The remaining stages typically add only a few ms, so `e2e ≈ vision + reasoner`
in practice today, but that ratio shifts the moment a real vision tower or
heavier validator is wired in.

## Section-by-section guide to summary.md

The headings below map 1:1 to those in
`benchmarks/results/<gpu>/summary.md`. Use this as the reference for
"what does column X mean / where does it come from / when do I look
here?"

### Core findings (auto)

Auto-generated from the dataset by `summary.py`. Bullet 1 is always the
winner (lowest `e2e_p50_ms`); subsequent bullets cover framework gaps,
non-competitive outliers, validity floor, energy spread, mem-bw
saturation, TTFT-vs-decode share, and section-4 status — each
emitted only when the data warrants it. Caps at 10 bullets.

The **Why** / **How to improve** fields on underperformer bullets are
filled from `docs/findings/knowledge.yaml` when a
`(gpu, framework, model[/variant], symptom)` match exists. Unmatched
items show `[TBD]` for an operator pass. **No LLM is required** to
populate or read the knowledge file — anyone can edit the YAML and
re-run `python -m benchmarks.summary --gpu <gpu>`. See the saved
core-findings style guide for the bullet shape.

### §1 Decision metrics

The headline pass/fail table. Columns:

| Column | BenchmarkResult field | Meaning | Target |
| --- | --- | --- | --- |
| `framework` / `label` | `framework`, `run_label` | which engine + variant tag (e.g. `baseline`, `eager`, `chunked_off`) | — |
| `quant` / `run_id` / `n` | `quantization`, `run_id`, `n_requests` | identity / repro pointer | — |
| `grammar_valid` | `grammar_validity_rate` | fraction passing schema **AND** safety validator (today: pydantic-parse + arg-schema + banned-keys + ≤16 commands) | high — close to 1.0 |
| `exec_accept` | `command_success_rate` | fraction the executor accepted and ran. Today the executor is `DryRunExecutor`, which mirrors `grammar_valid` — wire a real downstream signal to make this independent. | high — task-suite threshold |
| `e2e p50/p95/p99` | `e2e_p50_ms` / `_p95_ms` / `_p99_ms` | percentiles of `total_ms` (see above), aggregated over **all completed requests** | meet interactive budget |

A run with great tokens/sec but a low validity rate is **a failed run**.
The decoder must produce something the validator accepts.

> **Field-name history:** the e2e fields used to be `valid_e2e_*`.
> The "valid_" prefix implied a validity filter that the impl never
> applied — renamed 2026-05-10. summary.py keeps a fallback-read so
> historical JSONs still render.

### §2 Latency diagnostics

Why an `e2e p50` moved. Two halves: client-side (TTFT, ITL — measured
by the runner around each `Pipeline.run`) and server-side (prefill /
decode / queue — scraped from the framework's Prometheus `/metrics`
endpoint at end of run).

| Column | Source | Meaning |
| --- | --- | --- |
| `ttft p50/p95/p99` | client; `LatencyBreakdown.reasoner_ttft_ms` | time from request-send to first token (after vision encoding). Subtract from `e2e` to bound prefill + queue + first-token decode. |
| `itl p50/p95/p99` | client; derived in `derive_itl()` | inter-token latency = `(total_ms − ttft) / (completion_tokens − 1)` per request. Speaks to steady-state decode cost. |
| `prefill p50` | server `/metrics` | server's own prefill histogram (vllm/sglang only; trtllm needs `return_perf_metrics: true` to expose) |
| `decode p50` | server `/metrics` | server's per-iter decode time histogram |
| `queue p50` | server `/metrics` | time the request waited in the scheduler queue |

When prefill/decode/queue are `-`, the server isn't exposing them. The
note above the table calls out which framework's `/metrics` is
incomplete (today: trtllm by default).

`vision_encoder_p50_ms` is computed but not currently surfaced as a
table column — today the encoder is a passthrough so the column would
be all-zero. It's available on `BenchmarkResult` for when a real
vision tower is wired.

### §3 Throughput & token counts

How fast (and how *useful*-fast) the system completes work.

| Column | Field | Meaning |
| --- | --- | --- |
| `seq/s` | `throughput_seq_per_s` | completed requests per wall-clock second over the whole timed loop |
| `goodput` | `goodput_seq_per_s` | `seq/s × grammar_validity_rate` — invalid completions don't count |
| `tok/s_decode` | `tokens_per_sec_decode` | decode-only tokens-per-second computed from `(completion_tokens − 1) / (e2e − ttft)` per request, **not** wall time. Use this to reason about decode cost; not for capacity planning. |
| `mean prompt_toks` | `mean_prompt_tokens` | for VLMs **includes vision tokens** — a 1024×1024 image is hundreds-to-thousands of tokens depending on patcher. |
| `mean comp_toks` | `mean_completion_tokens` | average completion length |

If `goodput` is 0 while `seq/s` is healthy, fix validity before tuning
throughput.

### §4 Cache & scheduling

Whether the engine is reusing prefill work and what scheduling knobs
were active.

| Column | Field | Source | Meaning |
| --- | --- | --- | --- |
| `prefix_cache_hit` | `prefix_cache_hit_rate` | server `/metrics`, **polled in-run** by `PromPoller` | fraction of prefill tokens served from a cached prefix. 0 (or `-`) when prefix caching is off (e.g. `--no-enable-prefix-caching`). |
| `kv_cache_usage` | `kv_cache_usage_pct` | server `/metrics`, polled in-run for peak | KV blocks in use at the run's peak (gauge). 0% when no requests in flight; the in-run poller catches the peak rather than the post-drain state. |
| `chunked_prefill` | `chunked_prefill_enabled` | inferred from launch args by `_detect_server_flags()` | on / off / `-` (`-` = framework has no equivalent at this layer). Falls back to framework defaults (vllm + sglang: on) when no explicit flag is passed. |
| `enforce_eager` | `enforce_eager` | inferred from launch args | on / off / `-`. Falls back to defaults (vllm + sglang: off). |

> **In-run polling history:** before 2026-05-10 these gauges were
> scraped once *after* the timed loop, so they read 0 (queue had
> drained) and the flag detector ignored framework defaults — the
> whole section was empty. `benchmarks/probes/prom_poller.py` now
> polls every 500 ms and tracks peaks; the detector now falls back
> to known framework defaults plus sglang's `--chunked-prefill-size`
> / `--disable-cuda-graph`.

### §5 GPU resource usage

Hardware-level evidence for *why* a framework is fast or slow.
Sampled by [`benchmarks/probes/gpu_sampler.py`](../benchmarks/probes/gpu_sampler.py)
(DCGM-first, nvidia-smi fallback) at 250 ms cadence around the timed loop.

| Column | Field | Sampler | Meaning |
| --- | --- | --- | --- |
| `sampler` | `sampler_backend` | — | `dcgm` / `nvidia-smi` / `none`. Determines what's available below. |
| `mem_bw p50` / `peak` | `mem_bw_util_pct_p50` / `_peak` | DCGM only (`DCGM_FI_PROF_DRAM_ACTIVE`, field 1005) | DRAM-bandwidth utilisation, 0–100%. ≥70% = bw-bound; <50% = headroom for concurrency. Speaks directly to the bandwidth thesis. `n/a` when only nvidia-smi is available. |
| `gpu_util p50` | `gpu_util_pct_p50` | both | SM occupancy proxy. High `gpu_util` with low `mem_bw` is compute-bound; both high = saturated. |
| `fb peak (GB)` | `fb_used_peak_gb` | both | peak framebuffer used. Plan headroom against the GPU's total VRAM. |
| `power avg / peak (W)` | `power_avg_w` / `_peak_w` | both | matters for energy/req and for capacity (single-GPU thermals). |
| `energy/req (J)` | `energy_per_request_j` | derived | `power_avg_w × wall_time_s / n_completed`. Long-tail TTFT or low-throughput runs pay more energy even at lower `power_avg`. |

If `sampler_backend = none`, GPU rows show `-` everywhere — install
DCGM or run the script on a host with `nvidia-smi`.

### §6 Cross-run deltas

Pairs `run_label` variants against the matching `baseline` for the same
`(framework, model)`. Driven by `_cross_run_section()` in `summary.py`
and the `_KNOWN_VARIANTS` map. Headline comparisons:

- **graph → eager** — `cuda_graph_speedup = e2e_p50(eager) / e2e_p50(baseline)`. Quantifies CUDA-graph capture impact.
- **bf16 → fp8 / int8** — `quant_accuracy_delta = grammar_validity(baseline) − grammar_validity(variant)` in pp; throughput uplift on `tok/s_decode`.
- **TP=1 → TP=2** — `tp_efficiency = e2e_p50(baseline) / (2 × e2e_p50(tp2))`. > 1 is a real win; ≤ 1 means PCIe overhead ate the parallelism (especially relevant on RTX 5090 — no NVLink).
- **chunked_prefill on/off** — TTFT and decode percentage shifts.

When the table is empty (`_no variant runs to pair with baseline_`), no
non-baseline `run_label` was seen for any `(framework, model)` — extend
the sweep `rounds:` in `benchmarks/configs/<gpu>.yaml` to enable.

### §7 Per-scenario detail

Per-(scenario, run_id) raw rows under each backend sub-heading. This is
where you go to debug *which* scenarios failed and why. Columns:

| Column | Source | Meaning |
| --- | --- | --- |
| `run_id` / `label` / `scenario` | per-row identity | match against `<gpu>/<backend>/<scenario>__<run_id>.json` |
| `total_ms` | `latency_ms.total_ms` | the same e2e the aggregate row percentiles, but per-request |
| `ttft_ms` | `latency_ms.reasoner_ttft_ms` | per-request TTFT |
| `schema_valid` / `safe` | `validation.*` | the two halves of `grammar_valid` — split here so you can see *which* gate failed |
| `executed` | `was_executed` | did the executor actually run (today: DryRunExecutor) |
| `error` | `error` | first 60 chars of the failure reason (JSON parse failure, schema mismatch, etc.) |

A row with `schema_valid=True, safe=False` failed the safety gate
(banned key, missing args, sequence > 16 commands). A row with
`schema_valid=False` failed pydantic — the model didn't emit the
expected JSON shape; check `error` for the parse error.

### §8 Environment

`(framework, framework_version, driver, cuda)` distinct rows seen in
this dataset. If you see `unknown` driver / cuda, the runner couldn't
read the host metadata file (`host_<hostname>.json` from
`scripts/gpu_probe.sh`) — re-run the probe and the next sweep will
populate.

## Policy accuracy-vs-gold (NitroGen)

For the **NitroGen** diffusion-policy backend there is no notion of grammar
validity vs gold *text* — the model emits a gamepad action, and "correct" means
"close to the dataset's recorded action." These fields are populated only for
the `nitrogen` backend (computed per-scenario from `ModelMeta.extras["gamepad"]`
vs the scenario's `gold_action.json` sidecar, then averaged) by
[`benchmarks/accuracy.py`](../benchmarks/accuracy.py):

| Field | Meaning | Target |
| --- | --- | --- |
| `action_mse` | MSE over the joined action vector (stick axes + the 17 shared buttons) | low |
| `button_agreement_rate` | fraction of the 17 dataset buttons whose 0/1 state matches gold | high — close to 1.0 |
| `joystick_mae` | mean absolute error over the 4 analog-stick axes ([-1,1]) | low |
| `denoise_steps` | flow-matching iterations this run (a knob, recorded for context) | — |

This is the lever for the optimization study: it tells you whether an FP8 /
NVFP4 / TensorRT / reduced-step run **still produces the same action** as the
BF16 reference, not just whether it's faster. Denoising noise is **seed-pinned**
across runs so the delta reflects precision, not sampling. The bf16/eager run is
the reference; `quant_accuracy_delta` (§6) pairs a quantized run against it.

### Which metrics apply to which backend

| Metric group | VLM backends | NitroGen |
| --- | --- | --- |
| e2e latency p50/p95/p99 | ✓ | ✓ |
| throughput / goodput | ✓ | ✓ |
| GPU util / power / energy (§5) | ✓ | ✓ (sampler is backend-agnostic) |
| TTFT / ITL / token counts (§2–3) | ✓ | — (no token stream) |
| prefix-cache / KV / prefill / decode / queue (§4) | ✓ | — (no Prometheus server) |
| `grammar_validity_rate` / `command_success_rate` | ✓ | ✓ (against the lossy `ActionSequence`) |
| `action_mse` / `button_agreement_rate` / `joystick_mae` | — | ✓ |

Fields that don't apply are recorded as `null` and the run's `notes` say why —
they are not silently zeroed.

## Reporting shape

Each benchmark cell produces one `BenchmarkResult` row; rows are joined
into a per-GPU summary table that lives in
`benchmarks/results/<gpu>/summary.md`. Per-scenario rows live under
`benchmarks/results/<gpu>/<framework>/<scenario>__<run_id>.json` so
history accumulates across runs rather than overwriting.

## Statistical hygiene

- Always report **p50, p95, p99**, not just averages.
- Warm up before timing (CUDA graph capture, KV warm).
- Run with realistic concurrency, not single-stream — concurrent
  pipelines are how the executor will actually use this.
- Hold seed and image set fixed across frameworks so comparisons are
  apples-to-apples.
- Record framework version, GPU, driver, and CUDA in every row.
