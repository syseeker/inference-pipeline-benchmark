# Config changes for the next run

## Status

Items 1–4c are **applied and committed** (`e835c0d`, `763cb92`, and the same-vlm
swap), and the 11 affected colocations are re-running. Items 5 onward are still
outstanding — they are tuning, not defects, and none of them blocks a run.

## What these fixes are worth

The 2026-08-04 run had **27 failures of 147**. The fixes below do not all
recover data — two of them simply stop a run that was never going to produce
any:

| | Runs | |
|---|---|---|
| **Recovered as measurements** | **17** | memory-pressure (12), SGLang venv (4), `cross-vlm-prefill-vs-llm` (1) |
| Removed from the matrix | 10 | gemma-4 pin-out (8), invalid `python` rung (2) |
| Total failures addressed | 27 | |

Pinning out gemma-4 makes `same-vlm` *skip* rather than fail; it produces no
VLM-pair measurement either way, because `qwen2.5-vl-7b` is the only
video-capable VLM in the roster that serves. That colocation stays unmeasured
until a second one is added — it is a **gap in the study**, not a fixed bug.


Derived from the 2026-08-04 run on 2× RTX PRO 6000. Nothing here has been
applied — the run in flight is deliberately left untouched. Rationale for each
item is in [contention-phases.md](../contention-phases.md#tuning-for-the-next-run).

## 1. ✅ APPLIED — Memory-pressure family: caps → `kv_budget_gb` (12 runs)

`gpu_memory_utilization` is a total-device target and subtracts other processes'
memory, so it cannot apportion a shared card. Measured: `-37.31 GiB`. Replace
the cap on every rung with the KV the rung is named for.

```yaml
cross-memory-pressure-kv03:
  tenants:
    - name: anchor
      kv_budget_gb: 2.0      # replaces gpu_memory_utilization: 0.51
    - name: neighbour
      kv_budget_gb: 1.0      # replaces gpu_memory_utilization: 0.19

cross-memory-pressure-kv13:
  tenants:
    - {name: anchor,    kv_budget_gb: 8.7}    # was 0.58
    - {name: neighbour, kv_budget_gb: 3.9}    # was 0.22

cross-memory-pressure-kv22:
  tenants:
    - {name: anchor,    kv_budget_gb: 14.4}   # was 0.64
    - {name: neighbour, kv_budget_gb: 7.8}    # was 0.26

cross-memory-pressure-kv29:
  tenants:
    - {name: anchor,    kv_budget_gb: 19.2}   # was 0.69
    - {name: neighbour, kv_budget_gb: 9.7}    # was 0.28
```

This sets the swept variable directly instead of inferring it from a cap, so it
also removes the ~3 GiB per-rung error the 72B weights estimate introduced.

## 2. ✅ APPLIED — `weights_gb`: correct the one real outlier

Seven of eight are accurate to within 0.10 GiB. Only the AWQ 72B is wrong:

```yaml
qwen2.5-72b:
  weights_gb: 41.6    # was 45.0; measured 38.77 GiB = 41.6 GB
```

Also rename `weights_gb` or note the unit — the values are GB, vLLM reports GiB,
and comparing them directly makes every estimate look ~7% high.

## 3. ✅ APPLIED — `kv_budget_gb` where KV should be constant (1 run, prevents more)

Both are two-vLLM colocations; `cross-vlm-prefill-vs-llm` already failed with
`No available memory for the cache blocks`.

```yaml
cross-vlm-prefill-vs-llm:   # add kv_budget_gb to both tenants
place-vlm-prefill-split:    # same; survived phase 5 only because tenants were on separate cards
```

## 4. ✅ APPLIED — `same-vlm` swapped to `qwen3-vl-32b-fp8`; gemma-4 pinned out (8 runs)

`transformers` 4.57.6 has no `gemma4` architecture.

```yaml
gemma-4-31b-it-fp8:
  unsupported_backends:
    vllm: "transformers 4.57 has no gemma4 architecture; needs a version bump"
```

Takes `same-vlm` with it — that colocation has no second VLM until either
gemma-4 loads or another video-capable model is added.

## 4b. ✅ APPLIED — `.venv-sglang` was broken; the whole SGLang arm produced no data (4 runs)

`secondary-backend-llm-a/b` sweep `backend: [vllm, sglang]`. Every SGLang rung
fails before the server starts, because `import sglang` itself raises:

```
transformers/integrations/hub_kernels.py:89  LayerRepository(
kernels/layer/layer.py:77  ValueError: Either a revision or a version must be specified.
```

The venv has **transformers 5.6.0 + kernels 0.16.0**, which are mutually
incompatible — transformers constructs a `LayerRepository` with neither a
revision nor a version, which `kernels` 0.16 rejects. `.venv-vllm` is unaffected
because it runs transformers 4.57.6 and has no `kernels` installed at all.

Fix in `.venv-sglang` (either, verify with `python -c "import sglang"`):

```bash
.venv-sglang/bin/pip uninstall -y kernels        # transformers treats it as optional
# or pin transformers to the version the vllm venv uses
.venv-sglang/bin/pip install "transformers==4.57.6"
```

Do this **before** the next run — the failure is at import, so every SGLang
tenant in the matrix is affected, not just phase 6.

## 4c. ✅ APPLIED — `triton_backend: python` is not valid for `yolov8-l` (2 runs)

```yaml
secondary-backend-cv-a:   # extends mix-llm-cv
  vary: {tenant: cv, field: triton_backend, values: [tensorrt, onnx, python]}
secondary-backend-cv-b:   # extends mix-memory-bound
  vary: {tenant: cv, field: triton_backend, values: [tensorrt, onnx, python]}
```

The `python` rung fails with

```
yolov8-l is a python-backend model but has no model.py at
benchmarks/triton_python_models/yolov8-l/model.py
```

Only `kosmos-2.5` has a hand-authored `model.py`; `yolov8-l` runs TensorRT and
ONNX. Either drop `python` from `values` (the backend comparison is still
meaningful with two rungs), or author a `yolov8-l/model.py`. Dropping it is the
smaller change and loses nothing the study asks about — the question is
TensorRT-vs-ONNX cost, and a Python-backend YOLO would be slow for reasons that
have nothing to do with contention.

## 1b. `kv03` is below vLLM's minimum viable KV — the starved end is unreachable

Measured on the re-run: the `kv03` **solo** anchor fails on an empty card at
`_check_enough_kv_cache_memory`. vLLM requires the cache to hold at least one
max-length sequence, and it cannot:

```
qwen2.5-72b: 80 layers, 8 KV heads, head_dim 128
  KV per token     = 320 KiB
  one 8192-tok seq = 2.50 GiB      <- the floor
  kv03 anchor      = 2.00 GiB      <- the rung
```

| Rung | anchor KV | vs 2.50 GiB floor | neighbour KV | vs 0.44 GiB floor |
|---|---|---|---|---|
| `kv03` | 2.0 | **FAIL** | 1.0 | OK |
| `kv13` | 8.7 | OK | 3.9 | OK |
| `kv22` | 14.4 | OK | 7.8 | OK |
| `kv29` | 19.2 | OK | 9.7 | OK |

**This rung never ran as specified, including in the original config.** Under
the old caps the anchor received **6.69 GiB** — more than three times its stated
2.0 GB — because the cap was derived from a `weights_gb` that overstated the 72B
by 3.4 GB, and the surplus silently became cache. Converting to `kv_budget_gb`
made the config finally provision what it claimed, which is what exposed this.

So the eviction cliff the family is hunting sits **below the reachable range on
this card**, at least at 8192 context. Three options, in order of preference:

1. **Move the bottom rung up.** The floor is 2.50 GiB, so the tightest feasible
   2:1 rung is roughly anchor 2.6 / neighbour 1.3 — total 3.9 GB. Rename it
   `kv04`; the curve keeps its shape and its name stays honest.
2. **Shorten the anchor's context.** At `--max-model-len 4096` the floor halves
   to 1.25 GiB and 2.0/1.0 becomes feasible — but context length then differs
   across rungs, which is a second variable.
3. **Drop `kv03`.** The curve starts at `kv13`; the starved end goes unmeasured
   and is documented as out of reach.

Do not simply raise the budget and keep the `kv03` name: the suffix is the
swept variable, and a rung named for 3 GB that provisions 3.9 is the same class
of error as the caps that provisioned 6.69.

## 5. Rates: the configured load is far below capacity

| Colocation | Current sweep | Top rung reaches | Suggested |
|---|---|---|---|
| `same-cv` | `[1, 10, 50, 200]` | 33% utilisation | `[50, 200, 600, 1500]` |
| `cross-llm-vs-cv-rps` | `[1, 10, 50, 200]` | 33% | same |
| `cross-ilm-vs-cv` | `[1, 10, 50, 200]` | 33% | same |
| `secondary-asymmetry-a/b` | `[4, 16, 64]` | ~10% | scale to the CV ceiling |
| `same-llm` | `[1, 4, 16, 64]` | crosses the knee | add a **32** rung |

Confirm `perf_analyzer` is not itself the bottleneck before exceeding ~600 req/s:
run the tenant solo at the intended top rate and check `achieved` tracks `offered`.

## 5b. `same-vlm`'s sweep asks for 12x what the tenant can serve

Measured on the first re-run: `qwen3-vl-32b-fp8` on `vlm_video_long` plateaus at
**~0.33 req/s**, and its weights are 33.64 GiB (against the 35.5 GB / 33.1 GiB
declared, so the cap is sound).

| Offered | Achieved |
|---|---|
| 0.5 | 0.32 |
| 1.0 | 0.33 |

`same-vlm` sweeps `[0.5, 1, 2, 4]` on both tenants. Three of those four rungs
sit above the ceiling, so they all deliver ~0.33 and the sweep collapses into
one load point measured four times. The degradation ratios stay valid — baseline
and contention share the ceiling — but the rungs stop being a sweep.

**The pair needs per-tenant rates — a `"*"` sweep cannot serve both.** The solo
baselines from this run settle it:

| Offered | `qwen2.5-vl-7b` | `qwen3-vl-32b-fp8` |
|---|---|---|
| 0.5 | 0.49 | 0.32 |
| 1.0 | 1.00 | 0.33 |
| 2.0 | 1.95 | — |
| 4.0 | 3.88 | — |

The 7B tracks its offered rate to 4 req/s; the 32B is pinned at ~0.33 from 0.5
upward. That is a **12x** spread at the top rung, so any single sweep either
saturates the 32B or idles the 7B. Lowering the whole sweep to the 32B's ceiling
(the obvious fix, and what an earlier draft of this section recommended) would
run the 7B at under a tenth of its capacity and measure nothing.

Replace the `"*"` sweep with per-tenant rates, each scaled to its own ceiling —
the same fraction-of-capacity idea as §5, applied within one colocation:

```yaml
same-vlm:
  # no rps_sweep: "*" — the tenants are 12x apart
  tenants:
    - name: vlm_a   # ceiling ~4+ req/s
      load: {pattern: poisson, rps: 2}
    - name: vlm_b   # ceiling ~0.33 req/s
      load: {pattern: poisson, rps: 0.2}
```

The asymmetry is already visible in the first contention window: `vlm_a` held at
0.48 against its 0.49 solo while `vlm_b` fell 0.32 → 0.30. The 7B is barely
touched and the 32B absorbs the contention — which is a finding, but not one
this colocation was designed to make.

## 6. `same-llm`: add an asymmetric arm

Both tenants currently run at the same rate. The common production shape — one
saturated tenant beside an idle one — is not measured. Add one tenant at 4 req/s
against one at 64.

## 7. `duration_s` must suit the highest rung that inherits it

`cross-ilm-vs-cv` inherits `duration_s: 600` (chosen for kosmos at 0.1 req/s =
60 requests); at its 200 req/s CV rung that asks for 120,000 requests. Set
`duration_s` explicitly on sweeping colocations, or scale it per rung.

## 8. Repetitions at low utilisation

`same-cv`'s `dinov2-base` reads 1.55× at 10 req/s and 1.05× at 50 — non-monotonic,
so at least one is noise. Either raise the rates (preferred, see §5) or raise
`repetitions` on the low rungs.

## Instrumentation, not config

- **Record TTFT in the coloc summary.** It is the leading indicator of the
  `same-llm` cliff — ITL stays under 2× while TTFT goes 348×. Currently it has to
  be dug out of the aiperf exports by hand.
- **Pinning KV removes the log line that reports it.** vLLM prints
  `Available KV cache memory: X GiB` only when it *derives* the cache from a
  cap. With `--kv-cache-memory-bytes` set it prints nothing, so the ground-truth
  cache size is no longer recoverable from the log for exactly the tenants whose
  cache we care most about. Record `kv_budget_gb` into the manifest (it already
  is) and treat the log line as unavailable for pinned tenants.
- **Parse `Model loading took X GiB` from the server logs** into the manifest.
  It is the `vram_after_load_gb` ground truth Tier 2 wants, and it is already
  being written — just not collected.
- **Give each driver its own timeout.** `ColocationOrchestrator.run` shares one
  deadline across tenants, so a hung driver can consume a healthy sibling's budget.
