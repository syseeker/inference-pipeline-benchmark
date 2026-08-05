# Config changes for the next run

Derived from the 2026-08-04 run on 2× RTX PRO 6000. Nothing here has been
applied — the run in flight is deliberately left untouched. Rationale for each
item is in [contention-phases.md](../contention-phases.md#tuning-for-the-next-run).

## 1. Memory-pressure family: caps → `kv_budget_gb` (recovers 12 runs)

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

## 2. `weights_gb`: correct the one real outlier

Seven of eight are accurate to within 0.10 GiB. Only the AWQ 72B is wrong:

```yaml
qwen2.5-72b:
  weights_gb: 41.6    # was 45.0; measured 38.77 GiB = 41.6 GB
```

Also rename `weights_gb` or note the unit — the values are GB, vLLM reports GiB,
and comparing them directly makes every estimate look ~7% high.

## 3. Add `kv_budget_gb` where KV should be constant (recovers 2+ runs)

Both are two-vLLM colocations; `cross-vlm-prefill-vs-llm` already failed with
`No available memory for the cache blocks`.

```yaml
cross-vlm-prefill-vs-llm:   # add kv_budget_gb to both tenants
place-vlm-prefill-split:    # same; survived phase 5 only because tenants were on separate cards
```

## 4. Pin out `gemma-4-31b-it-fp8` (recovers 8 runs)

`transformers` 4.57.6 has no `gemma4` architecture.

```yaml
gemma-4-31b-it-fp8:
  unsupported_backends:
    vllm: "transformers 4.57 has no gemma4 architecture; needs a version bump"
```

Takes `same-vlm` with it — that colocation has no second VLM until either
gemma-4 loads or another video-capable model is added.

## 4b. `.venv-sglang` is broken — the whole SGLang arm produces no data (4 runs)

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

## 4c. `triton_backend: python` is not valid for `yolov8-l` (2 runs)

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
- **Parse `Model loading took X GiB` from the server logs** into the manifest.
  It is the `vram_after_load_gb` ground truth Tier 2 wants, and it is already
  being written — just not collected.
- **Give each driver its own timeout.** `ColocationOrchestrator.run` shares one
  deadline across tenants, so a hung driver can consume a healthy sibling's budget.
