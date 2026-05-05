# Live local-backend smoke tests

These run the three game scenarios end-to-end against a real model server
(vLLM / SGLang / TRT-LLM). They exercise the pipeline wiring **plus** the
OpenAI-compatible adapter **plus** the model itself on actual game
screens — not just gold stubs covered by the offline smoke tests.

> Prerequisite: backends installed per
> [INFERENCE_BACKENDS.md](INFERENCE_BACKENDS.md) Mode B (and engine
> built for TRT-LLM).

## What the tests assert

For each (backend × scenario):
- Pipeline runs end-to-end without crashing.
- `latency.total_ms > 0` (timing was recorded).
- `model_meta.framework` matches the backend (`vllm` / `sglang` /
  `trtllm`).

The tests do **not** assert the model produced the gold action sequence —
real model output varies and accuracy belongs in the benchmark harness,
not smoke. To eyeball actual vs. gold on one scenario use
`python -m examples.run_scenario` (see end of this doc).

## Common pattern

Two shells per backend — one runs the server, one runs the test.

```
Shell 1: source .venv-<framework>/bin/activate && <serve command>
Shell 2: source .venv-<framework>/bin/activate && pytest -m <framework> tests/smoke/test_local_backends.py -v
```

If the server isn't reachable, the parametrised group is **skipped with
a clear reason** rather than failing — so you can run the whole file
against whatever happens to be up:

```bash
pytest tests/smoke/test_local_backends.py -v
```

## B.1 — vLLM (port 8000)

**Shell 1 — server:**

```bash
source .venv-vllm/bin/activate
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --port 8000 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 32
# Wait for: "Application startup complete."
```

**Shell 2 — tests:**

```bash
source .venv-vllm/bin/activate
pytest -m vllm tests/smoke/test_local_backends.py -v
```

**Overrides:**

```bash
VLLM_BASE_URL=http://localhost:9000/v1 VLLM_MODEL=Qwen/Qwen2-VL-7B-Instruct \
  pytest -m vllm tests/smoke/test_local_backends.py -v
```

If `VLLM_MODEL` isn't set the reasoner auto-discovers it from
`/v1/models` — usually correct, but pin it explicitly when the server
loads more than one model.

## B.2 — SGLang (port 30000)

**Shell 1 — server:**

```bash
source .venv-sglang/bin/activate
sglang serve --model-path Qwen/Qwen3-VL-8B-Instruct --port 30000
# Wait for: "The server is fired up and ready to roll!"
```

**Shell 2 — tests:**

```bash
source .venv-sglang/bin/activate
pytest -m sglang tests/smoke/test_local_backends.py -v
```

**Overrides:** `SGLANG_BASE_URL`, `SGLANG_MODEL`.

> The smoke test uses plain JSON-mode. SGLang's structured-output
> features (regex / EBNF / JSON-schema) belong in the benchmark
> harness — see [docs/frameworks.md](docs/frameworks.md).

## B.3 — TRT-LLM (port 8002)

`trtllm-serve` exposes the same OpenAI-compatible API as vLLM/SGLang.
The smoke test defaults to **port 8002** so it doesn't clash with vLLM
(8000) when all three are profiled on the same host.

**Shell 1 — server:**

```bash
source .venv-trtllm/bin/activate

# Engine + tokenizer paths from INFERENCE_BACKENDS.md B.3.
trtllm-serve trt_engines/qwen2-vl-7b-rtx_pro6000-bf16/llm \
  --tokenizer ./hf_models/qwen2-vl-7b \
  --port 8002 \
  --backend pytorch
# Wait for: "Uvicorn running on http://0.0.0.0:8002"
```

**Shell 2 — tests:**

```bash
source .venv-trtllm/bin/activate
pytest -m trtllm tests/smoke/test_local_backends.py -v
```

**Overrides:** `TRTLLM_BASE_URL`.

## Run a single scenario interactively

For eyeballing actual-vs-gold on one scenario (prints both action
sequences side by side, plus per-stage latency):

```bash
python -m examples.run_scenario 01_clash_of_clans_start_attack --backend vllm
python -m examples.run_scenario 02_catan_open_menu             --backend sglang
python -m examples.run_scenario 03_fps_engage_and_reload       --backend trtllm

python -m examples.run_scenario --list   # all available scenarios
```

## Sweep everything that's up

```bash
pytest tests/smoke/test_local_backends.py -v
```

Backends whose server isn't running are skipped, not failed. Useful when
multiple servers are running in parallel on the same host.

## Common failures

| Symptom | Cause / fix |
| --- | --- |
| All three tests for a backend skipped | Server isn't reachable on the expected port. Verify with `curl http://localhost:<port>/v1/models`. |
| `BadRequest: response_format json_object not supported` | vLLM/SGLang version too old. Upgrade to a recent build (vLLM ≥ 0.6, SGLang ≥ 0.4). |
| Long timeout on the first request | Cold KV cache + vision-encoder warmup. Re-run; subsequent requests are normal. |
| `Connection reset` mid-stream | Server OOM-killed — reduce `--gpu-memory-utilization` (vLLM) or `--mem-fraction-static` (SGLang), or pick a smaller checkpoint. |
| `model 'X' not found` | Auto-discovery picked the wrong id. Set `VLLM_MODEL` / `SGLANG_MODEL` explicitly to the id printed at server startup. |
