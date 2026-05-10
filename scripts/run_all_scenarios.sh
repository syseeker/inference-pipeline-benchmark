#!/usr/bin/env bash
# One-trigger scenario sweep across vLLM / SGLang / TRT-LLM.
#
# Per-backend launch parameters come from benchmarks/configs/<gpu>.yaml —
# see BENCHMARK_GUIDE.md and docs/models.md. This script only knows the
# *shape* of each command; everything else is yaml-driven.
#
# For each (backend × round) — a "round" is a (model, variant) pair from a
# sweep, or the default model with the chosen --variant for a one-off:
#   1. activate the matching venv
#   2. start the inference server in the background, with launch args
#      composed by benchmarks.scenario_config (extra_args + variant +
#      model.backend_args)
#   3. wait for /v1/models to return 200 (or time out)
#   4. run benchmarks.runner — writes per-scenario JSON + one aggregate
#      BenchmarkResult JSON under benchmarks/results/<gpu>/
#   5. stop the server cleanly
#
# After every backend, runs benchmarks.summary to produce
# benchmarks/results/<gpu>/summary.md.
#
# Usage:
#   scripts/run_all_scenarios.sh                                # default model on all 3 backends
#   scripts/run_all_scenarios.sh --gpu rtx5090                  # different GPU profile
#   scripts/run_all_scenarios.sh --backends "vllm sglang"       # subset of backends
#   scripts/run_all_scenarios.sh --model qwen3.6-27b-fp8        # one-off model override
#   scripts/run_all_scenarios.sh --variants "baseline eager"    # vllm-only knob comparison
#   scripts/run_all_scenarios.sh --sweep models                 # auto-iterate yaml `sweeps.models`
#   scripts/run_all_scenarios.sh --scenarios-dir my/scenarios   # custom scenario folder
#
# Mode selection (mutually exclusive):
#   --sweep <name>                 — iterate yaml `sweeps:<name>` rounds.
#   --variants "v1 v2 ..."         — for each variant, run all backends
#                                    on the default (or --model) model.
#                                    "baseline" = no variant (the default).
#   (neither)                      — single round on each backend with the
#                                    default model.
#
# Env overrides:
#   READY_TIMEOUT_S (default 600) — server-readiness wait. Models with
#                                   slow cold-cache loads can override per-
#                                   model via yaml `models.<id>.ready_timeout_s`.

set -euo pipefail

BACKENDS="vllm sglang trtllm"
GPU="rtx_pro6000"
SCENARIOS_DIR=""
MODEL=""
VARIANTS=""
SWEEP=""
READY_TIMEOUT_S="${READY_TIMEOUT_S:-600}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backends) BACKENDS="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --scenarios-dir) SCENARIOS_DIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --variants) VARIANTS="$2"; shift 2 ;;
    --sweep) SWEEP="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,42p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -n "$SWEEP" && -n "$VARIANTS" ]]; then
  echo "!! --sweep and --variants are mutually exclusive" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="benchmarks/results/${GPU}/server-logs"
mkdir -p "$LOG_DIR"

CFG_PATH="benchmarks/configs/${GPU}.yaml"
if [[ ! -f "$CFG_PATH" ]]; then
  echo "!! missing GPU config: $CFG_PATH" >&2
  exit 2
fi

SERVER_PID=""

# Recovery instructions printed by the failure handlers below. Centralised so
# every failure path teaches the same incantations.
print_kill_recipe() {
  cat >&2 <<'EOF'

   Recovery — kill any orphan inference servers and worker processes:
     pkill -f 'vllm serve|sglang.launch_server|trtllm-serve|tensorrt_llm' ; sleep 2
     pkill -9 -f 'vllm serve|sglang.launch_server|trtllm-serve|tensorrt_llm'

   Verify the GPU is empty (only the header line should print):
     nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

   Verify the port is free (replace 8000/30000/8002 as needed):
     ss -ltnp | grep -E ':(8000|30000|8002)' || echo "all ports free"

   Then rerun this script.
EOF
}

# Pre-flight 1: port already bound? Abort early with a clear error so the
# script doesn't quietly attach to a leftover server (as happened on
# 2026-05-09 when a TRT-LLM orphan was holding port 8002).
preflight_port() {
  local port="$1" backend="$2"
  if ss -ltnp 2>/dev/null | grep -qE ":${port}\b"; then
    echo "!! port ${port} already in use — refusing to start ${backend}" >&2
    echo "   listener:" >&2
    ss -ltnp 2>/dev/null | grep -E ":${port}\b" | sed 's/^/     /' >&2
    print_kill_recipe
    return 1
  fi
  return 0
}

# Pre-flight 2: enough free VRAM to plausibly load a VLM? Threshold is
# deliberately conservative — the actual loader will fail with an exact
# number if this passes but the model is bigger.
preflight_gpu() {
  local backend="$1"
  local free_mib min_free_mib=30000
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [[ -z "$free_mib" ]]; then
    echo ">> WARNING: nvidia-smi unavailable; skipping GPU pre-flight for ${backend}" >&2
    return 0
  fi
  if (( free_mib < min_free_mib )); then
    echo "!! GPU has only ${free_mib} MiB free (need >= ${min_free_mib} MiB to start ${backend})" >&2
    echo "   compute processes currently holding VRAM:" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 \
      | sed 's/^/     /' >&2
    print_kill_recipe
    return 1
  fi
  return 0
}

# Post-failure: scan the server log for known fingerprints and emit a
# focused diagnosis. Falls through to printing the last 30 lines if nothing
# matches.
diagnose_log() {
  local backend="$1" log_file="$2" port="${3:-?}"
  [[ -f "$log_file" ]] || { echo "   (no log at $log_file)" >&2; return; }
  local tail_text
  tail_text=$(tail -n 300 "$log_file")

  if echo "$tail_text" | grep -qE "Free memory on device cuda|CUDA out of memory|cudaErrorMemoryAllocation|torch\.OutOfMemoryError|Out of memory"; then
    echo "!! ${backend} died: GPU OOM" >&2
    echo "$tail_text" \
      | grep -E "Free memory on device cuda|CUDA out of memory|torch\.OutOfMemoryError|Out of memory" \
      | tail -1 | sed 's/^/   /' >&2
    print_kill_recipe
    return
  fi

  if echo "$tail_text" | grep -qiE "address already in use|EADDRINUSE|bind.*error.*98"; then
    echo "!! ${backend} died: port ${port} already in use" >&2
    echo "   Another process is on the target port (most likely an orphan from a previous run)." >&2
    print_kill_recipe
    return
  fi

  # `Killed` on its own line OR a python wrapper "signal 9" trace = OOM-killer.
  if echo "$tail_text" | grep -qE "(^|[^[:alnum:]])Killed$|signal 9|SIGKILL"; then
    echo "!! ${backend} was killed (signal 9) — likely host-RAM OOM by the kernel" >&2
    echo "   Free RAM was insufficient. Confirm with: dmesg | tail -50  (look for 'Out of memory: Killed process')." >&2
    print_kill_recipe
    return
  fi

  echo "!! ${backend} died with unrecognised error; last 30 lines of ${log_file}:" >&2
  tail -n 30 "$log_file" | sed 's/^/     /' >&2
}

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo ">> stopping server pid=$SERVER_PID"
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup EXIT INT TERM

# Resolve a single field from the new schema. Fields:
#   hf_id family quantization base_url port trtllm_backend variant
# Lists:
#   launch_args (pass --list)
cfg_field() {
  local backend="$1" field="$2" model="${3:-}" variant="${4:-}"
  local args=(--gpu "$GPU" --backend "$backend" --field "$field")
  [[ -n "$model" ]] && args+=(--model "$model")
  [[ -n "$variant" && "$variant" != "baseline" ]] && args+=(--variant "$variant")
  python -m benchmarks.scenario_config "${args[@]}"
}

cfg_launch_args() {
  local -n _out="$1"
  local backend="$2" model="${3:-}" variant="${4:-}"
  _out=()
  local args=(--gpu "$GPU" --backend "$backend" --field launch_args --list)
  [[ -n "$model" ]] && args+=(--model "$model")
  [[ -n "$variant" && "$variant" != "baseline" ]] && args+=(--variant "$variant")
  while IFS= read -r line; do
    [[ -n "$line" ]] && _out+=("$line")
  done < <(python -m benchmarks.scenario_config "${args[@]}")
}

# rc 0 if `backends.<backend>.variants.<variant>` exists. "baseline" is
# treated as "always exists" (= no variant).
cfg_has_variant() {
  local backend="$1" variant="$2"
  if [[ -z "$variant" || "$variant" == "baseline" ]]; then
    return 0
  fi
  python -m benchmarks.scenario_config \
    --gpu "$GPU" --backend "$backend" --has-variant "$variant"
}

wait_for_ready() {
  local url="$1" label="$2" timeout_s="${3:-$READY_TIMEOUT_S}"
  echo ">> waiting for ${label} at ${url} (timeout ${timeout_s}s)"
  local elapsed=0
  while (( elapsed < timeout_s )); do
    if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "!! ${label} server died before ready (see log)" >&2
      return 1
    fi
    if curl -fsS --max-time 3 "$url" >/dev/null 2>&1; then
      echo ">> ${label} is ready after ${elapsed}s"
      return 0
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
  echo "!! ${label} did not become ready within ${timeout_s}s" >&2
  return 1
}

# Start the server for (backend, model, variant). Sets SERVER_PID and WAIT_URL.
start_server() {
  local backend="$1" model="${2:-}" variant="${3:-}"
  local hf_id port
  local -a launch
  hf_id=$(cfg_field "$backend" hf_id "$model" "$variant")
  port=$(cfg_field "$backend" port "$model" "$variant")
  cfg_launch_args launch "$backend" "$model" "$variant"

  # Always close stdin on backgrounded servers (`< /dev/null`). Without
  # this, the engine's child workers inherit the script's stdin and can
  # silently drain whatever's piped in (e.g. sweep rounds via process
  # substitution), making the outer while-loop exit early with no error.
  case "$backend" in
    vllm)
      echo ">> vllm serve $hf_id --port $port ${launch[*]:-}"
      vllm serve "$hf_id" --port "$port" "${launch[@]}" \
        < /dev/null >> "$LOG_DIR/vllm.log" 2>&1 &
      ;;
    sglang)
      echo ">> python -m sglang.launch_server --model-path $hf_id --port $port ${launch[*]:-}"
      python -m sglang.launch_server \
        --model-path "$hf_id" --port "$port" "${launch[@]}" \
        < /dev/null >> "$LOG_DIR/sglang.log" 2>&1 &
      ;;
    trtllm)
      # Translate yaml `backend:` → trtllm-serve `--backend` (yaml uses `trtllm`
      # for the engine path; trtllm-serve calls it `tensorrt`).
      local trt_backend
      trt_backend=$(cfg_field trtllm trtllm_backend "$model" "$variant")
      [[ "$trt_backend" == "trtllm" ]] && trt_backend="tensorrt"
      echo ">> trtllm-serve $hf_id --backend $trt_backend --port $port ${launch[*]:-}"
      trtllm-serve "$hf_id" \
        --backend "$trt_backend" --port "$port" "${launch[@]}" \
        < /dev/null >> "$LOG_DIR/trtllm.log" 2>&1 &
      ;;
    *)
      echo "!! unknown backend: $backend" >&2
      return 1 ;;
  esac
  SERVER_PID=$!
  WAIT_URL="http://localhost:${port}/v1/models"
}

# Probe whether (backend, model) is hardware/version-incompatible per the
# YAML's `unsupported_backends:` field. Echoes the reason on stdout (empty
# = supported). Always rc 0.
cfg_unsupported_reason() {
  local backend="$1" model="${2:-}"
  local args=(--gpu "$GPU" --backend "$backend" --unsupported-reason)
  [[ -n "$model" ]] && args+=(--model "$model")
  python -m benchmarks.scenario_config "${args[@]}" 2>/dev/null || true
}

# Run one (backend, model, variant) round end-to-end.
run_round() {
  local backend="$1" model="${2:-}" variant="${3:-}"
  local venv=".venv-${backend}"
  if [[ ! -d "$venv" ]]; then
    echo "!! skipping ${backend} [${variant:-baseline}]: ${venv} missing" >&2
    return 0
  fi
  if ! cfg_has_variant "$backend" "$variant"; then
    echo ">> skipping ${backend} [${variant}]: variant not defined for this backend"
    return 0
  fi
  # Hardware/version incompatibility check — covers both single-round
  # (`--backends X --model Y`) and sweep-level "you happened to ask for
  # an unsupported combo via --variants" invocations. Prints the reason
  # and skips cleanly (rc 0, not a failure).
  local unsupported_reason
  unsupported_reason=$(cfg_unsupported_reason "$backend" "$model")
  if [[ -n "$unsupported_reason" ]]; then
    echo ">> skipping ${backend}/${model:-<default>}: ${unsupported_reason}"
    return 0
  fi

  echo
  echo "================================================================"
  echo " backend: ${backend}  model: ${model:-<default>}  variant: ${variant:-baseline}"
  echo "================================================================"

  # shellcheck source=/dev/null
  source "${venv}/bin/activate"

  # Pre-flight: port + GPU memory. Fail loud, not silent.
  local port
  port=$(cfg_field "$backend" port "$model" "$variant")
  if ! preflight_port "$port" "$backend"; then
    deactivate
    return 1
  fi
  if ! preflight_gpu "$backend"; then
    deactivate
    return 1
  fi

  WAIT_URL=""
  if ! start_server "$backend" "$model" "$variant"; then
    deactivate
    return 1
  fi

  # Per-model override (yaml `models.<id>.ready_timeout_s`); empty = use default.
  local round_timeout
  round_timeout=$(cfg_field "$backend" ready_timeout_s "$model" "$variant")
  : "${round_timeout:=$READY_TIMEOUT_S}"

  echo ">> ${backend} server pid=${SERVER_PID} (log: $LOG_DIR/${backend}.log)"
  if ! wait_for_ready "$WAIT_URL" "$backend" "$round_timeout"; then
    diagnose_log "$backend" "$LOG_DIR/${backend}.log" "$port"
    cleanup
    deactivate
    return 1
  fi

  set +e
  local args=(--backend "$backend" --gpu "$GPU")
  [[ -n "$model" ]]                       && args+=(--model "$model")
  [[ -n "$variant" && "$variant" != "baseline" ]] && args+=(--variant "$variant")
  [[ -n "$SCENARIOS_DIR" ]]               && args+=(--scenarios-dir "$SCENARIOS_DIR")
  python -m benchmarks.runner "${args[@]}"
  local rc=$?
  set -e

  cleanup
  deactivate

  if (( rc != 0 )); then
    echo "!! ${backend} runner exited with rc=${rc}" >&2
    return $rc
  fi
}

failed=()

# Fail-fast: the first run_round failure halts the script. When the cause
# is global (orphan VRAM, shared port), repeating the same diagnosis for
# every remaining backend is noise — the user has to fix the underlying
# issue before any backend can succeed.
if [[ -n "$SWEEP" ]]; then
  # Sweep mode: iterate the yaml's sweep rounds, run each on its named backends.
  echo ">> sweep '$SWEEP' on $GPU"
  # Read all rounds into an array up front rather than streaming via
  # `done < <(...)`. Process substitution would attach the FIFO to the
  # loop's stdin, and any backgrounded server inside run_round would
  # inherit that fd and silently drain it — so only the first 1-3 rounds
  # ran before the loop saw EOF. mapfile dodges the inheritance entirely.
  declare -a SWEEP_ROUNDS
  mapfile -t SWEEP_ROUNDS < <(python -m benchmarks.scenario_config --gpu "$GPU" --emit-rounds "$SWEEP")
  echo ">> sweep '$SWEEP' resolved to ${#SWEEP_ROUNDS[@]} round(s)"
  for round_json in "${SWEEP_ROUNDS[@]}"; do
    [[ -z "$round_json" ]] && continue
    backend=$(printf '%s' "$round_json" | python -c 'import json,sys;d=json.load(sys.stdin);print(d["backend"])')
    model=$(printf '%s'   "$round_json" | python -c 'import json,sys;d=json.load(sys.stdin);print(d["model_id"])')
    variant=$(printf '%s' "$round_json" | python -c 'import json,sys;d=json.load(sys.stdin);print(d.get("variant") or "")')
    # Honour --backends filter even inside a sweep.
    case " $BACKENDS " in
      *" $backend "*) ;;
      *) echo ">> skipping sweep round (backend $backend not in --backends)"; continue ;;
    esac
    if ! run_round "$backend" "$model" "$variant"; then
      failed+=("${backend}/${model}/${variant:-baseline}")
      echo "!! halting at first failure (sweep round: ${backend}/${model}/${variant:-baseline})" >&2
      break
    fi
  done
else
  # Variant / single-round mode.
  variant_list="${VARIANTS:-baseline}"
  for backend in $BACKENDS; do
    for variant in $variant_list; do
      if ! run_round "$backend" "$MODEL" "$variant"; then
        failed+=("${backend}/${MODEL:-<default>}/${variant}")
        echo "!! halting at first failure (backend=${backend} variant=${variant})" >&2
        break 2
      fi
    done
  done
fi

# Generate summary regardless of partial failures.
echo
echo ">> generating summary.md for ${GPU}"
for v in vllm sglang trtllm; do
  if [[ -d ".venv-${v}" ]]; then
    # shellcheck source=/dev/null
    source ".venv-${v}/bin/activate"
    python -m benchmarks.summary --gpu "$GPU"
    deactivate
    break
  fi
done

if (( ${#failed[@]} > 0 )); then
  echo
  echo "!! rounds with failures: ${failed[*]}" >&2
  exit 1
fi

echo
echo ">> done. summary at benchmarks/results/${GPU}/summary.md"
