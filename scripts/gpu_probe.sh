#!/usr/bin/env bash
# Captures GPU/driver/toolchain versions so each benchmark result row is
# self-describing. Output is JSON for easy ingestion.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:-${REPO_ROOT}/benchmarks/results/host_$(hostname).json}"
mkdir -p "$(dirname "$out")"

driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || echo unknown)"
gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo unknown)"
# `nvcc --version` prints the version on the line that starts with
# "Cuda compilation tools, release X.Y, V...". `tail -n1` grabs the wrong
# line ("Build cuda_X.Y/..."), so we grep the release line directly.
cuda="$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -n1 || echo unknown)"
[ -z "$cuda" ] && cuda="unknown"

python_v="$(python3 -V 2>&1 | awk '{print $2}' || echo unknown)"

# Probe each backend's dedicated venv first (matches INFERENCE_BACKENDS.md
# layout: .venv-vllm / .venv-sglang / .venv-trtllm). Fall back to whatever
# python3 is on PATH so single-venv hosts still work.
probe_pkg() {
  # $1 = import name, $2 = preferred venv dir
  # Some packages (e.g. tensorrt_llm) print a banner on import; we take
  # only the last non-empty line of stdout, which is our explicit print().
  local pkg="$1" venv="$2" py="" version_expr out
  version_expr="import ${pkg}; print(getattr(${pkg}, '__version__', 'unknown'))"
  if [ -x "${REPO_ROOT}/${venv}/bin/python" ]; then
    py="${REPO_ROOT}/${venv}/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    py="python3"
  else
    echo "not-installed"
    return
  fi
  if out="$("$py" -c "$version_expr" 2>/dev/null)"; then
    printf '%s\n' "$out" | awk 'NF' | tail -n1
  else
    echo "not-installed"
  fi
}

vllm_v="$(probe_pkg vllm .venv-vllm)"
sglang_v="$(probe_pkg sglang .venv-sglang)"
trtllm_v="$(probe_pkg tensorrt_llm .venv-trtllm)"
# modelopt + tritonclient ship inside the trtllm venv in this layout.
modelopt_v="$(probe_pkg modelopt .venv-trtllm)"
triton_v="$(probe_pkg tritonclient .venv-trtllm)"

cat > "$out" <<JSON
{
  "host": "$(hostname)",
  "gpu": "$gpu",
  "driver": "$driver",
  "cuda": "$cuda",
  "python": "$python_v",
  "vllm": "$vllm_v",
  "sglang": "$sglang_v",
  "tensorrt_llm": "$trtllm_v",
  "modelopt": "$modelopt_v",
  "tritonclient": "$triton_v"
}
JSON

echo "wrote $out"
