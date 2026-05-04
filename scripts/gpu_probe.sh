#!/usr/bin/env bash
# Captures GPU/driver/toolchain versions so each benchmark result row is
# self-describing. Output is JSON for easy ingestion.
set -euo pipefail

out="${1:-benchmarks/results/host_$(hostname).json}"
mkdir -p "$(dirname "$out")"

driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || echo unknown)"
gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo unknown)"
cuda="$(nvcc --version 2>/dev/null | tail -n1 | sed 's/.*release //; s/,.*//' || echo unknown)"

python_v="$(python -V 2>&1 | awk '{print $2}')"
vllm_v="$(python -c 'import vllm,sys;print(vllm.__version__)' 2>/dev/null || echo not-installed)"
sglang_v="$(python -c 'import sglang,sys;print(getattr(sglang,"__version__","unknown"))' 2>/dev/null || echo not-installed)"
trtllm_v="$(python -c 'import tensorrt_llm,sys;print(getattr(tensorrt_llm,"__version__","unknown"))' 2>/dev/null || echo not-installed)"
modelopt_v="$(python -c 'import modelopt,sys;print(getattr(modelopt,"__version__","unknown"))' 2>/dev/null || echo not-installed)"
triton_v="$(python -c 'import tritonclient,sys;print(getattr(tritonclient,"__version__","unknown"))' 2>/dev/null || echo not-installed)"

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
