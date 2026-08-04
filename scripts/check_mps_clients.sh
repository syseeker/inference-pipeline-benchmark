#!/usr/bin/env bash
# Are the processes on the GPU actually MPS clients?
#
# The question this answers: a tenant that cannot reach MPS still runs, still
# produces plausible numbers, and still shows up in nvidia-smi — it just
# time-slices instead of sharing the SMs, which makes its degradation ratio
# meaningless. Nothing else in the harness catches that at run time.
#
# Do NOT try to read this off `nvidia-smi` alone: on Volta and later every MPS
# client keeps its own address space and lists as its own process, so separate
# `vllm` and `tritonserver` entries are what a correctly shared GPU looks like.
# The MPS control socket is the only authority.
#
# Usage: scripts/check_mps_clients.sh
# Run it from a SECOND terminal while a contention window is live.

set -uo pipefail

mps() { echo "$1" | nvidia-cuda-mps-control 2>/dev/null; }

# --- GPU processes, excluding the MPS server itself (it is not a tenant) -----
mapfile -t gpu_procs < <(
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null \
    | sed 's/^ *//' | sort -u | grep -v nvidia-cuda-mps-server
)

if ! pgrep -f nvidia-cuda-mps-control >/dev/null; then
  echo "FAIL: no MPS control daemon. Tenants will time-slice — see Step 4."
  exit 1
fi

servers=$(mps get_server_list)

if [[ -z "$servers" ]]; then
  if [[ ${#gpu_procs[@]} -eq 0 ]]; then
    echo "IDLE: no MPS server and nothing on the GPU."
    echo "      The server starts on first client and exits with the last one,"
    echo "      so this is what BETWEEN windows looks like. Re-run once the"
    echo "      contention window starts."
    exit 0
  fi
  echo "FAIL: ${#gpu_procs[@]} process(es) on the GPU and no MPS server —"
  echo "      none of them is an MPS client."
  printf '        %s\n' "${gpu_procs[@]}"
  exit 1
fi

# --- collect every client PID across every server ---------------------------
clients=()
for s in $servers; do
  while read -r pid; do
    [[ -n "$pid" ]] && clients+=("$pid")
  done < <(mps "get_client_list $s")
done

echo "MPS server(s): $servers"
if [[ ${#clients[@]} -eq 0 ]]; then
  echo "  (no clients connected right now)"
else
  for pid in "${clients[@]}"; do
    printf '  client %-8s %s\n' "$pid" "$(ps -p "$pid" -o comm= 2>/dev/null || echo '<exited>')"
  done
fi

# --- the actual verdict: is any GPU process NOT a client? -------------------
echo
orphans=0
for row in "${gpu_procs[@]}"; do
  pid=${row%%,*}
  pid=${pid// /}
  found=0
  for c in "${clients[@]:-}"; do [[ "$c" == "$pid" ]] && found=1; done
  if [[ $found -eq 0 ]]; then
    echo "  NOT an MPS client: $row"
    orphans=$((orphans + 1))
  fi
done

if [[ ${#gpu_procs[@]} -eq 0 ]]; then
  echo "IDLE: MPS server is up but nothing is on the GPU yet."
  exit 0
elif [[ $orphans -gt 0 ]]; then
  echo "FAIL: $orphans of ${#gpu_procs[@]} GPU process(es) are outside MPS."
  echo "      That tenant time-slices; this window's ratios are void."
  echo "      Check CUDA_MPS_PIPE_DIRECTORY is exported (Step 4)."
  exit 1
fi

echo "PASS: all ${#gpu_procs[@]} GPU process(es) are MPS clients."
[[ ${#gpu_procs[@]} -lt 2 ]] && echo "      Only one tenant — this is a solo baseline, not the contention window."
exit 0
