#!/usr/bin/env bash
# List the model ids served at NIM_BASE_URL with the current NIM_API_KEY.
# Run this whenever you hit a 404 — the catalogue rotates.
#
#   bash scripts/list_nim_models.sh             # all models
#   bash scripts/list_nim_models.sh qwen        # filter by substring
#   bash scripts/list_nim_models.sh -v          # also include json metadata
set -euo pipefail

base_url="${NIM_BASE_URL:-https://integrate.api.nvidia.com/v1}"

if [[ -z "${NIM_API_KEY:-}" ]]; then
  echo "NIM_API_KEY is not set in the environment." >&2
  exit 2
fi

verbose=0
filter=""
for arg in "$@"; do
  case "$arg" in
    -v|--verbose) verbose=1 ;;
    *) filter="$arg" ;;
  esac
done

resp="$(curl -sS -H "Authorization: Bearer $NIM_API_KEY" "${base_url%/}/models")"

if [[ "$verbose" -eq 1 ]]; then
  if [[ -n "$filter" ]]; then
    echo "$resp" | jq --arg f "$filter" '.data | map(select(.id | test($f; "i")))'
  else
    echo "$resp" | jq '.data'
  fi
  exit 0
fi

ids="$(echo "$resp" | jq -r '.data[].id' 2>/dev/null || true)"
if [[ -z "$ids" ]]; then
  echo "no .data[].id in response — raw response below:" >&2
  echo "$resp" >&2
  exit 1
fi

if [[ -n "$filter" ]]; then
  echo "$ids" | grep -i "$filter" || { echo "no models match '$filter'" >&2; exit 1; }
else
  echo "$ids"
fi
