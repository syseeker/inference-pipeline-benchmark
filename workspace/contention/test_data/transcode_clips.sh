#!/usr/bin/env bash
# Transcode the customer-generated source clips into the two VLM workload specs.
#
# Run this on the GPU instance (needs ffmpeg; not installed on the laptop).
# Idempotent — safe to re-run; it overwrites the two outputs.
#
#   bash workspace/contention/test_data/transcode_clips.sh
#
# Why this exists: the two clips the harness actually feeds to a VLM must be
# H.264 (NVDEC-decodable) and at an exact frame count, because frame count is
# the VLM input-size dimension. The generated sources are H.264 already but at
# 24 fps, so they carry 72 and 240 frames instead of the 3 and 40 the design
# calls for. Feeding the wrong frame count would silently change the vision
# encoder's workload — the very thing being measured.
set -euo pipefail

cd "$(dirname "$0")"

SHORT_SRC="vlm/clip_3s_gemini.mp4"    # 1280x720, 3 s, 24 fps, 72 frames
LONG_SRC="vlm/clip_10s_gemini.mp4"    # 1280x720, 10 s, 24 fps, 240 frames

SHORT_OUT="vlm/clip_3s_224.mp4"       # 224x224,  3 s, 1 fps,  3 frames
LONG_OUT="vlm/clip_10s_720p.mp4"      # 1280x720, 10 s, 4 fps, 40 frames

command -v ffmpeg >/dev/null || {
  echo "ERROR: ffmpeg not found. Install it first:" >&2
  echo "  sudo apt-get update && sudo apt-get install -y ffmpeg" >&2
  exit 4
}

for f in "$SHORT_SRC" "$LONG_SRC"; do
  [[ -f "$f" ]] || { echo "ERROR: missing source clip: $f" >&2; exit 1; }
done

# -c:v libx264 is the load-bearing flag: it replaces the mp4v (MPEG-4 Part 2)
# encoding of the previous clips. mp4v has weak NVDEC routing in decord/PyAV,
# so it can silently fall back to CPU decode — which would turn a "GPU video
# tenant" into a partly-CPU tenant and contaminate every contention number.
#
# -an drops audio: the generated clips carry an AAC track the VLM never reads.
# Note the 224x224 output is aspect-distorted from 16:9. That matches what
# prepare_data.py already does for the CV images, and pixel count (not aspect)
# is what drives encoder cost.
echo ">> $SHORT_SRC -> $SHORT_OUT  (224x224, 1 fps, 3 frames)"
ffmpeg -y -loglevel error -i "$SHORT_SRC" \
  -vf "scale=224:224,fps=1" -t 3 \
  -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -an "$SHORT_OUT"

echo ">> $LONG_SRC -> $LONG_OUT  (1280x720, 4 fps, 40 frames)"
ffmpeg -y -loglevel error -i "$LONG_SRC" \
  -vf "scale=1280:720,fps=4" -t 10 \
  -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -an "$LONG_OUT"

echo
echo "=== verify ==="
fail=0
check() {  # path expected_codec expected_w expected_h expected_frames
  local p=$1 codec w h frames
  codec=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name \
          -of csv=p=0 "$p")
  w=$(ffprobe -v error -select_streams v:0 -show_entries stream=width  -of csv=p=0 "$p")
  h=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$p")
  frames=$(ffprobe -v error -select_streams v:0 -count_frames \
           -show_entries stream=nb_read_frames -of csv=p=0 "$p")
  printf '%-24s %-6s %sx%-5s %s frames' "$(basename "$p")" "$codec" "$w" "$h" "$frames"
  if [[ "$codec" == "$2" && "$w" == "$3" && "$h" == "$4" && "$frames" == "$5" ]]; then
    echo "   OK"
  else
    echo "   MISMATCH (want $2 $3x$4 $5 frames)"; fail=1
  fi
}
check "$SHORT_OUT" h264 224  224 3
check "$LONG_OUT"  h264 1280 720 40

[[ $fail -eq 0 ]] || { echo "ERROR: output did not match spec" >&2; exit 1; }
echo
echo "Both clips match spec and are H.264. Confirm NVDEC is actually used"
echo "before trusting video-tenant numbers (Phase 0 pre-flight)."
