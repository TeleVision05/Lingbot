#!/usr/bin/env bash
# Reconstruct the full lab video and export a GLB mesh/point object.
#
# Usage:
#   bash scripts/export_full_lab_glb.sh
#   bash scripts/export_full_lab_glb.sh /path/to/video.mp4
set -euo pipefail

case "${1:-}" in
  -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAP="$ROOT/lingbot-map"
OUT_DIR="$ROOT/outputs"
mkdir -p "$OUT_DIR"

INPUT="${1:-}"
if [ -z "$INPUT" ]; then
  if [ -d "$ROOT/full lab_frames" ]; then
    INPUT="$ROOT/full lab_frames"
  elif [ -f "$ROOT/data/full_lab.mp4" ]; then
    INPUT="$ROOT/data/full_lab.mp4"
  elif [ -f "$ROOT/full lab.mp4" ]; then
    INPUT="$ROOT/full lab.mp4"
  else
    echo "No video/frames found. Pass a path, or put the video at data/full_lab.mp4." >&2
    exit 1
  fi
fi

if [ ! -f "$MAP/.venv/bin/activate" ]; then
  echo "lingbot-map venv not found at $MAP/.venv. Run: bash scripts/setup.sh" >&2
  exit 1
fi
if [ ! -f "$MAP/checkpoints/lingbot-map-long.pt" ]; then
  echo "Model weights not found at $MAP/checkpoints/lingbot-map-long.pt." >&2
  echo "Run: bash scripts/download_model.sh  (or bash scripts/setup.sh)" >&2
  exit 1
fi

cd "$MAP"
# shellcheck disable=SC1091
source .venv/bin/activate

export LINGBOT_DEVICE="${LINGBOT_DEVICE:-cpu}"
export PYTHONUNBUFFERED=1

exec python "$ROOT/scripts/export_full_lab_glb.py" \
  --input "$INPUT" \
  --model_path "$MAP/checkpoints/lingbot-map-long.pt" \
  --output_dir "$OUT_DIR" \
  --fps "${FPS:-10}" \
  --point_stride "${POINT_STRIDE:-2}" \
  --spatial_stride "${SPATIAL_STRIDE:-4}"
