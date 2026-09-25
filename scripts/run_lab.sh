#!/usr/bin/env bash
# Run interactive lingbot-map demo on a video or image folder.
#
# Usage:
#   bash scripts/run_lab.sh                          # uses data/full_lab.mp4 or ../full lab.mp4
#   bash scripts/run_lab.sh /path/to/video.mp4
#   bash scripts/run_lab.sh /path/to/frames_dir
#
# Env overrides:
#   LINGBOT_DEVICE=cpu|mps|cuda   (auto if unset)
#   PORT=8080
#   FPS=10
#   STRIDE=1
#   FIRST_K=                 (optional limit)
set -euo pipefail

case "${1:-}" in
  -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAP="$ROOT/lingbot-map"
CKPT="$MAP/checkpoints/lingbot-map-long.pt"
PORT="${PORT:-8080}"
FPS="${FPS:-10}"

if [ ! -f "$MAP/.venv/bin/activate" ]; then
  echo "lingbot-map venv not found at $MAP/.venv. Run: bash scripts/setup.sh" >&2
  exit 1
fi
if [ ! -f "$CKPT" ]; then
  echo "Model weights not found at $CKPT." >&2
  echo "Run: bash scripts/download_model.sh  (or bash scripts/setup.sh)" >&2
  exit 1
fi

cd "$MAP"
# shellcheck disable=SC1091
source .venv/bin/activate

INPUT="${1:-}"
if [ -z "$INPUT" ]; then
  if [ -f "$ROOT/data/full_lab.mp4" ]; then
    INPUT="$ROOT/data/full_lab.mp4"
  elif [ -f "$ROOT/full lab.mp4" ]; then
    INPUT="$ROOT/full lab.mp4"
  else
    echo "Usage: bash scripts/run_lab.sh /path/to/video.mp4"
    echo "Or place the video at data/full_lab.mp4"
    exit 1
  fi
fi

ARGS=(
  --model_path "$CKPT"
  --port "$PORT"
  --conf_threshold 1.5
  --downsample_factor 3
  --point_size 0.008
)

# Auto device flags
if command -v nvidia-smi >/dev/null 2>&1 && [ "${LINGBOT_DEVICE:-}" != "cpu" ]; then
  # Prefer FlashInfer when present; otherwise SDPA
  python -c "import flashinfer" 2>/dev/null || ARGS+=(--use_sdpa)
else
  export LINGBOT_DEVICE="${LINGBOT_DEVICE:-cpu}"
  ARGS+=(--use_sdpa --camera_num_iterations 1 --num_scale_frames 2)
  echo "Non-CUDA host: LINGBOT_DEVICE=$LINGBOT_DEVICE (SDPA, lighter settings)"
fi

if [ -d "$INPUT" ]; then
  ARGS+=(--image_folder "$INPUT")
else
  ARGS+=(--video_path "$INPUT" --fps "$FPS")
fi

if [ -n "${STRIDE:-}" ]; then
  ARGS+=(--stride "$STRIDE")
fi
if [ -n "${FIRST_K:-}" ]; then
  ARGS+=(--first_k "$FIRST_K")
fi

# Long sequences: windowed by default when FIRST_K unset and using video
if [ -z "${FIRST_K:-}" ] && [ ! -d "$INPUT" ]; then
  ARGS+=(--mode windowed --window_size 64 --keyframe_interval 2 --overlap_keyframes 8)
else
  ARGS+=(--mode streaming --keyframe_interval 1)
fi

echo "Running: python demo.py ${ARGS[*]}"
echo "Viewer:  http://localhost:$PORT"
exec python demo.py "${ARGS[@]}"
