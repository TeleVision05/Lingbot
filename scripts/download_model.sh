#!/usr/bin/env bash
# Download lingbot-map-long.pt into lingbot-map/checkpoints/
set -euo pipefail

case "${1:-}" in
  -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$ROOT/lingbot-map/checkpoints"
mkdir -p "$DIR"

if [ -f "$DIR/lingbot-map-long.pt" ]; then
  echo "Checkpoint already present: $DIR/lingbot-map-long.pt"
  ls -lh "$DIR/lingbot-map-long.pt"
  exit 0
fi

if [ -x "$ROOT/lingbot-map/.venv/bin/python" ]; then
  PY="$ROOT/lingbot-map/.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi

"$PY" -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    'robbyant/lingbot-map',
    'lingbot-map-long.pt',
    local_dir='$DIR',
)
print('Downloaded:', path)
"
ls -lh "$DIR/lingbot-map-long.pt"
