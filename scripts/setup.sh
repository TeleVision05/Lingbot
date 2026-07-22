#!/usr/bin/env bash
# Install lingbot-map env + download the long-sequence checkpoint.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Syncing submodule"
git submodule update --init --recursive

cd "$ROOT/lingbot-map"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
echo "==> Creating venv (Python ${PYTHON_VERSION})"
uv venv --python "$PYTHON_VERSION" .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing PyTorch"
if command -v nvidia-smi >/dev/null 2>&1; then
  # CUDA 12.8 wheels (matches upstream lingbot-map README)
  uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
else
  # Mac / CPU
  uv pip install torch torchvision
fi

echo "==> Installing lingbot-map (+ vis)"
uv pip install -e ".[vis]"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> Installing FlashInfer (optional, CUDA)"
  uv pip install --index-url https://pypi.org/simple flashinfer-python || \
    echo "FlashInfer install failed; demo will use --use_sdpa"
fi

echo "==> Downloading checkpoint"
bash "$ROOT/scripts/download_model.sh"

echo
echo "Setup complete."
echo "  cd lingbot-map && source .venv/bin/activate"
echo "  # then: bash ../scripts/run_lab.sh /path/to/video.mp4"
