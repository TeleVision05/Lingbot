# Lingbot lab runner

Wrapper around [lingbot-map](https://github.com/robbyant/lingbot-map) so you can clone one repo, install, and reconstruct a lab walkthrough video.

- **Submodule:** [`TeleVision05/lingbot-map`](https://github.com/TeleVision05/lingbot-map) branch `mac-cpu-compat` (upstream + Mac/CPU fixes)
- **Not in git:** video files, extracted frames, model weights (~4.3 GB), Python venv

## Clone

```bash
git clone --recurse-submodules https://github.com/TeleVision05/Lingbot.git
cd Lingbot
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Setup

```bash
bash scripts/setup.sh
```

This will:

1. Init the `lingbot-map` submodule  
2. Create `lingbot-map/.venv` (Python 3.10 via `uv`)  
3. Install PyTorch (CUDA 12.8 if `nvidia-smi` exists, else CPU/Mac)  
4. Install `lingbot-map[vis]`  
5. Download `lingbot-map-long.pt` into `lingbot-map/checkpoints/`

## Run

Put your video at `data/full_lab.mp4` (or pass a path):

```bash
# default: data/full_lab.mp4 or ./full lab.mp4
bash scripts/run_lab.sh

# explicit
bash scripts/run_lab.sh /path/to/video.mp4

# already-extracted frames
bash scripts/run_lab.sh /path/to/frames_dir
```

Open the viser UI: **http://localhost:8080**

### Useful overrides

```bash
# Apple Silicon / no GPU (forced CPU)
LINGBOT_DEVICE=cpu FIRST_K=24 STRIDE=2 bash scripts/run_lab.sh data/full_lab.mp4

# Cloud GPU, subsample FPS
FPS=5 bash scripts/run_lab.sh data/full_lab.mp4

# Limit frames while testing
FIRST_K=32 bash scripts/run_lab.sh data/full_lab.mp4
```

## Cloud GPU (Colab / Kaggle / VM)

```bash
git clone --recurse-submodules https://github.com/TeleVision05/Lingbot.git
cd Lingbot
bash scripts/setup.sh

# upload video, then:
bash scripts/run_lab.sh data/full_lab.mp4
```

Expose port 8080 with localtunnel / ngrok / Cloudflare Tunnel if the host is remote.

On CUDA hosts, FlashInfer is installed when possible; otherwise the runner adds `--use_sdpa`.

## Layout

```text
Lingbot/
  data/                 # put full_lab.mp4 here (gitignored)
  scripts/
    setup.sh
    download_model.sh
    run_lab.sh
  lingbot-map/          # git submodule (TeleVision05/lingbot-map@mac-cpu-compat)
```

## Updating the submodule

```bash
cd lingbot-map
git fetch origin
git checkout mac-cpu-compat
git pull
cd ..
git add lingbot-map
git commit -m "Bump lingbot-map submodule"
```

## Upstream

Upstream project: [Robbyant/lingbot-map](https://github.com/robbyant/lingbot-map).  
Our fork tracks local/Mac/CPU compatibility patches on `mac-cpu-compat`.
