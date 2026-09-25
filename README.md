# Lingbot lab runner

Wrapper around [lingbot-map](https://github.com/robbyant/lingbot-map) so you can clone one repo, install, and reconstruct a lab walkthrough video into a 3D point cloud (interactive viewer, GLB/PLY export, orbit MP4).

**Status:** working on CPU / Apple Silicon (last full run 2026-07-22, 487 frames in ~59 min); GPU path untested by us. See [ROADMAP.md](ROADMAP.md).

- **Submodule:** [`TeleVision05/lingbot-map`](https://github.com/TeleVision05/lingbot-map) branch `mac-cpu-compat` (upstream + Mac/CPU fixes)
- **Not in git:** video files, extracted frames, model weights (~4.3 GB), Python venv, `outputs/`

## Install

```bash
git clone --recurse-submodules https://github.com/TeleVision05/Lingbot.git
cd Lingbot
bash scripts/setup.sh
```

If you already cloned without submodules: `git submodule update --init --recursive`.

`setup.sh` will:

1. Init the `lingbot-map` submodule
2. Install `uv` via `curl -LsSf https://astral.sh/uv/install.sh | sh` **if it is not already on PATH**
3. Create `lingbot-map/.venv` (Python 3.10)
4. Install PyTorch (CUDA 12.8 wheels if `nvidia-smi` exists, else CPU/Mac wheels)
5. Install `lingbot-map[vis]` and, on CUDA hosts, try FlashInfer
6. Download `lingbot-map-long.pt` into `lingbot-map/checkpoints/`

Requires `git`, `curl`, and (for the orbit render) `ffmpeg`.

## Configure

There are no secrets and no `.env` file. Everything is an environment variable read by the scripts:

| Variable | Used by | Default | Meaning |
|---|---|---|---|
| `LINGBOT_DEVICE` | `run_lab.sh`, `export_full_lab_glb.sh` | auto / `cpu` | Force `cpu`, `mps`, or `cuda` |
| `PORT` | `run_lab.sh` | `8080` | viser viewer port |
| `FPS` | `run_lab.sh`, `export_full_lab_glb.sh` | `10` | Frames per second sampled from the video |
| `STRIDE` | `run_lab.sh` | unset | Keep every Nth sampled frame |
| `FIRST_K` | `run_lab.sh` | unset | Limit to the first K frames (also switches to streaming mode) |
| `POINT_STRIDE` | `export_full_lab_glb.sh` | `2` | Keep every Nth frame when building the point cloud |
| `SPATIAL_STRIDE` | `export_full_lab_glb.sh` | `4` | Keep every Nth pixel in H/W when exporting |
| `PYTHON_VERSION` | `setup.sh` | `3.10` | Interpreter for the venv |

## Run

Put your video at `data/full_lab.mp4` (or pass a path).

Interactive viewer at **http://localhost:8080**:

```bash
bash scripts/run_lab.sh                      # data/full_lab.mp4
bash scripts/run_lab.sh /path/to/video.mp4
bash scripts/run_lab.sh /path/to/frames_dir  # already-extracted frames

# Apple Silicon / no GPU, quick smoke run
LINGBOT_DEVICE=cpu FIRST_K=24 STRIDE=2 bash scripts/run_lab.sh data/full_lab.mp4
```

Headless export and orbit video:

```bash
# Windowed reconstruction → outputs/full_lab.{glb,ply} + full_lab_predictions.npz
bash scripts/export_full_lab_glb.sh data/full_lab.mp4

# Software orbit render of the PLY → outputs/full_lab_orbit.mp4 (needs ffmpeg)
lingbot-map/.venv/bin/python scripts/render_lab_orbit.py
```

Every script supports `--help` and exits with a clear message if the venv, the model weights, or the input is missing.

On a remote GPU box, expose port 8080 with localtunnel / ngrok / Cloudflare Tunnel. On CUDA hosts FlashInfer is used when installed; otherwise the runner adds `--use_sdpa`.

## Test

Unit tests cover the pure-python helpers and the scripts' guard rails. They need only numpy + Pillow (no torch, weights, or video):

```bash
lingbot-map/.venv/bin/python -m unittest discover -s tests -v
```

## Layout

```text
Lingbot/
  data/                 # put full_lab.mp4 here (gitignored)
  outputs/              # GLB / PLY / NPZ / MP4 results (gitignored)
  scripts/
    setup.sh              # venv + torch + lingbot-map + weights
    download_model.sh     # weights only (~4.3 GB)
    run_lab.sh            # interactive viser viewer
    export_full_lab_glb.* # headless reconstruction → GLB/PLY
    render_lab_orbit.py   # point cloud → orbit MP4
  tests/                # unittest smoke tests
  lingbot-map/          # git submodule (TeleVision05/lingbot-map@mac-cpu-compat)
  ROADMAP.md            # current state, gaps, milestones
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
