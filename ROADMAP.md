# Lingbot roadmap

Wrapper repo around the `lingbot-map` fork: clone, install, reconstruct a lab
walkthrough video into a point cloud, export GLB/PLY, render an orbit MP4.

## Current state (2026-09-25)

**Status: working (CPU path), experimental (GPU path).**

What works, and how it was verified:

| Piece | Evidence |
|---|---|
| Interactive viewer (`scripts/run_lab.sh`) | Launches `lingbot-map/demo.py` on viser port 8080; device auto-select via `LINGBOT_DEVICE` (`demo.py:52`). Not re-run today. |
| Headless export (`scripts/export_full_lab_glb.sh` + `.py`) | Real run on 2026-07-22 (`outputs/export_full_lab.log`): 487 frames from `full lab_frames`, CPU windowed inference in 5 windows, 3533.6 s (~59 min) on Apple Silicon, `nan=0` poses, 1,408,368 points → `full_lab.glb` (22.5 MB), `full_lab.ply`, `full_lab_predictions.npz` (306 MB). |
| Orbit render (`scripts/render_lab_orbit.py`) | Real run on 2026-07-22 (`outputs/render_lab_orbit.log`): 180 frames → `full_lab_orbit.mp4` (24 MB) via ffmpeg. |
| Setup / download (`scripts/setup.sh`, `scripts/download_model.sh`) | `--help` exercised by tests; the venv at `lingbot-map/.venv` (Python 3.10 via uv) and `checkpoints/lingbot-map-long.pt` exist locally, so the path has run at least once. |
| Unit tests | `lingbot-map/.venv/bin/python -m unittest discover -s tests -v` → `Ran 9 tests in 0.132s`, `OK` (run 2026-09-25). Covers `invert_c2w`, `sparsify_points`, `look_at_matrix`, `render_frame`, friendly-exit guards, and `--help` on all four shell scripts. |

Important caveat: commit `b2814a8` refactored `scripts/export_full_lab_glb.py`
(lazy imports, `parse_args(argv)`, `main(argv)`) **after** the 2026-07-22 run.
The refactored file has only been exercised by the unit tests, never against
the real model. See Milestone 1.

## Known gaps and bugs

- **`main()` in `scripts/export_full_lab_glb.py` is 175 lines** (lines 125–299,
  measured with `ast` on 2026-09-25; the earlier note said ~140). It mixes
  image loading, model config, inference, NPZ save, point building, and
  trimesh/GLB assembly. Over the 75-line house limit.
- **Refactored export never run end to end** (see above). A regression in the
  `argv` plumbing or the lazy `_import_lingbot()` would only show up on a real
  ~1 h CPU run.
- **`scripts/setup.sh:17-21` installs `uv` via `curl ... | sh`** when it is not
  on PATH. Convenient, but it runs a remote script unreviewed. Documented in
  the README; not changed.
- **Machine-specific defaults.** `scripts/export_full_lab_glb.sh:20` and
  `scripts/run_lab.sh:45` fall back to `$ROOT/full lab_frames` and
  `$ROOT/full lab.mp4` (space in the name), which exist only on the author's
  machine. `data/full_lab.mp4` is the documented path.
- **Mode mismatch between the two entry points.** `run_lab.sh:86-90` uses
  `--mode windowed` only for video input with `FIRST_K` unset; a frames
  *folder* always goes `--mode streaming`. `export_full_lab_glb.py` is always
  windowed. A 487-frame folder therefore behaves differently in the viewer
  and the exporter.
- **Camera overlay is best-effort.** `export_full_lab_glb.py:278-295` wraps
  `predictions_to_glb` in a broad `except Exception` and silently exports
  points only. The July log shows it succeeded, but a failure would be a
  single printed line.
- **`outputs/view_lab.html`** is an untracked three.js viewer (gitignored by
  `**/outputs/`). It is not mentioned anywhere in the repo.
- **No pre-commit hook / linter.** The repo has no `.pre-commit-config.yaml`
  or shellcheck/ruff config, so nothing gates commits.
- **Unused names** in `sparsify_points` (`S, H, W, _ = points.shape`,
  `export_full_lab_glb.py:97`). Harmless; acts as an implicit 4-D check.
- **No CI.** Tests only run when someone remembers to.

## Milestones

### Milestone 1 — Prove the refactored export on the real model

- [ ] Run `bash scripts/export_full_lab_glb.sh` (CPU, ~1 h) on the current
      `b2814a8` code with `data/full_lab.mp4` or the frames folder.
- [ ] Diff point count and file sizes against the 2026-07-22 log
      (1,408,368 points, 22.5 MB GLB).
- [ ] Run `scripts/render_lab_orbit.py` on the fresh PLY.
- [ ] Record the numbers and date in this file.

**Done when:** `outputs/export_full_lab.log` from a post-`b2814a8` run ends in
`Done.` with `nan=0`, and a fresh `full_lab_orbit.mp4` plays.

### Milestone 2 — Split `main()` and lock it with tests

- [ ] Extract `load_frames(args, load_images)`, `build_model_ns(args, num_frames, use_sdpa)`,
      `run_inference(model, images, args, device, dtype)`, `save_intermediates(...)`,
      and `assemble_scene(flat_pts, flat_cols, ...)` from `main()`.
- [ ] Every extracted function under 75 lines; `main()` becomes orchestration only.
- [ ] Add unit tests for `build_model_ns` (namespace fields) and
      `assemble_scene` (writes a GLB and PLY from a tiny synthetic cloud, using
      trimesh only).
- [ ] Re-run Milestone 1's end-to-end check once after the split.

**Done when:** `python -m unittest discover -s tests -v` reports ≥ 12 tests
`OK`, no function in `scripts/export_full_lab_glb.py` exceeds 75 lines
(check with the `ast` one-liner in git history), and one post-split real run
matches the Milestone 1 numbers.

### Milestone 3 — Remove machine-specific paths and align the two entry points

- [ ] Drop the `full lab.mp4` / `full lab_frames` fallbacks from
      `run_lab.sh` and `export_full_lab_glb.sh`; keep only `data/full_lab.mp4`
      and `data/full_lab_frames/`.
- [ ] Give `run_lab.sh` a `MODE=windowed|streaming` env override with the
      same default rule as the exporter for long inputs.
- [ ] Add a `ShellScripts` test asserting the "no input" error message names
      `data/full_lab.mp4`.

**Done when:** `grep -rn 'full lab' scripts/` returns nothing and the new
test passes.

### Milestone 4 — Commit gates and CI

- [ ] Add `.pre-commit-config.yaml` with `ruff` (Python) and `shellcheck`
      (bash); commit the config.
- [ ] Add a GitHub Actions workflow that installs numpy + Pillow only and
      runs the unittest suite on push and PR (no torch, no weights).
- [ ] Fix whatever ruff/shellcheck report on the current scripts.

**Done when:** a PR to `main` shows a green `tests` check at
`https://github.com/TeleVision05/Lingbot/actions`, and
`pre-commit run --all-files` passes locally.

### Milestone 5 — Track and document the interactive viewer

- [ ] Move `outputs/view_lab.html` to `scripts/view_lab.html` (or a
      `viewer/` folder) so it is versioned.
- [ ] Document how to load `outputs/full_lab.glb` into it in the README.
- [ ] Decide whether it replaces the viser UI for sharing results.

**Done when:** `git ls-files` includes the viewer, and a screenshot of it
displaying the July GLB is linked from this file.

### Milestone 6 — GPU path verified once

- [ ] Run `bash scripts/setup.sh` and `bash scripts/run_lab.sh` on a CUDA
      box (Colab / Kaggle / VM) and record whether FlashInfer installed or
      the runner fell back to `--use_sdpa`.
- [ ] Run the export on GPU and note wall time vs the 59-minute CPU run.

**Done when:** this file has a dated row for a CUDA run with its frame count,
wall time, and which attention backend was used.

## Won't do / out of scope

- Committing video, extracted frames, model weights, or `outputs/` (all
  gitignored on purpose; the checkpoint alone is ~4.3 GB).
- Changing `lingbot-map` model code here. Model/Mac-compat changes go to the
  fork's `mac-cpu-compat` branch and are picked up by bumping the submodule.
- Replacing `uv` with pip/conda.
- Supporting Python versions other than 3.10 (matches upstream).
- A GUI or web service around the scripts.

## Open questions for the owner

1. Is `outputs/view_lab.html` something you want to keep and version, or a
   one-off? (Milestone 5 assumes keep.)
2. Should the export default stay CPU (`LINGBOT_DEVICE=cpu` in
   `export_full_lab_glb.sh:46`) even on a CUDA host, or auto-detect like
   `run_lab.sh` does?
3. Is a ~1 h CPU re-run for Milestone 1 acceptable on your Mac, or should it
   wait for a GPU box (Milestone 6) and be done there?
4. Do you want `setup.sh` to stop auto-installing `uv` via `curl | sh` and
   instead fail with an install hint?
5. Should the submodule pin move to upstream `robbyant/lingbot-map` once
   the Mac/CPU patches land upstream, or stay on the fork indefinitely?
