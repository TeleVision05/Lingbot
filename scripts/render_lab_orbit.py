#!/usr/bin/env python3
"""Software orbit render of a PLY/GLB point cloud → MP4 (no GPU/EGL needed)."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load_ply_xyz_rgb(path: Path, max_points: int = 600_000):
    """Load points via open3d if available, else trimesh."""
    try:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(path))
        pts = np.asarray(pcd.points, dtype=np.float64)
        cols = np.asarray(pcd.colors, dtype=np.float64) if pcd.has_colors() else None
    except Exception:
        import trimesh

        geom = trimesh.load(path)
        if hasattr(geom, "vertices"):
            pts = np.asarray(geom.vertices, dtype=np.float64)
            cols = np.asarray(geom.colors[:, :3], dtype=np.float64) / 255.0 if getattr(geom, "colors", None) is not None else None
        else:
            raise RuntimeError(f"Cannot read points from {path}")

    if cols is None or len(cols) != len(pts):
        cols = np.full((len(pts), 3), 0.65, dtype=np.float64)
    if cols.max() > 1.0 + 1e-3:
        cols = cols / 255.0

    finite = np.isfinite(pts).all(axis=1)
    pts, cols = pts[finite], cols[finite]

    # Robust center/scale — drop outlier tails
    lo = np.percentile(pts, 2, axis=0)
    hi = np.percentile(pts, 98, axis=0)
    in_box = np.all((pts >= lo) & (pts <= hi), axis=1)
    pts, cols = pts[in_box], cols[in_box]

    if len(pts) > max_points:
        idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, cols = pts[idx], cols[idx]

    return pts, np.clip(cols, 0, 1)


def look_at_matrix(eye, target, up):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-12)
    s = np.cross(f, up)
    s = s / (np.linalg.norm(s) + 1e-12)
    u = np.cross(s, f)
    # world → camera (OpenCV-ish: x right, y down, z forward)
    R = np.stack([s, -u, f], axis=0)
    t = -R @ eye
    return R, t


def project_points(pts, R, t, fx, fy, cx, cy):
    cam = (R @ pts.T).T + t
    z = cam[:, 2]
    valid = z > 1e-4
    u = fx * (cam[:, 0] / z) + cx
    v = fy * (cam[:, 1] / z) + cy
    return u, v, z, valid


def render_frame(pts, cols, eye, center, up, width, height, fov_deg=55.0):
    aspect = width / height
    f = 0.5 * height / math.tan(math.radians(fov_deg) * 0.5)
    fx = fy = f
    cx, cy = width * 0.5, height * 0.5

    R, t = look_at_matrix(eye, center, up)
    u, v, z, valid = project_points(pts, R, t, fx, fy, cx, cy)

    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    in_img = valid & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)

    ui, vi, z, c = ui[in_img], vi[in_img], z[in_img], cols[in_img]

    # Painter's algorithm by depth (far → near)
    order = np.argsort(-z)
    ui, vi, c = ui[order], vi[order], c[order]

    img = np.full((height, width, 3), 245, dtype=np.uint8)  # light bg
    rgb = (c * 255.0).astype(np.uint8)

    # Soft 2x2 splat for denser look
    for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):
        x = ui + du
        y = vi + dv
        m = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        img[y[m], x[m]] = rgb[m]

    return img


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Render an orbit MP4 around a PLY/GLB point cloud.")
    ap.add_argument("--input", type=Path, default=root / "outputs" / "full_lab.ply")
    ap.add_argument("--output", type=Path, default=root / "outputs" / "full_lab_orbit.mp4")
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--max_points", type=int, default=600_000)
    args = ap.parse_args(argv)

    if not args.input.is_file():
        sys.exit(
            f"Point cloud not found: {args.input}\n"
            "Run `bash scripts/export_full_lab_glb.sh` first, or pass --input."
        )
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg is required to encode the MP4. Install it and retry.")

    pts, cols = load_ply_xyz_rgb(args.input, max_points=args.max_points)
    center = np.median(pts, axis=0)
    extent = np.percentile(pts, 95, axis=0) - np.percentile(pts, 5, axis=0)
    radius = float(np.linalg.norm(extent)) * 0.85
    radius = max(radius, 0.5)
    up = np.array([0.0, -1.0, 0.0])

    frames_dir = args.output.parent / "_orbit_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(pts):,} points. Rendering {args.frames} frames...")

    for i in range(args.frames):
        ang = 2.0 * math.pi * i / args.frames
        elev = -0.25 + 0.08 * math.sin(ang * 2)  # slight bob
        eye = center + np.array([
            radius * math.cos(ang),
            elev * radius,
            radius * math.sin(ang),
        ])
        img = render_frame(pts, cols, eye, center, up, args.width, args.height)
        Image.fromarray(img).save(frames_dir / f"{i:04d}.png")
        if (i + 1) % 30 == 0 or i == 0:
            print(f"  frame {i + 1}/{args.frames}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-i", str(frames_dir / "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(args.output),
    ]
    print("Encoding", args.output)
    subprocess.run(cmd, check=True)

    for fp in frames_dir.glob("*.png"):
        fp.unlink()
    try:
        frames_dir.rmdir()
    except OSError:
        pass
    print(f"Done: {args.output}")


if __name__ == "__main__":
    main()
