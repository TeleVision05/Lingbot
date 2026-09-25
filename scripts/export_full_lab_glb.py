#!/usr/bin/env python3
"""Headless full-video lingbot-map reconstruction → GLB export.

Runs windowed CPU (or LINGBOT_DEVICE) inference over the whole sequence,
saves intermediate predictions, builds world points, and writes a .glb.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

MAP_ROOT = Path(__file__).resolve().parents[1] / "lingbot-map"


def _import_lingbot():
    """Import torch + lingbot-map lazily so --help and helpers work without them."""
    sys.path.insert(0, str(MAP_ROOT))
    os.chdir(MAP_ROOT)
    import torch  # noqa: F401
    import demo
    from lingbot_map.utils.geometry import unproject_depth_map_to_point_map
    from lingbot_map.vis.glb_export import predictions_to_glb

    return torch, demo, unproject_depth_map_to_point_map, predictions_to_glb


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Full-lab reconstruction → GLB")
    p.add_argument("--input", required=True, help="Video path or image folder")
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--image_size", type=int, default=518)
    p.add_argument("--patch_size", type=int, default=14)
    p.add_argument("--window_size", type=int, default=64)
    p.add_argument("--keyframe_interval", type=int, default=2)
    p.add_argument("--overlap_keyframes", type=int, default=8)
    p.add_argument("--num_scale_frames", type=int, default=2)
    p.add_argument("--camera_num_iterations", type=int, default=1)
    p.add_argument("--conf_thres", type=float, default=40.0)
    p.add_argument("--point_stride", type=int, default=2,
                    help="Keep every Nth frame when building the point cloud")
    p.add_argument("--spatial_stride", type=int, default=4,
                    help="Keep every Nth pixel in H/W when exporting points")
    p.add_argument("--max_points", type=int, default=2_000_000)
    return p.parse_args(argv)


def tensor_to_np(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def invert_c2w(extrinsic):
    """Invert (S,3,4) camera-to-world matrices into (S,3,4) world-to-camera."""
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    S = extrinsic.shape[0]
    T = np.tile(np.eye(4, dtype=np.float64), (S, 1, 1))
    T[:, :3, :4] = extrinsic
    return np.linalg.inv(T)[:, :3, :4]


def build_world_points(depth, extrinsic, intrinsic, point_stride, spatial_stride,
                       unproject_depth_map_to_point_map):
    """Unproject depth → world points, optionally sparsified for export size."""
    depth = tensor_to_np(depth)
    extrinsic = tensor_to_np(extrinsic)
    intrinsic = tensor_to_np(intrinsic)

    # Extrinsic from demo postprocess is c2w; unproject expects w2c (cam←world).
    # Invert each 3x4 into w2c for unproject.
    S = extrinsic.shape[0]
    w2c = invert_c2w(extrinsic)

    frame_ids = list(range(0, S, max(1, point_stride)))
    print(f"Unprojecting {len(frame_ids)}/{S} frames (point_stride={point_stride})...")
    t0 = time.time()
    pts = unproject_depth_map_to_point_map(
        depth[frame_ids], w2c[frame_ids], intrinsic[frame_ids]
    )
    print(f"  unproject done in {time.time() - t0:.1f}s, shape={pts.shape}")

    if spatial_stride > 1:
        pts = pts[:, ::spatial_stride, ::spatial_stride, :]
    return pts, frame_ids


def sparsify_points(points, colors, conf, conf_thres_pct, max_points):
    """Flatten HxW maps → Nx3, filter by confidence percentile, cap count."""
    S, H, W, _ = points.shape
    pts = points.reshape(-1, 3)
    cols = colors.reshape(-1, 3)
    if conf is not None:
        c = conf.reshape(-1)
    else:
        c = np.ones(len(pts), dtype=np.float32)

    finite = np.isfinite(pts).all(axis=1) & np.isfinite(c)
    pts, cols, c = pts[finite], cols[finite], c[finite]

    if len(pts) == 0:
        raise RuntimeError("No finite points after unprojection")

    # conf_thres is a *percentage* of low-confidence points to drop (glb_export style)
    if conf_thres_pct > 0 and len(c) > 0:
        thr = np.percentile(c, conf_thres_pct)
        keep = c >= thr
        pts, cols, c = pts[keep], cols[keep], c[keep]

    if len(pts) > max_points:
        idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        idx.sort()
        pts, cols = pts[idx], cols[idx]

    return pts, cols


def main(argv=None):
    args = parse_args(argv)
    if not Path(args.input).exists():
        sys.exit(f"Input not found: {args.input}\nPass a video file or a folder of frames with --input.")
    if not Path(args.model_path).is_file():
        sys.exit(
            f"Model weights not found: {args.model_path}\n"
            "Run `bash scripts/download_model.sh` (or `bash scripts/setup.sh`) first."
        )
    try:
        torch, demo, unproject_depth_map_to_point_map, predictions_to_glb = _import_lingbot()
    except ImportError as exc:
        sys.exit(
            f"lingbot-map is not installed ({exc}).\n"
            "Run `bash scripts/setup.sh`, then use lingbot-map/.venv/bin/python."
        )
    amp_autocast, load_images, load_model = demo.amp_autocast, demo.load_images, demo.load_model
    postprocess, prepare_for_visualization = demo.postprocess, demo.prepare_for_visualization
    select_device = demo.select_device

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "full_lab_predictions.npz"
    glb_path = out_dir / "full_lab.glb"
    ply_path = out_dir / "full_lab.ply"

    device = select_device()
    print(f"Device: {device}")
    use_sdpa = device.type != "cuda"

    # ---- Load images ----
    inp = args.input
    if os.path.isdir(inp):
        images, paths, folder = load_images(
            image_folder=inp, image_size=args.image_size, patch_size=args.patch_size
        )
    else:
        images, paths, folder = load_images(
            video_path=inp, fps=args.fps,
            image_size=args.image_size, patch_size=args.patch_size,
        )
    num_frames = images.shape[0]
    print(f"Loaded {num_frames} frames from {inp}")

    # ---- Model ----
    class A:  # minimal namespace for load_model
        pass

    ns = A()
    ns.mode = "windowed"
    ns.image_size = args.image_size
    ns.patch_size = args.patch_size
    ns.enable_3d_rope = True
    ns.max_frame_num = max(num_frames + 64, 512)
    ns.kv_cache_sliding_window = 64
    ns.num_scale_frames = args.num_scale_frames
    ns.use_sdpa = use_sdpa
    ns.camera_num_iterations = args.camera_num_iterations
    ns.model_path = args.model_path

    model = load_model(ns, device)
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    images = images.to(device)
    print(f"Running windowed inference on {num_frames} frames (dtype={dtype})...")
    t0 = time.time()
    with torch.no_grad(), amp_autocast(device, dtype):
        predictions = model.inference_windowed(
            images,
            window_size=args.window_size,
            overlap_size=None,
            overlap_keyframes=args.overlap_keyframes,
            num_scale_frames=args.num_scale_frames,
            keyframe_interval=args.keyframe_interval,
            output_device=torch.device("cpu"),
        )
    print(f"Inference done in {time.time() - t0:.1f}s")

    del images
    if device.type == "mps":
        torch.mps.empty_cache()

    predictions, images_cpu = postprocess(predictions, predictions["images"])

    # Health check
    ext = predictions["extrinsic"]
    n_nan = int(torch.isnan(ext).sum()) if isinstance(ext, torch.Tensor) else int(np.isnan(ext).sum())
    print(f"Pose health: nan={n_nan}")
    if n_nan:
        if isinstance(ext, torch.Tensor):
            predictions["extrinsic"] = torch.nan_to_num(ext)
        else:
            predictions["extrinsic"] = np.nan_to_num(ext)

    vis = prepare_for_visualization(predictions, images_cpu)
    depth = vis["depth"]
    extrinsic = vis["extrinsic"]
    intrinsic = vis["intrinsic"]
    depth_conf = vis.get("depth_conf")
    imgs = vis["images"]  # (S,3,H,W)

    # Save compact intermediates (no full world points yet)
    print(f"Saving intermediates → {pred_path}")
    save_kwargs = dict(
        depth=depth.astype(np.float16),
        extrinsic=extrinsic.astype(np.float32),
        intrinsic=intrinsic.astype(np.float32),
        images=(imgs.transpose(0, 2, 3, 1) * 255).astype(np.uint8)
        if imgs.ndim == 4 and imgs.shape[1] == 3
        else (imgs * 255).astype(np.uint8),
    )
    if depth_conf is not None:
        save_kwargs["depth_conf"] = depth_conf.astype(np.float16)
    np.savez_compressed(pred_path, **save_kwargs)

    world_pts, frame_ids = build_world_points(
        depth, extrinsic, intrinsic, args.point_stride, args.spatial_stride,
        unproject_depth_map_to_point_map,
    )

    # Colors for selected frames, same spatial stride
    if imgs.ndim == 4 and imgs.shape[1] == 3:
        colors = imgs[frame_ids].transpose(0, 2, 3, 1)  # S,H,W,3
    else:
        colors = imgs[frame_ids]
    if args.spatial_stride > 1:
        colors = colors[:, :: args.spatial_stride, :: args.spatial_stride, :]

    conf_sel = None
    if depth_conf is not None:
        conf_sel = depth_conf[frame_ids]
        if args.spatial_stride > 1:
            conf_sel = conf_sel[:, :: args.spatial_stride, :: args.spatial_stride]

    flat_pts, flat_cols = sparsify_points(
        world_pts, colors, conf_sel, args.conf_thres, args.max_points
    )
    print(f"Exporting {len(flat_pts):,} points → {glb_path}")

    # Build a minimal predictions dict for predictions_to_glb using already-flat
    # data via a tiny trimesh scene (more reliable than re-entering glb_export
    # frame maps after sparsify).
    import trimesh

    cols_u8 = (np.clip(flat_cols, 0, 1) * 255).astype(np.uint8) if flat_cols.max() <= 1.0 + 1e-3 else flat_cols.astype(np.uint8)
    cloud = trimesh.points.PointCloud(vertices=flat_pts.astype(np.float32), colors=cols_u8)
    scene = trimesh.Scene()
    scene.add_geometry(cloud, geom_name="lab_points")

    # Also attach camera frustums via predictions_to_glb on a light subsample
    try:
        light = {
            "world_points_from_depth": world_pts,
            "depth_conf": conf_sel if conf_sel is not None else np.ones(world_pts.shape[:3], np.float32),
            "images": colors,
            "extrinsic": extrinsic[frame_ids],
        }
        cam_scene = predictions_to_glb(
            light,
            conf_thres=args.conf_thres,
            show_cam=True,
            prediction_mode="Predicted Depthmap",
        )
        for name, geom in cam_scene.geometry.items():
            if "cam" in name.lower() or "frustum" in name.lower() or "camera" in name.lower():
                scene.add_geometry(geom, geom_name=name)
    except Exception as exc:  # noqa: BLE001
        print(f"Camera overlay skipped ({exc}); exporting points only")

    scene.export(glb_path)
    cloud.export(ply_path)
    print(f"Done.\n  GLB: {glb_path}\n  PLY: {ply_path}\n  NPZ: {pred_path}")


if __name__ == "__main__":
    main()
