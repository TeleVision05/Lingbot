"""Smoke tests for the pure-python helpers and script guard rails.

Needs only numpy + Pillow (no torch, no weights, no video):
    lingbot-map/.venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


export = _load("export_full_lab_glb")
orbit = _load("render_lab_orbit")


class ExportHelpers(unittest.TestCase):
    def test_invert_c2w_round_trips(self):
        c2w = np.zeros((2, 3, 4))
        c2w[:, :3, :3] = np.eye(3)
        c2w[0, :, 3] = [1.0, 2.0, 3.0]
        c2w[1, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        w2c = export.invert_c2w(c2w)
        for i in range(2):
            a = np.vstack([c2w[i], [0, 0, 0, 1]])
            b = np.vstack([w2c[i], [0, 0, 0, 1]])
            np.testing.assert_allclose(a @ b, np.eye(4), atol=1e-12)

    def test_sparsify_drops_nonfinite_and_low_conf_and_caps(self):
        pts = np.arange(2 * 4 * 5 * 3, dtype=np.float64).reshape(2, 4, 5, 3)
        pts[0, 0, 0] = np.nan
        cols = np.ones_like(pts) * 0.5
        conf = np.arange(40, dtype=np.float64).reshape(2, 4, 5)
        out_pts, out_cols = export.sparsify_points(pts, cols, conf, 50.0, 10)
        self.assertEqual(out_pts.shape, (10, 3))
        self.assertEqual(out_cols.shape, (10, 3))
        self.assertTrue(np.isfinite(out_pts).all())

    def test_sparsify_raises_when_nothing_finite(self):
        pts = np.full((1, 2, 2, 3), np.nan)
        with self.assertRaises(RuntimeError):
            export.sparsify_points(pts, np.zeros_like(pts), None, 0, 100)

    def test_main_fails_friendly_when_weights_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as ctx:
                export.main(["--input", tmp, "--model_path", f"{tmp}/nope.pt", "--output_dir", tmp])
            self.assertIn("Model weights not found", str(ctx.exception.code))


class OrbitHelpers(unittest.TestCase):
    def test_look_at_points_forward(self):
        R, t = orbit.look_at_matrix([0, 0, -5], [0, 0, 0], [0, -1, 0])
        cam = R @ np.zeros(3) + t
        self.assertAlmostEqual(cam[2], 5.0)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)

    def test_render_frame_draws_centre_point(self):
        pts = np.array([[0.0, 0.0, 0.0]])
        cols = np.array([[1.0, 0.0, 0.0]])
        img = orbit.render_frame(pts, cols, np.array([0, 0, -5.0]), np.zeros(3),
                                 np.array([0, -1.0, 0]), 64, 48)
        self.assertEqual(img.shape, (48, 64, 3))
        self.assertEqual(tuple(img[24, 32]), (255, 0, 0))
        self.assertEqual(tuple(img[0, 0]), (245, 245, 245))

    def test_main_fails_friendly_when_input_missing(self):
        with self.assertRaises(SystemExit) as ctx:
            orbit.main(["--input", "/nonexistent/lab.ply"])
        self.assertIn("Point cloud not found", str(ctx.exception.code))


class ShellScripts(unittest.TestCase):
    def test_help_exits_zero(self):
        for name in ("setup.sh", "download_model.sh", "run_lab.sh", "export_full_lab_glb.sh"):
            r = subprocess.run(["bash", str(SCRIPTS / name), "--help"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, name)
            self.assertTrue(r.stdout.strip(), name)

    def test_run_lab_fails_friendly_without_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp)
            (fake / "scripts").mkdir()
            shutil.copy(SCRIPTS / "run_lab.sh", fake / "scripts")
            venv = fake / "lingbot-map" / ".venv" / "bin"
            venv.mkdir(parents=True)
            (venv / "activate").write_text("")
            r = subprocess.run(["bash", str(fake / "scripts" / "run_lab.sh")],
                               capture_output=True, text=True, env={**os.environ})
            self.assertEqual(r.returncode, 1)
            self.assertIn("Model weights not found", r.stderr)


if __name__ == "__main__":
    sys.exit(unittest.main())
