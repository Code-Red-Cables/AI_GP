"""Synthetic tests for bearing-vector LS gate pose."""
from __future__ import annotations

import math
import unittest

import numpy as np

import camera_model as cm
from vision.gate_ls_pose import (
    GATE_OUTER_M,
    solve_keypoints_ls,
    solve_ring_ls,
)
from vision.yolo_pnp import KEYPOINT_OBJECT_POINTS, OUTER_RING_IDX


def _project_gate(
    t_gate: np.ndarray,
    *,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    gate_yaw: float = 0.0,
    noise_px: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project the 8 keypoints into the image for a known camera pose."""
    # NED <- gate, then gate <- NED for camera pose construction.
    cz, sz = math.cos(gate_yaw), math.sin(gate_yaw)
    z = np.array([cz, sz, 0.0])
    y = np.array([0.0, 0.0, 1.0])
    x = np.cross(y, z)
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    R_ng = np.column_stack([x, y, z])  # NED <- gate

    # Camera attitude in NED from AHRS.
    R_wb = cm.rot_world_body(roll, pitch, yaw)  # NED <- body
    R_wc = R_wb @ cm.R_BC                       # NED <- cam

    # Camera origin in NED = R_ng @ t_gate (t_gate is camera in gate frame).
    cam_ned = R_ng @ t_gate

    pts = []
    for p_gate in KEYPOINT_OBJECT_POINTS:
        p_ned = R_ng @ p_gate
        p_cam = R_wc.T @ (p_ned - cam_ned)
        assert p_cam[2] > 0.05, f'point behind camera: {p_cam}'
        u, v = cm.project(p_cam)
        pts.append([u, v])
    pts = np.asarray(pts, dtype=np.float64)
    if noise_px > 0.0:
        rng = rng or np.random.default_rng(0)
        pts = pts + rng.normal(0.0, noise_px, size=pts.shape)
    conf = np.ones(len(pts), dtype=np.float64)
    return pts, conf


class GateLSPoseTests(unittest.TestCase):
    def test_recovers_fronto_parallel_pose(self):
        # Camera 8 m in front of the gate, on centreline, level.
        t_true = np.array([0.0, 0.0, -8.0])
        pts, conf = _project_gate(t_true, yaw=0.0, gate_yaw=0.0)
        pose = solve_keypoints_ls(pts, conf, roll=0.0, pitch=0.0, yaw=0.0)
        self.assertIsNotNone(pose)
        assert pose is not None
        np.testing.assert_allclose(pose.t_gate, t_true, atol=0.05)
        self.assertLess(pose.residual_m, 0.05)
        self.assertAlmostEqual(pose.range_m, 8.0, delta=0.05)

    def test_recovers_lateral_offset(self):
        t_true = np.array([1.2, -0.4, -10.0])
        pts, conf = _project_gate(t_true)
        pose = solve_keypoints_ls(pts, conf)
        self.assertIsNotNone(pose)
        assert pose is not None
        np.testing.assert_allclose(pose.t_gate, t_true, atol=0.08)
        self.assertAlmostEqual(pose.lateral_m, 1.2, delta=0.08)
        self.assertAlmostEqual(pose.vertical_m, -0.4, delta=0.08)

    def test_ring_consistency_on_clean_projection(self):
        t_true = np.array([0.3, 0.1, -7.0])
        pts, conf = _project_gate(t_true)
        pose = solve_keypoints_ls(pts, conf)
        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertIsNotNone(pose.outer_t)
        self.assertIsNotNone(pose.inner_t)
        self.assertLess(pose.ring_disagree_m, 0.05)

    def test_tolerates_pixel_noise(self):
        t_true = np.array([0.0, 0.0, -8.0])
        rng = np.random.default_rng(1)
        errs = []
        for _ in range(20):
            pts, conf = _project_gate(t_true, noise_px=2.0, rng=rng)
            pose = solve_keypoints_ls(pts, conf)
            self.assertIsNotNone(pose)
            assert pose is not None
            errs.append(float(np.linalg.norm(pose.t_gate - t_true)))
        # Mean error under ~0.5 m with 2 px noise at 8 m is acceptable —
        # range is the fragile axis; bearing stays usable.
        self.assertLess(float(np.mean(errs)), 0.6)

    def test_outer_ring_only(self):
        t_true = np.array([-0.5, 0.2, -9.0])
        pts, conf = _project_gate(t_true)
        half = GATE_OUTER_M / 2.0
        obj = np.array([
            [-half, -half, 0.0],
            [+half, -half, 0.0],
            [+half, +half, 0.0],
            [-half, +half, 0.0],
        ])
        outer = pts[list(OUTER_RING_IDX)]
        pose = solve_ring_ls(outer, obj)
        self.assertIsNotNone(pose)
        assert pose is not None
        np.testing.assert_allclose(pose.t_gate, t_true, atol=0.1)

    def test_rejects_too_few_points(self):
        pts = np.zeros((8, 2))
        conf = np.zeros(8)
        conf[:3] = 1.0
        pts[:3] = [[100, 100], [200, 100], [200, 200]]
        self.assertIsNone(solve_keypoints_ls(pts, conf, min_keypoint_confidence=0.5))


if __name__ == '__main__':
    unittest.main()
