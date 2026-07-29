"""Tests for the YOLO-corner PnP solver (vision/yolo_pnp.py)."""

import unittest

import cv2
import numpy as np

import camera_model as cm
from vision.yolo_pnp import (
    MIN_CORNER_SPREAD_PX,
    OBJECT_POINTS,
    solve_corners_pnp,
)


def project_gate(rvec, tvec):
    """Project the gate's outer corners into pixels for a known pose."""
    projected, _ = cv2.projectPoints(
        OBJECT_POINTS,
        np.asarray(rvec, float),
        np.asarray(tvec, float),
        cm.K,
        None,
    )
    return projected.reshape(-1, 2)


class SolveCornersPnPTests(unittest.TestCase):
    def test_recovers_known_pose(self):
        # Gate 6 m ahead, slightly off-axis and yawed: a realistic approach view.
        rvec = np.array([0.05, -0.20, 0.02])
        tvec = np.array([0.8, -0.4, 6.0])
        corners = project_gate(rvec, tvec)
        gate = solve_corners_pnp(corners, confidence=0.9)
        self.assertIsNotNone(gate)
        self.assertLess(gate.reproj_err_px, 0.5)
        np.testing.assert_allclose(gate.t_cg, tvec, atol=0.05)
        self.assertAlmostEqual(
            gate.range_m, float(np.linalg.norm(tvec)), delta=0.05
        )
        expected_rotation, _ = cv2.Rodrigues(rvec)
        np.testing.assert_allclose(gate.R_cg, expected_rotation, atol=0.02)

    def test_center_body_points_forward(self):
        # Straight-ahead gate: the camera is tilted 20 deg up from the body
        # axis, so the optical +Z range vector maps mostly onto body +X.
        corners = project_gate([0.0, 0.0, 0.0], [0.0, 0.0, 8.0])
        gate = solve_corners_pnp(corners)
        self.assertIsNotNone(gate)
        body = gate.center_body()
        self.assertGreater(body[0], 7.0)
        self.assertAlmostEqual(float(body[1]), 0.0, delta=0.1)

    def test_degenerate_spread_is_rejected(self):
        cluster = np.full((4, 2), 320.0)
        cluster += np.random.default_rng(0).uniform(
            -MIN_CORNER_SPREAD_PX / 4.0, MIN_CORNER_SPREAD_PX / 4.0, (4, 2)
        )
        self.assertIsNone(solve_corners_pnp(cluster))

    def test_beyond_max_range_is_rejected(self):
        # At 60 m the quad still clears the pixel-spread floor
        # (320 * 2.7 / 60 = 14.4 px) but the range gate must drop it: corner
        # pixels are sub-pixel noise out there.
        corners = project_gate([0.0, 0.0, 0.0], [0.0, 0.0, 60.0])
        self.assertIsNone(solve_corners_pnp(corners))

    def test_wrong_corner_order_is_rejected(self):
        rvec = np.array([0.10, -0.30, 0.05])
        corners = project_gate(rvec, [0.5, -0.3, 6.0])
        swapped = corners[[1, 0, 2, 3]]   # TL <-> TR
        self.assertIsNone(solve_corners_pnp(swapped))

    def test_nonfinite_corners_are_rejected(self):
        corners = project_gate([0.0, 0.0, 0.0], [0.0, 0.0, 6.0])
        corners[2, 0] = float('nan')
        self.assertIsNone(solve_corners_pnp(corners))


if __name__ == '__main__':
    unittest.main()
