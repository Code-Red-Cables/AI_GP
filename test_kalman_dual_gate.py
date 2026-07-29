"""Smoke tests for dual-gate PnP + EKF + cascaded PID (Q2_kalman)."""

from __future__ import annotations

import math
import unittest

import numpy as np

import config
from control.cascaded_pid import CascadedPIDController
from ekf.drone_ekf import DroneEKF
from planning.dual_gate_path import DualGatePathPlanner


class TestPnPOrientations(unittest.TestCase):
    def test_corners_canonical_order_and_body_signs(self):
        import cv2
        import camera_model as cm
        from vision.yolo_pnp import (
            OBJECT_POINTS,
            corners_tl_tr_bl_br,
            solve_corners_pnp,
        )
        from vision.gate_detector import order_corners

        R = np.eye(3)
        t = np.array([[1.5], [0.5], [6.0]])  # right + below in camera
        img, _ = cv2.projectPoints(
            OBJECT_POINTS, cv2.Rodrigues(R)[0], t, cm.K, None
        )
        true = img.reshape(4, 2)
        # Scramble + pass through cyclic orderer (TL,TR,BR,BL).
        scrambled = true[[3, 2, 0, 1]]
        cyclic = order_corners(scrambled)
        canon = corners_tl_tr_bl_br(scrambled)
        np.testing.assert_allclose(canon[0], true[0], atol=1e-6)  # TL
        np.testing.assert_allclose(canon[1], true[1], atol=1e-6)  # TR
        np.testing.assert_allclose(canon[2], true[2], atol=1e-6)  # BL
        np.testing.assert_allclose(canon[3], true[3], atol=1e-6)  # BR
        # Cyclic BR/BL swap must be fixed before PnP.
        self.assertFalse(np.allclose(cyclic[2], true[2], atol=1e-6))

        g = solve_corners_pnp(scrambled, confidence=0.9)
        self.assertIsNotNone(g)
        self.assertTrue(g.solved)
        body = g.center_body()
        self.assertGreater(body[0], 0.0)  # forward
        self.assertGreater(body[1], 0.0)  # right
        # Through (+Z) into scene; gate +Y along camera-down (upright only).
        self.assertGreater(float(g.R_cg[2, 2]), 0.0)
        self.assertGreater(float(g.R_cg[1, 1]), 0.25)

    def test_upright_gate_constraint_filters_inverted_R(self):
        from vision.yolo_pnp import _is_upright_through_solution

        upright = np.eye(3)
        self.assertTrue(_is_upright_through_solution(upright))
        # 180° about camera Z — gate hanging upside down.
        inverted = np.diag([-1.0, -1.0, 1.0])
        self.assertFalse(_is_upright_through_solution(inverted))
        # Through pointing back at the camera.
        facing_camera = np.diag([1.0, 1.0, -1.0])
        self.assertFalse(_is_upright_through_solution(facing_camera))


class TestDroneEKF(unittest.TestCase):
    def test_predict_integrates_hover_accel(self):
        ekf = DroneEKF()
        t = 0.0
        # Level hover specific force in FRD: body-z ≈ -g; filter adds +g NED.
        gyro = np.zeros(3)
        accel = np.array([0.0, 0.0, -9.80665])
        for _ in range(50):
            t += 0.002
            ekf.predict(gyro, accel, t)
        st = ekf.state()
        self.assertLess(abs(st.velocity_ned[2]), 0.5)
        self.assertLess(abs(st.position_ned[2]), 0.5)

    def test_gravity_aligns_parked_pitch(self):
        ekf = DroneEKF()
        t = 0.0
        # ~18° nose-up tip at rest (matches OLD-sim pad reading).
        pitch = math.radians(18.0)
        accel = np.array(
            [-9.80665 * math.sin(pitch), 0.0, -9.80665 * math.cos(pitch)]
        )
        gyro = np.zeros(3)
        for _ in range(20):
            t += 0.002
            ekf.predict(gyro, accel, t)
        roll, ekf_pitch, _yaw = ekf.state().roll_pitch_yaw
        self.assertAlmostEqual(roll, 0.0, delta=math.radians(2.0))
        self.assertAlmostEqual(ekf_pitch, pitch, delta=math.radians(2.0))

    def test_gate2_survives_fov_dropout(self):
        ekf = DroneEKF()
        t = 0.0
        gyro = np.zeros(3)
        accel = np.array([0.0, 0.0, -9.80665])
        for _ in range(10):
            t += 0.002
            ekf.predict(gyro, accel, t)
        g1 = np.array([5.0, 0.0, 0.0])
        g2 = np.array([12.0, 2.0, 0.0])
        ekf.correct_dual_gate_body(g1, g2, t)
        remembered = ekf.state().gate2_ned.copy()
        self.assertIsNotNone(remembered)
        # Gate 2 leaves FOV: correct with only Gate 1 for a while.
        for _ in range(200):
            t += 0.002
            ekf.predict(gyro, accel, t)
            if _ % 15 == 0:
                ekf.correct_dual_gate_body(g1, None, t)
        st = ekf.state()
        self.assertIsNotNone(st.gate2_ned)
        self.assertFalse(st.gate2_fresh)
        self.assertLess(
            float(np.linalg.norm(st.gate2_ned - remembered)),
            1.5,
        )


class TestDualGatePath(unittest.TestCase):
    def test_approach_look_yaw_points_at_gate1(self):
        from planning.dual_gate_path import body_look_yaw

        planner = DualGatePathPlanner()
        p = np.array([0.0, 0.0, 0.0])
        g1 = np.array([5.0, 0.5, 0.0])
        g2 = np.array([10.0, 5.0, 0.0])
        path = planner.plan(p, g1, g2)
        # Must NOT look at Gate 2 during approach (that yawed us off G1).
        expected_g1 = math.atan2(0.5, 5.0)
        self.assertAlmostEqual(path.look_yaw_rad, expected_g1, places=5)
        self.assertEqual(path.phase, 'approach')
        self.assertLess(path.target_ned[0], g1[0])

        # Body look centres Gate 1 even when Gate 2 is far to the side
        # (bearing outside the soft-blend deadband).
        look = body_look_yaw(
            0.0,
            np.array([5.0, 1.2, 0.0]),
            np.array([12.0, 6.0, 0.0]),
            phase='approach',
        )
        self.assertAlmostEqual(look, math.atan2(1.2, 5.0), places=5)


class TestCascadedPID(unittest.TestCase):
    def test_yaw_lookat_and_lean_outputs(self):
        ctl = CascadedPIDController()
        cmd = ctl.update(
            position_ned=np.zeros(3),
            velocity_ned=np.zeros(3),
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            target_ned=np.array([3.0, 1.0, 0.0]),
            look_yaw_rad=math.radians(30.0),
            dt=0.02,
        )
        self.assertGreater(abs(cmd.yaw_rate), 0.0)
        self.assertNotEqual(cmd.desired_pitch, 0.0)
        self.assertTrue(math.isfinite(cmd.thrust))
        self.assertTrue(math.isfinite(cmd.roll_rate))


class TestKalmanPlanner(unittest.TestCase):
    def test_hover_without_gates(self):
        from kalman_planner import KalmanDualGatePlanner

        planner = KalmanDualGatePlanner()
        data = {'attitude': {'yaw': 0.0}, 'position_ned': {}}
        tgt = planner.compute_target(data)
        self.assertTrue(tgt['kalman'])
        self.assertEqual(tgt['roll_rate'], 0.0)
        self.assertEqual(tgt['yaw_rate'], 0.0)
        self.assertGreater(tgt['thrust'], 0.0)

    def test_body_gate_centres_yaw_and_leans_forward(self):
        from kalman_planner import KalmanDualGatePlanner

        planner = KalmanDualGatePlanner()
        data = {
            'attitude': {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
            'position_ned': {'z': 0.0},
            'dual_gate_pnp': {
                'gate1_body': [5.0, 0.5, 0.1],
                'gate2_body': [12.0, 2.0, 0.0],
                'gate1_norm_x': 0.25,
                'gate1_norm_y': 0.0,
                'n_solved': 2,
            },
        }
        tgt = planner.compute_target(data)
        # Gate on right → positive planner yaw (RATE_SIGN_YAW applied later).
        self.assertGreater(tgt['yaw_rate'], 0.0)
        self.assertTrue(tgt['kalman'])

    def test_off_center_yaws_toward_gate(self):
        from kalman_planner import KalmanDualGatePlanner

        planner = KalmanDualGatePlanner()
        data = {
            'attitude': {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
            'position_ned': {'z': 0.0},
            'dual_gate_pnp': {
                'gate1_body': [5.0, -2.0, 0.0],
                'gate1_norm_x': -0.85,
                'gate1_norm_y': 0.1,
                'n_solved': 1,
            },
        }
        tgt = planner.compute_target(data)
        self.assertLess(tgt['yaw_rate'], 0.0)

    def test_centered_gate_does_not_rocket_thrust(self):
        from kalman_planner import KalmanDualGatePlanner

        planner = KalmanDualGatePlanner()
        data = {
            'attitude': {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
            'position_ned': {'z': 0.0},
            'dual_gate_pnp': {
                'gate1_body': [5.0, 0.0, -1.7],
                'gate1_norm_x': 0.0,
                'gate1_norm_y': 0.0,
                'n_solved': 1,
            },
        }
        tgt = planner.compute_target(data)
        self.assertLess(tgt['thrust'], config.HOVER_THRUST + 0.03)
        self.assertGreater(tgt['thrust'], config.HOVER_THRUST - 0.03)


if __name__ == '__main__':
    unittest.main()
