"""Commanded-physics velocity: the quadrotor is its own accelerometer."""
from __future__ import annotations

import math
import unittest

import numpy as np

from ekf.commanded_accel import (
    G,
    BodyVelocityIntegrator,
    commanded_accel_ned,
    gravity_in_body,
    hover_is_observable,
    observe_hover_trim,
    specific_thrust,
)
from ekf.drone_ekf import DroneEKF, quat_from_rpy, quat_to_rot
from ekf.drag_ekf import DragEKF


class CommandedAccelTests(unittest.TestCase):
    def test_level_hover_is_zero(self):
        a = commanded_accel_ned(np.eye(3), 0.255, 0.255)
        np.testing.assert_allclose(a, 0.0, atol=1e-12)

    def test_extra_collective_climbs(self):
        # NED z down: more than hover → negative a_d.
        a = commanded_accel_ned(np.eye(3), 0.306, 0.255)
        self.assertAlmostEqual(a[0], 0.0, places=12)
        self.assertAlmostEqual(a[1], 0.0, places=12)
        self.assertAlmostEqual(a[2], -0.2 * G, places=6)

    def test_nose_down_makes_north_accel(self):
        pitch = math.radians(-20.0)
        R = quat_to_rot(quat_from_rpy(0.0, pitch, 0.0))
        a = commanded_accel_ned(R, 0.255, 0.255)
        self.assertGreater(a[0], 1.0)
        self.assertAlmostEqual(a[1], 0.0, places=9)

    def test_drag_opposes_body_velocity(self):
        v = np.array([5.0, 0.0, 0.0])
        k = np.array([-0.5, -0.5, -0.15])
        a = commanded_accel_ned(np.eye(3), 0.255, 0.255, v, k)
        self.assertAlmostEqual(a[0], -2.5, places=6)

    def test_specific_thrust_scales_from_trim(self):
        self.assertAlmostEqual(specific_thrust(0.255, 0.255), G)
        self.assertAlmostEqual(specific_thrust(0.510, 0.255), 2.0 * G)


class HoverTrimObserverTests(unittest.TestCase):
    def test_quiet_hover_walks_toward_collective(self):
        trim = observe_hover_trim(
            0.255,
            0.270,
            roll=0.0,
            pitch=0.0,
            rates_rad_s=np.zeros(3),
            vel_d=0.05,
            dt=2.0,
            tau_s=2.0,
        )
        self.assertGreater(trim, 0.255)
        self.assertLess(trim, 0.270)

    def test_slam_does_not_rewrite_trim(self):
        trim = observe_hover_trim(
            0.255,
            0.40,
            roll=0.0,
            pitch=math.radians(35.0),
            rates_rad_s=np.array([0.0, 3.0, 0.0]),
            vel_d=-2.0,
            dt=0.02,
            tau_s=2.0,
        )
        self.assertAlmostEqual(trim, 0.255)

    def test_observable_gate(self):
        self.assertTrue(
            hover_is_observable(0.0, 0.0, np.zeros(3), 0.0)
        )
        self.assertFalse(
            hover_is_observable(0.0, math.radians(20.0), np.zeros(3), 0.0)
        )


class DroneEKFCommandedTests(unittest.TestCase):
    def test_level_hover_holds(self):
        ekf = DroneEKF(hover_trim=0.255, drag_k_body=None)
        t = 0.0
        gyro = np.zeros(3)
        fake_imu = np.array([20.0, -8.0, 0.0])  # must be ignored
        for _ in range(100):
            t += 0.01
            ekf.predict(gyro, fake_imu, t, thrust=0.255, hover_trim=0.255)
        st = ekf.state()
        self.assertLess(abs(st.velocity_ned[0]), 0.05)
        self.assertLess(abs(st.velocity_ned[1]), 0.05)
        self.assertLess(abs(st.velocity_ned[2]), 0.05)

    def test_imu_accel_cannot_invent_velocity(self):
        ekf = DroneEKF(hover_trim=0.255, drag_k_body=None)
        t = 0.0
        gyro = np.zeros(3)
        fake_imu = np.array([50.0, 50.0, -9.8])
        for _ in range(50):
            t += 0.01
            ekf.predict(gyro, fake_imu, t, thrust=0.255)
        st = ekf.state()
        self.assertLess(float(np.linalg.norm(st.velocity_ned)), 0.08)

    def test_climb_collective_makes_up_velocity(self):
        ekf = DroneEKF(hover_trim=0.255, drag_k_body=None)
        t = 0.0
        gyro = np.zeros(3)
        accel = np.array([0.0, 0.0, -G])
        for _ in range(100):
            t += 0.01
            ekf.predict(gyro, accel, t, thrust=0.306, hover_trim=0.255)
        # 0.2 g climb for 1 s → ~2 m/s up = −2 m/s NED.
        self.assertAlmostEqual(ekf.state().velocity_ned[2], -0.2 * G, delta=0.15)

    def test_legacy_imu_path_still_available(self):
        ekf = DroneEKF(use_commanded_accel=False)
        t = 0.0
        gyro = np.zeros(3)
        accel = np.array([0.0, 0.0, -G])
        for _ in range(50):
            t += 0.002
            ekf.predict(gyro, accel, t)
        self.assertLess(abs(ekf.state().velocity_ned[2]), 0.5)


class BodyVelocityIntegratorTests(unittest.TestCase):
    def test_level_hover_stays_put(self):
        integ = BodyVelocityIntegrator(hover_trim=0.255, k_body=np.zeros(3))
        for _ in range(100):
            v = integ.step(0.01, 0.255, 0.0, 0.0, np.zeros(3))
        self.assertLess(float(np.linalg.norm(v)), 0.05)

    def test_nose_down_builds_forward_speed(self):
        integ = BodyVelocityIntegrator(hover_trim=0.255, k_body=np.zeros(3))
        pitch = math.radians(-20.0)
        for _ in range(50):
            v = integ.step(0.01, 0.255, 0.0, pitch, np.zeros(3))
        self.assertGreater(v[0], 0.5)

    def test_gravity_body_ignores_yaw(self):
        g0 = gravity_in_body(0.1, -0.2)
        # Same lean, any heading: formula has no yaw.
        self.assertAlmostEqual(g0[0], -G * math.sin(-0.2), places=9)


class DragEKFCommandedTests(unittest.TestCase):
    def test_commanded_hover_does_not_use_accel(self):
        ekf = DragEKF(k_x=-0.5, k_y=-0.5, hover_trim=0.255)
        ekf.reset()
        accel = np.array([-4.0, 3.0, -G])
        gyro = np.zeros(3)
        for _ in range(50):
            ekf.predict(
                accel, gyro, 0.0, 0.0, 0.0, 0.01,
                thrust=0.255, hover_trim=0.255,
            )
        v = ekf.body_velocity_xy()
        self.assertAlmostEqual(v[0], 0.0, delta=0.05)
        self.assertAlmostEqual(v[1], 0.0, delta=0.05)


if __name__ == '__main__':
    unittest.main()
