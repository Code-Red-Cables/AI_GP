"""Unit tests for the drag-model EKF."""
from __future__ import annotations

import math
import unittest

import numpy as np

from ekf.drag_ekf import DragEKF, G


class DragEKFTests(unittest.TestCase):
    def test_body_velocity_from_accel(self):
        ekf = DragEKF(k_x=-0.5, k_y=-0.5)
        ekf.reset()
        # a = k v  =>  v = a / k.  With a_x=-1 and k=-0.5, v_x=2.
        v = ekf.body_velocity_xy(np.array([-1.0, 0.0, -G]))
        self.assertAlmostEqual(v[0], 2.0, places=5)
        self.assertAlmostEqual(v[1], 0.0, places=5)

    def test_bias_integrates_once(self):
        """Constant accel bias must not quadratic-diverge position."""
        ekf = DragEKF(k_x=-0.5, k_y=-0.5)
        ekf.reset()
        # Inject a constant x-bias; with zero measured accel the inferred
        # velocity is -bias/k, so position walks linearly, not as t^2.
        ekf.state.x[ekf.IDX_BAX] = 0.5
        accel = np.array([0.0, 0.0, -G])
        gyro = np.zeros(3)
        for _ in range(100):
            ekf.predict(accel, gyro, roll=0.0, pitch=0.0, yaw=0.0, dt=0.01)
        # v_x = (0 - 0.5) / -0.5 = 1.0 m/s → ~1 m after 1 s.
        pos = ekf.position_ned()
        self.assertAlmostEqual(pos[0], 1.0, delta=0.15)
        self.assertLess(abs(pos[1]), 0.05)

    def test_vision_update_pulls_position(self):
        ekf = DragEKF(k_x=-0.5, k_y=-0.5, vision_noise=0.1)
        ekf.reset()
        ekf.state.x[0:3] = np.array([5.0, 0.0, 0.0])
        ekf.update_position(np.array([0.0, 0.0, 0.0]))
        pos = ekf.position_ned()
        self.assertLess(abs(pos[0]), 2.0)

    def test_rejects_zero_drag(self):
        with self.assertRaises(ValueError):
            DragEKF(k_x=0.0, k_y=-0.5)


if __name__ == '__main__':
    unittest.main()
