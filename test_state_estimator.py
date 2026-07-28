"""Unit tests for the VIO state estimator (IMU dead-reckoning + PnP fixes).

The estimator's background thread is stopped immediately after construction;
each test drives ``_propagate`` / ``_apply_fix`` / ``_publish`` directly so
the fusion behaviour is deterministic.
"""

import math
import os
import tempfile
import threading
import time
import unittest

import numpy as np

import camera_model as cm
import state_estimator as se
from state_estimator import G, P_GATE2NED, StateEstimator


def make_synthetic_fix(R_gb, p_gb, ts=None):
    """Build a pnp_fix dict from a desired body pose in the gate-NED frame.

    Inverts the reconstruction in StateEstimator._apply_fix:
      R_gb = P_GATE2NED @ R_cg.T @ R_CB    and
      p_gb = P_GATE2NED @ (-R_cg.T @ t_cg)
    """
    R_gb = np.asarray(R_gb, float)
    p_gb = np.asarray(p_gb, float)
    R_cg = (P_GATE2NED.T @ R_gb @ cm.R_CB.T).T
    t_cg = -R_cg @ (P_GATE2NED.T @ p_gb)
    return {
        'ts': int(ts if ts is not None else time.time_ns()),
        'R_cg': R_cg.tolist(),
        't_cg': t_cg.tolist(),
        'reproj_err_px': 0.5,
    }


def rot_x(angle_rad):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


class StateEstimatorTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.data = {'lock': threading.Lock(), 'flight_started': False}
        self.est = StateEstimator(
            self.data,
            anchors_path=os.path.join(self._tmp, 'anchors.json'),
        )
        # Stop the background loop; tests drive the estimator directly.
        self.est.get_thread_for_join().join(timeout=2.0)
        self.assertFalse(self.est.thread.is_alive())

    def _feed_imu(self, samples, accel, gyro=(0.0, 0.0, 0.0), dt_us=5000):
        start = self.est._last_imu_us or 1_000_000
        for index in range(samples):
            self.est._propagate({
                'ts_us': start + (index + 1) * dt_us,
                'xacc': accel[0], 'yacc': accel[1], 'zacc': accel[2],
                'xgyro': gyro[0], 'ygyro': gyro[1], 'zgyro': gyro[2],
            })

    # ------------------------------------------------------------------
    def test_attitude_converges_to_gravity(self):
        self.est._roll = 0.3
        self.est._pitch = -0.2
        # Stationary drone: accelerometer reads pure minus-gravity in body.
        self._feed_imu(400, accel=(0.0, 0.0, -G))
        self.assertLess(abs(self.est._roll), 0.01)
        self.assertLess(abs(self.est._pitch), 0.01)

    def test_preflight_zupt_pins_position(self):
        # Slightly wrong accel would normally dead-reckon the belief away,
        # but before flight_started the drone is parked on its spawn point.
        self._feed_imu(400, accel=(0.5, 0.0, -G + 0.5))
        self.assertEqual(float(np.linalg.norm(self.est._pos)), 0.0)
        self.assertEqual(float(np.linalg.norm(self.est._vel)), 0.0)

    def test_blind_flight_velocity_decay(self):
        self.data['flight_started'] = True
        self.est._last_fix_wall = time.monotonic() - 10.0  # blind regime
        self.est._vel = np.array([2.0, 0.0, 0.0])
        # Level hover: specific force cancels gravity, so a_ned = 0 and the
        # only velocity change is the blind leak.
        self._feed_imu(200, accel=(0.0, 0.0, -G))
        self.assertLess(abs(self.est._vel[0]), 0.2 * 2.0)

    # ------------------------------------------------------------------
    def test_first_fix_anchors_gate(self):
        fix = make_synthetic_fix(np.eye(3), [-5.0, 0.0, 0.0])
        self.est._apply_fix(fix, 0)
        self.assertIn(0, self.est.anchors)
        self.assertEqual(self.est._n_fix, 1)
        # Anchored from the origin belief with the drone 5 m before the
        # gate, the gate itself sits 5 m ahead in the world frame.
        anchor = np.asarray(self.est.anchors[0]['pos'], float)
        np.testing.assert_allclose(anchor, [5.0, 0.0, 0.0], atol=1e-9)

    def test_fix_blends_position_toward_measurement(self):
        first = make_synthetic_fix(np.eye(3), [-5.0, 0.0, 0.0])
        self.est._apply_fix(first, 0)
        # Dead-reckoning walked 1 m off; the same gate view must pull the
        # belief back toward the measured position (the origin).
        self.est._pos = np.array([1.0, 0.0, 0.0])
        second = make_synthetic_fix(
            np.eye(3), [-5.0, 0.0, 0.0], ts=first['ts'] + int(0.2e9)
        )
        self.est._apply_fix(second, 0)
        self.assertEqual(self.est._n_fix, 2)
        self.assertAlmostEqual(
            float(self.est._pos[0]), 1.0 - se.FIX_POS_GAIN, places=6
        )

    def test_fix_corrects_roll_pitch_absolutely(self):
        first = make_synthetic_fix(np.eye(3), [-5.0, 0.0, 0.0])
        self.est._apply_fix(first, 0)
        self.est._roll = 0.2
        second = make_synthetic_fix(
            np.eye(3), [-5.0, 0.0, 0.0], ts=first['ts'] + int(0.2e9)
        )
        self.est._apply_fix(second, 0)
        # Gates hang upright: a level gate view is an absolute roll fix.
        self.assertAlmostEqual(
            self.est._roll, 0.2 * (1.0 - se.FIX_RP_GAIN), places=6
        )

    def test_wild_tilt_fix_is_rejected(self):
        fix = make_synthetic_fix(rot_x(math.radians(70.0)), [-5.0, 0.0, 0.0])
        self.est._apply_fix(fix, 0)
        self.assertEqual(self.est._n_fix, 0)
        self.assertEqual(self.est._n_fix_rej, 1)
        self.assertNotIn(0, self.est.anchors)

    def test_large_innovation_fix_is_rejected(self):
        first = make_synthetic_fix(np.eye(3), [-5.0, 0.0, 0.0])
        self.est._apply_fix(first, 0)
        self.est._pos = np.array([se.INNOV_REJECT_M + 2.0, 0.0, 0.0])
        second = make_synthetic_fix(
            np.eye(3), [-5.0, 0.0, 0.0], ts=first['ts'] + int(0.2e9)
        )
        self.est._apply_fix(second, 0)
        self.assertEqual(self.est._n_fix, 1)
        self.assertEqual(self.est._n_fix_rej, 1)

    # ------------------------------------------------------------------
    def test_publish_owns_attitude_and_position(self):
        self.est._publish()
        attitude = self.data['attitude']
        position = self.data['position_ned']
        self.assertEqual(attitude['source'], 'vio')
        self.assertEqual(position['source'], 'vio')
        for key in ('roll', 'pitch', 'yaw', 'rollspeed', 'ts'):
            self.assertIn(key, attitude)
        for key in ('x', 'y', 'z', 'vx', 'vy', 'vz', 'ts'):
            self.assertIn(key, position)


if __name__ == '__main__':
    unittest.main()
