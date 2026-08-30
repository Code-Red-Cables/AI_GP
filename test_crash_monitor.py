"""Policy crash re-arm: floor hits only — not EKF z runaway."""
from __future__ import annotations

import time
import unittest

import config
from main import CrashMonitor, _crash_cooldown_s, _crash_grace_s, _env_slam


class CrashMonitorPolicyTests(unittest.TestCase):
    def setUp(self):
        self._mode = config.FLIGHT_MODE
        self._confirm = config.CRASH_CONFIRM_S
        config.CRASH_CONFIRM_S = 0.20

    def tearDown(self):
        config.FLIGHT_MODE = self._mode
        config.CRASH_CONFIRM_S = self._confirm

    def _armed(self, mode: str, shared=None) -> CrashMonitor:
        config.FLIGHT_MODE = mode
        shared = shared if shared is not None else {
            'local_position_ned': {'z': 0.0},
        }
        monitor = CrashMonitor()
        monitor.note_armed(shared)
        monitor.grace_until = 0.0
        monitor.last_reset_at = 0.0
        return monitor

    def _confirmed(self, monitor, shared) -> bool:
        now = time.monotonic()
        monitor.update(shared, now)
        return monitor.update(shared, now + 0.25)

    def test_policy_grace_and_cooldown_are_short(self):
        config.FLIGHT_MODE = 'policy'
        self.assertAlmostEqual(_crash_grace_s(), 0.40)
        self.assertAlmostEqual(_crash_cooldown_s(), 0.40)
        config.FLIGHT_MODE = 'kalman'
        self.assertGreaterEqual(_crash_grace_s(), 2.0)
        self.assertGreaterEqual(_crash_cooldown_s(), 2.0)

    def test_note_armed_writes_attempt(self):
        config.FLIGHT_MODE = 'policy'
        shared = {'local_position_ned': {'z': 0.0}}
        monitor = CrashMonitor()
        monitor.note_armed(shared)
        self.assertEqual(shared['attempt'], 1)
        monitor.note_armed(shared)
        self.assertEqual(shared['attempt'], 2)

    def test_policy_ignores_ekf_z_runaway(self):
        """092128: EKF pos_d → +23 m mid-air must not auto-reset."""
        shared = {
            'local_position_ned': None,
            'position_ned': {'z': 23.5},
        }
        monitor = self._armed('policy', shared)
        self.assertFalse(self._confirmed(monitor, shared))

    def test_policy_odo_below_spawn_after_climb_is_a_crash(self):
        shared = {'local_position_ned': {'z': 0.0}}
        monitor = self._armed('policy', shared)
        shared['local_position_ned'] = {'z': -1.0}
        self.assertFalse(self._confirmed(monitor, shared))
        shared['local_position_ned'] = {'z': 0.60}
        self.assertTrue(self._confirmed(monitor, shared))

    def test_policy_hard_env_slam_without_z(self):
        shared = {
            'local_position_ned': None,
            'position_ned': {'z': 4.0},
            'collision': {
                'id': 1002,
                'impulse': 2.01,
                'ts': time.time_ns(),
            },
        }
        monitor = self._armed('policy', shared)
        self.assertTrue(self._confirmed(monitor, shared))

    def test_policy_soft_pad_scrape_is_not_a_crash(self):
        config.FLIGHT_MODE = 'policy'
        shared = {
            'collision': {
                'id': 1002,
                'impulse': 0.17,
                'ts': time.time_ns(),
            },
        }
        self.assertFalse(_env_slam(shared, climbed=0.0, peak_climb=0.0))

    def test_kalman_still_requires_climb_for_env(self):
        config.FLIGHT_MODE = 'kalman'
        shared = {
            'collision': {
                'id': 1002,
                'impulse': 0.17,
                'ts': time.time_ns(),
            },
        }
        self.assertFalse(_env_slam(shared, climbed=0.0, peak_climb=0.0))

    def test_policy_invert_is_not_a_crash(self):
        shared = {
            'local_position_ned': {'z': -1.2},
            'attitude_raw': {'roll': 0.0, 'pitch': 1.80},
        }
        monitor = self._armed('policy', shared)
        self.assertFalse(self._confirmed(monitor, shared))


if __name__ == '__main__':
    unittest.main()
