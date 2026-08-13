"""The logger must record sim ODOMETRY separately from the estimator's belief.

pos_*/vel_* come from position_ned / local_position_ned, which on this build is
IMU dead reckoning -- earlier logs held those fields populated 100% of the time
while ranging to +81264 m and -10770500 m. odo_* must carry the sim's measured
state instead, or HG-DAgger has no ground truth to validate the observation
against and every recorded run is unusable for Phase 3.
"""
from __future__ import annotations

import csv
import tempfile
import time
import unittest
from pathlib import Path

from logger import Logger


def _row_from(shared: dict) -> dict:
    """Drive Logger._write_row once and read back the single CSV row."""
    with tempfile.TemporaryDirectory() as tmp:
        logger = Logger.__new__(Logger)
        logger.data = shared
        logger._t0 = time.time()
        logger._csv_writer = None
        logger._csv_file = None
        logger._csv_path = str(Path(tmp) / 'telem.csv')
        logger._write_row()
        logger._csv_file.close()
        with open(logger._csv_path, newline='') as fh:
            return next(iter(csv.DictReader(fh)))


class LoggerOdometryTests(unittest.TestCase):
    def test_odometry_columns_are_written(self):
        shared = {
            'odometry': {
                'x': 1.5, 'y': -2.25, 'z': -3.0,
                'vx': 4.0, 'vy': 0.5, 'vz': -0.25,
                'q': [1.0, 0.0, 0.0, 0.0],
                'roll': 0.1, 'pitch': -0.2, 'yaw': 1.57,
                'rollspeed': 0.01, 'pitchspeed': 0.02, 'yawspeed': 0.03,
            },
        }
        row = _row_from(shared)
        self.assertAlmostEqual(float(row['odo_x']), 1.5, places=3)
        self.assertAlmostEqual(float(row['odo_y']), -2.25, places=3)
        self.assertAlmostEqual(float(row['odo_z']), -3.0, places=3)
        self.assertAlmostEqual(float(row['odo_vx']), 4.0, places=3)
        self.assertAlmostEqual(float(row['odo_qw']), 1.0, places=3)
        self.assertAlmostEqual(float(row['odo_yaw']), 1.57, places=3)
        self.assertAlmostEqual(float(row['odo_yawspeed']), 0.03, places=3)

    def test_odometry_is_not_the_estimator_belief(self):
        """A diverged EKF must not leak into the odo_* columns."""
        shared = {
            'position_ned': {
                'x': 81264.4, 'y': -10770500.0, 'z': -5.0,
                'vx': 900.0, 'vy': 0.0, 'vz': 0.0,
            },
            'odometry': {
                'x': 2.0, 'y': 3.0, 'z': -1.0,
                'vx': 1.0, 'vy': 0.0, 'vz': 0.0,
            },
        }
        row = _row_from(shared)
        # The belief still lands in pos_*, unchanged.
        self.assertAlmostEqual(float(row['pos_n']), 81264.4, places=1)
        # Ground truth stays clean.
        self.assertAlmostEqual(float(row['odo_x']), 2.0, places=3)
        self.assertAlmostEqual(float(row['odo_y']), 3.0, places=3)

    def test_missing_odometry_is_nan_not_zero(self):
        """VQ2 publishes no ODOMETRY; absence must be nan, never a fake 0.0."""
        row = _row_from({})
        for key in ('odo_x', 'odo_y', 'odo_z', 'odo_vx', 'odo_qw', 'odo_yaw'):
            self.assertEqual(row[key], 'nan', f'{key} should be nan when absent')

    def test_control_authority_defaults_to_policy(self):
        row = _row_from({})
        self.assertEqual(row['control_authority'], 'policy')
        self.assertEqual(row['intervention_id'], '')

    def test_control_authority_records_human_takeover(self):
        row = _row_from({'control_authority': 'human', 'intervention_id': 7})
        self.assertEqual(row['control_authority'], 'human')
        self.assertEqual(row['intervention_id'], '7')


if __name__ == '__main__':
    unittest.main()
