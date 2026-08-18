"""Autonomous policy flyer: env, flags, and pad-vision arm rules."""
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

from tools.run_policy import (
    DEFAULT_WEIGHTS,
    apply_flight_env,
    build_parser,
)


class ApplyFlightEnvTests(unittest.TestCase):
    def setUp(self):
        self._saved = {
            key: os.environ.get(key)
            for key in (
                'FLIGHT_MODE', 'POLICY_WEIGHTS', 'YOLO_POSE_MODEL_PATH',
                'POLICY_LOOP_HZ', 'RUN_MAX_SECONDS', 'OBS_PANEL',
                'EKF_USE_PNP', 'VISION_DISPLAY', 'TAKEOFF_DURATION_S',
                'GATE_DETECTOR_BACKEND',
            )
        }
        for key in self._saved:
            os.environ.pop(key, None)

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_defaults_are_policy_and_seed_weights(self):
        args = SimpleNamespace(
            weights=DEFAULT_WEIGHTS, yolo=None, hz=20.0, seconds=0.0,
            panel=False, panel_scale=1.0, detector=None,
        )
        applied = apply_flight_env(args)
        self.assertEqual(os.environ['FLIGHT_MODE'], 'policy')
        self.assertEqual(os.environ['POLICY_WEIGHTS'], DEFAULT_WEIGHTS)
        self.assertEqual(os.environ['EKF_USE_PNP'], '0')
        self.assertNotIn('OBS_PANEL', os.environ)
        self.assertEqual(applied['FLIGHT_MODE'], 'policy')

    def test_cli_overrides_weights_yolo_and_panel(self):
        args = build_parser().parse_args([
            '--weights', 'models/other.pt',
            '--yolo', 'models/gate_pose_v5.pt',
            '--panel',
            '--hz', '10',
            '--seconds', '90',
        ])
        apply_flight_env(args)
        self.assertEqual(os.environ['POLICY_WEIGHTS'], 'models/other.pt')
        self.assertEqual(os.environ['YOLO_POSE_MODEL_PATH'], 'models/gate_pose_v5.pt')
        self.assertEqual(os.environ['OBS_PANEL'], '1')
        self.assertEqual(os.environ['POLICY_LOOP_HZ'], '10.0')
        self.assertEqual(os.environ['RUN_MAX_SECONDS'], '90.0')


class PadVisionTests(unittest.TestCase):
    def test_policy_arms_on_yolo_not_pnp(self):
        from main import _pad_wants_pnp
        self.assertFalse(_pad_wants_pnp('policy'))
        self.assertFalse(_pad_wants_pnp('assist'))
        self.assertFalse(_pad_wants_pnp('race'))
        self.assertTrue(_pad_wants_pnp('kalman'))
        self.assertTrue(_pad_wants_pnp('spline'))


class TuneFlightPolicyModeTests(unittest.TestCase):
    def test_parser_has_policy_subcommand(self):
        from tools.tune_flight import build_parser
        args = build_parser().parse_args([
            'policy', '--weights', 'models/policy_seed_17.pt', '--panel',
        ])
        self.assertEqual(args.mode, 'policy')
        self.assertEqual(args.weights, 'models/policy_seed_17.pt')
        self.assertTrue(args.panel)


if __name__ == '__main__':
    unittest.main()
