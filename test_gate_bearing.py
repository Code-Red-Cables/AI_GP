"""Unit tests for the Phase 3b observation hard-gate helpers."""
from __future__ import annotations

import math
import unittest

import numpy as np

from gate_bearing import (
    detector_bearing_rad,
    gate_by_active,
    keypoint_centroid_px,
    pearson,
    true_bearing_rad,
    true_gate_body,
)


class GateBearingTests(unittest.TestCase):
    def test_gate_by_active_accepts_probe_ids(self):
        gates = [
            {'gate_id': 0, 'position_ned': [10.0, 0.0, 0.0]},
            {'gate_id': 1, 'position_ned': [20.0, 5.0, 0.0]},
        ]
        g = gate_by_active(gates, 1)
        self.assertIsNotNone(g)
        self.assertEqual(g['id'], 1)
        self.assertEqual(g['pos'][0], 20.0)

    def test_true_bearing_straight_ahead(self):
        odo = {
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
        }
        gate = {'id': 0, 'pos': (10.0, 0.0, 0.0)}
        body = true_gate_body(odo, gate)
        self.assertIsNotNone(body)
        self.assertAlmostEqual(true_bearing_rad(body), 0.0, places=5)

    def test_true_bearing_to_the_right(self):
        odo = {
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
        }
        gate = {'id': 0, 'pos': (10.0, 10.0, 0.0)}
        body = true_gate_body(odo, gate)
        self.assertAlmostEqual(true_bearing_rad(body), math.pi / 4, places=5)

    def test_detector_bearing_matches_image_right(self):
        # Pixel to the right of centre should yield positive body-y bearing.
        b = detector_bearing_rad(400.0, 180.0)
        self.assertIsNotNone(b)
        self.assertGreater(b, 0.0)

    def test_centroid_ignores_unseen(self):
        kps = [(0.0, 0.0), (100.0, 50.0), (120.0, 60.0)] + [(0.0, 0.0)] * 5
        confs = [0.0, 0.9, 0.9] + [0.0] * 5
        c = keypoint_centroid_px(kps, confs)
        self.assertAlmostEqual(c[0], 110.0)
        self.assertAlmostEqual(c[1], 55.0)

    def test_pearson_perfect(self):
        xs = [0.0, 1.0, 2.0, 3.0]
        self.assertAlmostEqual(pearson(xs, xs), 1.0, places=6)
        self.assertTrue(math.isnan(pearson([1.0], [1.0])))


if __name__ == '__main__':
    unittest.main()
