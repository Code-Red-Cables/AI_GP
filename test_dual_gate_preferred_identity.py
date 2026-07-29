"""dual_gate_pnp must honour YOLO preferred identity (no nearest-range steal)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from vision.dual_gate_pnp import observe_two_closest_gates


def _fake_candidate(cx, cy, half, conf=0.9, source_index=0):
    """Minimal pose candidate with reliable outer corners for PnP."""
    corners = np.asarray(
        [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ],
        dtype=np.float32,
    )
    box = SimpleNamespace(
        center=(float(cx), float(cy)),
        area=float((2 * half) ** 2),
        confidence=float(conf),
        bbox=(cx - half, cy - half, cx + half, cy + half),
        source_index=source_index,
    )
    return SimpleNamespace(
        box=box,
        keypoints=corners,
        keypoint_confidences=np.ones(4, dtype=np.float32),
        hsv_confirmed=True,
    )


class DualGatePreferredIdentityTests(unittest.TestCase):
    def test_preferred_near_gate_wins_over_farther_solved(self):
        # Larger half → nearer in size/PnP; smaller half → farther.
        near = _fake_candidate(320, 160, 80, conf=0.85, source_index=0)
        far = _fake_candidate(400, 200, 25, conf=0.99, source_index=1)
        obs = observe_two_closest_gates(
            [far, near],
            timestamp=0.0,
            min_confidence=0.45,
            preferred=near,
        )
        self.assertIsNotNone(obs)
        # Preferred identity is gate1 even if far scores higher confidence.
        self.assertLess(obs.gate1.range_m, obs.gate2.range_m)

    def test_preferred_pnp_fail_returns_none_not_nearest(self):
        # Preferred with degenerate corners so PnP fails; far is solvable.
        bad = _fake_candidate(320, 160, 2, conf=0.9, source_index=0)
        far = _fake_candidate(400, 200, 40, conf=0.95, source_index=1)
        # Collapse preferred keypoints to a line so solve fails / is rejected.
        bad.keypoints = np.asarray(
            [[320, 160], [321, 160], [322, 160], [323, 160]],
            dtype=np.float32,
        )
        obs = observe_two_closest_gates(
            [bad, far],
            timestamp=0.0,
            min_confidence=0.45,
            preferred=bad,
        )
        self.assertIsNone(obs)


if __name__ == '__main__':
    unittest.main()
