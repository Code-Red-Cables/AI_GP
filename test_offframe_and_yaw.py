"""Two bugs found from flight observations, pinned so they cannot return.

1. A pose model predicts all eight keypoints whether they are in frame or not.
   Clamping an off-screen prediction to the image border and flagging it "seen"
   lies to the network about where a corner is.
2. ``GateLSPose.t_gate`` is the camera's position *in the gate frame*, so
   ``bearing_rad`` is the drone's angle as seen from the gate -- the opposite of
   the gate's angle as seen from the drone. Steering yaw on +bearing turns away.
"""
from __future__ import annotations

import math
import unittest

from race_obs import (
    FRAME_H,
    FRAME_W,
    KEYPOINT_COUNT,
    NOT_SEEN,
    OFF_FRAME_MARGIN,
    build_observation,
)

VIS0 = KEYPOINT_COUNT * 2


def _obs(points):
    kps = list(points) + [(0.0, 0.0)] * (KEYPOINT_COUNT - len(points))
    return build_observation(kps, [0.9] * KEYPOINT_COUNT)


class OffFrameKeypointTests(unittest.TestCase):
    def test_in_frame_corner_is_seen(self):
        obs = _obs([(320.0, 180.0)])
        self.assertEqual(obs[VIS0], 1.0)
        self.assertAlmostEqual(obs[0], 0.5, places=3)

    def test_just_outside_is_still_believed(self):
        """A partially visible gate legitimately puts corners past the edge."""
        u = -0.05 * FRAME_W
        obs = _obs([(u, 180.0)])
        self.assertEqual(obs[VIS0], 1.0)

    def test_far_outside_is_reported_not_seen(self):
        for u in (-0.5 * FRAME_W, 1.6 * FRAME_W):
            obs = _obs([(u, 180.0)])
            self.assertEqual(obs[VIS0], 0.0, f'u={u} should be unseen')
            self.assertEqual(obs[0], NOT_SEEN)
            self.assertEqual(obs[1], NOT_SEEN)

    def test_vertical_axis_too(self):
        obs = _obs([(320.0, -0.4 * FRAME_H)])
        self.assertEqual(obs[VIS0], 0.0)

    def test_margin_boundary(self):
        inside = -(OFF_FRAME_MARGIN * 0.9) * FRAME_W
        outside = -(OFF_FRAME_MARGIN * 1.1) * FRAME_W
        self.assertEqual(_obs([(inside, 180.0)])[VIS0], 1.0)
        self.assertEqual(_obs([(outside, 180.0)])[VIS0], 0.0)

    def test_off_frame_no_longer_masquerades_as_an_edge_corner(self):
        """The old behaviour clamped to 0.0 and claimed the corner was seen."""
        obs = _obs([(-300.0, 180.0)])
        self.assertNotEqual(obs[0], 0.0)
        self.assertEqual(obs[0], NOT_SEEN)


class _Pose:
    """Minimal stand-in for GateLSPose."""

    def __init__(self, lateral, through=-5.0):
        self.lateral_m = lateral
        self.vertical_m = 0.0
        self.through_m = through
        self.range_m = 5.0
        self.residual_m = 0.1
        self.ring_disagree_m = 0.1
        self.body_forward_range = 5.0
        self.bearing_rad = math.atan2(lateral, max(-through, 1e-6))


class DeadReckonedPoseTests(unittest.TestCase):
    """Blind stretches of several seconds must not stop the controller."""

    def _planner(self):
        from race_planner import RacePlanner

        return RacePlanner()

    def test_pose_is_held_and_propagated_when_vision_drops(self):
        p = self._planner()
        p._held_pose = {
            'lateral_m': 1.0, 'vertical_m': 0.0, 'through_m': -6.0,
            'range_m': 6.08, 'age_s': 0.0,
        }
        held = p._propagate_pose(v_fwd=3.0, v_lat=0.0, dt=0.5)
        # Flying forward 1.5 m closes the gap.
        self.assertAlmostEqual(held['through_m'], -4.5, places=6)
        self.assertAlmostEqual(held['age_s'], 0.5, places=6)

    def test_lateral_drift_moves_the_gate_the_other_way(self):
        p = self._planner()
        p._held_pose = {
            'lateral_m': 0.0, 'vertical_m': 0.0, 'through_m': -6.0,
            'range_m': 6.0, 'age_s': 0.0,
        }
        held = p._propagate_pose(v_fwd=0.0, v_lat=+2.0, dt=0.5)
        self.assertLess(held['lateral_m'], 0.0)

    def test_no_held_pose_returns_none(self):
        p = self._planner()
        p._held_pose = None
        self.assertIsNone(p._propagate_pose(1.0, 0.0, 0.1))

    def test_held_pose_quacks_like_a_solved_pose(self):
        from race_planner import _HeldPose

        h = _HeldPose({'lateral_m': 1.0, 'vertical_m': 0.2,
                       'through_m': -5.0, 'range_m': 5.1, 'age_s': 0.3})
        for attr in ('lateral_m', 'vertical_m', 'range_m', 'bearing_rad',
                     'residual_m', 'ring_disagree_m', 'body_forward_range'):
            self.assertTrue(hasattr(h, attr), attr)
        self.assertTrue(h.held)
        self.assertGreater(h.bearing_rad, 0.0)


class PaperControlLawTests(unittest.TestCase):
    """Straight-part law must match paper eq. 22, not an invented variant."""

    def _align(self, lateral, v_lat=0.0):
        from race_planner import RacePlanner

        return RacePlanner()._align_command(
            _Pose(lateral), roll=0.0, pitch=0.0, yaw=0.0, v_fwd=3.0, dt=0.02,
            v_lat=v_lat,
        )

    def test_heading_is_held_not_steered(self):
        """psi_c = 0. A bearing-driven yaw loop is not in the paper."""
        for lateral in (-2.0, 0.0, +2.0):
            self.assertAlmostEqual(self._align(lateral)['yaw_rate'], 0.0,
                                   places=9)

    def test_roll_nulls_lateral_offset(self):
        self.assertLess(self._align(+1.5)['desired_roll'], 0.0)
        self.assertGreater(self._align(-1.5)['desired_roll'], 0.0)
        self.assertAlmostEqual(self._align(0.0)['desired_roll'], 0.0, places=9)

    def test_damping_uses_lateral_velocity(self):
        """kd acts on ydot; sliding right must reduce the rightward command."""
        still = self._align(-1.0, v_lat=0.0)['desired_roll']
        sliding_right = self._align(-1.0, v_lat=+1.0)['desired_roll']
        self.assertLess(sliding_right, still)

    def test_thrust_ignores_the_gate_vertical(self):
        """Altitude is a separate loop in the paper; gate height must not move it."""
        from race_planner import RacePlanner

        p = RacePlanner()
        low = _Pose(0.0)
        low.vertical_m = +5.0
        high = _Pose(0.0)
        high.vertical_m = -5.0
        a = p._align_command(low, roll=0.0, pitch=0.0, yaw=0.0, v_fwd=3.0, dt=0.02)
        b = p._align_command(high, roll=0.0, pitch=0.0, yaw=0.0, v_fwd=3.0, dt=0.02)
        self.assertAlmostEqual(a['thrust'], b['thrust'], places=9)

    def test_forward_drive_is_positive_pitch(self):
        """Forward is +pitch on this plant; the paper's sign convention is not."""
        import config
        from race_planner import RacePlanner

        p = RacePlanner()
        t = p._align_command(
            _Pose(0.0), roll=0.0, pitch=0.0, yaw=0.0, v_fwd=3.0, dt=0.02
        )
        self.assertGreater(t['desired_pitch'], 0.15,
                           'must command real forward lean, not a hover')

    def test_thrust_rises_with_lean(self):
        """Leaning without more collective trades altitude for speed."""
        from race_planner import _tilt_compensate

        level = _tilt_compensate(0.255, 0.0, 0.0)
        leaned = _tilt_compensate(0.255, 0.0, math.radians(35.0))
        self.assertAlmostEqual(level, 0.255, places=6)
        self.assertGreater(leaned, level * 1.15)

    def test_only_roll_corrects_lateral_error(self):
        """Roll does the work; yaw stays out of it (paper eq. 22)."""
        from race_planner import RacePlanner

        t = RacePlanner()._align_command(
            _Pose(+1.5), roll=0.0, pitch=0.0, yaw=0.0, v_fwd=3.0, dt=0.02
        )
        self.assertLess(t['desired_roll'], 0.0)
        self.assertAlmostEqual(t['yaw_rate'], 0.0, places=9)


if __name__ == '__main__':
    unittest.main()
