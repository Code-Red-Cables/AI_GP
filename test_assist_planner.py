"""Offline tests for AssistImagePlanner (no sim)."""

from __future__ import annotations

import math
import time
import unittest

import numpy as np

import camera_model as cm
import config
from assist_planner import AssistImagePlanner, image_gate_norm, next_gate_hint


def _airborne(planner, shared):
    planner._arm_z = 0.0
    planner._lift_start_t = time.monotonic() - 1.0
    planner._airborne_t = time.monotonic() - 1.0
    planner._climb_f = 1.2
    planner._left_pad = True
    shared['flight_started'] = True
    shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -1.2}
    shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -1.2}


def _body_cam(cx=0.0, cy=0.0, cz=12.0):
    return cm.cam_to_body(np.array([cx, cy, cz], dtype=np.float64)).tolist()


class ImageGateNormTests(unittest.TestCase):
    def test_prefers_yolo_box_over_dual_pnp(self):
        shared = {
            'gate_detection': {
                'center_px': (400.0, 200.0),
                'area_px': 2000.0,
            },
            'dual_gate_pnp': {
                'gate1_norm_x': 0.9,
                'gate1_norm_y': 0.9,
                'n_solved': 2,
            },
        }
        nx, ny, src = image_gate_norm(shared)
        self.assertEqual(src, 'yolo')
        self.assertAlmostEqual(nx, (400.0 - 320.0) / 320.0, places=5)

    def test_falls_back_to_dual_pnp(self):
        shared = {
            'gate_detection': {},
            'dual_gate_pnp': {
                'gate1_norm_x': 0.25,
                'gate1_norm_y': -0.10,
                'n_solved': 1,
            },
        }
        nx, ny, src = image_gate_norm(shared)
        self.assertEqual(src, 'dual_pnp')
        self.assertAlmostEqual(nx, 0.25)


class AssistPlannerTests(unittest.TestCase):
    def _shared(self, *, nx=0.0, ny=0.1, area=2500.0, body=None, range_m=12.0):
        cx = 320.0 + nx * 320.0
        cy = 180.0 + ny * 180.0
        if body is None:
            body = _body_cam(0.0, 0.0, range_m)
        return {
            'attitude': {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
            'position_ned': {'x': 0.0, 'y': 0.0, 'z': -1.2},
            'local_position_ned': {'x': 0.0, 'y': 0.0, 'z': -1.2},
            'gate_detection': {
                'center_px': (cx, cy),
                'area_px': area,
            },
            'dual_gate_pnp': {
                'gate1_norm_x': nx,
                'gate1_norm_y': ny,
                'gate1_range_m': range_m,
                'gate1_body': body,
                'n_solved': 1,
            },
            'race_status': {'active_gate': 0},
            'flight_started': True,
        }

    def test_centered_gate_commands_forward_lean(self):
        planner = AssistImagePlanner()
        shared = self._shared(nx=0.0, ny=0.12)
        _airborne(planner, shared)
        tgt = planner.compute_target(shared)
        self.assertTrue(tgt['kalman'])
        self.assertGreater(abs(float(tgt['desired_pitch'])), math.radians(2.0))
        self.assertEqual(shared['kalman_path']['phase'], 'chase')

    def test_left_gate_yaws(self):
        planner = AssistImagePlanner()
        shared = self._shared(nx=-0.40, ny=0.12, body=_body_cam(-4.0, 0.0, 12.0))
        _airborne(planner, shared)
        for _ in range(5):
            tgt = planner.compute_target(shared)
        self.assertLess(float(tgt['yaw_rate']), 0.0)

    def test_image_pitch_keeps_gate_in_frame(self):
        """Gate low in frame → more forward pitch; gate high → less (not nose-up)."""
        planner_lo = AssistImagePlanner()
        planner_hi = AssistImagePlanner()
        body = _body_cam(0.0, 0.0, 12.0)
        low = self._shared(nx=0.0, ny=0.50, body=body)
        high = self._shared(nx=0.0, ny=-0.40, body=body)
        _airborne(planner_lo, low)
        _airborne(planner_hi, high)
        p_lo = float(planner_lo.compute_target(low)['desired_pitch'])
        p_hi = float(planner_hi.compute_target(high)['desired_pitch'])
        self.assertGreater(abs(p_lo), abs(p_hi))
        # Never command nose-up loft from look-up (040724).
        self.assertGreaterEqual(p_hi * float(config.FORWARD_PITCH_SIGN), -1e-6)

    def test_altitude_from_gate1_pose_only(self):
        """Thrust follows gate1 cam-Y; image ny must not drive thrust."""
        hover = float(config.HOVER_THRUST)
        planner_a = AssistImagePlanner()
        planner_b = AssistImagePlanner()
        # Same pose (on boresight) — different image ny ⇒ same altitude intent.
        body = _body_cam(0.0, 0.0, 12.0)
        a = self._shared(nx=0.0, ny=0.55, body=body)
        b = self._shared(nx=0.0, ny=-0.40, body=body)
        _airborne(planner_a, a)
        _airborne(planner_b, b)
        thr_a = float(planner_a.compute_target(a)['thrust'])
        thr_b = float(planner_b.compute_target(b)['thrust'])
        self.assertAlmostEqual(thr_a, thr_b, delta=0.01)
        self.assertAlmostEqual(thr_a, hover, delta=0.025)

        planner_lo = AssistImagePlanner()
        planner_hi = AssistImagePlanner()
        lo = self._shared(nx=0.0, ny=0.1, body=_body_cam(0.0, -1.0, 12.0))
        hi = self._shared(nx=0.0, ny=0.1, body=_body_cam(0.0, 1.0, 12.0))
        _airborne(planner_lo, lo)
        _airborne(planner_hi, hi)
        thr_lo = float(planner_lo.compute_target(lo)['thrust'])
        thr_hi = float(planner_hi.compute_target(hi)['thrust'])
        self.assertGreater(thr_lo, hover + 0.004)
        self.assertLess(thr_hi, hover - 0.004)
        self.assertIn('climb', lo['kalman_path']['vert_src'])
        self.assertIn('sink', hi['kalman_path']['vert_src'])

    def test_lost_gate_falls_to_hover(self):
        planner = AssistImagePlanner()
        shared = self._shared()
        planner.compute_target(shared)
        shared['gate_detection'] = {}
        shared['dual_gate_pnp'] = {'n_solved': 0}
        planner._last_see_t = 0.0
        tgt = planner.compute_target(shared)
        self.assertIn(shared['kalman_path']['phase'], ('lost', 'search', 'hover'))
        self.assertAlmostEqual(float(tgt['yaw_rate']), 0.0, places=3)

    def test_coast_aborts_toward_next_gate_sideways(self):
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.55, ny=-0.15, area=1800.0,
            body=_body_cam(4.0, 0.0, 22.0), range_m=22.0,
        )
        _airborne(planner, shared)
        planner._climb_f = 2.0
        shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -2.0}
        shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -2.0}
        planner._coast_until = time.monotonic() + 2.0
        planner._seek_until = time.monotonic() + 10.0
        planner._have_filt = False
        tgt = planner.compute_target(shared)
        self.assertNotEqual(shared['kalman_path']['phase'], 'coast')
        self.assertGreater(float(tgt['yaw_rate']), 0.0)

    def test_next_gate_hint_from_gate2_body(self):
        shared = {
            'dual_gate_pnp': {
                'gate2_body': [15.0, 6.0, -3.0],
                'n_solved': 2,
            },
        }
        nx, ny, src, rng = next_gate_hint(shared)
        self.assertEqual(src, 'gate2_body')
        self.assertAlmostEqual(nx, 6.0 / 15.0, places=4)


if __name__ == '__main__':
    unittest.main()
