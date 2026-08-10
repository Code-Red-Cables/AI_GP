"""Offline tests for derived-position spline following. No simulator needed.

    python test_spline_mission.py

Covers the three things that can silently break this branch:
  1. path geometry (passes through waypoints, brakes for corners and the end)
  2. capture -> JSON -> load round trip in the shape mission.py expects
  3. the planner actually converging on the path, flown against a kinematic
     model of the plant, plus every fail-safe returning hover
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import threading
import unittest

import numpy as np

os.environ.setdefault('FLIGHT_MODE', 'spline')

import config  # noqa: E402
from mission import (  # noqa: E402
    Mission,
    Waypoint,
    keep_until_gate,
    load_mission,
    save_mission,
)
from planning.spline_path import (  # noqa: E402
    build_spline_path,
    path_curvature,
    speed_profile,
)

SQUARE = [(0.0, 0.0, -2.0), (12.0, 0.0, -2.0),
          (12.0, 10.0, -2.5), (0.0, 10.0, -2.5)]


def _mission_file(points=SQUARE, path=None):
    wps = [
        Waypoint(n, e, d, yaw_deg=0.0, name=f'wp{i}')
        for i, (n, e, d) in enumerate(points)
    ]
    target = path or os.path.join(tempfile.mkdtemp(), 'wp.json')
    save_mission(Mission(wps, loop=False, name='test'), target)
    return target


class TestGeometry(unittest.TestCase):
    def test_path_passes_through_every_waypoint(self):
        pos = np.array(SQUARE, dtype=float)
        pts, cum_s, wp_s = build_spline_path(pos)
        self.assertGreater(len(pts), len(pos) * 10)
        for i, w in enumerate(pos):
            closest = float(np.linalg.norm(pts - w, axis=1).min())
            self.assertLess(closest, 1e-6, f'waypoint {i} not on path')
        # arc length is monotonic and ends at the path length
        self.assertTrue(np.all(np.diff(cum_s) >= 0.0))
        self.assertAlmostEqual(float(cum_s[-1]), float(wp_s[-1]), places=6)

    def test_speed_profile_brakes_for_corners_and_finish(self):
        pos = np.array(SQUARE, dtype=float)
        pts, cum_s, _ = build_spline_path(pos)
        curv = path_curvature(pts, cum_s)
        cruise = 4.0
        v = speed_profile(curv, cum_s, cruise=cruise, a_lat=4.0, a_lon=2.5,
                          end_speed=0.0)
        self.assertLessEqual(float(v.max()), cruise + 1e-9)
        self.assertAlmostEqual(float(v[-1]), 0.0, places=6)
        # the corner (max curvature) must be slower than the straight start
        self.assertLess(float(v[int(np.argmax(curv))]), float(v[0]) + 1e-9)

    def test_two_waypoints_is_enough(self):
        pts, cum_s, _ = build_spline_path(
            np.array([[0, 0, -2.0], [5, 0, -2.0]], dtype=float)
        )
        self.assertGreater(len(pts), 1)
        self.assertGreater(float(cum_s[-1]), 4.0)


class TestCaptureRoundTrip(unittest.TestCase):
    def test_capture_writes_loadable_mission(self):
        sys.argv = ['x']
        from tools.tune_flight import WaypointCapture

        target = os.path.join(tempfile.mkdtemp(), 'cap.json')
        cap = WaypointCapture(target)
        for i, (n, e, d) in enumerate(SQUARE):
            shared = {
                'position_ned': {'x': n, 'y': e, 'z': d},
                'attitude': {'yaw': math.radians(15.0 * i)},
            }
            msg = cap.mark(shared)
            self.assertIn(f'wp{i}', msg)
        saved = cap.save()
        self.assertIsNotNone(saved)

        loaded = load_mission(str(saved))
        self.assertEqual(len(loaded.waypoints), len(SQUARE))
        for w, (n, e, d) in zip(loaded.waypoints, SQUARE):
            self.assertAlmostEqual(w.pos[0], n, places=6)
            self.assertAlmostEqual(w.pos[1], e, places=6)
            self.assertAlmostEqual(w.pos[2], d, places=6)
        # the EKF_USE_PNP setting is recorded so a mismatched replay is visible
        payload = json.loads(open(saved, encoding='utf-8').read())
        self.assertIn('ekf_use_pnp', payload)

    def test_mark_without_pose_is_rejected(self):
        sys.argv = ['x']
        from tools.tune_flight import WaypointCapture

        cap = WaypointCapture(os.path.join(tempfile.mkdtemp(), 'c.json'))
        self.assertIn('REJECTED', cap.mark({}))
        self.assertEqual(cap.waypoints, [])

    def test_save_refuses_a_single_waypoint(self):
        sys.argv = ['x']
        from tools.tune_flight import WaypointCapture

        cap = WaypointCapture(os.path.join(tempfile.mkdtemp(), 'c.json'))
        cap.mark({'position_ned': {'x': 0, 'y': 0, 'z': -1},
                  'attitude': {'yaw': 0.0}})
        self.assertIsNone(cap.save())


class TestKeepUntilGate(unittest.TestCase):
    def test_trim_by_name(self):
        wps = [
            Waypoint(0, 0, -2, 0, name='start'),
            Waypoint(5, 0, -2, 0, name='gate1'),
            Waypoint(10, 2, -2, 10, name='gate2'),
            Waypoint(15, 4, -2, 20, name='gate3'),
        ]
        trimmed = keep_until_gate(Mission(wps, name='course'), 2)
        self.assertEqual(len(trimmed), 3)
        self.assertEqual(trimmed.waypoints[-1].name, 'gate2')
        self.assertEqual(trimmed.keep_until_gate, 2)

    def test_trim_by_gate_pass_event(self):
        wps = [
            Waypoint(0, 0, -2, 0, name='wp0'),
            Waypoint(3, 0, -2, 0, name='wp1'),
            Waypoint(6, 0, -2, 0, name='gate1',
                     active_gate=1, event='gate_pass'),
            Waypoint(12, 0, -2, 0, name='gate2',
                     active_gate=2, event='gate_pass'),
        ]
        trimmed = keep_until_gate(Mission(wps), 1)
        self.assertEqual(len(trimmed), 3)
        self.assertEqual(trimmed.waypoints[-1].event, 'gate_pass')
        self.assertEqual(trimmed.waypoints[-1].active_gate, 1)

    def test_sparse_gate_only_synthesizes_start(self):
        wps = [
            Waypoint(6, 0, -2, 0, name='gate1',
                     active_gate=1, event='gate_pass'),
            Waypoint(12, 0, -2, 0, name='gate2',
                     active_gate=2, event='gate_pass'),
        ]
        trimmed = keep_until_gate(Mission(wps), 1)
        self.assertEqual(len(trimmed), 2)
        self.assertEqual(trimmed.waypoints[0].name, 'start')
        self.assertEqual(trimmed.waypoints[1].name, 'gate1')

    def test_missing_gate_raises(self):
        wps = [
            Waypoint(0, 0, -2, 0, name='start'),
            Waypoint(5, 0, -2, 0, name='gate1'),
        ]
        with self.assertRaises(ValueError):
            keep_until_gate(Mission(wps), 3)


class _Plant:
    """Kinematic plant: rate commands -> attitude -> lean -> NED motion.

    Deliberately crude — enough to prove the guidance loop converges and is
    stable, not to model the airframe.
    """

    G = 9.80665

    def __init__(self, pos, yaw=0.0):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.zeros(3)
        self.roll = self.pitch = 0.0
        self.yaw = float(yaw)

    def blackboard(self):
        return {
            'lock': threading.Lock(),
            'position_ned': {'x': self.pos[0], 'y': self.pos[1],
                             'z': self.pos[2]},
            'attitude': {'roll': self.roll, 'pitch': self.pitch,
                         'yaw': self.yaw},
            'ekf_state': {'velocity_ned': list(self.vel)},
        }

    def step(self, tgt, dt):
        self.roll += float(tgt['roll_rate']) * dt
        self.pitch += float(tgt['pitch_rate']) * dt
        self.yaw += float(tgt['yaw_rate']) * dt
        # lean -> horizontal accel in body, rotated to NED
        a_fwd = self.G * math.tan(self.pitch) * config.FORWARD_PITCH_SIGN
        a_right = self.G * math.tan(self.roll) * config.LATERAL_LEAN_SIGN
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        a_n = a_fwd * cy - a_right * sy
        a_e = a_fwd * sy + a_right * cy
        a_d = -self.G * (float(tgt['thrust']) / config.HOVER_THRUST - 1.0)
        acc = np.array([a_n, a_e, a_d]) - 0.35 * self.vel   # light drag
        self.vel += acc * dt
        self.pos += self.vel * dt


class TestSplinePlannerClosedLoop(unittest.TestCase):
    def setUp(self):
        from spline_planner import SplinePlanner
        self.path = _mission_file()
        self.planner = SplinePlanner(self.path)

    def _fly(self, start=(0.0, 0.0, -2.0), seconds=60.0, dt=0.02):
        plant = _Plant(start)
        worst_xte = 0.0
        progress = 0.0
        for _ in range(int(seconds / dt)):
            shared = plant.blackboard()
            tgt = self.planner.compute_target(shared, dt=dt)
            plant.step(tgt, dt)
            info = shared.get('spline') or {}
            if info.get('phase') == 'track':
                worst_xte = max(worst_xte, float(info['cross_track_m']))
                progress = max(progress, float(info['progress']))
            for key in ('roll_rate', 'pitch_rate', 'yaw_rate', 'thrust'):
                self.assertTrue(math.isfinite(float(tgt[key])), key)
        return plant, worst_xte, progress

    def test_target_contract(self):
        shared = _Plant((0.0, 0.0, -2.0)).blackboard()
        tgt = self.planner.compute_target(shared)
        for key in ('kalman', 'roll_rate', 'pitch_rate', 'yaw_rate', 'thrust'):
            self.assertIn(key, tgt)
        # must take the controller's tuned rate path, not the velocity fallback
        self.assertTrue(tgt['kalman'])

    def test_follows_the_path_and_makes_progress(self):
        _, worst_xte, progress = self._fly(seconds=90.0)
        self.assertGreater(progress, 0.85, f'only {progress:.2f} of path flown')
        self.assertLess(worst_xte, config.SPLINE_MAX_XTE_M,
                        f'cross-track {worst_xte:.2f} m hit the guard')

    def test_converges_from_an_offset_start(self):
        # start 2 m off the path laterally
        _, worst_xte, progress = self._fly(start=(0.0, 2.0, -2.0), seconds=90.0)
        self.assertGreater(progress, 0.75)
        self.assertLess(worst_xte, config.SPLINE_MAX_XTE_M)

    def test_thrust_stays_inside_limits(self):
        plant = _Plant((0.0, 0.0, -2.0))
        for _ in range(1500):
            shared = plant.blackboard()
            tgt = self.planner.compute_target(shared, dt=0.02)
            self.assertGreaterEqual(tgt['thrust'], config.MIN_THRUST)
            self.assertLessEqual(tgt['thrust'], config.MAX_THRUST)
            plant.step(tgt, 0.02)


class TestSplineFailSafes(unittest.TestCase):
    def setUp(self):
        from spline_planner import SplinePlanner
        self.planner = SplinePlanner(_mission_file())

    def _assert_hover(self, tgt, shared, reason):
        self.assertEqual(tgt['roll_rate'], 0.0)
        self.assertEqual(tgt['pitch_rate'], 0.0)
        self.assertAlmostEqual(tgt['thrust'], config.HOVER_THRUST, places=9)
        self.assertEqual((shared.get('spline') or {}).get('reason'), reason)

    def test_no_derived_pose_hovers(self):
        shared = {'attitude': {'yaw': 0.0}}
        self._assert_hover(self.planner.compute_target(shared), shared,
                           'no_derived_pose')

    def test_missing_yaw_hovers(self):
        shared = {'position_ned': {'x': 0, 'y': 0, 'z': -2}, 'attitude': {}}
        self._assert_hover(self.planner.compute_target(shared), shared,
                           'no_derived_pose')

    def test_altitude_guard_hovers(self):
        shared = {
            'position_ned': {'x': 0, 'y': 0,
                             'z': -(config.SPLINE_MAX_ALT_M + 5.0)},
            'attitude': {'yaw': 0.0, 'roll': 0.0, 'pitch': 0.0},
        }
        self._assert_hover(self.planner.compute_target(shared), shared,
                           'alt_guard')

    def test_cross_track_guard_hovers(self):
        shared = {
            'position_ned': {'x': 0.0,
                             'y': config.SPLINE_MAX_XTE_M + 20.0, 'z': -2.0},
            'attitude': {'yaw': 0.0, 'roll': 0.0, 'pitch': 0.0},
        }
        self._assert_hover(self.planner.compute_target(shared), shared,
                           'xte_guard')

    def test_nonfinite_pose_hovers(self):
        shared = {
            'position_ned': {'x': float('nan'), 'y': 0.0, 'z': -2.0},
            'attitude': {'yaw': 0.0},
        }
        self._assert_hover(self.planner.compute_target(shared), shared,
                           'no_derived_pose')

    def test_reaching_the_end_finishes_rather_than_tripping_xte(self):
        # Fly the whole path, then keep going. Terminal state must be
        # 'finished' -- an overshoot past the last sample used to be reported
        # as xte_guard because the guard was checked first.
        plant = _Plant((0.0, 0.0, -2.0))
        reason = None
        for _ in range(int(80.0 / 0.02)):
            shared = plant.blackboard()
            tgt = self.planner.compute_target(shared, dt=0.02)
            plant.step(tgt, 0.02)
            info = shared.get('spline') or {}
            if info.get('phase') == 'hover':
                reason = info.get('reason')
                break
        self.assertEqual(reason, 'finished')

    def test_reset_episode_rewinds_progress(self):
        shared = _Plant((6.0, 0.0, -2.0)).blackboard()
        self.planner.compute_target(shared)
        self.assertGreater(self.planner._i, 0)
        self.planner.reset_episode()
        self.assertEqual(self.planner._i, 0)


class TestMissionNeedsTwoWaypoints(unittest.TestCase):
    def test_single_waypoint_rejected(self):
        from spline_planner import SplinePlanner
        path = _mission_file(points=[(0.0, 0.0, -2.0)])
        with self.assertRaises(ValueError):
            SplinePlanner(path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
