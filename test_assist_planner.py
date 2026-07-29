"""Offline tests for AssistImagePlanner (no sim)."""

from __future__ import annotations

import math
import time
import unittest

import numpy as np

import camera_model as cm
import config
from assist_planner import (
    AssistImagePlanner,
    image_gate_norm,
    next_gate_hint,
    pose_aim_y_m,
    pose_bearing_yaw_rad,
)


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

    def test_speed_cap_brakes_forward_lean(self):
        """Over ASSIST_SPEED_CAP_MPS → less forward pitch (all phases)."""
        old_cap = getattr(config, 'ASSIST_SPEED_CAP_MPS', 4.0)
        config.ASSIST_SPEED_CAP_MPS = 3.0
        try:
            planner = AssistImagePlanner()
            # Pitch-lean proxy: large commanded lean implies high v_fwd.
            shared = self._shared(nx=0.0, ny=0.12, range_m=14.0)
            shared['attitude'] = {
                'roll': 0.0,
                'pitch': math.radians(12.0) * float(config.FORWARD_PITCH_SIGN),
                'yaw': 0.0,
            }
            # Inject NED velocity well over the cap.
            shared['local_position_ned'] = {
                'x': 0.0, 'y': 0.0, 'z': -1.2,
                'vx': 6.5, 'vy': 0.0, 'vz': 0.0,
            }
            _airborne(planner, shared)
            tgt = planner.compute_target(shared)
            look = float(tgt['desired_pitch']) * float(config.FORWARD_PITCH_SIGN)
            # Braking: reduced forward lean — never reverse-nod (125233).
            self.assertLess(look, math.radians(8.0))
            self.assertGreaterEqual(look, 0.0)
        finally:
            config.ASSIST_SPEED_CAP_MPS = old_cap

    def test_pose_aim_y_nudge_is_proportional_to_range(self):
        """Lateral aim is body metres — angular nudge shrinks with range."""
        old = getattr(config, 'ASSIST_POSE_AIM_Y_M', 0.0)
        config.ASSIST_POSE_AIM_Y_M = 0.15
        try:
            self.assertAlmostEqual(pose_aim_y_m(), 0.15)
            # Far: small bearing from aim alone; near: larger.
            far = pose_bearing_yaw_rad([20.0, 0.0, 0.0])
            near = pose_bearing_yaw_rad([8.0, 0.0, 0.0])
            self.assertIsNotNone(far)
            self.assertIsNotNone(near)
            # aim_y>0 with ey=0 → negative yaw (steer left), |near| > |far|.
            self.assertLess(float(far), 0.0)
            self.assertLess(float(near), float(far))
        finally:
            config.ASSIST_POSE_AIM_Y_M = old

    def test_left_gate_yaws(self):
        planner = AssistImagePlanner()
        shared = self._shared(nx=-0.40, ny=0.12, body=_body_cam(-4.0, 0.0, 12.0))
        _airborne(planner, shared)
        for _ in range(5):
            tgt = planner.compute_target(shared)
        self.assertLess(float(tgt['yaw_rate']), 0.0)

    def test_yaw_from_pose_bearing_more_than_image(self):
        """Near-centre image: pose bearing still yaws toward the gate."""
        # Gate ~3 m right at 15 m forward; image on nx_aim so pose owns yaw.
        body = [15.0, 3.0, 0.0]
        self.assertGreater(pose_bearing_yaw_rad(body), 0.15)
        planner = AssistImagePlanner()
        nx_aim = float(getattr(config, 'ASSIST_NX_AIM', 0.03))
        shared = self._shared(nx=nx_aim, ny=0.20, body=body, range_m=15.0)
        _airborne(planner, shared)
        for _ in range(10):
            t = time.monotonic()
            planner._last_t = t - 0.05
            tgt = planner.compute_target(shared)
        # Near-centre pose fill: rightward, not plant-saturated.
        self.assertGreater(float(tgt['yaw_rate']), 0.05)
        self.assertLess(float(tgt['yaw_rate']), 0.55)

    def test_pitch_not_used_to_keep_gate_in_frame(self):
        """Same body/range → same pitch regardless of image ny (092525)."""
        planner_lo = AssistImagePlanner()
        planner_hi = AssistImagePlanner()
        body = _body_cam(0.0, 0.0, 12.0)
        low = self._shared(nx=0.0, ny=0.80, body=body)
        high = self._shared(nx=0.0, ny=-0.40, body=body)
        _airborne(planner_lo, low)
        _airborne(planner_hi, high)
        p_lo = float(planner_lo.compute_target(low)['desired_pitch'])
        p_hi = float(planner_hi.compute_target(high)['desired_pitch'])
        self.assertAlmostEqual(p_lo, p_hi, delta=math.radians(0.5))
        # Never command nose-up loft (040724).
        self.assertGreaterEqual(p_hi * float(config.FORWARD_PITCH_SIGN), -1e-6)

    def test_larger_pose_dz_sinks_harder(self):
        """Bigger geometric height error → stronger sink thrust offset."""
        hover = float(config.HOVER_THRUST)
        old_bias = getattr(config, 'ASSIST_CAM_TILT_BIAS', 1.0)
        old_high = getattr(config, 'ASSIST_APPROACH_HIGH_M', 0.28)
        config.ASSIST_CAM_TILT_BIAS = 0.0
        config.ASSIST_APPROACH_HIGH_M = 0.0
        try:
            planner_s = AssistImagePlanner()
            planner_l = AssistImagePlanner()
            small = self._shared(
                nx=0.0, ny=0.55, body=[15.0, 0.0, 0.8], range_m=15.0,
            )
            large = self._shared(
                nx=0.0, ny=0.85, body=[15.0, 0.0, 2.5], range_m=15.0,
            )
            _airborne(planner_s, small)
            _airborne(planner_l, large)
            thr_s = float(planner_s.compute_target(small)['thrust'])
            thr_l = float(planner_l.compute_target(large)['thrust'])
            self.assertIn('sink', small['kalman_path']['vert_src'])
            self.assertIn('sink', large['kalman_path']['vert_src'])
            self.assertLess(thr_s, hover - 0.002)
            # Quadratic shape: 2.5 m error much harder than 0.8 m.
            self.assertLess(thr_l, thr_s - 0.010)
        finally:
            config.ASSIST_CAM_TILT_BIAS = old_bias
            config.ASSIST_APPROACH_HIGH_M = old_high

    def test_altitude_from_gate1_geometric_height(self):
        """Thrust follows gate1 NED height; image ny must not drive thrust.

        Approach aims a bit above pose-matched height (bottom-rail clearance).
        """
        hover = float(config.HOVER_THRUST)
        planner_a = AssistImagePlanner()
        planner_b = AssistImagePlanner()
        # Same geometric height → same thrust (ny must not diverge them).
        body = [12.0, 0.0, 0.0]
        a = self._shared(nx=0.0, ny=0.10, body=body, range_m=18.0)
        b = self._shared(nx=0.0, ny=0.05, body=body, range_m=18.0)
        _airborne(planner_a, a)
        _airborne(planner_b, b)
        thr_a = float(planner_a.compute_target(a)['thrust'])
        thr_b = float(planner_b.compute_target(b)['thrust'])
        self.assertAlmostEqual(thr_a, thr_b, delta=0.01)
        # Level geometric height → hold (approach-high default off).
        self.assertAlmostEqual(thr_a, hover, delta=0.025)
        self.assertIn('hold', a['kalman_path']['vert_src'])

        planner_above = AssistImagePlanner()  # gate above us → climb
        planner_below = AssistImagePlanner()  # gate below us → sink
        # Disable cam-tilt cancel so pure geometry is visible.
        old_bias = getattr(config, 'ASSIST_CAM_TILT_BIAS', 1.0)
        config.ASSIST_CAM_TILT_BIAS = 0.0
        try:
            ny_aim = float(getattr(config, 'ASSIST_NY_AIM', 0.12))
            above = self._shared(
                nx=0.0, ny=min(-0.05, ny_aim - 0.15), body=[12.0, 0.0, -1.0],
            )
            below = self._shared(nx=0.0, ny=0.82, body=[12.0, 0.0, 1.8])
            _airborne(planner_above, above)
            _airborne(planner_below, below)
            thr_up = float(planner_above.compute_target(above)['thrust'])
            thr_dn = float(planner_below.compute_target(below)['thrust'])
            self.assertGreater(thr_up, hover + 0.002)
            self.assertLess(thr_dn, hover - 0.004)
            self.assertIn('climb', above['kalman_path']['vert_src'])
            self.assertIn('sink', below['kalman_path']['vert_src'])
        finally:
            config.ASSIST_CAM_TILT_BIAS = old_bias

    def test_no_climb_when_gate_already_low_in_frame(self):
        """090736: ny≫aim must not climb on residual negative pose_dz."""
        hover = float(config.HOVER_THRUST)
        planner = AssistImagePlanner()
        # Optical-axis-ish body (raw dz negative) but gate low in image.
        shared = self._shared(
            nx=0.0, ny=0.55, body=_body_cam(0.0, 0.0, 16.0), range_m=16.0,
        )
        _airborne(planner, shared)
        thr = float(planner.compute_target(shared)['thrust'])
        self.assertNotIn('climb', shared['kalman_path']['vert_src'])
        self.assertLessEqual(thr, hover + 0.008)

    def test_approach_ignores_pose_sink_when_image_height_ok(self):
        """095706: on approach, pose-below must not sink if ny is near aim."""
        hover = float(config.HOVER_THRUST)
        old_bias = getattr(config, 'ASSIST_CAM_TILT_BIAS', 1.0)
        config.ASSIST_CAM_TILT_BIAS = 0.0
        try:
            planner = AssistImagePlanner()
            # Pose says gate below; image height still near aim → hold.
            shared = self._shared(
                nx=0.0, ny=0.28, body=[10.0, 0.0, 1.0], range_m=10.0,
            )
            _airborne(planner, shared)
            thr = float(planner.compute_target(shared)['thrust'])
            self.assertNotIn('sink', shared['kalman_path']['vert_src'])
            self.assertGreater(thr, hover - 0.004)
        finally:
            config.ASSIST_CAM_TILT_BIAS = old_bias

    def test_approach_tip_sinks_when_gate_low_in_frame(self):
        """124213: clear tip-low above min alt/range must sink (not pad dig)."""
        hover = float(config.HOVER_THRUST)
        old_bias = getattr(config, 'ASSIST_CAM_TILT_BIAS', 1.0)
        config.ASSIST_CAM_TILT_BIAS = 0.0
        try:
            planner = AssistImagePlanner()
            shared = self._shared(
                nx=0.02, ny=0.50, body=[10.0, 0.2, 0.05], range_m=10.0,
                area=4000.0,
            )
            _airborne(planner, shared)
            planner._climb_f = 1.70
            shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -1.70}
            shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -1.70}
            planner._last_t = time.monotonic() - 0.05
            thr = float(planner.compute_target(shared)['thrust'])
            self.assertIn('sink', shared['kalman_path']['vert_src'])
            self.assertLessEqual(thr, hover + 0.002)
        finally:
            config.ASSIST_CAM_TILT_BIAS = old_bias

    def test_coast_settles_when_tip_low(self):
        """Through-slot with tip-low must not coast_lift into the top rail."""
        hover = float(config.HOVER_THRUST)
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.02, ny=0.40, body=[6.0, 0.1, 0.3], range_m=6.0, area=12000.0,
        )
        _airborne(planner, shared)
        planner._climb_f = 1.70
        shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -1.70}
        shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -1.70}
        planner._coast_until = time.monotonic() + 2.0
        planner._seek_until = 0.0
        planner._have_filt = True
        planner._nx_f = 0.02
        planner._ny_f = 0.40
        planner._last_t = time.monotonic() - 0.05
        thr = float(planner.compute_target(shared)['thrust'])
        self.assertEqual(shared['kalman_path']['phase'], 'coast')
        self.assertEqual(shared['kalman_path']['vert_src'], 'coast_settle')
        self.assertLess(thr, hover)

    def test_coast_lifts_when_tip_high(self):
        """124438: ny below aim on coast → lift (bottom-rail scrape)."""
        hover = float(config.HOVER_THRUST)
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.02, ny=-0.10, body=[6.5, 0.1, -0.2], range_m=6.5, area=14000.0,
        )
        _airborne(planner, shared)
        planner._climb_f = 1.35
        shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -1.35}
        shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -1.35}
        planner._coast_until = time.monotonic() + 2.0
        planner._seek_until = 0.0
        planner._have_filt = True
        planner._nx_f = 0.02
        planner._ny_f = -0.10
        planner._last_t = time.monotonic() - 0.05
        thr = float(planner.compute_target(shared)['thrust'])
        self.assertEqual(shared['kalman_path']['phase'], 'coast')
        self.assertEqual(shared['kalman_path']['vert_src'], 'coast_lift')
        self.assertGreater(thr, hover + 0.004)

    def test_cam_tilt_bias_holds_on_optical_axis_at_hover(self):
        """085654: optical-axis gate at hover must not loft (cam-tilt bias)."""
        hover = float(config.HOVER_THRUST)
        old_high = getattr(config, 'ASSIST_APPROACH_HIGH_M', 0.28)
        config.ASSIST_APPROACH_HIGH_M = 0.0
        try:
            planner = AssistImagePlanner()
            # Optical-axis body at 20 m ⇒ raw NED dz ≈ −6.8 m; bias cancels.
            shared = self._shared(
                nx=0.0, ny=-0.02, body=_body_cam(0.0, 0.0, 20.0), range_m=20.0,
            )
            _airborne(planner, shared)
            thr = float(planner.compute_target(shared)['thrust'])
            path = shared['kalman_path']
            self.assertGreater(float(path['tilt_bias']), 3.0)
            # Near-centre optical-axis must not command a loft.
            self.assertNotIn('climb', path['vert_src'])
            self.assertLessEqual(thr, hover + 0.008)
        finally:
            config.ASSIST_APPROACH_HIGH_M = old_high

    def test_cam_tilt_bias_grows_with_speed(self):
        """Faster / more pitched ⇒ stronger cam-tilt bias (drone tilts more)."""
        from assist_planner import cam_tilt_height_bias_m
        bias_slow = cam_tilt_height_bias_m(-6.8, 18.8, 0.0, 0.0, 0.0)
        bias_fast = cam_tilt_height_bias_m(-6.8, 18.8, 0.0, 0.0, 6.0)
        bias_pitched = cam_tilt_height_bias_m(
            -6.8, 18.8, 0.0, math.radians(10.0), 0.0
        )
        self.assertGreater(bias_slow, 5.0)  # hover still cancels optical-axis loft
        self.assertGreaterEqual(bias_fast, bias_slow)
        self.assertGreaterEqual(bias_pitched, bias_slow)

    def test_cam_bank_lateral_bias_symmetric_and_speed(self):
        """Fly left/right → opposite nx bias; |bias| grows with speed."""
        from assist_planner import cam_bank_lateral_bias_nx
        lean = math.radians(10.0)
        # Bank left / move left → +bias (gate looked too left).
        left_slow = cam_bank_lateral_bias_nx(
            -lean * 0.6, -1.0, 2.0, lean,
        )
        left_fast = cam_bank_lateral_bias_nx(
            -lean * 0.6, -5.0, 8.0, lean,
        )
        right_fast = cam_bank_lateral_bias_nx(
            lean * 0.6, 5.0, 8.0, lean,
        )
        self.assertGreater(left_slow, 0.0)
        self.assertGreater(left_fast, left_slow)
        self.assertLess(right_fast, 0.0)
        self.assertAlmostEqual(left_fast, -right_fast, delta=0.05)

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

    def test_ignores_memorized_course_bearing(self):
        """Chase the live gate image — never default_right / course memory."""
        planner = AssistImagePlanner()
        shared = self._shared(nx=-0.35, ny=0.40, area=1200.0, range_m=20.0)
        shared['course_bearing'] = {
            'nx': 0.28, 'ny': -0.06, 'range_m': 8.0,
            'source': 'default_right', 'ts': time.monotonic(),
        }
        _airborne(planner, shared)
        planner._seek_until = time.monotonic() + 10.0
        planner._coast_until = 0.0
        for _ in range(5):
            tgt = planner.compute_target(shared)
        # Live YOLO is left → yaw left; do not snap to memorized +0.28.
        self.assertLess(float(tgt['yaw_rate']), 0.0)
        self.assertLess(float(shared['kalman_path']['norm_x']), 0.0)

    def test_sinks_when_live_gate_pose_is_below(self):
        """Gate below us in PnP → sink on approach; seeking uses image ny."""
        hover = float(config.HOVER_THRUST)
        old_bias = getattr(config, 'ASSIST_CAM_TILT_BIAS', 1.0)
        config.ASSIST_CAM_TILT_BIAS = 0.0
        try:
            planner = AssistImagePlanner()
            shared = self._shared(
                nx=0.05, ny=0.55, body=[18.0, 0.5, 1.2], range_m=18.0,
            )
            _airborne(planner, shared)
            # Approach (not seeking) may sink on pose-below.
            thr = float(planner.compute_target(shared)['thrust'])
            self.assertIn('sink', shared['kalman_path']['vert_src'])
            self.assertLess(thr, hover - 0.002)

            # After a pass: image-ny owns altitude (gate mid-low → seek_sink).
            planner2 = AssistImagePlanner()
            seek = self._shared(
                nx=0.05, ny=0.55, body=[18.0, 0.5, 1.2], range_m=18.0,
            )
            _airborne(planner2, seek)
            planner2._climb_f = 2.5
            seek['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -2.5}
            seek['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -2.5}
            planner2._seek_until = time.monotonic() + 10.0
            planner2._coast_until = 0.0
            planner2._pass_t = time.monotonic() - 2.0
            thr2 = float(planner2.compute_target(seek)['thrust'])
            self.assertIn(
                seek['kalman_path']['vert_src'],
                ('seek_sink', 'seek_pose_sink', 'pose_g1:sink'),
            )
            self.assertLess(thr2, hover)
        finally:
            config.ASSIST_CAM_TILT_BIAS = old_bias

    def test_sinks_when_live_gate_is_low_in_frame(self):
        """Approach tip-sinks when ny≫aim; seeking also sinks on low ny."""
        hover = float(config.HOVER_THRUST)
        # Level geometric gate but tip-low → approach tip sink (124213).
        planner_g1 = AssistImagePlanner()
        g1 = self._shared(
            nx=0.0, ny=0.52, body=[18.0, 0.0, 0.0], range_m=18.0,
        )
        _airborne(planner_g1, g1)
        thr_g1 = float(planner_g1.compute_target(g1)['thrust'])
        self.assertIn('sink', g1['kalman_path']['vert_src'])
        self.assertLessEqual(thr_g1, hover + 0.005)

        # Seeking next gate with low ny → seek_sink (bring gate up in frame).
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.05, ny=0.80, body=[18.0, 0.0, 0.0], range_m=18.0,
        )
        _airborne(planner, shared)
        planner._climb_f = 2.0
        shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -2.0}
        shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -2.0}
        planner._seek_until = time.monotonic() + 10.0
        planner._coast_until = 0.0
        thr = float(planner.compute_target(shared)['thrust'])
        self.assertIn(
            shared['kalman_path']['vert_src'],
            ('seek_sink', 'seek_pose_sink'),
        )
        self.assertLess(thr, hover)

    def test_next_gate_hint_ignores_course_bearing(self):
        shared = {
            'course_bearing': {
                'nx': 0.28, 'ny': -0.06, 'ts': time.monotonic(),
                'source': 'default_right',
            },
            'dual_gate_pnp': {'n_solved': 0},
        }
        nx, ny, src, _rng = next_gate_hint(shared)
        self.assertEqual(src, 'none')
        self.assertIsNone(nx)

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
        # Abort only after a real pass (not during visual-commit slot).
        planner._pass_t = time.monotonic() - 0.5
        planner._have_filt = False
        tgt = planner.compute_target(shared)
        self.assertNotEqual(shared['kalman_path']['phase'], 'coast')
        self.assertGreater(float(tgt['yaw_rate']), 0.0)

    def test_coast_stays_through_near_gate(self):
        """094154: do not abort coast while still on the current gate."""
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.05, ny=0.25, area=15000.0, range_m=9.1,
        )
        _airborne(planner, shared)
        planner._coast_until = time.monotonic() + 2.0
        planner._seek_until = time.monotonic() + 10.0
        planner._have_filt = True
        planner._nx_f = 0.05
        planner._ny_f = 0.25
        planner.compute_target(shared)
        self.assertEqual(shared['kalman_path']['phase'], 'coast')

    def test_yaw_trusts_image_when_pose_disagrees(self):
        """Gate right in image but body says left → still yaw right."""
        planner = AssistImagePlanner()
        # body y negative = left; image nx positive = right.
        shared = self._shared(
            nx=0.20, ny=0.25, body=[12.0, -2.5, 0.0], range_m=8.0,
        )
        _airborne(planner, shared)
        for _ in range(8):
            planner._last_t = time.monotonic() - 0.05
            tgt = planner.compute_target(shared)
        self.assertGreater(float(tgt['yaw_rate']), 0.04)
        self.assertLess(float(tgt['yaw_rate']), 0.55)

    def test_yaw_coarse_then_fine(self):
        """Far offset → large yaw; near centre → small fine yaw."""
        planner_far = AssistImagePlanner()
        planner_near = AssistImagePlanner()
        nx_aim = float(getattr(config, 'ASSIST_NX_AIM', 0.03))
        far = self._shared(
            nx=0.45, ny=0.25, body=[12.0, 5.0, 0.0], range_m=10.0,
        )
        # Relative to nx_aim so fine yaw is still clearly rightward.
        near = self._shared(
            nx=nx_aim + 0.08, ny=0.25, body=[12.0, 0.8, 0.0], range_m=10.0,
        )
        _airborne(planner_far, far)
        _airborne(planner_near, near)
        # Inject realistic control dt so yaw slew can accumulate.
        for _ in range(8):
            t = time.monotonic()
            planner_far._last_t = t - 0.05
            planner_near._last_t = t - 0.05
            y_far = float(planner_far.compute_target(far)['yaw_rate'])
            y_near = float(planner_near.compute_target(near)['yaw_rate'])
        self.assertGreater(y_far, 0.18)
        self.assertGreater(y_near, 0.0)
        self.assertLess(y_near, 0.22)
        self.assertLess(y_far, 1.70)  # bang-bang may hit plant ~95°/s
        self.assertGreater(y_far, y_near * 1.6)

    def test_extreme_nx_yaws_harder_than_mild(self):
        """Far L/R in frame must command clearly stronger yaw than mild offset."""
        planner_hi = AssistImagePlanner()
        planner_lo = AssistImagePlanner()
        hi = self._shared(nx=0.55, ny=0.20, body=None, range_m=16.0)
        lo = self._shared(nx=0.15, ny=0.20, body=None, range_m=16.0)
        _airborne(planner_hi, hi)
        _airborne(planner_lo, lo)
        for _ in range(10):
            t = time.monotonic()
            planner_hi._last_t = t - 0.05
            planner_lo._last_t = t - 0.05
            y_hi = float(planner_hi.compute_target(hi)['yaw_rate'])
            y_lo = float(planner_lo.compute_target(lo)['yaw_rate'])
        self.assertGreater(y_hi, 0.0)
        self.assertGreater(y_lo, 0.0)
        self.assertGreater(y_hi, y_lo * 1.5)

    def test_visual_commit_keeps_aim_and_needs_align(self):
        """092927: do not drop lock on commit; refuse commit when |nx| large."""
        # Misaligned close gate must NOT commit.
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.28, ny=0.20, area=10000.0, range_m=7.5,
        )
        _airborne(planner, shared)
        planner.compute_target(shared)
        self.assertNotEqual(shared['kalman_path']['phase'], 'coast')
        self.assertFalse(shared.get('vision_begin_next_gate'))

        # Aligned close gate commits but keeps filt / no next-gate punch.
        planner2 = AssistImagePlanner()
        aligned = self._shared(
            nx=0.05, ny=0.20, area=10000.0, range_m=7.5,
        )
        _airborne(planner2, aligned)
        planner2.compute_target(aligned)  # seed filt
        planner2.compute_target(aligned)
        self.assertEqual(aligned['kalman_path']['phase'], 'coast')
        self.assertTrue(planner2._have_filt)
        self.assertFalse(aligned.get('vision_begin_next_gate'))

    def test_coast_rolls_toward_image_nx(self):
        """Through-slot coast must strafe/yaw on live nx (not roll=0)."""
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.30, ny=0.25, area=8000.0, range_m=6.0,
        )
        _airborne(planner, shared)
        planner._coast_until = time.monotonic() + 2.0
        planner._seek_until = time.monotonic() + 10.0
        planner._have_filt = True
        planner._nx_f = 0.30
        planner._ny_f = 0.25
        tgt = planner.compute_target(shared)
        self.assertEqual(shared['kalman_path']['phase'], 'coast')
        # Gate right in frame → yaw right; roll sign follows LATERAL_LEAN_SIGN.
        self.assertGreater(float(tgt['yaw_rate']), 0.0)
        self.assertNotAlmostEqual(float(tgt['desired_roll']), 0.0, places=3)

    def test_next_gate_hint_from_gate2_body(self):
        shared = {
            'dual_gate_pnp': {
                # z/x must stay >= ASSIST_LATCH_NY_MIN (−0.05); −3/15 was
                # a cleared-gate high residual (122209).
                'gate2_body': [15.0, 6.0, 1.5],
                'n_solved': 2,
            },
        }
        nx, ny, src, rng = next_gate_hint(shared)
        self.assertEqual(src, 'gate2_body')
        self.assertAlmostEqual(nx, 6.0 / 15.0, places=4)
        self.assertAlmostEqual(ny, 1.5 / 15.0, places=4)

    def test_next_gate_hint_rejects_high_ny(self):
        """122209: above-frame residual must not latch as next."""
        shared = {
            'dual_gate_pnp': {
                'gate2_body': [15.0, 2.0, -3.0],
                'n_solved': 2,
            },
        }
        nx, ny, src, rng = next_gate_hint(shared)
        self.assertEqual(src, 'none')
        self.assertIsNone(nx)

    def test_ghost_does_not_reinject_high_latch_after_low_dig(self):
        """122209: after losing a low dig, skip high residual latch → scan."""
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.05, ny=0.55, body=[16.0, 0.8, 2.0], range_m=16.0, area=900.0,
        )
        _airborne(planner, shared)
        planner._seek_until = time.monotonic() + 10.0
        planner._coast_until = 0.0
        planner._pass_t = time.monotonic() - 2.0
        planner._reset_gate_lock()
        for _ in range(4):
            planner._last_t = time.monotonic() - 0.05
            planner.compute_target(shared)
        self.assertGreater(float(planner._ny_f), 0.35)
        # Blind + high-ish latch that still passes LATCH_NY_MIN / min-ahead.
        shared['vision'] = {}
        shared['gate_detection'] = {}
        shared['dual_gate_pnp'] = {}
        planner._next_nx = 0.05
        planner._next_ny = -0.04
        planner._next_rng = 16.0
        planner._next_t = time.monotonic() - 0.2
        planner._next_live_t = time.monotonic() - 0.2
        ghost_s = float(getattr(config, 'ASSIST_SEEK_GHOST_S', 1.20))
        planner._last_see_t = time.monotonic() - ghost_s - 0.05
        planner._last_t = time.monotonic() - 0.05
        planner.compute_target(shared)
        self.assertFalse(planner._have_filt)
        self.assertNotEqual(
            shared['kalman_path'].get('src'), 'next_latch'
        )

    def test_post_pass_yaws_toward_glimpse_until_lock(self):
        """Gate slightly in frame after pass → live soft yaw."""
        hover = float(config.HOVER_THRUST)
        bleed = float(getattr(config, 'ASSIST_SEEK_THRUST_BLEED', 0.014))
        ny_aim = float(getattr(config, 'ASSIST_NY_AIM', 0.22))
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.45, ny=ny_aim, body=[16.0, 6.0, 0.0], range_m=17.0, area=1800.0,
        )
        _airborne(planner, shared)
        planner._climb_f = 2.5
        shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -2.5}
        shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -2.5}
        planner._seek_until = time.monotonic() + 10.0
        planner._coast_until = 0.0
        planner._pass_t = time.monotonic() - 2.0
        planner._reset_gate_lock()

        # First ticks: unlocked soft chase on the live glimpse.
        for _ in range(2):
            planner._last_t = time.monotonic() - 0.05
            tgt = planner.compute_target(shared)
        self.assertEqual(shared['kalman_path']['phase'], 'seek_yaw')
        self.assertFalse(shared['kalman_path']['gate_lock'])
        # Far-right glimpse (|nx|≥bang) saturates yaw right.
        self.assertGreater(float(tgt['yaw_rate']), math.radians(20.0))
        self.assertAlmostEqual(float(tgt['desired_roll']), 0.0, places=3)
        look = float(tgt['desired_pitch']) * float(config.FORWARD_PITCH_SIGN)
        # Post-pass: keep cam-level tip (not crawl-only sky look).
        self.assertGreater(look, math.radians(10.0))
        self.assertLess(look, math.radians(17.0))
        self.assertEqual(shared['kalman_path']['vert_src'], 'seek_hold')
        tilt = math.cos(abs(float(tgt['desired_pitch'])))
        tilted_hover = hover / max(0.88, tilt)
        self.assertLess(float(tgt['thrust']), tilted_hover - 0.5 * bleed)

        for _ in range(8):
            planner._last_t = time.monotonic() - 0.05
            tgt = planner.compute_target(shared)
        self.assertTrue(shared['kalman_path']['gate_lock'])
        self.assertEqual(shared['kalman_path']['phase'], 'seek_chase')
        self.assertGreater(float(tgt['yaw_rate']), 0.0)
        # Still tipped after lock — camera stays forward.
        look_locked = float(tgt['desired_pitch']) * float(
            config.FORWARD_PITCH_SIGN
        )
        self.assertGreater(look_locked, math.radians(10.0))

    def test_post_pass_holds_soft_yaw_then_clears_ghost(self):
        """After vision drops: soft-yaw on last box, unlock, then clear."""
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.40, ny=0.35, body=[18.0, 5.0, 0.0], range_m=18.0, area=1600.0,
        )
        _airborne(planner, shared)
        planner._seek_until = time.monotonic() + 10.0
        planner._coast_until = 0.0
        planner._pass_t = time.monotonic() - 2.0
        planner._reset_gate_lock()

        for _ in range(6):
            planner._last_t = time.monotonic() - 0.05
            planner.compute_target(shared)
        self.assertTrue(planner._have_filt)
        self.assertTrue(planner._seek_seen)

        shared['vision'] = {}
        shared['gate_detection'] = {}
        shared['dual_gate_pnp'] = {}

        # Was locked, then vision gone — unlock immediately, keep soft yaw.
        planner._gate_lock = True
        planner._lock_count = 8
        planner._last_see_t = time.monotonic() - 0.40
        planner._last_t = time.monotonic() - 0.05
        tgt = planner.compute_target(shared)
        self.assertTrue(planner._have_filt)
        self.assertFalse(shared['kalman_path']['gate_lock'])
        self.assertEqual(shared['kalman_path']['phase'], 'seek_yaw')
        self.assertGreater(float(tgt['yaw_rate']), 0.0)
        look = float(tgt['desired_pitch']) * float(config.FORWARD_PITCH_SIGN)
        self.assertGreater(look, math.radians(4.0))

        # After ghost timeout with latch cleared — blind L/R scan (not mute).
        planner._clear_next_latch()
        planner._last_see_t = time.monotonic() - 1.40
        planner._last_t = time.monotonic() - 0.05
        tgt = planner.compute_target(shared)
        self.assertFalse(planner._have_filt)
        self.assertEqual(shared['kalman_path']['phase'], 'seek_scan')
        # Scan eases in; after a short age it must be non-zero most of the time.
        # Force age past ease-in so the sine is away from a zero crossing.
        planner._pass_t = time.monotonic() - 1.25
        planner._last_t = time.monotonic() - 0.05
        tgt = planner.compute_target(shared)
        self.assertNotAlmostEqual(float(tgt['yaw_rate']), 0.0, places=3)

    def test_blind_seek_scans_and_caps_loft(self):
        """121451: no latch → scan yaw; above cruise → bleed thrust."""
        hover = float(config.HOVER_THRUST)
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.0, ny=0.2, body=None, range_m=None, area=None,
        )
        shared['gate_detection'] = {}
        shared['dual_gate_pnp'] = {'n_solved': 0}
        _airborne(planner, shared)
        planner._climb_f = 2.50
        shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -2.50}
        shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -2.50}
        planner._seek_until = time.monotonic() + 10.0
        planner._coast_until = 0.0
        planner._pass_t = time.monotonic() - 1.25
        planner._reset_gate_lock()
        planner._clear_next_latch()
        planner._have_filt = False
        planner._last_see_t = time.monotonic() - 2.0
        planner._last_t = time.monotonic() - 0.05
        tgt = planner.compute_target(shared)
        self.assertEqual(shared['kalman_path']['phase'], 'seek_scan')
        self.assertNotAlmostEqual(float(tgt['yaw_rate']), 0.0, places=3)
        self.assertIn(
            shared['kalman_path']['vert_src'],
            ('seek_scan_cap', 'seek_ceiling'),
        )
        self.assertLess(float(tgt['thrust']), hover - 0.004)

    def test_seek_lock_yaws_from_latch_after_ghost(self):
        """115959: after ghost clear, keep soft-yaw from next-gate latch."""
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.40, ny=0.35, body=[18.0, 5.0, 0.0], range_m=18.0, area=1600.0,
        )
        _airborne(planner, shared)
        planner._seek_until = time.monotonic() + 10.0
        planner._coast_until = 0.0
        planner._pass_t = time.monotonic() - 2.0
        planner._reset_gate_lock()
        for _ in range(6):
            planner._last_t = time.monotonic() - 0.05
            planner.compute_target(shared)
        shared['vision'] = {}
        shared['gate_detection'] = {}
        shared['dual_gate_pnp'] = {}
        planner._next_nx = 0.35
        planner._next_ny = 0.40
        planner._next_rng = 18.0
        planner._next_t = time.monotonic() - 0.5
        planner._have_filt = False
        planner._last_see_t = time.monotonic() - 2.0
        planner._last_t = time.monotonic() - 0.05
        tgt = planner.compute_target(shared)
        self.assertGreater(float(tgt['yaw_rate']), 0.0)
        self.assertIn(
            shared['kalman_path']['phase'],
            ('seek_yaw', 'seek_lock', 'seek_scan', 'seek_chase'),
        )

    def test_pass_seeds_seek_from_pre_pass_gate2_latch(self):
        """Gate2 seen before pass is kept and used to soft-aim after."""
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.0, ny=0.20, body=[8.0, 0.0, 0.0], range_m=8.0, area=8000.0,
        )
        shared['dual_gate_pnp']['gate2_body'] = [20.0, 5.0, 1.5]
        shared['dual_gate_pnp']['n_solved'] = 2
        shared['race_status'] = {'active_gate': 0}
        _airborne(planner, shared)
        planner._active_gate = 0
        planner._last_t = time.monotonic() - 0.05
        planner.compute_target(shared)
        self.assertIsNotNone(planner._next_nx)
        self.assertAlmostEqual(planner._next_nx, 5.0 / 20.0, places=3)

        # Pass with vision empty — must still soft-yaw from latched gate2.
        shared['race_status'] = {'active_gate': 1}
        shared['gate_detection'] = {}
        shared['dual_gate_pnp'] = {'n_solved': 0}
        planner._last_t = time.monotonic() - 0.05
        tgt = planner.compute_target(shared)
        self.assertTrue(planner._have_filt)
        self.assertTrue(planner._seek_seen)
        self.assertGreater(float(planner._nx_f), 0.15)
        self.assertIn(
            shared['kalman_path']['phase'],
            ('coast', 'seek_yaw', 'seek_chase'),
        )
        self.assertGreater(float(tgt['yaw_rate']), 0.0)

    def test_approach_snap_seeds_pass_after_latch_ages(self):
        """115959: approach snapshot survives dual dropout / stale _next_t."""
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.02, ny=0.22, body=[7.0, 0.2, 0.1], range_m=7.0, area=9000.0,
        )
        shared['dual_gate_pnp']['gate2_body'] = [18.0, 4.5, 1.2]
        shared['dual_gate_pnp']['n_solved'] = 2
        shared['race_status'] = {'active_gate': 1}
        _airborne(planner, shared)
        planner._active_gate = 1
        planner._climb_f = 2.5
        shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -2.5}
        shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -2.5}
        planner._last_t = time.monotonic() - 0.05
        planner.compute_target(shared)
        self.assertIsNotNone(planner._snap_next_nx)
        self.assertAlmostEqual(planner._snap_next_nx, 4.5 / 18.0, places=3)

        # Age out live latch; dual gone — pass must still seed from snap.
        planner._next_t = time.monotonic() - 8.0
        shared['race_status'] = {'active_gate': 2}
        shared['gate_detection'] = {}
        shared['dual_gate_pnp'] = {'n_solved': 0}
        planner._last_t = time.monotonic() - 0.05
        tgt = planner.compute_target(shared)
        self.assertTrue(planner._have_filt)
        self.assertGreater(float(planner._nx_f), 0.15)
        self.assertGreater(float(tgt['yaw_rate']), 0.0)

    def test_reject_near_latch_on_pass(self):
        """120804: ~8.7 m residual must not seed seek after GATE_PASSED."""
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.0, ny=0.2, body=[8.0, 0.0, 0.0], range_m=8.0, area=8000.0,
        )
        shared['gate_detection'] = {}
        shared['dual_gate_pnp'] = {'n_solved': 0}
        shared['race_status'] = {'active_gate': 1}
        _airborne(planner, shared)
        planner._active_gate = 0
        planner._next_nx = 0.05
        planner._next_ny = -0.22
        planner._next_rng = 8.7
        planner._next_t = time.monotonic()
        planner._next_live_t = time.monotonic()
        planner._snap_next_nx = 0.05
        planner._snap_next_ny = -0.22
        planner._snap_next_rng = 8.7
        planner._snap_next_t = time.monotonic()
        planner._last_t = time.monotonic() - 0.05
        planner.compute_target(shared)
        self.assertIsNone(planner._snap_next_nx)
        # Poisoned 8.7 m residual dropped; course-2 memory may replace it.
        self.assertNotAlmostEqual(
            float(shared['kalman_path'].get('norm_x') or 0.0), 0.05, places=2
        )
        if planner._course_mem:
            self.assertGreater(float(planner._next_nx), 0.20)
            self.assertGreaterEqual(
                float(planner._next_rng), float(config.ASSIST_LATCH_MIN_AHEAD_M)
            )

    def test_course2_memory_seeds_right_after_gate1(self):
        """Course 2: after g1 with no live latch → yaw right, slight climb."""
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.0, ny=0.2, body=None, range_m=None, area=None,
        )
        shared['gate_detection'] = {}
        shared['dual_gate_pnp'] = {'n_solved': 0}
        shared['race_status'] = {'active_gate': 1}
        shared['attitude'] = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        _airborne(planner, shared)
        planner._active_gate = 0
        planner._climb_f = 1.40
        shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -1.40}
        shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -1.40}
        planner._last_t = time.monotonic() - 0.05
        tgt = planner.compute_target(shared)
        self.assertTrue(planner._course_mem)
        self.assertTrue(planner._course_mem_spent)
        self.assertIsNotNone(planner._course_mem_yaw_tgt)
        self.assertGreater(float(planner._nx_f), 0.25)
        # Heading hold: right while off target; settles (no perpetual hunt).
        yaws = []
        for i in range(8):
            planner._pass_t = time.monotonic() - 1.0
            planner._last_t = time.monotonic() - 0.05
            shared['attitude'] = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
            shared['gate_detection'] = {}
            shared['dual_gate_pnp'] = {'n_solved': 0}
            yaws.append(float(planner.compute_target(shared)['yaw_rate']))
        self.assertTrue(all(y >= -0.02 for y in yaws))
        self.assertGreater(max(yaws), 0.08)
        # At target heading → hold (no perpetual spin / reverse).
        shared['attitude'] = {
            'roll': 0.0, 'pitch': 0.0,
            'yaw': float(planner._course_mem_yaw_tgt),
        }
        planner._pass_t = time.monotonic() - 1.0
        planner._last_t = time.monotonic() - 0.05
        shared['gate_detection'] = {}
        shared['dual_gate_pnp'] = {'n_solved': 0}
        settled = float(planner.compute_target(shared)['yaw_rate'])
        self.assertLess(abs(settled), 0.08)

    def test_course2_memory_not_after_later_gates(self):
        """Right-yaw memory is g1→g2 only — never after gate 2+."""
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.0, ny=0.2, body=None, range_m=None, area=None,
        )
        shared['gate_detection'] = {}
        shared['dual_gate_pnp'] = {'n_solved': 0}
        shared['race_status'] = {'active_gate': 2}
        shared['attitude'] = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        _airborne(planner, shared)
        planner._active_gate = 1
        planner._course_mem_spent = False  # even if flag were wrong…
        planner._climb_f = 1.40
        shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -1.40}
        shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -1.40}
        planner._last_t = time.monotonic() - 0.05
        planner.compute_target(shared)
        self.assertFalse(planner._course_mem)
        self.assertIsNone(planner._course_mem_yaw_tgt)

    def test_course2_memory_one_shot_only(self):
        """Do not re-seed right yaw if g1 pass somehow fires again."""
        planner = AssistImagePlanner()
        self.assertTrue(
            planner._seed_course_memory(time.monotonic(), 1, 0.0)
        )
        self.assertTrue(planner._course_mem_spent)
        planner._course_mem = False
        planner._course_mem_yaw_tgt = None
        self.assertFalse(
            planner._seed_course_memory(time.monotonic(), 1, 0.0)
        )

    def test_course2_memory_ignores_left_steal(self):
        """132028: left ghost must not clear g1→g2 right memory."""
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.0, ny=0.2, body=None, range_m=None, area=None,
        )
        shared['gate_detection'] = {}
        shared['dual_gate_pnp'] = {'n_solved': 0}
        shared['race_status'] = {'active_gate': 1}
        shared['attitude'] = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        _airborne(planner, shared)
        planner._active_gate = 0
        planner._climb_f = 1.40
        shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -1.40}
        shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -1.40}
        planner._last_t = time.monotonic() - 0.05
        planner.compute_target(shared)
        self.assertTrue(planner._course_mem)
        # Inject a live left ghost (wrong side for g2).
        nx, ny = -0.48, 0.20
        shared['gate_detection'] = {
            'center_px': (320.0 + nx * 320.0, 180.0 + ny * 180.0),
            'area_px': 800.0,
            'ts': time.monotonic(),
            'source': 'yolo',
        }
        shared['dual_gate_pnp'] = {'n_solved': 0}
        yaws = []
        for _ in range(8):
            planner._pass_t = time.monotonic() - 0.5
            planner._last_t = time.monotonic() - 0.05
            yaws.append(float(planner.compute_target(shared)['yaw_rate']))
        self.assertTrue(planner._course_mem)
        self.assertTrue(all(y > 0.05 for y in yaws))

    def test_course2_memory_keeps_yaw_until_heading(self):
        """132813: live right box must not kill the turn before budget done."""
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.0, ny=0.2, body=None, range_m=None, area=None,
        )
        shared['gate_detection'] = {}
        shared['dual_gate_pnp'] = {'n_solved': 0}
        shared['race_status'] = {'active_gate': 1}
        shared['attitude'] = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        _airborne(planner, shared)
        planner._active_gate = 0
        planner._climb_f = 1.40
        shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -1.40}
        shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -1.40}
        planner._last_t = time.monotonic() - 0.05
        planner.compute_target(shared)
        self.assertTrue(planner._course_mem)
        tgt = float(planner._course_mem_yaw_tgt)
        # Strong right live (the 132813 dual_pnp handoff) while still short.
        nx, ny = 0.53, 0.05
        shared['gate_detection'] = {
            'center_px': (320.0 + nx * 320.0, 180.0 + ny * 180.0),
            'area_px': 2000.0,
            'ts': time.monotonic(),
            'source': 'yolo',
        }
        shared['dual_gate_pnp'] = {
            'n_solved': 2,
            'gate1_range_m': 18.0,
            'gate1_norm_x': 0.53,
            'gate1_norm_y': 0.05,
        }
        yaws = []
        for i in range(8):
            # Advance heading only partway — must keep yawing right.
            shared['attitude'] = {
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.15 * tgt * (i / 8.0),
            }
            planner._pass_t = time.monotonic() - 0.3
            planner._last_t = time.monotonic() - 0.05
            yaws.append(float(planner.compute_target(shared)['yaw_rate']))
        self.assertFalse(planner._course_mem_done)
        self.assertTrue(all(y > math.radians(12.0) for y in yaws))

    def test_course2_memory_stops_after_budget(self):
        """133354: turn must release after commanded angle budget."""
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.0, ny=0.2, body=None, range_m=None, area=None,
        )
        shared['gate_detection'] = {}
        shared['dual_gate_pnp'] = {'n_solved': 0}
        shared['race_status'] = {'active_gate': 1}
        shared['attitude'] = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        _airborne(planner, shared)
        planner._active_gate = 0
        planner._climb_f = 1.40
        shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -1.40}
        shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -1.40}
        planner._last_t = time.monotonic() - 0.05
        planner.compute_target(shared)
        budget = float(planner._course_mem_yaw_budget)
        self.assertGreater(budget, 0.0)
        # Simulate enough commanded integral without attitude catching up.
        for _ in range(40):
            planner._pass_t = time.monotonic() - 0.5
            planner._last_t = time.monotonic() - 0.05
            shared['attitude'] = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
            shared['gate_detection'] = {}
            yr = float(planner.compute_target(shared)['yaw_rate'])
            if planner._course_mem_done:
                self.assertLess(abs(yr), 0.05)
                break
        self.assertTrue(planner._course_mem_done)
        # Synthetic aim must not linger after the turn (133934 floor chase).
        self.assertFalse(planner._next_course_mem)
        self.assertIsNone(planner._next_nx)

    def test_course2_handoff_center_live_after_turn(self):
        """133934: after right turn, near-center live g2 must be chaseable."""
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.0, ny=0.2, body=None, range_m=None, area=None,
        )
        shared['gate_detection'] = {}
        shared['dual_gate_pnp'] = {'n_solved': 0}
        shared['race_status'] = {'active_gate': 1}
        shared['attitude'] = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        _airborne(planner, shared)
        planner._active_gate = 0
        planner._climb_f = 1.40
        shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -1.40}
        shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -1.40}
        planner._last_t = time.monotonic() - 0.05
        planner.compute_target(shared)
        # Finish the bounded turn.
        planner._course_mem_mark_done()
        planner._pass_t = time.monotonic() - 0.5
        planner._seek_until = time.monotonic() + 10.0
        nx, ny = -0.10, 0.08
        shared['gate_detection'] = {
            'center_px': (320.0 + nx * 320.0, 180.0 + ny * 180.0),
            'area_px': 1600.0,
            'ts': time.monotonic(),
            'source': 'yolo',
        }
        shared['dual_gate_pnp'] = {'n_solved': 0}
        yaws = []
        for _ in range(6):
            planner._pass_t = time.monotonic() - 0.5
            planner._last_t = time.monotonic() - 0.05
            yaws.append(float(planner.compute_target(shared)['yaw_rate']))
        self.assertTrue(planner._have_filt)
        self.assertLess(abs(float(planner._nx_f) + 0.10), 0.08)
        # Fine left allowed to center — not hard-zeroed by post-g1 lock.
        self.assertTrue(any(y < -0.05 for y in yaws))

    def test_course2_rejects_floor_live_after_g1(self):
        """133934: floor-band box must not become the post-g1 aim."""
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.0, ny=0.2, body=None, range_m=None, area=None,
        )
        shared['gate_detection'] = {}
        shared['dual_gate_pnp'] = {'n_solved': 0}
        shared['race_status'] = {'active_gate': 1}
        shared['attitude'] = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        _airborne(planner, shared)
        planner._active_gate = 0
        planner._climb_f = 1.40
        shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -1.40}
        shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -1.40}
        planner._last_t = time.monotonic() - 0.05
        planner.compute_target(shared)
        planner._course_mem_mark_done()
        planner._pass_t = time.monotonic() - 0.5
        planner._seek_until = time.monotonic() + 10.0
        nx, ny = 0.40, 0.86
        shared['gate_detection'] = {
            'center_px': (320.0 + nx * 320.0, 180.0 + ny * 180.0),
            'area_px': 1200.0,
            'ts': time.monotonic(),
            'source': 'yolo',
        }
        planner.compute_target(shared)
        self.assertFalse(planner._have_filt)
        self.assertIsNone(planner._next_nx)

    def test_pitch_slew_prevents_tip_flip(self):
        """125233: speed-brake must not reverse-nod tip every tick."""
        planner = AssistImagePlanner()
        shared = self._shared(nx=0.0, ny=0.12, range_m=14.0)
        shared['attitude'] = {
            'roll': 0.0,
            'pitch': math.radians(12.0) * float(config.FORWARD_PITCH_SIGN),
            'yaw': 0.0,
        }
        shared['local_position_ned'] = {
            'x': 0.0, 'y': 0.0, 'z': -1.2,
            'vx': 7.0, 'vy': 0.0, 'vz': 0.0,
        }
        _airborne(planner, shared)
        pitches = []
        for _ in range(10):
            planner._last_t = time.monotonic() - 0.05
            shared['local_position_ned']['vx'] = 7.0 if _ % 2 == 0 else 3.0
            tgt = planner.compute_target(shared)
            look = float(tgt['desired_pitch']) * float(config.FORWARD_PITCH_SIGN)
            pitches.append(look)
            self.assertGreaterEqual(look, -math.radians(1.0))
        # Adjacent samples must not jump by a full tip (~16°).
        for a, b in zip(pitches, pitches[1:]):
            self.assertLess(abs(a - b), math.radians(8.0))

    def test_freeze_skips_next_not_behind_primary(self):
        """Do not snapshot a 'next' at nearly the same range as the active gate."""
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.02, ny=0.22, body=[8.0, 0.3, 0.1], range_m=8.0, area=9000.0,
        )
        # gate2 at ~9.5 m — ahead of min? 9.5 < 12 min_ahead → hint ignored.
        # Use 13 m next with primary 11 m → not behind by margin 3.
        shared['dual_gate_pnp']['gate2_body'] = [12.5, 1.0, 0.5]
        shared['dual_gate_pnp']['gate1_range_m'] = 11.0
        shared['dual_gate_pnp']['n_solved'] = 2
        shared['race_status'] = {'active_gate': 1}
        _airborne(planner, shared)
        planner._active_gate = 1
        planner._last_t = time.monotonic() - 0.05
        planner.compute_target(shared)
        # 12.5-ish rng may latch as next, but must not freeze as snap vs 11 m.
        if planner._next_rng is not None:
            self.assertGreaterEqual(
                float(planner._next_rng),
                float(config.ASSIST_LATCH_MIN_AHEAD_M) - 0.05,
            )
        self.assertIsNone(planner._snap_next_nx)

    def test_post_pass_cruise_climbs_when_low_and_blind(self):
        """After pass below cruise with no live dig — climb to cruise."""
        hover = float(config.HOVER_THRUST)
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.0, ny=0.2, body=None, range_m=None, area=None,
        )
        shared['gate_detection'] = {}
        shared['dual_gate_pnp'] = {'n_solved': 0}
        _airborne(planner, shared)
        # Above floor, below cruise — must lift eyes before tip-crawl.
        planner._climb_f = 0.80
        shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -0.80}
        shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -0.80}
        planner._seek_until = time.monotonic() + 10.0
        planner._coast_until = 0.0
        planner._pass_t = time.monotonic() - 0.5
        planner._reset_gate_lock()
        planner._next_nx = 0.25
        planner._next_ny = 0.30
        planner._next_rng = 16.0
        planner._next_t = time.monotonic() - 0.2
        planner._have_filt = True
        planner._nx_f = 0.25
        planner._ny_f = 0.30
        planner._last_see_t = time.monotonic() - 0.2
        planner._last_t = time.monotonic() - 0.05
        thr = float(planner.compute_target(shared)['thrust'])
        self.assertEqual(shared['kalman_path']['vert_src'], 'seek_cruise')
        self.assertGreater(thr, hover + 0.008)

    def test_post_pass_accepts_low_small_next_gate(self):
        """Gate-2 style low/small box must soft-yaw, not sit in seek_lock."""
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.08, ny=0.88, body=[20.0, 1.5, 2.0], range_m=20.0, area=250.0,
        )
        _airborne(planner, shared)
        planner._climb_f = 2.5
        shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -2.5}
        shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -2.5}
        planner._seek_until = time.monotonic() + 10.0
        planner._coast_until = 0.0
        planner._pass_t = time.monotonic() - 2.0
        planner._reset_gate_lock()
        planner._last_t = time.monotonic() - 0.05
        tgt = planner.compute_target(shared)
        self.assertEqual(shared['kalman_path']['phase'], 'seek_yaw')
        self.assertTrue(shared['kalman_path']['chaseable'])
        # Nearly centered — yaw may be tiny; tip + not seek_lock is the point.
        look = float(tgt['desired_pitch']) * float(config.FORWARD_PITCH_SIGN)
        self.assertGreater(look, math.radians(5.0))
        self.assertEqual(shared['kalman_path']['vert_src'], 'seek_sink')

    def test_seek_sinks_when_gate_low_rises_when_high(self):
        """Seek altitude follows image ny: low→sink, high→climb."""
        hover = float(config.HOVER_THRUST)
        ny_aim = float(getattr(config, 'ASSIST_NY_AIM', 0.20))

        def _seek_thr(ny, climb=2.5):
            planner = AssistImagePlanner()
            shared = self._shared(
                nx=0.10, ny=ny, body=[18.0, 2.0, 0.0], range_m=18.0, area=1200.0,
            )
            _airborne(planner, shared)
            planner._climb_f = float(climb)
            shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -float(climb)}
            shared['local_position_ned'] = {
                'x': 0.0, 'y': 0.0, 'z': -float(climb),
            }
            planner._seek_until = time.monotonic() + 10.0
            planner._coast_until = 0.0
            planner._pass_t = time.monotonic() - 2.0
            planner._reset_gate_lock()
            planner._last_t = time.monotonic() - 0.05
            tgt = planner.compute_target(shared)
            return float(tgt['thrust']), shared['kalman_path']['vert_src']

        thr_low, vert_low = _seek_thr(0.85, climb=2.5)
        thr_high, vert_high = _seek_thr(-0.10, climb=2.5)
        thr_aim, vert_aim = _seek_thr(ny_aim, climb=2.5)
        self.assertEqual(vert_low, 'seek_sink')
        self.assertEqual(vert_high, 'seek_climb')
        self.assertEqual(vert_aim, 'seek_hold')
        self.assertLess(thr_low, thr_aim)
        self.assertGreater(thr_high, thr_aim)
        self.assertLess(thr_low, hover)

        # Hard floor near pad / below seek min alt.
        _thr_floor, vert_floor = _seek_thr(0.85, climb=0.30)
        self.assertEqual(vert_floor, 'seek_floor')

    def test_post_pass_allows_proportional_sink(self):
        """After pass, sink toward low gate immediately (no alt freeze)."""
        hover = float(config.HOVER_THRUST)
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.10, ny=0.80, body=[18.0, 2.0, 1.5], range_m=18.0, area=1200.0,
        )
        _airborne(planner, shared)
        planner._climb_f = 2.5
        shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -2.5}
        shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -2.5}
        planner._seek_until = time.monotonic() + 10.0
        planner._coast_until = 0.0
        planner._pass_t = time.monotonic() - 0.05
        planner._reset_gate_lock()
        planner._last_t = time.monotonic() - 0.05
        thr = float(planner.compute_target(shared)['thrust'])
        self.assertIn(
            shared['kalman_path']['vert_src'],
            ('seek_sink', 'seek_pose_sink', 'pose_g1:sink'),
        )
        self.assertLess(thr, hover - 0.005)

    def test_seek_floor_stops_pose_sink(self):
        """110520: pose_sink must not dig below SEEK_MIN_ALT."""
        hover = float(config.HOVER_THRUST)
        min_alt = float(getattr(config, 'ASSIST_SEEK_MIN_ALT_M', 1.80))
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.05, ny=0.70, body=[14.0, 1.0, 1.5], range_m=14.0, area=1500.0,
        )
        _airborne(planner, shared)
        climb = min_alt - 0.20
        planner._climb_f = climb
        shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -climb}
        shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -climb}
        planner._seek_until = time.monotonic() + 10.0
        planner._coast_until = 0.0
        planner._pass_t = time.monotonic() - 1.0
        planner._last_t = time.monotonic() - 0.05
        thr = float(planner.compute_target(shared)['thrust'])
        self.assertEqual(shared['kalman_path']['vert_src'], 'seek_floor')
        self.assertGreaterEqual(thr, hover + 0.005)

    def test_seek_ignores_far_low_gate_for_sink(self):
        """110826: far low box (gate 3) must not drive dig after gate 1."""
        hover = float(config.HOVER_THRUST)
        # Near next-gate body in dual; far YOLO box tries to steal.
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.05, ny=0.85, body=[12.0, 0.5, 0.4], range_m=40.0, area=400.0,
        )
        shared['dual_gate_pnp'] = {
            'gate1_norm_x': 0.05,
            'gate1_norm_y': 0.85,
            'gate1_range_m': 40.0,
            'gate1_body': [40.0, 1.0, 8.0],
            'gate2_body': [12.0, 0.5, 0.4],
            'n_solved': 2,
        }
        _airborne(planner, shared)
        planner._climb_f = 2.6
        shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -2.6}
        shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -2.6}
        planner._seek_until = time.monotonic() + 10.0
        planner._coast_until = 0.0
        planner._pass_t = time.monotonic() - 0.2
        planner._next_nx = 0.04
        planner._next_ny = 0.22
        planner._next_rng = 12.0
        planner._next_body = np.array([12.0, 0.5, 0.0], dtype=np.float64)
        planner._next_t = time.monotonic() - 0.1
        planner._have_filt = True
        planner._nx_f = 0.04
        planner._ny_f = 0.22
        planner._body_f = planner._next_body.copy()
        planner._last_see_t = time.monotonic()
        planner._last_t = time.monotonic() - 0.05
        thr = float(planner.compute_target(shared)['thrust'])
        path = shared['kalman_path']
        # Kept near latch — did not adopt the far low box as the sink target.
        self.assertLess(float(path['range_m']), 18.0)
        self.assertLess(abs(float(path['norm_y'])), 0.40)
        # Not max-rail dig (0.20) toward gate 3.
        self.assertGreater(thr, 0.220)

    def test_seek_rejects_far_midframe_range_jump(self):
        """111515: far mid-frame box must not overwrite near latch / yaw."""
        planner = AssistImagePlanner()
        # Live YOLO: ~44 m, almost centered vertically (not "low").
        shared = self._shared(
            nx=-0.10, ny=0.16, body=[40.0, -4.0, 2.0], range_m=44.0, area=500.0,
        )
        shared['dual_gate_pnp'] = {
            'gate1_norm_x': -0.10,
            'gate1_norm_y': 0.16,
            'gate1_range_m': 44.0,
            'gate1_body': [40.0, -4.0, 2.0],
            # Far "gate2" that would poison the latch if accepted.
            'gate2_body': [42.0, -5.0, 3.0],
            'n_solved': 2,
        }
        _airborne(planner, shared)
        planner._climb_f = 2.8
        shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -2.8}
        shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -2.8}
        planner._seek_until = time.monotonic() + 10.0
        planner._coast_until = 0.0
        planner._pass_t = time.monotonic() - 0.5
        planner._next_nx = 0.24
        planner._next_ny = 0.06
        planner._next_rng = 8.0
        planner._next_body = np.array([8.0, 1.9, 0.5], dtype=np.float64)
        planner._next_t = time.monotonic() - 0.2
        planner._have_filt = True
        planner._nx_f = 0.24
        planner._ny_f = 0.06
        planner._body_f = planner._next_body.copy()
        planner._last_range_m = 8.0
        planner._last_see_t = time.monotonic()
        planner._last_t = time.monotonic() - 0.05
        tgt = planner.compute_target(shared)
        path = shared['kalman_path']
        # Latch range/aim held — not the 44 m steal.
        self.assertLess(float(path['range_m']), 14.0)
        self.assertGreater(float(path['norm_x']), 0.12)
        self.assertAlmostEqual(float(planner._next_rng), 8.0, places=2)
        # Soft-yaw should still pull toward latched +right, not far left.
        self.assertGreater(float(tgt['yaw_rate']), 0.0)

    def test_seek_accepts_live_within_two_gates(self):
        """112030: ~20 m live gate-2 must win over an 8 m sky latch."""
        hover = float(config.HOVER_THRUST)
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.06, ny=0.55, body=[18.0, 1.0, 1.2], range_m=20.0, area=1200.0,
        )
        _airborne(planner, shared)
        planner._climb_f = 2.6
        shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -2.6}
        shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -2.6}
        planner._seek_until = time.monotonic() + 10.0
        planner._coast_until = 0.0
        planner._pass_t = time.monotonic() - 0.3
        planner._next_nx = 0.05
        planner._next_ny = -0.22  # bad sky latch from 112030
        planner._next_rng = 8.0
        planner._next_body = np.array([8.0, 0.4, -0.9], dtype=np.float64)
        planner._next_t = time.monotonic() - 0.1
        planner._have_filt = True
        planner._nx_f = 0.05
        planner._ny_f = -0.22
        planner._body_f = planner._next_body.copy()
        planner._last_range_m = 8.0
        planner._last_see_t = time.monotonic()
        planner._last_t = time.monotonic() - 0.05
        thr = float(planner.compute_target(shared)['thrust'])
        path = shared['kalman_path']
        # Live low box wins — not stuck on sky latch ny=-0.22.
        self.assertGreater(float(path['norm_y']), 0.30)
        self.assertLess(float(path['range_m']), 28.0)
        self.assertIn(
            path['vert_src'],
            ('seek_sink', 'seek_pose_sink', 'pose_g1:sink', 'seek_hold'),
        )
        self.assertNotEqual(path['vert_src'], 'seek_climb')
        # Hold may sit near hover/tilt (~+0.01); just no climb punch.
        self.assertLess(thr, hover + 0.018)

    def test_seek_stops_sink_when_pose_level(self):
        """Pose at gate height + YOLO near aim → stop dig."""
        hover = float(config.HOVER_THRUST)
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.05, ny=0.35, body=[10.0, 0.5, 0.05], range_m=10.0, area=2000.0,
        )
        _airborne(planner, shared)
        planner._climb_f = 1.20
        shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -1.20}
        shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -1.20}
        planner._seek_until = time.monotonic() + 10.0
        planner._coast_until = 0.0
        planner._pass_t = time.monotonic() - 1.0
        planner._gate_lock = True
        planner._last_t = time.monotonic() - 0.05
        thr = float(planner.compute_target(shared)['thrust'])
        self.assertNotEqual(shared['kalman_path']['vert_src'], 'seek_sink')
        self.assertNotEqual(shared['kalman_path']['vert_src'], 'seek_pose_sink')
        self.assertGreaterEqual(thr, hover - 0.020)

    def test_seek_brakes_fast_descent(self):
        """Near gate height + fast fall → raise thrust (not while still high/low)."""
        hover = float(config.HOVER_THRUST)
        planner = AssistImagePlanner()
        # ny near aim + already low — full brake allowed.
        shared = self._shared(
            nx=0.05, ny=0.32, body=[8.0, 0.3, 0.2], range_m=8.0, area=3500.0,
        )
        _airborne(planner, shared)
        planner._climb_f = 0.95
        shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -0.95}
        shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -0.95}
        planner._seek_until = time.monotonic() + 10.0
        planner._coast_until = 0.0
        planner._pass_t = time.monotonic() - 1.0
        planner._gate_lock = True
        planner._climb_rate = -1.50
        planner._climb_rate_t = time.monotonic() - 0.05
        planner._climb_rate_z = 1.05  # implies falling into 0.95
        planner._last_t = time.monotonic() - 0.05
        thr = float(planner.compute_target(shared)['thrust'])
        # Near height + fast fall must not keep a hard dig below hover.
        self.assertGreaterEqual(thr, hover - 0.005)

    def test_seek_keeps_sinking_when_still_high_and_low(self):
        """115153: don't hover-brake at ~2 m while ny still clearly low."""
        hover = float(config.HOVER_THRUST)
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.05, ny=0.65, body=[18.0, 0.5, 0.8], range_m=18.0, area=1200.0,
        )
        _airborne(planner, shared)
        planner._climb_f = 2.00
        shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -2.00}
        shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -2.00}
        planner._seek_until = time.monotonic() + 10.0
        planner._coast_until = 0.0
        planner._pass_t = time.monotonic() - 0.5
        planner._gate_lock = True
        planner._climb_rate = -0.90
        planner._climb_rate_t = time.monotonic()
        planner._climb_rate_z = 2.00
        planner._last_t = time.monotonic() - 0.05
        thr = float(planner.compute_target(shared)['thrust'])
        self.assertNotEqual(
            shared['kalman_path']['vert_src'], 'seek_descent_brake'
        )
        self.assertLess(thr, hover - 0.010)

    def test_seek_holds_near_cruise_despite_tip_ny(self):
        """Near cruise + pose_dz≈0 + tip → hold (do not dig to pad)."""
        hover = float(config.HOVER_THRUST)
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.05, ny=0.70, body=[18.0, 0.5, 0.02], range_m=18.0, area=1200.0,
        )
        _airborne(planner, shared)
        planner._climb_f = 1.70
        shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -1.70}
        shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -1.70}
        planner._seek_until = time.monotonic() + 10.0
        planner._coast_until = 0.0
        planner._pass_t = time.monotonic() - 0.5
        planner._gate_lock = True
        planner._last_t = time.monotonic() - 0.05
        thr = float(planner.compute_target(shared)['thrust'])
        self.assertNotEqual(shared['kalman_path']['vert_src'], 'seek_sink')
        self.assertGreaterEqual(thr, hover - 0.012)

    def test_seek_digs_from_high_tip_toward_cruise(self):
        """123610: pose≈0 + tip at ~3 m must sink (not hold then loft)."""
        hover = float(config.HOVER_THRUST)
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.05, ny=0.70, body=[18.0, 0.5, 0.02], range_m=18.0, area=1200.0,
        )
        _airborne(planner, shared)
        planner._climb_f = 3.10
        shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -3.10}
        shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -3.10}
        planner._seek_until = time.monotonic() + 10.0
        planner._coast_until = 0.0
        planner._pass_t = time.monotonic() - 0.5
        planner._gate_lock = True
        planner._last_t = time.monotonic() - 0.05
        thr = float(planner.compute_target(shared)['thrust'])
        self.assertIn(
            shared['kalman_path']['vert_src'],
            ('seek_sink', 'seek_ceiling', 'seek_pose_sink', 'pose_g1:sink'),
        )
        self.assertLess(thr, hover - 0.008)

    def test_blind_scan_cap_persists_after_post_pass_window(self):
        """123610: >8 s after pass, blind scan must still bleed loft."""
        hover = float(config.HOVER_THRUST)
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.0, ny=0.2, body=None, range_m=None, area=None,
        )
        shared['gate_detection'] = {}
        shared['dual_gate_pnp'] = {'n_solved': 0}
        _airborne(planner, shared)
        planner._climb_f = 3.50
        shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -3.50}
        shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -3.50}
        planner._seek_until = time.monotonic() + 20.0
        planner._coast_until = 0.0
        planner._pass_t = time.monotonic() - 10.0  # past 8 s window
        planner._reset_gate_lock()
        planner._clear_next_latch()
        planner._have_filt = False
        planner._last_see_t = time.monotonic() - 2.0
        planner._last_t = time.monotonic() - 0.05
        thr = float(planner.compute_target(shared)['thrust'])
        self.assertIn(
            shared['kalman_path']['vert_src'],
            ('seek_scan_cap', 'seek_ceiling'),
        )
        self.assertLess(thr, hover - 0.015)

    def test_seek_stops_tip_dig_near_height_with_zero_pose(self):
        """115626: at ~1 m, pose_dz≈0 wins over tipped ny — don't bury rail."""
        hover = float(config.HOVER_THRUST)
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.05, ny=0.65, body=[13.0, 0.3, 0.02], range_m=13.0, area=2500.0,
        )
        _airborne(planner, shared)
        planner._climb_f = 1.00
        shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -1.00}
        shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -1.00}
        planner._seek_until = time.monotonic() + 10.0
        planner._coast_until = 0.0
        planner._pass_t = time.monotonic() - 1.0
        planner._gate_lock = True
        planner._climb_rate = -1.80
        planner._climb_rate_t = time.monotonic()
        planner._climb_rate_z = 1.00
        planner._last_t = time.monotonic() - 0.05
        thr = float(planner.compute_target(shared)['thrust'])
        self.assertNotEqual(shared['kalman_path']['vert_src'], 'seek_sink')
        self.assertGreaterEqual(thr, hover - 0.005)

    def test_seek_punch_holds_when_close(self):
        """Close live gate: stop dig and punch (no pose required)."""
        hover = float(config.HOVER_THRUST)
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.05, ny=0.65, body=[7.0, 0.3, 0.8], range_m=7.5, area=4000.0,
        )
        _airborne(planner, shared)
        planner._climb_f = 1.0
        shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -1.0}
        shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -1.0}
        planner._seek_until = time.monotonic() + 10.0
        planner._coast_until = 0.0
        planner._pass_t = time.monotonic() - 1.0
        planner._gate_lock = True
        planner._last_t = time.monotonic() - 0.05
        thr = float(planner.compute_target(shared)['thrust'])
        self.assertNotEqual(shared['kalman_path']['vert_src'], 'seek_sink')
        self.assertGreater(thr, hover - 0.025)

    def test_latch_only_does_not_climb(self):
        """112030: next_latch with ny above aim must hold, not loft."""
        hover = float(config.HOVER_THRUST)
        planner = AssistImagePlanner()
        shared = self._shared(nx=0.05, ny=-0.22, body=[8.0, 0.4, -0.9], range_m=8.0)
        # No live image — only the pre-pass latch remains.
        shared['gate_detection'] = {}
        shared['dual_gate_pnp'] = {'n_solved': 0}
        _airborne(planner, shared)
        planner._climb_f = 2.6
        shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -2.6}
        shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -2.6}
        planner._seek_until = time.monotonic() + 10.0
        planner._coast_until = 0.0
        planner._pass_t = time.monotonic() - 0.4
        planner._next_nx = 0.05
        planner._next_ny = -0.22
        planner._next_rng = 8.0
        planner._next_body = np.array([8.0, 0.4, -0.9], dtype=np.float64)
        planner._next_t = time.monotonic() - 0.1
        planner._have_filt = True
        planner._nx_f = 0.05
        planner._ny_f = -0.22
        planner._body_f = planner._next_body.copy()
        planner._last_range_m = 8.0
        planner._last_see_t = time.monotonic()
        planner._seek_seen = True
        planner._gate_lock = True
        planner._last_t = time.monotonic() - 0.05
        thr = float(planner.compute_target(shared)['thrust'])
        path = shared['kalman_path']
        self.assertNotIn('climb', path['vert_src'])
        self.assertLessEqual(thr, hover + 0.002)

    def test_approach_high_optional_climbs_when_enabled(self):
        """ASSIST_APPROACH_HIGH_M>0 shifts hold upward (default is 0)."""
        hover = float(config.HOVER_THRUST)
        old_bias = getattr(config, 'ASSIST_CAM_TILT_BIAS', 1.0)
        old_high = getattr(config, 'ASSIST_APPROACH_HIGH_M', 0.0)
        config.ASSIST_CAM_TILT_BIAS = 0.0
        config.ASSIST_APPROACH_HIGH_M = 0.50
        try:
            planner = AssistImagePlanner()
            shared = self._shared(
                nx=0.0, ny=0.05, body=[10.0, 0.0, 0.0], range_m=10.0,
            )
            _airborne(planner, shared)
            thr = float(planner.compute_target(shared)['thrust'])
            self.assertIn('climb', shared['kalman_path']['vert_src'])
            self.assertGreater(thr, hover + 0.001)
        finally:
            config.ASSIST_CAM_TILT_BIAS = old_bias
            config.ASSIST_APPROACH_HIGH_M = old_high

    def test_no_false_climb_when_ny_near_aim(self):
        """105106: residual pose_dz<0 at ny≈aim must not loft."""
        hover = float(config.HOVER_THRUST)
        old_bias = getattr(config, 'ASSIST_CAM_TILT_BIAS', 1.0)
        config.ASSIST_CAM_TILT_BIAS = 0.0
        try:
            planner = AssistImagePlanner()
            # Large "gate above" pose but image already at/below aim.
            ny_aim = float(getattr(config, 'ASSIST_NY_AIM', 0.22))
            shared = self._shared(
                nx=0.0, ny=ny_aim + 0.05, body=[14.0, 0.0, -2.8], range_m=14.0,
            )
            _airborne(planner, shared)
            thr = float(planner.compute_target(shared)['thrust'])
            self.assertNotIn('climb', shared['kalman_path']['vert_src'])
            self.assertLessEqual(thr, hover + 0.008)
        finally:
            config.ASSIST_CAM_TILT_BIAS = old_bias

    def test_coast_keeps_lift_while_seeking(self):
        """Through-slot coast must not get seek thrust bleed (104433)."""
        hover = float(config.HOVER_THRUST)
        planner = AssistImagePlanner()
        shared = self._shared(
            nx=0.0, ny=0.20, body=[8.0, 0.0, 0.0], range_m=8.0, area=6000.0,
        )
        _airborne(planner, shared)
        planner._climb_f = 2.9
        shared['position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -2.9}
        shared['local_position_ned'] = {'x': 0.0, 'y': 0.0, 'z': -2.9}
        # Post-pass coast (seeking + coasting) — used to overwrite with seek_hold.
        planner._seek_until = time.monotonic() + 10.0
        planner._coast_until = time.monotonic() + 2.0
        planner._pass_t = time.monotonic() - 0.05
        planner._have_filt = True
        planner._nx_f = 0.0
        planner._ny_f = 0.20
        planner._last_t = time.monotonic() - 0.05
        tgt = planner.compute_target(shared)
        self.assertEqual(shared['kalman_path']['phase'], 'coast')
        # High coast settles (123610 exited at 3.3 m); low coast lifts.
        self.assertIn(
            shared['kalman_path']['vert_src'],
            ('coast_lift', 'coast_settle'),
        )
        if shared['kalman_path']['vert_src'] == 'coast_lift':
            self.assertGreaterEqual(float(tgt['thrust']), hover)


if __name__ == '__main__':
    unittest.main()
