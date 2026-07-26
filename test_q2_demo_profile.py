"""Regression tests for the deployed profile ported from collect_demos.py."""

import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import config
from controller import Controller
from dreamer_drone.config import load_config
from dreamer_drone.env.spaces import VECTOR_OBS_FIELDS, scale_action
from test_opencv_gate_navigation import detection_at
from vision.navigation import (
    GateNavigator,
    NavigationState,
    q2_demo_navigation_config,
)
from vision.gate_tracker import GateTracker, q2_demo_tracker_config


class _FakeMav:
    def __init__(self):
        self.attitude_targets = []

    def set_attitude_target_send(self, *args):
        self.attitude_targets.append(args)

    def command_long_send(self, *args):
        pass


class _FakeConnection:
    target_system = 1
    target_component = 1

    def __init__(self):
        self.mav = _FakeMav()


class DemoNavigationProfileTests(unittest.TestCase):
    def test_profile_matches_collect_demos_tuning(self):
        cfg = q2_demo_navigation_config()
        self.assertAlmostEqual(cfg.search_forward_mps, 0.0)
        self.assertAlmostEqual(cfg.maximum_approach_mps, 0.65)
        self.assertAlmostEqual(cfg.vertical_setpoint_normalized, 0.24)
        self.assertAlmostEqual(cfg.vertical_deadband, 0.20)
        self.assertAlmostEqual(
            cfg.vertical_control_min_area_ratio,
            0.0,
        )
        self.assertAlmostEqual(cfg.horizontal_yaw_kp, 0.75)
        self.assertAlmostEqual(cfg.max_yaw_rate_rps, 0.48)
        self.assertEqual(cfg.path_lateral_kp, 0.0)
        self.assertEqual(cfg.path_yaw_kp, 0.0)
        self.assertGreater(
            cfg.next_gate_max_right_mps,
            cfg.next_gate_max_yaw_rate_rps,
        )
        self.assertAlmostEqual(cfg.commit_opening_area_ratio, 0.030)
        self.assertAlmostEqual(cfg.commit_alignment_tolerance, 0.40)
        self.assertAlmostEqual(cfg.commit_horizontal_tolerance, 0.10)

    def test_recorded_close_gate_enters_commit_before_dropout(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        far = detection_at(
            opening_width=50,
            opening_height=40,
            stable_frames=5,
        )
        navigator.update(far, 1.0)
        navigator.update(far, 1.1)
        navigator.update(far, 1.2)
        close = detection_at(
            # Latest gate-one telemetry places the opening near -0.20 at
            # commit; this remains inside the raised flight-line tolerance.
            ny=-0.20,
            opening_width=100,
            opening_height=82,
            stable_frames=1,
            size_rate=12.0,
        )
        command = navigator.update(close, 1.3)
        self.assertEqual(command.state.value, 'COMMIT')
        after_dropout = navigator.update(None, 1.4)
        self.assertEqual(after_dropout.state.value, 'PASS_THROUGH')

    def test_close_but_off_center_gate_does_not_commit(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        aligned_far = detection_at(
            opening_width=50,
            opening_height=40,
            stable_frames=5,
        )
        navigator.update(aligned_far, 1.0)
        navigator.update(aligned_far, 1.1)
        navigator.update(aligned_far, 1.2)
        off_center_close = detection_at(
            nx=0.18,
            ny=-0.30,
            opening_width=100,
            opening_height=82,
            stable_frames=5,
        )
        command = navigator.update(off_center_close, 1.3)
        self.assertEqual(
            command.state, NavigationState.ALIGN_AND_APPROACH
        )

    def test_q2_tracker_rejects_recorded_false_gate_shapes(self):
        false_gates = (
            detection_at(
                confidence=0.67,
                opening_width=27,
                opening_height=26,
            ),
            detection_at(
                nx=-0.72,
                confidence=0.67,
                opening_width=50,
                opening_height=40,
            ),
            detection_at(
                ny=0.88,
                confidence=0.67,
                opening_width=50,
                opening_height=40,
            ),
        )
        for false_gate in false_gates:
            with self.subTest(
                nx=false_gate.normalized_x,
                ny=false_gate.normalized_y,
                area=false_gate.area_px,
            ):
                tracker = GateTracker(q2_demo_tracker_config())
                self.assertIsNone(
                    tracker.update(false_gate, timestamp=1.0)
                )

    def test_q2_tracker_accepts_forward_course_gate(self):
        tracker = GateTracker(q2_demo_tracker_config())
        forward_gate = detection_at(
            nx=-0.52,
            ny=0.10,
            confidence=0.63,
            opening_width=55,
            opening_height=43,
        )
        self.assertIsNotNone(
            tracker.update(forward_gate, timestamp=1.0)
        )

    def test_no_gate_stops_forward_flight_and_scans(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        command = navigator.update(None, 1.0)
        self.assertAlmostEqual(command.forward_mps, 0.0)
        self.assertGreater(command.yaw_rate_rps, 0.0)

    def test_close_gate_slows_and_brakes_before_commit(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        far = detection_at(
            opening_width=42,
            opening_height=36,
        )
        close = detection_at(
            opening_width=60,
            opening_height=60,
        )
        very_close = detection_at(
            opening_width=80,
            opening_height=75,
        )
        far_speed = navigator._approach_speed(far, 0.0, 0.0)
        close_speed = navigator._approach_speed(close, 0.0, 0.0)
        very_close_speed = navigator._approach_speed(
            very_close, 0.0, 0.0
        )
        self.assertGreater(far_speed, close_speed)
        self.assertGreater(close_speed, very_close_speed)
        self.assertLess(very_close_speed, 0.0)

    def test_gate_capture_favors_translation_over_camera_yaw(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        right_gate = detection_at(
            nx=0.55,
            opening_width=50,
            opening_height=40,
            stable_frames=5,
        )
        navigator.update(right_gate, 1.0)
        navigator.update(right_gate, 1.1)
        command = navigator.update(right_gate, 1.2)

        self.assertEqual(
            command.state, NavigationState.ALIGN_AND_APPROACH
        )
        self.assertGreater(command.right_mps, 1.5)
        self.assertLessEqual(abs(command.yaw_rate_rps), 0.48)

    def test_confirmed_pass_releases_old_gate_immediately(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        navigator.state = NavigationState.COMMIT
        navigator.confirm_gate_pass(2.0)
        self.assertEqual(navigator.state, NavigationState.SEARCH)
        self.assertEqual(navigator._state_since, 2.0)

    def test_vertical_servo_keeps_distant_gate_in_frame(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        far = detection_at(ny=0.70, opening_width=35)
        _, far_vertical, _ = navigator._errors(far)
        self.assertAlmostEqual(far_vertical, 0.26)

        close = detection_at(ny=0.60, opening_width=80)
        _, close_vertical, _ = navigator._errors(close)
        # (0.60 - 0.24 target) - 0.20 framing deadband.
        self.assertAlmostEqual(close_vertical, 0.16)

    def test_gate_near_frame_edge_stops_forward_approach(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        edge_gate = detection_at(
            nx=0.90,
            opening_width=50,
            opening_height=40,
            stable_frames=5,
        )
        command = navigator.update(edge_gate, 1.0)
        self.assertLess(command.forward_mps, 0.0)
        self.assertGreater(command.right_mps, 0.0)

    def test_q2_controller_matches_demo_gain_damping_and_rate_cap(self):
        connection = _FakeConnection()
        data = {
            'attitude': {
                'roll': 0.0,
                'pitch': 0.0,
                'yaw': 0.0,
                'rollspeed': 0.0,
                'pitchspeed': 0.0,
            },
            'highres_imu': {
                'xgyro': 0.0,
                'ygyro': 0.0,
                'zgyro': 0.0,
                'xacc': 0.0,
                'yacc': 0.0,
                'zacc': -9.81,
                'ts_us': 1_000_000,
                'ts': time.time_ns(),
            },
            'planner_target': {
                'vn': 1.0,
                've': 3.0,
                'vd': 0.0,
                'yaw_rate': 9.0,
            },
        }
        controller = Controller(connection, data, system_boot_ms=0)
        with patch('controller.time.sleep'):
            controller.update()

        output = data['control_output']
        self.assertAlmostEqual(output['pitch_rate'], -0.18)
        # Lateral capture uses a stronger bank gain and reaches the 25-degree
        # lean cap for this deliberately large 3 m/s request.
        self.assertAlmostEqual(
            output['roll_rate'],
            config.MAX_LEAN_RAD * config.KP_ATT,
        )
        self.assertAlmostEqual(output['yaw_rate'], config.MAX_RATE_RAD_S)
        for key in ('roll_rate', 'pitch_rate', 'yaw_rate'):
            self.assertLessEqual(
                abs(output[key]),
                config.MAX_RATE_RAD_S,
            )
        self.assertEqual(len(connection.mav.attitude_targets), 1)

    def test_stale_imu_forces_neutral_hover(self):
        connection = _FakeConnection()
        data = {
            'attitude': {'yaw': 0.0},
            'highres_imu': {
                'xgyro': 0.0,
                'ygyro': 0.0,
                'zgyro': 0.0,
                'xacc': 0.0,
                'yacc': 0.0,
                'zacc': -9.81,
                'ts_us': 1_000_000,
                'ts': time.time_ns() - 2_000_000_000,
            },
            'planner_target': {
                'vn': 1.0,
                've': 1.0,
                'vd': -1.0,
                'yaw_rate': 1.0,
            },
        }
        controller = Controller(connection, data, system_boot_ms=0)
        with patch('controller.time.sleep'):
            controller.update()
        output = data['control_output']
        self.assertFalse(output['telemetry_ok'])
        self.assertAlmostEqual(output['thrust'], config.HOVER_THRUST)
        self.assertAlmostEqual(output['yaw_rate'], 0.0)

    def test_arming_applies_bounded_takeoff_boost(self):
        connection = _FakeConnection()
        data = {
            'attitude': {'yaw': 0.0},
            'highres_imu': {
                'xgyro': 0.0,
                'ygyro': 0.0,
                'zgyro': 0.0,
                'xacc': 0.0,
                'yacc': 0.0,
                'zacc': -9.81,
                'ts_us': 1_000_000,
                'ts': time.time_ns(),
            },
            'planner_target': {
                'vn': 0.0,
                've': 0.0,
                'vd': 0.4,
                'yaw_rate': 0.0,
            },
        }
        controller = Controller(connection, data, system_boot_ms=0)
        controller.arm()
        with patch('controller.time.sleep'):
            controller.update()
        self.assertAlmostEqual(
            data['control_output']['thrust'],
            config.TAKEOFF_THRUST,
        )

    def test_controller_replays_saved_demo_pitch_commands(self):
        episode_path = (
            Path(__file__).resolve().parent
            / 'artifacts'
            / 'demos_orig'
            / 'episode_000.npz'
        )
        episode = np.load(episode_path)
        vectors = episode['vector']
        actions = episode['action']
        field = {
            name: index
            for index, name in enumerate(VECTOR_OBS_FIELDS)
        }
        action_cfg = load_config(None).action

        connection = _FakeConnection()
        data = {
            'attitude': {'yaw': 0.0},
            'planner_target': {
                'vn': 1.0,
                've': 0.0,
                'vd': 0.0,
                'yaw_rate': 0.0,
            },
        }
        controller = Controller(connection, data, system_boot_ms=0)
        imu_ts_us = 1_000_000
        errors = []
        with patch('controller.time.sleep'):
            for index, (vector, action) in enumerate(
                zip(vectors, actions)
            ):
                dt = max(0.001, float(vector[field['dt']]))
                imu_ts_us += int(dt * 1e6)
                data['highres_imu'] = {
                    'xgyro': float(vector[field['gyro_x']]),
                    'ygyro': float(vector[field['gyro_y']]),
                    'zgyro': float(vector[field['gyro_z']]),
                    'xacc': float(vector[field['ax']]),
                    'yacc': float(vector[field['ay']]),
                    'zacc': float(vector[field['az']]),
                    'ts_us': imu_ts_us,
                    'ts': time.time_ns(),
                }
                controller.update()
                if index:
                    expected = scale_action(action, action_cfg).pitch_rate
                    actual = data['control_output']['pitch_rate']
                    errors.append(abs(actual - expected))

        self.assertLess(max(errors), 1e-5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
