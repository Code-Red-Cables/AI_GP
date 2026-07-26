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
from vision.navigation import GateNavigator, q2_demo_navigation_config


class _FakeMav:
    def __init__(self):
        self.attitude_targets = []

    def set_attitude_target_send(self, *args):
        self.attitude_targets.append(args)


class _FakeConnection:
    target_system = 1
    target_component = 1

    def __init__(self):
        self.mav = _FakeMav()


class DemoNavigationProfileTests(unittest.TestCase):
    def test_profile_matches_collect_demos_tuning(self):
        cfg = q2_demo_navigation_config()
        self.assertAlmostEqual(cfg.search_forward_mps, 1.0)
        self.assertAlmostEqual(cfg.maximum_approach_mps, 1.0)
        self.assertAlmostEqual(cfg.vertical_setpoint_normalized, 0.16)
        self.assertAlmostEqual(cfg.vertical_deadband, 0.30)
        self.assertAlmostEqual(
            cfg.vertical_control_min_area_ratio,
            40.0 / 4096.0,
        )
        self.assertAlmostEqual(cfg.horizontal_yaw_kp, 2.4)
        self.assertAlmostEqual(cfg.max_yaw_rate_rps, 1.05)

    def test_no_gate_keeps_demo_ballistic_approach_without_yaw(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        command = navigator.update(None, 1.0)
        self.assertAlmostEqual(command.forward_mps, 1.0)
        self.assertAlmostEqual(command.yaw_rate_rps, 0.0)

    def test_vertical_servo_is_close_range_with_demo_deadband(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        far = detection_at(ny=0.70, opening_width=35)
        _, far_vertical, _ = navigator._errors(far)
        self.assertEqual(far_vertical, 0.0)

        close = detection_at(ny=0.60, opening_width=80)
        _, close_vertical, _ = navigator._errors(close)
        # (0.60 - 0.16 target) - 0.30 demonstrated deadband.
        self.assertAlmostEqual(close_vertical, 0.14)

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
        self.assertAlmostEqual(output['roll_rate'], 0.54)
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
