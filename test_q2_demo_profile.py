"""Regression tests for the deployed profile ported from collect_demos.py."""

import math
import time
import unittest
from dataclasses import replace
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
        tracker_cfg = q2_demo_tracker_config()
        self.assertAlmostEqual(cfg.search_forward_mps, 0.0)
        self.assertAlmostEqual(cfg.search_close_forward_mps, 0.06)
        self.assertAlmostEqual(cfg.maximum_approach_mps, 0.22)
        self.assertAlmostEqual(cfg.vertical_setpoint_normalized, 0.0)
        self.assertAlmostEqual(
            cfg.post_pass_vertical_setpoint_normalized,
            -0.08,
        )
        self.assertAlmostEqual(cfg.vertical_deadband, 0.06)
        self.assertAlmostEqual(cfg.vertical_descent_deadband, 0.02)
        self.assertAlmostEqual(cfg.vertical_kp, 0.8)
        self.assertAlmostEqual(cfg.vertical_kd, 2.30)
        self.assertAlmostEqual(
            cfg.severe_horizontal_error_normalized,
            0.50,
        )
        self.assertAlmostEqual(
            cfg.severe_horizontal_forward_cap_mps,
            0.08,
        )
        self.assertAlmostEqual(
            config.MAX_DESCENT_THRUST_REDUCTION,
            0.020,
        )
        self.assertAlmostEqual(
            cfg.vertical_control_min_area_ratio,
            0.0,
        )
        self.assertAlmostEqual(
            cfg.vertical_control_max_horizontal_error,
            0.35,
        )
        self.assertAlmostEqual(cfg.horizontal_yaw_kp, 0.60)
        self.assertAlmostEqual(cfg.horizontal_yaw_kd, 0.75)
        self.assertAlmostEqual(cfg.lateral_kp, 1.15)
        self.assertAlmostEqual(cfg.lateral_kd, 2.50)
        self.assertAlmostEqual(cfg.yaw_first_lateral_minimum_scale, 0.20)
        self.assertAlmostEqual(
            cfg.horizontal_capture_brake_lateral_gain, 1.50
        )
        self.assertAlmostEqual(cfg.horizontal_capture_brake_yaw_gain, 0.00)
        self.assertAlmostEqual(
            cfg.horizontal_yaw_capture_prediction_horizon_s, 8.0
        )
        self.assertAlmostEqual(
            cfg.horizontal_yaw_capture_max_error_normalized, 0.75
        )
        self.assertAlmostEqual(cfg.yaw_capture_lateral_scale, 0.55)
        self.assertAlmostEqual(
            cfg.yaw_first_lateral_full_normalized, 0.35
        )
        self.assertAlmostEqual(
            cfg.yaw_first_lateral_zero_normalized, 0.75
        )
        self.assertAlmostEqual(cfg.inward_capture_max_lateral_mps, 0.0)
        self.assertAlmostEqual(
            cfg.horizontal_capture_release_normalized,
            0.25,
        )
        self.assertAlmostEqual(
            cfg.horizontal_capture_prediction_horizon_s,
            8.0,
        )
        self.assertAlmostEqual(cfg.lateral_countersteer_gain, 3.00)
        self.assertAlmostEqual(cfg.countersteer_max_lateral_mps, 0.40)
        self.assertAlmostEqual(cfg.yaw_countersteer_gain, 2.25)
        self.assertAlmostEqual(cfg.countersteer_forward_floor_mps, 0.18)
        self.assertAlmostEqual(cfg.command_lpf_alpha, 0.68)
        self.assertAlmostEqual(cfg.max_lateral_acceleration, 3.2)
        self.assertAlmostEqual(cfg.max_right_mps, 0.40)
        self.assertAlmostEqual(cfg.max_up_mps, 0.32)
        self.assertAlmostEqual(cfg.max_down_mps, 0.40)
        self.assertAlmostEqual(cfg.recover_forward_mps, 0.14)
        self.assertAlmostEqual(cfg.recover_close_forward_mps, 0.12)
        self.assertAlmostEqual(cfg.recover_lateral_mps, 0.0)
        self.assertAlmostEqual(cfg.recover_prediction_scale, 0.65)
        self.assertAlmostEqual(cfg.max_yaw_rate_rps, 0.45)
        self.assertAlmostEqual(cfg.commit_maximum_duration_s, 1.20)
        self.assertAlmostEqual(cfg.recover_local_duration_s, 0.75)
        self.assertEqual(cfg.minimum_forward_mps, 0.0)
        self.assertEqual(cfg.close_approach_mps, 0.15)
        self.assertEqual(cfg.minimum_approach_mps, 0.18)
        self.assertEqual(cfg.track_forward_mps, 0.16)
        self.assertEqual(cfg.commit_forward_mps, 0.24)
        self.assertAlmostEqual(cfg.prepass_lookahead_weight, 0.55)
        self.assertAlmostEqual(cfg.secondary_contact_yaw_kp, 0.55)
        self.assertAlmostEqual(cfg.next_gate_maximum_primary_horizontal, 0.90)
        self.assertEqual(cfg.path_lateral_kp, 0.0)
        self.assertEqual(cfg.path_yaw_kp, 0.0)
        self.assertGreater(
            cfg.next_gate_max_right_mps,
            cfg.next_gate_max_yaw_rate_rps,
        )
        self.assertAlmostEqual(cfg.commit_opening_area_ratio, 0.030)
        self.assertAlmostEqual(cfg.commit_alignment_tolerance, 0.10)
        self.assertAlmostEqual(cfg.commit_horizontal_tolerance, 0.08)
        self.assertEqual(cfg.commit_stable_frames, 3)
        self.assertTrue(cfg.commit_straight_through)
        self.assertAlmostEqual(cfg.minimum_detection_confidence, 0.18)
        self.assertAlmostEqual(cfg.reliable_confidence, 0.30)
        self.assertAlmostEqual(cfg.commit_minimum_confidence, 0.32)
        self.assertAlmostEqual(tracker_cfg.minimum_seed_area_ratio, 0.0032)
        self.assertAlmostEqual(
            tracker_cfg.trusted_minimum_seed_confidence, 0.30
        )
        self.assertAlmostEqual(
            tracker_cfg.trusted_minimum_seed_area_ratio, 0.0012
        )
        self.assertEqual(config.YOLO_MIN_GATE_AREA_PX, 100.0)
        self.assertEqual(config.GATE_MIN_CONTOUR_AREA, 12.0)
        self.assertEqual(config.GATE_HSV_LOWER, (0, 75, 140))
        self.assertEqual(config.GATE_HSV_UPPER, (23, 255, 255))
        self.assertAlmostEqual(config.KP_LEAN, 0.10)
        self.assertAlmostEqual(config.OPENCV_KP_LEAN, 0.16)
        self.assertAlmostEqual(config.MAX_ASCENT_THRUST_INCREASE, 0.030)
        self.assertAlmostEqual(config.HOVER_THRUST, 0.24)
        self.assertAlmostEqual(config.KP_ROLL_ATT, 2.6)
        self.assertAlmostEqual(config.KD_ROLL_ATT, 0.22)
        self.assertAlmostEqual(config.YOLO_CONFIDENCE_THRESHOLD, 0.45)
        self.assertEqual(config.YOLO_ACQUISITION_CONFIRMATION_FRAMES, 1)
        self.assertFalse(config.YOLO_REQUIRE_HSV_CONFIRMATION)
        self.assertFalse(config.GLOBAL_HSV_FALLBACK_ENABLED)
        self.assertEqual(config.GATE_DETECTOR_BACKEND, 'yolo_pose')
        self.assertAlmostEqual(config.YOLO_HSV_CENTER_BLEND, 0.0)
        self.assertFalse(config.RESET_SIM_ON_START)
        self.assertEqual(config.OPENCV_MAX_SECONDS, 0.0)

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
            # Current gate-one telemetry places the opening near optical
            # center at commit, inside the camera-tilt-adjusted flight line.
            ny=0.02,
            opening_width=100,
            opening_height=82,
            stable_frames=3,
            size_rate=12.0,
        )
        command = navigator.update(close, 1.3)
        self.assertEqual(command.state.value, 'COMMIT')
        after_dropout = navigator.update(None, 1.4)
        self.assertEqual(after_dropout.state.value, 'PASS_THROUGH')

    def test_visible_centered_gate_does_not_time_out_commit_early(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        gate = detection_at(
            ny=0.24,
            opening_width=100,
            opening_height=82,
            stable_frames=5,
        )
        navigator.update(gate, 1.0)
        navigator.update(gate, 1.1)
        committed = navigator.update(gate, 1.25)
        self.assertEqual(committed.state, NavigationState.COMMIT)

        still_committed = navigator.update(gate, 2.0)
        self.assertEqual(still_committed.state, NavigationState.COMMIT)

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

    def test_commit_aborts_and_recenters_if_gate_drifts_sideways(self):
        navigator = GateNavigator(
            replace(
                q2_demo_navigation_config(),
                commit_straight_through=False,
            )
        )
        navigator.state = NavigationState.COMMIT
        navigator._state_since = 1.0
        navigator._last_alignment_command[:] = 0.0
        drifted = detection_at(
            nx=0.12,
            ny=0.24,
            opening_width=110,
            opening_height=90,
            stable_frames=5,
        )

        command = navigator.update(drifted, 1.2)

        self.assertEqual(
            command.state, NavigationState.ALIGN_AND_APPROACH
        )
        self.assertGreater(command.right_mps, 0.0)

    def test_q2_commit_stays_latched_and_flies_straight(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        navigator.state = NavigationState.COMMIT
        navigator._state_since = 1.0
        navigator._last_alignment_command[:] = (-0.4, 0.2, -0.1)
        close_drift = detection_at(
            nx=-0.11,
            ny=0.30,
            opening_width=260,
            opening_height=250,
            stable_frames=5,
        )

        command = navigator.update(close_drift, 1.2)

        self.assertEqual(command.state, NavigationState.COMMIT)
        self.assertGreater(command.forward_mps, 0.0)
        self.assertAlmostEqual(command.right_mps, 0.0)
        self.assertAlmostEqual(command.down_mps, 0.0)
        self.assertAlmostEqual(command.yaw_rate_rps, 0.0)

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

    def test_q2_tracker_accepts_trusted_green_gate_near_frame_edge(self):
        tracker = GateTracker(q2_demo_tracker_config())
        trusted_green_gate = replace(
            detection_at(
                nx=0.82,
                ny=0.75,
                confidence=0.34,
                opening_width=20,
                opening_height=18,
            ),
            method="yolo_pose_hsv_box_center_no_orientation",
        )

        self.assertIsNotNone(
            tracker.update(trusted_green_gate, timestamp=1.0)
        )

    def test_q2_tracker_rejects_nested_gate_size_hop(self):
        tracker = GateTracker(q2_demo_tracker_config())
        active = detection_at(
            opening_width=150,
            opening_height=140,
            confidence=0.9,
        )
        tracker.update(active, timestamp=1.0)
        nested_next = detection_at(
            opening_width=105,
            opening_height=98,
            confidence=0.95,
        )

        result = tracker.update(nested_next, timestamp=1.1)

        self.assertTrue(result.predicted)
        self.assertGreater(result.opening_width, 140)

    def test_q2_tracker_accepts_recorded_monotonic_gate_growth(self):
        tracker = GateTracker(q2_demo_tracker_config())
        recorded_sizes = (
            (18.371, 35.374),
            (20.679, 38.446),
            (22.816, 42.394),
            (24.899, 47.081),
        )

        results = []
        for index, (width, height) in enumerate(recorded_sizes):
            measurement = replace(
                detection_at(
                    nx=0.17 + 0.02 * index,
                    ny=0.37 + 0.01 * index,
                    confidence=0.75,
                    opening_width=width,
                    opening_height=height,
                ),
                method="yolo_pose_hsv_refined_center",
            )
            results.append(
                tracker.update(
                    measurement,
                    timestamp=1.0 + 0.1 * index,
                )
            )

        self.assertTrue(all(result is not None for result in results))
        self.assertTrue(all(not result.predicted for result in results))

    def test_q2_tracker_accepts_rapid_growth_of_locked_pose_gate(self):
        tracker = GateTracker(q2_demo_tracker_config())
        first = replace(
            detection_at(
                nx=0.68,
                ny=-0.19,
                confidence=0.75,
                opening_width=76,
                opening_height=70,
            ),
            method="yolo_pose_hsv_refined_center",
        )
        tracker.update(first, timestamp=1.0)
        closer = replace(
            detection_at(
                nx=0.72,
                ny=0.01,
                confidence=0.80,
                opening_width=110,
                opening_height=118,
            ),
            method="yolo_pose_hsv_refined_center",
        )

        result = tracker.update(closer, timestamp=1.1)

        self.assertIsNotNone(result)
        self.assertFalse(result.predicted)
        self.assertGreater(result.opening_width, first.opening_width)

    def test_no_gate_stops_forward_flight_and_scans(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        command = navigator.update(None, 1.0)
        self.assertAlmostEqual(command.forward_mps, 0.0)
        self.assertAlmostEqual(command.yaw_rate_rps, 0.0)

    def test_tracker_prediction_cannot_move_unconfirmed_gate_target(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        measured = detection_at(
            nx=0.30,
            confidence=0.90,
            stable_frames=1,
        )
        navigator.update(measured, 1.0)
        predicted = detection_at(
            nx=0.40,
            confidence=0.70,
            predicted=True,
            stable_frames=1,
        )

        command = navigator.update(predicted, 1.1)

        self.assertEqual(command.state, NavigationState.TRACK)
        self.assertAlmostEqual(command.forward_mps, 0.0)
        self.assertAlmostEqual(command.right_mps, 0.0)
        self.assertAlmostEqual(command.down_mps, 0.0)
        self.assertAlmostEqual(command.yaw_rate_rps, 0.0)

    def test_tracker_prediction_brakes_existing_image_motion(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        measured = detection_at(
            nx=0.40,
            ny=-0.20,
            confidence=0.90,
            stable_frames=1,
        )
        navigator.update(measured, 1.0)
        moving_prediction = detection_at(
            nx=0.30,
            ny=-0.20,
            confidence=0.70,
            predicted=True,
            stable_frames=1,
            velocity_x=-0.15,
            velocity_y=0.18,
        )

        command = navigator.update(moving_prediction, 1.1)

        self.assertEqual(command.state, NavigationState.TRACK)
        self.assertAlmostEqual(command.forward_mps, 0.0)
        self.assertLess(command.right_mps, 0.0)
        self.assertGreater(command.down_mps, 0.0)
        self.assertLess(command.yaw_rate_rps, 0.0)

    def test_search_keeps_forward_crawl_for_close_prediction_only(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        close_prediction = detection_at(
            opening_width=175,
            opening_height=171,
            predicted=True,
        )

        command = navigator.update(close_prediction, 1.0)

        self.assertEqual(command.state, NavigationState.SEARCH)
        self.assertGreater(command.forward_mps, 0.0)
        self.assertLessEqual(
            command.forward_mps,
            navigator.config.search_close_forward_mps,
        )
        self.assertAlmostEqual(command.right_mps, 0.0)
        self.assertAlmostEqual(command.yaw_rate_rps, 0.0)

    def test_close_gate_slows_but_keeps_forward_authority(self):
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
        self.assertEqual(very_close_speed, 0.15)

    def test_gate_capture_coordinates_yaw_and_bank_near_edge(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        right_gate = detection_at(
            nx=0.35,
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
        self.assertLessEqual(
            command.right_mps,
            navigator.config.max_right_mps,
        )
        self.assertGreater(command.yaw_rate_rps, 0.0)
        self.assertLessEqual(
            abs(command.right_mps),
            navigator.config.max_right_mps,
        )

    def test_extreme_edge_gate_uses_yaw_before_lateral_bank(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        navigator.state = NavigationState.ALIGN_AND_APPROACH
        navigator._state_since = 1.0
        navigator._last_update = 1.0
        high_right_edge_gate = detection_at(
            nx=0.87,
            ny=-0.25,
            opening_width=50,
            opening_height=40,
            stable_frames=5,
        )

        command = navigator.update(high_right_edge_gate, 1.1)

        self.assertGreater(command.right_mps, 0.0)
        self.assertLessEqual(command.right_mps, 0.12)
        self.assertLessEqual(
            command.right_mps,
            navigator.config.max_right_mps,
        )
        self.assertGreater(command.yaw_rate_rps, 0.0)

    def test_lateral_velocity_brakes_before_gate_crosses_center(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        navigator.state = NavigationState.ALIGN_AND_APPROACH
        navigator._state_since = 1.0
        navigator._last_update = 1.0
        approaching_center = detection_at(
            nx=0.15,
            ny=0.24,
            opening_width=50,
            opening_height=40,
            stable_frames=5,
            velocity_x=-0.15,
        )

        command = navigator.update(approaching_center, 1.1)

        # The opening is still right of center, but its leftward image
        # velocity predicts an overshoot. Counter-steer before it crosses.
        self.assertGreater(approaching_center.normalized_x, 0.0)
        self.assertLess(command.right_mps, 0.0)
        self.assertLess(command.yaw_rate_rps, 0.0)

    def test_static_off_center_gate_still_commands_toward_opening(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        navigator.state = NavigationState.ALIGN_AND_APPROACH
        navigator._state_since = 1.0
        navigator._last_update = 1.0
        right_gate = detection_at(
            nx=0.15,
            ny=0.24,
            opening_width=50,
            opening_height=40,
            stable_frames=5,
            velocity_x=0.0,
        )

        command = navigator.update(right_gate, 1.1)

        self.assertGreater(command.right_mps, 0.0)
        self.assertGreater(command.yaw_rate_rps, 0.0)

    def test_inward_gate_motion_limits_further_lateral_acceleration(self):
        cfg = q2_demo_navigation_config()
        navigator = GateNavigator(cfg)
        navigator.state = NavigationState.ALIGN_AND_APPROACH
        navigator._state_since = 1.0
        navigator._last_update = 1.0
        navigator._last_command[1] = 0.55
        gate_moving_inward = detection_at(
            nx=0.50,
            ny=0.24,
            opening_width=50,
            opening_height=40,
            stable_frames=5,
            velocity_x=-0.11,
        )

        command = navigator.update(gate_moving_inward, 1.1)

        self.assertGreater(gate_moving_inward.normalized_x, 0.0)
        self.assertLess(gate_moving_inward.velocity_x, 0.0)
        self.assertAlmostEqual(command.right_mps, -0.165)
        self.assertAlmostEqual(
            navigator._last_command[1],
            command.right_mps,
        )

    def test_inward_gate_motion_limits_further_yaw_acceleration(self):
        cfg = q2_demo_navigation_config()
        navigator = GateNavigator(cfg)
        navigator.state = NavigationState.ALIGN_AND_APPROACH
        navigator._state_since = 1.0
        navigator._last_update = 1.0
        navigator._last_command[3] = 0.40
        gate_moving_inward = detection_at(
            nx=0.50,
            ny=0.24,
            opening_width=50,
            opening_height=40,
            stable_frames=5,
            velocity_x=-0.11,
        )

        command = navigator.update(gate_moving_inward, 1.1)

        self.assertAlmostEqual(command.yaw_rate_rps, 0.0)
        self.assertAlmostEqual(
            navigator._last_command[3],
            command.yaw_rate_rps,
        )

    def test_far_off_axis_gate_holds_hover_before_vertical_centering(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        navigator._confirmed_gate_passes = 1

        _, far_vertical, _ = navigator._errors(
            detection_at(nx=0.70, ny=0.30)
        )
        _, aligned_vertical, _ = navigator._errors(
            detection_at(nx=0.30, ny=0.30)
        )

        self.assertAlmostEqual(far_vertical, 0.0)
        self.assertGreater(aligned_vertical, 0.0)

    def test_capture_hysteresis_ignores_one_outward_velocity_sample(self):
        cfg = q2_demo_navigation_config()
        navigator = GateNavigator(cfg)
        navigator.state = NavigationState.ALIGN_AND_APPROACH
        navigator._state_since = 1.0
        navigator._last_update = 1.0
        inward = detection_at(
            nx=0.50,
            ny=0.24,
            stable_frames=5,
            velocity_x=-0.11,
        )
        navigator.update(inward, 1.1)
        noisy_outward = detection_at(
            nx=0.64,
            ny=0.24,
            stable_frames=5,
            velocity_x=0.05,
        )

        command = navigator.update(noisy_outward, 1.2)

        self.assertLessEqual(
            command.yaw_rate_rps,
            cfg.inward_capture_max_yaw_rate_rps,
        )
        self.assertEqual(navigator._horizontal_capture_side, 1.0)

    def test_capture_hysteresis_releases_in_center_corridor(self):
        cfg = q2_demo_navigation_config()
        navigator = GateNavigator(cfg)
        navigator.state = NavigationState.ALIGN_AND_APPROACH
        navigator._state_since = 1.0
        navigator._last_update = 1.0
        navigator.update(
            detection_at(
                nx=0.50,
                ny=0.24,
                stable_frames=5,
                velocity_x=-0.11,
            ),
            1.1,
        )

        navigator.update(
            detection_at(
                nx=0.20,
                ny=0.24,
                stable_frames=5,
                velocity_x=0.0,
            ),
            1.2,
        )

        self.assertEqual(navigator._horizontal_capture_side, 0.0)

    def test_vertical_image_motion_cannot_reverse_descent_into_climb(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        navigator.state = NavigationState.ALIGN_AND_APPROACH
        navigator._state_since = 1.0
        navigator._last_update = 1.0
        low_gate_moving_up = detection_at(
            nx=0.0,
            ny=0.50,
            opening_width=50,
            opening_height=40,
            stable_frames=5,
            velocity_y=-0.15,
        )

        command = navigator.update(low_gate_moving_up, 1.1)

        self.assertGreater(
            low_gate_moving_up.normalized_y,
            navigator.config.vertical_setpoint_normalized,
        )
        self.assertGreaterEqual(command.down_mps, 0.0)

    def test_recorded_gate_two_motion_countersteers_while_still_right(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        navigator.state = NavigationState.ALIGN_AND_APPROACH
        navigator._state_since = 1.0
        navigator._last_update = 1.0
        recorded_motion = detection_at(
            nx=0.086,
            ny=0.24,
            opening_width=50,
            opening_height=44,
            stable_frames=5,
            velocity_x=-0.079,
        )

        command = navigator.update(recorded_motion, 1.1)

        self.assertGreater(recorded_motion.normalized_x, 0.0)
        self.assertLess(command.right_mps, 0.0)
        self.assertLess(command.yaw_rate_rps, 0.0)

    def test_predicted_recovery_brakes_inward_gate_motion(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        navigator.state = NavigationState.RECOVER
        navigator._state_since = 1.0
        navigator._last_seen_at = 1.0
        navigator._last_update = 1.0
        recorded_prediction = detection_at(
            nx=0.58,
            ny=0.82,
            confidence=0.48,
            opening_width=45,
            opening_height=45,
            predicted=True,
            velocity_x=-0.128,
        )

        command = navigator.update(recorded_prediction, 1.1)

        # This reproduces gate two at 19.65-20.20 s: the predicted gate was
        # still right of centre but moving inward quickly. Continuing to bank
        # right carried the vehicle through centre before roll could reverse.
        self.assertEqual(command.state, NavigationState.RECOVER)
        self.assertLess(command.right_mps, 0.0)
        self.assertLess(command.yaw_rate_rps, 0.0)

    def test_countersteer_boost_adds_authority_without_moving_trigger(self):
        boosted_config = q2_demo_navigation_config()
        baseline_config = replace(
            boosted_config,
            lateral_countersteer_gain=1.0,
            countersteer_max_lateral_mps=math.inf,
            yaw_countersteer_gain=1.0,
        )
        boosted = GateNavigator(boosted_config)
        baseline = GateNavigator(baseline_config)
        for navigator in (boosted, baseline):
            navigator.state = NavigationState.ALIGN_AND_APPROACH
            navigator._state_since = 1.0
            navigator._last_update = 1.0
        approaching_center = detection_at(
            nx=0.15,
            ny=0.24,
            opening_width=50,
            opening_height=40,
            stable_frames=5,
            velocity_x=-0.15,
        )

        boosted_command = boosted.update(approaching_center, 1.2)
        baseline_command = baseline.update(approaching_center, 1.2)

        self.assertLess(boosted_command.right_mps, 0.0)
        self.assertLess(boosted_command.yaw_rate_rps, 0.0)
        self.assertGreater(
            abs(boosted_command.right_mps),
            abs(baseline_command.right_mps),
        )
        self.assertLessEqual(
            abs(boosted_command.right_mps),
            boosted_config.countersteer_max_lateral_mps,
        )
        self.assertGreater(
            abs(boosted_command.yaw_rate_rps),
            abs(baseline_command.yaw_rate_rps),
        )

    def test_countersteering_keeps_forward_approach_active(self):
        forward_config = q2_demo_navigation_config()
        stopping_config = replace(
            forward_config,
            countersteer_forward_floor_mps=0.0,
        )
        forward = GateNavigator(forward_config)
        stopping = GateNavigator(stopping_config)
        for navigator in (forward, stopping):
            navigator.state = NavigationState.ALIGN_AND_APPROACH
            navigator._state_since = 1.0
            navigator._last_update = 1.0
        close_gate = detection_at(
            nx=0.15,
            ny=0.24,
            opening_width=100,
            opening_height=90,
            stable_frames=5,
            velocity_x=-0.25,
        )

        for index in range(1, 7):
            now = 1.0 + 0.2 * index
            forward_command = forward.update(close_gate, now)
            stopping_command = stopping.update(close_gate, now)

        self.assertLess(forward_command.right_mps, 0.0)
        self.assertGreater(
            forward_command.forward_mps,
            stopping_command.forward_mps,
        )
        self.assertGreaterEqual(forward_command.forward_mps, 0.17)

    def test_confirmed_pass_releases_old_gate_immediately(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        navigator.state = NavigationState.COMMIT
        navigator.confirm_gate_pass(2.0)
        self.assertEqual(navigator.state, NavigationState.SEARCH)
        self.assertEqual(navigator._state_since, 2.0)

    def test_pass_through_drops_old_left_correction_then_turns_right(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        navigator.state = NavigationState.ALIGN_AND_APPROACH
        navigator._state_since = 1.0
        navigator._last_alignment_command[:] = (-2.0, 0.0, -0.3)
        primary = detection_at(
            nx=0.0,
            ny=0.24,
            opening_width=100,
            opening_height=80,
            stable_frames=5,
        )

        navigator.update(primary, 1.1, next_gate_horizontal=0.60)
        navigator.state = NavigationState.COMMIT
        navigator._state_since = 1.1
        clearance = navigator.update(None, 1.2)
        still_clearing = navigator.update(None, 1.75)
        toward_next = navigator.update(None, 1.95)

        self.assertEqual(clearance.state, NavigationState.PASS_THROUGH)
        self.assertGreaterEqual(clearance.right_mps, 0.0)
        self.assertAlmostEqual(still_clearing.right_mps, 0.0)
        self.assertGreater(toward_next.right_mps, 0.0)
        self.assertGreater(toward_next.yaw_rate_rps, 0.0)

    def test_confirmed_pass_scans_toward_latched_right_gate(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        navigator.state = NavigationState.ALIGN_AND_APPROACH
        navigator._state_since = 1.0
        primary = detection_at(
            nx=0.0,
            ny=0.24,
            opening_width=100,
            opening_height=80,
            stable_frames=5,
        )
        navigator.update(primary, 1.1, next_gate_horizontal=0.60)
        navigator._last_direction = -1.0

        navigator.confirm_gate_pass(1.2)
        command = navigator.update(None, 1.3)

        self.assertEqual(command.state, NavigationState.SEARCH)
        # Multi-gate / latched next-gate bearing must rotate SEARCH toward the
        # remembered right-hand gate. Open-loop post-pass yaw is sign-flipped
        # on VQ2 relative to image IBVS (run 022119).
        self.assertLess(command.yaw_rate_rps, 0.0)

    def test_recovery_banks_toward_last_gate_without_reversing(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        navigator.state = NavigationState.ALIGN_AND_APPROACH
        navigator._state_since = 1.0
        navigator._last_update = 1.0
        navigator._last_seen_at = 1.0
        navigator._last_direction = 1.0
        navigator._last_command[:] = (-0.12, 0.0, 0.0, 0.0)

        command = navigator.update(None, 1.1)

        self.assertEqual(command.state, NavigationState.RECOVER)
        # A previously commanded reverse is neutralized on the first recovery
        # update; subsequent updates accelerate toward recover_forward_mps.
        self.assertEqual(command.forward_mps, 0.0)
        self.assertAlmostEqual(command.right_mps, 0.0)

    def test_recovery_uses_predicted_motion_not_stale_acquisition_side(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        navigator.state = NavigationState.RECOVER
        navigator._state_since = 1.0
        navigator._last_update = 1.0
        navigator._last_seen_at = 1.0
        navigator._last_direction = 1.0
        predicted = detection_at(
            nx=0.01,
            ny=0.20,
            opening_width=60,
            predicted=True,
            velocity_x=-0.14,
            velocity_y=-0.05,
        )

        command = navigator.update(predicted, 1.1)

        self.assertEqual(command.state, NavigationState.RECOVER)
        self.assertGreater(command.forward_mps, 0.0)
        self.assertLess(command.right_mps, 0.0)
        self.assertLess(command.yaw_rate_rps, 0.0)

    def test_close_uncommitted_prediction_brakes_forward(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        navigator.state = NavigationState.RECOVER
        navigator._state_since = 1.0
        navigator._last_update = 1.0
        navigator._last_seen_at = 1.0
        predicted = detection_at(
            nx=0.16,
            ny=0.14,
            opening_width=175,
            opening_height=171,
            predicted=True,
            velocity_x=-0.05,
        )

        command = navigator.update(predicted, 1.1)

        self.assertEqual(command.state, NavigationState.RECOVER)
        self.assertGreater(command.forward_mps, 0.0)
        self.assertLessEqual(
            command.forward_mps,
            navigator.config.recover_close_forward_mps,
        )
        self.assertNotEqual(command.right_mps, 0.0)

    def test_tracker_predictions_do_not_extend_recovery_timeout(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        navigator.state = NavigationState.RECOVER
        navigator._state_since = 1.0
        navigator._last_update = 1.0
        navigator._last_seen_at = 1.0
        predicted = detection_at(
            nx=-0.20,
            ny=0.24,
            predicted=True,
            velocity_x=-0.10,
        )

        navigator.update(predicted, 1.1)
        command = navigator.update(predicted, 1.8)

        self.assertEqual(command.state, NavigationState.SEARCH)
        self.assertAlmostEqual(navigator._last_seen_at, 1.0)
        self.assertAlmostEqual(command.right_mps, 0.0)
        self.assertAlmostEqual(command.down_mps, 0.0)
        self.assertAlmostEqual(command.yaw_rate_rps, 0.0)

    def test_vertical_servo_keeps_distant_gate_in_frame(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        far = detection_at(ny=0.70, opening_width=35)
        _, far_vertical, _ = navigator._errors(far)
        self.assertAlmostEqual(far_vertical, 0.44)

        close = detection_at(ny=0.60, opening_width=80)
        _, close_vertical, _ = navigator._errors(close)
        # (0.60 - 0.24 target) - 0.02 descent deadband.
        self.assertAlmostEqual(close_vertical, 0.34)

    def test_vertical_target_holds_high_approach_after_confirmed_pass(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        high_approach = detection_at(
            ny=-0.25,
            opening_width=55,
            stable_frames=5,
        )
        _, before_pass, _ = navigator._errors(high_approach)

        navigator.confirm_gate_pass(1.0)
        _, after_pass, _ = navigator._errors(high_approach)

        self.assertLess(before_pass, 0.0)
        self.assertAlmostEqual(after_pass, 0.0)
        self.assertEqual(navigator._confirmed_gate_passes, 1)

    def test_vertical_velocity_brakes_before_gate_crosses_setpoint(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        high_gate = detection_at(
            ny=-0.05,
            velocity_y=0.40,
            opening_width=55,
            stable_frames=5,
        )
        _, vertical, _ = navigator._errors(high_gate)

        command = navigator._vertical_control(
            high_gate,
            vertical,
            0.20,
        )

        self.assertLess(vertical, 0.0)
        self.assertGreater(command, 0.0)
        self.assertLessEqual(
            command,
            navigator.config.vertical_countersteer_max_mps,
        )

    def test_recorded_gate_two_vertical_motion_brakes_at_y141(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        recorded = detection_at(
            ny=2.0 * 141.0 / 360.0 - 1.0,
            velocity_y=0.094,
            opening_width=55,
            stable_frames=5,
        )
        _, vertical, _ = navigator._errors(recorded)

        command = navigator._vertical_control(recorded, vertical, 0.20)

        self.assertLess(vertical, 0.0)
        self.assertGreater(command, 0.0)
        self.assertLessEqual(
            command,
            navigator.config.vertical_countersteer_max_mps,
        )

    def test_vertical_reversal_is_blocked_when_gate_moves_outward(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        high_gate = detection_at(
            ny=-0.05,
            velocity_y=-0.40,
            opening_width=55,
            stable_frames=5,
        )
        _, vertical, _ = navigator._errors(high_gate)
        # Prime a derivative reversal, then prove outward motion may not use
        # the countersteer allowance.
        navigator._vertical_pid.update(
            vertical,
            0.20,
            measurement_rate=0.40,
        )

        command = navigator._vertical_control(
            high_gate,
            vertical,
            0.20,
        )

        self.assertLess(vertical, 0.0)
        self.assertLessEqual(command, 0.0)

    def test_gate_near_frame_edge_retains_coordinated_turn_progress(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        edge_gate = detection_at(
            nx=0.90,
            opening_width=50,
            opening_height=40,
            stable_frames=5,
        )
        command = navigator.update(edge_gate, 1.0)
        self.assertGreater(command.forward_mps, 0.0)
        self.assertLess(
            command.forward_mps,
            command.requested_forward_mps,
        )
        self.assertGreater(command.right_mps, 0.0)
        self.assertLessEqual(
            command.right_mps,
            navigator.config.max_right_mps,
        )
        self.assertGreater(command.yaw_rate_rps, 0.0)
        self.assertTrue(command.framing_limited)

    def test_gate_at_frame_boundary_stops_forward_approach(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        edge_gate = detection_at(
            nx=1.0,
            opening_width=50,
            opening_height=40,
            stable_frames=5,
        )

        command = navigator.update(edge_gate, 1.0)

        self.assertEqual(command.forward_mps, 0.0)
        self.assertGreater(command.right_mps, 0.0)
        self.assertLess(command.right_mps, 0.24)
        self.assertGreater(command.yaw_rate_rps, 0.0)
        self.assertTrue(command.framing_limited)

    def test_low_gate_with_corrective_descent_keeps_forward_approach(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        low_gate = detection_at(
            nx=0.30,
            ny=0.87,
            opening_width=35,
            opening_height=32,
            stable_frames=5,
        )

        navigator.update(low_gate, 1.0)
        command = navigator.update(low_gate, 1.25)

        self.assertEqual(command.state, NavigationState.ALIGN_AND_APPROACH)
        self.assertGreater(command.down_mps, 0.0)
        self.assertGreater(command.forward_mps, 0.10)
        self.assertFalse(command.framing_limited)
        self.assertGreaterEqual(
            command.requested_forward_mps,
            navigator.config.minimum_approach_mps,
        )

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
        # The calibrated OpenCV lateral mapping reverses desired roll before
        # the simulator's outgoing rate-axis inversion.
        self.assertAlmostEqual(
            output['roll_rate'],
            config.MAX_RATE_RAD_S,
        )
        self.assertAlmostEqual(output['yaw_rate'], -config.MAX_RATE_RAD_S)
        for key in ('roll_rate', 'pitch_rate', 'yaw_rate'):
            self.assertLessEqual(
                abs(output[key]),
                config.MAX_RATE_RAD_S,
            )
        self.assertEqual(len(connection.mav.attitude_targets), 1)

    def test_opencv_racing_uses_stronger_forward_lean_mapping(self):
        connection = _FakeConnection()
        data = {
            'attitude': {'yaw': 0.0},
            'planner_mode': 'opencv_align_and_approach',
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
                'vn': 0.34,
                've': 0.30,
                'vd': 0.0,
                'yaw_rate': 0.20,
            },
        }
        controller = Controller(connection, data, system_boot_ms=0)

        with patch('controller.time.sleep'):
            controller.update()

        self.assertAlmostEqual(
            data['control_output']['desired_pitch'],
            0.34 * config.OPENCV_KP_LEAN,
        )
        self.assertAlmostEqual(
            data['control_output']['desired_roll'],
            0.30 * 0.24 * config.OPENCV_LATERAL_LEAN_SIGN,
        )
        self.assertLess(
            data['control_output']['desired_roll'],
            0.0,
        )
        self.assertAlmostEqual(
            data['control_output']['yaw_rate'],
            0.20
            + config.OPENCV_YAW_RATE_FEEDBACK_KP * 0.20,
        )

    def test_opencv_yaw_feedback_brakes_carried_turn_rate(self):
        connection = _FakeConnection()
        data = {
            'attitude': {'yaw': 0.0},
            'planner_mode': 'opencv_align_and_approach',
            'highres_imu': {
                'xgyro': 0.0,
                'ygyro': 0.0,
                # Positive/right physical yaw is negative on the raw body-z
                # gyro in VQ2.
                'zgyro': -0.70,
                'xacc': 0.0,
                'yacc': 0.0,
                'zacc': -9.81,
                'ts_us': 1_000_000,
                'ts': time.time_ns(),
            },
            'planner_target': {
                'vn': 0.34,
                've': 0.0,
                'vd': 0.0,
                'yaw_rate': 0.12,
            },
        }
        controller = Controller(connection, data, system_boot_ms=0)

        with patch('controller.time.sleep'):
            controller.update()

        output = data['control_output']
        self.assertAlmostEqual(output['requested_yaw_rate'], 0.12)
        self.assertAlmostEqual(output['measured_yaw_rate'], 0.70)
        self.assertLess(output['yaw_rate_feedback'], 0.0)
        self.assertAlmostEqual(output['yaw_rate_feedback'], -0.12)
        self.assertAlmostEqual(output['yaw_rate'], 0.0)

    def test_opencv_zero_yaw_request_uses_soft_nonreversing_brake(self):
        connection = _FakeConnection()
        data = {
            'attitude': {'yaw': 0.0},
            'planner_mode': 'opencv_align_and_approach',
            'highres_imu': {
                'xgyro': 0.0,
                'ygyro': 0.0,
                'zgyro': -0.70,
                'xacc': 0.0,
                'yacc': 0.0,
                'zacc': -9.81,
                'ts_us': 1_000_000,
                'ts': time.time_ns(),
            },
            'planner_target': {
                'vn': 0.34,
                've': 0.0,
                'vd': 0.0,
                'yaw_rate': 0.0,
            },
        }
        controller = Controller(connection, data, system_boot_ms=0)

        with patch('controller.time.sleep'):
            controller.update()

        output = data['control_output']
        self.assertAlmostEqual(output['requested_yaw_rate'], 0.0)
        self.assertAlmostEqual(output['measured_yaw_rate'], 0.70)
        self.assertAlmostEqual(output['yaw_rate_feedback'], 0.0)
        self.assertAlmostEqual(output['yaw_rate'], 0.0)

    def test_opencv_attitude_estimator_retains_real_gyro_bank(self):
        connection = _FakeConnection()
        data = {
            'attitude': {'yaw': 0.0},
            'planner_mode': 'opencv_align_and_approach',
            'planner_target': {
                'vn': 0.0,
                've': 0.0,
                'vd': 0.0,
                'yaw_rate': 0.0,
            },
        }
        controller = Controller(connection, data, system_boot_ms=0)
        imu_ts_us = 1_000_000
        with patch('controller.time.sleep'):
            for _ in range(30):
                imu_ts_us += 10_000
                data['highres_imu'] = {
                    'xgyro': 0.5,
                    'ygyro': 0.0,
                    'zgyro': 0.0,
                    # VQ2 reports a gravity-like level acceleration even while
                    # the camera and gyro show a sustained roll maneuver.
                    'xacc': 0.0,
                    'yacc': 0.0,
                    'zacc': -9.81,
                    'ts_us': imu_ts_us,
                    'ts': time.time_ns(),
                }
                controller.update()

        self.assertAlmostEqual(config.OPENCV_AHRS_GYRO_WEIGHT, 0.995)
        self.assertGreater(data['control_output']['ahrs_roll'], 0.12)
        self.assertGreater(
            data['control_output']['ahrs_roll'],
            1.5 * controller._ahrs.roll,
        )

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

    def test_controller_compensates_hover_thrust_while_banked(self):
        connection = _FakeConnection()
        data = {
            'attitude': {'yaw': 0.0},
            'planner_target': {
                'vn': 0.0,
                've': 0.0,
                'vd': 0.0,
                'yaw_rate': 0.0,
            },
        }
        controller = Controller(connection, data, system_boot_ms=0)
        bank = math.radians(20.0)
        with (
            patch.object(
                controller,
                '_demo_attitude',
                return_value=(bank, 0.0, 0.0, 0.0, 0.0, True),
            ),
            patch('controller.time.sleep'),
        ):
            controller.update()
        self.assertAlmostEqual(
            data['control_output']['thrust'],
            config.HOVER_THRUST / math.cos(bank),
        )

    def test_visual_descent_cannot_remove_hover_lift(self):
        connection = _FakeConnection()
        data = {
            'attitude': {'yaw': 0.0},
            'planner_target': {
                'vn': 0.0,
                've': 0.0,
                'vd': 0.4,
                'yaw_rate': 0.0,
            },
        }
        controller = Controller(connection, data, system_boot_ms=0)
        with (
            patch.object(
                controller,
                '_demo_attitude',
                return_value=(0.0, 0.0, 0.0, 0.0, 0.0, True),
            ),
            patch('controller.time.sleep'),
        ):
            controller.update()
        self.assertAlmostEqual(
            data['control_output']['thrust'],
            config.HOVER_THRUST - config.MAX_DESCENT_THRUST_REDUCTION,
        )

    def test_visual_climb_cannot_run_away_above_hover(self):
        connection = _FakeConnection()
        data = {
            'attitude': {'yaw': 0.0},
            'planner_target': {
                'vn': 0.0,
                've': 0.0,
                'vd': -0.6,
                'yaw_rate': 0.0,
            },
        }
        controller = Controller(connection, data, system_boot_ms=0)
        with (
            patch.object(
                controller,
                '_demo_attitude',
                return_value=(0.0, 0.0, 0.0, 0.0, 0.0, True),
            ),
            patch('controller.time.sleep'),
        ):
            controller.update()
        self.assertAlmostEqual(
            data['control_output']['thrust'],
            config.HOVER_THRUST + config.MAX_ASCENT_THRUST_INCREASE,
        )

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
