"""Deterministic tests for gate opening detection, tracking, and navigation."""

from __future__ import annotations

import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from gate_estimator import build_gate_object_points, estimate_gate
from opencv_gate_planner import OpenCVGatePlanner
from vision_rx import VisionRX, course_lookahead_horizontal
from vision.gate_detector import (
    CandidateDebug,
    DetectorDebug,
    GateDetection,
    GateVisionConfig,
    OrangeGateDetector,
    normalized_image_coordinates,
    order_corners,
    quadrilateral_diagonal_center,
    draw_detection,
)
from vision.gate_tracker import GateTracker, GateTrackerConfig
from vision.mode_router import (
    GateNavigationMode,
    ModeRouterConfig,
    VisionModeRouter,
)
from vision.navigation import GateNavigator, NavigationConfig, NavigationState


ORANGE = (0, 105, 255)
DIM_ORANGE = (0, 75, 180)
BACKGROUND = (42, 42, 42)
FRAME_SIZE = (360, 640)


def blank_frame() -> np.ndarray:
    return np.full((*FRAME_SIZE, 3), BACKGROUND, dtype=np.uint8)


def square_points(center, size, angle=0.0) -> np.ndarray:
    return cv2.boxPoints((center, (size, size), angle)).astype(np.int32)


def synthetic_gate(
    center=(320, 180),
    size=150,
    thickness=22,
    angle=0.0,
    filled=False,
    broken_side=None,
    perspective=False,
    uneven=False,
) -> np.ndarray:
    image = blank_frame()
    if perspective:
        outer = np.array([[205, 70], [440, 105], [405, 300], [235, 270]], np.int32)
        inner = np.array([[250, 120], [390, 140], [370, 250], [270, 235]], np.int32)
        cv2.fillConvexPoly(image, outer, ORANGE)
        if not filled:
            cv2.fillConvexPoly(image, inner, BACKGROUND)
        return image

    if broken_side is not None:
        half = size // 2
        x0, y0 = int(center[0] - half), int(center[1] - half)
        x1, y1 = int(center[0] + half), int(center[1] + half)
        sides = {
            "top": ((x0, y0), (x1, y0)),
            "right": ((x1, y0), (x1, y1)),
            "bottom": ((x1, y1), (x0, y1)),
            "left": ((x0, y1), (x0, y0)),
        }
        for name, endpoints in sides.items():
            if name != broken_side:
                cv2.line(image, *endpoints, ORANGE, thickness)
        return image

    outer = square_points(center, size, angle)
    cv2.fillConvexPoly(image, outer, ORANGE)
    if not filled:
        inner_size = max(2, size - 2 * thickness)
        inner_center = center
        if uneven:
            # Uneven border: the true opening is deliberately off the outer centre.
            inner_center = (center[0] + 13, center[1] - 9)
        inner = square_points(inner_center, inner_size, angle)
        cv2.fillConvexPoly(image, inner, BACKGROUND)
        if uneven:
            cv2.line(image, tuple(outer[0]), tuple(outer[1]), DIM_ORANGE, 7)
    return image


def add_gate(image, center, size, thickness=18):
    outer = square_points(center, size)
    inner = square_points(center, size - 2 * thickness)
    cv2.fillConvexPoly(image, outer, ORANGE)
    cv2.fillConvexPoly(image, inner, BACKGROUND)


def detection_at(
    nx=0.0,
    ny=0.0,
    confidence=0.85,
    opening_width=100.0,
    opening_height=None,
    predicted=False,
    stable_frames=5,
    velocity_x=0.0,
    velocity_y=0.0,
    size_rate=0.0,
    timestamp=1.0,
):
    opening_height = opening_width if opening_height is None else opening_height
    cx = (nx + 1.0) * 640.0 / 2.0
    cy = (ny + 1.0) * 360.0 / 2.0
    return GateDetection(
        found=True,
        center_x=cx,
        center_y=cy,
        normalized_x=nx,
        normalized_y=ny,
        opening_width=opening_width,
        opening_height=opening_height,
        apparent_area=opening_width * opening_height,
        confidence=confidence,
        method="inner_contour",
        predicted=predicted,
        stable_frames=stable_frames,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        size_rate=size_rate,
        frame_width=640,
        frame_height=360,
        timestamp=timestamp,
        bbox=(
            int(cx - opening_width / 2),
            int(cy - opening_height / 2),
            int(opening_width),
            int(opening_height),
        ),
    )


class DetectorPositionTests(unittest.TestCase):
    def setUp(self):
        self.detector = OrangeGateDetector()

    def _assert_position(self, center, x_sign=0, y_sign=0):
        result = self.detector.detect(synthetic_gate(center=center))
        self.assertTrue(result.found)
        self.assertEqual(np.sign(result.normalized_x), x_sign)
        self.assertEqual(np.sign(result.normalized_y), y_sign)
        return result

    def test_01_centered_gate(self):
        result = self._assert_position((320, 180), 0, 0)
        self.assertEqual(result.method, "inner_contour")

    def test_02_gate_left(self):
        self._assert_position((180, 180), -1, 0)

    def test_03_gate_right(self):
        self._assert_position((470, 180), 1, 0)

    def test_04_gate_above(self):
        self._assert_position((320, 100), 0, -1)

    def test_05_gate_below(self):
        self._assert_position((320, 270), 0, 1)

    def test_06_normalized_coordinate_convention(self):
        self.assertEqual(normalized_image_coordinates(0, 0, 640, 360), (-1.0, -1.0))
        self.assertEqual(normalized_image_coordinates(320, 180, 640, 360), (0.0, 0.0))
        self.assertEqual(normalized_image_coordinates(640, 360, 640, 360), (1.0, 1.0))


class DetectorRobustnessTests(unittest.TestCase):
    def setUp(self):
        self.detector = OrangeGateDetector()

    def test_07_rotated_gate(self):
        result = self.detector.detect(synthetic_gate(angle=31.0))
        self.assertTrue(result.found)
        self.assertEqual(result.method, "inner_contour")
        self.assertAlmostEqual(abs(result.angle_degrees), 31.0, delta=5.0)

    def test_08_perspective_distorted_gate(self):
        result = self.detector.detect(synthetic_gate(perspective=True))
        self.assertTrue(result.found)
        self.assertEqual(result.method, "inner_contour")
        self.assertAlmostEqual(result.center_x, 320, delta=18)
        self.assertAlmostEqual(result.center_y, 190, delta=18)

    def test_09_partially_clipped_gate_reconstructs_center(self):
        result = self.detector.detect(
            synthetic_gate(center=(20, 180), size=140, thickness=20)
        )
        self.assertTrue(result.found)
        self.assertEqual(result.method, "partial_gate")
        self.assertAlmostEqual(result.center_x, 20, delta=18)

    def test_10_gate_with_one_broken_side(self):
        result = self.detector.detect(
            synthetic_gate(broken_side="top", thickness=12)
        )
        self.assertTrue(result.found)
        self.assertIn(result.method, {"line_reconstruction", "quadrilateral"})
        self.assertAlmostEqual(result.center_x, 320, delta=20)

    def test_11_uneven_lighting_and_border_targets_opening(self):
        result = self.detector.detect(synthetic_gate(uneven=True))
        self.assertTrue(result.found)
        self.assertEqual(result.method, "inner_contour")
        self.assertAlmostEqual(result.center_x, 333, delta=5)
        self.assertAlmostEqual(result.center_y, 171, delta=5)

    def test_12_motion_blur(self):
        image = synthetic_gate()
        kernel = np.zeros((9, 9), dtype=np.float32)
        kernel[4, :] = 1.0 / 9.0
        blurred = cv2.filter2D(image, -1, kernel)
        result = self.detector.detect(blurred)
        self.assertTrue(result.found)
        self.assertAlmostEqual(result.center_x, 320, delta=10)

    def test_12b_hough_line_shapes_are_platform_independent(self):
        segments = np.array(
            [[10, 20, 30, 40], [50, 60, 70, 80], [90, 10, 20, 30]],
            dtype=np.int32,
        )
        windows_shape = segments.copy()
        singleton_shape = segments[:, np.newaxis, :]

        normalized_windows = self.detector._normalize_hough_lines(
            windows_shape
        )
        normalized_singleton = self.detector._normalize_hough_lines(
            singleton_shape
        )

        np.testing.assert_array_equal(normalized_windows, segments)
        np.testing.assert_array_equal(normalized_singleton, segments)

    def test_12c_debug_overlay_ignores_empty_cross_platform_contours(self):
        mask = np.zeros(FRAME_SIZE, dtype=np.uint8)
        debug = DetectorDebug(
            raw_mask=mask,
            cleaned_mask=mask,
            candidates=[
                CandidateDebug(
                    outer_contour=np.empty((0, 2), dtype=np.float32),
                    opening_contour=None,
                    accepted=False,
                    score=0.0,
                    confidence=0.0,
                    reason='empty',
                    method='none',
                    center=(0.0, 0.0),
                    bbox=(0, 0, 0, 0),
                ),
                CandidateDebug(
                    outer_contour=np.array(
                        [[10.0, 10.0], [30.0, 10.0], [20.0, 30.0]],
                        dtype=np.float32,
                    ),
                    opening_contour=None,
                    accepted=True,
                    score=0.5,
                    confidence=0.5,
                    reason='accepted',
                    method='line_reconstruction',
                    center=(20.0, 20.0),
                    bbox=(10, 10, 20, 20),
                ),
            ],
        )
        overlay = draw_detection(blank_frame(), None, debug=debug)
        self.assertEqual(overlay.shape, (*FRAME_SIZE, 3))

    def test_13_multiple_orange_objects_rejects_filled_floor_marking(self):
        image = synthetic_gate(center=(210, 160), size=110, thickness=16)
        cv2.rectangle(image, (350, 240), (630, 350), ORANGE, -1)
        result = self.detector.detect(image)
        self.assertTrue(result.found)
        self.assertLess(result.center_x, 300)

    def test_14_largest_valid_gate_overrides_temporal_hint(self):
        image = blank_frame()
        add_gate(image, (190, 180), 180, 24)
        add_gate(image, (470, 180), 100, 16)
        hint = detection_at(nx=(470 - 320) / 320, opening_width=68)
        result = self.detector.detect(image, hint=hint)
        self.assertTrue(result.found)
        self.assertLess(result.center_x, 300)

    def test_15_isolated_orange_post_is_rejected(self):
        image = blank_frame()
        cv2.rectangle(image, (310, 70), (330, 290), ORANGE, -1)
        self.assertFalse(self.detector.detect(image).found)

    def test_16_filled_rectangle_is_rejected(self):
        self.assertFalse(
            self.detector.detect(synthetic_gate(filled=True)).found
        )

    def test_17_small_orange_noise_is_rejected(self):
        image = blank_frame()
        for point in ((100, 100), (300, 200), (500, 70)):
            cv2.circle(image, point, 2, ORANGE, -1)
        self.assertFalse(self.detector.detect(image).found)

    def test_18_corner_ordering_and_diagonal_center(self):
        scrambled = np.array([[9, 9], [1, 1], [1, 9], [9, 1]], np.float32)
        ordered = order_corners(scrambled)
        np.testing.assert_allclose(
            ordered, np.array([[1, 1], [9, 1], [9, 9], [1, 9]], np.float32)
        )
        self.assertEqual(quadrilateral_diagonal_center(scrambled), (5.0, 5.0))
        diamond = order_corners(
            np.array([[5, 0], [10, 5], [5, 10], [0, 5]], np.float32)
        )
        self.assertEqual(len(np.unique(diamond, axis=0)), 4)

    def test_19_pnp_inputs_and_validation(self):
        points = build_gate_object_points(2.0, 1.0)
        np.testing.assert_allclose(points[0], [-1.0, 0.5, 0.0])
        estimate = estimate_gate(self.detector.detect(synthetic_gate()))
        self.assertEqual(estimate["method"], "pnp")
        self.assertLessEqual(estimate["pnp_reprojection_error"], 6.0)

    def test_19b_largest_gate_stays_primary_with_nested_gate(self):
        image = blank_frame()
        add_gate(image, (320, 180), 340, thickness=25)
        add_gate(image, (355, 170), 80, thickness=12)

        selected = OrangeGateDetector().detect(image)

        self.assertTrue(selected.found)
        self.assertGreater(selected.opening_width, 250.0)
        self.assertAlmostEqual(selected.center_x, 320.0, delta=10.0)


class TrackerTests(unittest.TestCase):
    def test_20_timestamped_velocity_and_short_dropout(self):
        tracker = GateTracker(GateTrackerConfig(maximum_missing_frames=2))
        tracker.update(detection_at(nx=0.0, timestamp=1.0), 1.0)
        moved = tracker.update(detection_at(nx=0.1, timestamp=1.1), 1.1)
        self.assertGreater(moved.velocity_x, 0.0)
        predicted = tracker.update(None, 1.2)
        self.assertTrue(predicted.predicted)
        self.assertEqual(predicted.method, "tracker_prediction")
        self.assertIsNotNone(tracker.update(None, 1.3))
        self.assertIsNone(tracker.update(None, 1.4))

    def test_21_center_jump_rejection(self):
        tracker = GateTracker(
            GateTrackerConfig(maximum_center_jump_ratio=0.2)
        )
        tracker.update(detection_at(nx=-0.4), 1.0)
        result = tracker.update(detection_at(nx=0.7), 1.1)
        self.assertTrue(result.predicted)
        self.assertLess(result.normalized_x, 0.0)

    def test_22_size_jump_rejection(self):
        tracker = GateTracker(
            GateTrackerConfig(maximum_size_change_ratio=0.5)
        )
        tracker.update(detection_at(opening_width=80), 1.0)
        result = tracker.update(detection_at(opening_width=200), 1.1)
        self.assertTrue(result.predicted)
        self.assertLess(result.opening_width, 100)


def navigator_for_tests(**overrides):
    values = dict(
        minimum_state_duration_s=0.0,
        command_lpf_alpha=1.0,
        max_forward_acceleration=100.0,
        max_lateral_acceleration=100.0,
        max_vertical_acceleration=100.0,
        max_yaw_acceleration=100.0,
    )
    values.update(overrides)
    return GateNavigator(NavigationConfig(**values))


def advance_to_align(navigator, detection):
    navigator.update(detection, 1.0)
    navigator.update(detection, 1.1)
    return navigator.update(detection, 1.2)


class NavigationTests(unittest.TestCase):
    def test_23_track_confirmation_stage(self):
        navigator = navigator_for_tests(track_confirmation_frames=3)
        noisy = detection_at(stable_frames=1, opening_width=80)
        self.assertEqual(
            navigator.update(noisy, 1.0).state, NavigationState.TRACK
        )
        self.assertEqual(
            navigator.update(noisy, 1.1).state, NavigationState.TRACK
        )

    def test_24_control_deadband(self):
        navigator = navigator_for_tests(commit_opening_area_ratio=0.5)
        centered = detection_at(nx=0.02, ny=-0.02, opening_width=70)
        command = advance_to_align(navigator, centered)
        self.assertEqual(command.state, NavigationState.ALIGN_AND_APPROACH)
        self.assertAlmostEqual(command.right_mps, 0.0, delta=1e-8)
        self.assertAlmostEqual(command.down_mps, 0.0, delta=1e-8)
        self.assertAlmostEqual(command.yaw_rate_rps, 0.0, delta=1e-8)

    def test_25_command_signs_and_clipping(self):
        navigator = navigator_for_tests(recovery_alignment_error=2.0)
        gate = detection_at(nx=1.0, ny=1.0, opening_width=60)
        command = advance_to_align(navigator, gate)
        self.assertGreater(command.right_mps, 0.0)
        self.assertGreater(command.down_mps, 0.0)
        self.assertGreater(command.yaw_rate_rps, 0.0)
        self.assertLessEqual(command.right_mps, navigator.config.max_right_mps)
        self.assertLessEqual(command.down_mps, navigator.config.max_down_mps)
        self.assertLessEqual(
            command.yaw_rate_rps, navigator.config.max_yaw_rate_rps
        )

    def test_26_forward_speed_rewards_alignment(self):
        aligned_nav = navigator_for_tests(commit_opening_area_ratio=0.5)
        poor_nav = navigator_for_tests(
            commit_opening_area_ratio=0.5, recovery_alignment_error=2.0
        )
        aligned = advance_to_align(
            aligned_nav, detection_at(nx=0.05, opening_width=60)
        )
        poor = advance_to_align(
            poor_nav, detection_at(nx=0.55, opening_width=60)
        )
        self.assertGreater(aligned.forward_mps, poor.forward_mps)

    def test_27_commit_requires_stability_and_size(self):
        navigator = navigator_for_tests(track_confirmation_frames=1)
        unstable = detection_at(
            opening_width=150, stable_frames=1, velocity_x=0.8
        )
        advance_to_align(navigator, unstable)
        self.assertEqual(
            navigator.update(unstable, 1.3).state,
            NavigationState.ALIGN_AND_APPROACH,
        )
        stable = detection_at(opening_width=150, stable_frames=6)
        self.assertEqual(
            navigator.update(stable, 1.4).state, NavigationState.COMMIT
        )

    def test_28_continue_forward_after_committed_gate_disappears(self):
        navigator = navigator_for_tests()
        close = detection_at(opening_width=160, stable_frames=7)
        advance_to_align(navigator, close)
        self.assertEqual(
            navigator.update(close, 1.3).state, NavigationState.COMMIT
        )
        command = navigator.update(None, 1.4)
        self.assertEqual(command.state, NavigationState.PASS_THROUGH)
        self.assertGreater(command.forward_mps, 0.0)
        self.assertEqual(
            navigator.update(None, 2.3).state, NavigationState.SEARCH
        )

    def test_29_unexpected_loss_enters_recovery(self):
        navigator = navigator_for_tests(recover_local_duration_s=1.0)
        far = detection_at(opening_width=70)
        advance_to_align(navigator, far)
        recovered = navigator.update(None, 1.3)
        self.assertEqual(recovered.state, NavigationState.RECOVER)
        self.assertEqual(recovered.forward_mps, 0.0)
        self.assertEqual(
            navigator.update(None, 2.3).state, NavigationState.SEARCH
        )

    def test_30_planner_axis_mapping(self):
        now = time.time_ns()
        data = {
            "control_source": "opencv",
            "navigation": {
                "ts": now,
                "forward_mps": 1.0,
                "right_mps": 0.3,
                "down_mps": 0.2,
                "yaw_rate_rps": 0.4,
                "state": "ALIGN_AND_APPROACH",
            },
            "attitude": {
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "ts": now,
            },
        }
        target = OpenCVGatePlanner().compute_target(data)
        self.assertGreater(target["vn"], 0.0)
        self.assertGreater(target["ve"], 0.0)
        self.assertGreater(target["vd"], 0.0)
        self.assertGreater(target["yaw_rate"], 0.0)

    def test_30b_vertical_setpoint_compensates_camera_tilt(self):
        navigator = navigator_for_tests(
            vertical_setpoint_normalized=0.60,
            commit_opening_area_ratio=0.5,
        )
        optical_centered = detection_at(ny=0.0, opening_width=70)
        command = advance_to_align(navigator, optical_centered)
        self.assertLess(command.down_mps, 0.0)

        navigator = navigator_for_tests(
            vertical_setpoint_normalized=0.60,
            commit_opening_area_ratio=0.5,
        )
        body_forward_centered = detection_at(ny=0.60, opening_width=70)
        command = advance_to_align(navigator, body_forward_centered)
        self.assertAlmostEqual(command.down_mps, 0.0, delta=1e-8)


class ModeTests(unittest.TestCase):
    def test_31_opencv_and_existing_ai_are_exclusive(self):
        opencv = VisionModeRouter(
            ModeRouterConfig(mode=GateNavigationMode.OPENCV)
        )
        existing_ai = VisionModeRouter(
            ModeRouterConfig(mode=GateNavigationMode.EXISTING_AI)
        )
        self.assertEqual(opencv.update(ai_available=True), "opencv")
        self.assertEqual(existing_ai.update(ai_available=True), "ai")
        self.assertEqual(existing_ai.update(ai_available=False), "safe")

    def test_31b_live_display_contains_detection_and_mask_panels(self):
        annotated = np.zeros((360, 640, 3), dtype=np.uint8)
        mask = np.zeros((180, 320), dtype=np.uint8)
        display = VisionRX.build_display_frame(annotated, mask)
        self.assertEqual(display.shape, (360, 1280, 3))

    def test_31c_course_lookahead_finds_supported_second_gate(self):
        image = np.full((360, 640, 3), BACKGROUND, dtype=np.uint8)
        add_gate(image, (320, 190), 170, 22)
        add_gate(image, (500, 135), 70, 10)
        detector = OrangeGateDetector()
        primary = detector.detect(image)
        lookahead = course_lookahead_horizontal(
            primary, detector.last_debug
        )
        self.assertIsNotNone(lookahead)
        self.assertGreater(lookahead, 0.45)


class RepositoryFrameTests(unittest.TestCase):
    def test_32_floor_markers_rejected_and_real_gate_detected(self):
        frames = Path(__file__).resolve().parent / "frames"
        detector = OrangeGateDetector()
        no_gate = detector.detect(
            cv2.imread(str(frames / "f_00004.png"), cv2.IMREAD_COLOR)
        )
        self.assertFalse(no_gate.found)

        real_gate = detector.detect(
            cv2.imread(str(frames / "f_00070.png"), cv2.IMREAD_COLOR)
        )
        self.assertTrue(real_gate.found)
        self.assertEqual(real_gate.method, "inner_contour")
        self.assertGreater(real_gate.confidence, 0.80)
        self.assertAlmostEqual(real_gate.center_x, 332.0, delta=12.0)
        self.assertAlmostEqual(real_gate.center_y, 270.0, delta=12.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
