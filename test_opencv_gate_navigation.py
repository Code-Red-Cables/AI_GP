"""Focused deterministic tests for OpenCV gate perception and navigation."""

from __future__ import annotations

import math
import threading
import time
import unittest

import cv2
import numpy as np

from gate_estimator import build_gate_object_points, estimate_gate
from planner import Planner
from vision.gate_detector import (
    GateDetection,
    OrangeGateDetector,
    normalized_image_coordinates,
    order_corners,
)
from vision.gate_tracker import GateTracker, TrackerConfig
from vision.mode_router import ModeRouterConfig, VisionMode, VisionModeRouter
from vision.navigation import GateNavigator, NavigationConfig, NavigationState


ORANGE = (0, 100, 255)
BACKGROUND = (45, 45, 45)


def synthetic_gate(
    center=(320, 180),
    size=150,
    thickness=22,
    angle=0.0,
    filled=False,
):
    image = np.full((360, 640, 3), BACKGROUND, dtype=np.uint8)
    outer = cv2.boxPoints((center, (size, size), angle)).astype(np.int32)
    cv2.fillConvexPoly(image, outer, ORANGE)
    if not filled:
        inner_size = max(1, size - 2 * thickness)
        inner = cv2.boxPoints((center, (inner_size, inner_size), angle)).astype(np.int32)
        cv2.fillConvexPoly(image, inner, BACKGROUND)
    return image


def detection_at(
    nx=0.0,
    ny=0.0,
    confidence=0.9,
    distance=8.0,
    width=100.0,
    predicted=False,
):
    cx = (nx + 1.0) * 0.5 * 639.0
    cy = (ny + 1.0) * 0.5 * 359.0
    return GateDetection(
        found=True,
        center_x=cx,
        center_y=cy,
        normalized_x=nx,
        normalized_y=ny,
        width=width,
        height=width,
        area=width * width,
        confidence=confidence,
        distance_m=distance,
        predicted=predicted,
        frame_width=640,
        frame_height=360,
        bbox=(int(cx - width / 2), int(cy - width / 2), int(width), int(width)),
    )


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = OrangeGateDetector()

    def test_orange_hsv_segmentation_and_center(self):
        result = self.detector.detect(synthetic_gate())
        self.assertTrue(result.found)
        self.assertGreater(result.confidence, 0.5)
        self.assertAlmostEqual(result.normalized_x, 0.0, delta=0.02)
        self.assertAlmostEqual(result.normalized_y, 0.0, delta=0.02)
        self.assertGreater(cv2.countNonZero(self.detector.last_debug.mask), 100)

    def test_normalized_coordinates(self):
        self.assertEqual(normalized_image_coordinates(0, 0, 641, 361), (-1.0, -1.0))
        nx, ny = normalized_image_coordinates(320, 180, 641, 361)
        self.assertAlmostEqual(nx, 0.0)
        self.assertAlmostEqual(ny, 0.0)
        self.assertEqual(normalized_image_coordinates(640, 360, 641, 361), (1.0, 1.0))

    def test_rotated_gate_and_corner_order(self):
        result = self.detector.detect(synthetic_gate(angle=27.0))
        self.assertTrue(result.found)
        self.assertIsNotNone(result.corners)
        ordered = order_corners(np.array([[9, 9], [1, 1], [9, 1], [1, 9]], np.float32))
        np.testing.assert_allclose(
            ordered, np.array([[1, 1], [9, 1], [9, 9], [1, 9]], np.float32)
        )
        diamond = order_corners(
            np.array([[5, 0], [10, 5], [5, 10], [0, 5]], np.float32)
        )
        self.assertEqual(len(np.unique(diamond, axis=0)), 4)

    def test_reject_tiny_and_filled_orange_objects(self):
        tiny = np.full((360, 640, 3), BACKGROUND, dtype=np.uint8)
        cv2.circle(tiny, (320, 180), 2, ORANGE, -1)
        self.assertFalse(self.detector.detect(tiny).found)
        self.assertFalse(self.detector.detect(synthetic_gate(filled=True)).found)

    def test_partially_visible_gate(self):
        result = self.detector.detect(
            synthetic_gate(center=(20, 180), size=130, thickness=20)
        )
        self.assertTrue(result.found)
        self.assertLess(result.normalized_x, -0.8)

    def test_multiple_shapes_selects_largest_gate(self):
        image = synthetic_gate(center=(210, 180), size=150, thickness=22)
        small = synthetic_gate(center=(480, 180), size=70, thickness=12)
        orange_pixels = np.any(small != np.array(BACKGROUND, dtype=np.uint8), axis=2)
        image[orange_pixels] = small[orange_pixels]
        result = self.detector.detect(image)
        self.assertTrue(result.found)
        self.assertLess(result.center_x, 320)

    def test_pnp_input_and_validated_pose(self):
        points = build_gate_object_points(2.0, 1.0)
        self.assertEqual(points.shape, (4, 3))
        np.testing.assert_allclose(points[0], [-1.0, 0.5, 0.0])
        detection = self.detector.detect(synthetic_gate())
        estimate = estimate_gate(detection)
        self.assertEqual(estimate["method"], "pnp")
        self.assertLessEqual(estimate["pnp_reprojection_error"], 6.0)


class TrackerControllerTests(unittest.TestCase):
    def test_short_dropout_prediction_then_reset(self):
        tracker = GateTracker(TrackerConfig(max_missing_frames=2))
        measured = tracker.update(detection_at(nx=0.2))
        self.assertFalse(measured.predicted)
        self.assertTrue(tracker.update(None).predicted)
        self.assertTrue(tracker.update(None).predicted)
        self.assertIsNone(tracker.update(None))

    def test_implausible_jump_is_rejected(self):
        tracker = GateTracker(TrackerConfig(max_center_jump=0.2))
        tracker.update(detection_at(nx=-0.5))
        result = tracker.update(detection_at(nx=0.7))
        self.assertTrue(result.predicted)
        self.assertLess(result.normalized_x, 0.0)

    def test_controller_signs_and_clipping(self):
        cfg = NavigationConfig(
            minimum_state_dwell_s=0.0,
            command_lpf_alpha=1.0,
            max_command_delta=10.0,
        )
        navigator = GateNavigator(cfg)
        command = navigator.update(detection_at(nx=0.8, ny=0.7), 1.0)
        self.assertEqual(command.state, NavigationState.ALIGN)
        self.assertGreater(command.right_mps, 0.0)
        self.assertGreater(command.down_mps, 0.0)
        self.assertGreater(command.yaw_rate_rps, 0.0)
        self.assertLessEqual(abs(command.right_mps), cfg.max_right_mps)
        self.assertLessEqual(abs(command.down_mps), cfg.max_down_mps)
        self.assertLessEqual(abs(command.yaw_rate_rps), cfg.max_yaw_rate_rps)

    def test_state_transitions_commit_and_pass_dropout(self):
        cfg = NavigationConfig(minimum_state_dwell_s=0.0, command_lpf_alpha=1.0)
        navigator = GateNavigator(cfg)
        close = detection_at(distance=2.0, width=220)
        self.assertEqual(navigator.update(close, 1.0).state, NavigationState.ALIGN)
        self.assertEqual(navigator.update(close, 1.1).state, NavigationState.APPROACH)
        self.assertEqual(navigator.update(close, 1.2).state, NavigationState.COMMIT)
        self.assertEqual(navigator.update(None, 1.4).state, NavigationState.COMMIT)
        passed = navigator.update(None, 1.8)
        self.assertEqual(passed.state, NavigationState.PASS_THROUGH)
        self.assertGreater(passed.forward_mps, 0.0)
        self.assertEqual(navigator.update(None, 2.8).state, NavigationState.SEARCH)

    def test_unexpected_loss_recovers_locally(self):
        cfg = NavigationConfig(
            minimum_state_dwell_s=0.0,
            recover_duration_s=1.0,
            command_lpf_alpha=1.0,
        )
        navigator = GateNavigator(cfg)
        navigator.update(detection_at(nx=0.4), 1.0)
        recover = navigator.update(None, 1.1)
        self.assertEqual(recover.state, NavigationState.RECOVER)
        self.assertEqual(recover.forward_mps, 0.0)
        self.assertGreater(recover.yaw_rate_rps, 0.0)
        self.assertEqual(navigator.update(None, 2.1).state, NavigationState.SEARCH)

    def test_planner_command_axis_signs(self):
        now = time.time_ns()
        data = {
            "lock": threading.RLock(),
            "control_source": "opencv",
            "navigation": {
                "ts": now,
                "forward_mps": 1.0,
                "right_mps": 0.3,
                "down_mps": 0.2,
                "yaw_rate_rps": 0.4,
                "state": "ALIGN",
            },
            "attitude": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "ts": now},
        }
        target = Planner(data).compute_target()
        self.assertGreater(target["vel_ned"][0], 0.0)
        self.assertGreater(target["vel_ned"][1], 0.0)
        self.assertGreater(target["vel_ned"][2], 0.0)
        self.assertGreater(target["yaw"], 0.0)


class HybridModeTests(unittest.TestCase):
    def test_switching_hysteresis_and_cooldown(self):
        router = VisionModeRouter(
            ModeRouterConfig(
                mode=VisionMode.HYBRID,
                low_confidence_frames=3,
                recovery_frames=2,
                cooldown_frames=2,
            )
        )
        self.assertEqual(router.update(0.0, True), "opencv")
        self.assertEqual(router.update(0.0, True), "opencv")
        self.assertEqual(router.update(0.0, True), "ai")
        # Cooldown blocks an immediate flip-flop.
        self.assertEqual(router.update(1.0, True), "ai")
        self.assertEqual(router.update(1.0, True), "opencv")


if __name__ == "__main__":
    unittest.main(verbosity=2)
