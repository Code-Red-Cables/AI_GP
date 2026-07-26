"""Deterministic blue-course path detection and navigation-assist tests."""

import unittest

import cv2
import numpy as np

from test_opencv_gate_navigation import detection_at
from vision.navigation import (
    GateNavigator,
    NavigationState,
    q2_demo_navigation_config,
)
from vision.path_detector import BluePathDetection, BluePathDetector
from vision_rx import VisionRX


def path_frame(shift: int = 0) -> np.ndarray:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cyan = (255, 180, 0)
    cv2.line(frame, (90 + shift, 359), (285 + shift, 150), cyan, 9)
    cv2.line(frame, (550 + shift, 359), (355 + shift, 150), cyan, 9)
    return frame


class BluePathDetectorTests(unittest.TestCase):
    def test_centered_converging_path_detected(self):
        detection = BluePathDetector().detect(path_frame())
        self.assertTrue(detection.found)
        self.assertGreaterEqual(detection.confidence, 0.30)
        self.assertAlmostEqual(detection.normalized_offset, 0.0, delta=0.08)
        self.assertAlmostEqual(detection.normalized_heading, 0.0, delta=0.08)

    def test_right_shifted_path_has_positive_offset(self):
        detection = BluePathDetector().detect(path_frame(shift=55))
        self.assertTrue(detection.found)
        self.assertGreater(detection.normalized_offset, 0.10)

    def test_blue_object_above_path_roi_is_ignored(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 20), (540, 100), (255, 180, 0), -1)
        detection = BluePathDetector().detect(frame)
        self.assertFalse(detection.found)

    def test_narrow_path_visible_through_gate_is_detected(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cyan = (255, 180, 0)
        cv2.line(frame, (285, 359), (310, 150), cyan, 7)
        cv2.line(frame, (355, 359), (330, 150), cyan, 7)
        detection = BluePathDetector().detect(frame)
        self.assertTrue(detection.found)
        self.assertAlmostEqual(
            detection.normalized_offset, 0.0, delta=0.08
        )


class Q2BluePathIsolationTests(unittest.TestCase):
    @staticmethod
    def right_path() -> BluePathDetection:
        return BluePathDetection(
            found=True,
            confidence=0.9,
            normalized_offset=0.40,
            normalized_heading=0.10,
        )

    def test_q2_profile_ignores_path_only_detection(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        command = navigator.update(None, 1.0, path=self.right_path())
        self.assertEqual(command.right_mps, 0.0)
        self.assertEqual(command.yaw_rate_rps, 0.0)
        self.assertGreater(command.forward_mps, 0.0)

    def test_q2_profile_ignores_path_with_gate(self):
        path = self.right_path()
        centered_gate = detection_at(
            nx=0.0, ny=0.16, confidence=0.9, stable_frames=1
        )
        with_gate = GateNavigator(q2_demo_navigation_config()).update(
            centered_gate, 1.0, path=path
        )
        without_gate_path = GateNavigator(
            q2_demo_navigation_config()
        ).update(centered_gate, 1.0)
        self.assertEqual(with_gate.right_mps, without_gate_path.right_mps)
        self.assertEqual(
            with_gate.yaw_rate_rps, without_gate_path.yaw_rate_rps
        )

    def test_path_assist_is_suspended_during_gate_commit(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        navigator.state = NavigationState.COMMIT
        command = navigator.update(None, 1.0, path=self.right_path())
        self.assertAlmostEqual(command.right_mps, 0.0)
        self.assertAlmostEqual(command.yaw_rate_rps, 0.0)

    def test_path_assist_stays_disabled_during_pass_through(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        navigator.state = NavigationState.PASS_THROUGH
        navigator._state_since = 1.0
        early = navigator.update(None, 1.10, path=self.right_path())
        self.assertAlmostEqual(early.right_mps, 0.0)
        active = navigator.update(None, 1.20, path=self.right_path())
        self.assertEqual(active.right_mps, 0.0)
        self.assertEqual(active.yaw_rate_rps, 0.0)

    def test_visible_next_gate_adds_bounded_early_turn(self):
        cfg = q2_demo_navigation_config()
        primary = detection_at(
            nx=0.0,
            ny=-0.20,
            opening_width=55,
            opening_height=40,
            stable_frames=5,
        )
        without = GateNavigator(cfg)
        without.update(primary, 1.0)
        without.update(primary, 1.1)
        base = without.update(primary, 1.2)
        with_lookahead = GateNavigator(cfg)
        with_lookahead.update(primary, 1.0)
        with_lookahead.update(primary, 1.1)
        anticipated = with_lookahead.update(
            primary, 1.2, next_gate_horizontal=0.50
        )
        self.assertGreater(anticipated.right_mps, base.right_mps)
        self.assertGreater(
            anticipated.yaw_rate_rps, base.yaw_rate_rps
        )
        self.assertLessEqual(
            anticipated.right_mps, cfg.next_gate_max_right_mps
        )
        self.assertLessEqual(
            anticipated.yaw_rate_rps,
            cfg.next_gate_max_yaw_rate_rps,
        )
    def test_accepted_target_panel_does_not_show_path(self):
        panel = VisionRX.build_accepted_target_frame(
            (360, 640, 3), None
        )
        self.assertEqual(panel.shape, (360, 640, 3))
        self.assertEqual(np.count_nonzero(panel), 0)


if __name__ == '__main__':
    unittest.main()
