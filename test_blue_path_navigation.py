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


class BluePathNavigationTests(unittest.TestCase):
    @staticmethod
    def right_path() -> BluePathDetection:
        return BluePathDetection(
            found=True,
            confidence=0.9,
            normalized_offset=0.40,
            normalized_heading=0.10,
        )

    def test_path_only_commands_right_correction(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        command = navigator.update(None, 1.0, path=self.right_path())
        self.assertGreater(command.right_mps, 0.0)
        self.assertGreater(command.yaw_rate_rps, 0.0)
        self.assertGreater(command.forward_mps, 0.0)

    def test_gate_keeps_path_assist_conservative(self):
        path = self.right_path()
        path_only = GateNavigator(q2_demo_navigation_config()).update(
            None, 1.0, path=path
        )
        centered_gate = detection_at(
            nx=0.0, ny=0.16, confidence=0.9, stable_frames=1
        )
        with_gate = GateNavigator(q2_demo_navigation_config()).update(
            centered_gate, 1.0, path=path
        )
        self.assertLess(abs(with_gate.right_mps), abs(path_only.right_mps))
        self.assertLess(
            abs(with_gate.yaw_rate_rps), abs(path_only.yaw_rate_rps)
        )

    def test_path_assist_is_suspended_during_gate_commit(self):
        navigator = GateNavigator(q2_demo_navigation_config())
        navigator.state = NavigationState.COMMIT
        command = navigator.update(None, 1.0, path=self.right_path())
        self.assertAlmostEqual(command.right_mps, 0.0)
        self.assertAlmostEqual(command.yaw_rate_rps, 0.0)

    def test_accepted_target_panel_contains_path(self):
        path = BluePathDetector().detect(path_frame())
        panel = VisionRX.build_accepted_target_frame(
            (360, 640, 3), None, path
        )
        self.assertEqual(panel.shape, (360, 640, 3))
        self.assertGreater(np.count_nonzero(panel), 0)


if __name__ == '__main__':
    unittest.main()
