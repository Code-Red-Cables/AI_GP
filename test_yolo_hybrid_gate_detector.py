"""Deterministic tests for YOLO-isolated crop-local gate extraction."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from vision.yolo_gate_detector import (
    HybridGateConfig,
    InnerGateCorners,
    YoloGateBox,
    YoloHybridGateDetector,
    calculate_gate_center,
    convert_crop_coordinates,
    crop_target_gate,
    detect_gates_yolo,
    extract_inner_gate_corners,
    select_target_gate,
)


ORANGE = (0, 105, 255)


class _FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _FakeBoxes:
    def __init__(self, rows):
        rows = list(rows)
        self.xyxy = _FakeTensor([row[:4] for row in rows])
        self.conf = _FakeTensor([row[4] for row in rows])
        self.cls = _FakeTensor([row[5] for row in rows])


class _FakeResult:
    def __init__(self, rows, names):
        self.boxes = _FakeBoxes(rows)
        self.names = names


class _FakeModel:
    def __init__(self, frames, names=None):
        self.frames = list(frames)
        self.names = names or {0: "gate"}
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        rows = self.frames.pop(0) if self.frames else []
        return [_FakeResult(rows, self.names)]


def gate_frame(
    shape=(360, 640, 3),
    outer=(170, 80, 330, 280),
    inner=(210, 120, 290, 240),
):
    frame = np.zeros(shape, dtype=np.uint8)
    cv2.rectangle(frame, outer[:2], outer[2:], ORANGE, -1)
    cv2.rectangle(frame, inner[:2], inner[2:], (20, 20, 20), -1)
    return frame


class TargetSelectionTests(unittest.TestCase):
    def setUp(self):
        self.config = HybridGateConfig(
            minimum_gate_area_px=100.0,
            log_interval_s=999.0,
        )

    def test_largest_gate_is_selected_at_acquisition(self):
        detections = [
            YoloGateBox((40, 40, 140, 140), 0.95),
            YoloGateBox((180, 50, 430, 300), 0.80),
        ]

        selected = select_target_gate(
            detections,
            None,
            (360, 640, 3),
            self.config,
            lock_active=False,
        )

        self.assertEqual(selected.bbox, detections[1].bbox)

    def test_lock_prefers_overlap_with_previous_target(self):
        previous = YoloGateBox((190, 90, 350, 250), 0.82)
        detections = [
            YoloGateBox((80, 40, 430, 330), 0.96),
            YoloGateBox((194, 94, 354, 254), 0.76),
        ]

        selected = select_target_gate(
            detections,
            previous,
            (360, 640, 3),
            self.config,
            lock_active=True,
        )

        self.assertEqual(selected.bbox, detections[1].bbox)

    def test_tiny_and_mostly_outside_boxes_are_rejected(self):
        config = HybridGateConfig(
            minimum_gate_area_px=500.0,
            maximum_outside_fraction=0.25,
        )
        detections = [
            YoloGateBox((10, 10, 20, 20), 0.99),
            YoloGateBox((-500, 20, 50, 300), 0.95),
        ]

        selected = select_target_gate(
            detections,
            None,
            (360, 640, 3),
            config,
            lock_active=False,
        )

        self.assertIsNone(selected)


class CropCornerTests(unittest.TestCase):
    def setUp(self):
        self.config = HybridGateConfig(
            minimum_opening_area_px=30.0,
            minimum_opening_side_px=8.0,
        )

    def test_crop_and_coordinate_conversion(self):
        frame = gate_frame()
        crop, crop_bbox = crop_target_gate(
            frame, (160, 70, 340, 290), padding_px=10
        )
        corners, mask, reason = extract_inner_gate_corners(
            crop, self.config
        )

        self.assertEqual(reason, "inner_corners")
        self.assertIsNotNone(corners)
        self.assertGreater(np.count_nonzero(mask), 0)
        full = convert_crop_coordinates(corners, crop_bbox)
        center = calculate_gate_center(full)
        self.assertAlmostEqual(center[0], 250.0, delta=3.0)
        self.assertAlmostEqual(center[1], 180.0, delta=3.0)
        self.assertLess(full.top_left[0], full.top_right[0])
        self.assertLess(full.top_left[1], full.bottom_left[1])

    def test_calculate_center_averages_all_four_named_corners(self):
        corners = InnerGateCorners(
            top_left=(10, 20),
            top_right=(30, 10),
            bottom_left=(20, 50),
            bottom_right=(40, 40),
        )
        self.assertEqual(calculate_gate_center(corners), (25.0, 30.0))

    def test_white_letter_has_no_valid_inner_opening(self):
        crop = np.full((260, 260, 3), ORANGE, dtype=np.uint8)
        cv2.putText(
            crop,
            "A",
            (35, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            7.0,
            (255, 255, 255),
            30,
            cv2.LINE_AA,
        )

        corners, _, _ = extract_inner_gate_corners(crop, self.config)

        self.assertIsNone(corners)


class HybridDetectorTests(unittest.TestCase):
    def setUp(self):
        self.config = HybridGateConfig(
            model_path="unused-in-test.pt",
            minimum_gate_area_px=100.0,
            minimum_opening_area_px=30.0,
            minimum_opening_side_px=8.0,
            previous_center_frames=2,
            log_interval_s=999.0,
        )

    def test_predict_uses_non_agnostic_high_iou_nms(self):
        model = _FakeModel([[(10, 20, 110, 120, 0.8, 0)]])

        detections = detect_gates_yolo(
            np.zeros((180, 320, 3), dtype=np.uint8),
            model,
            self.config,
        )

        self.assertEqual(len(detections), 1)
        self.assertFalse(model.calls[0]["agnostic_nms"])
        self.assertAlmostEqual(
            model.calls[0]["iou"], self.config.nms_iou_threshold
        )

    def test_inner_corner_center_is_published_in_full_coordinates(self):
        model = _FakeModel(
            [[(160, 70, 340, 290, 0.91, 0)]]
        )
        detector = YoloHybridGateDetector(self.config, model=model)

        result = detector.detect(gate_frame(), timestamp=1.0)

        self.assertTrue(result.found)
        self.assertEqual(result.method, "yolo_inner_corners")
        self.assertAlmostEqual(result.center_x, 250.0, delta=3.0)
        self.assertAlmostEqual(result.center_y, 180.0, delta=3.0)
        self.assertEqual(
            detector.last_hybrid_debug.center_source, "inner_corners"
        )

    def test_yolo_box_center_is_safe_fallback_without_orange_hole(self):
        model = _FakeModel(
            [[(100, 60, 300, 260, 0.80, 0)]]
        )
        detector = YoloHybridGateDetector(self.config, model=model)
        blank = np.zeros((360, 640, 3), dtype=np.uint8)

        result = detector.detect(blank, timestamp=1.0)

        self.assertTrue(result.found)
        self.assertEqual(result.method, "yolo_box_fallback")
        self.assertEqual(result.center_px, (200.0, 160.0))
        self.assertEqual(
            detector.last_hybrid_debug.center_source,
            "yolo_box_fallback",
        )

    def test_previous_center_fallback_is_finite(self):
        model = _FakeModel(
            [
                [(160, 70, 340, 290, 0.91, 0)],
                [],
                [],
                [],
            ]
        )
        detector = YoloHybridGateDetector(self.config, model=model)
        frame = gate_frame()
        measured = detector.detect(frame, timestamp=1.0)
        first_missing = detector.detect(frame, timestamp=1.1)
        second_missing = detector.detect(frame, timestamp=1.2)
        expired = detector.detect(frame, timestamp=2.0)

        self.assertTrue(measured.found)
        self.assertTrue(first_missing.predicted)
        self.assertTrue(second_missing.predicted)
        self.assertFalse(expired.found)
        self.assertEqual(
            first_missing.method, "yolo_previous_fallback"
        )

    def test_generic_weights_are_rejected(self):
        model = _FakeModel(
            [[(10, 20, 110, 120, 0.8, 0)]],
            names={0: "person"},
        )

        with self.assertRaisesRegex(RuntimeError, "custom class 'gate'"):
            detect_gates_yolo(
                np.zeros((180, 320, 3), dtype=np.uint8),
                model,
                self.config,
            )


if __name__ == "__main__":
    unittest.main()
