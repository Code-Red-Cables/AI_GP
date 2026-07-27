"""Deterministic tests for four-keypoint gate pose inference."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from vision.yolo_pose_gate_detector import (
    PoseGateCandidate,
    PoseGateConfig,
    YoloPoseGateDetector,
    detect_gate_poses,
    reliable_pose_corners,
    select_pose_target,
)
from vision.yolo_gate_detector import YoloGateBox


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


class _FakeKeypoints:
    def __init__(self, points, confidences):
        self.xy = _FakeTensor(points)
        self.conf = _FakeTensor(confidences)


class _FakeResult:
    def __init__(self, rows, points, confidences, names):
        self.boxes = _FakeBoxes(rows)
        self.keypoints = _FakeKeypoints(points, confidences)
        self.names = names


class _FakeModel:
    def __init__(self, frames, names=None):
        self.frames = list(frames)
        self.names = names or {0: "gate"}
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        if self.frames:
            rows, points, confidences = self.frames.pop(0)
        else:
            rows, points, confidences = [], [], []
        return [
            _FakeResult(rows, points, confidences, self.names)
        ]


def _frame(
    rows=((100, 60, 300, 260, 0.90, 0),),
    points=(((125, 90), (275, 90), (125, 230), (275, 230)),),
    confidences=((0.9, 0.9, 0.9, 0.9),),
):
    return list(rows), list(points), list(confidences)


def _candidate(bbox, confidence, source_index, points):
    return PoseGateCandidate(
        box=YoloGateBox(
            bbox=bbox,
            confidence=confidence,
            source_index=source_index,
        ),
        keypoints=np.asarray(points, dtype=np.float32),
        keypoint_confidences=np.full(4, 0.9, dtype=np.float32),
    )


def test_pose_inference_preserves_instances_and_non_agnostic_nms():
    model = _FakeModel([_frame()])
    config = PoseGateConfig(minimum_gate_area_px=100)

    candidates = detect_gate_poses(
        np.zeros((360, 640, 3), dtype=np.uint8), model, config
    )

    assert len(candidates) == 1
    assert candidates[0].box.source_index == 0
    assert candidates[0].keypoints.shape == (4, 2)
    assert model.calls[0]["agnostic_nms"] is False
    assert model.calls[0]["iou"] == config.nms_iou_threshold


def test_generic_pose_weights_are_rejected():
    model = _FakeModel([_frame()], names={0: "person"})

    with pytest.raises(RuntimeError, match="custom class 'gate'"):
        detect_gate_poses(
            np.zeros((360, 640, 3), dtype=np.uint8),
            model,
            PoseGateConfig(),
        )


def test_pose_uses_box_center_and_keeps_outer_corners_debug_only():
    detector = YoloPoseGateDetector(
        PoseGateConfig(
            minimum_gate_area_px=100,
            log_interval_s=999,
        ),
        model=_FakeModel([_frame()]),
    )

    result = detector.detect(
        np.zeros((360, 640, 3), dtype=np.uint8), timestamp=1.0
    )

    assert result.found
    assert result.method == "yolo_pose_box_center"
    assert result.center_px == pytest.approx((200.0, 160.0))
    assert not result.corners_reliable
    assert result.corners is None
    assert np.allclose(
        detector.last_pose_debug.corners.as_polygon(),
        np.asarray([[125, 90], [275, 90], [275, 230], [125, 230]]),
    )
    assert detector.last_pose_debug.center_source == "yolo_box_center"


def test_asymmetric_keypoints_cannot_shift_steering_off_box_center():
    detector = YoloPoseGateDetector(
        PoseGateConfig(minimum_gate_area_px=100, log_interval_s=999),
        model=_FakeModel(
            [
                _frame(
                    points=(
                        (
                            (115, 70),
                            (245, 80),
                            (105, 200),
                            (250, 220),
                        ),
                    )
                )
            ]
        ),
    )

    result = detector.detect(
        np.zeros((360, 640, 3), dtype=np.uint8), timestamp=1.0
    )

    assert result.center_px == (200.0, 160.0)
    assert result.angle_degrees != 0.0


def test_largest_pose_instance_is_acquired_without_merging():
    candidates = [
        _candidate(
            (10, 20, 110, 120),
            0.95,
            0,
            ((20, 30), (100, 30), (20, 110), (100, 110)),
        ),
        _candidate(
            (180, 50, 430, 300),
            0.80,
            1,
            ((200, 70), (410, 70), (200, 280), (410, 280)),
        ),
    ]

    selected = select_pose_target(
        candidates,
        None,
        (360, 640, 3),
        PoseGateConfig(minimum_gate_area_px=100),
        lock_active=False,
    )

    assert selected is not None
    assert selected.box.source_index == 1
    assert selected.keypoints[0].tolist() == [200, 70]


def test_lock_keeps_the_same_physical_pose_instance():
    previous = YoloGateBox((190, 90, 350, 250), 0.82)
    candidates = [
        _candidate(
            (80, 40, 430, 330),
            0.96,
            0,
            ((90, 50), (420, 50), (90, 320), (420, 320)),
        ),
        _candidate(
            (194, 94, 354, 254),
            0.76,
            1,
            ((205, 105), (343, 105), (205, 243), (343, 243)),
        ),
    ]

    selected = select_pose_target(
        candidates,
        previous,
        (360, 640, 3),
        PoseGateConfig(minimum_gate_area_px=100),
        lock_active=True,
    )

    assert selected is not None
    assert selected.box.source_index == 1


def test_low_confidence_corner_uses_box_center_fallback():
    rows, points, confidences = _frame(
        confidences=((0.9, 0.1, 0.9, 0.9),)
    )
    detector = YoloPoseGateDetector(
        PoseGateConfig(
            minimum_gate_area_px=100,
            keypoint_confidence_threshold=0.25,
            log_interval_s=999,
        ),
        model=_FakeModel([(rows, points, confidences)]),
    )

    result = detector.detect(
        np.zeros((360, 640, 3), dtype=np.uint8), timestamp=1.0
    )

    assert result.found
    assert result.method == "yolo_pose_box_center_no_orientation"
    assert result.center_px == (200.0, 160.0)
    assert not result.corners_reliable


def test_missing_pose_uses_only_bounded_previous_frame_fallback():
    detector = YoloPoseGateDetector(
        PoseGateConfig(
            minimum_gate_area_px=100,
            previous_center_frames=2,
            log_interval_s=999,
        ),
        model=_FakeModel(
            [
                _frame(),
                ([], [], []),
                ([], [], []),
                ([], [], []),
            ]
        ),
    )
    image = np.zeros((360, 640, 3), dtype=np.uint8)

    measured = detector.detect(image, timestamp=1.0)
    first = detector.detect(image, timestamp=1.1)
    second = detector.detect(image, timestamp=1.2)
    expired = detector.detect(image, timestamp=2.0)

    assert measured.found
    assert first.predicted and second.predicted
    assert first.method == "yolo_pose_previous_fallback"
    assert not expired.found


def test_new_pose_target_requires_consecutive_confirmation_frames():
    detector = YoloPoseGateDetector(
        PoseGateConfig(
            minimum_gate_area_px=100,
            acquisition_confirmation_frames=3,
            log_interval_s=999,
        ),
        model=_FakeModel([_frame(), _frame(), _frame()]),
    )
    image = np.zeros((360, 640, 3), dtype=np.uint8)

    first = detector.detect(image, timestamp=1.0)
    second = detector.detect(image, timestamp=1.1)
    confirmed = detector.detect(image, timestamp=1.2)

    assert not first.found
    assert not second.found
    assert confirmed.found
    assert confirmed.center_px == (200.0, 160.0)


def test_inconsistent_pose_candidate_cannot_become_steering_target():
    shifted = _frame(
        rows=((360, 70, 500, 230, 0.92, 0),),
        points=(((375, 90), (485, 90), (375, 210), (485, 210)),),
    )
    detector = YoloPoseGateDetector(
        PoseGateConfig(
            minimum_gate_area_px=100,
            acquisition_confirmation_frames=3,
            log_interval_s=999,
        ),
        model=_FakeModel([_frame(), shifted, _frame()]),
    )
    image = np.zeros((360, 640, 3), dtype=np.uint8)

    results = [
        detector.detect(image, timestamp=timestamp)
        for timestamp in (1.0, 1.1, 1.2)
    ]

    assert not any(result.found for result in results)


def test_hsv_confirmation_rejects_yolo_box_without_orange_gate():
    detector = YoloPoseGateDetector(
        PoseGateConfig(
            minimum_gate_area_px=100,
            require_hsv_confirmation=True,
            log_interval_s=999,
        ),
        model=_FakeModel([_frame()]),
    )

    result = detector.detect(
        np.zeros((360, 640, 3), dtype=np.uint8), timestamp=1.0
    )

    assert not result.found
    assert len(detector.last_pose_debug.candidates) == 1
    assert not detector.last_pose_debug.candidates[0].hsv_confirmed
    assert not detector.last_debug.candidates[0].accepted


def test_hsv_confirmation_accepts_orange_supported_gate_frame():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (100, 60), (300, 260), (0, 105, 255), -1)
    cv2.rectangle(image, (135, 95), (265, 225), (0, 0, 0), -1)
    detector = YoloPoseGateDetector(
        PoseGateConfig(
            minimum_gate_area_px=100,
            require_hsv_confirmation=True,
            log_interval_s=999,
        ),
        model=_FakeModel([_frame()]),
    )

    result = detector.detect(image, timestamp=1.0)
    candidate = detector.last_pose_debug.candidates[0]

    assert result.found
    assert candidate.hsv_confirmed
    assert candidate.hsv_supported_sides == 4
    assert 0.12 <= candidate.hsv_orange_ratio <= 0.72
    assert np.count_nonzero(detector.last_debug.cleaned_mask) > 0


def test_hsv_confirmation_rejects_local_orange_reflection_patch():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (170, 130), (230, 190), (0, 105, 255), -1)
    detector = YoloPoseGateDetector(
        PoseGateConfig(
            minimum_gate_area_px=100,
            require_hsv_confirmation=True,
            log_interval_s=999,
        ),
        model=_FakeModel([_frame()]),
    )

    result = detector.detect(image, timestamp=1.0)
    candidate = detector.last_pose_debug.candidates[0]

    assert not result.found
    assert not candidate.hsv_confirmed
    assert candidate.hsv_supported_sides < 3


def test_reliable_corners_require_all_four_points():
    candidate = _candidate(
        (100, 60, 300, 260),
        0.9,
        0,
        ((125, 90), (275, 90), (125, 230), (275, 230)),
    )
    candidate = PoseGateCandidate(
        box=candidate.box,
        keypoints=candidate.keypoints,
        keypoint_confidences=np.asarray([0.9, 0.9, 0.0, 0.9]),
    )

    corners, reason = reliable_pose_corners(
        candidate,
        PoseGateConfig(keypoint_confidence_threshold=0.25),
    )

    assert corners is None
    assert reason == "low_keypoint_confidence"
