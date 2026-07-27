"""Deterministic tests for four-keypoint gate pose inference."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from vision.yolo_pose_gate_detector import (
    PoseGateCandidate,
    PoseGateConfig,
    YoloPoseGateDetector,
    build_pose_orange_mask,
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


def test_acquisition_prefers_large_edge_gate_over_tiny_center_gate():
    candidates = [
        _candidate(
            (500, 100, 580, 171),
            0.55,
            0,
            ((505, 105), (575, 105), (505, 166), (575, 166)),
        ),
        _candidate(
            (344, 177, 362, 208),
            0.99,
            1,
            ((346, 179), (360, 179), (346, 206), (360, 206)),
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
    assert selected.box.source_index == 0


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


def test_lock_reacquires_fast_same_scale_gate_after_edge_yaw():
    previous = YoloGateBox((603, 172, 640, 251), 0.89)
    candidates = [
        _candidate(
            (487, 249, 537, 335),
            0.84,
            0,
            ((492, 254), (532, 254), (492, 330), (532, 330)),
        ),
        _candidate(
            (361, 217, 386, 265),
            0.91,
            1,
            ((364, 220), (383, 220), (364, 262), (383, 262)),
        ),
    ]

    selected = select_pose_target(
        candidates,
        previous,
        (360, 640, 3),
        PoseGateConfig(
            minimum_gate_area_px=100,
            target_association_center_span=1.85,
            target_association_min_area_ratio=0.45,
            target_association_max_area_ratio=2.20,
        ),
        lock_active=True,
    )

    assert selected is not None
    assert selected.box.source_index == 0


def test_lock_does_not_reacquire_distant_different_scale_gate():
    previous = YoloGateBox((603, 172, 640, 251), 0.89)
    candidates = [
        _candidate(
            (350, 190, 455, 350),
            0.98,
            0,
            ((360, 200), (445, 200), (360, 340), (445, 340)),
        ),
    ]

    selected = select_pose_target(
        candidates,
        previous,
        (360, 640, 3),
        PoseGateConfig(
            minimum_gate_area_px=100,
            target_association_center_span=1.85,
            target_association_min_area_ratio=0.45,
            target_association_max_area_ratio=2.20,
        ),
        lock_active=True,
    )

    assert selected is None


def test_persistent_lock_survives_timeout_and_requires_explicit_reset():
    continuation_and_larger_gate = _frame(
        rows=(
            (110, 65, 310, 265, 0.72, 0),
            (350, 30, 630, 330, 0.98, 0),
        ),
        points=(
            ((130, 85), (290, 85), (130, 245), (290, 245)),
            ((375, 55), (605, 55), (375, 305), (605, 305)),
        ),
        confidences=(
            (0.9, 0.9, 0.9, 0.9),
            (0.9, 0.9, 0.9, 0.9),
        ),
    )
    unrelated_gate = _frame(
        rows=((350, 30, 630, 330, 0.98, 0),),
        points=(
            ((375, 55), (605, 55), (375, 305), (605, 305)),
        ),
    )
    detector = YoloPoseGateDetector(
        PoseGateConfig(
            minimum_gate_area_px=100,
            target_lock_seconds=0.25,
            persistent_target_lock=True,
            previous_center_frames=1,
            log_interval_s=999,
        ),
        model=_FakeModel(
            [
                _frame(),
                continuation_and_larger_gate,
                unrelated_gate,
                unrelated_gate,
                unrelated_gate,
            ]
        ),
    )
    image = np.zeros((360, 640, 3), dtype=np.uint8)

    first = detector.detect(image, timestamp=1.0)
    continued = detector.detect(image, hint=first, timestamp=2.0)
    fallback = detector.detect(image, hint=continued, timestamp=3.0)
    missing = detector.detect(image, hint=fallback, timestamp=3.1)

    assert first.center_px == (200.0, 160.0)
    assert continued.center_px == (210.0, 165.0)
    assert fallback.predicted
    assert fallback.center_px == continued.center_px
    assert not missing.found
    assert detector._previous_target is not None

    detector.reset_target_lock()
    reacquired = detector.detect(image, timestamp=4.0)

    assert reacquired.found
    assert reacquired.center_px == (490.0, 180.0)


def test_post_pass_acquisition_rejects_oversized_gate_remnant():
    post_pass_scene = _frame(
        rows=(
            (0, 0, 430, 360, 0.96, 0),
            (500, 90, 620, 230, 0.86, 0),
        ),
        points=(
            ((10, 10), (420, 10), (10, 350), (420, 350)),
            ((510, 100), (610, 100), (510, 220), (610, 220)),
        ),
        confidences=(
            (0.9, 0.9, 0.9, 0.9),
            (0.9, 0.9, 0.9, 0.9),
        ),
    )
    detector = YoloPoseGateDetector(
        PoseGateConfig(
            minimum_gate_area_px=100,
            post_pass_rejection_seconds=0.8,
            post_pass_max_area_ratio=0.18,
            log_interval_s=999,
        ),
        model=_FakeModel([post_pass_scene, post_pass_scene]),
    )
    image = np.zeros((360, 640, 3), dtype=np.uint8)

    detector.begin_next_gate_acquisition(1.0)
    next_gate = detector.detect(image, timestamp=1.1)
    detector.reset_target_lock()
    after_guard = detector.detect(image, timestamp=2.0)

    assert next_gate.found
    assert next_gate.center_px == (560.0, 160.0)
    assert after_guard.found
    assert after_guard.center_px == (215.0, 180.0)


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
    assert result.confidence == pytest.approx(0.90)
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


def test_pose_mask_blur_and_opening_remove_isolated_noise():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[10, 10] = (0, 105, 255)
    cv2.rectangle(image, (40, 25), (120, 95), (0, 105, 255), 5)

    mask = build_pose_orange_mask(
        image,
        PoseGateConfig(
            hsv_blur_kernel=3,
            hsv_opening_kernel=3,
            hsv_closing_kernel=3,
        ),
    )

    assert mask[10, 10] == 0
    assert np.count_nonzero(mask[25:96, 40:121]) > 0


def test_hsv_refined_center_is_used_only_for_small_supported_shift():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (112, 65), (292, 255), (0, 105, 255), -1)
    cv2.rectangle(image, (145, 100), (260, 220), (0, 0, 0), -1)
    detector = YoloPoseGateDetector(
        PoseGateConfig(
            minimum_gate_area_px=100,
            require_hsv_confirmation=True,
            hsv_center_blend=0.5,
            hsv_center_max_shift_fraction=0.20,
            log_interval_s=999,
        ),
        model=_FakeModel([_frame()]),
    )

    result = detector.detect(image, timestamp=1.0)

    assert result.found
    assert result.method.startswith("yolo_pose_hsv_refined_center")
    assert result.center_x > 200.0
    assert detector.last_pose_debug.center_source == "hsv_refined_center"


def test_global_hsv_fallback_is_explicit_and_lower_confidence():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (170, 80), (330, 280), (0, 105, 255), -1)
    cv2.rectangle(image, (210, 120), (290, 240), (0, 0, 0), -1)
    detector = YoloPoseGateDetector(
        PoseGateConfig(
            global_hsv_fallback_enabled=True,
            global_hsv_fallback_confidence_scale=0.5,
            log_interval_s=999,
        ),
        model=_FakeModel([([], [], [])]),
    )

    result = detector.detect(image, timestamp=1.0)

    assert result.found
    assert result.method == "global_hsv_fallback"
    assert result.confidence <= 0.5


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
