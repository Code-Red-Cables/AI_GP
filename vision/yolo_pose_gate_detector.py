"""YOLO pose detector for separately labeled overlapping racing gates.

The model separates physical gate instances and predicts the annotated outer
gate corners. The YOLO box owns the steering center because the training
keypoints nearly duplicate the box corners and are noisier. Keypoints provide
orientation only and are intentionally excluded from inner-opening PnP.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np

import camera_model as cm
from .gate_detector import (
    CandidateDebug,
    DetectorDebug,
    GateDetection,
    normalized_image_coordinates,
    order_corners,
)
from .yolo_gate_detector import (
    InnerGateCorners,
    YoloGateBox,
    select_target_gate,
)


@dataclass(frozen=True)
class PoseGateConfig:
    model_path: str = "models/gate_pose.pt"
    gate_class_name: str = "gate"
    confidence_threshold: float = 0.25
    keypoint_confidence_threshold: float = 0.25
    nms_iou_threshold: float = 0.70
    target_lock_seconds: float = 0.75
    acquisition_confirmation_frames: int = 1
    minimum_gate_area_px: float = 400.0
    maximum_outside_fraction: float = 0.35
    minimum_opening_area_px: float = 64.0
    minimum_opening_side_px: float = 8.0
    previous_center_frames: int = 5
    estimated_opening_scale: float = 0.72
    inference_size: int = 640
    device: Optional[str] = None
    log_interval_s: float = 1.0
    gate_inner_width_m: float = cm.GATE_INNER_M
    focal_length_px: float = cm.FX


@dataclass(frozen=True)
class PoseGateCandidate:
    box: YoloGateBox
    keypoints: np.ndarray
    keypoint_confidences: np.ndarray

    @property
    def keypoint_center(self) -> tuple[float, float]:
        points = np.asarray(self.keypoints, dtype=np.float32).reshape(-1, 2)
        if len(points) == 0:
            return self.box.center
        center = points.mean(axis=0)
        return float(center[0]), float(center[1])


@dataclass
class PoseGateDebug:
    candidates: list[PoseGateCandidate] = field(default_factory=list)
    selected: Optional[PoseGateCandidate] = None
    corners: Optional[InnerGateCorners] = None
    center: Optional[tuple[float, float]] = None
    center_source: str = "none"
    keypoint_reason: str = "no_target"
    missing_frames: int = 0


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _model_names(result: Any, model: Any) -> dict[int, str]:
    names = getattr(result, "names", None)
    if names is None:
        names = getattr(model, "names", None)
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def detect_gate_poses(
    frame: np.ndarray,
    model: Any,
    config: PoseGateConfig,
) -> list[PoseGateCandidate]:
    """Run one pose inference and retain per-instance boxes and keypoints."""
    predict_kwargs = {
        "source": frame,
        "conf": config.confidence_threshold,
        "iou": config.nms_iou_threshold,
        "agnostic_nms": False,
        "verbose": False,
        "imgsz": config.inference_size,
    }
    if config.device:
        predict_kwargs["device"] = config.device
    results = model.predict(**predict_kwargs)
    if not results:
        return []
    result = results[0]
    names = _model_names(result, model)
    gate_ids = {
        class_id
        for class_id, name in names.items()
        if name.strip().lower() == config.gate_class_name.strip().lower()
    }
    if not gate_ids:
        available = ", ".join(
            f"{class_id}:{name}" for class_id, name in sorted(names.items())
        ) or "<none>"
        raise RuntimeError(
            "YOLO pose weights do not define the required custom class "
            f"{config.gate_class_name!r}; classes are {available}."
        )

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = _to_numpy(getattr(boxes, "xyxy", None)).reshape(-1, 4)
    box_confidences = _to_numpy(
        getattr(boxes, "conf", None)
    ).reshape(-1)
    classes = _to_numpy(getattr(boxes, "cls", None)).reshape(-1)
    if len(xyxy) == 0:
        return []

    keypoints = getattr(result, "keypoints", None)
    if keypoints is None:
        raise RuntimeError(
            "model returned boxes without keypoints; use custom pose weights"
        )
    coordinates = _to_numpy(getattr(keypoints, "xy", None))
    if coordinates.ndim != 3 or coordinates.shape[1:] != (4, 2):
        raise RuntimeError(
            "gate pose model must return exactly four (x, y) keypoints "
            f"per gate, received shape {coordinates.shape}"
        )
    point_confidences = _to_numpy(getattr(keypoints, "conf", None))
    if point_confidences.shape != coordinates.shape[:2]:
        data = _to_numpy(getattr(keypoints, "data", None))
        if data.ndim == 3 and data.shape[:2] == coordinates.shape[:2]:
            point_confidences = data[:, :, 2]
        else:
            point_confidences = np.ones(
                coordinates.shape[:2], dtype=np.float32
            )

    count = min(
        len(xyxy),
        len(box_confidences),
        len(classes),
        len(coordinates),
        len(point_confidences),
    )
    candidates = []
    for index in range(count):
        class_id = int(classes[index])
        if class_id not in gate_ids:
            continue
        bbox = tuple(float(value) for value in xyxy[index])
        points = np.asarray(coordinates[index], dtype=np.float32)
        confidences = np.asarray(
            point_confidences[index], dtype=np.float32
        ).reshape(4)
        if (
            not all(math.isfinite(value) for value in bbox)
            or not np.all(np.isfinite(points))
            or not np.all(np.isfinite(confidences))
        ):
            continue
        candidates.append(
            PoseGateCandidate(
                box=YoloGateBox(
                    bbox=bbox,
                    confidence=float(box_confidences[index]),
                    class_id=class_id,
                    label=names[class_id],
                    source_index=index,
                ),
                keypoints=points,
                keypoint_confidences=confidences,
            )
        )
    return candidates


def select_pose_target(
    candidates: Sequence[PoseGateCandidate],
    previous_target: Optional[YoloGateBox],
    frame_shape: tuple[int, ...],
    config: PoseGateConfig,
    *,
    lock_active: bool,
) -> Optional[PoseGateCandidate]:
    """Apply the shared largest-acquisition and identity-locking policy."""
    selected_box = select_target_gate(
        [candidate.box for candidate in candidates],
        previous_target,
        frame_shape,
        config,
        lock_active=lock_active,
    )
    if selected_box is None:
        return None
    for candidate in candidates:
        if candidate.box.source_index == selected_box.source_index:
            return replace(candidate, box=selected_box)
    return min(
        candidates,
        key=lambda candidate: math.hypot(
            candidate.box.center[0] - selected_box.center[0],
            candidate.box.center[1] - selected_box.center[1],
        ),
    )


def reliable_pose_corners(
    candidate: PoseGateCandidate,
    config: PoseGateConfig,
) -> tuple[Optional[InnerGateCorners], str]:
    """Validate the learned outer gate corners for orientation only."""
    confidences = candidate.keypoint_confidences
    if np.any(confidences < config.keypoint_confidence_threshold):
        return None, "low_keypoint_confidence"
    ordered = order_corners(candidate.keypoints)
    area = abs(float(cv2.contourArea(ordered.reshape(-1, 1, 2))))
    top_width = np.linalg.norm(ordered[1] - ordered[0])
    bottom_width = np.linalg.norm(ordered[2] - ordered[3])
    left_height = np.linalg.norm(ordered[3] - ordered[0])
    right_height = np.linalg.norm(ordered[2] - ordered[1])
    pose_width = 0.5 * (top_width + bottom_width)
    pose_height = 0.5 * (left_height + right_height)
    if area < config.minimum_opening_area_px:
        return None, "pose_area"
    if min(pose_width, pose_height) < config.minimum_opening_side_px:
        return None, "pose_side"
    corners = InnerGateCorners(
        top_left=tuple(float(value) for value in ordered[0]),
        top_right=tuple(float(value) for value in ordered[1]),
        bottom_right=tuple(float(value) for value in ordered[2]),
        bottom_left=tuple(float(value) for value in ordered[3]),
        reliable=True,
    )
    return corners, "outer_pose_orientation"


def draw_pose_debug_overlay(
    frame: np.ndarray,
    debug: Optional[PoseGateDebug],
) -> np.ndarray:
    """Draw all pose instances, selected keypoints, and center source."""
    if debug is None:
        return frame
    output = frame.copy()
    for candidate in debug.candidates:
        x1, y1, x2, y2 = (
            int(round(value)) for value in candidate.box.bbox
        )
        selected = (
            debug.selected is not None
            and candidate.box.source_index
            == debug.selected.box.source_index
        )
        color = (0, 255, 0) if selected else (255, 170, 0)
        thickness = 3 if selected else 1
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            output,
            f"gate {candidate.box.confidence:.2f}",
            (x1, max(14, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
        for index, (point, confidence) in enumerate(
            zip(candidate.keypoints, candidate.keypoint_confidences)
        ):
            if confidence <= 0.0:
                continue
            point_xy = tuple(int(round(value)) for value in point)
            cv2.circle(output, point_xy, 4 if selected else 2, color, -1)
            if selected:
                cv2.putText(
                    output,
                    f"k{index}:{confidence:.2f}",
                    (point_xy[0] + 4, point_xy[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.34,
                    color,
                    1,
                    cv2.LINE_AA,
                )
    if debug.corners is not None:
        polygon = np.round(
            debug.corners.as_polygon()
        ).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(output, [polygon], True, (255, 0, 255), 2)
        for label, point in (
            ("TL", debug.corners.top_left),
            ("TR", debug.corners.top_right),
            ("BR", debug.corners.bottom_right),
            ("BL", debug.corners.bottom_left),
        ):
            point_xy = tuple(int(round(value)) for value in point)
            cv2.putText(
                output,
                label,
                (point_xy[0] + 4, point_xy[1] + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )
    if debug.center is not None:
        center = tuple(int(round(value)) for value in debug.center)
        cv2.drawMarker(
            output, center, (0, 0, 255), cv2.MARKER_CROSS, 18, 2
        )
    cv2.putText(
        output,
        (
            f"POSE n={len(debug.candidates)} "
            f"source={debug.center_source} "
            f"keypoints={debug.keypoint_reason}"
        ),
        (10, max(18, output.shape[0] - 32)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


class YoloPoseGateDetector:
    """Drop-in detector using YOLO boxes for target center and scale."""

    def __init__(
        self,
        config: PoseGateConfig,
        model: Any = None,
    ):
        self.config = config
        self.model = model if model is not None else self._load_model()
        self.last_debug = DetectorDebug(
            raw_mask=np.zeros((1, 1), dtype=np.uint8),
            cleaned_mask=np.zeros((1, 1), dtype=np.uint8),
        )
        self.last_pose_debug = PoseGateDebug()
        self._previous_target: Optional[YoloGateBox] = None
        self._previous_valid_detection: Optional[GateDetection] = None
        self._lock_until = 0.0
        self._missing_frames = 0
        self._pending_target: Optional[YoloGateBox] = None
        self._pending_target_frames = 0
        self._last_log_at = -math.inf

    def _load_model(self):
        model_path = Path(self.config.model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"custom YOLO gate pose weights not found: {model_path}"
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics with YOLO pose support is required; run "
                "python -m pip install -r requirements.txt"
            ) from exc
        model = YOLO(str(model_path))
        task = getattr(model, "task", None)
        if task not in (None, "pose"):
            raise RuntimeError(
                f"expected pose weights, but model task is {task!r}"
            )
        return model

    def reset_target_lock(self) -> None:
        self._previous_target = None
        self._previous_valid_detection = None
        self._lock_until = 0.0
        self._missing_frames = 0
        self._pending_target = None
        self._pending_target_frames = 0

    @staticmethod
    def _same_acquisition_candidate(
        current: YoloGateBox,
        pending: YoloGateBox,
    ) -> bool:
        """Require a spatially consistent instance before starting a lock."""
        current_span = max(
            current.bbox[2] - current.bbox[0],
            current.bbox[3] - current.bbox[1],
            1.0,
        )
        pending_span = max(
            pending.bbox[2] - pending.bbox[0],
            pending.bbox[3] - pending.bbox[1],
            1.0,
        )
        center_distance = math.hypot(
            current.center[0] - pending.center[0],
            current.center[1] - pending.center[1],
        )
        area_ratio = current.area / max(pending.area, 1.0)
        return bool(
            center_distance <= 0.45 * max(current_span, pending_span)
            and 0.50 <= area_ratio <= 2.0
        )

    def _confirm_acquisition(
        self,
        selected: Optional[PoseGateCandidate],
    ) -> Optional[PoseGateCandidate]:
        """Suppress one-frame candidates before they can steer the vehicle."""
        if selected is None:
            self._pending_target = None
            self._pending_target_frames = 0
            return None
        required = max(1, int(self.config.acquisition_confirmation_frames))
        if self._pending_target is not None and self._same_acquisition_candidate(
            selected.box, self._pending_target
        ):
            self._pending_target_frames += 1
        else:
            self._pending_target = selected.box
            self._pending_target_frames = 1
        if self._pending_target_frames < required:
            return None
        self._pending_target = None
        self._pending_target_frames = 0
        return selected

    def _to_gate_detection(
        self,
        candidate: PoseGateCandidate,
        pose_corners: Optional[InnerGateCorners],
        frame_shape: tuple[int, ...],
        timestamp: float,
    ) -> GateDetection:
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = candidate.box.bbox
        bbox = (
            int(round(x1)),
            int(round(y1)),
            int(round(x2 - x1)),
            int(round(y2 - y1)),
        )
        center_x, center_y = candidate.box.center
        opening_scale = float(
            np.clip(self.config.estimated_opening_scale, 0.05, 1.0)
        )
        opening_width = max(1.0, (x2 - x1) * opening_scale)
        opening_height = max(1.0, (y2 - y1) * opening_scale)
        if pose_corners is not None:
            polygon = pose_corners.as_polygon()
            method = "yolo_pose_box_center"
            point_quality = float(
                np.mean(candidate.keypoint_confidences)
            )
            confidence = (
                0.70 * candidate.box.confidence + 0.30 * point_quality
            )
            top_edge = polygon[1] - polygon[0]
            angle = math.degrees(math.atan2(top_edge[1], top_edge[0]))
        else:
            method = "yolo_pose_box_center_no_orientation"
            confidence = 0.70 * candidate.box.confidence
            angle = 0.0
        normalized_x, normalized_y = normalized_image_coordinates(
            center_x, center_y, width, height
        )
        apparent_side = max(opening_width, opening_height, 1.0)
        distance = (
            self.config.focal_length_px
            * self.config.gate_inner_width_m
            / apparent_side
        )
        return GateDetection(
            found=True,
            center_x=float(center_x),
            center_y=float(center_y),
            normalized_x=normalized_x,
            normalized_y=normalized_y,
            opening_width=opening_width,
            opening_height=opening_height,
            apparent_area=opening_width * opening_height,
            angle_degrees=float(angle),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            # The learned points describe the outer gate, not the flyable
            # inner opening. Publishing them here would make gate_estimator
            # run physically invalid inner-square PnP.
            corners=None,
            method=method,
            corners_reliable=False,
            distance_m=float(distance),
            frame_width=width,
            frame_height=height,
            bbox=bbox,
            timestamp=timestamp,
        )

    def _publish_debug(
        self,
        frame_shape: tuple[int, ...],
        candidates: list[PoseGateCandidate],
        selected: Optional[PoseGateCandidate],
        corners: Optional[InnerGateCorners],
        center: Optional[tuple[float, float]],
        source: str,
        reason: str,
        timings: dict[str, float],
    ) -> None:
        height, width = frame_shape[:2]
        empty_mask = np.zeros((height, width), dtype=np.uint8)
        debug_candidates = []
        for candidate in candidates:
            x1, y1, x2, y2 = candidate.box.bbox
            contour = np.asarray(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                dtype=np.int32,
            ).reshape(-1, 1, 2)
            pose_corners, _ = reliable_pose_corners(
                candidate, self.config
            )
            debug_candidates.append(
                CandidateDebug(
                    outer_contour=contour,
                    opening_contour=(
                        pose_corners.as_polygon()
                        .astype(np.int32)
                        .reshape(-1, 1, 2)
                        if pose_corners is not None
                        else None
                    ),
                    accepted=True,
                    score=candidate.box.confidence,
                    confidence=candidate.box.confidence,
                    reason="yolo_pose_gate",
                    method="yolo_pose_gate",
                    center=candidate.box.center,
                    bbox=(
                        int(round(x1)),
                        int(round(y1)),
                        int(round(x2 - x1)),
                        int(round(y2 - y1)),
                    ),
                    features={
                        "supported_sides": (
                            4.0 if pose_corners is not None else 0.0
                        )
                    },
                )
            )
        selected_contour = None
        if selected is not None:
            x1, y1, x2, y2 = selected.box.bbox
            selected_contour = np.asarray(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                dtype=np.int32,
            ).reshape(-1, 1, 2)
        opening_contour = (
            corners.as_polygon().astype(np.int32).reshape(-1, 1, 2)
            if corners is not None
            else None
        )
        self.last_debug = DetectorDebug(
            raw_mask=empty_mask.copy(),
            cleaned_mask=empty_mask,
            candidates=debug_candidates,
            selected_contour=selected_contour,
            selected_opening_contour=opening_contour,
            raw_center=center,
            scale=1.0,
            timings_ms=timings,
        )
        self.last_pose_debug = PoseGateDebug(
            candidates=candidates,
            selected=selected,
            corners=corners,
            center=center,
            center_source=source,
            keypoint_reason=reason,
            missing_frames=self._missing_frames,
        )

    def _maybe_log(self, now: float) -> None:
        if now - self._last_log_at < self.config.log_interval_s:
            return
        debug = self.last_pose_debug
        selected_confidence = (
            "none"
            if debug.selected is None
            else f"{debug.selected.box.confidence:.2f}"
        )
        point_confidences = (
            "none"
            if debug.selected is None
            else np.array2string(
                debug.selected.keypoint_confidences,
                precision=2,
                separator=",",
            )
        )
        print(
            "[YOLO_POSE] "
            f"detections={len(debug.candidates)} "
            f"selected_conf={selected_confidence} "
            f"keypoint_conf={point_confidences} "
            f"keypoints={debug.keypoint_reason} "
            f"center={debug.center} "
            f"source={debug.center_source}",
            flush=True,
        )
        self._last_log_at = now

    def draw_debug_overlay(self, frame: np.ndarray) -> np.ndarray:
        return draw_pose_debug_overlay(frame, self.last_pose_debug)

    def detect(
        self,
        frame: np.ndarray,
        hint: Optional[GateDetection] = None,
        timestamp: Optional[float] = None,
    ) -> GateDetection:
        del hint
        started = time.perf_counter()
        now = time.monotonic() if timestamp is None else float(timestamp)
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            empty = GateDetection(timestamp=now)
            self._publish_debug(
                (1, 1, 3),
                [],
                None,
                None,
                None,
                "none",
                "empty_frame",
                {"total": 0.0},
            )
            return empty

        inference_started = time.perf_counter()
        candidates = detect_gate_poses(frame, self.model, self.config)
        inference_done = time.perf_counter()
        selected = select_pose_target(
            candidates,
            self._previous_target,
            frame.shape,
            self.config,
            lock_active=now <= self._lock_until,
        )
        acquiring = self._previous_target is None or now > self._lock_until
        if acquiring:
            selected = self._confirm_acquisition(selected)
            if selected is None:
                # An expired target must not remain a green steering target
                # while a different candidate is awaiting confirmation.
                self._previous_target = None
                self._previous_valid_detection = None
                self._lock_until = 0.0
        else:
            self._pending_target = None
            self._pending_target_frames = 0
        selection_done = time.perf_counter()

        corners = None
        center = None
        source = "none"
        reason = "no_yolo_pose_target"
        result = GateDetection(
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            timestamp=now,
        )
        if selected is not None:
            corners, reason = reliable_pose_corners(
                selected, self.config
            )
            source = "yolo_box_center"
            result = self._to_gate_detection(
                selected, corners, frame.shape, now
            )
            center = result.center_px
            self._previous_target = selected.box
            self._previous_valid_detection = result
            self._lock_until = now + self.config.target_lock_seconds
            self._missing_frames = 0
        else:
            self._missing_frames += 1
            if (
                self._previous_valid_detection is not None
                and self._missing_frames
                <= self.config.previous_center_frames
            ):
                result = replace(
                    self._previous_valid_detection,
                    predicted=True,
                    missing_frames=self._missing_frames,
                    method="yolo_pose_previous_fallback",
                    corners=None,
                    corners_reliable=False,
                    timestamp=now,
                )
                center = result.center_px
                source = "previous_frame_fallback"
                reason = "target_temporarily_missing"
            elif (
                now > self._lock_until
                and self._pending_target is None
            ):
                self.reset_target_lock()
        extraction_done = time.perf_counter()
        timings = {
            "pose_inference": (inference_done - inference_started) * 1000.0,
            "target_selection": (selection_done - inference_done) * 1000.0,
            "keypoint_validation": (
                extraction_done - selection_done
            ) * 1000.0,
            "total": (extraction_done - started) * 1000.0,
        }
        self._publish_debug(
            frame.shape,
            candidates,
            selected,
            corners,
            center,
            source,
            reason,
            timings,
        )
        self._maybe_log(now)
        return result
