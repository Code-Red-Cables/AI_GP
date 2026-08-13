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
    GateVisionConfig,
    OrangeGateDetector,
    normalized_image_coordinates,
    order_corners,
)
from .yolo_gate_detector import (
    InnerGateCorners,
    YoloGateBox,
    score_gate_candidate,
    select_target_gate,
)


@dataclass(frozen=True)
class PoseGateConfig:
    model_path: str = "models/gate_pose.pt"
    gate_class_name: str = "gate"
    confidence_threshold: float = 0.25
    keypoint_confidence_threshold: float = 0.25
    nms_iou_threshold: float = 0.70
    target_lock_seconds: float = 2.0
    persistent_target_lock: bool = True
    acquisition_confirmation_frames: int = 1
    require_hsv_confirmation: bool = False
    hsv_ranges: tuple[
        tuple[tuple[int, int, int], tuple[int, int, int]], ...
    ] = (((3, 105, 180), (17, 255, 255)),)
    hsv_min_orange_ratio: float = 0.12
    hsv_max_orange_ratio: float = 0.72
    hsv_side_band_fraction: float = 0.28
    hsv_min_side_density: float = 0.10
    hsv_min_supported_sides: int = 3
    minimum_gate_area_px: float = 400.0
    maximum_outside_fraction: float = 0.35
    minimum_opening_area_px: float = 64.0
    minimum_opening_side_px: float = 8.0
    previous_center_frames: int = 5
    estimated_opening_scale: float = 0.72
    inference_size: int = 640
    device: Optional[str] = None
    log_interval_s: float = 1.0
    score_confidence_weight: float = 0.40
    score_center_weight: float = 0.30
    score_area_weight: float = 0.30
    score_reference_area_ratio: float = 0.08
    target_association_center_span: float = 1.85
    target_association_min_area_ratio: float = 0.45
    target_association_max_area_ratio: float = 4.0
    post_pass_rejection_seconds: float = 0.0
    post_pass_max_area_ratio: float = 1.0
    hsv_blur_kernel: int = 5
    hsv_opening_kernel: int = 3
    hsv_closing_kernel: int = 5
    hsv_center_blend: float = 0.25
    hsv_center_max_shift_fraction: float = 0.12
    global_hsv_fallback_enabled: bool = False
    global_hsv_fallback_confidence_scale: float = 0.55
    # Also run the colour fallback while a target lock is alive but this frame
    # produced nothing. Without it the fallback only ever fires pre-acquisition.
    global_hsv_fallback_during_lock: bool = True
    gate_inner_width_m: float = cm.GATE_INNER_M
    focal_length_px: float = cm.FX


# The pose model labels the 2.7 m outer ring as keypoints 0-3 and the 1.5 m
# opening as 4-7, each clockwise from top-left. Older four-keypoint weights
# carry the outer ring alone, so both shapes stay loadable.
SUPPORTED_KEYPOINT_COUNTS = (4, 8)
OUTER_RING_IDX = (0, 1, 2, 3)


@dataclass(frozen=True)
class PoseGateCandidate:
    box: YoloGateBox
    keypoints: np.ndarray
    keypoint_confidences: np.ndarray
    hsv_confirmed: bool = True
    hsv_orange_ratio: float = 1.0
    hsv_supported_sides: int = 4
    hsv_refined_center: Optional[tuple[float, float]] = None
    hsv_geometry_score: float = 0.0

    @property
    def keypoint_center(self) -> tuple[float, float]:
        points = np.asarray(self.keypoints, dtype=np.float32).reshape(-1, 2)
        if len(points) == 0:
            return self.box.center
        center = points.mean(axis=0)
        return float(center[0]), float(center[1])

    @property
    def outer_keypoints(self) -> np.ndarray:
        """The four outer corners, whichever ring count the model returned."""
        points = np.asarray(self.keypoints, dtype=np.float32).reshape(-1, 2)
        return points[list(OUTER_RING_IDX)]

    @property
    def outer_keypoint_confidences(self) -> np.ndarray:
        conf = np.asarray(
            self.keypoint_confidences, dtype=np.float32
        ).reshape(-1)
        return conf[list(OUTER_RING_IDX)]


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
    if (
        coordinates.ndim != 3
        or coordinates.shape[2] != 2
        or coordinates.shape[1] not in SUPPORTED_KEYPOINT_COUNTS
    ):
        raise RuntimeError(
            "gate pose model must return four (outer ring) or eight "
            "(outer + inner ring) (x, y) keypoints per gate, received "
            f"shape {coordinates.shape}"
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
        ).reshape(-1)
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


def build_pose_orange_mask(
    frame: np.ndarray,
    config: PoseGateConfig,
) -> np.ndarray:
    """Build the calibrated orange mask used to confirm YOLO proposals."""
    blur_kernel = max(1, int(config.hsv_blur_kernel))
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    filtered = (
        cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)
        if blur_kernel > 1
        else frame
    )
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for lower, upper in config.hsv_ranges:
        mask = cv2.bitwise_or(
            mask,
            cv2.inRange(
                hsv,
                np.asarray(lower, dtype=np.uint8),
                np.asarray(upper, dtype=np.uint8),
            ),
        )
    for operation, size in (
        (cv2.MORPH_OPEN, config.hsv_opening_kernel),
        (cv2.MORPH_CLOSE, config.hsv_closing_kernel),
    ):
        size = max(1, int(size))
        if size > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (size, size)
            )
            mask = cv2.morphologyEx(mask, operation, kernel, iterations=1)
    return mask


def confirm_pose_candidates_with_hsv(
    candidates: Sequence[PoseGateCandidate],
    orange_mask: np.ndarray,
    config: PoseGateConfig,
) -> list[PoseGateCandidate]:
    """Require orange frame support inside each proposed YOLO gate box."""
    if not config.require_hsv_confirmation:
        return list(candidates)
    frame_height, frame_width = orange_mask.shape[:2]
    confirmed = []
    for candidate in candidates:
        x1, y1, x2, y2 = candidate.box.bbox
        left = max(0, int(math.floor(x1)))
        top = max(0, int(math.floor(y1)))
        right = min(frame_width, int(math.ceil(x2)))
        bottom = min(frame_height, int(math.ceil(y2)))
        crop = orange_mask[top:bottom, left:right]
        if crop.size == 0:
            confirmed.append(
                replace(
                    candidate,
                    hsv_confirmed=False,
                    hsv_orange_ratio=0.0,
                    hsv_supported_sides=0,
                )
            )
            continue
        height, width = crop.shape
        band_fraction = float(
            np.clip(config.hsv_side_band_fraction, 0.05, 0.50)
        )
        band_x = max(1, int(round(width * band_fraction)))
        band_y = max(1, int(round(height * band_fraction)))
        side_regions = (
            crop[:band_y, :],
            crop[-band_y:, :],
            crop[:, :band_x],
            crop[:, -band_x:],
        )
        side_densities = [
            float(np.count_nonzero(region)) / max(region.size, 1)
            for region in side_regions
        ]
        supported_sides = sum(
            density >= config.hsv_min_side_density
            for density in side_densities
        )
        orange_ratio = float(np.count_nonzero(crop)) / crop.size
        orange_y, orange_x = np.nonzero(crop)
        refined_center = None
        geometry_score = 0.0
        if orange_x.size >= 20:
            x_low, x_high = np.percentile(orange_x, (2.0, 98.0))
            y_low, y_high = np.percentile(orange_y, (2.0, 98.0))
            span_x = max(float(x_high - x_low), 1.0)
            span_y = max(float(y_high - y_low), 1.0)
            refined_center = (
                float(left + 0.5 * (x_low + x_high)),
                float(top + 0.5 * (y_low + y_high)),
            )
            aspect_quality = min(span_x, span_y) / max(span_x, span_y)
            geometry_score = float(
                np.clip(
                    0.65 * aspect_quality
                    + 0.35 * supported_sides / 4.0,
                    0.0,
                    1.0,
                )
            )
        is_confirmed = bool(
            config.hsv_min_orange_ratio
            <= orange_ratio
            <= config.hsv_max_orange_ratio
            and supported_sides >= config.hsv_min_supported_sides
        )
        confirmed.append(
            replace(
                candidate,
                hsv_confirmed=is_confirmed,
                hsv_orange_ratio=orange_ratio,
                hsv_supported_sides=supported_sides,
                hsv_refined_center=refined_center,
                hsv_geometry_score=geometry_score,
            )
        )
    return confirmed


def select_pose_target(
    candidates: Sequence[PoseGateCandidate],
    previous_target: Optional[YoloGateBox],
    frame_shape: tuple[int, ...],
    config: PoseGateConfig,
    *,
    lock_active: bool,
) -> Optional[PoseGateCandidate]:
    """Apply the shared largest-acquisition and identity-locking policy."""
    selection_boxes = []
    for candidate in candidates:
        if config.require_hsv_confirmation:
            hsv_quality = float(
                np.clip(
                    0.65 * candidate.hsv_geometry_score
                    + 0.35 * candidate.hsv_supported_sides / 4.0,
                    0.0,
                    1.0,
                )
            )
            confidence = float(
                np.clip(
                    0.80 * candidate.box.confidence + 0.20 * hsv_quality,
                    0.0,
                    1.0,
                )
            )
            selection_boxes.append(
                replace(candidate.box, confidence=confidence)
            )
        else:
            selection_boxes.append(candidate.box)
    selected_box = select_target_gate(
        selection_boxes,
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
    # Orientation has always come from the outer ring; the inner ring feeds
    # PnP only, so this gate keeps judging exactly the same four points.
    confidences = candidate.outer_keypoint_confidences
    if np.any(confidences < config.keypoint_confidence_threshold):
        return None, "low_keypoint_confidence"
    ordered = order_corners(candidate.outer_keypoints)
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
    """Draw outer + inner keypoints only — no boxes or corner outlines."""
    if debug is None:
        return frame
    output = frame.copy()
    # Outer ring = cyan, inner ring = magenta. Dimmer for non-selected gates.
    outer_selected = (255, 255, 0)
    inner_selected = (255, 0, 255)
    outer_other = (160, 160, 0)
    inner_other = (160, 0, 160)
    for candidate in debug.candidates:
        selected = (
            debug.selected is not None
            and candidate.box.source_index
            == debug.selected.box.source_index
        )
        points = np.asarray(candidate.keypoints, dtype=np.float32).reshape(
            -1, 2
        )
        confidences = np.asarray(
            candidate.keypoint_confidences, dtype=np.float32
        ).reshape(-1)
        for index, (point, confidence) in enumerate(zip(points, confidences)):
            if (
                confidence <= 0.0
                or not np.isfinite(point).all()
                or (point[0] == 0.0 and point[1] == 0.0)
            ):
                continue
            inner = index >= 4
            color = (
                (inner_selected if inner else outer_selected)
                if selected
                else (inner_other if inner else outer_other)
            )
            point_xy = (int(round(point[0])), int(round(point[1])))
            radius = 5 if selected else 3
            cv2.circle(output, point_xy, radius, color, -1, cv2.LINE_AA)
            if selected:
                cv2.putText(
                    output,
                    f"{'i' if inner else 'o'}{index % 4}",
                    (point_xy[0] + 4, point_xy[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.34,
                    color,
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
            f"cyan=outer magenta=inner "
            f"src={debug.center_source}"
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
        model_was_injected = model is not None
        self.model = model if model_was_injected else self._load_model()
        if not model_was_injected:
            self._warm_up_model()
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
        self._last_association_target: Optional[YoloGateBox] = None
        self._post_pass_rejection_until = -math.inf
        self._skip_confirm_until = -math.inf
        self._prefer_nx: Optional[float] = None
        self._global_hsv_detector = OrangeGateDetector(
            GateVisionConfig(
                hsv_ranges=config.hsv_ranges,
                blur_kernel=config.hsv_blur_kernel,
                opening_kernel_size=config.hsv_opening_kernel,
                closing_kernel_size=config.hsv_closing_kernel,
                min_contour_area=config.minimum_opening_area_px,
            )
        )

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

    def _warm_up_model(self) -> None:
        """Pay the backend's first-inference cost before flight can arm.

        The device and the warm timing are printed because a CPU fallback is
        otherwise silent and looks like nothing worse than a sluggish log.
        It is not: the simulator emits 30 fps, CPU inference sustains about
        4, and the drone then flies the gaps blind. That is invisible while
        Cheat Engine holds the sim at 0.2x, where 4 fps of wall clock is a
        comfortable 20 fps of simulated time, and it only bites at race speed.
        """
        print("[VISION] warming YOLO pose inference...", flush=True)
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        detect_gate_poses(blank, self.model, self.config)
        started = time.perf_counter()
        detect_gate_poses(blank, self.model, self.config)
        warm_ms = (time.perf_counter() - started) * 1000.0

        device = self.config.device or "auto"
        try:
            device = str(next(self.model.model.parameters()).device)
        except (AttributeError, StopIteration, TypeError):
            pass
        note = "" if warm_ms <= 60.0 else "  <-- too slow for 30 fps vision"
        print(
            f"[VISION] YOLO pose inference ready on {device} "
            f"({warm_ms:.0f} ms/frame){note}",
            flush=True,
        )

    def reset_target_lock(self) -> None:
        self._previous_target = None
        self._previous_valid_detection = None
        self._lock_until = 0.0
        self._missing_frames = 0
        self._pending_target = None
        self._pending_target_frames = 0
        self._last_association_target = None

    def reset_episode(self) -> None:
        """Clear lock + post-pass filter after a sim/crash reset.

        0855: post_pass_rejection_until survived floor resets, so max_area
        kept wiping the restart first-gate (~80–180 px) and association stuck
        on a stale edge remnant → selected_conf=none for whole attempts.
        """
        self.reset_target_lock()
        self._post_pass_rejection_until = -math.inf
        self._prefer_nx = None
        self._skip_confirm_until = time.monotonic() + 3.0
        print('[VISION] YOLO episode reset (post-pass/lock cleared)', flush=True)

    def set_prefer_horizontal(self, nx: Optional[float]) -> None:
        """Bias post-pass acquisition toward a latched course bearing."""
        if nx is None or not math.isfinite(float(nx)):
            self._prefer_nx = None
            return
        self._prefer_nx = float(np.clip(nx, -1.0, 1.0))

    def begin_next_gate_acquisition(self, timestamp: float) -> None:
        """Reset identity and briefly reject the just-passed gate remnant."""
        self.reset_target_lock()
        self._post_pass_rejection_until = (
            float(timestamp)
            + max(0.0, self.config.post_pass_rejection_seconds)
        )
        self._skip_confirm_until = float(timestamp) + 2.0

    def _target_from_hint(
        self,
        hint: Optional[GateDetection],
    ) -> Optional[YoloGateBox]:
        """Translate a tracker prediction into an association-only YOLO box."""
        if hint is None or not hint.found:
            return None
        _, _, bbox_width, bbox_height = hint.bbox
        if bbox_width <= 0 or bbox_height <= 0:
            return None
        half_width = 0.5 * float(bbox_width)
        half_height = 0.5 * float(bbox_height)
        return YoloGateBox(
            bbox=(
                float(hint.center_x) - half_width,
                float(hint.center_y) - half_height,
                float(hint.center_x) + half_width,
                float(hint.center_y) + half_height,
            ),
            confidence=float(hint.confidence),
            label=self.config.gate_class_name,
        )

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
        center_source = "yolo_box_center"
        if (
            pose_corners is not None
            and
            self.config.require_hsv_confirmation
            and candidate.hsv_confirmed
            and candidate.hsv_refined_center is not None
        ):
            refined_x, refined_y = candidate.hsv_refined_center
            box_diagonal = max(math.hypot(x2 - x1, y2 - y1), 1.0)
            shift_fraction = (
                math.hypot(refined_x - center_x, refined_y - center_y)
                / box_diagonal
            )
            if shift_fraction <= self.config.hsv_center_max_shift_fraction:
                blend = float(np.clip(self.config.hsv_center_blend, 0.0, 1.0))
                center_x = (1.0 - blend) * center_x + blend * refined_x
                center_y = (1.0 - blend) * center_y + blend * refined_y
                if blend > 0.0:
                    center_source = "hsv_refined_center"
        opening_scale = float(
            np.clip(self.config.estimated_opening_scale, 0.05, 1.0)
        )
        opening_width = max(1.0, (x2 - x1) * opening_scale)
        opening_height = max(1.0, (y2 - y1) * opening_scale)
        if pose_corners is not None:
            polygon = pose_corners.as_polygon()
            method = (
                (
                    "yolo_pose_hsv_refined_center"
                    if center_source == "hsv_refined_center"
                    else "yolo_pose_hsv_box_center"
                )
                if self.config.require_hsv_confirmation
                else "yolo_pose_box_center"
            )
            point_quality = float(
                np.mean(candidate.keypoint_confidences)
            )
            if self.config.require_hsv_confirmation:
                confidence = (
                    0.65 * candidate.box.confidence
                    + 0.20 * point_quality
                    + 0.15 * candidate.hsv_geometry_score
                )
            else:
                confidence = (
                    0.70 * candidate.box.confidence + 0.30 * point_quality
                )
            top_edge = polygon[1] - polygon[0]
            angle = math.degrees(math.atan2(top_edge[1], top_edge[0]))
        else:
            method = (
                (
                    "yolo_pose_hsv_refined_center_no_orientation"
                    if center_source == "hsv_refined_center"
                    else "yolo_pose_hsv_box_center_no_orientation"
                )
                if self.config.require_hsv_confirmation
                else "yolo_pose_box_center_no_orientation"
            )
            # Keep the YOLO box center when orientation is unavailable. The
            # 21:29 Training Two trace changed from x=505 to x=553 on a
            # no-orientation HSV center, reversing bank and losing gate two.
            # Missing keypoints still must not demote an otherwise
            # YOLO+HSV-confirmed gate below navigation thresholds.
            confidence = (
                0.80 * candidate.box.confidence
                + 0.20 * candidate.hsv_geometry_score
                if self.config.require_hsv_confirmation
                else candidate.box.confidence
            )
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
        orange_mask: Optional[np.ndarray],
        candidates: list[PoseGateCandidate],
        selected: Optional[PoseGateCandidate],
        corners: Optional[InnerGateCorners],
        center: Optional[tuple[float, float]],
        source: str,
        reason: str,
        timings: dict[str, float],
    ) -> None:
        height, width = frame_shape[:2]
        if orange_mask is None or orange_mask.shape != (height, width):
            orange_mask = np.zeros((height, width), dtype=np.uint8)
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
                    accepted=candidate.hsv_confirmed,
                    score=candidate.box.confidence,
                    confidence=candidate.box.confidence,
                    reason=(
                        "yolo_pose_hsv_confirmed"
                        if candidate.hsv_confirmed
                        else "yolo_pose_hsv_rejected"
                    ),
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
                            float(candidate.hsv_supported_sides)
                        ),
                        "orange_ratio": candidate.hsv_orange_ratio,
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
            raw_mask=orange_mask.copy(),
            cleaned_mask=orange_mask.copy(),
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
        confirmed_count = sum(
            candidate.hsv_confirmed for candidate in debug.candidates
        )
        selected_hsv = (
            "none"
            if debug.selected is None
            else (
                f"{debug.selected.hsv_orange_ratio:.2f}/"
                f"{debug.selected.hsv_supported_sides}"
            )
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
        association = self._last_association_target
        association_text = (
            "none"
            if association is None
            else (
                f"({association.center[0]:.1f},{association.center[1]:.1f};"
                f"{association.bbox[2] - association.bbox[0]:.1f}x"
                f"{association.bbox[3] - association.bbox[1]:.1f})"
            )
        )
        candidate_text = ";".join(
            (
                f"{candidate.box.source_index}:"
                f"({candidate.box.center[0]:.1f},{candidate.box.center[1]:.1f};"
                f"{candidate.box.bbox[2] - candidate.box.bbox[0]:.1f}x"
                f"{candidate.box.bbox[3] - candidate.box.bbox[1]:.1f};"
                f"hsv={int(candidate.hsv_confirmed)})"
            )
            for candidate in debug.candidates
        ) or "none"
        print(
            "[YOLO_POSE] "
            f"detections={len(debug.candidates)} "
            f"hsv_confirmed={confirmed_count} "
            f"selected_conf={selected_confidence} "
            f"selected_hsv={selected_hsv} "
            f"keypoint_conf={point_confidences} "
            f"keypoints={debug.keypoint_reason} "
            f"center={debug.center} "
            f"source={debug.center_source} "
            f"association={association_text} "
            f"candidates={candidate_text}",
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
        started = time.perf_counter()
        now = time.monotonic() if timestamp is None else float(timestamp)
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            empty = GateDetection(timestamp=now)
            self._publish_debug(
                (1, 1, 3),
                None,
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
        orange_mask = None
        if self.config.require_hsv_confirmation:
            orange_mask = build_pose_orange_mask(frame, self.config)
            candidates = confirm_pose_candidates_with_hsv(
                candidates, orange_mask, self.config
            )
            eligible_candidates = [
                candidate
                for candidate in candidates
                if candidate.hsv_confirmed
            ]
        else:
            # YOLO-only: trust pose instances; skip HSV mask entirely.
            eligible_candidates = list(candidates)
        hsv_done = time.perf_counter()
        post_pass = now <= self._post_pass_rejection_until
        if post_pass:
            frame_area = max(float(frame.shape[0] * frame.shape[1]), 1.0)
            maximum_area = (
                float(
                    np.clip(
                        self.config.post_pass_max_area_ratio,
                        0.0,
                        1.0,
                    )
                )
                * frame_area
            )
            # 0902: next gate often ~650–2200 px right after punch; 700 still
            # dropped the bearing-side box so only right remnants remained.
            minimum_area = max(500.0, 0.002 * frame_area)
            eligible_candidates = [
                candidate
                for candidate in eligible_candidates
                if minimum_area <= candidate.box.area <= maximum_area
            ]
            # 0750: reject flat bottom-bar instances that steal lock at v≈330.
        frame_h = float(frame.shape[0])
        filtered = []
        for candidate in eligible_candidates:
            box = candidate.box
            x1, y1, x2, y2 = box.bbox
            bw = max(1.0, float(x2 - x1))
            bh = max(1.0, float(y2 - y1))
            cy = 0.5 * (float(y1) + float(y2))
            if bw / bh > 2.8 and cy > 0.70 * frame_h:
                continue
            if cy > 0.88 * frame_h and box.area < 0.45 * frame_h * float(
                frame.shape[1]
            ):
                continue
            # 0908: post-pass chase of v≈297–315 floor junk → dive.
            # 100134: prefer_nx bot_frac=0.94 let cy≈337 floor box through
            # (never acquired real gate 2). Cap at 0.82 even when hunting right.
            bot_frac = 0.82 if post_pass else 0.88
            if post_pass and cy > bot_frac * frame_h:
                continue
            filtered.append(candidate)
        if filtered:
            eligible_candidates = filtered
        association_target = None if post_pass else self._previous_target
        # Drop FOV-edge sticky locks so a centered gate can reacquire
        # (phase5 174620: lock stuck at y≈5 after climb-out).
        # Do NOT wipe a large approaching gate at the top of frame — that
        # cleared persistent lock mid-approach and reacquired a far gate.
        if association_target is not None and not post_pass:
            try:
                ay = float(association_target.center[1])
                ah = float(frame.shape[0])
                aw = float(frame.shape[1])
                area = float(association_target.area)
                frame_area = max(ah * aw, 1.0)
                large_close = area >= 0.012 * frame_area  # ~2.7k @ 640x360
                top_junk = ay < 0.10 * ah and not large_close
                # 100134: sticky assoc at y≈350 blocked all next-gate picks.
                bot_junk = ay > 0.82 * ah
                if top_junk or bot_junk:
                    association_target = None
                    self._previous_target = None
                    self._previous_valid_detection = None
                    self._lock_until = 0.0
            except (TypeError, ValueError, AttributeError, IndexError):
                pass
        # Recompute after FOV wipe so we do not acquire freely with a
        # stale lock_active=True and previous=None.
        persistent_lock = bool(
            self.config.persistent_target_lock
            and self._previous_target is not None
            and not post_pass
        )
        # 0928: sticky near-course lock after pass so we don't drop 378→599
        # when the box drifts a few pixels lower.
        if (
            post_pass
            and self._previous_target is not None
            and self._prefer_nx is not None
        ):
            prev_nx = (
                float(self._previous_target.center[0]) - 0.5 * float(frame.shape[1])
            ) / (0.5 * float(frame.shape[1]))
            if (
                abs(prev_nx - float(self._prefer_nx)) <= 0.45
                and abs(prev_nx) <= 0.55
            ):
                association_target = self._previous_target
        if persistent_lock:
            predicted_target = self._target_from_hint(hint)
            if predicted_target is not None:
                association_target = predicted_target
        self._last_association_target = association_target
        # Acquisition uses max(area) which after a pass locks onto the largest
        # side remnant (u≈600). During post-pass, pick by center-weighted score
        # and heavily down-weight FOV-edge remnants (0855: locked u≈623).
        if post_pass and eligible_candidates:
            frame_w = float(frame.shape[1])
            prefer = self._prefer_nx

            def _cand_nx(candidate: PoseGateCandidate) -> float:
                return (float(candidate.box.center[0]) - 0.5 * frame_w) / (
                    0.5 * frame_w
                )

            def _post_pass_score(candidate: PoseGateCandidate) -> float:
                score = score_gate_candidate(
                    candidate.box, frame.shape, self.config
                )
                nx = _cand_nx(candidate)
                if abs(nx) > 0.85:
                    score *= 0.08
                elif abs(nx) > 0.70:
                    score *= 0.35
                elif abs(nx) > 0.55:
                    score *= 0.70
                # 0912: prefer_nx over-weighted a small bearing-side speck
                # over a larger near-center next gate (~6k). Same-side + area.
                # 0928: edge remnant 562 (~6.8k) beat next gate 388 (~0.7k)
                # despite prefer=+0.22 — angular match must dominate size.
                if prefer is not None:
                    # When hunting right, do not treat near-center-left as
                    # "same" (0929: u≈301 beat u≈518 with abs(nx)<0.12).
                    if float(prefer) >= 0.18:
                        same = float(nx) >= -0.05
                    else:
                        same = (
                            abs(float(nx)) < 0.20
                            or float(nx) * float(prefer) >= 0.0
                        )
                    if not same:
                        score *= 0.08
                    else:
                        score *= float(
                            np.exp(-3.0 * abs(float(nx) - float(prefer)))
                        )
                        score *= 0.55 + 0.45 * min(
                            float(candidate.box.area) / 6000.0, 1.0
                        )
                # Prefer sticky association when still near-course — but do
                # not let a drifted center lock beat a larger right grower
                # (0930: u≈324 stole over u≈526 while prefer=+0.28).
                if association_target is not None:
                    try:
                        dx = float(candidate.box.center[0]) - float(
                            association_target.center[0]
                        )
                        dy = float(candidate.box.center[1]) - float(
                            association_target.center[1]
                        )
                        dist = math.hypot(dx, dy)
                        if dist < 80.0:
                            if (
                                prefer is not None
                                and float(prefer) >= 0.18
                                and float(nx) < 0.10
                            ):
                                score *= 0.85
                            else:
                                score *= 1.8
                    except (TypeError, ValueError, AttributeError):
                        pass
                return score

            pool = list(eligible_candidates)
            if prefer is not None:
                # 0929: next gate at u≈504 (nx≈+0.58) with prefer=+0.32 was
                # excluded by abs(nx)<=0.55 — only edge junk remained.
                nx_lim = 0.75 if abs(float(prefer)) >= 0.18 else 0.55
                near = [
                    c
                    for c in eligible_candidates
                    if abs(_cand_nx(c) - float(prefer)) <= 0.45
                    and abs(_cand_nx(c)) <= nx_lim
                    and float(c.box.area) >= 400.0
                    # Prefer side: drop opposite / near-left when hunting right.
                    and (
                        (
                            float(prefer) >= 0.18
                            and _cand_nx(c) >= -0.05
                        )
                        or (
                            float(prefer) < 0.18
                            and (
                                float(prefer) * _cand_nx(c) >= 0.0
                                or abs(_cand_nx(c)) < 0.12
                            )
                        )
                    )
                ]
                if near:
                    pool = near
            selected = max(pool, key=_post_pass_score)
        else:
            selected = select_pose_target(
                eligible_candidates,
                association_target,
                frame.shape,
                self.config,
                lock_active=persistent_lock or now <= self._lock_until,
            )
        acquiring = self._previous_target is None or (
            not persistent_lock and now > self._lock_until
        )
        # 0850/0859: skip confirm during post-pass and right after episode
        # reset so restart first-gate / next-gate locks aren't delayed.
        skip_confirm = bool(
            post_pass or now <= self._skip_confirm_until
        )
        if acquiring and not skip_confirm:
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
            result = self._to_gate_detection(
                selected, corners, frame.shape, now
            )
            center = result.center_px
            source = (
                "hsv_refined_center"
                if "hsv_refined_center" in result.method
                else "yolo_box_center"
            )
            self._previous_target = selected.box
            self._previous_valid_detection = result
            self._lock_until = now + self.config.target_lock_seconds
            self._missing_frames = 0
        else:
            self._missing_frames += 1
            if (
                now > self._post_pass_rejection_until
                and self._previous_valid_detection is not None
                and self._missing_frames
                <= self.config.previous_center_frames
            ):
                # 0719: post-pass previous_frame_fallback stuck at u≈75/600.
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
                and not persistent_lock
            ):
                self.reset_target_lock()
        # The colour fallback used to require that no target had ever been
        # locked, which meant it fired only before acquisition and never during
        # a race — precisely when it is wanted. It now also runs while a lock
        # exists but this frame produced nothing, which is what a roll past 45
        # degrees does. Confidence is still scaled down, so a colour hit never
        # outranks a real YOLO detection downstream.
        if (
            not result.found
            and self.config.global_hsv_fallback_enabled
            and (
                self.config.global_hsv_fallback_during_lock
                or (
                    self._previous_target is None
                    and self._pending_target is None
                )
            )
        ):
            fallback = self._global_hsv_detector.detect(
                frame,
                hint=None,
                timestamp=now,
            )
            if fallback.found:
                result = replace(
                    fallback,
                    confidence=float(
                        np.clip(
                            fallback.confidence
                            * self.config.global_hsv_fallback_confidence_scale,
                            0.0,
                            1.0,
                        )
                    ),
                    method="global_hsv_fallback",
                )
                center = result.center_px
                source = "global_hsv_fallback"
                reason = "yolo_missing_global_hsv"
        extraction_done = time.perf_counter()
        timings = {
            "pose_inference": (inference_done - inference_started) * 1000.0,
            "hsv_confirmation": (hsv_done - inference_done) * 1000.0,
            "target_selection": (selection_done - hsv_done) * 1000.0,
            "keypoint_validation": (
                extraction_done - selection_done
            ) * 1000.0,
            "total": (extraction_done - started) * 1000.0,
        }
        self._publish_debug(
            frame.shape,
            orange_mask,
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
