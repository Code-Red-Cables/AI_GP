"""YOLO target isolation followed by crop-local orange opening extraction.

The custom detector owns only perception. It returns the established
``GateDetection`` contract, so tracking, navigation, planner mapping, MAVLink,
and the rate controller remain unchanged.
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


@dataclass(frozen=True)
class HybridGateConfig:
    model_path: str = "models/gate_detector.pt"
    gate_class_name: str = "gate"
    confidence_threshold: float = 0.35
    nms_iou_threshold: float = 0.70
    target_lock_seconds: float = 0.75
    crop_padding_px: int = 14
    minimum_gate_area_px: float = 400.0
    maximum_outside_fraction: float = 0.35
    previous_center_frames: int = 5
    inference_size: int = 640
    device: Optional[str] = None
    log_interval_s: float = 1.0
    score_confidence_weight: float = 0.40
    score_center_weight: float = 0.30
    score_area_weight: float = 0.30
    score_reference_area_ratio: float = 0.08
    target_association_center_span: float = 0.75
    target_association_min_area_ratio: float = 0.0
    target_association_max_area_ratio: float = math.inf
    minimum_opening_area_px: float = 45.0
    minimum_opening_side_px: float = 12.0
    minimum_opening_aspect: float = 0.45
    maximum_opening_aspect: float = 2.20
    maximum_bright_neutral_fraction: float = 0.55
    opening_kernel_size: int = 3
    closing_kernel_size: int = 3
    hsv_ranges: tuple[
        tuple[tuple[int, int, int], tuple[int, int, int]], ...
    ] = (((3, 105, 180), (17, 255, 255)),)
    gate_inner_width_m: float = cm.GATE_INNER_M
    focal_length_px: float = cm.FX


@dataclass(frozen=True)
class YoloGateBox:
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int = 0
    label: str = "gate"
    source_index: int = -1

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass(frozen=True)
class InnerGateCorners:
    top_left: tuple[float, float]
    top_right: tuple[float, float]
    bottom_left: tuple[float, float]
    bottom_right: tuple[float, float]
    reliable: bool = False

    def as_polygon(self) -> np.ndarray:
        """Return cyclic TL, TR, BR, BL points for OpenCV consumers."""
        return np.asarray(
            [
                self.top_left,
                self.top_right,
                self.bottom_right,
                self.bottom_left,
            ],
            dtype=np.float32,
        )


@dataclass
class HybridGateDebug:
    detections: list[YoloGateBox] = field(default_factory=list)
    selected: Optional[YoloGateBox] = None
    crop_bbox: Optional[tuple[int, int, int, int]] = None
    crop_mask: Optional[np.ndarray] = None
    corners: Optional[InnerGateCorners] = None
    center: Optional[tuple[float, float]] = None
    center_source: str = "none"
    extraction_reason: str = "no_target"
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


def detect_gates_yolo(
    frame: np.ndarray,
    model: Any,
    config: HybridGateConfig,
) -> list[YoloGateBox]:
    """Run one custom YOLO inference with overlapping-box preservation."""
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
            "YOLO weights do not define the required custom class "
            f"{config.gate_class_name!r}; classes are {available}. "
            "Do not use generic COCO weights for racing-gate inference."
        )

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = _to_numpy(getattr(boxes, "xyxy", None)).reshape(-1, 4)
    confidences = _to_numpy(getattr(boxes, "conf", None)).reshape(-1)
    classes = _to_numpy(getattr(boxes, "cls", None)).reshape(-1)
    count = min(len(xyxy), len(confidences), len(classes))
    detections = []
    for index in range(count):
        class_id = int(classes[index])
        if class_id not in gate_ids:
            continue
        bbox = tuple(float(value) for value in xyxy[index])
        if not all(math.isfinite(value) for value in bbox):
            continue
        detections.append(
            YoloGateBox(
                bbox=bbox,
                confidence=float(confidences[index]),
                class_id=class_id,
                label=names[class_id],
                source_index=index,
            )
        )
    return detections


def _inside_box(
    detection: YoloGateBox,
    frame_shape: tuple[int, ...],
) -> tuple[YoloGateBox, float]:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = detection.bbox
    original_area = detection.area
    clipped = (
        float(np.clip(x1, 0, width)),
        float(np.clip(y1, 0, height)),
        float(np.clip(x2, 0, width)),
        float(np.clip(y2, 0, height)),
    )
    clipped_detection = replace(detection, bbox=clipped)
    inside_fraction = clipped_detection.area / max(original_area, 1e-6)
    return clipped_detection, inside_fraction


def _bbox_iou(left: YoloGateBox, right: YoloGateBox) -> float:
    lx1, ly1, lx2, ly2 = left.bbox
    rx1, ry1, rx2, ry2 = right.bbox
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    union = left.area + right.area - intersection
    return intersection / max(union, 1e-6)


def score_gate_candidate(
    detection: YoloGateBox,
    frame_shape: tuple[int, ...],
    config: HybridGateConfig,
) -> float:
    """Score a valid gate without making raw orange/box area authoritative."""
    frame_height, frame_width = frame_shape[:2]
    frame_area = max(float(frame_height * frame_width), 1.0)
    center_x, center_y = detection.center
    normalized_x = (center_x - frame_width / 2.0) / max(
        frame_width / 2.0, 1.0
    )
    normalized_y = (center_y - frame_height / 2.0) / max(
        frame_height / 2.0, 1.0
    )
    center_quality = float(
        np.clip(
            1.0 - math.hypot(normalized_x, normalized_y) / math.sqrt(2.0),
            0.0,
            1.0,
        )
    )
    reference_area = max(config.score_reference_area_ratio, 1e-6)
    area_quality = float(
        np.clip(math.sqrt(detection.area / frame_area / reference_area), 0.0, 1.0)
    )
    weights = np.asarray(
        [
            max(0.0, config.score_confidence_weight),
            max(0.0, config.score_center_weight),
            max(0.0, config.score_area_weight),
        ],
        dtype=np.float64,
    )
    if float(weights.sum()) <= 1e-9:
        weights[:] = 1.0
    weights /= weights.sum()
    return float(
        weights[0] * np.clip(detection.confidence, 0.0, 1.0)
        + weights[1] * center_quality
        + weights[2] * area_quality
    )


def select_target_gate(
    detections: Sequence[YoloGateBox],
    previous_target: Optional[YoloGateBox],
    frame_shape: tuple[int, ...],
    config: HybridGateConfig,
    *,
    lock_active: bool,
) -> Optional[YoloGateBox]:
    """Select a scored gate at acquisition and preserve target identity.

    While locked, overlap with the old box is authoritative. Center and size
    similarity break ties when two overlapping gates have comparable IoU.
    """
    valid = []
    for detection in detections:
        clipped, inside_fraction = _inside_box(detection, frame_shape)
        if (
            clipped.area < config.minimum_gate_area_px
            or inside_fraction < 1.0 - config.maximum_outside_fraction
        ):
            continue
        valid.append(clipped)
    if not valid:
        return None
    if previous_target is None or not lock_active:
        # A newly acquired target must be the largest visible gate.  Center
        # proximity is not a safe primary signal after a pass: a tiny,
        # high-confidence distant gate near image center can otherwise beat
        # the substantially larger next gate at the edge of the frame.
        # Detector confidence and the composite score remain tie-breakers.
        return max(
            valid,
            key=lambda item: (
                item.area,
                item.confidence,
                score_gate_candidate(item, frame_shape, config),
            ),
        )

    previous_span = max(
        previous_target.bbox[2] - previous_target.bbox[0],
        previous_target.bbox[3] - previous_target.bbox[1],
        1.0,
    )
    matches = []
    for detection in valid:
        overlap = _bbox_iou(detection, previous_target)
        center_distance = math.hypot(
            detection.center[0] - previous_target.center[0],
            detection.center[1] - previous_target.center[1],
        )
        center_similarity = math.exp(-center_distance / previous_span)
        size_similarity = math.exp(
            -abs(math.log(max(detection.area, 1.0) / max(previous_target.area, 1.0)))
        )
        # A locked gate can move by more than one old box width between slow
        # pose inferences when a yaw correction pulls it in from the frame
        # edge. Permit that motion only when its scale remains consistent;
        # this retains the physical-instance lock without accepting a larger
        # unrelated gate elsewhere in the scene.
        if overlap < 0.02:
            area_ratio = detection.area / max(previous_target.area, 1.0)
            if (
                center_distance
                > config.target_association_center_span * previous_span
                or area_ratio < config.target_association_min_area_ratio
                or area_ratio > config.target_association_max_area_ratio
            ):
                continue
        association = (
            0.65 * overlap
            + 0.23 * center_similarity
            + 0.10 * size_similarity
            + 0.02 * score_gate_candidate(detection, frame_shape, config)
        )
        matches.append((association, detection))
    return max(matches, key=lambda item: item[0])[1] if matches else None


def crop_target_gate(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    padding_px: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Return one padded crop and its full-image (x, y, width, height)."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    left = max(0, int(math.floor(x1)) - max(0, padding_px))
    top = max(0, int(math.floor(y1)) - max(0, padding_px))
    right = min(width, int(math.ceil(x2)) + max(0, padding_px))
    bottom = min(height, int(math.ceil(y2)) + max(0, padding_px))
    if right <= left or bottom <= top:
        return frame[0:0, 0:0], (left, top, 0, 0)
    return frame[top:bottom, left:right], (
        left,
        top,
        right - left,
        bottom - top,
    )


def _crop_orange_mask(
    crop: np.ndarray,
    config: HybridGateConfig,
) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
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
        (cv2.MORPH_OPEN, config.opening_kernel_size),
        (cv2.MORPH_CLOSE, config.closing_kernel_size),
    ):
        size = max(1, int(size))
        if size > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
            mask = cv2.morphologyEx(mask, operation, kernel, iterations=1)
    return hsv, mask


def _contour_depth(index: int, hierarchy: np.ndarray) -> int:
    depth = 0
    parent = int(hierarchy[index][3])
    while parent != -1:
        depth += 1
        parent = int(hierarchy[parent][3])
    return depth


def extract_inner_gate_corners(
    crop: np.ndarray,
    config: HybridGateConfig,
) -> tuple[Optional[InnerGateCorners], np.ndarray, str]:
    """Extract the best enclosed non-orange opening from one YOLO crop."""
    if crop is None or crop.size == 0:
        return None, np.zeros((1, 1), dtype=np.uint8), "empty_crop"
    hsv, mask = _crop_orange_mask(crop, config)
    contours, hierarchy_raw = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy_raw is None:
        return None, mask, "no_contours"
    hierarchy = hierarchy_raw[0]
    crop_center = np.asarray(
        [crop.shape[1] / 2.0, crop.shape[0] / 2.0],
        dtype=np.float64,
    )
    candidates = []
    for index, contour in enumerate(contours):
        # Odd hierarchy depth is a non-orange hole enclosed by orange.
        if _contour_depth(index, hierarchy) % 2 != 1:
            continue
        area = float(cv2.contourArea(contour))
        if area < config.minimum_opening_area_px:
            continue
        rect = cv2.minAreaRect(contour)
        (_, _), (rect_width, rect_height), _ = rect
        if (
            rect_width < config.minimum_opening_side_px
            or rect_height < config.minimum_opening_side_px
        ):
            continue
        aspect = rect_width / max(rect_height, 1e-6)
        if not (
            config.minimum_opening_aspect
            <= aspect
            <= config.maximum_opening_aspect
        ):
            continue
        rect_area = max(rect_width * rect_height, 1.0)
        rectangularity = float(np.clip(area / rect_area, 0.0, 1.0))
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
        reliable = len(approx) == 4 and cv2.isContourConvex(approx)
        polygon = (
            order_corners(approx.reshape(4, 2))
            if reliable
            else order_corners(cv2.boxPoints(rect))
        )

        interior = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillConvexPoly(interior, np.round(polygon).astype(np.int32), 255)
        non_orange = (interior > 0) & (mask == 0)
        if np.count_nonzero(non_orange) >= 20:
            pixels = hsv[non_orange]
            bright_neutral = float(
                np.mean((pixels[:, 1] <= 80) & (pixels[:, 2] >= 180))
            )
            if bright_neutral > config.maximum_bright_neutral_fraction:
                continue
        center = np.mean(polygon, axis=0)
        center_distance = float(
            np.linalg.norm(center - crop_center)
            / max(math.hypot(*crop_center), 1.0)
        )
        score = (
            area
            * (0.55 + 0.45 * rectangularity)
            * math.exp(-0.75 * center_distance)
        )
        candidates.append((score, polygon, reliable))
    if not candidates:
        return None, mask, "no_enclosed_opening"
    _, polygon, reliable = max(candidates, key=lambda item: item[0])
    ordered = order_corners(polygon)
    corners = InnerGateCorners(
        top_left=tuple(float(value) for value in ordered[0]),
        top_right=tuple(float(value) for value in ordered[1]),
        bottom_left=tuple(float(value) for value in ordered[3]),
        bottom_right=tuple(float(value) for value in ordered[2]),
        reliable=reliable,
    )
    return corners, mask, "inner_corners"


def convert_crop_coordinates(
    corners: InnerGateCorners,
    crop_bbox: tuple[int, int, int, int],
) -> InnerGateCorners:
    """Translate crop-local corners back into full-image coordinates."""
    offset_x, offset_y = crop_bbox[:2]

    def translate(point):
        return (point[0] + offset_x, point[1] + offset_y)

    return InnerGateCorners(
        top_left=translate(corners.top_left),
        top_right=translate(corners.top_right),
        bottom_left=translate(corners.bottom_left),
        bottom_right=translate(corners.bottom_right),
        reliable=corners.reliable,
    )


def calculate_gate_center(
    corners: InnerGateCorners,
) -> tuple[float, float]:
    points = np.asarray(
        [
            corners.top_left,
            corners.top_right,
            corners.bottom_left,
            corners.bottom_right,
        ],
        dtype=np.float64,
    )
    center = np.mean(points, axis=0)
    return (float(center[0]), float(center[1]))


def _opening_dimensions(corners: InnerGateCorners) -> tuple[float, float]:
    tl = np.asarray(corners.top_left)
    tr = np.asarray(corners.top_right)
    bl = np.asarray(corners.bottom_left)
    br = np.asarray(corners.bottom_right)
    width = 0.5 * (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl))
    height = 0.5 * (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr))
    return (float(width), float(height))


def draw_gate_debug_overlay(
    frame: np.ndarray,
    debug: Optional[HybridGateDebug],
) -> np.ndarray:
    """Draw all YOLO boxes and the selected crop/corner center source."""
    if debug is None:
        return frame
    output = frame.copy()
    for detection in debug.detections:
        x1, y1, x2, y2 = (
            int(round(value)) for value in detection.bbox
        )
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 170, 0), 1)
        cv2.putText(
            output,
            f"YOLO gate {detection.confidence:.2f}",
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (255, 170, 0),
            1,
            cv2.LINE_AA,
        )
    if debug.selected is not None:
        x1, y1, x2, y2 = (
            int(round(value)) for value in debug.selected.bbox
        )
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 3)
    if debug.crop_bbox is not None:
        x, y, width, height = debug.crop_bbox
        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + height),
            (0, 255, 255),
            1,
        )
    if debug.corners is not None:
        for label, point in (
            ("TL", debug.corners.top_left),
            ("TR", debug.corners.top_right),
            ("BL", debug.corners.bottom_left),
            ("BR", debug.corners.bottom_right),
        ):
            pixel = tuple(int(round(value)) for value in point)
            cv2.circle(output, pixel, 4, (255, 0, 255), -1)
            cv2.putText(
                output,
                label,
                (pixel[0] + 4, pixel[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (255, 0, 255),
                1,
            )
    if debug.center is not None:
        center = tuple(int(round(value)) for value in debug.center)
        cv2.drawMarker(
            output, center, (0, 0, 255), cv2.MARKER_CROSS, 18, 2
        )
    cv2.putText(
        output,
        (
            f"YOLO n={len(debug.detections)} "
            f"source={debug.center_source} "
            f"corners={debug.extraction_reason}"
        ),
        (10, max(18, output.shape[0] - 32)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


class YoloHybridGateDetector:
    """Drop-in gate detector using YOLO boxes to isolate HSV processing."""

    def __init__(
        self,
        config: HybridGateConfig,
        model: Any = None,
    ):
        self.config = config
        self.model = model if model is not None else self._load_model()
        self.last_debug = DetectorDebug(
            raw_mask=np.zeros((1, 1), dtype=np.uint8),
            cleaned_mask=np.zeros((1, 1), dtype=np.uint8),
        )
        self.last_hybrid_debug = HybridGateDebug()
        self._previous_target: Optional[YoloGateBox] = None
        self._previous_valid_detection: Optional[GateDetection] = None
        self._lock_until = 0.0
        self._missing_frames = 0
        self._last_log_at = -math.inf

    def _load_model(self):
        model_path = Path(self.config.model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"custom YOLO gate weights not found: {model_path}"
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is required for yolo_hybrid mode; run "
                "python -m pip install -r requirements.txt"
            ) from exc
        return YOLO(str(model_path))

    def reset_target_lock(self) -> None:
        self._previous_target = None
        self._previous_valid_detection = None
        self._lock_until = 0.0
        self._missing_frames = 0

    def _to_gate_detection(
        self,
        selected: YoloGateBox,
        corners: Optional[InnerGateCorners],
        frame_shape: tuple[int, ...],
        timestamp: float,
    ) -> GateDetection:
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = selected.bbox
        bbox = (
            int(round(x1)),
            int(round(y1)),
            int(round(x2 - x1)),
            int(round(y2 - y1)),
        )
        if corners is not None:
            center_x, center_y = calculate_gate_center(corners)
            opening_width, opening_height = _opening_dimensions(corners)
            polygon = corners.as_polygon()
            method = "yolo_inner_corners"
            confidence = selected.confidence
        else:
            center_x, center_y = selected.center
            opening_width = max(1.0, (x2 - x1) * 0.72)
            opening_height = max(1.0, (y2 - y1) * 0.72)
            polygon = None
            method = "yolo_box_fallback"
            confidence = 0.75 * selected.confidence
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
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            corners=polygon,
            method=method,
            corners_reliable=bool(corners and corners.reliable),
            distance_m=float(distance),
            frame_width=width,
            frame_height=height,
            bbox=bbox,
            timestamp=timestamp,
        )

    def _publish_debug(
        self,
        frame_shape: tuple[int, ...],
        detections: list[YoloGateBox],
        selected: Optional[YoloGateBox],
        crop_bbox: Optional[tuple[int, int, int, int]],
        crop_mask: Optional[np.ndarray],
        corners: Optional[InnerGateCorners],
        center: Optional[tuple[float, float]],
        source: str,
        reason: str,
        timings: dict[str, float],
    ) -> None:
        height, width = frame_shape[:2]
        full_mask = np.zeros((height, width), dtype=np.uint8)
        if crop_bbox is not None and crop_mask is not None:
            x, y, crop_width, crop_height = crop_bbox
            if crop_width > 0 and crop_height > 0:
                full_mask[
                    y : y + crop_height,
                    x : x + crop_width,
                ] = crop_mask[:crop_height, :crop_width]
        debug_candidates = []
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            contour = np.asarray(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                dtype=np.int32,
            ).reshape(-1, 1, 2)
            debug_candidates.append(
                CandidateDebug(
                    outer_contour=contour,
                    opening_contour=None,
                    accepted=True,
                    score=detection.confidence,
                    confidence=detection.confidence,
                    reason="yolo_gate",
                    method="yolo_gate",
                    center=detection.center,
                    bbox=(
                        int(round(x1)),
                        int(round(y1)),
                        int(round(x2 - x1)),
                        int(round(y2 - y1)),
                    ),
                    features={"supported_sides": 4.0},
                )
            )
        selected_contour = None
        if selected is not None:
            x1, y1, x2, y2 = selected.bbox
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
            raw_mask=full_mask.copy(),
            cleaned_mask=full_mask,
            candidates=debug_candidates,
            selected_contour=selected_contour,
            selected_opening_contour=opening_contour,
            raw_center=center,
            scale=1.0,
            timings_ms=timings,
        )
        self.last_hybrid_debug = HybridGateDebug(
            detections=detections,
            selected=selected,
            crop_bbox=crop_bbox,
            crop_mask=crop_mask,
            corners=corners,
            center=center,
            center_source=source,
            extraction_reason=reason,
            missing_frames=self._missing_frames,
        )

    def _maybe_log(self, now: float) -> None:
        if now - self._last_log_at < self.config.log_interval_s:
            return
        debug = self.last_hybrid_debug
        selected_confidence = (
            "none"
            if debug.selected is None
            else f"{debug.selected.confidence:.2f}"
        )
        print(
            "[YOLO_GATE] "
            f"detections={len(debug.detections)} "
            f"selected_conf={selected_confidence} "
            f"bbox={None if debug.selected is None else debug.selected.bbox} "
            f"corners={debug.extraction_reason} "
            f"center={debug.center} "
            f"source={debug.center_source}",
            flush=True,
        )
        self._last_log_at = now

    def draw_debug_overlay(self, frame: np.ndarray) -> np.ndarray:
        return draw_gate_debug_overlay(frame, self.last_hybrid_debug)

    def detect(
        self,
        frame: np.ndarray,
        hint: Optional[GateDetection] = None,
        timestamp: Optional[float] = None,
    ) -> GateDetection:
        del hint  # YOLO box identity owns pre-tracker association.
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
                None,
                None,
                "none",
                "empty_frame",
                {"total": 0.0},
            )
            return empty

        inference_started = time.perf_counter()
        detections = detect_gates_yolo(frame, self.model, self.config)
        inference_done = time.perf_counter()
        selected = select_target_gate(
            detections,
            self._previous_target,
            frame.shape,
            self.config,
            lock_active=now <= self._lock_until,
        )
        selection_done = time.perf_counter()

        crop_bbox = None
        crop_mask = None
        full_corners = None
        center = None
        source = "none"
        reason = "no_yolo_target"
        result = GateDetection(
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            timestamp=now,
        )
        if selected is not None:
            crop, crop_bbox = crop_target_gate(
                frame, selected.bbox, self.config.crop_padding_px
            )
            local_corners, crop_mask, reason = extract_inner_gate_corners(
                crop, self.config
            )
            if local_corners is not None:
                full_corners = convert_crop_coordinates(
                    local_corners, crop_bbox
                )
                source = "inner_corners"
            else:
                source = "yolo_box_fallback"
            result = self._to_gate_detection(
                selected,
                full_corners,
                frame.shape,
                now,
            )
            center = result.center_px
            self._previous_target = selected
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
                    method="yolo_previous_fallback",
                    timestamp=now,
                )
                center = result.center_px
                source = "previous_frame_fallback"
                reason = "target_temporarily_missing"
            elif now > self._lock_until:
                self.reset_target_lock()
        extraction_done = time.perf_counter()
        timings = {
            "yolo_inference": (inference_done - inference_started) * 1000.0,
            "target_selection": (selection_done - inference_done) * 1000.0,
            "crop_extraction": (extraction_done - selection_done) * 1000.0,
            "total": (extraction_done - started) * 1000.0,
        }
        self._publish_debug(
            frame.shape,
            detections,
            selected,
            crop_bbox,
            crop_mask,
            full_corners,
            center,
            source,
            reason,
            timings,
        )
        self._maybe_log(now)
        return result
