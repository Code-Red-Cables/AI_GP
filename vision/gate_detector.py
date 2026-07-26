"""Deterministic orange gate-opening detection.

The live receiver decodes JPEG frames with ``cv2.IMREAD_COLOR``, therefore the
normal input is BGR. Image-space signs are:

* normalized_x: -1 left, 0 centre, +1 right.
* normalized_y: -1 top, 0 centre, +1 bottom.

The target is the centre of the *opening*, never the centroid of orange pixels.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import camera_model as cm


@dataclass(frozen=True)
class GateVisionConfig:
    """Central configuration for segmentation and all candidate strategies."""

    hsv_ranges: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...] = (
        # Calibrated from the live Q2 gate material: its illuminated face is
        # H=3..17, strongly saturated, and bright. Floor/wall glow is primarily
        # H=18..20 and substantially dimmer.
        ((3, 105, 180), (17, 255, 255)),
    )
    input_format: str = "BGR"
    process_width: Optional[int] = None
    blur_kernel: int = 3
    opening_kernel_size: int = 3
    closing_kernel_size: int = 3
    opening_iterations: int = 1
    closing_iterations: int = 1
    dilation_iterations: int = 0

    min_contour_area: float = 45.0
    min_opening_area: float = 12.0
    min_gate_width: float = 5.0
    min_gate_height: float = 5.0
    # Below seven pixels the saved simulator floor markers become
    # indistinguishable from tiny outlined gates; a real 1.5 m opening remains
    # larger than this throughout the configured useful range.
    min_opening_width: float = 18.0
    min_opening_height: float = 18.0
    min_aspect_ratio: float = 0.35
    max_aspect_ratio: float = 2.8
    polygon_epsilon_ratio: float = 0.035
    min_confidence: float = 0.22
    maximum_candidates: int = 30
    # Retained for configuration compatibility. Q2 now always selects the
    # largest valid opening instead of handing off through a nearer opening.
    handoff_bbox_area_ratio: float = 0.30

    estimated_opening_scale: float = 0.72
    minimum_supported_sides: int = 3
    minimum_side_coverage: float = 0.16
    maximum_center_orange_fraction: float = 0.42
    filled_object_density: float = 0.72
    compound_child_area_fraction: float = 0.18
    single_opening_min_aspect_ratio: float = 0.50
    single_opening_max_aspect_ratio: float = 2.00

    enable_line_reconstruction: bool = True
    hough_threshold: int = 20
    hough_min_line_fraction: float = 0.055
    hough_max_line_gap: int = 14
    hough_angle_tolerance_degrees: float = 20.0

    temporal_center_sigma: float = 0.38
    temporal_size_sigma: float = 0.75
    temporal_angle_sigma_degrees: float = 40.0
    # Once the tracker owns a gate, keep that identity until its opening
    # disappears or becomes implausible. This prevents a farther off-axis gate
    # from stealing control for a few frames as contour areas fluctuate.
    target_lock_center_radius_normalized: float = 0.30
    target_lock_min_area_ratio: float = 0.40
    target_lock_max_area_ratio: float = 2.50
    target_lock_min_confidence: float = 0.32
    gate_inner_width_m: float = cm.GATE_INNER_M
    focal_length_px: float = cm.FX


# Compatibility name retained for existing imports.
DetectorConfig = GateVisionConfig


@dataclass
class CandidateDebug:
    outer_contour: np.ndarray
    opening_contour: Optional[np.ndarray]
    accepted: bool
    score: float
    confidence: float
    reason: str
    method: str
    center: tuple[float, float]
    bbox: tuple[int, int, int, int]
    features: dict[str, float] = field(default_factory=dict)

    @property
    def contour(self) -> np.ndarray:
        return self.outer_contour


@dataclass
class DetectorDebug:
    raw_mask: np.ndarray
    cleaned_mask: np.ndarray
    hsv: Optional[np.ndarray] = None
    candidates: list[CandidateDebug] = field(default_factory=list)
    selected_contour: Optional[np.ndarray] = None
    selected_opening_contour: Optional[np.ndarray] = None
    raw_center: Optional[tuple[float, float]] = None
    scale: float = 1.0
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def mask(self) -> np.ndarray:
        return self.cleaned_mask


@dataclass
class GateDetection:
    found: bool = False
    center_x: float = 0.0
    center_y: float = 0.0
    normalized_x: float = 0.0
    normalized_y: float = 0.0
    opening_width: float = 0.0
    opening_height: float = 0.0
    apparent_area: float = 0.0
    angle_degrees: float = 0.0
    confidence: float = 0.0
    corners: Optional[np.ndarray] = None
    method: str = "none"
    corners_reliable: bool = False
    distance_m: Optional[float] = None
    pnp_reprojection_error: Optional[float] = None
    pnp_rvec: Optional[np.ndarray] = None
    pnp_tvec: Optional[np.ndarray] = None
    predicted: bool = False
    missing_frames: int = 0
    frame_width: int = 0
    frame_height: int = 0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    timestamp: Optional[float] = None
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    size_rate: float = 0.0
    stable_frames: int = 0

    # Compatibility aliases used by the existing estimator/planner.
    @property
    def width(self) -> float:
        return self.opening_width

    @property
    def height(self) -> float:
        return self.opening_height

    @property
    def area(self) -> float:
        return self.apparent_area

    @property
    def angle(self) -> float:
        return self.angle_degrees

    @property
    def center_px(self) -> tuple[float, float]:
        return (self.center_x, self.center_y)

    @property
    def corners_px(self) -> Optional[list[tuple[float, float]]]:
        if self.corners is None:
            return None
        return [tuple(float(v) for v in point) for point in self.corners]

    @property
    def bbox_px(self) -> tuple[int, int, int, int]:
        return self.bbox

    @property
    def area_px(self) -> float:
        return self.apparent_area

    @property
    def opening_area_ratio(self) -> float:
        frame_area = self.frame_width * self.frame_height
        if frame_area <= 0:
            return 0.0
        return self.opening_width * self.opening_height / frame_area

    @property
    def opening_width_ratio(self) -> float:
        return (
            self.opening_width / self.frame_width
            if self.frame_width > 0
            else 0.0
        )

    @property
    def opening_height_ratio(self) -> float:
        return (
            self.opening_height / self.frame_height
            if self.frame_height > 0
            else 0.0
        )


@dataclass
class _Candidate:
    outer_contour: np.ndarray
    opening_contour: Optional[np.ndarray]
    center: tuple[float, float]
    opening_width: float
    opening_height: float
    outer_width: float
    outer_height: float
    angle: float
    bbox: tuple[int, int, int, int]
    corners: Optional[np.ndarray]
    corners_reliable: bool
    method: str
    confidence: float
    score: float
    features: dict[str, float]


def normalized_image_coordinates(
    center_x: float, center_y: float, width: int, height: int
) -> tuple[float, float]:
    """Return normalized image coordinates with +x right and +y down."""
    if width <= 0 or height <= 0:
        return 0.0, 0.0
    nx = (float(center_x) - width / 2.0) / (width / 2.0)
    ny = (float(center_y) - height / 2.0) / (height / 2.0)
    return float(np.clip(nx, -1.0, 1.0)), float(np.clip(ny, -1.0, 1.0))


def order_corners(points: np.ndarray) -> np.ndarray:
    """Order four points TL, TR, BR, BL using a cyclic centroid ordering."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    cyclic = pts[np.argsort(angles, kind="stable")]
    start = int(np.argmin(cyclic.sum(axis=1)))
    return np.roll(cyclic, -start, axis=0).astype(np.float32)


_order_corners = order_corners


def quadrilateral_diagonal_center(corners: np.ndarray) -> tuple[float, float]:
    """Intersect TL-BR and TR-BL; fall back to their mean if nearly parallel."""
    pts = order_corners(corners).astype(np.float64)
    p, r = pts[0], pts[2] - pts[0]
    q, s = pts[1], pts[3] - pts[1]
    cross = r[0] * s[1] - r[1] * s[0]
    if abs(cross) < 1e-8:
        center = pts.mean(axis=0)
    else:
        qp = q - p
        t = (qp[0] * s[1] - qp[1] * s[0]) / cross
        center = p + t * r
    return float(center[0]), float(center[1])


def _contour_center(contour: np.ndarray) -> Optional[tuple[float, float]]:
    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-8:
        return None
    return (
        float(moments["m10"] / moments["m00"]),
        float(moments["m01"] / moments["m00"]),
    )


def _angle_delta(a: float, b: float) -> float:
    return abs((a - b + 90.0) % 180.0 - 90.0)


def _rect_angle(rect: tuple) -> float:
    (_, _), (width, height), raw = rect
    angle = float(raw + (90.0 if width < height else 0.0))
    angle = (angle + 90.0) % 180.0 - 90.0
    return 0.0 if abs(angle) >= 89.0 else angle


def _edge_lengths(corners: np.ndarray) -> tuple[float, float]:
    pts = order_corners(corners)
    horizontal = 0.5 * (
        np.linalg.norm(pts[1] - pts[0]) + np.linalg.norm(pts[2] - pts[3])
    )
    vertical = 0.5 * (
        np.linalg.norm(pts[3] - pts[0]) + np.linalg.norm(pts[2] - pts[1])
    )
    return float(horizontal), float(vertical)


def _corner_angle_quality(corners: Optional[np.ndarray]) -> float:
    """Score four cyclic corner angles; 1 is rectangular, 0 is degenerate."""
    if corners is None:
        return 0.0
    points = order_corners(corners).astype(np.float64)
    absolute_cosines = []
    for index in range(4):
        previous = points[(index - 1) % 4] - points[index]
        following = points[(index + 1) % 4] - points[index]
        denominator = np.linalg.norm(previous) * np.linalg.norm(following)
        if denominator <= 1e-8:
            return 0.0
        absolute_cosines.append(
            abs(float(np.dot(previous, following) / denominator))
        )
    return float(np.clip(1.0 - np.mean(absolute_cosines), 0.0, 1.0))


class OrangeGateDetector:
    """Multiple-strategy detector whose output target is the gate opening."""

    METHOD_PRIOR = {
        "inner_contour": 1.00,
        "compound_split": 0.95,
        "quadrilateral": 0.82,
        "line_reconstruction": 0.68,
        "partial_gate": 0.58,
        "rotated_rectangle": 0.48,
    }

    def __init__(self, config: Optional[GateVisionConfig] = None):
        self.config = config or GateVisionConfig()
        self.last_debug: Optional[DetectorDebug] = None

    def _to_bgr(self, frame: np.ndarray) -> np.ndarray:
        fmt = self.config.input_format.upper()
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if fmt == "BGR":
            return frame
        conversions = {
            "RGB": cv2.COLOR_RGB2BGR,
            "BGRA": cv2.COLOR_BGRA2BGR,
            "RGBA": cv2.COLOR_RGBA2BGR,
        }
        if fmt not in conversions:
            raise ValueError(f"unsupported input_format {self.config.input_format!r}")
        return cv2.cvtColor(frame, conversions[fmt])

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        bgr = self._to_bgr(frame)
        scale = 1.0
        if self.config.process_width and bgr.shape[1] > self.config.process_width:
            scale = self.config.process_width / float(bgr.shape[1])
            bgr = cv2.resize(
                bgr,
                (self.config.process_width, max(1, round(bgr.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        kernel = max(1, int(self.config.blur_kernel))
        kernel += 1 - kernel % 2
        if kernel > 1:
            bgr = cv2.GaussianBlur(bgr, (kernel, kernel), 0)
        return bgr, scale

    def _segment(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        raw = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.config.hsv_ranges:
            raw = cv2.bitwise_or(
                raw,
                cv2.inRange(
                    hsv,
                    np.asarray(lower, dtype=np.uint8),
                    np.asarray(upper, dtype=np.uint8),
                ),
            )
        cleaned = raw.copy()
        operations = (
            (
                cv2.MORPH_OPEN,
                self.config.opening_kernel_size,
                self.config.opening_iterations,
            ),
            (
                cv2.MORPH_CLOSE,
                self.config.closing_kernel_size,
                self.config.closing_iterations,
            ),
        )
        for operation, size, iterations in operations:
            size = max(1, int(size))
            if size > 1 and iterations > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
                cleaned = cv2.morphologyEx(
                    cleaned, operation, kernel, iterations=int(iterations)
                )
        if self.config.dilation_iterations > 0:
            cleaned = cv2.dilate(
                cleaned, None, iterations=int(self.config.dilation_iterations)
            )
        return hsv, raw, cleaned

    @staticmethod
    def _children(
        index: int, contours: Sequence[np.ndarray], hierarchy: np.ndarray
    ):
        child = int(hierarchy[index][2])
        while child != -1:
            yield contours[child]
            child = int(hierarchy[child][0])

    def _border_support(
        self, mask: np.ndarray, corners: np.ndarray
    ) -> tuple[list[float], float]:
        ordered = order_corners(corners)
        width, height = _edge_lengths(ordered)
        warp_width = int(np.clip(round(width), 16, 360))
        warp_height = int(np.clip(round(height), 16, 360))
        destination = np.array(
            [
                [0, 0],
                [warp_width - 1, 0],
                [warp_width - 1, warp_height - 1],
                [0, warp_height - 1],
            ],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(ordered, destination)
        warped = cv2.warpPerspective(mask, transform, (warp_width, warp_height))
        band = max(2, round(min(warp_width, warp_height) * 0.16))
        strips = (
            warped[:band, :],
            warped[-band:, :],
            warped[:, :band],
            warped[:, -band:],
        )
        coverage = [float(np.count_nonzero(side) / max(1, side.size)) for side in strips]
        center = warped[band:-band, band:-band]
        center_orange = (
            float(np.count_nonzero(center) / center.size) if center.size else 1.0
        )
        return coverage, center_orange

    def _partial_center(
        self,
        bbox: tuple[int, int, int, int],
        frame_shape: tuple[int, int],
        estimated_side: float,
    ) -> tuple[float, float]:
        x, y, width, height = bbox
        frame_height, frame_width = frame_shape
        cx, cy = x + width / 2.0, y + height / 2.0
        if x <= 1:
            cx = x + width - estimated_side / 2.0
        elif x + width >= frame_width - 1:
            cx = x + estimated_side / 2.0
        if y <= 1:
            cy = y + height - estimated_side / 2.0
        elif y + height >= frame_height - 1:
            cy = y + estimated_side / 2.0
        return cx, cy

    def _temporal_features(
        self,
        center: tuple[float, float],
        width: float,
        height: float,
        angle: float,
        frame_shape: tuple[int, int],
        hint: Optional[GateDetection],
    ) -> tuple[float, float, float, float]:
        frame_height, frame_width = frame_shape
        nx, ny = normalized_image_coordinates(
            center[0], center[1], frame_width, frame_height
        )
        center_prior = math.exp(-math.hypot(nx, ny) / 1.25)
        if hint is None or not hint.found:
            return 0.55, 0.55, 0.55, center_prior
        distance = math.hypot(nx - hint.normalized_x, ny - hint.normalized_y)
        center_quality = math.exp(
            -distance / max(self.config.temporal_center_sigma, 1e-6)
        )
        new_area = max(width * height, 1.0)
        old_area = max(hint.opening_width * hint.opening_height, 1.0)
        size_quality = math.exp(
            -abs(math.log(new_area / old_area))
            / max(self.config.temporal_size_sigma, 1e-6)
        )
        angle_quality = math.exp(
            -_angle_delta(angle, hint.angle_degrees)
            / max(self.config.temporal_angle_sigma_degrees, 1e-6)
        )
        return center_quality, size_quality, angle_quality, center_prior

    def _finalize_candidate(
        self,
        *,
        outer: np.ndarray,
        opening: Optional[np.ndarray],
        center: tuple[float, float],
        opening_width: float,
        opening_height: float,
        outer_width: float,
        outer_height: float,
        angle: float,
        bbox: tuple[int, int, int, int],
        corners: Optional[np.ndarray],
        corners_reliable: bool,
        method: str,
        mask: np.ndarray,
        contour_area: float,
        rectangularity: float,
        convexity: float,
        hint: Optional[GateDetection],
    ) -> tuple[Optional[_Candidate], CandidateDebug]:
        cfg = self.config
        outer_corners = order_corners(cv2.boxPoints(cv2.minAreaRect(outer)))
        coverage, center_orange = self._border_support(mask, outer_corners)
        supported_sides = sum(
            value >= cfg.minimum_side_coverage for value in coverage
        )
        # A disconnected three-sided frame has no child opening contour and a
        # low filled-area ratio.  Treat its fitted rectangle as an explicit
        # missing-side reconstruction so downstream logs expose what happened.
        if (
            method == "rotated_rectangle"
            and supported_sides >= cfg.minimum_supported_sides
            and rectangularity < 0.55
        ):
            method = "line_reconstruction"
        border_quality = float(np.mean(np.clip(np.asarray(coverage) / 0.55, 0, 1)))
        center_clear = 1.0 - center_orange
        aspect_quality = min(outer_width, outer_height) / max(
            outer_width, outer_height, 1e-6
        )
        frame_area = float(mask.shape[0] * mask.shape[1])
        size_quality = float(
            np.clip(math.sqrt(max(contour_area, 1.0) / frame_area) / 0.16, 0, 1)
        )
        opening_quality = self.METHOD_PRIOR[method]
        corner_angles = _corner_angle_quality(corners)
        corner_quality = (
            0.55 + 0.45 * corner_angles
            if corners_reliable
            else (0.55 if corners is not None else 0.0)
        )
        temporal_center, temporal_size, temporal_angle, center_prior = (
            self._temporal_features(
                center,
                opening_width,
                opening_height,
                angle,
                mask.shape,
                hint,
            )
        )
        temporal_quality = (
            0.55 * temporal_center + 0.30 * temporal_size + 0.15 * temporal_angle
        )
        features = {
            "size": size_quality,
            "aspect": aspect_quality,
            "rectangularity": rectangularity,
            "convexity": convexity,
            "opening": opening_quality,
            "border": border_quality,
            "center_clear": center_clear,
            "corners": corner_quality,
            "corner_angles": corner_angles,
            "temporal": temporal_quality,
            "center_prior": center_prior,
            "supported_sides": float(supported_sides),
        }
        confidence = (
            0.08 * size_quality
            + 0.11 * aspect_quality
            + 0.08 * rectangularity
            + 0.05 * convexity
            + 0.22 * opening_quality
            + 0.13 * border_quality
            + 0.10 * center_clear
            + 0.08 * corner_quality
            + 0.15 * temporal_quality
        )

        orange_density = float(
            np.count_nonzero(mask[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]])
            / max(1, bbox[2] * bbox[3])
        )
        reason = "accepted"
        opening_aspect = opening_width / max(opening_height, 1e-6)
        if (
            opening_width < cfg.min_opening_width
            or opening_height < cfg.min_opening_height
        ):
            confidence -= 0.65
            reason = "tiny_opening"
        elif not (
            cfg.single_opening_min_aspect_ratio
            <= opening_aspect
            <= cfg.single_opening_max_aspect_ratio
        ):
            confidence -= 0.45
            reason = "implausible_opening"
        if supported_sides < 2:
            confidence -= 0.55
            reason = "single_post"
        if center_orange > cfg.maximum_center_orange_fraction:
            # A valid opening may contain a farther concentric gate.  That is
            # useful race-course geometry, not evidence that the nearer frame
            # is a filled false positive.
            confidence -= 0.12 if opening is not None else 0.45
            reason = "center_on_orange"
        if opening is None and orange_density >= cfg.filled_object_density:
            confidence -= 0.55
            reason = "filled"
        confidence = float(np.clip(confidence, 0.0, 1.0))
        score = float(
            np.clip(
                0.68 * confidence
                + 0.23 * temporal_quality
                + 0.06 * center_prior
                + 0.03 * opening_quality,
                0.0,
                1.0,
            )
        )
        debug = CandidateDebug(
            outer,
            opening,
            confidence >= cfg.min_confidence,
            score,
            confidence,
            reason,
            method,
            center,
            bbox,
            features,
        )
        if confidence < cfg.min_confidence:
            return None, debug
        return (
            _Candidate(
                outer,
                opening,
                center,
                opening_width,
                opening_height,
                outer_width,
                outer_height,
                angle,
                bbox,
                corners,
                corners_reliable,
                method,
                confidence,
                score,
                features,
            ),
            debug,
        )

    def _contour_candidate(
        self,
        index: int,
        contour: np.ndarray,
        contours: Sequence[np.ndarray],
        hierarchy: np.ndarray,
        mask: np.ndarray,
        hint: Optional[GateDetection],
    ) -> tuple[Optional[_Candidate], CandidateDebug]:
        cfg = self.config
        area = float(cv2.contourArea(contour))
        x, y, bbox_width, bbox_height = cv2.boundingRect(contour)
        bbox = (int(x), int(y), int(bbox_width), int(bbox_height))
        empty_debug = CandidateDebug(
            contour, None, False, 0.0, 0.0, "tiny_area", "none", (0.0, 0.0), bbox
        )
        if area < cfg.min_contour_area:
            return None, empty_debug

        rect = cv2.minAreaRect(contour)
        (_, _), (rect_width, rect_height), _ = rect
        if (
            rect_width < cfg.min_gate_width
            or rect_height < cfg.min_gate_height
            or rect_width <= 0
            or rect_height <= 0
        ):
            empty_debug.reason = "tiny_side"
            return None, empty_debug
        ratio = rect_width / max(rect_height, 1e-6)
        if not (cfg.min_aspect_ratio <= ratio <= cfg.max_aspect_ratio):
            empty_debug.reason = "single_post"
            return None, empty_debug

        rect_area = max(rect_width * rect_height, 1.0)
        rectangularity = float(np.clip(area / rect_area, 0.0, 1.0))
        hull_area = max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)
        convexity = float(np.clip(area / hull_area, 0.0, 1.0))
        angle = _rect_angle(rect)

        children = [
            child
            for child in self._children(index, contours, hierarchy)
            if cv2.contourArea(child) >= cfg.min_opening_area
        ]
        children.sort(key=cv2.contourArea, reverse=True)
        compound = bool(
            len(children) >= 2
            and cv2.contourArea(children[1])
            >= cfg.compound_child_area_fraction
            * max(cv2.contourArea(children[0]), 1.0)
        )
        plausible_openings = []
        for child in children:
            (_, _), (child_width, child_height), _ = cv2.minAreaRect(child)
            child_aspect = child_width / max(child_height, 1e-6)
            if (
                cfg.single_opening_min_aspect_ratio
                <= child_aspect
                <= cfg.single_opening_max_aspect_ratio
            ):
                plausible_openings.append(child)
        if compound and plausible_openings and hint is not None and hint.found:
            hint_center = np.array(
                [
                    (hint.normalized_x + 1.0) * mask.shape[1] / 2.0,
                    (hint.normalized_y + 1.0) * mask.shape[0] / 2.0,
                ],
                dtype=np.float64,
            )
            opening = min(
                plausible_openings,
                key=lambda child: float(
                    np.linalg.norm(
                        np.asarray(_contour_center(child)) - hint_center
                    )
                ),
            )
        else:
            opening = (
                max(plausible_openings, key=cv2.contourArea)
                if plausible_openings
                else None
            )
        if opening is not None:
            opening_center = _contour_center(opening)
            opening_rect = cv2.minAreaRect(opening)
            (_, _), (opening_width, opening_height), _ = opening_rect
            perimeter = cv2.arcLength(opening, True)
            approx = cv2.approxPolyDP(
                opening, cfg.polygon_epsilon_ratio * perimeter, True
            )
            reliable = len(approx) == 4 and cv2.isContourConvex(approx)
            corners = (
                order_corners(approx.reshape(4, 2))
                if reliable
                else order_corners(cv2.boxPoints(opening_rect))
            )
            center = opening_center or quadrilateral_diagonal_center(corners)
            # Always keep published/debug geometry local to this one opening.
            # The parent orange contour can be the connected union of two
            # overlapping gates even when hierarchy heuristics do not label it
            # compound.
            opening_box = order_corners(cv2.boxPoints(opening_rect))
            opening_center_array = np.asarray(center, dtype=np.float32)
            expansion = 1.0 / max(cfg.estimated_opening_scale, 1e-6)
            local_outer_corners = (
                opening_center_array
                + expansion * (opening_box - opening_center_array)
            )
            local_outer_corners[:, 0] = np.clip(
                local_outer_corners[:, 0], 0, mask.shape[1] - 1
            )
            local_outer_corners[:, 1] = np.clip(
                local_outer_corners[:, 1], 0, mask.shape[0] - 1
            )
            local_outer = np.round(local_outer_corners).astype(
                np.int32
            ).reshape(-1, 1, 2)
            local_x, local_y, local_width, local_height = cv2.boundingRect(
                local_outer
            )
            local_bbox = (
                int(local_x),
                int(local_y),
                int(local_width),
                int(local_height),
            )
            candidate_outer = contour
            candidate_bbox = local_bbox
            candidate_outer_width = float(rect_width)
            candidate_outer_height = float(rect_height)
            candidate_area = area
            candidate_method = "inner_contour"
            if compound:
                # A connected orange component may contain two overlapping
                # gates. Build local bounds around one plausible opening so
                # neither apparent size nor debug geometry represents their
                # combined union.
                candidate_outer = local_outer
                candidate_outer_width = float(opening_width * expansion)
                candidate_outer_height = float(opening_height * expansion)
                candidate_area = max(
                    cfg.min_contour_area,
                    float(cv2.contourArea(candidate_outer))
                    - float(cv2.contourArea(opening)),
                )
                candidate_method = "compound_split"
            candidate, debug = self._finalize_candidate(
                outer=candidate_outer,
                opening=opening,
                center=center,
                opening_width=float(opening_width),
                opening_height=float(opening_height),
                outer_width=candidate_outer_width,
                outer_height=candidate_outer_height,
                angle=angle,
                bbox=candidate_bbox,
                corners=corners,
                corners_reliable=reliable,
                method=candidate_method,
                mask=mask,
                contour_area=candidate_area,
                rectangularity=rectangularity,
                convexity=convexity,
                hint=hint,
            )
            # Scoring for a normal gate still uses its measured parent frame,
            # but consumers and overlays must never receive that union contour.
            debug.outer_contour = local_outer
            debug.bbox = local_bbox
            if candidate is not None:
                candidate.outer_contour = local_outer
                candidate.bbox = local_bbox
            return candidate, debug

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(
            contour, cfg.polygon_epsilon_ratio * perimeter, True
        )
        touches_edge = (
            x <= 1
            or y <= 1
            or x + bbox_width >= mask.shape[1] - 1
            or y + bbox_height >= mask.shape[0] - 1
        )
        method = "rotated_rectangle"
        outer_corners = order_corners(cv2.boxPoints(rect))
        center = (float(rect[0][0]), float(rect[0][1]))
        if len(approx) == 4 and cv2.isContourConvex(approx):
            outer_corners = order_corners(approx.reshape(4, 2))
            center = quadrilateral_diagonal_center(outer_corners)
            method = "quadrilateral"
        elif touches_edge:
            estimated_side = max(rect_width, rect_height, bbox_width, bbox_height)
            center = self._partial_center(bbox, mask.shape, estimated_side)
            method = "partial_gate"
            if hint is not None and hint.found:
                hint_center = (
                    (hint.normalized_x + 1.0) * mask.shape[1] / 2.0,
                    (hint.normalized_y + 1.0) * mask.shape[0] / 2.0,
                )
                center = (
                    0.65 * center[0] + 0.35 * hint_center[0],
                    0.65 * center[1] + 0.35 * hint_center[1],
                )

        scale = cfg.estimated_opening_scale
        opening_width = rect_width * scale
        opening_height = rect_height * scale
        center_array = np.asarray(center, dtype=np.float32)
        corners = center_array + scale * (outer_corners - center_array)
        return self._finalize_candidate(
            outer=contour,
            opening=None,
            center=center,
            opening_width=float(opening_width),
            opening_height=float(opening_height),
            outer_width=float(rect_width),
            outer_height=float(rect_height),
            angle=angle,
            bbox=bbox,
            corners=np.asarray(corners, dtype=np.float32),
            corners_reliable=False,
            method=method,
            mask=mask,
            contour_area=area,
            rectangularity=rectangularity,
            convexity=convexity,
            hint=hint,
        )

    @staticmethod
    def _line_angle_difference(angle: float, reference: float) -> float:
        return abs((angle - reference + 90.0) % 180.0 - 90.0)

    @staticmethod
    def _normalize_hough_lines(lines: np.ndarray) -> np.ndarray:
        """Return Hough segments as N x 4 on every OpenCV platform.

        OpenCV wheels have returned both N x 1 x 4 and N x 4 arrays for
        HoughLinesP. Flattening only the singleton dimensions avoids coupling
        the detector to a platform-specific wrapper shape.
        """
        array = np.asarray(lines)
        if array.size == 0 or array.size % 4:
            return np.empty((0, 4), dtype=np.float32)
        return array.reshape(-1, 4)

    def _line_candidate(
        self, mask: np.ndarray, hint: Optional[GateDetection]
    ) -> tuple[Optional[_Candidate], Optional[CandidateDebug]]:
        cfg = self.config
        edges = cv2.Canny(mask, 60, 160)
        minimum_length = max(
            6, round(min(mask.shape) * cfg.hough_min_line_fraction)
        )
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            cfg.hough_threshold,
            minLineLength=minimum_length,
            maxLineGap=cfg.hough_max_line_gap,
        )
        if lines is None:
            return None, None
        segments = self._normalize_hough_lines(lines)
        if len(segments) < 3:
            return None, None

        records = []
        for raw in segments:
            x1, y1, x2, y2 = (float(value) for value in raw)
            length = math.hypot(x2 - x1, y2 - y1)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
            midpoint = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            if hint is not None and hint.found:
                hx = (hint.normalized_x + 1.0) * mask.shape[1] / 2.0
                hy = (hint.normalized_y + 1.0) * mask.shape[0] / 2.0
                radius = 1.8 * max(hint.opening_width, hint.opening_height, 20.0)
                if math.hypot(midpoint[0] - hx, midpoint[1] - hy) > radius:
                    continue
            records.append((raw.astype(np.float32), length, angle))
        if len(records) < 3:
            return None, None
        dominant = max(records, key=lambda item: item[1])[2]
        tolerance = cfg.hough_angle_tolerance_degrees
        parallel = [
            item
            for item in records
            if self._line_angle_difference(item[2], dominant) <= tolerance
        ]
        perpendicular = [
            item
            for item in records
            if abs(self._line_angle_difference(item[2], dominant) - 90.0)
            <= tolerance
        ]
        if not parallel or not perpendicular or len(parallel) + len(perpendicular) < 3:
            return None, None

        used = parallel + perpendicular
        points = np.array(
            [
                [[line[0][0], line[0][1]], [line[0][2], line[0][3]]]
                for line in used
            ]
        )
        points = points.reshape(-1, 2).astype(np.float32)
        rect = cv2.minAreaRect(points)
        (_, _), (width, height), _ = rect
        if (
            width < cfg.min_gate_width
            or height < cfg.min_gate_height
            or not (
                cfg.min_aspect_ratio
                <= width / max(height, 1e-6)
                <= cfg.max_aspect_ratio
            )
        ):
            return None, None
        corners = order_corners(cv2.boxPoints(rect))
        center = quadrilateral_diagonal_center(corners)
        contour = cv2.convexHull(points.reshape(-1, 1, 2))
        x, y, bbox_width, bbox_height = cv2.boundingRect(contour)
        bbox = (x, y, bbox_width, bbox_height)
        area = max(float(cv2.contourArea(contour)), 1.0)
        candidate, debug = self._finalize_candidate(
            outer=contour,
            opening=None,
            center=center,
            opening_width=float(width * cfg.estimated_opening_scale),
            opening_height=float(height * cfg.estimated_opening_scale),
            outer_width=float(width),
            outer_height=float(height),
            angle=_rect_angle(rect),
            bbox=bbox,
            corners=(
                np.asarray(center, dtype=np.float32)
                + cfg.estimated_opening_scale
                * (corners - np.asarray(center, dtype=np.float32))
            ),
            corners_reliable=False,
            method="line_reconstruction",
            mask=mask,
            contour_area=area,
            rectangularity=1.0,
            convexity=1.0,
            hint=hint,
        )
        return candidate, debug

    def detect(
        self,
        frame: np.ndarray,
        hint: Optional[GateDetection] = None,
        timestamp: Optional[float] = None,
    ) -> GateDetection:
        """Detect the best supported gate opening in one frame."""
        start = time.perf_counter()
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            empty = np.zeros((1, 1), dtype=np.uint8)
            self.last_debug = DetectorDebug(empty, empty)
            return GateDetection(timestamp=timestamp)

        original_height, original_width = frame.shape[:2]
        bgr, scale = self._preprocess(frame)
        preprocess_done = time.perf_counter()
        hsv, raw_mask, cleaned_mask = self._segment(bgr)
        mask_done = time.perf_counter()
        contours, hierarchy_raw = cv2.findContours(
            cleaned_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        contours_done = time.perf_counter()

        candidates: list[_Candidate] = []
        debug_candidates: list[CandidateDebug] = []
        if hierarchy_raw is not None:
            hierarchy = hierarchy_raw[0]

            def is_orange_outer_boundary(index: int) -> bool:
                # RETR_TREE alternates orange-object boundaries (even depth)
                # and their dark openings (odd depth). Include a farther gate
                # nested inside a nearer gate's opening for handoff scoring.
                depth = 0
                parent = int(hierarchy[index][3])
                while parent != -1:
                    depth += 1
                    parent = int(hierarchy[parent][3])
                return depth % 2 == 0

            outer_indices = [
                index
                for index in range(len(contours))
                if is_orange_outer_boundary(index)
            ]
            outer_indices.sort(
                key=lambda index: cv2.contourArea(contours[index]), reverse=True
            )
            for index in outer_indices[: self.config.maximum_candidates]:
                candidate, debug = self._contour_candidate(
                    index,
                    contours[index],
                    contours,
                    hierarchy,
                    cleaned_mask,
                    hint,
                )
                debug_candidates.append(debug)
                if candidate is not None:
                    candidates.append(candidate)
        contour_scoring_done = time.perf_counter()

        strong_contour = max(
            (candidate.confidence for candidate in candidates), default=0.0
        )
        if (
            self.config.enable_line_reconstruction
            and strong_contour < 0.55
        ):
            line_candidate, line_debug = self._line_candidate(cleaned_mask, hint)
            if line_debug is not None:
                debug_candidates.append(line_debug)
            if line_candidate is not None:
                candidates.append(line_candidate)
        line_done = time.perf_counter()

        # Acquire the largest valid opening. While a tracked hint exists,
        # retain the matching gate so normal contour-size jitter cannot cause
        # a brief switch to the next off-axis gate.
        largest = max(
            candidates,
            key=lambda candidate: (
                candidate.opening_width * candidate.opening_height,
                candidate.score,
            ),
            default=None,
        )
        selected = largest
        if hint is not None and hint.found and candidates:
            hint_area_ratio = max(hint.opening_area_ratio, 1e-9)
            locked_candidates = []
            for candidate in candidates:
                nx, ny = normalized_image_coordinates(
                    candidate.center[0],
                    candidate.center[1],
                    cleaned_mask.shape[1],
                    cleaned_mask.shape[0],
                )
                center_distance = math.hypot(
                    nx - hint.normalized_x,
                    ny - hint.normalized_y,
                )
                candidate_area_ratio = (
                    candidate.opening_width
                    * candidate.opening_height
                    / float(cleaned_mask.shape[0] * cleaned_mask.shape[1])
                )
                relative_area = candidate_area_ratio / hint_area_ratio
                if (
                    center_distance
                    <= self.config.target_lock_center_radius_normalized
                    and self.config.target_lock_min_area_ratio
                    <= relative_area
                    <= self.config.target_lock_max_area_ratio
                    and candidate.confidence
                    >= self.config.target_lock_min_confidence
                ):
                    locked_candidates.append(candidate)
            if locked_candidates:
                selected = max(
                    locked_candidates,
                    key=lambda candidate: candidate.score,
                )
        timings = {
            "preprocess": (preprocess_done - start) * 1000.0,
            "mask_generation": (mask_done - preprocess_done) * 1000.0,
            "contours": (contours_done - mask_done) * 1000.0,
            "candidate_scoring": (contour_scoring_done - contours_done) * 1000.0,
            "line_reconstruction": (line_done - contour_scoring_done) * 1000.0,
            "total": (line_done - start) * 1000.0,
        }
        self.last_debug = DetectorDebug(
            raw_mask=raw_mask,
            cleaned_mask=cleaned_mask,
            hsv=hsv,
            candidates=debug_candidates,
            selected_contour=selected.outer_contour if selected else None,
            selected_opening_contour=selected.opening_contour if selected else None,
            raw_center=selected.center if selected else None,
            scale=scale,
            timings_ms=timings,
        )
        if selected is None:
            return GateDetection(
                frame_width=original_width,
                frame_height=original_height,
                timestamp=timestamp,
            )

        inverse_scale = 1.0 / scale
        center_x = selected.center[0] * inverse_scale
        center_y = selected.center[1] * inverse_scale
        opening_width = selected.opening_width * inverse_scale
        opening_height = selected.opening_height * inverse_scale
        corners = (
            selected.corners * inverse_scale
            if selected.corners is not None
            else None
        )
        x, y, width, height = selected.bbox
        bbox = tuple(
            int(round(value * inverse_scale)) for value in (x, y, width, height)
        )
        normalized_x, normalized_y = normalized_image_coordinates(
            center_x, center_y, original_width, original_height
        )
        focal = self.config.focal_length_px / scale
        apparent_side = max(opening_width, opening_height, 1.0)
        distance = focal * self.config.gate_inner_width_m / apparent_side
        return GateDetection(
            found=True,
            center_x=center_x,
            center_y=center_y,
            normalized_x=normalized_x,
            normalized_y=normalized_y,
            opening_width=opening_width,
            opening_height=opening_height,
            apparent_area=opening_width * opening_height,
            angle_degrees=selected.angle,
            confidence=selected.confidence,
            corners=corners,
            method=selected.method,
            corners_reliable=selected.corners_reliable,
            distance_m=float(distance),
            frame_width=original_width,
            frame_height=original_height,
            bbox=bbox,
            timestamp=timestamp,
        )


def _legacy_config(cfg: Optional[dict]) -> GateVisionConfig:
    if not cfg:
        return GateVisionConfig()
    ranges = []
    if cfg.get("lower_hsv") is not None and cfg.get("upper_hsv") is not None:
        ranges.append((tuple(cfg["lower_hsv"]), tuple(cfg["upper_hsv"])))
    if cfg.get("lower_hsv2") is not None and cfg.get("upper_hsv2") is not None:
        ranges.append((tuple(cfg["lower_hsv2"]), tuple(cfg["upper_hsv2"])))
    base = GateVisionConfig()
    kernel = int(cfg.get("kernel_size", base.opening_kernel_size))
    return replace(
        base,
        hsv_ranges=tuple(ranges) or base.hsv_ranges,
        opening_kernel_size=kernel,
        closing_kernel_size=kernel,
        min_contour_area=float(cfg.get("min_area", base.min_contour_area)),
        polygon_epsilon_ratio=float(
            cfg.get("approx_eps_frac", base.polygon_epsilon_ratio)
        ),
        blur_kernel=int(cfg.get("blur_ksize", base.blur_kernel)),
        handoff_bbox_area_ratio=float(
            cfg.get(
                "handoff_bbox_frac",
                base.handoff_bbox_area_ratio,
            )
        ),
    )


def _make_cfg(cfg: Optional[dict]) -> dict:
    """Compatibility dictionary for Q2 debug/training utilities."""
    base = GateVisionConfig()
    first = base.hsv_ranges[0]
    second = base.hsv_ranges[1] if len(base.hsv_ranges) > 1 else (None, None)
    merged = {
        "lower_hsv": first[0],
        "upper_hsv": first[1],
        "lower_hsv2": second[0],
        "upper_hsv2": second[1],
        "kernel_size": base.opening_kernel_size,
        "min_area": base.min_contour_area,
        "approx_eps_frac": base.polygon_epsilon_ratio,
        "blur_ksize": base.blur_kernel,
        "handoff_bbox_frac": base.handoff_bbox_area_ratio,
    }
    if cfg:
        merged.update(cfg)
    return merged


def _build_mask(hsv: np.ndarray, cfg: dict) -> np.ndarray:
    """Compatibility mask builder for tools that already hold an HSV image."""
    config = _legacy_config(_make_cfg(cfg))
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in config.hsv_ranges:
        mask = cv2.bitwise_or(
            mask,
            cv2.inRange(
                hsv,
                np.asarray(lower, dtype=np.uint8),
                np.asarray(upper, dtype=np.uint8),
            ),
        )
    operations = (
        (
            cv2.MORPH_OPEN,
            config.opening_kernel_size,
            config.opening_iterations,
        ),
        (
            cv2.MORPH_CLOSE,
            config.closing_kernel_size,
            config.closing_iterations,
        ),
    )
    for operation, size, iterations in operations:
        if size > 1 and iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
            mask = cv2.morphologyEx(
                mask, operation, kernel, iterations=int(iterations)
            )
    return mask


def detect_gate(
    bgr: np.ndarray, cfg: Optional[dict] = None
) -> Optional[GateDetection]:
    """Legacy wrapper returning ``None`` when no gate is found."""
    detection = OrangeGateDetector(_legacy_config(cfg)).detect(bgr)
    return detection if detection.found else None


def _scaled_contour(contour: np.ndarray, scale: float) -> np.ndarray:
    """Normalize any OpenCV contour wrapper shape to contiguous N x 1 x 2."""
    array = np.asarray(contour)
    if array.size < 2 or array.size % 2:
        return np.empty((0, 1, 2), dtype=np.int32)
    points = array.reshape(-1, 2).astype(np.float64)
    points = points[np.all(np.isfinite(points), axis=1)]
    if not len(points):
        return np.empty((0, 1, 2), dtype=np.int32)
    return np.ascontiguousarray(
        np.round(points / max(float(scale), 1e-9))
        .astype(np.int32)
        .reshape(-1, 1, 2)
    )


def draw_detection(
    bgr: np.ndarray,
    detection: Optional[GateDetection],
    debug: Optional[DetectorDebug] = None,
    state: Optional[str] = None,
    command: Optional[object] = None,
    raw_detection: Optional[GateDetection] = None,
    total_time_ms: Optional[float] = None,
    show_rejected_candidates: bool = True,
    show_mask_insets: bool = True,
) -> np.ndarray:
    """Render candidates, opening geometry, tracked target, state, and timing."""
    output = bgr.copy()
    height, width = output.shape[:2]
    image_center = (width // 2, height // 2)
    cv2.drawMarker(
        output, image_center, (255, 255, 0), cv2.MARKER_CROSS, 22, 1
    )

    if debug:
        for item in debug.candidates:
            if not item.accepted and not show_rejected_candidates:
                continue
            contour = _scaled_contour(item.outer_contour, debug.scale)
            color = (0, 145, 0) if item.accepted else (80, 80, 190)
            if len(contour):
                cv2.drawContours(output, [contour], -1, color, 1)
            x, y, _, _ = item.bbox
            label_point = (
                int(x / debug.scale),
                max(12, int(y / debug.scale)),
            )
            cv2.putText(
                output,
                (
                    f"{item.method}:{item.confidence:.2f}"
                    if item.accepted
                    else f"reject:{item.reason}"
                ),
                label_point,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                color,
                1,
                cv2.LINE_AA,
            )
        if debug.selected_opening_contour is not None:
            opening = _scaled_contour(
                debug.selected_opening_contour, debug.scale
            )
            if len(opening):
                cv2.drawContours(output, [opening], -1, (255, 180, 0), 2)
        if debug.selected_contour is not None:
            selected_outer = _scaled_contour(
                debug.selected_contour, debug.scale
            )
            if len(selected_outer):
                cv2.drawContours(
                    output,
                    [selected_outer],
                    -1,
                    (40, 255, 40),
                    3,
                )

        if show_mask_insets:
            # Offline diagnostics retain compact raw/clean mask previews.
            inset_width = min(120, max(24, (width - 12) // 2))
            inset_height = max(
                1,
                min(
                    max(1, height - 4),
                    max(18, round(inset_width * height / width)),
                ),
            )
            for inset_index, (mask, label) in enumerate(
                ((debug.raw_mask, "RAW"), (debug.cleaned_mask, "CLEAN"))
            ):
                inset = cv2.resize(
                    mask,
                    (inset_width, inset_height),
                    interpolation=cv2.INTER_NEAREST,
                )
                inset = cv2.cvtColor(inset, cv2.COLOR_GRAY2BGR)
                x0 = width - (2 - inset_index) * (inset_width + 4)
                y0 = height - inset_height - 4
                output[y0 : y0 + inset_height, x0 : x0 + inset_width] = inset
                cv2.putText(
                    output,
                    label,
                    (x0 + 3, y0 + 13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    if raw_detection is not None and raw_detection.found:
        raw_center = (
            int(round(raw_detection.center_x)),
            int(round(raw_detection.center_y)),
        )
        cv2.drawMarker(
            output, raw_center, (0, 165, 255), cv2.MARKER_TILTED_CROSS, 12, 2
        )
        cv2.putText(
            output,
            "RAW",
            (raw_center[0] + 5, raw_center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 165, 255),
            1,
        )

    if detection is None or not detection.found:
        cv2.putText(
            output, "NO GATE", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2
        )
    else:
        if detection.corners is not None:
            opening_points = np.round(detection.corners).astype(
                np.int32
            ).reshape(-1, 1, 2)
            x, y, bbox_width, bbox_height = cv2.boundingRect(opening_points)
        else:
            x, y, bbox_width, bbox_height = detection.bbox
        cv2.rectangle(
            output, (x, y), (x + bbox_width, y + bbox_height), (0, 255, 0), 2
        )
        tracked_center = (
            int(round(detection.center_x)),
            int(round(detection.center_y)),
        )
        marker_color = (0, 255, 255) if detection.predicted else (0, 0, 255)
        cv2.circle(output, tracked_center, 5, marker_color, -1)
        cv2.putText(
            output,
            "PRED" if detection.predicted else "TRACK",
            (tracked_center[0] + 6, tracked_center[1] + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            marker_color,
            1,
            cv2.LINE_AA,
        )
        cv2.line(output, image_center, tracked_center, (0, 255, 255), 1)
        if detection.corners is not None:
            corners = np.round(detection.corners).astype(np.int32)
            cv2.polylines(output, [corners], True, (255, 0, 255), 2)
            for label, point in zip(("TL", "TR", "BR", "BL"), corners):
                cv2.putText(
                    output,
                    label,
                    tuple(point),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.34,
                    (255, 0, 255),
                    1,
                )
        if detection.pnp_rvec is not None and detection.pnp_tvec is not None:
            cv2.drawFrameAxes(
                output,
                cm.K,
                None,
                detection.pnp_rvec,
                detection.pnp_tvec,
                0.5,
                2,
            )
        distance = (
            "?" if detection.distance_m is None else f"{detection.distance_m:.1f}m"
        )
        prediction = " PRED" if detection.predicted else ""
        cv2.putText(
            output,
            (
                f"{detection.method}{prediction} conf={detection.confidence:.2f} "
                f"d={distance} area={detection.opening_area_ratio:.3f}"
            ),
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            (
                f"err=({detection.normalized_x:+.2f},{detection.normalized_y:+.2f}) "
                f"vel=({detection.velocity_x:+.2f},{detection.velocity_y:+.2f})"
            ),
            (10, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    text_y = 64
    if state:
        cv2.putText(
            output,
            f"state={state}",
            (10, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
        )
        text_y += 19
    if command is not None:
        cv2.putText(
            output,
            (
                f"cmd f={getattr(command, 'forward_mps', 0):+.2f} "
                f"r={getattr(command, 'right_mps', 0):+.2f} "
                f"d={getattr(command, 'down_mps', 0):+.2f} "
                f"yaw={getattr(command, 'yaw_rate_rps', 0):+.2f}"
            ),
            (10, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
        )
        text_y += 19
    if debug and debug.timings_ms:
        detector_total = debug.timings_ms.get("total", 0.0)
        total = total_time_ms if total_time_ms is not None else detector_total
        fps = 1000.0 / total if total > 0 else 0.0
        cv2.putText(
            output,
            (
                f"detector={detector_total:.2f}ms "
                f"total={total:.2f}ms ({fps:.1f} fps)"
            ),
            (10, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
        )
    return output


def _synthesize_test_image() -> tuple[np.ndarray, dict]:
    image = np.full((360, 640, 3), 60, dtype=np.uint8)
    cv2.rectangle(image, (230, 90), (410, 270), (0, 100, 255), 24)
    return image, {}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?")
    parser.add_argument("--output", default="_detect_debug.png")
    args = parser.parse_args()
    if args.image:
        source = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if source is None:
            raise SystemExit(f"failed to read {args.image}")
    else:
        source, _ = _synthesize_test_image()
    detector = OrangeGateDetector()
    result = detector.detect(source)
    cv2.imwrite(
        args.output,
        draw_detection(source, result, detector.last_debug, raw_detection=result),
    )
    print(result)
    print(detector.last_debug.timings_ms if detector.last_debug else {})
