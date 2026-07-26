"""Deterministic orange/red race-gate detection.

The simulator delivers JPEG frames decoded by :func:`cv2.imdecode`, so the live
pipeline passes BGR images here.  ``OrangeGateDetector`` also accepts RGB, BGRA,
and RGBA when configured explicitly.

Image-space signs are intentionally independent of the flight controller:

* normalized_x: -1 at the left edge, 0 at image centre, +1 at the right edge.
* normalized_y: -1 at the top edge, 0 at image centre, +1 at the bottom edge.

The legacy ``detect_gate`` function remains as a compatibility wrapper for the
existing estimator and smoke tests.
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
class DetectorConfig:
    """All detector tuning values in one place."""

    # Calibrated for the simulator's glowing red/orange gate.  The second range
    # handles OpenCV hue wrap-around at 0/179.
    hsv_ranges: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...] = (
        ((0, 55, 75), (18, 255, 255)),
        ((165, 55, 75), (179, 255, 255)),
    )
    input_format: str = "BGR"
    process_width: Optional[int] = None
    blur_kernel: int = 3
    open_kernel: int = 3
    close_kernel: int = 3
    dilate_iterations: int = 0

    min_contour_area: float = 45.0
    min_side_px: float = 5.0
    min_aspect: float = 0.35
    max_aspect: float = 2.8
    min_confidence: float = 0.20
    approx_epsilon_fraction: float = 0.035
    filled_object_density: float = 0.72
    max_candidates: int = 30

    gate_inner_width_m: float = cm.GATE_INNER_M
    focal_length_px: float = cm.FX


@dataclass
class CandidateDebug:
    contour: np.ndarray
    accepted: bool
    score: float
    reason: str
    bbox: tuple[int, int, int, int]


@dataclass
class DetectorDebug:
    mask: np.ndarray
    candidates: list[CandidateDebug] = field(default_factory=list)
    selected_contour: Optional[np.ndarray] = None
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class GateDetection:
    found: bool = False
    center_x: float = 0.0
    center_y: float = 0.0
    normalized_x: float = 0.0
    normalized_y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    area: float = 0.0
    angle: float = 0.0
    confidence: float = 0.0
    corners: Optional[np.ndarray] = None
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

    # Compatibility properties used by gate_estimator.py and existing tooling.
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
        return self.area


@dataclass
class _Candidate:
    contour: np.ndarray
    center: tuple[float, float]
    width: float
    height: float
    angle: float
    area: float
    bbox: tuple[int, int, int, int]
    corners: Optional[np.ndarray]
    corners_reliable: bool
    confidence: float
    score: float


def normalized_image_coordinates(
    center_x: float, center_y: float, width: int, height: int
) -> tuple[float, float]:
    """Return image coordinates in [-1,+1], with +x right and +y down."""
    if width <= 1 or height <= 1:
        return 0.0, 0.0
    nx = 2.0 * float(center_x) / float(width - 1) - 1.0
    ny = 2.0 * float(center_y) / float(height - 1) - 1.0
    return float(np.clip(nx, -1.0, 1.0)), float(np.clip(ny, -1.0, 1.0))


def order_corners(points: np.ndarray) -> np.ndarray:
    """Order four image points TL, TR, BR, BL.

    Sort cyclically around the centroid, then rotate the cycle to the point with
    the smallest x+y. This remains unique for diamond-shaped 45-degree gates,
    where the usual independent sum/difference shortcut can duplicate points.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    cyclic = pts[np.argsort(angles, kind="stable")]
    start = int(np.argmin(cyclic.sum(axis=1)))
    return np.roll(cyclic, -start, axis=0).astype(np.float32)


# Backward-compatible private name used by older tests.
_order_corners = order_corners


class OrangeGateDetector:
    """HSV/contour detector with gate-specific scoring and debug products.

    Confidence is a weighted sum of:

    * 20% apparent size,
    * 20% square/rectangular aspect quality,
    * 10% contour rectangularity,
    * 5% convex-hull quality,
    * 25% visible inner opening,
    * 10% plausible orange-border density,
    * 10% four-corner quality.

    Filled orange rectangles receive an additional 0.45 penalty.  Candidate
    selection also includes a small area multiplier so the nearest of several
    nested race gates wins without making confidence depend on image position.
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
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
        target_width = self.config.process_width
        if target_width and bgr.shape[1] > target_width:
            scale = target_width / float(bgr.shape[1])
            bgr = cv2.resize(
                bgr, (target_width, max(1, round(bgr.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        k = max(1, int(self.config.blur_kernel))
        if k % 2 == 0:
            k += 1
        if k > 1:
            bgr = cv2.GaussianBlur(bgr, (k, k), 0)
        return bgr, scale

    def _segment(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.config.hsv_ranges:
            mask = cv2.bitwise_or(
                mask,
                cv2.inRange(
                    hsv,
                    np.asarray(lower, dtype=np.uint8),
                    np.asarray(upper, dtype=np.uint8),
                ),
            )
        for operation, size in (
            (cv2.MORPH_OPEN, self.config.open_kernel),
            (cv2.MORPH_CLOSE, self.config.close_kernel),
        ):
            size = max(1, int(size))
            if size > 1:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
                mask = cv2.morphologyEx(mask, operation, kernel)
        if self.config.dilate_iterations > 0:
            mask = cv2.dilate(
                mask, None, iterations=int(self.config.dilate_iterations)
            )
        return mask

    @staticmethod
    def _child_contours(index: int, contours: Sequence[np.ndarray], hierarchy: np.ndarray):
        child_index = int(hierarchy[index][2])
        while child_index != -1:
            yield contours[child_index]
            child_index = int(hierarchy[child_index][0])

    def _score_candidate(
        self,
        index: int,
        contour: np.ndarray,
        contours: Sequence[np.ndarray],
        hierarchy: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[Optional[_Candidate], CandidateDebug]:
        cfg = self.config
        contour_area = float(cv2.contourArea(contour))
        x, y, bw, bh = cv2.boundingRect(contour)
        bbox = (int(x), int(y), int(bw), int(bh))
        if contour_area < cfg.min_contour_area:
            return None, CandidateDebug(contour, False, 0.0, "tiny_area", bbox)

        rect = cv2.minAreaRect(contour)
        (cx, cy), (rw, rh), raw_angle = rect
        if min(rw, rh) < cfg.min_side_px or rw <= 0 or rh <= 0:
            return None, CandidateDebug(contour, False, 0.0, "tiny_side", bbox)
        aspect_ratio = rw / max(rh, 1e-6)
        if not (cfg.min_aspect <= aspect_ratio <= cfg.max_aspect):
            return None, CandidateDebug(contour, False, 0.0, "aspect", bbox)

        rect_area = float(rw * rh)
        rectangularity = float(np.clip(contour_area / max(rect_area, 1.0), 0.0, 1.0))
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        convexity = float(np.clip(contour_area / max(hull_area, 1.0), 0.0, 1.0))
        aspect_quality = float(
            np.clip(min(rw, rh) / max(rw, rh), 0.0, 1.0)
        )

        children = list(self._child_contours(index, contours, hierarchy))
        inner = max(children, key=cv2.contourArea) if children else None
        inner_area = float(cv2.contourArea(inner)) if inner is not None else 0.0
        hole_ratio = inner_area / max(contour_area, 1.0)
        opening_quality = float(np.clip(hole_ratio / 0.35, 0.0, 1.0))

        roi = mask[y : y + bh, x : x + bw]
        orange_density = float(np.count_nonzero(roi) / max(1, roi.size))
        # Gate borders are neither empty nor solid. This broad trapezoid tolerates
        # perspective, clipping, glow, and fragmented sides.
        border_quality = float(
            np.clip(min(orange_density / 0.12, (0.75 - orange_density) / 0.25), 0.0, 1.0)
        )

        corner_source = inner if inner is not None and inner_area > 8.0 else contour
        perimeter = cv2.arcLength(corner_source, True)
        approx = cv2.approxPolyDP(
            corner_source, cfg.approx_epsilon_fraction * perimeter, True
        )
        corners = None
        corner_quality = 0.0
        if len(approx) == 4 and cv2.isContourConvex(approx):
            corners = order_corners(approx.reshape(4, 2))
            corner_quality = 1.0
        elif len(approx) in (3, 5, 6):
            corner_quality = 0.45

        frame_area = float(mask.shape[0] * mask.shape[1])
        size_quality = float(np.clip(math.sqrt(contour_area / max(frame_area, 1.0)) / 0.18, 0.0, 1.0))
        confidence = (
            0.20 * size_quality
            + 0.20 * aspect_quality
            + 0.10 * rectangularity
            + 0.05 * convexity
            + 0.25 * opening_quality
            + 0.10 * border_quality
            + 0.10 * corner_quality
        )

        filled = inner is None and orange_density >= cfg.filled_object_density
        if filled:
            confidence -= 0.45

        # A clipped gate often has no closed hole. Keep it detectable at reduced
        # confidence when it touches an image edge and still looks frame-like.
        touches_edge = x <= 1 or y <= 1 or x + bw >= mask.shape[1] - 1 or y + bh >= mask.shape[0] - 1
        if inner is None and touches_edge and orange_density < cfg.filled_object_density:
            confidence += 0.12

        confidence = float(np.clip(confidence, 0.0, 1.0))
        if confidence < cfg.min_confidence:
            reason = "filled" if filled else "low_confidence"
            return None, CandidateDebug(contour, False, confidence, reason, bbox)

        corners_reliable = corners is not None and inner is not None
        if corners is None:
            corners = order_corners(cv2.boxPoints(rect))

        # OpenCV's minAreaRect angle changes convention when width/height swap.
        angle = float(raw_angle)
        if rw < rh:
            angle += 90.0
        while angle >= 90.0:
            angle -= 180.0
        while angle < -90.0:
            angle += 180.0
        if abs(angle) >= 89.0:
            angle = 0.0

        # Prefer larger (nearer) gates after confidence, but never let raw pixels
        # overwhelm shape quality.
        score = confidence * (1.0 + 0.35 * size_quality)
        candidate = _Candidate(
            contour=contour,
            center=(float(cx), float(cy)),
            width=float(rw),
            height=float(rh),
            angle=angle,
            area=contour_area,
            bbox=bbox,
            corners=corners,
            corners_reliable=corners_reliable,
            confidence=confidence,
            score=score,
        )
        return candidate, CandidateDebug(contour, True, score, "accepted", bbox)

    def detect(self, frame: np.ndarray) -> GateDetection:
        """Return a structured result; ``found`` is false when no candidate passes."""
        start = time.perf_counter()
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            self.last_debug = DetectorDebug(mask=np.zeros((1, 1), dtype=np.uint8))
            return GateDetection()

        original_h, original_w = frame.shape[:2]
        bgr, scale = self._preprocess(frame)
        preprocess_done = time.perf_counter()
        mask = self._segment(bgr)
        segment_done = time.perf_counter()
        contours, hierarchy_raw = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        contours_done = time.perf_counter()

        debug_candidates: list[CandidateDebug] = []
        candidates: list[_Candidate] = []
        if hierarchy_raw is not None:
            hierarchy = hierarchy_raw[0]
            outer_indices = [
                i for i in range(len(contours)) if int(hierarchy[i][3]) == -1
            ]
            outer_indices.sort(key=lambda i: cv2.contourArea(contours[i]), reverse=True)
            for i in outer_indices[: self.config.max_candidates]:
                candidate, debug = self._score_candidate(
                    i, contours[i], contours, hierarchy, mask
                )
                debug_candidates.append(debug)
                if candidate is not None:
                    candidates.append(candidate)
        scoring_done = time.perf_counter()

        selected = max(candidates, key=lambda item: item.score) if candidates else None
        timings = {
            "preprocess": (preprocess_done - start) * 1000.0,
            "segmentation": (segment_done - preprocess_done) * 1000.0,
            "contours": (contours_done - segment_done) * 1000.0,
            "scoring": (scoring_done - contours_done) * 1000.0,
            "total": (scoring_done - start) * 1000.0,
        }
        self.last_debug = DetectorDebug(
            mask=mask,
            candidates=debug_candidates,
            selected_contour=selected.contour if selected else None,
            timings_ms=timings,
        )
        if selected is None:
            return GateDetection(frame_width=original_w, frame_height=original_h)

        inverse_scale = 1.0 / scale
        center_x = selected.center[0] * inverse_scale
        center_y = selected.center[1] * inverse_scale
        width = selected.width * inverse_scale
        height = selected.height * inverse_scale
        corners = selected.corners * inverse_scale if selected.corners is not None else None
        x, y, bw, bh = selected.bbox
        bbox = tuple(int(round(value * inverse_scale)) for value in (x, y, bw, bh))
        nx, ny = normalized_image_coordinates(
            center_x, center_y, original_w, original_h
        )
        apparent_side = max(width, height, 1.0)
        focal = self.config.focal_length_px / scale
        distance = focal * self.config.gate_inner_width_m / apparent_side
        return GateDetection(
            found=True,
            center_x=center_x,
            center_y=center_y,
            normalized_x=nx,
            normalized_y=ny,
            width=width,
            height=height,
            area=selected.area * inverse_scale * inverse_scale,
            angle=selected.angle,
            confidence=selected.confidence,
            corners=corners,
            corners_reliable=selected.corners_reliable,
            distance_m=float(distance),
            frame_width=original_w,
            frame_height=original_h,
            bbox=bbox,
        )


def _legacy_config(cfg: Optional[dict]) -> DetectorConfig:
    if not cfg:
        return DetectorConfig()
    ranges = []
    if cfg.get("lower_hsv") is not None and cfg.get("upper_hsv") is not None:
        ranges.append((tuple(cfg["lower_hsv"]), tuple(cfg["upper_hsv"])))
    if cfg.get("lower_hsv2") is not None and cfg.get("upper_hsv2") is not None:
        ranges.append((tuple(cfg["lower_hsv2"]), tuple(cfg["upper_hsv2"])))
    base = DetectorConfig()
    return replace(
        base,
        hsv_ranges=tuple(ranges) or base.hsv_ranges,
        open_kernel=int(cfg.get("kernel_size", base.open_kernel)),
        close_kernel=int(cfg.get("kernel_size", base.close_kernel)),
        min_contour_area=float(cfg.get("min_area", base.min_contour_area)),
        approx_epsilon_fraction=float(
            cfg.get("approx_eps_frac", base.approx_epsilon_fraction)
        ),
    )


def detect_gate(bgr: np.ndarray, cfg: Optional[dict] = None) -> Optional[GateDetection]:
    """Compatibility wrapper returning ``None`` instead of ``found=False``."""
    detection = OrangeGateDetector(_legacy_config(cfg)).detect(bgr)
    return detection if detection.found else None


def draw_detection(
    bgr: np.ndarray,
    detection: Optional[GateDetection],
    debug: Optional[DetectorDebug] = None,
    state: Optional[str] = None,
    command: Optional[object] = None,
) -> np.ndarray:
    """Draw detector, tracker, state, command, and timing information."""
    out = bgr.copy()
    h, w = out.shape[:2]
    cv2.line(out, (w // 2 - 12, h // 2), (w // 2 + 12, h // 2), (255, 255, 0), 1)
    cv2.line(out, (w // 2, h // 2 - 12), (w // 2, h // 2 + 12), (255, 255, 0), 1)

    if debug:
        for item in debug.candidates:
            color = (0, 160, 0) if item.accepted else (80, 80, 180)
            cv2.drawContours(out, [item.contour], -1, color, 1)

    if detection is None or not detection.found:
        cv2.putText(out, "NO GATE", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    else:
        x, y, bw, bh = detection.bbox
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        center = (int(round(detection.center_x)), int(round(detection.center_y)))
        cv2.circle(out, center, 5, (0, 0, 255), -1)
        cv2.line(out, (w // 2, h // 2), center, (0, 255, 255), 1)
        if detection.corners is not None:
            quad = np.round(detection.corners).astype(np.int32)
            cv2.polylines(out, [quad], True, (255, 0, 255), 2)
            for label, point in zip(("TL", "TR", "BR", "BL"), quad):
                cv2.putText(out, label, tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
        if detection.pnp_rvec is not None and detection.pnp_tvec is not None:
            cv2.drawFrameAxes(
                out, cm.K, None, detection.pnp_rvec, detection.pnp_tvec, 0.5, 2
            )
        distance = "?" if detection.distance_m is None else f"{detection.distance_m:.1f}m"
        prediction = " PRED" if detection.predicted else ""
        line = (
            f"conf={detection.confidence:.2f} d={distance} "
            f"err=({detection.normalized_x:+.2f},{detection.normalized_y:+.2f}) "
            f"a={detection.angle:+.1f}{prediction}"
        )
        cv2.putText(out, line, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 1, cv2.LINE_AA)

    y_text = 48
    if state:
        cv2.putText(out, f"state={state}", (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
        y_text += 20
    if command is not None:
        line = (
            f"cmd f={getattr(command, 'forward_mps', 0):+.2f} "
            f"r={getattr(command, 'right_mps', 0):+.2f} "
            f"d={getattr(command, 'down_mps', 0):+.2f} "
            f"yaw={getattr(command, 'yaw_rate_rps', 0):+.2f}"
        )
        cv2.putText(out, line, (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        y_text += 20
    if debug and debug.timings_ms:
        total = debug.timings_ms.get("total", 0.0)
        fps = 1000.0 / total if total > 0 else 0.0
        cv2.putText(out, f"vision={total:.2f}ms ({fps:.1f} fps)", (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    return out


def _synthesize_test_image() -> tuple[np.ndarray, dict]:
    image = np.full((360, 640, 3), 60, dtype=np.uint8)
    # BGR orange maps into the default first HSV range.
    cv2.rectangle(image, (230, 90), (410, 270), (0, 100, 255), 24)
    return image, {}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?")
    parser.add_argument("--output", default="_detect_debug.png")
    args = parser.parse_args()
    if args.image:
        source = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
        if source is None:
            raise SystemExit(f"failed to read {args.image}")
    else:
        source, _ = _synthesize_test_image()
    detector = OrangeGateDetector()
    result = detector.detect(source)
    cv2.imwrite(args.output, draw_detection(source, result, detector.last_debug))
    print(result)
    print(detector.last_debug.timings_ms if detector.last_debug else {})
