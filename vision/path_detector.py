"""Detect the cyan course lane and estimate its image-space centerline."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class BluePathConfig:
    hsv_lower: tuple[int, int, int] = (88, 70, 70)
    hsv_upper: tuple[int, int, int] = (110, 255, 255)
    roi_top_ratio: float = 0.40
    reference_y_ratio: float = 0.88
    lookahead_y_ratio: float = 0.55
    minimum_line_length_ratio: float = 0.07
    maximum_line_gap: int = 28
    hough_threshold: int = 18
    minimum_abs_dx: float = 8.0
    minimum_abs_dy: float = 10.0
    # Near a gate, the course rails are visible only through the opening and
    # can narrow to roughly 8-10% of the frame before the next turn.
    minimum_lane_width_ratio: float = 0.06
    maximum_lane_width_ratio: float = 1.60
    ema_alpha: float = 0.42
    prediction_frames: int = 4
    prediction_decay: float = 0.72


@dataclass
class BluePathDetection:
    found: bool = False
    confidence: float = 0.0
    normalized_offset: float = 0.0
    normalized_heading: float = 0.0
    left_line: Optional[tuple[tuple[int, int], tuple[int, int]]] = None
    right_line: Optional[tuple[tuple[int, int], tuple[int, int]]] = None
    center_line: Optional[tuple[tuple[int, int], tuple[int, int]]] = None
    mask: Optional[np.ndarray] = None
    predicted: bool = False
    timestamp: Optional[float] = None
    segment_count: int = 0


class BluePathDetector:
    """Fit converging left/right lane boundaries in the lower camera image."""

    def __init__(self, config: Optional[BluePathConfig] = None):
        self.config = config or BluePathConfig()
        self._last: Optional[BluePathDetection] = None
        self._missing_frames = 0
        self.last_time_ms = 0.0

    @staticmethod
    def _x_at_y(segment: np.ndarray, y: float) -> Optional[float]:
        x1, y1, x2, y2 = (float(value) for value in segment)
        dy = y2 - y1
        if abs(dy) < 1e-6:
            return None
        return x1 + (y - y1) * (x2 - x1) / dy

    @staticmethod
    def _line(
        x_reference: float,
        x_lookahead: float,
        y_reference: int,
        y_lookahead: int,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return (
            (int(round(x_reference)), y_reference),
            (int(round(x_lookahead)), y_lookahead),
        )

    def _prediction(
        self,
        mask: np.ndarray,
        timestamp: Optional[float],
    ) -> BluePathDetection:
        self._missing_frames += 1
        if (
            self._last is None
            or self._missing_frames > self.config.prediction_frames
        ):
            return BluePathDetection(mask=mask, timestamp=timestamp)
        predicted = replace(
            self._last,
            confidence=(
                self._last.confidence
                * self.config.prediction_decay
            ),
            mask=mask,
            predicted=True,
            timestamp=timestamp,
            segment_count=0,
        )
        self._last = predicted
        return predicted

    def detect(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> BluePathDetection:
        started = cv2.getTickCount()
        if frame is None or frame.size == 0:
            return BluePathDetection(timestamp=timestamp)

        cfg = self.config
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.asarray(cfg.hsv_lower, dtype=np.uint8),
            np.asarray(cfg.hsv_upper, dtype=np.uint8),
        )
        roi_top = int(round(height * cfg.roi_top_ratio))
        mask[:roi_top] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        segments_raw = cv2.HoughLinesP(
            mask,
            1,
            np.pi / 180.0,
            cfg.hough_threshold,
            minLineLength=max(
                12,
                int(round(width * cfg.minimum_line_length_ratio)),
            ),
            maxLineGap=cfg.maximum_line_gap,
        )
        if segments_raw is None:
            result = self._prediction(mask, timestamp)
            self.last_time_ms = (
                cv2.getTickCount() - started
            ) * 1000.0 / cv2.getTickFrequency()
            return result

        segments = np.asarray(segments_raw).reshape(-1, 4)
        y_reference = int(round(height * cfg.reference_y_ratio))
        y_lookahead = int(round(height * cfg.lookahead_y_ratio))
        image_center = width / 2.0
        left: list[tuple[float, float, float]] = []
        right: list[tuple[float, float, float]] = []
        for segment in segments:
            x1, y1, x2, y2 = (float(value) for value in segment)
            dx, dy = x2 - x1, y2 - y1
            if (
                abs(dx) < cfg.minimum_abs_dx
                or abs(dy) < cfg.minimum_abs_dy
            ):
                continue
            x_reference = self._x_at_y(segment, y_reference)
            x_lookahead = self._x_at_y(segment, y_lookahead)
            if x_reference is None or x_lookahead is None:
                continue
            if not (
                -0.5 * width <= x_reference <= 1.5 * width
                and -0.5 * width <= x_lookahead <= 1.5 * width
            ):
                continue
            length = math.hypot(dx, dy)
            if x_reference < image_center and x_lookahead > x_reference:
                left.append((x_reference, x_lookahead, length))
            elif x_reference > image_center and x_lookahead < x_reference:
                right.append((x_reference, x_lookahead, length))

        if not left or not right:
            result = self._prediction(mask, timestamp)
            self.last_time_ms = (
                cv2.getTickCount() - started
            ) * 1000.0 / cv2.getTickFrequency()
            return result

        def weighted(values, index):
            numerator = sum(row[index] * row[2] for row in values)
            denominator = sum(row[2] for row in values)
            return numerator / max(denominator, 1e-6)

        left_reference = weighted(left, 0)
        left_lookahead = weighted(left, 1)
        right_reference = weighted(right, 0)
        right_lookahead = weighted(right, 1)
        lane_width = right_reference - left_reference
        if not (
            cfg.minimum_lane_width_ratio * width
            <= lane_width
            <= cfg.maximum_lane_width_ratio * width
        ):
            result = self._prediction(mask, timestamp)
            self.last_time_ms = (
                cv2.getTickCount() - started
            ) * 1000.0 / cv2.getTickFrequency()
            return result

        center_reference = (left_reference + right_reference) / 2.0
        center_lookahead = (left_lookahead + right_lookahead) / 2.0
        raw_offset = (center_reference - image_center) / (width / 2.0)
        raw_heading = (
            center_lookahead - center_reference
        ) / (width / 2.0)
        if self._last is not None and self._last.found:
            alpha = cfg.ema_alpha
            raw_offset = (
                alpha * raw_offset
                + (1.0 - alpha) * self._last.normalized_offset
            )
            raw_heading = (
                alpha * raw_heading
                + (1.0 - alpha) * self._last.normalized_heading
            )

        segment_quality = min(1.0, (len(left) + len(right)) / 8.0)
        symmetry = 1.0 - min(
            1.0,
            abs(
                (image_center - left_reference)
                - (right_reference - image_center)
            )
            / max(lane_width, 1.0),
        )
        confidence = float(
            np.clip(0.65 * segment_quality + 0.35 * symmetry, 0.0, 1.0)
        )
        result = BluePathDetection(
            found=True,
            confidence=confidence,
            normalized_offset=float(np.clip(raw_offset, -1.0, 1.0)),
            normalized_heading=float(np.clip(raw_heading, -1.0, 1.0)),
            left_line=self._line(
                left_reference,
                left_lookahead,
                y_reference,
                y_lookahead,
            ),
            right_line=self._line(
                right_reference,
                right_lookahead,
                y_reference,
                y_lookahead,
            ),
            center_line=self._line(
                center_reference,
                center_lookahead,
                y_reference,
                y_lookahead,
            ),
            mask=mask,
            predicted=False,
            timestamp=timestamp,
            segment_count=len(left) + len(right),
        )
        self._last = result
        self._missing_frames = 0
        self.last_time_ms = (
            cv2.getTickCount() - started
        ) * 1000.0 / cv2.getTickFrequency()
        return result


def draw_blue_path(
    image: np.ndarray,
    detection: Optional[BluePathDetection],
) -> np.ndarray:
    """Draw accepted blue lane geometry without rendering rejected segments."""
    if detection is None or not detection.found:
        return image
    color = (255, 190, 0)
    for line in (detection.left_line, detection.right_line):
        if line is not None:
            cv2.line(image, line[0], line[1], color, 3, cv2.LINE_AA)
    if detection.center_line is not None:
        cv2.line(
            image,
            detection.center_line[0],
            detection.center_line[1],
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        image,
        (
            f'blue path conf={detection.confidence:.2f} '
            f'offset={detection.normalized_offset:+.2f} '
            f'heading={detection.normalized_heading:+.2f}'
        ),
        (10, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        color,
        1,
        cv2.LINE_AA,
    )
    return image
