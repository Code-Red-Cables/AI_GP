"""Timestamped alpha-beta tracking for gate-opening detections."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .gate_detector import GateDetection


@dataclass(frozen=True)
class GateTrackerConfig:
    maximum_missing_frames: int = 5
    maximum_center_jump_ratio: float = 0.48
    maximum_size_change_ratio: float = 0.75
    minimum_update_confidence: float = 0.22
    prediction_confidence_decay: float = 0.78
    alpha: float = 0.48
    beta: float = 0.12
    size_alpha: float = 0.38
    angle_alpha: float = 0.30
    minimum_dt_seconds: float = 1.0 / 120.0
    maximum_dt_seconds: float = 0.20
    stable_center_residual: float = 0.055
    stable_size_change_ratio: float = 0.14
    near_gate_minimum_area_ratio: float = 0.020
    near_edge_margin_fraction: float = 0.04
    maximum_near_edge_shrink_fraction: float = 0.15
    # Optional seed gate prevents a new track from latching onto tiny side
    # signs/posts after the previously approached gate leaves the frame.
    minimum_seed_confidence: float = 0.0
    minimum_seed_area_ratio: float = 0.0
    maximum_seed_abs_horizontal: float = 1.0
    maximum_seed_normalized_y: float = 1.0


# Compatibility name retained for existing imports/tests.
TrackerConfig = GateTrackerConfig


class GateTracker:
    """Alpha-beta centre filter with size gating and finite prediction lifetime."""

    def __init__(self, config: Optional[GateTrackerConfig] = None):
        self.config = config or GateTrackerConfig()
        self._track: Optional[GateDetection] = None
        self._velocity = np.zeros(2, dtype=np.float64)
        self._missing_frames = 0
        self._last_reliable_direction = 1.0
        self.last_update_ms = 0.0

    @property
    def last_reliable_direction(self) -> float:
        return self._last_reliable_direction

    @property
    def current(self) -> Optional[GateDetection]:
        return self._track

    def reset(self) -> None:
        self._track = None
        self._velocity[:] = 0.0
        self._missing_frames = 0

    @staticmethod
    def _timestamp(
        detection: Optional[GateDetection], timestamp: Optional[float]
    ) -> float:
        if timestamp is not None:
            return float(timestamp)
        if detection is not None and detection.timestamp is not None:
            return float(detection.timestamp)
        return time.monotonic()

    def _dt(self, timestamp: float) -> float:
        if self._track is None or self._track.timestamp is None:
            return 1.0 / 30.0
        return float(
            np.clip(
                timestamp - self._track.timestamp,
                self.config.minimum_dt_seconds,
                self.config.maximum_dt_seconds,
            )
        )

    @staticmethod
    def _pixel_center(
        normalized_x: float,
        normalized_y: float,
        frame_width: int,
        frame_height: int,
    ) -> tuple[float, float]:
        return (
            (normalized_x + 1.0) * frame_width / 2.0,
            (normalized_y + 1.0) * frame_height / 2.0,
        )

    @staticmethod
    def _angle_blend(old: float, new: float, alpha: float) -> float:
        delta = (new - old + 90.0) % 180.0 - 90.0
        return old + alpha * delta

    @staticmethod
    def _near_frame_edge(
        detection: GateDetection, margin_fraction: float
    ) -> bool:
        if detection.frame_width <= 0 or detection.frame_height <= 0:
            return False
        x, y, width, height = detection.bbox
        horizontal_margin = detection.frame_width * margin_fraction
        vertical_margin = detection.frame_height * margin_fraction
        return bool(
            x <= horizontal_margin
            or y <= vertical_margin
            or x + width
            >= detection.frame_width - horizontal_margin
            or y + height
            >= detection.frame_height - vertical_margin
        )

    def hint(self, timestamp: Optional[float] = None) -> Optional[GateDetection]:
        """Return a non-mutating predicted association hint for candidate scoring."""
        if self._track is None:
            return None
        now = time.monotonic() if timestamp is None else float(timestamp)
        dt = self._dt(now)
        nx = float(
            np.clip(self._track.normalized_x + self._velocity[0] * dt, -1.0, 1.0)
        )
        ny = float(
            np.clip(self._track.normalized_y + self._velocity[1] * dt, -1.0, 1.0)
        )
        cx, cy = self._pixel_center(
            nx, ny, self._track.frame_width, self._track.frame_height
        )
        return replace(
            self._track,
            center_x=cx,
            center_y=cy,
            normalized_x=nx,
            normalized_y=ny,
            timestamp=now,
            velocity_x=float(self._velocity[0]),
            velocity_y=float(self._velocity[1]),
        )

    def _prediction(self, timestamp: float) -> Optional[GateDetection]:
        if self._track is None:
            return None
        self._missing_frames += 1
        if self._missing_frames > self.config.maximum_missing_frames:
            self.reset()
            return None
        dt = self._dt(timestamp)
        nx = float(
            np.clip(self._track.normalized_x + self._velocity[0] * dt, -1.0, 1.0)
        )
        ny = float(
            np.clip(self._track.normalized_y + self._velocity[1] * dt, -1.0, 1.0)
        )
        cx, cy = self._pixel_center(
            nx, ny, self._track.frame_width, self._track.frame_height
        )
        confidence = (
            self._track.confidence * self.config.prediction_confidence_decay
        )
        predicted = replace(
            self._track,
            center_x=cx,
            center_y=cy,
            normalized_x=nx,
            normalized_y=ny,
            confidence=confidence,
            corners=None,
            corners_reliable=False,
            method="tracker_prediction",
            predicted=True,
            missing_frames=self._missing_frames,
            timestamp=timestamp,
            velocity_x=float(self._velocity[0]),
            velocity_y=float(self._velocity[1]),
        )
        self._track = predicted
        return predicted

    def update(
        self,
        detection: Optional[GateDetection],
        timestamp: Optional[float] = None,
    ) -> Optional[GateDetection]:
        """Update from a measurement or predict briefly through a dropout."""
        started = time.perf_counter()
        now = self._timestamp(detection, timestamp)
        usable = bool(
            detection is not None
            and detection.found
            and detection.confidence >= self.config.minimum_update_confidence
        )
        if not usable:
            result = self._prediction(now)
            self.last_update_ms = (time.perf_counter() - started) * 1000.0
            return result

        assert detection is not None
        if self._track is None:
            cfg = self.config
            plausible_seed = bool(
                detection.confidence >= cfg.minimum_seed_confidence
                and detection.opening_area_ratio
                >= cfg.minimum_seed_area_ratio
                and abs(detection.normalized_x)
                <= cfg.maximum_seed_abs_horizontal
                and detection.normalized_y
                <= cfg.maximum_seed_normalized_y
            )
            if not plausible_seed:
                self.last_update_ms = (
                    time.perf_counter() - started
                ) * 1000.0
                return None
            seeded = replace(
                detection,
                predicted=False,
                missing_frames=0,
                timestamp=now,
                velocity_x=0.0,
                velocity_y=0.0,
                size_rate=0.0,
                stable_frames=1,
            )
            self._track = seeded
            self._missing_frames = 0
            if abs(seeded.normalized_x) > 0.02:
                self._last_reliable_direction = (
                    1.0 if seeded.normalized_x > 0 else -1.0
                )
            self.last_update_ms = (time.perf_counter() - started) * 1000.0
            return seeded

        previous = self._track
        dt = self._dt(now)
        predicted_center = np.array(
            [
                previous.normalized_x + self._velocity[0] * dt,
                previous.normalized_y + self._velocity[1] * dt,
            ],
            dtype=np.float64,
        )
        measured_center = np.array(
            [detection.normalized_x, detection.normalized_y], dtype=np.float64
        )
        residual = measured_center - predicted_center
        center_jump = float(np.linalg.norm(residual))

        old_area = max(previous.opening_width * previous.opening_height, 1.0)
        new_area = max(detection.opening_width * detection.opening_height, 1.0)
        area_ratio = new_area / old_area
        size_change = max(area_ratio, 1.0 / area_ratio) - 1.0
        edge_limited_close_gate = bool(
            previous.opening_area_ratio
            >= self.config.near_gate_minimum_area_ratio
            and (
                self._near_frame_edge(
                    previous, self.config.near_edge_margin_fraction
                )
                or self._near_frame_edge(
                    detection, self.config.near_edge_margin_fraction
                )
            )
        )
        unexpected_close_gate_shrink = bool(
            new_area
            < old_area
            * (1.0 - self.config.maximum_near_edge_shrink_fraction)
        )
        if (
            center_jump > self.config.maximum_center_jump_ratio
            or size_change > self.config.maximum_size_change_ratio
            or (
                edge_limited_close_gate
                and unexpected_close_gate_shrink
            )
        ):
            result = self._prediction(now)
            self.last_update_ms = (time.perf_counter() - started) * 1000.0
            return result

        filtered_center = predicted_center + self.config.alpha * residual
        self._velocity += self.config.beta * residual / dt
        self._velocity = np.clip(self._velocity, -4.0, 4.0)
        nx, ny = (float(value) for value in np.clip(filtered_center, -1.0, 1.0))
        cx, cy = self._pixel_center(
            nx, ny, detection.frame_width, detection.frame_height
        )
        size_alpha = self.config.size_alpha
        opening_width = (
            (1.0 - size_alpha) * previous.opening_width
            + size_alpha * detection.opening_width
        )
        opening_height = (
            (1.0 - size_alpha) * previous.opening_height
            + size_alpha * detection.opening_height
        )
        size_rate = math.log(new_area / old_area) / dt
        stable = (
            center_jump <= self.config.stable_center_residual
            and size_change <= self.config.stable_size_change_ratio
            and not detection.predicted
        )
        stable_frames = previous.stable_frames + 1 if stable else 1
        corners = detection.corners
        if previous.corners is not None and detection.corners is not None:
            corners = (
                (1.0 - size_alpha) * previous.corners
                + size_alpha * detection.corners
            )
        distance = detection.distance_m
        if previous.distance_m is not None and detection.distance_m is not None:
            distance = (
                (1.0 - size_alpha) * previous.distance_m
                + size_alpha * detection.distance_m
            )
        tracked = replace(
            detection,
            center_x=cx,
            center_y=cy,
            normalized_x=nx,
            normalized_y=ny,
            opening_width=float(opening_width),
            opening_height=float(opening_height),
            apparent_area=float(opening_width * opening_height),
            angle_degrees=self._angle_blend(
                previous.angle_degrees,
                detection.angle_degrees,
                self.config.angle_alpha,
            ),
            confidence=float(
                np.clip(
                    0.35 * previous.confidence + 0.65 * detection.confidence,
                    0.0,
                    1.0,
                )
            ),
            corners=corners,
            distance_m=distance,
            predicted=False,
            missing_frames=0,
            timestamp=now,
            velocity_x=float(self._velocity[0]),
            velocity_y=float(self._velocity[1]),
            size_rate=float(size_rate),
            stable_frames=stable_frames,
        )
        self._track = tracked
        self._missing_frames = 0
        if abs(tracked.normalized_x) > 0.02:
            self._last_reliable_direction = (
                1.0 if tracked.normalized_x > 0 else -1.0
            )
        self.last_update_ms = (time.perf_counter() - started) * 1000.0
        return tracked


def q2_demo_tracker_config() -> GateTrackerConfig:
    """Reject the post-pass false tracks observed in the Q2 FlightSim run."""
    return GateTrackerConfig(
        minimum_seed_confidence=0.55,
        minimum_seed_area_ratio=0.004,
        maximum_seed_abs_horizontal=0.60,
        maximum_seed_normalized_y=0.70,
    )
