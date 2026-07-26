"""Lightweight temporal tracking for gate detections."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .gate_detector import GateDetection


@dataclass(frozen=True)
class TrackerConfig:
    smoothing_alpha: float = 0.42
    velocity_alpha: float = 0.35
    min_detection_confidence: float = 0.20
    max_missing_frames: int = 5
    max_center_jump: float = 0.48
    prediction_confidence_decay: float = 0.78


class GateTracker:
    """EMA tracker with short prediction, jump rejection, and bounded staleness."""

    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()
        self._track: Optional[GateDetection] = None
        self._velocity = np.zeros(2, dtype=np.float64)
        self._missing = 0

    def reset(self) -> None:
        self._track = None
        self._velocity[:] = 0.0
        self._missing = 0

    @staticmethod
    def _valid(detection: Optional[GateDetection], minimum: float) -> bool:
        return bool(
            detection is not None
            and detection.found
            and detection.confidence >= minimum
        )

    def _predict_missing(self) -> Optional[GateDetection]:
        if self._track is None:
            return None
        self._missing += 1
        if self._missing > self.config.max_missing_frames:
            self.reset()
            return None
        track = self._track
        nx = float(np.clip(track.normalized_x + self._velocity[0], -1.0, 1.0))
        ny = float(np.clip(track.normalized_y + self._velocity[1], -1.0, 1.0))
        cx = (nx + 1.0) * 0.5 * max(1, track.frame_width - 1)
        cy = (ny + 1.0) * 0.5 * max(1, track.frame_height - 1)
        confidence = track.confidence * self.config.prediction_confidence_decay
        predicted = replace(
            track,
            center_x=cx,
            center_y=cy,
            normalized_x=nx,
            normalized_y=ny,
            confidence=confidence,
            predicted=True,
            missing_frames=self._missing,
            corners=None,
        )
        self._track = predicted
        return predicted

    def update(self, detection: Optional[GateDetection]) -> Optional[GateDetection]:
        """Accept a measurement or predict briefly when it is absent.

        A measurement jumping more than ``max_center_jump`` in normalized image
        space is treated as one missing frame.  The tracker resets after the
        configured number of misses, so stale gates cannot persist indefinitely.
        """
        if not self._valid(detection, self.config.min_detection_confidence):
            return self._predict_missing()

        assert detection is not None
        if self._track is None:
            self._track = replace(detection, predicted=False, missing_frames=0)
            self._missing = 0
            return self._track

        previous = self._track
        delta = np.array(
            [
                detection.normalized_x - previous.normalized_x,
                detection.normalized_y - previous.normalized_y,
            ],
            dtype=np.float64,
        )
        if float(np.linalg.norm(delta)) > self.config.max_center_jump:
            return self._predict_missing()

        a = float(np.clip(self.config.smoothing_alpha, 0.0, 1.0))
        va = float(np.clip(self.config.velocity_alpha, 0.0, 1.0))
        self._velocity = va * delta + (1.0 - va) * self._velocity

        def blend(old: float, new: float) -> float:
            return (1.0 - a) * old + a * new

        def blend_angle(old: float, new: float) -> float:
            delta = (new - old + 90.0) % 180.0 - 90.0
            return old + a * delta

        corners = detection.corners
        if previous.corners is not None and detection.corners is not None:
            corners = (1.0 - a) * previous.corners + a * detection.corners
        distance = detection.distance_m
        if previous.distance_m is not None and detection.distance_m is not None:
            distance = blend(previous.distance_m, detection.distance_m)

        tracked = replace(
            detection,
            center_x=blend(previous.center_x, detection.center_x),
            center_y=blend(previous.center_y, detection.center_y),
            normalized_x=blend(previous.normalized_x, detection.normalized_x),
            normalized_y=blend(previous.normalized_y, detection.normalized_y),
            width=blend(previous.width, detection.width),
            height=blend(previous.height, detection.height),
            area=blend(previous.area, detection.area),
            angle=blend_angle(previous.angle, detection.angle),
            confidence=blend(previous.confidence, detection.confidence),
            corners=corners,
            distance_m=distance,
            predicted=False,
            missing_frames=0,
        )
        self._track = tracked
        self._missing = 0
        return tracked
