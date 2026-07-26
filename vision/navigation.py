"""Image-space gate navigation state machine and bounded visual controller."""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .gate_detector import GateDetection


class NavigationState(str, enum.Enum):
    SEARCH = "SEARCH"
    ALIGN = "ALIGN"
    APPROACH = "APPROACH"
    COMMIT = "COMMIT"
    PASS_THROUGH = "PASS_THROUGH"
    RECOVER = "RECOVER"


@dataclass(frozen=True)
class NavigationConfig:
    min_confidence: float = 0.24
    align_tolerance: float = 0.18
    approach_tolerance: float = 0.11
    commit_tolerance: float = 0.13
    commit_distance_m: float = 3.0
    commit_width_fraction: float = 0.28
    commit_duration_s: float = 0.55
    pass_through_duration_s: float = 0.85
    recover_duration_s: float = 1.4
    minimum_state_dwell_s: float = 0.12

    search_forward_mps: float = 0.08
    search_yaw_rate_rps: float = 0.20
    align_forward_mps: float = 0.25
    max_forward_mps: float = 1.8
    commit_forward_mps: float = 1.9
    max_right_mps: float = 0.45
    max_down_mps: float = 0.65
    max_yaw_rate_rps: float = math.radians(55.0)

    yaw_gain: float = 1.15
    lateral_gain: float = 0.20
    vertical_gain: float = 0.85
    center_deadband: float = 0.035
    command_lpf_alpha: float = 0.45
    max_command_delta: float = 0.22


@dataclass
class NavigationCommand:
    # Body axes: +forward, +right, +down.  +yaw_rate turns the view right.
    forward_mps: float = 0.0
    right_mps: float = 0.0
    down_mps: float = 0.0
    yaw_rate_rps: float = 0.0
    state: NavigationState = NavigationState.SEARCH
    confidence: float = 0.0
    predicted: bool = False


class GateNavigator:
    """Readable state machine between perception and the existing flight layer."""

    def __init__(self, config: Optional[NavigationConfig] = None):
        self.config = config or NavigationConfig()
        self.state = NavigationState.SEARCH
        self._state_since = 0.0
        self._last_seen_at = 0.0
        self._last_direction = 1.0
        self._last_command = np.zeros(4, dtype=np.float64)
        self._commit_started = 0.0

    def reset(self) -> None:
        self.state = NavigationState.SEARCH
        self._state_since = 0.0
        self._last_seen_at = 0.0
        self._last_direction = 1.0
        self._last_command[:] = 0.0
        self._commit_started = 0.0

    def _transition(self, state: NavigationState, now: float, force: bool = False) -> None:
        if state == self.state:
            return
        if (
            not force
            and self._state_since
            and now - self._state_since < self.config.minimum_state_dwell_s
        ):
            return
        self.state = state
        self._state_since = now
        if state == NavigationState.COMMIT:
            self._commit_started = now

    @staticmethod
    def _usable(detection: Optional[GateDetection], confidence: float) -> bool:
        return bool(
            detection is not None
            and detection.found
            and detection.confidence >= confidence
        )

    def _condition(self, values: np.ndarray) -> np.ndarray:
        cfg = self.config
        limits = np.array(
            [
                cfg.max_forward_mps if self.state != NavigationState.COMMIT else cfg.commit_forward_mps,
                cfg.max_right_mps,
                cfg.max_down_mps,
                cfg.max_yaw_rate_rps,
            ],
            dtype=np.float64,
        )
        values = np.clip(values, -limits, limits)
        delta = np.clip(
            values - self._last_command,
            -cfg.max_command_delta,
            cfg.max_command_delta,
        )
        slew_limited = self._last_command + delta
        alpha = float(np.clip(cfg.command_lpf_alpha, 0.0, 1.0))
        filtered = (1.0 - alpha) * self._last_command + alpha * slew_limited
        self._last_command = filtered
        return filtered

    def update(
        self, detection: Optional[GateDetection], now: float
    ) -> NavigationCommand:
        cfg = self.config
        usable = self._usable(detection, cfg.min_confidence)
        if usable:
            assert detection is not None
            self._last_seen_at = now
            if abs(detection.normalized_x) > cfg.center_deadband:
                self._last_direction = 1.0 if detection.normalized_x > 0 else -1.0

        if self._state_since == 0.0:
            self._state_since = now

        if self.state == NavigationState.SEARCH:
            if usable:
                self._transition(NavigationState.ALIGN, now, force=True)
        elif self.state == NavigationState.ALIGN:
            if not usable:
                self._transition(NavigationState.RECOVER, now)
            elif max(abs(detection.normalized_x), abs(detection.normalized_y)) <= cfg.approach_tolerance:
                self._transition(NavigationState.APPROACH, now)
        elif self.state == NavigationState.APPROACH:
            if not usable:
                self._transition(NavigationState.RECOVER, now)
            else:
                error = max(abs(detection.normalized_x), abs(detection.normalized_y))
                if error > cfg.align_tolerance * 1.35:
                    self._transition(NavigationState.ALIGN, now)
                width_fraction = detection.width / max(1, detection.frame_width)
                close = (
                    detection.distance_m is not None
                    and detection.distance_m <= cfg.commit_distance_m
                ) or width_fraction >= cfg.commit_width_fraction
                if close and error <= cfg.commit_tolerance and not detection.predicted:
                    self._transition(NavigationState.COMMIT, now)
        elif self.state == NavigationState.COMMIT:
            if now - self._commit_started >= cfg.commit_duration_s:
                self._transition(NavigationState.PASS_THROUGH, now, force=True)
        elif self.state == NavigationState.PASS_THROUGH:
            if now - self._state_since >= cfg.pass_through_duration_s:
                self._transition(NavigationState.SEARCH, now, force=True)
        elif self.state == NavigationState.RECOVER:
            if usable:
                self._transition(NavigationState.ALIGN, now, force=True)
            elif now - self._last_seen_at >= cfg.recover_duration_s:
                self._transition(NavigationState.SEARCH, now, force=True)

        raw = np.zeros(4, dtype=np.float64)
        if self.state == NavigationState.SEARCH:
            raw[:] = (
                cfg.search_forward_mps,
                0.0,
                0.0,
                cfg.search_yaw_rate_rps * self._last_direction,
            )
        elif self.state == NavigationState.RECOVER:
            raw[3] = 0.55 * cfg.search_yaw_rate_rps * self._last_direction
        elif self.state in (NavigationState.COMMIT, NavigationState.PASS_THROUGH):
            raw[0] = cfg.commit_forward_mps
            if usable and self.state == NavigationState.COMMIT:
                raw[2] = 0.25 * cfg.vertical_gain * detection.normalized_y
                raw[3] = 0.20 * cfg.yaw_gain * detection.normalized_x
        elif usable:
            horizontal = (
                0.0
                if abs(detection.normalized_x) <= cfg.center_deadband
                else detection.normalized_x
            )
            vertical = (
                0.0
                if abs(detection.normalized_y) <= cfg.center_deadband
                else detection.normalized_y
            )
            alignment = max(0.0, 1.0 - abs(horizontal) - abs(vertical))
            if self.state == NavigationState.ALIGN:
                raw[0] = cfg.align_forward_mps * alignment
            else:
                confidence_scale = float(np.clip(detection.confidence, 0.0, 1.0))
                angle_scale = max(0.25, 1.0 - abs(detection.angle) / 90.0)
                raw[0] = cfg.align_forward_mps + (
                    cfg.max_forward_mps - cfg.align_forward_mps
                ) * alignment * confidence_scale * angle_scale
            raw[1] = cfg.lateral_gain * horizontal
            raw[2] = cfg.vertical_gain * vertical
            raw[3] = cfg.yaw_gain * horizontal

        conditioned = self._condition(raw)
        return NavigationCommand(
            forward_mps=float(max(0.0, conditioned[0])),
            right_mps=float(conditioned[1]),
            down_mps=float(conditioned[2]),
            yaw_rate_rps=float(conditioned[3]),
            state=self.state,
            confidence=detection.confidence if usable else 0.0,
            predicted=bool(detection.predicted) if usable else False,
        )
