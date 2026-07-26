"""Focused gate-opening navigation state machine and timestamped PD control."""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .gate_detector import GateDetection


class NavigationState(str, enum.Enum):
    SEARCH = "SEARCH"
    TRACK = "TRACK"
    ALIGN_AND_APPROACH = "ALIGN_AND_APPROACH"
    COMMIT = "COMMIT"
    PASS_THROUGH = "PASS_THROUGH"
    RECOVER = "RECOVER"

    # Compatibility aliases for older log readers.
    ALIGN = "TRACK"
    APPROACH = "ALIGN_AND_APPROACH"


@dataclass(frozen=True)
class NavigationConfig:
    minimum_detection_confidence: float = 0.22
    reliable_confidence: float = 0.42
    track_confirmation_frames: int = 3
    commit_stable_frames: int = 4
    commit_alignment_tolerance: float = 0.12
    commit_opening_area_ratio: float = 0.075
    commit_minimum_confidence: float = 0.52
    commit_max_center_speed: float = 0.35
    commit_max_size_rate: float = 2.4
    maximum_alignment_error: float = 0.65
    recovery_alignment_error: float = 0.82
    center_deadband: float = 0.035
    # Optical normalized-y that represents body-forward flight. This is zero
    # for a level camera and positive for Q2's upward-tilted camera.
    vertical_setpoint_normalized: float = 0.0
    vertical_deadband: float = 0.035
    vertical_control_min_area_ratio: float = 0.0
    track_alignment_scale: float = 0.45
    recover_forward_mps: float = 0.0

    minimum_state_duration_s: float = 0.10
    commit_maximum_duration_s: float = 0.65
    pass_through_duration_s: float = 0.85
    recover_local_duration_s: float = 1.35

    search_forward_mps: float = 0.04
    search_yaw_rate_rps: float = 0.20
    track_forward_mps: float = 0.18
    minimum_approach_mps: float = 0.28
    maximum_approach_mps: float = 1.75
    commit_forward_mps: float = 1.90

    horizontal_yaw_kp: float = 1.05
    horizontal_yaw_kd: float = 0.12
    lateral_kp: float = 0.18
    lateral_kd: float = 0.04
    vertical_kp: float = 0.82
    vertical_kd: float = 0.10

    max_right_mps: float = 0.42
    max_down_mps: float = 0.62
    max_yaw_rate_rps: float = math.radians(55.0)
    max_forward_acceleration: float = 2.5
    max_lateral_acceleration: float = 2.2
    max_vertical_acceleration: float = 2.4
    max_yaw_acceleration: float = math.radians(180.0)
    command_lpf_alpha: float = 0.48


@dataclass
class NavigationCommand:
    # Body axes: +forward, +right, +down; +yaw turns the view right.
    forward_mps: float = 0.0
    right_mps: float = 0.0
    down_mps: float = 0.0
    yaw_rate_rps: float = 0.0
    state: NavigationState = NavigationState.SEARCH
    confidence: float = 0.0
    predicted: bool = False
    alignment_error: float = 1.0


class GateNavigator:
    """Confirm, align, commit, and continue through a disappearing gate."""

    def __init__(self, config: Optional[NavigationConfig] = None):
        self.config = config or NavigationConfig()
        self.state = NavigationState.SEARCH
        self._state_since: Optional[float] = None
        self._last_update: Optional[float] = None
        self._last_seen_at: Optional[float] = None
        self._last_direction = 1.0
        self._last_command = np.zeros(4, dtype=np.float64)
        self._last_alignment_command = np.zeros(3, dtype=np.float64)
        self._previous_alignment_error = 1.0

    def reset(self) -> None:
        self.state = NavigationState.SEARCH
        self._state_since = None
        self._last_update = None
        self._last_seen_at = None
        self._last_direction = 1.0
        self._last_command[:] = 0.0
        self._last_alignment_command[:] = 0.0
        self._previous_alignment_error = 1.0

    def _transition(
        self, new_state: NavigationState, now: float, force: bool = False
    ) -> None:
        if new_state == self.state:
            return
        if (
            not force
            and self._state_since is not None
            and now - self._state_since < self.config.minimum_state_duration_s
        ):
            return
        self.state = new_state
        self._state_since = now

    def _dt(self, now: float) -> float:
        if self._last_update is None:
            return 1.0 / 30.0
        return float(np.clip(now - self._last_update, 1.0 / 120.0, 0.20))

    @staticmethod
    def _usable(
        detection: Optional[GateDetection], minimum_confidence: float
    ) -> bool:
        return bool(
            detection is not None
            and detection.found
            and detection.confidence >= minimum_confidence
        )

    def _errors(
        self, detection: GateDetection
    ) -> tuple[float, float, float]:
        cfg = self.config
        horizontal = (
            0.0
            if abs(detection.normalized_x) <= cfg.center_deadband
            else detection.normalized_x
        )
        vertical_error = detection.normalized_y - cfg.vertical_setpoint_normalized
        if (
            detection.opening_area_ratio
            < cfg.vertical_control_min_area_ratio
            or abs(vertical_error) <= cfg.vertical_deadband
        ):
            vertical = 0.0
        else:
            vertical = math.copysign(
                abs(vertical_error) - cfg.vertical_deadband,
                vertical_error,
            )
        return horizontal, vertical, max(abs(horizontal), abs(vertical))

    def _condition(self, desired: np.ndarray, dt: float) -> np.ndarray:
        cfg = self.config
        limits = np.array(
            [
                cfg.commit_forward_mps,
                cfg.max_right_mps,
                cfg.max_down_mps,
                cfg.max_yaw_rate_rps,
            ],
            dtype=np.float64,
        )
        accelerations = np.array(
            [
                cfg.max_forward_acceleration,
                cfg.max_lateral_acceleration,
                cfg.max_vertical_acceleration,
                cfg.max_yaw_acceleration,
            ],
            dtype=np.float64,
        )
        desired = np.clip(desired, -limits, limits)
        maximum_delta = accelerations * dt
        slew_limited = self._last_command + np.clip(
            desired - self._last_command, -maximum_delta, maximum_delta
        )
        alpha = float(np.clip(cfg.command_lpf_alpha, 0.0, 1.0))
        filtered = (1.0 - alpha) * self._last_command + alpha * slew_limited
        self._last_command = filtered
        return filtered

    def _approach_speed(
        self,
        detection: GateDetection,
        alignment_error: float,
        error_rate: float,
    ) -> float:
        cfg = self.config
        alignment_quality = float(
            np.clip(
                1.0 - alignment_error / cfg.maximum_alignment_error, 0.0, 1.0
            )
        )
        confidence_quality = float(
            np.clip(
                (detection.confidence - cfg.minimum_detection_confidence)
                / max(1e-6, 1.0 - cfg.minimum_detection_confidence),
                0.0,
                1.0,
            )
        )
        center_speed = math.hypot(detection.velocity_x, detection.velocity_y)
        stability_quality = float(np.clip(1.0 - center_speed / 1.2, 0.20, 1.0))
        edge_quality = float(
            np.clip(
                1.0
                - max(abs(detection.normalized_x), abs(detection.normalized_y))
                / 0.95,
                0.15,
                1.0,
            )
        )
        worsening_quality = math.exp(-max(0.0, error_rate) * 1.8)
        prediction_quality = 0.30 if detection.predicted else 1.0
        quality = (
            alignment_quality
            * confidence_quality
            * stability_quality
            * edge_quality
            * worsening_quality
            * prediction_quality
        )
        return cfg.minimum_approach_mps + (
            cfg.maximum_approach_mps - cfg.minimum_approach_mps
        ) * quality

    def update(
        self, detection: Optional[GateDetection], now: float
    ) -> NavigationCommand:
        cfg = self.config
        dt = self._dt(now)
        if self._state_since is None:
            self._state_since = now

        usable = self._usable(detection, cfg.minimum_detection_confidence)
        measured = bool(usable and detection is not None and not detection.predicted)
        if usable:
            assert detection is not None
            self._last_seen_at = now
            if abs(detection.normalized_x) > cfg.center_deadband:
                self._last_direction = (
                    1.0 if detection.normalized_x > 0.0 else -1.0
                )

        horizontal = vertical = 0.0
        alignment_error = 1.0
        if usable:
            horizontal, vertical, alignment_error = self._errors(detection)

        if self.state == NavigationState.SEARCH:
            if measured:
                self._transition(NavigationState.TRACK, now, force=True)
        elif self.state == NavigationState.TRACK:
            if not usable:
                self._transition(NavigationState.RECOVER, now)
            elif (
                measured
                and detection.stable_frames >= cfg.track_confirmation_frames
                and detection.confidence >= cfg.reliable_confidence
            ):
                self._transition(NavigationState.ALIGN_AND_APPROACH, now)
        elif self.state == NavigationState.ALIGN_AND_APPROACH:
            if not measured:
                self._transition(NavigationState.RECOVER, now)
            elif alignment_error >= cfg.recovery_alignment_error:
                self._transition(NavigationState.RECOVER, now)
            else:
                center_speed = math.hypot(
                    detection.velocity_x, detection.velocity_y
                )
                ready = (
                    alignment_error <= cfg.commit_alignment_tolerance
                    and detection.opening_area_ratio
                    >= cfg.commit_opening_area_ratio
                    and detection.confidence >= cfg.commit_minimum_confidence
                    and detection.stable_frames >= cfg.commit_stable_frames
                    and center_speed <= cfg.commit_max_center_speed
                    and abs(detection.size_rate) <= cfg.commit_max_size_rate
                )
                if ready:
                    self._transition(NavigationState.COMMIT, now)
        elif self.state == NavigationState.COMMIT:
            # Once committed, disappearance is expected: punch through instead
            # of reacting to orange edges leaving the image.
            if not measured:
                self._transition(NavigationState.PASS_THROUGH, now, force=True)
            elif (
                self._state_since is not None
                and now - self._state_since >= cfg.commit_maximum_duration_s
            ):
                self._transition(NavigationState.PASS_THROUGH, now, force=True)
        elif self.state == NavigationState.PASS_THROUGH:
            if (
                self._state_since is not None
                and now - self._state_since >= cfg.pass_through_duration_s
            ):
                self._transition(NavigationState.SEARCH, now, force=True)
        elif self.state == NavigationState.RECOVER:
            if measured:
                self._transition(NavigationState.TRACK, now, force=True)
            elif (
                self._last_seen_at is None
                or now - self._last_seen_at >= cfg.recover_local_duration_s
            ):
                self._transition(NavigationState.SEARCH, now, force=True)

        desired = np.zeros(4, dtype=np.float64)
        if self.state == NavigationState.SEARCH:
            desired[:] = (
                cfg.search_forward_mps,
                0.0,
                0.0,
                cfg.search_yaw_rate_rps * self._last_direction,
            )
        elif self.state == NavigationState.TRACK:
            desired[0] = cfg.track_forward_mps if usable else 0.0
            if usable:
                scale = cfg.track_alignment_scale
                desired[1] = scale * cfg.lateral_kp * horizontal
                desired[2] = scale * cfg.vertical_kp * vertical
                desired[3] = scale * cfg.horizontal_yaw_kp * horizontal
        elif self.state == NavigationState.ALIGN_AND_APPROACH and usable:
            error_rate = (
                alignment_error - self._previous_alignment_error
            ) / dt
            desired[0] = self._approach_speed(
                detection, alignment_error, error_rate
            )
            desired[1] = (
                cfg.lateral_kp * horizontal
                + cfg.lateral_kd * detection.velocity_x
            )
            desired[2] = (
                cfg.vertical_kp * vertical
                + cfg.vertical_kd * detection.velocity_y
            )
            desired[3] = (
                cfg.horizontal_yaw_kp * horizontal
                + cfg.horizontal_yaw_kd * detection.velocity_x
            )
            self._last_alignment_command[:] = (
                desired[1],
                desired[2],
                desired[3],
            )
        elif self.state == NavigationState.COMMIT:
            desired[0] = cfg.commit_forward_mps
            desired[1:] = 0.22 * self._last_alignment_command
            if measured:
                desired[2] += 0.15 * cfg.vertical_kp * vertical
                desired[3] += 0.12 * cfg.horizontal_yaw_kp * horizontal
        elif self.state == NavigationState.PASS_THROUGH:
            desired[0] = cfg.commit_forward_mps
            desired[1:] = 0.18 * self._last_alignment_command
        elif self.state == NavigationState.RECOVER:
            desired[0] = cfg.recover_forward_mps
            desired[3] = (
                0.55 * cfg.search_yaw_rate_rps * self._last_direction
            )

        conditioned = self._condition(desired, dt)
        self._last_update = now
        self._previous_alignment_error = alignment_error
        return NavigationCommand(
            forward_mps=float(max(0.0, conditioned[0])),
            right_mps=float(conditioned[1]),
            down_mps=float(conditioned[2]),
            yaw_rate_rps=float(conditioned[3]),
            state=self.state,
            confidence=detection.confidence if usable else 0.0,
            predicted=bool(detection.predicted) if usable else False,
            alignment_error=alignment_error,
        )


def q2_demo_navigation_config() -> NavigationConfig:
    """Reproduce the gate-passing profile used by collect_demos.py.

    The mapping preserves the demonstrated physical behavior through Q2's
    velocity planner: forward lean 0.10 -> 1.0 m/s, bank gain 0.30 ->
    3.0 m/s lateral target, yaw gain 0.8 normalized -> 2.4 rad/s before
    the demonstrated 1.05 rad/s cap, and vertical thrust gain 0.4 ->
    0.8 m/s down-velocity gain.
    """
    return NavigationConfig(
        center_deadband=0.0,
        vertical_setpoint_normalized=2.0 * 0.58 - 1.0,
        vertical_deadband=0.30,
        vertical_control_min_area_ratio=40.0 / 4096.0,
        search_forward_mps=1.0,
        search_yaw_rate_rps=0.0,
        track_forward_mps=1.0,
        minimum_approach_mps=1.0,
        maximum_approach_mps=1.0,
        commit_forward_mps=1.0,
        recover_forward_mps=1.0,
        horizontal_yaw_kp=2.4,
        horizontal_yaw_kd=0.0,
        lateral_kp=3.0,
        lateral_kd=0.0,
        vertical_kp=0.8,
        vertical_kd=0.0,
        max_right_mps=3.0,
        max_down_mps=0.60,
        max_yaw_rate_rps=1.05,
        max_forward_acceleration=100.0,
        max_lateral_acceleration=100.0,
        max_vertical_acceleration=100.0,
        max_yaw_acceleration=100.0,
        command_lpf_alpha=1.0,
        track_alignment_scale=1.0,
    )
