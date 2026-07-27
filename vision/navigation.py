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
    commit_horizontal_tolerance: float = 0.12
    commit_abort_alignment_tolerance: float = 0.20
    commit_abort_horizontal_tolerance: float = 0.20
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
    minimum_forward_mps: float = 0.0
    maximum_forward_mps: float = 2.0
    approach_slowdown_start_area_ratio: float = 1.0
    approach_slowdown_end_area_ratio: float = 1.0
    close_approach_mps: float = 0.0

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

    # Optional blue-lane assist. Zero gains keep the generic navigator's
    # historical behavior; q2_demo_navigation_config enables conservative
    # centering while keeping the orange gate authoritative.
    path_minimum_confidence: float = 0.25
    path_heading_weight: float = 0.60
    path_lateral_kp: float = 0.0
    path_yaw_kp: float = 0.0
    path_max_right_mps: float = 0.0
    path_max_yaw_rate_rps: float = 0.0
    path_blend_with_gate: float = 0.20
    path_pass_through_delay_s: float = 0.0
    path_pass_through_weight: float = 0.0
    next_gate_minimum_primary_area_ratio: float = 1.0
    next_gate_maximum_primary_horizontal: float = 0.0
    next_gate_lateral_kp: float = 0.0
    next_gate_yaw_kp: float = 0.0
    next_gate_max_right_mps: float = 0.0
    next_gate_max_yaw_rate_rps: float = 0.0
    prepass_lookahead_weight: float = 1.0
    pass_through_lookahead_delay_s: float = 0.20
    framing_soft_edge_normalized: float = 0.58
    framing_hard_edge_normalized: float = 0.82
    framing_retreat_mps: float = -0.12


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
        self._pending_next_gate_horizontal: Optional[float] = None

    def reset(self) -> None:
        self.state = NavigationState.SEARCH
        self._state_since = None
        self._last_update = None
        self._last_seen_at = None
        self._last_direction = 1.0
        self._last_command[:] = 0.0
        self._last_alignment_command[:] = 0.0
        self._previous_alignment_error = 1.0
        self._pending_next_gate_horizontal = None

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
                cfg.maximum_forward_mps,
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
        speed = cfg.minimum_approach_mps + (
            cfg.maximum_approach_mps - cfg.minimum_approach_mps
        ) * quality
        slowdown_span = (
            cfg.approach_slowdown_end_area_ratio
            - cfg.approach_slowdown_start_area_ratio
        )
        if slowdown_span > 1e-6:
            proximity = float(
                np.clip(
                    (
                        detection.opening_area_ratio
                        - cfg.approach_slowdown_start_area_ratio
                    )
                    / slowdown_span,
                    0.0,
                    1.0,
                )
            )
            speed = (
                (1.0 - proximity) * speed
                + proximity * cfg.close_approach_mps
            )
        return speed

    def confirm_gate_pass(self, now: float) -> None:
        """Release the old visual target after the race timer confirms a pass."""
        self._last_seen_at = now
        self._last_alignment_command[:] = 0.0
        self._last_command[1:] = 0.0
        if (
            self._pending_next_gate_horizontal is not None
            and abs(self._pending_next_gate_horizontal) > 0.02
        ):
            self._last_direction = (
                1.0 if self._pending_next_gate_horizontal > 0.0 else -1.0
            )
        # Race confirmation arrives after the physical crossing. Do not add
        # another blind pass-through interval; stop and acquire the largest
        # visible gate immediately.
        self._transition(NavigationState.SEARCH, now, force=True)

    def update(
        self,
        detection: Optional[GateDetection],
        now: float,
        path: Optional[object] = None,
        next_gate_horizontal: Optional[float] = None,
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
            if measured and abs(detection.normalized_x) > cfg.center_deadband:
                self._last_direction = (
                    1.0 if detection.normalized_x > 0.0 else -1.0
                )

        horizontal = vertical = 0.0
        alignment_error = 1.0
        if usable:
            horizontal, vertical, alignment_error = self._errors(detection)

        starting_state = self.state
        lookahead_supported = bool(
            measured
            and detection is not None
            and next_gate_horizontal is not None
            and detection.opening_area_ratio
            >= cfg.next_gate_minimum_primary_area_ratio
            and abs(detection.normalized_x)
            <= cfg.next_gate_maximum_primary_horizontal
        )
        if (
            lookahead_supported
            and starting_state
            in (
                NavigationState.TRACK,
                NavigationState.ALIGN_AND_APPROACH,
                NavigationState.COMMIT,
            )
        ):
            self._pending_next_gate_horizontal = float(
                np.clip(next_gate_horizontal, -1.0, 1.0)
            )
        elif starting_state == NavigationState.SEARCH and measured:
            # This is the newly acquired gate. Do not reuse the turn that led
            # from the previous gate unless a fresh farther gate is also seen.
            self._pending_next_gate_horizontal = None

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
                    abs(horizontal) <= cfg.commit_horizontal_tolerance
                    and abs(vertical) <= cfg.commit_alignment_tolerance
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
            # Stay committed only while the opening remains centered. A gate
            # drifting sideways is not a successful pass trajectory.
            if (
                measured
                and (
                    abs(horizontal)
                    > cfg.commit_abort_horizontal_tolerance
                    or abs(vertical)
                    > cfg.commit_abort_alignment_tolerance
                )
            ):
                self._transition(
                    NavigationState.ALIGN_AND_APPROACH, now, force=True
                )
            # Once a centered commit makes the gate disappear, punch through
            # instead of reacting to orange edges leaving the image.
            elif not measured:
                self._last_command[1:] = 0.0
                self._transition(NavigationState.PASS_THROUGH, now, force=True)
            elif (
                self._state_since is not None
                and now - self._state_since >= cfg.commit_maximum_duration_s
            ):
                self._last_command[1:] = 0.0
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
            desired[1:] = 0.35 * self._last_alignment_command
            if measured:
                # Recompute lateral centering throughout commit. Holding only
                # the pre-commit command let inertia carry gate two across the
                # image and into the frame.
                desired[1] += 0.75 * cfg.lateral_kp * horizontal
                desired[2] += 0.15 * cfg.vertical_kp * vertical
                desired[3] += 0.20 * cfg.horizontal_yaw_kp * horizontal
        elif self.state == NavigationState.PASS_THROUGH:
            desired[0] = cfg.commit_forward_mps
            # The final correction for the passed gate points back toward that
            # gate and may have the opposite sign from the next turn. Fly
            # straight until the frame has cleared, then translate and yaw
            # toward the latched next gate.
            pass_elapsed = (
                0.0
                if self._state_since is None
                else max(0.0, now - self._state_since)
            )
            if (
                self._pending_next_gate_horizontal is not None
                and pass_elapsed >= cfg.pass_through_lookahead_delay_s
            ):
                lookahead = self._pending_next_gate_horizontal
                desired[1] = float(
                    np.clip(
                        cfg.next_gate_lateral_kp * lookahead,
                        -cfg.next_gate_max_right_mps,
                        cfg.next_gate_max_right_mps,
                    )
                )
                desired[3] = float(
                    np.clip(
                        cfg.next_gate_yaw_kp * lookahead,
                        -cfg.next_gate_max_yaw_rate_rps,
                        cfg.next_gate_max_yaw_rate_rps,
                    )
                )
        elif self.state == NavigationState.RECOVER:
            desired[0] = cfg.recover_forward_mps
            desired[3] = (
                0.55 * cfg.search_yaw_rate_rps * self._last_direction
            )

        next_gate_usable = bool(
            lookahead_supported
            and self.state
            in (NavigationState.TRACK, NavigationState.ALIGN_AND_APPROACH)
        )
        if next_gate_usable:
            lookahead = float(np.clip(next_gate_horizontal, -1.0, 1.0))
            proximity = float(
                np.clip(
                    detection.opening_area_ratio
                    / max(cfg.commit_opening_area_ratio, 1e-6),
                    0.0,
                    1.0,
                )
            )
            lookahead_weight = float(
                np.clip(cfg.prepass_lookahead_weight, 0.0, 1.0)
            )
            desired[1] += lookahead_weight * proximity * float(
                np.clip(
                    cfg.next_gate_lateral_kp * lookahead,
                    -cfg.next_gate_max_right_mps,
                    cfg.next_gate_max_right_mps,
                )
            )
            desired[3] += lookahead_weight * proximity * float(
                np.clip(
                    cfg.next_gate_yaw_kp * lookahead,
                    -cfg.next_gate_max_yaw_rate_rps,
                    cfg.next_gate_max_yaw_rate_rps,
                )
            )

        # Never drive a measured gate out of the camera. As its center enters
        # the outer frame margin, progressively remove forward speed and then
        # back away while horizontal/vertical centering remains active.
        if (
            usable
            and detection is not None
            and self.state
            not in (NavigationState.COMMIT, NavigationState.PASS_THROUGH)
        ):
            frame_edge = max(
                abs(detection.normalized_x),
                abs(detection.normalized_y),
            )
            soft = cfg.framing_soft_edge_normalized
            hard = max(soft + 1e-6, cfg.framing_hard_edge_normalized)
            if frame_edge > soft:
                edge_progress = float(
                    np.clip((frame_edge - soft) / (hard - soft), 0.0, 1.0)
                )
                safe_forward = (
                    (1.0 - edge_progress) * max(0.0, desired[0])
                    + edge_progress * cfg.framing_retreat_mps
                )
                desired[0] = min(desired[0], safe_forward)

        path_found = bool(
            path is not None
            and getattr(path, 'found', False)
            and getattr(path, 'confidence', 0.0)
            >= cfg.path_minimum_confidence
        )
        path_assist_allowed = self.state != NavigationState.COMMIT
        path_weight = cfg.path_blend_with_gate if usable else 1.0
        if self.state == NavigationState.PASS_THROUGH:
            pass_elapsed = (
                0.0
                if self._state_since is None
                else max(0.0, now - self._state_since)
            )
            path_assist_allowed = bool(
                cfg.path_pass_through_weight > 0.0
                and pass_elapsed >= cfg.path_pass_through_delay_s
            )
            path_weight = cfg.path_pass_through_weight
        if path_found and path_assist_allowed and (
            cfg.path_lateral_kp != 0.0 or cfg.path_yaw_kp != 0.0
        ):
            path_error = float(
                np.clip(
                    getattr(path, 'normalized_offset', 0.0)
                    + cfg.path_heading_weight
                    * getattr(path, 'normalized_heading', 0.0),
                    -1.0,
                    1.0,
                )
            )
            desired[1] += path_weight * float(
                np.clip(
                    cfg.path_lateral_kp * path_error,
                    -cfg.path_max_right_mps,
                    cfg.path_max_right_mps,
                )
            )
            desired[3] += path_weight * float(
                np.clip(
                    cfg.path_yaw_kp * path_error,
                    -cfg.path_max_yaw_rate_rps,
                    cfg.path_max_yaw_rate_rps,
                )
            )

        conditioned = self._condition(desired, dt)
        self._last_update = now
        self._previous_alignment_error = alignment_error
        return NavigationCommand(
            forward_mps=float(
                max(cfg.minimum_forward_mps, conditioned[0])
            ),
            right_mps=float(conditioned[1]),
            down_mps=float(conditioned[2]),
            yaw_rate_rps=float(conditioned[3]),
            state=self.state,
            confidence=detection.confidence if usable else 0.0,
            predicted=bool(detection.predicted) if usable else False,
            alignment_error=alignment_error,
        )


def q2_demo_navigation_config() -> NavigationConfig:
    """Adapt the demonstrated gate-passing profile to the multi-gate course.

    The attitude loop retains collect_demos.py's physical conventions. Gate
    centering deliberately favors lateral bank over yaw so rotating the camera
    cannot make an off-axis gate look aligned before the vehicle has moved
    toward its opening.
    """
    return NavigationConfig(
        # The recorded first pass peaked at 6.49% frame opening area before
        # dropout. Commit earlier, while the centered opening is still measured,
        # instead of falling into recovery and acquiring background orange.
        commit_opening_area_ratio=0.030,
        commit_alignment_tolerance=0.10,
        commit_horizontal_tolerance=0.05,
        commit_abort_alignment_tolerance=0.16,
        commit_abort_horizontal_tolerance=0.08,
        track_confirmation_frames=1,
        commit_stable_frames=3,
        commit_max_center_speed=2.0,
        commit_max_size_rate=100.0,
        center_deadband=0.0,
        # Hold the opening lower in the camera (drone higher in the opening)
        # and start correcting sooner to clear the first gate's bottom rail.
        vertical_setpoint_normalized=2.0 * 0.62 - 1.0,
        vertical_deadband=0.20,
        # Distant gates still control altitude so they cannot drift out of the
        # top or bottom of the frame.
        vertical_control_min_area_ratio=0.0,
        search_forward_mps=0.0,
        search_yaw_rate_rps=0.16,
        track_forward_mps=0.24,
        minimum_approach_mps=0.18,
        maximum_approach_mps=0.36,
        commit_forward_mps=0.40,
        recover_forward_mps=0.0,
        minimum_forward_mps=-0.12,
        maximum_forward_mps=0.65,
        approach_slowdown_start_area_ratio=0.008,
        approach_slowdown_end_area_ratio=0.025,
        close_approach_mps=-0.12,
        # Horizontal error primarily commands bank/translation. The earlier
        # 2.4 yaw gain centered gate two by rotating the camera while the
        # vehicle remained outside its lateral flight line.
        horizontal_yaw_kp=0.42,
        horizontal_yaw_kd=0.0,
        lateral_kp=1.20,
        lateral_kd=0.0,
        vertical_kp=0.8,
        vertical_kd=0.0,
        max_right_mps=0.75,
        max_down_mps=0.60,
        max_yaw_rate_rps=0.28,
        max_forward_acceleration=1.4,
        max_lateral_acceleration=1.8,
        max_vertical_acceleration=1.4,
        max_yaw_acceleration=1.0,
        command_lpf_alpha=0.45,
        track_alignment_scale=1.0,
        # Blue-path steering is deliberately disabled for Q2. Only accepted
        # orange gates and orange next-gate look-ahead can affect flight.
        path_lateral_kp=0.0,
        path_yaw_kp=0.0,
        path_max_right_mps=0.0,
        path_max_yaw_rate_rps=0.0,
        path_blend_with_gate=0.0,
        path_pass_through_delay_s=0.0,
        path_pass_through_weight=0.0,
        next_gate_minimum_primary_area_ratio=0.008,
        next_gate_maximum_primary_horizontal=0.18,
        next_gate_lateral_kp=0.90,
        next_gate_yaw_kp=0.25,
        next_gate_max_right_mps=0.45,
        next_gate_max_yaw_rate_rps=0.12,
        # Retain the next gate's direction, but never let it pull the drone
        # away from the center of the gate that has not yet been cleared.
        prepass_lookahead_weight=0.0,
        pass_through_lookahead_delay_s=0.70,
    )
