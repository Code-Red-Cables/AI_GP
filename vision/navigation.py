"""Focused gate-opening navigation state machine and timestamped PD control."""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from control.pid import PIDConfig, PIDController

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
    # Once strict close-and-centered entry criteria are satisfied, ignore
    # unreliable close-range image drift and fly level through the opening.
    commit_straight_through: bool = False
    maximum_alignment_error: float = 0.65
    recovery_alignment_error: float = 0.82
    center_deadband: float = 0.035
    # Optical normalized-y that represents body-forward flight. This is zero
    # for a level camera and positive for Q2's upward-tilted camera.
    vertical_setpoint_normalized: float = 0.0
    post_pass_vertical_setpoint_normalized: Optional[float] = None
    vertical_deadband: float = 0.035
    vertical_descent_deadband: float = 0.035
    vertical_control_min_area_ratio: float = 0.0
    vertical_control_max_horizontal_error: float = math.inf
    track_alignment_scale: float = 0.45
    recover_forward_mps: float = 0.0
    recover_close_forward_mps: float = 0.0
    recover_lateral_mps: float = 0.0
    recover_prediction_scale: float = 0.0

    minimum_state_duration_s: float = 0.10
    commit_maximum_duration_s: float = 0.65
    pass_through_duration_s: float = 0.85
    recover_local_duration_s: float = 1.35

    search_forward_mps: float = 0.04
    search_close_forward_mps: float = 0.0
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
    severe_horizontal_error_normalized: float = math.inf
    severe_horizontal_forward_cap_mps: float = math.inf

    horizontal_yaw_kp: float = 1.05
    horizontal_yaw_ki: float = 0.0
    horizontal_yaw_kd: float = 0.12
    lateral_kp: float = 0.18
    lateral_ki: float = 0.0
    lateral_kd: float = 0.04
    yaw_first_lateral_full_normalized: float = 1.0
    yaw_first_lateral_zero_normalized: float = 1.0
    yaw_first_lateral_minimum_scale: float = 0.0
    yaw_capture_lateral_scale: float = 1.0
    inward_capture_max_lateral_mps: float = math.inf
    inward_capture_max_yaw_rate_rps: float = math.inf
    horizontal_capture_release_normalized: float = 0.0
    horizontal_capture_prediction_horizon_s: float = 0.0
    horizontal_yaw_capture_prediction_horizon_s: float = 0.0
    horizontal_yaw_capture_max_error_normalized: float = math.inf
    horizontal_capture_outward_release_speed: float = math.inf
    horizontal_yaw_capture_outward_release_speed: float = math.inf
    horizontal_capture_brake_lateral_gain: float = 0.0
    horizontal_capture_brake_yaw_gain: float = 0.0
    lateral_countersteer_gain: float = 1.0
    countersteer_max_lateral_mps: float = math.inf
    yaw_countersteer_gain: float = 1.0
    countersteer_forward_floor_mps: float = 0.0
    vertical_kp: float = 0.82
    vertical_ki: float = 0.0
    vertical_kd: float = 0.10
    vertical_countersteer_max_mps: float = 0.0
    image_pid_integral_limit: float = 0.35
    image_pid_derivative_filter_tau_s: float = 0.0

    max_right_mps: float = 0.42
    max_up_mps: float = 0.62
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
    requested_forward_mps: float = 0.0
    framing_limited: bool = False
    framing_edge: float = 0.0


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
        self._horizontal_capture_side = 0.0
        self._yaw_capture_side = 0.0
        self._confirmed_gate_passes = 0
        self._lateral_pid = self._image_pid(
            self.config.lateral_kp,
            self.config.lateral_ki,
            self.config.lateral_kd,
            self.config.max_right_mps,
        )
        self._vertical_pid = PIDController(
            PIDConfig(
                kp=self.config.vertical_kp,
                ki=self.config.vertical_ki,
                kd=self.config.vertical_kd,
                output_min=-self.config.max_up_mps,
                output_max=self.config.max_down_mps,
                integral_min=-self.config.image_pid_integral_limit,
                integral_max=self.config.image_pid_integral_limit,
                derivative_filter_tau_s=(
                    self.config.image_pid_derivative_filter_tau_s
                ),
                minimum_dt_s=1.0 / 120.0,
                maximum_dt_s=0.20,
            )
        )
        self._yaw_pid = self._image_pid(
            self.config.horizontal_yaw_kp,
            self.config.horizontal_yaw_ki,
            self.config.horizontal_yaw_kd,
            self.config.max_yaw_rate_rps,
        )

    def _image_pid(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_limit: float,
    ) -> PIDController:
        return PIDController(
            PIDConfig(
                kp=kp,
                ki=ki,
                kd=kd,
                output_min=-output_limit,
                output_max=output_limit,
                integral_min=-self.config.image_pid_integral_limit,
                integral_max=self.config.image_pid_integral_limit,
                derivative_filter_tau_s=(
                    self.config.image_pid_derivative_filter_tau_s
                ),
                minimum_dt_s=1.0 / 120.0,
                maximum_dt_s=0.20,
            )
        )

    def _reset_image_pids(self) -> None:
        self._lateral_pid.reset()
        self._vertical_pid.reset()
        self._yaw_pid.reset()

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
        self._horizontal_capture_side = 0.0
        self._yaw_capture_side = 0.0
        self._confirmed_gate_passes = 0
        self._reset_image_pids()

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
        # Each flight phase has a different target contract. Do not carry
        # image-error integral or derivative history into the next phase.
        self._reset_image_pids()

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
        vertical_setpoint = cfg.vertical_setpoint_normalized
        if (
            self._confirmed_gate_passes > 0
            and cfg.post_pass_vertical_setpoint_normalized is not None
        ):
            vertical_setpoint = cfg.post_pass_vertical_setpoint_normalized
        vertical_error = detection.normalized_y - vertical_setpoint
        vertical_deadband = (
            cfg.vertical_descent_deadband
            if vertical_error > 0.0
            else cfg.vertical_deadband
        )
        if (
            detection.opening_area_ratio
            < cfg.vertical_control_min_area_ratio
            or abs(detection.normalized_x)
            > cfg.vertical_control_max_horizontal_error
            or abs(vertical_error) <= vertical_deadband
        ):
            vertical = 0.0
        else:
            vertical = math.copysign(
                abs(vertical_error) - vertical_deadband,
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
        lower_limits = -limits
        # Up and down need separate bounds. Forward pitch moves a gate upward
        # in the image even without a matching altitude error; treating that
        # motion as an unrestricted climb request drove Q2 into the ceiling.
        lower_limits[2] = -cfg.max_up_mps
        desired = np.minimum(np.maximum(desired, lower_limits), limits)
        maximum_delta = accelerations * dt
        slew_limited = self._last_command + np.clip(
            desired - self._last_command, -maximum_delta, maximum_delta
        )
        alpha = float(np.clip(cfg.command_lpf_alpha, 0.0, 1.0))
        filtered = (1.0 - alpha) * self._last_command + alpha * slew_limited
        self._last_command = filtered
        return filtered

    def _horizontal_control(
        self,
        detection: GateDetection,
        horizontal: float,
        dt: float,
    ) -> tuple[float, float]:
        """Return PID lateral/yaw commands with image-velocity braking."""
        cfg = self.config
        # Positive image-x means the gate is right of center, so positive
        # lateral/yaw commands move and turn right. The tracker reports
        # d(error)/dt; PID expects measurement rate and negates it internally.
        lateral = self._lateral_pid.update(
            horizontal,
            dt,
            measurement_rate=-detection.velocity_x,
        )
        yaw = self._yaw_pid.update(
            horizontal,
            dt,
            measurement_rate=-detection.velocity_x,
        )
        # Preserve the prediction horizon (the command sign-change point).
        # Once an off-center gate starts moving inward, stop adding a large
        # same-direction lateral command. The camera motion is the only
        # lateral-velocity feedback available here; this capture limit keeps
        # the initial turn responsive while preventing sideways momentum from
        # continuing to build before the derivative term changes command sign.
        image_side = detection.normalized_x
        moving_inward = image_side * detection.velocity_x < 0.0
        release = max(0.0, cfg.horizontal_capture_release_normalized)
        horizon = max(0.0, cfg.horizontal_capture_prediction_horizon_s)
        projected_side = image_side + horizon * detection.velocity_x
        capture_ready = bool(
            moving_inward
            and (
                image_side * projected_side <= 0.0
                or abs(projected_side) <= release
            )
        )
        moving_outward_again = bool(
            self._horizontal_capture_side * detection.velocity_x
            > max(0.0, cfg.horizontal_capture_outward_release_speed)
        )
        if (
            abs(image_side) <= release
            or self._horizontal_capture_side * image_side < 0.0
            or moving_outward_again
        ):
            self._horizontal_capture_side = 0.0
        if capture_ready and abs(image_side) > release:
            self._horizontal_capture_side = math.copysign(1.0, image_side)
        capture_active = (
            self._horizontal_capture_side * image_side > 0.0
            and abs(image_side) > release
        )
        yaw_horizon = max(
            horizon,
            cfg.horizontal_yaw_capture_prediction_horizon_s,
        )
        yaw_projected_side = (
            image_side + yaw_horizon * detection.velocity_x
        )
        yaw_capture_ready = bool(
            moving_inward
            and abs(image_side)
            <= cfg.horizontal_yaw_capture_max_error_normalized
            and (
                image_side * yaw_projected_side <= 0.0
                or abs(yaw_projected_side) <= release
            )
        )
        yaw_moving_outward_again = bool(
            self._yaw_capture_side * detection.velocity_x
            > max(
                0.0,
                cfg.horizontal_yaw_capture_outward_release_speed,
            )
        )
        if (
            abs(image_side) <= release
            or self._yaw_capture_side * image_side < 0.0
            or yaw_moving_outward_again
        ):
            self._yaw_capture_side = 0.0
        if yaw_capture_ready and abs(image_side) > release:
            self._yaw_capture_side = math.copysign(1.0, image_side)
        yaw_capture_active = (
            self._yaw_capture_side * image_side > 0.0
            and abs(image_side) > release
        )
        if capture_active and image_side * lateral > 0.0:
            if (
                moving_inward
                and cfg.horizontal_capture_brake_lateral_gain > 0.0
            ):
                lateral = float(
                    np.clip(
                        cfg.horizontal_capture_brake_lateral_gain
                        * detection.velocity_x,
                        -cfg.countersteer_max_lateral_mps,
                        cfg.countersteer_max_lateral_mps,
                    )
                )
            else:
                lateral = float(
                    np.clip(
                        lateral,
                        -cfg.inward_capture_max_lateral_mps,
                        cfg.inward_capture_max_lateral_mps,
                    )
                )
        if yaw_capture_active and image_side * yaw > 0.0:
            if moving_inward and cfg.horizontal_capture_brake_yaw_gain > 0.0:
                yaw = float(
                    np.clip(
                        cfg.horizontal_capture_brake_yaw_gain
                        * detection.velocity_x,
                        -cfg.max_yaw_rate_rps,
                        cfg.max_yaw_rate_rps,
                    )
                )
            else:
                yaw = float(
                    np.clip(
                        yaw,
                        -cfg.inward_capture_max_yaw_rate_rps,
                        cfg.inward_capture_max_yaw_rate_rps,
                    )
                )
        # Only amplify a command that opposes the gate's current image-side,
        # which is counter-steering intended to remove existing momentum.
        if not capture_active and image_side * lateral < 0.0:
            lateral *= cfg.lateral_countersteer_gain
            lateral = float(
                np.clip(
                    lateral,
                    -cfg.countersteer_max_lateral_mps,
                    cfg.countersteer_max_lateral_mps,
                )
            )
        if not yaw_capture_active and image_side * yaw < 0.0:
            yaw *= cfg.yaw_countersteer_gain
        yaw_first_span = (
            cfg.yaw_first_lateral_zero_normalized
            - cfg.yaw_first_lateral_full_normalized
        )
        if yaw_first_span > 1e-6:
            # A high, far-edge target initially moves farther outward when the
            # vehicle banks toward it because camera roll rotates that image
            # quadrant before lateral translation develops. Use yaw alone at
            # the extreme edge, then fade full translation back in as the gate
            # enters a geometrically safe capture corridor.
            if yaw_capture_active:
                # Heading is now gyro-braked. Transition from the yaw-first
                # acquisition phase to full bounded translation so the
                # airframe, not just the camera, enters the gate flight line.
                lateral_scale = float(
                    np.clip(cfg.yaw_capture_lateral_scale, 0.0, 1.0)
                )
            else:
                lateral_scale = float(
                    np.clip(
                        (
                            cfg.yaw_first_lateral_zero_normalized
                            - abs(image_side)
                        )
                        / yaw_first_span,
                        0.0,
                        1.0,
                    )
                )
                lateral_scale = max(
                    float(
                        np.clip(
                            cfg.yaw_first_lateral_minimum_scale,
                            0.0,
                            1.0,
                        )
                    ),
                    lateral_scale,
                )
            lateral *= lateral_scale
        return lateral, yaw

    def _vertical_control(
        self,
        detection: GateDetection,
        vertical: float,
        dt: float,
    ) -> float:
        """Return vertical control without reversing across a visible error."""
        if vertical == 0.0:
            return 0.0
        command = self._vertical_pid.update(
            vertical,
            dt,
            measurement_rate=-detection.velocity_y,
        )
        if vertical * command < 0.0:
            # Permit a bounded *downward* reversal only when a high gate is
            # already moving toward the setpoint. This brakes accumulated
            # climb before the image center crosses. The symmetric upward
            # reversal stays disabled: forward pitch can move a low gate
            # upward optically and previously drove the aircraft toward the
            # ceiling.
            moving_inward = vertical * detection.velocity_y < 0.0
            limit = max(0.0, self.config.vertical_countersteer_max_mps)
            if not moving_inward or command <= 0.0 or limit <= 0.0:
                return 0.0
            return float(np.clip(command, 0.0, limit))
        return command

    def _prediction_braking_control(
        self,
        detection: GateDetection,
        horizontal: float,
        vertical: float,
        dt: float,
    ) -> tuple[float, float, float]:
        """Brake measured image motion during a short inference dropout."""
        cfg = self.config
        lateral, yaw = self._horizontal_control(
            detection, horizontal, dt
        )
        if horizontal * detection.velocity_x < 0.0:
            # The target is already moving toward image center. Require at
            # least derivative-only countersteer so stale position error
            # cannot continue accelerating through the setpoint.
            lateral_brake = cfg.lateral_kd * detection.velocity_x
            yaw_brake = cfg.horizontal_yaw_kd * detection.velocity_x
            if detection.velocity_x < 0.0:
                lateral = min(lateral, lateral_brake)
                yaw = min(yaw, yaw_brake)
            else:
                lateral = max(lateral, lateral_brake)
                yaw = max(yaw, yaw_brake)

        down = self._vertical_control(detection, vertical, dt)
        if vertical * detection.velocity_y < 0.0:
            # Apply the same predictive braking vertically. This is limited to
            # a moving tracker prediction; a static unconfirmed target still
            # cannot steer the aircraft.
            vertical_brake = cfg.vertical_kd * detection.velocity_y
            if detection.velocity_y < 0.0:
                down = min(down, vertical_brake)
            else:
                down = max(down, vertical_brake)
        return lateral, down, yaw

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
        if (
            abs(detection.normalized_x)
            >= cfg.severe_horizontal_error_normalized
        ):
            speed = min(speed, cfg.severe_horizontal_forward_cap_mps)
        return speed

    def confirm_gate_pass(self, now: float) -> None:
        """Release the old visual target after the race timer confirms a pass."""
        self._confirmed_gate_passes += 1
        self._last_seen_at = now
        self._last_alignment_command[:] = 0.0
        self._last_command[1:] = 0.0
        self._horizontal_capture_side = 0.0
        self._yaw_capture_side = 0.0
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
            # A tracker extrapolation is not a new camera observation. Letting
            # predictions refresh this timestamp made RECOVER effectively
            # unbounded: each synthetic point extended the recovery window and
            # could keep banking after the real gate had left the frame.
            if measured:
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
            elif not measured:
                self._last_command[:] = 0.0
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
                not cfg.commit_straight_through
                and measured
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
                # Recovery has exhausted the last real observation. Do not
                # low-pass an old bank/descent/yaw command into blind SEARCH;
                # level the vehicle immediately so attitude control can brake
                # the motion before the gate is reacquired on the other side.
                self._last_command[1:] = 0.0
                self._transition(NavigationState.SEARCH, now, force=True)

        desired = np.zeros(4, dtype=np.float64)
        if self.state == NavigationState.SEARCH:
            desired[:] = (
                cfg.search_forward_mps,
                0.0,
                0.0,
                cfg.search_yaw_rate_rps * self._last_direction,
            )
            if (
                usable
                and detection is not None
                and detection.predicted
                and detection.opening_area_ratio
                >= cfg.commit_opening_area_ratio
            ):
                # Recovery expiry must not turn a close tracked gate into a
                # full stop/backoff. Preserve only a slow forward crawl; stale
                # lateral/yaw commands remain cleared until YOLO+HSV returns.
                desired[0] = max(
                    desired[0],
                    cfg.search_close_forward_mps,
                )
        elif self.state == NavigationState.TRACK:
            # A tracker prediction is useful for retaining identity, but it
            # does not satisfy the live YOLO+HSV gate confirmation contract.
            # Never approach it. During a short gap, however, its measured
            # image velocity may brake momentum already created by the last
            # confirmed frame; zeroing every axis here caused the observed
            # gate-two left/right oscillation between 3-5 FPS detections.
            desired[0] = cfg.track_forward_mps if measured else 0.0
            if measured:
                scale = cfg.track_alignment_scale
                lateral, yaw = self._horizontal_control(
                    detection, horizontal, dt
                )
                desired[1] = scale * lateral
                desired[2] = scale * self._vertical_control(
                    detection, vertical, dt
                )
                desired[3] = scale * yaw
            elif (
                usable
                and detection is not None
                and detection.predicted
                and (
                    abs(detection.velocity_x) > 0.01
                    or abs(detection.velocity_y) > 0.01
                )
            ):
                scale = cfg.recover_prediction_scale
                lateral, down, yaw = self._prediction_braking_control(
                    detection, horizontal, vertical, dt
                )
                desired[1] = scale * lateral
                desired[2] = scale * down
                desired[3] = scale * yaw
        elif self.state == NavigationState.ALIGN_AND_APPROACH and usable:
            error_rate = (
                alignment_error - self._previous_alignment_error
            ) / dt
            desired[0] = self._approach_speed(
                detection, alignment_error, error_rate
            )
            desired[1], desired[3] = self._horizontal_control(
                detection, horizontal, dt
            )
            if (
                detection.normalized_x * desired[1] < 0.0
                or detection.normalized_x * desired[3] < 0.0
            ):
                # Counter-steering removes lateral/yaw momentum; it is not a
                # reason to stop approaching a measured gate. Keep translating
                # forward while the independent axes brake the sideways motion.
                countersteer_floor = cfg.countersteer_forward_floor_mps
                if (
                    abs(detection.normalized_x)
                    >= cfg.severe_horizontal_error_normalized
                ):
                    countersteer_floor = min(
                        countersteer_floor,
                        cfg.severe_horizontal_forward_cap_mps,
                    )
                desired[0] = max(desired[0], countersteer_floor)
            desired[2] = self._vertical_control(
                detection, vertical, dt
            )
            self._last_alignment_command[:] = (
                desired[1],
                desired[2],
                desired[3],
            )
        elif self.state == NavigationState.COMMIT:
            desired[0] = cfg.commit_forward_mps
            if not cfg.commit_straight_through:
                desired[1:] = 0.35 * self._last_alignment_command
            if measured and not cfg.commit_straight_through:
                # Recompute lateral centering throughout commit. Holding only
                # the pre-commit command let inertia carry gate two across the
                # image and into the frame.
                lateral, yaw = self._horizontal_control(
                    detection, horizontal, dt
                )
                desired[1] += 0.75 * lateral
                desired[2] += 0.15 * self._vertical_control(
                    detection, vertical, dt
                )
                desired[3] += 0.20 * yaw
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
            if (
                usable
                and detection is not None
                and detection.predicted
                and cfg.recover_prediction_scale > 0.0
            ):
                # A short detector dropout still carries a bounded tracker
                # prediction. Use its image motion to finish braking instead
                # of steering toward the side on which the gate was first
                # seen; that stale side sent gate two out of the opposite edge.
                scale = cfg.recover_prediction_scale
                lateral, down, yaw = self._prediction_braking_control(
                    detection, horizontal, vertical, dt
                )
                desired[1] = scale * lateral
                desired[2] = scale * down
                desired[3] = scale * yaw
                if (
                    detection.opening_area_ratio
                    >= cfg.commit_opening_area_ratio
                ):
                    # A close gate that vanished before COMMIT was not centered
                    # enough for a safe blind pass. Continuing at recovery
                    # speed clipped the frame and the collision rebounded the
                    # drone backward even though nav_fwd stayed positive.
                    desired[0] = min(
                        desired[0],
                        cfg.recover_close_forward_mps,
                    )
            else:
                desired[1] = (
                    cfg.recover_lateral_mps * self._last_direction
                )
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

        requested_forward_mps = float(desired[0])
        framing_limited = False
        framing_edge = 0.0
        # Never drive a measured gate out of the camera. As its center enters
        # the outer frame margin, progressively remove forward speed while
        # horizontal/vertical centering remains active.
        #
        # A gate low in the image is the important exception: forward pitch
        # and a positive body-down command both move that gate back upward.
        # Treating the lower image edge like a lateral edge reduced forward
        # speed to zero on Training One's second gate even while the vehicle
        # was correctly descending toward it. Keep horizontal framing active,
        # but do not let a corrective descent cancel the approach.
        if (
            usable
            and detection is not None
            and self.state
            not in (NavigationState.COMMIT, NavigationState.PASS_THROUGH)
        ):
            horizontal_edge = abs(detection.normalized_x)
            vertical_edge = abs(detection.normalized_y)
            soft = cfg.framing_soft_edge_normalized
            hard = max(soft + 1e-6, cfg.framing_hard_edge_normalized)
            low_gate_correcting_inward = bool(
                detection.normalized_y > soft and desired[2] > 0.0
            )
            frame_edge = (
                horizontal_edge
                if low_gate_correcting_inward
                else max(horizontal_edge, vertical_edge)
            )
            framing_edge = float(frame_edge)
            if frame_edge > soft:
                edge_progress = float(
                    np.clip((frame_edge - soft) / (hard - soft), 0.0, 1.0)
                )
                safe_forward = (
                    (1.0 - edge_progress) * max(0.0, desired[0])
                    + edge_progress * cfg.framing_retreat_mps
                )
                desired[0] = min(desired[0], safe_forward)
                framing_limited = desired[0] < requested_forward_mps - 1e-6

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
        if (
            usable
            and detection is not None
            and self._horizontal_capture_side * detection.normalized_x > 0.0
            and abs(detection.normalized_x)
            > cfg.horizontal_capture_release_normalized
        ):
            # This guard must run after command slew and low-pass filtering.
            # Applying the same limit only to the raw PID target still allowed
            # a previously queued ±0.5 m/s command to build lateral momentum
            # for several frames after measured gate motion showed capture.
            capture_moving_inward = (
                detection.normalized_x * detection.velocity_x < 0.0
            )
            if (
                capture_moving_inward
                and cfg.horizontal_capture_brake_lateral_gain > 0.0
            ):
                conditioned[1] = float(
                    np.clip(
                        cfg.horizontal_capture_brake_lateral_gain
                        * detection.velocity_x,
                        -cfg.countersteer_max_lateral_mps,
                        cfg.countersteer_max_lateral_mps,
                    )
                )
                self._last_command[1] = conditioned[1]
            elif detection.normalized_x * conditioned[1] > 0.0:
                conditioned[1] = float(
                    np.clip(
                        conditioned[1],
                        -cfg.inward_capture_max_lateral_mps,
                        cfg.inward_capture_max_lateral_mps,
                    )
                )
                self._last_command[1] = conditioned[1]
        if (
            usable
            and detection is not None
            and self._yaw_capture_side * detection.normalized_x > 0.0
            and abs(detection.normalized_x)
            > cfg.horizontal_capture_release_normalized
        ):
            yaw_capture_moving_inward = (
                detection.normalized_x * detection.velocity_x < 0.0
            )
            if (
                yaw_capture_moving_inward
                and cfg.horizontal_capture_brake_yaw_gain > 0.0
            ):
                conditioned[3] = float(
                    np.clip(
                        cfg.horizontal_capture_brake_yaw_gain
                        * detection.velocity_x,
                        -cfg.max_yaw_rate_rps,
                        cfg.max_yaw_rate_rps,
                    )
                )
                self._last_command[3] = conditioned[3]
            elif detection.normalized_x * conditioned[3] > 0.0:
                conditioned[3] = float(
                    np.clip(
                        conditioned[3],
                        -cfg.inward_capture_max_yaw_rate_rps,
                        cfg.inward_capture_max_yaw_rate_rps,
                    )
                )
                self._last_command[3] = conditioned[3]
        if (
            self.state == NavigationState.RECOVER
            and conditioned[0] < 0.0
        ):
            # Do not carry a close-approach reverse command into recovery.
            conditioned[0] = 0.0
            self._last_command[0] = 0.0
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
            requested_forward_mps=requested_forward_mps,
            framing_limited=framing_limited,
            framing_edge=framing_edge,
        )


def q2_demo_navigation_config() -> NavigationConfig:
    """Adapt the demonstrated gate-passing profile to the multi-gate course.

    The attitude loop retains collect_demos.py's physical conventions. Gate
    centering deliberately favors lateral bank over yaw so rotating the camera
    cannot make an off-axis gate look aligned before the vehicle has moved
    toward its opening.
    """
    return NavigationConfig(
        # A gate must already pass YOLO, orange HSV support, and temporal
        # acquisition before reaching navigation. Do not reject that trusted
        # target again merely because it is distant or lacks pose orientation.
        minimum_detection_confidence=0.18,
        reliable_confidence=0.30,
        commit_minimum_confidence=0.32,
        # The recorded first pass peaked at 6.49% frame opening area before
        # dropout. Commit earlier, while the centered opening is still measured,
        # instead of falling into recovery and acquiring background orange.
        commit_opening_area_ratio=0.030,
        commit_alignment_tolerance=0.10,
        # Two consecutive runs reached a stable, high-confidence gate at
        # 4.1-4.3 m with only 6.7-7.8% horizontal error, then lost the detector
        # without committing.  This is still well inside the opening and
        # matches the existing abort tolerance.
        commit_horizontal_tolerance=0.08,
        commit_abort_alignment_tolerance=0.16,
        commit_abort_horizontal_tolerance=0.08,
        # A valid YOLO+HSV gate can first appear near x=0.85 after a pass.
        # Keep measured edge targets in active yaw-first capture instead of
        # dropping immediately into blind RECOVER.
        recovery_alignment_error=0.98,
        commit_maximum_duration_s=1.20,
        recover_local_duration_s=0.75,
        track_confirmation_frames=1,
        commit_stable_frames=3,
        commit_max_center_speed=2.0,
        commit_max_size_rate=100.0,
        # Close-range centre estimates become noisy as the frame fills. Once
        # the strict commit gate is satisfied, fly level through the opening
        # and let race-status confirmation release the old target.
        commit_straight_through=True,
        center_deadband=0.035,
        # Hold the opening lower in the camera (drone higher in the opening)
        # and start correcting sooner to clear the first gate's bottom rail.
        vertical_setpoint_normalized=2.0 * 0.62 - 1.0,
        # The lower image target clears gate one's bottom rail. Reusing it
        # after the confirmed pass made gate two's y≈128 acquisition command
        # a large climb. Even an optical-center target still requested climb
        # for the first 0.8 s and gate two then fell out of the bottom of the
        # image before yaw capture completed. Hold subsequent gates on their
        # natural high approach line instead; any downward drift then requests
        # descent early, while no climb momentum is deliberately introduced.
        post_pass_vertical_setpoint_normalized=-0.25,
        vertical_deadband=0.20,
        # The post-pass setpoint is y≈135. Waiting for the gate to fall past
        # y≈149 before descending let vertical image speed build beyond the
        # open-loop thrust controller's stopping authority. Begin descent
        # within roughly four pixels of the target line.
        vertical_descent_deadband=0.02,
        # Distant gates still control altitude so they cannot drift out of the
        # top or bottom of the frame.
        vertical_control_min_area_ratio=0.0,
        # Gate-two y is not a reliable altitude cue during the initial hard
        # right turn. The 21:13 A/B commanded up to 0.60 m/s descent while
        # the gate still fell from y=131 to y=306 and struck the frame.
        # Hold hover until yaw/lateral alignment enters this corridor.
        vertical_control_max_horizontal_error=0.35,
        search_forward_mps=0.0,
        search_close_forward_mps=0.12,
        # Every Q2 gate is already visible from the preceding gate. A blind
        # yaw sweep during a detector dropout rotated the correctly aligned
        # second gate through the opposite edge and into the Station 22
        # pillar. Hold heading until a measured gate can steer deliberately.
        search_yaw_rate_rps=0.0,
        track_forward_mps=0.32,
        minimum_approach_mps=0.34,
        maximum_approach_mps=0.42,
        commit_forward_mps=0.46,
        # Preserve modest forward progress through a brief detector dropout.
        # This is below the close-approach speed and is bounded by the real
        # sighting-based recovery timeout above.
        recover_forward_mps=0.28,
        # Training One still backed from 2.1 m to 2.3 m at the old 0.14 m/s
        # close floor. Keep nearly all normal recovery momentum; the remaining
        # 0.04 m/s reduction leaves modest collision braking while lateral
        # centering finishes.
        recover_close_forward_mps=0.24,
        recover_lateral_mps=0.0,
        recover_prediction_scale=0.65,
        minimum_forward_mps=0.0,
        maximum_forward_mps=0.65,
        approach_slowdown_start_area_ratio=0.008,
        approach_slowdown_end_area_ratio=0.025,
        close_approach_mps=0.30,
        # When gate two is still in the outer half of the image, the previous
        # 0.30-0.33 m/s approach moved it from y=132 to the bottom in roughly
        # two seconds. Keep a positive crawl while yaw/bank align the flight
        # line, then automatically restore normal speed inside the corridor.
        severe_horizontal_error_normalized=0.50,
        severe_horizontal_forward_cap_mps=0.16,
        # Horizontal error primarily commands bank/translation. The earlier
        # 2.4 yaw gain centered gate two by rotating the camera while the
        # vehicle remained outside its lateral flight line.
        # After the axis-isolated sign fix, gate two still carried +0.10 to
        # +0.15 normalized outward image speed while measured yaw held near
        # 0.5 rad/s. Increase position authority inside the existing 0.65
        # rad/s ceiling; gyro feedback and projected capture provide braking.
        horizontal_yaw_kp=0.60,
        # The tracked gate-center velocity is its angular motion through the
        # camera. It includes both lateral translation and vehicle yaw, so the
        # derivative terms begin counter-steering before the center crossing.
        # The latest gate-two run still crossed center with substantial image
        # velocity, so brake farther ahead: about 1.6 s laterally and 1.2 s
        # in yaw at the proportional gains below.
        horizontal_yaw_kd=0.75,
        lateral_kp=1.15,
        # Training One crossed from x=253 back through center before the roll
        # reversal took effect, then continued to x=454. More image-velocity
        # damping flips the desired bank earlier without increasing the
        # bounded lateral speed or countersteer multiplier.
        lateral_kd=2.50,
        # Gate two first appears high-right behind the Station 22 pillar.
        # Immediate full bank rotates that upper image quadrant outward before
        # lateral translation develops. A yaw-only trace kept the gate visible
        # and moved it inward; ramp corrected lateral bank in as yaw brings the
        # target from the edge toward the flight corridor.
        yaw_first_lateral_full_normalized=0.35,
        yaw_first_lateral_zero_normalized=0.75,
        # Gate two initially appears high-right. In the 21:20 trace, the 80%
        # edge handoff held about 4 degrees of right bank while the gate made
        # a 1.4 s inward/outward excursion and rotated from y=136 to y=305.
        # Retain a small flight-line handoff, then ramp to full translation
        # only as yaw brings the opening into the capture corridor.
        yaw_first_lateral_minimum_scale=0.20,
        # Start active braking only when measured inward image velocity
        # projects into the central corridor. Latching at the first inward
        # sample previously stopped correction near x=500, far from center.
        inward_capture_max_lateral_mps=0.0,
        inward_capture_max_yaw_rate_rps=0.0,
        # A 2.5 s image-space stopping horizon matches the observed delay from
        # desired-bank reversal to lateral response. If the target builds a
        # real outward velocity again, release the latch and recapture it
        # instead of holding zero command all the way to the frame edge.
        horizontal_capture_release_normalized=0.25,
        horizontal_capture_prediction_horizon_s=8.0,
        # Yaw reaches its measured response much sooner than lateral
        # translation. Brake heading on a longer projection while allowing
        # bank to keep moving the airframe onto the gate's flight line.
        horizontal_yaw_capture_prediction_horizon_s=8.0,
        # Do not capture heading while the target is still in the outer
        # quarter of the image.  Prediction alone latched near x=521/640 and
        # stopped the right turn before the airframe faced gate two.
        horizontal_yaw_capture_max_error_normalized=0.75,
        horizontal_capture_outward_release_speed=0.06,
        # Once heading capture has stopped the turn, do not restart yaw on a
        # single outward image-velocity sample. The 09:00 trace captured at
        # x=509, reached x=496, then one +0.121 sample released yaw and caused
        # the next oscillation. Hold heading until the target reaches/crosses
        # the center corridor; lateral control remains free to recenter it.
        horizontal_yaw_capture_outward_release_speed=math.inf,
        # Simply leveling at capture left the accumulated lateral/yaw momentum
        # untouched: gate two moved x=539 -> 473, then rebounded to x=538.
        # Reverse in proportion to measured image velocity and write the brake
        # through the output conditioner so queued same-side commands cannot
        # prolong the oscillation.
        horizontal_capture_brake_lateral_gain=1.50,
        # The gyro feedback loop already supplies braking torque. A negative
        # yaw-rate setpoint drove measured yaw through zero to -0.19 rad/s and
        # rebounded gate two from x=527 to x=568. Capture therefore targets
        # zero yaw rate while lateral translation continues.
        horizontal_capture_brake_yaw_gain=0.00,
        # A full bank step after yaw capture moved the gate outward again.
        # Use a moderate translation handoff while the yaw loop holds heading.
        yaw_capture_lateral_scale=0.55,
        lateral_countersteer_gain=3.00,
        # Preserve the early derivative-based reversal, but keep it inside a
        # semi-strict flight corridor. The latest run held -0.60 m/s left for
        # about a second and crossed violently after the initial right turn.
        countersteer_max_lateral_mps=0.40,
        yaw_countersteer_gain=2.25,
        countersteer_forward_floor_mps=0.34,
        vertical_kp=0.8,
        # Gate-two logs show the image center moving upward while proportional
        # error still requested descent. Brake vertical momentum well before
        # the center crossing; thrust limits remain the final safety bound.
        vertical_kd=2.30,
        # The 06:46 trace still climbed at gate y=141 with vy=0.094, then
        # needed saturated descent after y=155. Stronger damping changes that
        # recorded sample into a small early descent. The asymmetric guard
        # below bounds this reversal and keeps pre-crossing climb disabled.
        vertical_countersteer_max_mps=0.22,
        # The first post-guard Training One run proved the complete command
        # chain was working: +0.539 m/s became -0.129 rad desired roll and
        # -0.085 rad measured roll before gate motion turned inward. That bank
        # carried the drone into the parked aircraft despite immediate capture
        # limiting. Bound normal and braking authority to about 5.5 degrees of
        # desired bank through the controller's 0.24 lean gain.
        max_right_mps=0.40,
        max_up_mps=0.35,
        max_down_mps=0.60,
        # Controlled lateral-sign A/B runs both saturated this ceiling while
        # gate two still moved outward; the correct demo sign was better but
        # could not overcome carried post-gate heading before the image edge.
        # Give acquisition more turn authority. Measured inward capture still
        # zeros the request and the IMU feedback loop brakes the actual rate.
        max_yaw_rate_rps=0.65,
        max_forward_acceleration=1.4,
        # Gate-two logs show the high-authority countersteer taking roughly
        # 0.7-0.8 s to reverse through the old slew limit.  Reverse promptly
        # enough to brake before the image center crosses, while retaining the
        # 0.60 m/s absolute lateral-command bound.
        max_lateral_acceleration=3.2,
        max_vertical_acceleration=1.4,
        max_yaw_acceleration=0.8,
        command_lpf_alpha=0.68,
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
        # The corrected capture trace moved gate two inward (x=528→508), then
        # reversed outward exactly as this envelope reduced forward speed from
        # 0.33 to 0.18 m/s. Keep coordinated translation through the turn and
        # taper only in the last part of the actual camera boundary.
        framing_soft_edge_normalized=0.84,
        framing_hard_edge_normalized=0.99,
        framing_retreat_mps=0.0,
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
