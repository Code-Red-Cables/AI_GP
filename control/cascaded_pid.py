"""Cascaded PID for dual-gate EKF flight.

Outer loop (position): NED position / altitude errors → desired roll, pitch,
collective thrust.

Yaw channel: dedicated PID on heading error vs look-at vector to Gate 2.

Inner loop (attitude): desired roll/pitch → body rates.

VQ2 does not expose raw motor throttles — the inner loop emits body rates +
thrust for ``SET_ATTITUDE_TARGET`` (the sim's supported actuator interface).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from control.pid import PIDConfig, PIDController


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


@dataclass
class CascadedCommand:
    roll_rate: float
    pitch_rate: float
    yaw_rate: float
    thrust: float
    desired_roll: float
    desired_pitch: float
    desired_yaw: float
    position_error_ned: np.ndarray


@dataclass
class CascadedPIDConfig:
    # Outer position → lean / climb
    kp_pos_xy: float = 0.35
    kd_pos_xy: float = 0.20
    kp_pos_z: float = 0.55
    kd_pos_z: float = 0.25
    max_lean_rad: float = math.radians(18.0)
    hover_thrust: float = 0.28
    max_climb_thrust_delta: float = 0.08
    # Yaw look-at Gate 2
    kp_yaw: float = 1.8
    kd_yaw: float = 0.15
    max_yaw_rate: float = math.radians(70.0)
    # Inner attitude → rate
    kp_att: float = 2.4
    kd_att: float = 0.12
    max_rate: float = 1.05


class CascadedPIDController:
    def __init__(self, config: CascadedPIDConfig | None = None):
        self.cfg = config or CascadedPIDConfig()
        self._yaw_pid = PIDController(
            PIDConfig(
                kp=self.cfg.kp_yaw,
                kd=self.cfg.kd_yaw,
                output_min=-self.cfg.max_yaw_rate,
                output_max=self.cfg.max_yaw_rate,
            )
        )
        self._roll_pid = PIDController(
            PIDConfig(
                kp=self.cfg.kp_att,
                kd=self.cfg.kd_att,
                output_min=-self.cfg.max_rate,
                output_max=self.cfg.max_rate,
            )
        )
        self._pitch_pid = PIDController(
            PIDConfig(
                kp=self.cfg.kp_att,
                kd=self.cfg.kd_att,
                output_min=-self.cfg.max_rate,
                output_max=self.cfg.max_rate,
            )
        )
        self._prev_pos = None
        self._prev_t = None

    def reset(self) -> None:
        self._yaw_pid.reset()
        self._roll_pid.reset()
        self._pitch_pid.reset()
        self._prev_pos = None
        self._prev_t = None

    def update(
        self,
        *,
        position_ned: np.ndarray,
        velocity_ned: np.ndarray,
        roll: float,
        pitch: float,
        yaw: float,
        target_ned: np.ndarray,
        look_yaw_rad: float,
        dt: float,
    ) -> CascadedCommand:
        p = np.asarray(position_ned, dtype=np.float64).reshape(3)
        v = np.asarray(velocity_ned, dtype=np.float64).reshape(3)
        target = np.asarray(target_ned, dtype=np.float64).reshape(3)
        err = target - p

        # Outer position PD in NED → desired body lean.
        # Forward (body x) ≈ N*cos(yaw)+E*sin(yaw); right ≈ -N*sin+E*cos.
        cy, sy = math.cos(yaw), math.sin(yaw)
        err_fwd = err[0] * cy + err[1] * sy
        err_right = -err[0] * sy + err[1] * cy
        vel_fwd = v[0] * cy + v[1] * sy
        vel_right = -v[0] * sy + v[1] * cy

        des_pitch = float(
            np.clip(
                -(
                    self.cfg.kp_pos_xy * err_fwd
                    - self.cfg.kd_pos_xy * vel_fwd
                ),
                -self.cfg.max_lean_rad,
                self.cfg.max_lean_rad,
            )
        )
        # Positive right error → negative roll in FRD/NED lean convention used
        # by the Q2 rate path (LATERAL_LEAN_SIGN = -1).
        des_roll = float(
            np.clip(
                -(
                    self.cfg.kp_pos_xy * err_right
                    - self.cfg.kd_pos_xy * vel_right
                ),
                -self.cfg.max_lean_rad,
                self.cfg.max_lean_rad,
            )
        )

        # Altitude: NED down positive; negative err_z means need to climb.
        climb = -(
            self.cfg.kp_pos_z * err[2] - self.cfg.kd_pos_z * v[2]
        )
        thrust = float(
            np.clip(
                self.cfg.hover_thrust
                + np.clip(
                    climb * 0.08,
                    -self.cfg.max_climb_thrust_delta,
                    self.cfg.max_climb_thrust_delta,
                ),
                0.05,
                0.90,
            )
        )

        yaw_err = _wrap(look_yaw_rad - yaw)
        yaw_rate = self._yaw_pid.update(yaw_err, dt)

        roll_rate = self._roll_pid.update(des_roll - roll, dt)
        pitch_rate = self._pitch_pid.update(des_pitch - pitch, dt)

        return CascadedCommand(
            roll_rate=float(roll_rate),
            pitch_rate=float(pitch_rate),
            yaw_rate=float(yaw_rate),
            thrust=thrust,
            desired_roll=des_roll,
            desired_pitch=des_pitch,
            desired_yaw=look_yaw_rad,
            position_error_ned=err,
        )
