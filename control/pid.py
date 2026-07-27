"""Timestamp-aware PID with filtering, clamping, and conditional anti-windup."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PIDConfig:
    kp: float
    ki: float = 0.0
    kd: float = 0.0
    output_min: float = -math.inf
    output_max: float = math.inf
    integral_min: float = -math.inf
    integral_max: float = math.inf
    derivative_filter_tau_s: float = 0.0
    minimum_dt_s: float = 1e-3
    maximum_dt_s: float = 0.1


class PIDController:
    """PID controller suitable for attitude and image-space control loops.

    ``measurement_rate`` enables derivative-on-measurement, which avoids a
    derivative kick when a new setpoint arrives. If it is omitted, the
    derivative is computed from consecutive errors.
    """

    def __init__(self, config: PIDConfig):
        if config.output_min > config.output_max:
            raise ValueError("PID output_min must not exceed output_max")
        if config.integral_min > config.integral_max:
            raise ValueError("PID integral_min must not exceed integral_max")
        self.config = config
        self.integral = 0.0
        self._previous_error: float | None = None
        self._filtered_derivative = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self._previous_error = None
        self._filtered_derivative = 0.0

    @staticmethod
    def _clip(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def update(
        self,
        error: float,
        dt: float,
        *,
        measurement_rate: float | None = None,
    ) -> float:
        cfg = self.config
        values = (error, dt)
        if measurement_rate is not None:
            values += (measurement_rate,)
        if not all(math.isfinite(float(value)) for value in values):
            self.reset()
            return 0.0

        dt = self._clip(float(dt), cfg.minimum_dt_s, cfg.maximum_dt_s)
        if measurement_rate is None:
            raw_derivative = (
                0.0
                if self._previous_error is None
                else (float(error) - self._previous_error) / dt
            )
        else:
            # d(error)/dt = -d(measurement)/dt for a locally fixed setpoint.
            raw_derivative = -float(measurement_rate)

        tau = max(0.0, cfg.derivative_filter_tau_s)
        alpha = 1.0 if tau == 0.0 else dt / (tau + dt)
        self._filtered_derivative += alpha * (
            raw_derivative - self._filtered_derivative
        )

        candidate_integral = self._clip(
            self.integral + float(error) * dt,
            cfg.integral_min,
            cfg.integral_max,
        )
        candidate = (
            cfg.kp * float(error)
            + cfg.ki * candidate_integral
            + cfg.kd * self._filtered_derivative
        )
        output = self._clip(candidate, cfg.output_min, cfg.output_max)

        # Conditional integration: freeze the integrator when saturation and
        # the current error point in the same direction. Integration resumes
        # immediately when the error would drive the actuator out of saturation.
        saturated_high = candidate > cfg.output_max and error > 0.0
        saturated_low = candidate < cfg.output_min and error < 0.0
        if not (saturated_high or saturated_low):
            self.integral = candidate_integral
        else:
            candidate = (
                cfg.kp * float(error)
                + cfg.ki * self.integral
                + cfg.kd * self._filtered_derivative
            )
            output = self._clip(candidate, cfg.output_min, cfg.output_max)

        self._previous_error = float(error)
        return float(output)
