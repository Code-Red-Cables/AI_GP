"""World acceleration from commanded thrust and known attitude.

The competition accelerometer is not a velocity sensor. Horizontal correlation
with true velocity is ~0, and integrating IMU-z just walks the bias. A
quadrotor can only accelerate by tilting and throttling, so world acceleration
is computable at the control rate without touching the accelerometer:

    a_ned = G − (T/m) · (R @ [0, 0, 1]) + drag(v)
            [NED, z down; FRD body]

T/m = (thr / hover_trim) · g. Hover trim is the 1 g calibration: the
collective that holds altitude while level. That scale is observable in
flight, so the thrust model is self-calibrating.

This is a Kalman *process input*, not a measurement. During a vision gap the
filter propagates with the actual commanded physics instead of coasting on
stale velocity or integrating a dead IMU.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

G = 9.80665
G_NED = np.array([0.0, 0.0, G], dtype=np.float64)
BODY_DOWN = np.array([0.0, 0.0, 1.0], dtype=np.float64)

# Quiet-hover window for the trim observer. Loose enough to catch a real
# hover, tight enough that a slam or a climb punch cannot rewrite the scale.
_TRIM_MAX_TILT_RAD = math.radians(8.0)
_TRIM_MAX_RATE_RAD = math.radians(20.0)
_TRIM_MAX_VZ = 0.40
_TRIM_LO = 0.15
_TRIM_HI = 0.45


def specific_thrust(thrust: float, hover_trim: float) -> float:
    """T/m in m/s^2 from collective and the 1 g hover calibration."""
    thr = float(thrust)
    trim = float(hover_trim)
    if not math.isfinite(thr) or not math.isfinite(trim) or trim <= 1e-6:
        return G
    return (thr / trim) * G


def commanded_accel_ned(
    R_ned_body: np.ndarray,
    thrust: float,
    hover_trim: float,
    velocity_ned: Optional[np.ndarray] = None,
    k_body: Optional[np.ndarray] = None,
) -> np.ndarray:
    """World acceleration from thrust, attitude, and optional linear drag.

    ``R_ned_body`` is body → NED. ``k_body`` is a 3-vector with a = k ⊙ v
    in FRD (k < 0 opposes motion). Omit ``k_body`` / ``velocity_ned`` for
    the drag-free model.
    """
    R = np.asarray(R_ned_body, dtype=np.float64).reshape(3, 3)
    tm = specific_thrust(thrust, hover_trim)
    a = G_NED - tm * (R @ BODY_DOWN)
    if velocity_ned is not None and k_body is not None:
        k = np.asarray(k_body, dtype=np.float64).reshape(3)
        v_ned = np.asarray(velocity_ned, dtype=np.float64).reshape(3)
        if np.all(np.isfinite(k)) and np.all(np.isfinite(v_ned)):
            v_body = R.T @ v_ned
            a = a + R @ (k * v_body)
    return a


def hover_is_observable(
    roll: float,
    pitch: float,
    rates_rad_s: np.ndarray,
    vel_d: float,
) -> bool:
    """True when current collective is a usable 1 g sample."""
    if abs(float(roll)) > _TRIM_MAX_TILT_RAD:
        return False
    if abs(float(pitch)) > _TRIM_MAX_TILT_RAD:
        return False
    if abs(float(vel_d)) > _TRIM_MAX_VZ:
        return False
    omega = np.asarray(rates_rad_s, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(omega)):
        return False
    return float(np.linalg.norm(omega)) <= _TRIM_MAX_RATE_RAD


def observe_hover_trim(
    trim: float,
    thrust: float,
    *,
    roll: float,
    pitch: float,
    rates_rad_s: np.ndarray,
    vel_d: float,
    dt: float,
    tau_s: float,
) -> float:
    """EMA hover trim toward the collective that is currently holding 1 g.

    Returns ``trim`` unchanged when the craft is not quiet, ``tau_s`` is
    non-positive, or the sample is not finite.
    """
    current = float(trim)
    if tau_s <= 0.0 or dt <= 0.0:
        return current
    sample = float(thrust)
    if not math.isfinite(sample) or not (_TRIM_LO <= sample <= _TRIM_HI):
        return current
    if not hover_is_observable(roll, pitch, rates_rad_s, vel_d):
        return current
    alpha = 1.0 - math.exp(-float(dt) / float(tau_s))
    updated = current + alpha * (sample - current)
    return float(min(_TRIM_HI, max(_TRIM_LO, updated)))


def gravity_in_body(roll: float, pitch: float) -> np.ndarray:
    """Gravity (NED +z down) expressed in FRD. Yaw does not enter."""
    cr, sr = math.cos(float(roll)), math.sin(float(roll))
    cp, sp = math.cos(float(pitch)), math.sin(float(pitch))
    return np.array([-G * sp, G * sr * cp, G * cr * cp], dtype=np.float64)


class BodyVelocityIntegrator:
    """Body-frame commanded velocity. Train and flight must share this.

    Integrated in FRD so absolute yaw is never required:

        a = g_body − (T/m) e_z + k ⊙ v − ω × v
    """

    def __init__(
        self,
        *,
        hover_trim: float = 0.255,
        k_body: Optional[np.ndarray] = None,
        max_dt: float = 0.05,
    ):
        self.hover_trim = float(hover_trim)
        self.k_body = (
            np.array([-0.50, -0.50, -0.15], dtype=np.float64)
            if k_body is None
            else np.asarray(k_body, dtype=np.float64).reshape(3)
        )
        self.max_dt = float(max_dt)
        self.v = np.zeros(3, dtype=np.float64)

    def reset(self) -> None:
        self.v[:] = 0.0

    def step(
        self,
        dt: float,
        thrust: float,
        roll: float,
        pitch: float,
        omega: Optional[np.ndarray] = None,
        hover_trim: Optional[float] = None,
    ) -> np.ndarray:
        if hover_trim is not None and math.isfinite(float(hover_trim)):
            self.hover_trim = float(hover_trim)
        step = float(dt)
        if not math.isfinite(step) or step <= 0.0:
            return self.v.copy()
        step = min(step, self.max_dt)
        thr = float(thrust) if math.isfinite(float(thrust)) else self.hover_trim
        roll = 0.0 if not math.isfinite(float(roll)) else float(roll)
        pitch = 0.0 if not math.isfinite(float(pitch)) else float(pitch)
        g_body = gravity_in_body(roll, pitch)
        tm = specific_thrust(thr, self.hover_trim)
        a = g_body - tm * BODY_DOWN + self.k_body * self.v
        if omega is not None:
            w = np.asarray(omega, dtype=np.float64).reshape(3)
            if np.all(np.isfinite(w)):
                a = a - np.cross(w, self.v)
        self.v = self.v + a * step
        return self.v.copy()
