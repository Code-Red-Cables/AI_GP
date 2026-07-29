"""Complementary-filter AHRS: fuse gyro (short-term) + accelerometer gravity (long-term)
into a clean roll/pitch estimate that survives dynamic motion — unlike accel-only tilt,
which is only valid near 1 g.

Feeds the rate controller's attitude loop (controller.py). Yaw is integrated from gyro for
relative heading but drifts (no magnetometer) — heading is steered from vision / the
dual-gate PnP fix, not from absolute yaw.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_G = 9.81


@dataclass
class AHRSConfig:
    alpha: float = 0.95            # gyro trust (short-term); (1-alpha) = accel trust
    gyro_sign_roll: float = 1.0    # flip if the estimate diverges from accel (tune live)
    gyro_sign_pitch: float = 1.0
    gyro_sign_yaw: float = 1.0
    accel_gate: float = 0.35       # only apply gravity correction when |a| within this of 1g


class ComplementaryAHRS:
    def __init__(self, cfg: AHRSConfig | None = None):
        self.cfg = cfg or AHRSConfig()
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0            # relative, drifts
        self.last_accel_roll = 0.0
        self.last_accel_pitch = 0.0
        self.divergence = 0.0    # |estimate - accel| when corrected; live sign-tuning aid

    def reset(self) -> None:
        self.roll = self.pitch = self.yaw = 0.0

    @staticmethod
    def _accel_angles(ax: float, ay: float, az: float) -> tuple[float, float]:
        # accel_z reads ~-9.8 at rest (measured), so gravity-up = -az → level maps to ~0.
        g_up = -az
        roll = math.atan2(ay, g_up)
        pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az) + 1e-6)
        return roll, pitch

    def update(self, gyro: tuple[float, float, float],
               accel: tuple[float, float, float], dt: float) -> tuple[float, float, float]:
        c = self.cfg
        gx, gy, gz = gyro
        ax, ay, az = accel
        dt = max(1e-3, min(0.1, dt))

        # gyro integration (body rates ≈ euler rates for the modest lean angles we fly)
        roll_g = self.roll + c.gyro_sign_roll * gx * dt
        pitch_g = self.pitch + c.gyro_sign_pitch * gy * dt
        self.yaw += c.gyro_sign_yaw * gz * dt

        # gravity correction only when acceleration is near 1 g (else accel is not gravity)
        amag = math.sqrt(ax * ax + ay * ay + az * az)
        if abs(amag - _G) <= c.accel_gate * _G:
            roll_a, pitch_a = self._accel_angles(ax, ay, az)
            self.last_accel_roll, self.last_accel_pitch = roll_a, pitch_a
            self.roll = c.alpha * roll_g + (1 - c.alpha) * roll_a
            self.pitch = c.alpha * pitch_g + (1 - c.alpha) * pitch_a
            self.divergence = 0.9 * self.divergence + 0.1 * (
                abs(self.roll - roll_a) + abs(self.pitch - pitch_a))
        else:
            # high-g / rotating: trust gyro alone this step
            self.roll, self.pitch = roll_g, pitch_g
        return self.roll, self.pitch, self.yaw
