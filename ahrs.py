"""Complementary-filter AHRS: gyro short-term + accelerometer gravity long-term."""
from __future__ import annotations

import math
from dataclasses import dataclass

_G = 9.81


@dataclass
class AHRSConfig:
    alpha: float = 0.95
    gyro_sign_roll: float = 1.0
    gyro_sign_pitch: float = 1.0
    gyro_sign_yaw: float = 1.0
    accel_gate: float = 0.35


class ComplementaryAHRS:
    def __init__(self, cfg: AHRSConfig | None = None):
        self.cfg = cfg or AHRSConfig()
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.last_accel_roll = 0.0
        self.last_accel_pitch = 0.0
        self.divergence = 0.0

    def reset(self) -> None:
        self.roll = self.pitch = self.yaw = 0.0

    @staticmethod
    def _accel_angles(ax: float, ay: float, az: float) -> tuple[float, float]:
        g_up = -az
        roll = math.atan2(ay, g_up)
        pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az) + 1e-6)
        return roll, pitch

    def update(
        self,
        gyro: tuple[float, float, float],
        accel: tuple[float, float, float],
        dt: float,
    ) -> tuple[float, float, float]:
        c = self.cfg
        gx, gy, gz = gyro
        ax, ay, az = accel
        dt = max(1e-3, min(0.1, dt))

        roll_g = self.roll + c.gyro_sign_roll * gx * dt
        pitch_g = self.pitch + c.gyro_sign_pitch * gy * dt
        self.yaw += c.gyro_sign_yaw * gz * dt

        amag = math.sqrt(ax * ax + ay * ay + az * az)
        if abs(amag - _G) <= c.accel_gate * _G:
            roll_a, pitch_a = self._accel_angles(ax, ay, az)
            self.last_accel_roll, self.last_accel_pitch = roll_a, pitch_a
            self.roll = c.alpha * roll_g + (1 - c.alpha) * roll_a
            self.pitch = c.alpha * pitch_g + (1 - c.alpha) * pitch_a
            self.divergence = 0.9 * self.divergence + 0.1 * (
                abs(self.roll - roll_a) + abs(self.pitch - pitch_a)
            )
        else:
            self.roll, self.pitch = roll_g, pitch_g
        return self.roll, self.pitch, self.yaw
