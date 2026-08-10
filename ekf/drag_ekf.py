"""Drag-model EKF for vision-based racing without ground-truth position.

Li & de Croon (RAS 2020) replace the classic double-integrator prediction
with an aerodynamic drag model. Horizontal body velocity is *read* from the
accelerometer:

    v_body_xy = K^{-1} @ (a_meas_xy - bias_xy)

so accelerometer bias integrates once into position, not twice. That is the
direct fix for the 10^7 m divergence seen with ``ekf.drone_ekf`` under VQ2
(no LOCAL_POSITION_NED / ODOMETRY).

State (7):
  x, y, z     — position in a local NED frame (m), origin = filter start
  vz_B        — vertical velocity in the body frame (m/s)
  bax, bay, baz — accelerometer bias (m/s^2)

Vision updates come from ``GateLSPose`` expressed in the same local frame
once a course / gate map supplies the gate origin. Until then the filter can
still run open prediction for feed-forward arc control using drag velocity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

import camera_model as cm

G = 9.80665


@dataclass
class DragEKFState:
    x: np.ndarray          # shape (7,)
    P: np.ndarray          # shape (7, 7)
    initialized: bool = False


class DragEKF:
    """7-state drag-model EKF."""

    IDX_X, IDX_Y, IDX_Z, IDX_VZ, IDX_BAX, IDX_BAY, IDX_BAZ = range(7)

    def __init__(
        self,
        k_x: float = -0.5,
        k_y: float = -0.5,
        accel_noise: float = 0.5,
        bias_noise: float = 1e-4,
        vision_noise: float = 0.35,
    ):
        # a = k * v  with k < 0 (force opposes motion). Stored as given.
        if abs(k_x) < 1e-6 or abs(k_y) < 1e-6:
            raise ValueError('drag coefficients must be non-zero')
        self.k_x = float(k_x)
        self.k_y = float(k_y)
        self.accel_noise = float(accel_noise)
        self.bias_noise = float(bias_noise)
        self.vision_noise = float(vision_noise)
        self.state = DragEKFState(
            x=np.zeros(7, dtype=np.float64),
            P=np.eye(7, dtype=np.float64) * 1.0,
        )

    def reset(self, position_ned: Optional[np.ndarray] = None) -> None:
        self.state.x[:] = 0.0
        if position_ned is not None:
            self.state.x[0:3] = np.asarray(position_ned, dtype=np.float64).reshape(3)
        self.state.P = np.eye(7, dtype=np.float64) * 1.0
        self.state.initialized = True

    def body_velocity_xy(self, accel_body: np.ndarray) -> np.ndarray:
        """Horizontal body velocity from specific force and current bias."""
        ax, ay = float(accel_body[0]), float(accel_body[1])
        bax = float(self.state.x[self.IDX_BAX])
        bay = float(self.state.x[self.IDX_BAY])
        return np.array([
            (ax - bax) / self.k_x,
            (ay - bay) / self.k_y,
        ], dtype=np.float64)

    def predict(
        self,
        accel_body: np.ndarray,
        gyro_body: np.ndarray,
        roll: float,
        pitch: float,
        yaw: float,
        dt: float,
    ) -> np.ndarray:
        """Propagate state by ``dt`` seconds using IMU + AHRS."""
        if dt <= 0.0:
            return self.position_ned()
        if not self.state.initialized:
            self.reset()

        accel = np.asarray(accel_body, dtype=np.float64).reshape(3)
        gyro = np.asarray(gyro_body, dtype=np.float64).reshape(3)
        R_wb = cm.rot_world_body(roll, pitch, yaw)

        v_xy = self.body_velocity_xy(accel)
        vz = float(self.state.x[self.IDX_VZ])
        v_body = np.array([v_xy[0], v_xy[1], vz], dtype=np.float64)
        v_ned = R_wb @ v_body

        # Position integrates velocity; bias is random walk.
        self.state.x[self.IDX_X] += float(v_ned[0]) * dt
        self.state.x[self.IDX_Y] += float(v_ned[1]) * dt
        self.state.x[self.IDX_Z] += float(v_ned[2]) * dt

        # Vertical body accel dynamics (paper eq. 17, simplified).
        bax, bay, baz = self.state.x[self.IDX_BAX:self.IDX_BAZ + 1]
        p, q = float(gyro[0]), float(gyro[1])
        az = float(accel[2]) - float(baz)
        dvz = (
            az
            + G * math.cos(pitch) * math.cos(roll)
            + q * ((float(accel[0]) - float(bax)) / self.k_x)
            - p * ((float(accel[1]) - float(bay)) / self.k_y)
        )
        self.state.x[self.IDX_VZ] += float(dvz) * dt

        # Covariance: crude but stable process noise.
        q = np.zeros((7, 7), dtype=np.float64)
        q[0:3, 0:3] = np.eye(3) * (self.accel_noise * dt) ** 2
        q[self.IDX_VZ, self.IDX_VZ] = (self.accel_noise * dt) ** 2
        q[4:7, 4:7] = np.eye(3) * (self.bias_noise * dt)
        # Discrete Lyapunov with identity transition (bias RW, pos from v).
        self.state.P = self.state.P + q
        return self.position_ned()

    def update_position(self, position_ned: np.ndarray, noise: Optional[float] = None) -> None:
        """Correct with a vision-derived NED position fix."""
        if not self.state.initialized:
            self.reset(position_ned)
            return
        z = np.asarray(position_ned, dtype=np.float64).reshape(3)
        r = float(self.vision_noise if noise is None else noise)
        H = np.zeros((3, 7), dtype=np.float64)
        H[0, self.IDX_X] = 1.0
        H[1, self.IDX_Y] = 1.0
        H[2, self.IDX_Z] = 1.0
        R = np.eye(3, dtype=np.float64) * (r * r)
        y = z - self.state.x[0:3]
        S = H @ self.state.P @ H.T + R
        K = self.state.P @ H.T @ np.linalg.inv(S)
        self.state.x = self.state.x + K @ y
        self.state.P = (np.eye(7) - K @ H) @ self.state.P

    def position_ned(self) -> np.ndarray:
        return self.state.x[0:3].copy()

    def velocity_ned(self, accel_body: np.ndarray, roll: float, pitch: float, yaw: float) -> np.ndarray:
        R_wb = cm.rot_world_body(roll, pitch, yaw)
        v_xy = self.body_velocity_xy(accel_body)
        v_body = np.array([v_xy[0], v_xy[1], float(self.state.x[self.IDX_VZ])])
        return R_wb @ v_body

    def velocity_body_forward(self, accel_body: np.ndarray) -> float:
        return float(self.body_velocity_xy(accel_body)[0])
