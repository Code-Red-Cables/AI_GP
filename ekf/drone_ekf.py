"""NumPy Extended Kalman Filter for dual-gate racing.

State (16):
  p (3)  — NED position (m), origin = arm / filter start
  v (3)  — NED velocity (m/s)
  q (4)  — body→NED quaternion (w, x, y, z)
  ba (3) — accelerometer bias (m/s^2)
  bg (3) — gyro bias (rad/s)

Prediction: IMU (gyro + accel) at high rate (~100–500 Hz samples).
Correction: dual-gate PnP body-frame centres rotated into NED at ~30 Hz.

When Gate 2 leaves the FOV the filter keeps predicting from IMU alone —
dead-reckoning memory — so the yaw look-at and path toward the last Gate-2
fix remain continuous until a fresh PnP arrives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

G = 9.80665
G_NED = np.array([0.0, 0.0, G], dtype=np.float64)


def _wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_from_gyro(omega: np.ndarray, dt: float) -> np.ndarray:
    angle = float(np.linalg.norm(omega) * dt)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = omega / (np.linalg.norm(omega) + 1e-12)
    half = 0.5 * angle
    s = math.sin(half)
    return np.array(
        [math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s],
        dtype=np.float64,
    )


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Body → NED rotation matrix."""
    q = quat_normalize(q)
    w, x, y, z = q
    return np.array(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    tr = float(np.trace(R))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return quat_normalize(np.array([w, x, y, z], dtype=np.float64))


def quat_to_rpy(q: np.ndarray) -> tuple[float, float, float]:
    R = quat_to_rot(q)
    pitch = -math.asin(float(np.clip(R[2, 0], -1.0, 1.0)))
    roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
    yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    return roll, pitch, yaw


@dataclass
class EKFState:
    position_ned: np.ndarray
    velocity_ned: np.ndarray
    quaternion: np.ndarray
    accel_bias: np.ndarray
    gyro_bias: np.ndarray
    timestamp: float
    gate1_ned: Optional[np.ndarray] = None
    gate2_ned: Optional[np.ndarray] = None
    gate2_fresh: bool = False

    @property
    def roll_pitch_yaw(self) -> tuple[float, float, float]:
        return quat_to_rpy(self.quaternion)


class DroneEKF:
    """16-state IMU / dual-gate PnP EKF."""

    IDX_P = 0
    IDX_V = 3
    IDX_Q = 6
    IDX_BA = 10
    IDX_BG = 13
    N = 16

    def __init__(
        self,
        *,
        accel_noise: float = 0.35,
        gyro_noise: float = 0.02,
        bias_noise: float = 1e-4,
        pnp_pos_noise: float = 0.45,
    ):
        self.x = np.zeros(self.N, dtype=np.float64)
        self.x[self.IDX_Q] = 1.0  # identity quaternion
        self.P = np.eye(self.N, dtype=np.float64)
        self.P[self.IDX_P : self.IDX_P + 3] *= 1.0
        self.P[self.IDX_V : self.IDX_V + 3] *= 1.0
        self.P[self.IDX_Q : self.IDX_Q + 4] *= 0.05
        self.P[self.IDX_BA : self.IDX_BA + 3] *= 0.1
        self.P[self.IDX_BG : self.IDX_BG + 3] *= 0.01
        self.accel_noise = accel_noise
        self.gyro_noise = gyro_noise
        self.bias_noise = bias_noise
        self.pnp_pos_noise = pnp_pos_noise
        self._last_t: Optional[float] = None
        self._gate1_ned: Optional[np.ndarray] = None
        self._gate2_ned: Optional[np.ndarray] = None
        self._gate2_fresh = False
        self._last_gate2_update = 0.0

    def reset(self, timestamp: float = 0.0) -> None:
        self.__init__(
            accel_noise=self.accel_noise,
            gyro_noise=self.gyro_noise,
            bias_noise=self.bias_noise,
            pnp_pos_noise=self.pnp_pos_noise,
        )
        self._last_t = timestamp

    def state(self) -> EKFState:
        return EKFState(
            position_ned=self.x[self.IDX_P : self.IDX_P + 3].copy(),
            velocity_ned=self.x[self.IDX_V : self.IDX_V + 3].copy(),
            quaternion=quat_normalize(self.x[self.IDX_Q : self.IDX_Q + 4]),
            accel_bias=self.x[self.IDX_BA : self.IDX_BA + 3].copy(),
            gyro_bias=self.x[self.IDX_BG : self.IDX_BG + 3].copy(),
            timestamp=float(self._last_t or 0.0),
            gate1_ned=None
            if self._gate1_ned is None
            else self._gate1_ned.copy(),
            gate2_ned=None
            if self._gate2_ned is None
            else self._gate2_ned.copy(),
            gate2_fresh=bool(self._gate2_fresh),
        )

    def predict(
        self,
        gyro_rad_s: np.ndarray,
        accel_m_s2: np.ndarray,
        timestamp: float,
    ) -> EKFState:
        """Propagate with IMU. Safe to call at ~100–500 Hz sample rate."""
        gyro = np.asarray(gyro_rad_s, dtype=np.float64).reshape(3)
        accel = np.asarray(accel_m_s2, dtype=np.float64).reshape(3)
        if self._last_t is None:
            self._last_t = timestamp
            return self.state()
        dt = float(timestamp - self._last_t)
        if dt <= 0.0 or dt > 0.05:
            self._last_t = timestamp
            return self.state()
        self._last_t = timestamp

        ba = self.x[self.IDX_BA : self.IDX_BA + 3]
        bg = self.x[self.IDX_BG : self.IDX_BG + 3]
        omega = gyro - bg
        acc_body = accel - ba
        q = quat_normalize(self.x[self.IDX_Q : self.IDX_Q + 4])
        R = quat_to_rot(q)
        acc_ned = R @ acc_body + G_NED

        self.x[self.IDX_P : self.IDX_P + 3] += (
            self.x[self.IDX_V : self.IDX_V + 3] * dt
            + 0.5 * acc_ned * dt * dt
        )
        self.x[self.IDX_V : self.IDX_V + 3] += acc_ned * dt
        dq = quat_from_gyro(omega, dt)
        self.x[self.IDX_Q : self.IDX_Q + 4] = quat_normalize(
            quat_multiply(q, dq)
        )

        # Covariance: diagonal process noise (practical racing EKF).
        q_pos = (0.5 * self.accel_noise * dt * dt) ** 2
        q_vel = (self.accel_noise * dt) ** 2
        q_att = (self.gyro_noise * dt) ** 2
        q_bias = (self.bias_noise * dt) ** 2
        for i in range(3):
            self.P[self.IDX_P + i, self.IDX_P + i] += q_pos
            self.P[self.IDX_V + i, self.IDX_V + i] += q_vel
            self.P[self.IDX_BA + i, self.IDX_BA + i] += q_bias
            self.P[self.IDX_BG + i, self.IDX_BG + i] += q_bias
        for i in range(4):
            self.P[self.IDX_Q + i, self.IDX_Q + i] += q_att

        # Mark Gate-2 belief as stale once it ages (still used for look-at).
        # Never promote stale→fresh here; only PnP corrections set fresh=True.
        if (
            self._gate2_ned is not None
            and (timestamp - self._last_gate2_update) >= 0.25
        ):
            self._gate2_fresh = False
        return self.state()

    def correct_dual_gate_body(
        self,
        gate1_body: np.ndarray,
        gate2_body: Optional[np.ndarray],
        timestamp: float,
    ) -> EKFState:
        """Update with PnP gate centres expressed in the body frame.

        With the filter origin at the current drone pose, a body-frame gate
        vector ``g_b`` maps to world as ``p + R @ g_b``. We treat the relative
        vector as a position measurement of the gate in NED and keep a
        filtered gate location; the drone state is corrected so
        ``R @ g_b ≈ gate_ned - p``.
        """
        g1 = np.asarray(gate1_body, dtype=np.float64).reshape(3)
        q = quat_normalize(self.x[self.IDX_Q : self.IDX_Q + 4])
        R = quat_to_rot(q)
        p = self.x[self.IDX_P : self.IDX_P + 3]
        gate1_meas = p + R @ g1
        if self._gate1_ned is None:
            self._gate1_ned = gate1_meas.copy()
        else:
            self._gate1_ned = 0.65 * self._gate1_ned + 0.35 * gate1_meas

        # Innovation: predicted relative = R.T @ (gate - p) vs measured body.
        pred_body = R.T @ (self._gate1_ned - p)
        innov = g1 - pred_body
        self._scalar_body_update(innov, R)

        if gate2_body is not None:
            g2 = np.asarray(gate2_body, dtype=np.float64).reshape(3)
            q = quat_normalize(self.x[self.IDX_Q : self.IDX_Q + 4])
            R = quat_to_rot(q)
            p = self.x[self.IDX_P : self.IDX_P + 3]
            gate2_meas = p + R @ g2
            if self._gate2_ned is None:
                self._gate2_ned = gate2_meas.copy()
            else:
                self._gate2_ned = 0.70 * self._gate2_ned + 0.30 * gate2_meas
            self._last_gate2_update = timestamp
            self._gate2_fresh = True
            pred_body2 = R.T @ (self._gate2_ned - p)
            innov2 = g2 - pred_body2
            self._scalar_body_update(innov2, R, noise_scale=1.25)
        else:
            # Dead-reckoning buffer: keep last Gate-2 NED; mark not fresh.
            self._gate2_fresh = False

        self._last_t = timestamp
        return self.state()

    def _scalar_body_update(
        self,
        innov: np.ndarray,
        R: np.ndarray,
        *,
        noise_scale: float = 1.0,
    ) -> None:
        """Correct position (and lightly velocity) from a body-frame innov."""
        # Map body innovation to NED position correction: dp = -R @ innov
        # (if measured body vector is longer/shorter/wrong direction).
        dp = -R @ innov
        if float(np.linalg.norm(dp)) > 6.0:
            return  # reject outlier PnP
        r = (self.pnp_pos_noise * noise_scale) ** 2
        for i in range(3):
            p_ii = self.P[self.IDX_P + i, self.IDX_P + i]
            k = p_ii / (p_ii + r)
            self.x[self.IDX_P + i] += k * dp[i]
            self.P[self.IDX_P + i, self.IDX_P + i] *= 1.0 - k
            # Soft velocity pull toward zero relative motion vs gate.
            v_ii = self.P[self.IDX_V + i, self.IDX_V + i]
            kv = 0.15 * v_ii / (v_ii + r)
            self.x[self.IDX_V + i] *= 1.0 - kv
