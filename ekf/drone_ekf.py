"""NumPy Extended Kalman Filter for dual-gate racing.

State (16):
  p (3)  — NED position (m), origin = arm / filter start
  v (3)  — NED velocity (m/s)
  q (4)  — body→NED quaternion (w, x, y, z)
  ba (3) — accelerometer bias (m/s^2)
  bg (3) — gyro bias (rad/s)

Prediction: gyro for attitude; commanded thrust + attitude + drag for
velocity (the quadrotor is its own accelerometer — HIGHRES_IMU accel is
not integrated). Correction: dual-gate PnP at ~30 Hz.

When Gate 2 leaves the FOV the filter keeps predicting from IMU alone —
dead-reckoning memory — so the yaw look-at and path toward the last Gate-2
fix remain continuous until a fresh PnP arrives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ekf.commanded_accel import commanded_accel_ned

G = 9.80665
G_NED = np.array([0.0, 0.0, G], dtype=np.float64)


def _wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _clamp(v: float, limit: float) -> float:
    return max(-limit, min(limit, float(v)))


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


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return quat_normalize(
        np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=np.float64,
        )
    )


def accel_to_roll_pitch(accel_m_s2: np.ndarray) -> tuple[float, float]:
    """Tilt from specific force. Resting zacc ≈ -g (same as ahrs.py)."""
    ax, ay, az = (float(v) for v in np.asarray(accel_m_s2).reshape(3))
    g_up = -az
    roll = math.atan2(ay, g_up)
    pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az) + 1e-6)
    return roll, pitch


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
        accel_tilt_gain: float = 0.0,
        accel_tilt_max_acc: float = 1.5,
        accel_tilt_max_rate: float = math.radians(25.0),
        gate_horizon_gain: float = 0.0,
        gate_horizon_max_step: float = math.radians(1.0),
        gate_horizon_bias_gain: float = 0.30,
        gate_horizon_pitch_scale: float = 0.25,
        gate_yaw_gain: float = 0.0,
        gate_yaw_max_step: float = math.radians(1.0),
        gate_yaw_bias_gain: float = 0.20,
        gate_yaw_anchor_n: int = 15,
        gate_bias_innov_max: float = math.radians(8.0),
        gyro_bias_limit: float = math.radians(1.5),
        use_commanded_accel: bool = True,
        hover_trim: float = 0.255,
        drag_k_body: Optional[np.ndarray] = None,
        commanded_accel_noise: float = 0.14,
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
        self.accel_tilt_gain = accel_tilt_gain
        self.accel_tilt_max_acc = accel_tilt_max_acc
        self.accel_tilt_max_rate = accel_tilt_max_rate
        self.gate_horizon_gain = gate_horizon_gain
        self.gate_horizon_max_step = gate_horizon_max_step
        self.gate_horizon_bias_gain = gate_horizon_bias_gain
        self.gate_horizon_pitch_scale = gate_horizon_pitch_scale
        self.gate_yaw_gain = gate_yaw_gain
        self.gate_yaw_max_step = gate_yaw_max_step
        self.gate_yaw_bias_gain = gate_yaw_bias_gain
        self.gate_yaw_anchor_n = gate_yaw_anchor_n
        self.gate_bias_innov_max = gate_bias_innov_max
        self.gyro_bias_limit = gyro_bias_limit
        self.use_commanded_accel = bool(use_commanded_accel)
        self.hover_trim = float(hover_trim)
        self.drag_k_body = (
            None
            if drag_k_body is None
            else np.asarray(drag_k_body, dtype=np.float64).reshape(3)
        )
        self.commanded_accel_noise = float(commanded_accel_noise)
        self.last_accel_ned = np.zeros(3, dtype=np.float64)
        self.last_horizon_innov = (0.0, 0.0)
        self.last_yaw_innov = 0.0
        self._last_horizon_t: Optional[float] = None
        self._last_yaw_t: Optional[float] = None
        self._last_t: Optional[float] = None
        self._gravity_aligned: bool = False
        self._gate1_ned: Optional[np.ndarray] = None
        self._gate2_ned: Optional[np.ndarray] = None
        self._gate2_fresh = False
        self._last_gate2_update = 0.0
        self._yaw_anchor: Optional[float] = None
        self._yaw_anchor_pos: Optional[np.ndarray] = None
        self._yaw_anchor_acc = np.zeros(2, dtype=np.float64)
        self._yaw_anchor_n = 0

    def reset(self, timestamp: float = 0.0) -> None:
        self.__init__(
            accel_noise=self.accel_noise,
            gyro_noise=self.gyro_noise,
            bias_noise=self.bias_noise,
            pnp_pos_noise=self.pnp_pos_noise,
            accel_tilt_gain=self.accel_tilt_gain,
            accel_tilt_max_acc=self.accel_tilt_max_acc,
            accel_tilt_max_rate=self.accel_tilt_max_rate,
            gate_horizon_gain=self.gate_horizon_gain,
            gate_horizon_max_step=self.gate_horizon_max_step,
            gate_horizon_bias_gain=self.gate_horizon_bias_gain,
            gate_horizon_pitch_scale=self.gate_horizon_pitch_scale,
            gate_yaw_gain=self.gate_yaw_gain,
            gate_yaw_max_step=self.gate_yaw_max_step,
            gate_yaw_bias_gain=self.gate_yaw_bias_gain,
            gate_yaw_anchor_n=self.gate_yaw_anchor_n,
            gate_bias_innov_max=self.gate_bias_innov_max,
            gyro_bias_limit=self.gyro_bias_limit,
            use_commanded_accel=self.use_commanded_accel,
            hover_trim=self.hover_trim,
            drag_k_body=self.drag_k_body,
            commanded_accel_noise=self.commanded_accel_noise,
        )
        self._last_t = timestamp

    def realign_gravity(self, accel_m_s2: np.ndarray) -> bool:
        """Snap roll/pitch to accelerometer tilt; keep yaw.

        Use only when nearly level and quiet — continuous accel blending under
        thrust falsely reads body-level while leaned. Returns True if applied.
        """
        accel = np.asarray(accel_m_s2, dtype=np.float64).reshape(3)
        amag = float(np.linalg.norm(accel))
        if abs(amag - G) > 0.25 * G:
            return False
        roll_a, pitch_a = accel_to_roll_pitch(accel)
        # Refuse large "corrections" — likely still maneuvering / bad sample.
        if abs(roll_a) > math.radians(35.0) or abs(pitch_a) > math.radians(35.0):
            return False
        _roll, _pitch, yaw = quat_to_rpy(self.x[self.IDX_Q : self.IDX_Q + 4])
        self.x[self.IDX_Q : self.IDX_Q + 4] = quat_from_rpy(
            roll_a, pitch_a, yaw
        )
        self._gravity_aligned = True
        return True

    def zero_tilt(self) -> tuple[float, float, float]:
        """Declare current pose as level: set roll/pitch to 0, keep yaw.

        For pure stick flying when the EKF has drifted and there is no vision
        attitude aid. Clear roll/pitch gyro bias so the filter does not
        immediately re-tilt. Returns ``(roll, pitch, yaw)`` after the snap
        (roll/pitch are 0).
        """
        _roll, _pitch, yaw = quat_to_rpy(self.x[self.IDX_Q : self.IDX_Q + 4])
        self.x[self.IDX_Q : self.IDX_Q + 4] = quat_from_rpy(0.0, 0.0, yaw)
        # Roll / pitch gyro bias only — leave yaw bias alone.
        self.x[self.IDX_BG] = 0.0
        self.x[self.IDX_BG + 1] = 0.0
        self._gravity_aligned = True
        return 0.0, 0.0, float(yaw)

    def correct_gate_horizon(self, gate_down_body: np.ndarray) -> bool:
        """Absolute roll/pitch from an upright gate's own vertical axis.

        A gate that hangs true makes its DOWN axis the gravity direction, so
        PnP hands back a horizon that cannot drift and — unlike the
        accelerometer — does not care that the craft is accelerating. This is
        the only bounded attitude reference available mid-race.

        Structured as a Mahony filter: a proportional pull on attitude plus an
        integral term into gyro bias. The bias term is what actually matters —
        it removes the *cause* of the drift, so attitude keeps holding through
        the 0.6–2.5 s stretches with no gate in view (152912).

        Outliers are screened on solve geometry and reprojection error, never
        on how far the measurement sits from the filter's own belief: a drifted
        filter would then reject exactly the evidence that it has drifted.
        Instead every accepted frame is applied with a clamped step, so a rare
        bad pose can only nudge, while a persistent error still gets corrected.
        """
        if self.gate_horizon_gain <= 0.0:
            return False
        d = np.asarray(gate_down_body, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(d)):
            return False
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            return False
        d = d / n
        # Edge-on / degenerate solve: the gate's vertical axis should still
        # point mostly downward in body. Near-horizontal means a bad pose.
        if float(d[2]) < 0.35:
            return False
        # d is world-down seen in body, i.e. the third ROW of the body→NED
        # rotation. Inverting quat_to_rpy's own definition of that row keeps
        # this in exactly the filter's euler convention — routing it through
        # accel_to_roll_pitch instead flips roll, because that helper takes
        # specific force (up-positive), not the gravity direction.
        pitch_g = -math.asin(float(np.clip(d[0], -1.0, 1.0)))
        roll_g = math.atan2(float(d[1]), float(d[2]))
        roll, pitch, yaw = quat_to_rpy(self.x[self.IDX_Q : self.IDX_Q + 4])
        d_roll = _wrap_angle(roll_g - roll)
        d_pitch = _wrap_angle(pitch_g - pitch)
        self.last_horizon_innov = (d_roll, d_pitch)

        # Roll and pitch are NOT equally trustworthy here, and treating them
        # as if they were is what let the pitch channel poison the estimate.
        # A near-frontal planar square barely constrains its own out-of-plane
        # tilt, so corner jitter lands almost entirely in pitch: at 15 m, 1 px
        # of corner error is 0.9 deg of roll but 7.6 deg of pitch. Weight the
        # fragile channel down rather than believing it.
        step = float(self.gate_horizon_max_step)
        w = float(self.gate_horizon_gain)
        w_pitch = w * float(self.gate_horizon_pitch_scale)
        self.x[self.IDX_Q : self.IDX_Q + 4] = quat_from_rpy(
            roll + _clamp(w * d_roll, step),
            pitch + _clamp(w_pitch * d_pitch, step),
            yaw,
        )

        # Integral term: teach the filter its gyro bias. omega = gyro - bg, so
        # a persistent under-read of roll means bg[0] is too big.
        #
        # Anti-windup is not optional here. The integral is only meaningful
        # once the proportional term has pulled the innovation small — a large
        # innovation means the attitude is simply wrong, or the pose is bad,
        # and integrating it drives the bias to its rail. Runs 155234/155700
        # both pinned bg at 8 deg/s that way, which is a rotation rate no real
        # gyro bias reaches, and the filter then flew on that fiction.
        t_now = self._last_t
        if self.gate_horizon_bias_gain > 0.0 and t_now is not None:
            settled = (
                max(abs(d_roll), abs(d_pitch)) <= self.gate_bias_innov_max
            )
            if self._last_horizon_t is not None and settled:
                dt = float(t_now - self._last_horizon_t)
                if 0.0 < dt <= 1.0:
                    ki = float(self.gate_horizon_bias_gain) * dt
                    lim = float(self.gyro_bias_limit)
                    self.x[self.IDX_BG + 0] = _clamp(
                        self.x[self.IDX_BG + 0] - ki * d_roll, lim
                    )
                    self.x[self.IDX_BG + 1] = _clamp(
                        self.x[self.IDX_BG + 1]
                        - ki * float(self.gate_horizon_pitch_scale) * d_pitch,
                        lim,
                    )
            self._last_horizon_t = t_now

        self._gravity_aligned = True
        return True

    def correct_gate_yaw(self, gate_normal_body: np.ndarray) -> bool:
        """Absolute yaw from the gate plane, anchored per gate.

        The gate's through-axis gives heading *relative* to that gate. The
        first solid sighting anchors what the gate's world heading is; later
        sightings then pin yaw against that anchor instead of letting it
        integrate away. Re-anchors when the tracked gate changes.
        """
        if self.gate_yaw_gain <= 0.0:
            return False
        v = np.asarray(gate_normal_body, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(v)):
            return False
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            return False
        v = v / n
        if float(v[0]) < 0.0:
            v = -v  # normal may point either way through the gate
        _roll, _pitch, yaw = quat_to_rpy(self.x[self.IDX_Q : self.IDX_Q + 4])
        # Rotate into NED with the full attitude rather than yaw alone, so
        # roll/pitch do not leak into the heading.
        v_ned = quat_to_rot(self.x[self.IDX_Q : self.IDX_Q + 4]) @ v
        # Gate seen edge-on / normal near vertical: no usable heading.
        if math.hypot(float(v_ned[0]), float(v_ned[1])) < 0.30:
            return False
        heading_now = math.atan2(float(v_ned[1]), float(v_ned[0]))

        gate_pos = self._gate1_ned
        moved = (
            gate_pos is None
            or self._yaw_anchor_pos is None
            or float(np.linalg.norm(gate_pos - self._yaw_anchor_pos)) > 4.0
        )
        if moved:
            self._yaw_anchor_pos = (
                None if gate_pos is None else np.asarray(gate_pos).copy()
            )
            self._yaw_anchor = None
            self._yaw_anchor_acc = np.zeros(2, dtype=np.float64)
            self._yaw_anchor_n = 0

        if self._yaw_anchor_n < self.gate_yaw_anchor_n:
            # Average the opening sightings so one bad pose cannot define the
            # reference, then freeze. An EMA anchor is no good here: it simply
            # walks along with the drift it is supposed to be catching.
            self._yaw_anchor_acc += np.array(
                [math.cos(heading_now), math.sin(heading_now)]
            )
            self._yaw_anchor_n += 1
            if self._yaw_anchor_n >= self.gate_yaw_anchor_n:
                self._yaw_anchor = math.atan2(
                    float(self._yaw_anchor_acc[1]),
                    float(self._yaw_anchor_acc[0]),
                )
            return False

        innov = _wrap_angle(float(self._yaw_anchor) - heading_now)
        self.last_yaw_innov = innov
        self.x[self.IDX_Q : self.IDX_Q + 4] = quat_from_rpy(
            _roll,
            _pitch,
            _wrap_angle(
                yaw
                + _clamp(
                    float(self.gate_yaw_gain) * innov,
                    float(self.gate_yaw_max_step),
                )
            ),
        )
        t_now = self._last_t
        if self.gate_yaw_bias_gain > 0.0 and t_now is not None:
            settled = abs(innov) <= self.gate_bias_innov_max
            if self._last_yaw_t is not None and settled:
                dt = float(t_now - self._last_yaw_t)
                if 0.0 < dt <= 1.0:
                    self.x[self.IDX_BG + 2] = _clamp(
                        self.x[self.IDX_BG + 2]
                        - float(self.gate_yaw_bias_gain) * dt * innov,
                        float(self.gyro_bias_limit),
                    )
            self._last_yaw_t = t_now
        return True

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
        *,
        thrust: Optional[float] = None,
        hover_trim: Optional[float] = None,
    ) -> EKFState:
        """Propagate attitude from gyro; velocity from commanded physics.

        ``accel_m_s2`` is used only for the parked-pad gravity snap (and the
        optional coasting tilt aid). Position / velocity integrate
        ``commanded_accel_ned`` when ``use_commanded_accel`` is set — missing
        thrust is treated as hover (1 g along body −z).
        """
        gyro = np.asarray(gyro_rad_s, dtype=np.float64).reshape(3)
        accel = np.asarray(accel_m_s2, dtype=np.float64).reshape(3)
        if hover_trim is not None and math.isfinite(float(hover_trim)):
            self.hover_trim = float(hover_trim)
        if self._last_t is None:
            self._last_t = timestamp
            return self.state()
        dt = float(timestamp - self._last_t)
        if dt <= 0.0:
            return self.state()
        # CE / slow packet gaps: do not drop the sample (old 50 ms skip left
        # attitude frozen → shallow pilot lean). Cap one step; sub-step rest.
        if dt > 0.25:
            self._last_t = timestamp
            return self.state()
        self._last_t = timestamp

        ba = self.x[self.IDX_BA : self.IDX_BA + 3]
        bg = self.x[self.IDX_BG : self.IDX_BG + 3]
        omega = gyro - bg
        acc_body = accel - ba
        thr = self.hover_trim if thrust is None else float(thrust)
        if not math.isfinite(thr):
            thr = self.hover_trim

        # Gravity tilt before using R for position (else parked tip reads as
        # identity and NED accel is wrong — localize 171202).
        amag = float(np.linalg.norm(accel))
        near_1g = abs(amag - G) <= 0.35 * G
        if near_1g:
            roll_a, pitch_a = accel_to_roll_pitch(accel)
            roll, pitch, yaw = quat_to_rpy(
                self.x[self.IDX_Q : self.IDX_Q + 4]
            )
            if not self._gravity_aligned:
                self.x[self.IDX_Q : self.IDX_Q + 4] = quat_from_rpy(
                    roll_a, pitch_a, yaw
                )
                self._gravity_aligned = True
            elif self.accel_tilt_gain > 0.0:
                # Off by default — see EKF_ACCEL_TILT_GAIN. A wrong attitude
                # estimate fakes the same |acc_ned| as a real lean, so this
                # gate self-locks past ~8° of error and cannot be loosened
                # without re-breaking the held step. Kept for experiments;
                # correct_gate_horizon is the aid that actually bounds drift.
                R_now = quat_to_rot(self.x[self.IDX_Q : self.IDX_Q + 4])
                acc_ned = R_now @ acc_body + G_NED
                coasting = (
                    float(np.linalg.norm(acc_ned)) <= self.accel_tilt_max_acc
                )
                quiet = float(np.linalg.norm(omega)) <= self.accel_tilt_max_rate
                if coasting and quiet:
                    w = min(1.0, float(self.accel_tilt_gain) * dt)
                    self.x[self.IDX_Q : self.IDX_Q + 4] = quat_from_rpy(
                        roll + w * _wrap_angle(roll_a - roll),
                        pitch + w * _wrap_angle(pitch_a - pitch),
                        yaw,
                    )

        remaining = dt
        vel_noise = (
            self.commanded_accel_noise
            if self.use_commanded_accel
            else self.accel_noise
        )
        while remaining > 1e-9:
            step = min(remaining, 0.05)
            remaining -= step
            q = quat_normalize(self.x[self.IDX_Q : self.IDX_Q + 4])
            R = quat_to_rot(q)
            if self.use_commanded_accel:
                acc_ned = commanded_accel_ned(
                    R,
                    thr,
                    self.hover_trim,
                    self.x[self.IDX_V : self.IDX_V + 3],
                    self.drag_k_body,
                )
            else:
                acc_ned = R @ acc_body + G_NED
            self.last_accel_ned = acc_ned

            self.x[self.IDX_P : self.IDX_P + 3] += (
                self.x[self.IDX_V : self.IDX_V + 3] * step
                + 0.5 * acc_ned * step * step
            )
            self.x[self.IDX_V : self.IDX_V + 3] += acc_ned * step
            dq = quat_from_gyro(omega, step)
            self.x[self.IDX_Q : self.IDX_Q + 4] = quat_normalize(
                quat_multiply(q, dq)
            )

            # Covariance: diagonal process noise (practical racing EKF).
            q_pos = (0.5 * vel_noise * step * step) ** 2
            q_vel = (vel_noise * step) ** 2
            q_att = (self.gyro_noise * step) ** 2
            q_bias = (self.bias_noise * step) ** 2
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
