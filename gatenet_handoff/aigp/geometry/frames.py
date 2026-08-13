"""All coordinate frames, rotations and rigid transforms live HERE and only here.

Frames
------
  World  (W): local NED — x north, y east, z DOWN. Anchored at the arming pose
              (drone at rest at the deterministic start defines the origin and yaw 0).
  Body   (B): FRD — x forward, y right, z down (MAVLink body frame).
  Camera (C): OpenCV — x right in image, y down in image, z = optical axis.
              Mounted pitched UP by spec.camera.tilt_up_deg from body +x.
  Gate   (G): x right along the gate face, y UP along the gate face, z out of
              the gate plane toward the approaching drone (matches the OpenCV
              SOLVEPNP_IPPE_SQUARE object convention).

Notation
--------
  R_ab: rotation taking frame-b vectors into frame a  (x_a = R_ab @ x_b).
  Pose T_ab: rigid transform b -> a. Composition: T_ac = T_ab @ T_bc.
  Quaternions: Hamilton, [w, x, y, z] (MAVLink order). q_ab rotates b into a.
  Euler (debug only): aerospace ZYX yaw-pitch-roll.

NED sign reminders (the "above/below" bug lives or dies here):
  up          => negative z
  gate above  => gate z_w < drone z_w
"""

from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Basic rotations
# ---------------------------------------------------------------------------


def R_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def R_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def R_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def rpy_to_rot(roll, pitch, yaw):
    """Aerospace ZYX: R_wb = R_z(yaw) @ R_y(pitch) @ R_x(roll)."""
    return R_z(yaw) @ R_y(pitch) @ R_x(roll)


def rot_to_rpy(R):
    """Inverse of rpy_to_rot. Returns (roll, pitch, yaw)."""
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# Camera mount: the 20-degree tilt appears in this section ONLY
# ---------------------------------------------------------------------------

# Zero-tilt body-from-camera: cam x -> body y, cam y -> body z, cam z -> body x.
_R_BC0 = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])


def R_bc(tilt_up_deg):
    """Body-from-camera. Positive tilt pitches the optical axis UP from body +x
    (rotation about body +y by +tilt: x-axis tips up because NED-body z is down).
    """
    return R_y(np.deg2rad(tilt_up_deg)) @ _R_BC0


def R_cb(tilt_up_deg):
    """Camera-from-body (transpose of R_bc)."""
    return R_bc(tilt_up_deg).T


# ---------------------------------------------------------------------------
# Quaternions — Hamilton, [w, x, y, z]
# ---------------------------------------------------------------------------

def quat_identity():
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    q = q / n
    return q if q[0] >= 0.0 else -q   # canonical sign


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_rot(q):
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rot_to_quat(R):
    """Shepperd's method, numerically stable for all cases."""
    m = np.asarray(R, dtype=float)
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return quat_normalize(np.array([w, x, y, z]))


def quat_exp(rotvec):
    """Rotation vector (axis*angle, rad) -> quaternion."""
    v = np.asarray(rotvec, dtype=float)
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return quat_normalize(np.array([1.0, *(0.5 * v)]))
    axis = v / angle
    return np.array([np.cos(angle / 2), *(axis * np.sin(angle / 2))])


def quat_log(q):
    """Quaternion -> rotation vector (inverse of quat_exp)."""
    q = quat_normalize(q)
    w = np.clip(q[0], -1.0, 1.0)
    vn = np.linalg.norm(q[1:])
    if vn < 1e-12:
        return 2.0 * q[1:]
    angle = 2.0 * np.arctan2(vn, w)
    return q[1:] / vn * angle


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


# ---------------------------------------------------------------------------
# Rigid transforms
# ---------------------------------------------------------------------------

class Pose(NamedTuple):
    """T_ab: rigid transform frame b -> frame a. x_a = R @ x_b + t."""
    R: np.ndarray
    t: np.ndarray

    @staticmethod
    def identity():
        return Pose(np.eye(3), np.zeros(3))

    def __matmul__(self, other):
        if isinstance(other, Pose):
            return Pose(self.R @ other.R, self.R @ other.t + self.t)
        raise TypeError("Pose @ expects Pose; use .apply() for points")

    def inverse(self):
        Rt = self.R.T
        return Pose(Rt, -Rt @ self.t)

    def apply(self, x):
        """Transform point(s): x is (3,) or (N,3)."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return self.R @ x + self.t
        return x @ self.R.T + self.t


def T_bc(tilt_up_deg):
    """Body-from-camera pose. Camera at body origin (no lever arm in spec)."""
    return Pose(R_bc(tilt_up_deg), np.zeros(3))


def body_pose_from_gate_obs(T_wg, T_cg, tilt_up_deg):
    """Localization equation: gate map pose + PnP observation -> body pose.

    Derivation:  T_wc = T_wg @ T_gc  where T_gc = T_cg.inverse()
                 T_wb = T_wc @ T_cb  where T_cb = T_bc.inverse()
    """
    return T_wg @ T_cg.inverse() @ T_bc(tilt_up_deg).inverse()
