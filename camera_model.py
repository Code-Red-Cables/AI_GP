"""Camera intrinsics and coordinate-frame transforms for the FPV pipeline.

numpy-only (no cv2, no other repo imports). Holds the camera pinhole model and
all conversions between pixel, camera-optical, body, and world (NED) frames.

Frames
------
- Body (NED body):       x = forward, y = right,  z = down.
- Camera-optical (cv2):  x = right,   y = down,   z = forward (optical axis).
- World (NED):           x = North,   y = East,   z = Down (origin at arm point).

The camera optical axis is body-forward pitched UP by ``CAMERA_TILT_UP_DEG``
(rotation about the body-y / right axis).
"""

import numpy as np

# --------------------------------------------------------------------------------------
# Intrinsics / constants (spec §3.7/§3.8)
# --------------------------------------------------------------------------------------
WIDTH, HEIGHT = 640, 360
CX, CY = 320.0, 180.0
FX, FY = 320.0, 320.0
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], float)  # trust this; ignore "VFoV=90"
CAMERA_TILT_UP_DEG = 20.0
GATE_INNER_M = 1.5  # flyable inner square side (1500 mm)

# --------------------------------------------------------------------------------------
# Body -> camera-optical rotation  (v_cam = R_CB @ v_body)
# Axis permutation (body x-fwd/y-right/z-down -> cam x-right/y-down/z-fwd) composed with
# a 20deg pitch-up about the body-y axis.
# --------------------------------------------------------------------------------------
_t = np.radians(CAMERA_TILT_UP_DEG)
R_CB = np.array([
    [0.0,         1.0,  0.0       ],
    [np.sin(_t),  0.0,  np.cos(_t)],
    [np.cos(_t),  0.0, -np.sin(_t)],
])
R_BC = R_CB.T  # cam -> body (orthonormal)


# --------------------------------------------------------------------------------------
# Pinhole model
# --------------------------------------------------------------------------------------
def pixel_to_ray(u, v):
    """Unit viewing ray (camera-optical frame) for pixel (u, v)."""
    ray = np.array([(u - CX) / FX, (v - CY) / FY, 1.0])
    return ray / np.linalg.norm(ray)


def project(point_cam):
    """Pinhole-project a camera-optical point (z > 0) to pixel coords (u, v)."""
    x, y, z = point_cam
    u = FX * (x / z) + CX
    v = FY * (y / z) + CY
    return float(u), float(v)


def deproject(u, v, Z):
    """Camera-optical point at optical-axis depth Z for pixel (u, v)."""
    return Z * np.array([(u - CX) / FX, (v - CY) / FY, 1.0])


def range_from_size(pixel_size, real_size=GATE_INNER_M, f=FX):
    """Fronto-parallel range estimate: Z = f * real_size / pixel_size."""
    return float(f * real_size / pixel_size)


# --------------------------------------------------------------------------------------
# Camera <-> body
# --------------------------------------------------------------------------------------
def cam_to_body(p):
    """Transform a camera-optical vector/point into the body frame."""
    return R_BC @ np.asarray(p, float)


def body_to_cam(p):
    """Transform a body vector/point into the camera-optical frame."""
    return R_CB @ np.asarray(p, float)


# --------------------------------------------------------------------------------------
# Body <-> world (NED), standard aerospace ZYX (yaw-pitch-roll)
# --------------------------------------------------------------------------------------
def rot_world_body(roll, pitch, yaw):
    """World<-body rotation R_wb = Rz(yaw) @ Ry(pitch) @ Rx(roll) (radians)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], float)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], float)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], float)
    return Rz @ Ry @ Rx


def body_to_ned(p, roll, pitch, yaw):
    """Transform a body vector/point into the world (NED) frame."""
    return rot_world_body(roll, pitch, yaw) @ np.asarray(p, float)


def ned_to_body(p, roll, pitch, yaw):
    """Transform a world (NED) vector/point into the body frame."""
    return rot_world_body(roll, pitch, yaw).T @ np.asarray(p, float)
