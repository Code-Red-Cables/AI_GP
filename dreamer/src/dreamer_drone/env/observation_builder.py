"""Assemble the LEGAL, deployment-clean observation.

This module is deliberately handed *only* competition-legal runtime fields. It has no
access to `sim/privileged_state.py`. If a privileged value ever needs to reach the
policy, it must be added to `spaces.VECTOR_OBS_FIELDS` and re-audited — which the
leakage test will catch.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..config import ObsConfig
from .spaces import VECTOR_DIM, image_shape

try:
    import cv2
    _HAVE_CV2 = True
except Exception:  # pragma: no cover - cv2 always present in the real env
    _HAVE_CV2 = False


def build_vector(
    imu: Optional[dict],
    prev_action_norm: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Build the fixed-schema vector observation from LEGAL telemetry only.

    In this VQ2 build ATTITUDE is absent (measured), so everything comes from HIGHRES_IMU:
    body angular rates (gyro), linear accel, and an accelerometer-derived gravity tilt
    (roll/pitch). Yaw is unobservable (no magnetometer) and is intentionally omitted — the
    camera provides heading and the RSSM integrates orientation over the sequence.

    `imu` : dict with x/y/zgyro + x/y/zacc (HIGHRES_IMU) or None.
    `prev_action_norm` : previous normalized action in [-1,1]^4.
    `dt`  : measured seconds since the previous processed frame.
    """
    im = imu or {}
    ax = float(im.get("xacc", 0.0))
    ay = float(im.get("yacc", 0.0))
    az = float(im.get("zacc", 0.0))
    # accelerometer gravity tilt — driftless. MEASURED: acc_z reads ~-9.8 at rest, so use
    # gravity-up = -az; otherwise atan2(ay, az) sits at ±pi exactly at hover (the wrap
    # discontinuity at the operating point). With -az, level flight maps to tilt ≈ 0.
    g_up = -az
    tilt_roll = math.atan2(ay, g_up)
    tilt_pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az) + 1e-6)

    pa = np.asarray(prev_action_norm, dtype=np.float32).reshape(-1)
    if pa.shape[0] != 4:
        pa = np.zeros(4, dtype=np.float32)

    vec = np.array([
        float(im.get("xgyro", 0.0)),
        float(im.get("ygyro", 0.0)),
        float(im.get("zgyro", 0.0)),
        ax, ay, az,
        tilt_roll, tilt_pitch,
        pa[0], pa[1], pa[2], pa[3],
        float(dt),
    ], dtype=np.float32)
    assert vec.shape[0] == VECTOR_DIM, (vec.shape, VECTOR_DIM)
    return vec


def build_image(frame_bgr: Optional[np.ndarray], cfg: ObsConfig) -> np.ndarray:
    """Resize (and optionally grayscale) a camera frame to the obs image.

    Returns uint8 [H, W, C]. A missing frame yields zeros (paired with valid=0 upstream).
    Uses the newest causal frame only — never a future frame.
    """
    h, w, c = image_shape(cfg)
    if frame_bgr is None or frame_bgr.size == 0:
        return np.zeros((h, w, c), dtype=np.uint8)

    if _HAVE_CV2:
        img = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
        if cfg.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., None]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:  # numpy nearest-neighbour fallback (tests without cv2)
        src_h, src_w = frame_bgr.shape[:2]
        ys = (np.linspace(0, src_h - 1, h)).astype(np.int64)
        xs = (np.linspace(0, src_w - 1, w)).astype(np.int64)
        img = frame_bgr[np.ix_(ys, xs)]
        if cfg.grayscale:
            img = img.mean(axis=2, keepdims=True).astype(np.uint8)
    return np.ascontiguousarray(img.astype(np.uint8))


def build_obs(
    frame_bgr: Optional[np.ndarray],
    imu: Optional[dict],
    prev_action_norm: np.ndarray,
    dt: float,
    cfg: ObsConfig,
    image_valid: bool = True,
    telem_valid: bool = True,
) -> dict:
    """Full LEGAL observation dict consumed by the agent and the deploy controller.

    `valid` marks whether the image/telemetry are fresh (not fabricated/held), so the
    world model can learn to distinguish real observations from held ones instead of
    silently trusting duplicated data.
    """
    return {
        "image": build_image(frame_bgr, cfg),
        "vector": build_vector(imu, prev_action_norm, dt),
        "valid": np.array(
            [1.0 if image_valid else 0.0, 1.0 if telem_valid else 0.0],
            dtype=np.float32,
        ),
    }
