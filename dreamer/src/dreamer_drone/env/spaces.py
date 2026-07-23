"""Observation & action space definitions — the single source of truth shared by the
training env, the world model, and the deployment controller. Keeping the vector
schema and the action scaling here guarantees zero train/deploy skew.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..config import ActionConfig, ObsConfig

# --- Vector observation schema (LEGAL-only; see docs/interface_inventory.md) ----------
# Order is binding: the encoder, replay, and deploy controller all rely on it.
#
# MEASURED (probe 2026-07-23): ATTITUDE does NOT arrive in this VQ2 build, so there is no
# direct orientation/yaw. Body rates + linear accel come from HIGHRES_IMU (~114 Hz).
# Orientation is left for the RSSM to integrate; absolute tilt is given driftlessly by the
# accelerometer (gravity); yaw is unobservable without a magnetometer and is carried by the
# camera (you see the gate). This is exactly a human FPV pilot's information set.
VECTOR_OBS_FIELDS: tuple[str, ...] = (
    "gyro_x", "gyro_y", "gyro_z",          # body angular rates (HIGHRES_IMU)
    "ax", "ay", "az",                      # body linear accel (HIGHRES_IMU)
    "tilt_roll", "tilt_pitch",             # accelerometer gravity tilt (driftless, no yaw)
    "prev_thrust", "prev_roll_rate", "prev_pitch_rate", "prev_yaw_rate",
    "dt",
)
VECTOR_DIM: int = len(VECTOR_OBS_FIELDS)  # 13

ACTION_DIM: int = 4  # [thrust, roll_rate, pitch_rate, yaw_rate], normalized to [-1, 1]


class PhysicalAction(NamedTuple):
    thrust: float       # 0..1 collective
    roll_rate: float    # rad/s, body
    pitch_rate: float   # rad/s, body
    yaw_rate: float     # rad/s, body


def image_shape(obs: ObsConfig) -> tuple[int, int, int]:
    c = 1 if obs.grayscale else 3
    return (obs.image_h, obs.image_w, c)


def scale_action(norm_action: np.ndarray, cfg: ActionConfig) -> PhysicalAction:
    """Map a normalized policy action in [-1, 1]^4 to a physical SET_ATTITUDE_TARGET.

    Pure and deterministic. Slew-rate limiting / LPF / watchdog live in the stateful
    `sim/action_sender.py` wrapper; this is the memoryless scaling both train & deploy use.
    """
    a = np.asarray(norm_action, dtype=np.float32).reshape(-1)
    if a.shape[0] != ACTION_DIM:
        raise ValueError(f"action must have {ACTION_DIM} elements, got {a.shape[0]}")
    a = np.clip(a, -1.0, 1.0)

    thrust = cfg.hover_thrust + a[0] * cfg.thrust_span
    thrust = float(np.clip(thrust, cfg.thrust_min, cfg.thrust_max))

    roll_rate = float(a[1] * cfg.max_rate_rad_s * cfg.rate_sign_roll)
    pitch_rate = float(a[2] * cfg.max_rate_rad_s * cfg.rate_sign_pitch)
    yaw_rate = float(a[3] * cfg.max_rate_rad_s * cfg.rate_sign_yaw)
    return PhysicalAction(thrust, roll_rate, pitch_rate, yaw_rate)


def neutral_action() -> np.ndarray:
    """Emergency / hold action: zero rates, hover thrust (a0=0 → hover_thrust)."""
    return np.zeros(ACTION_DIM, dtype=np.float32)
