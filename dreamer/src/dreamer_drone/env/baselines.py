"""Non-learned baselines for environment validation (prompt Phase 4).

These are NOT for deployment — they exist to prove that resets work, rewards point the
right way, actions are mapped correctly, and the course is solvable through the wrapper
before blaming DreamerV3. Each is a `policy(obs) -> normalized action in [-1,1]^4`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..config import ActionConfig
from .spaces import ACTION_DIM, VECTOR_OBS_FIELDS

_I_TILT_ROLL = VECTOR_OBS_FIELDS.index("tilt_roll")
_I_TILT_PITCH = VECTOR_OBS_FIELDS.index("tilt_pitch")


class RandomPolicy:
    """Strictly-limited random-action smoke test."""

    def __init__(self, thrust_range: tuple[float, float] = (-0.3, 0.3),
                 rate_scale: float = 0.3):
        self.thrust_range = thrust_range
        self.rate_scale = rate_scale

    def __call__(self, obs: dict) -> np.ndarray:
        a = np.random.uniform(-1, 1, ACTION_DIM).astype(np.float32) * self.rate_scale
        a[0] = np.random.uniform(*self.thrust_range)
        return a


class ScriptedController:
    """Stabilized lean-hold attitude controller (the sim is rate-controlled, so a constant
    rate would just tumble — we must hold a lean ANGLE via accelerometer tilt feedback).

    Loop (all from LEGAL obs):
      roll_rate  = kp_att * (0            - tilt_roll)     # keep wings level
      pitch_rate = kp_att * (forward_lean - tilt_pitch)    # hold a small forward lean
      yaw_rate   = kp_yaw * horizontal_gate_error          # visual-servo toward the gate
      thrust     = hover + thrust_bias                     # open-loop (no altitude in VQ2)

    What this validates live: whether it stays airborne (no tumble) and drifts toward
    gates. If it drifts the WRONG way, flip `forward_lean` sign; if it rolls/pitches away
    unstably, the corresponding `kp_att` term needs its sign flipped (report it and I'll flip).
    """

    def __init__(self, cfg: ActionConfig, forward_lean: float = 0.12, kp_att: float = 2.5,
                 kp_yaw: float = 1.0, rate_cap: float = 0.4, thrust_bias: float = 0.0,
                 detect_gate: Optional[Callable] = None):
        self.cfg = cfg
        self.forward_lean = forward_lean
        self.kp_att = kp_att
        self.kp_yaw = kp_yaw
        self.rate_cap = rate_cap
        self.thrust_bias = thrust_bias
        self._detect = detect_gate or _try_load_detector()

    def __call__(self, obs: dict) -> np.ndarray:
        vec = obs["vector"]
        tilt_roll = float(vec[_I_TILT_ROLL])
        tilt_pitch = float(vec[_I_TILT_PITCH])
        cap = self.rate_cap

        a = np.zeros(ACTION_DIM, dtype=np.float32)
        a[0] = self.thrust_bias                                              # 0 => hover
        a[1] = np.clip(self.kp_att * (0.0 - tilt_roll), -cap, cap)           # level
        a[2] = np.clip(self.kp_att * (self.forward_lean - tilt_pitch), -cap, cap)  # forward lean

        img = obs.get("image")
        if self._detect is not None and img is not None and img.shape[-1] == 3:
            det = self._detect(img[..., ::-1])  # RGB->BGR for the detector
            if det is not None:
                w = img.shape[1]
                err = (det.center_px[0] - w / 2.0) / (w / 2.0)  # [-1,1] horizontal
                a[3] = float(np.clip(self.kp_yaw * err, -cap, cap))  # yaw toward gate
        return np.clip(a, -1, 1)


def _try_load_detector():
    try:
        repo_root = Path(__file__).resolve().parents[4]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from vision.gate_detector import detect_gate  # type: ignore
        return detect_gate
    except Exception:
        return None
