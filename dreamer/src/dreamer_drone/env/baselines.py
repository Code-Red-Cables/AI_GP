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
from .ahrs import AHRSConfig, ComplementaryAHRS
from .spaces import ACTION_DIM, VECTOR_OBS_FIELDS

_I_TILT_ROLL = VECTOR_OBS_FIELDS.index("tilt_roll")
_I_TILT_PITCH = VECTOR_OBS_FIELDS.index("tilt_pitch")
_I_GX = VECTOR_OBS_FIELDS.index("gyro_x")
_I_GY = VECTOR_OBS_FIELDS.index("gyro_y")
_I_GZ = VECTOR_OBS_FIELDS.index("gyro_z")
_I_AX = VECTOR_OBS_FIELDS.index("ax")
_I_AY = VECTOR_OBS_FIELDS.index("ay")
_I_AZ = VECTOR_OBS_FIELDS.index("az")
_I_DT = VECTOR_OBS_FIELDS.index("dt")


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


class StabilizedController:
    """AHRS-stabilized autopilot demonstrator (training-time replay seeding only).

    Complementary-filter attitude hold + forward lean + visual gate centering + open-loop
    thrust. Because the AHRS gives a trustworthy roll/pitch even during motion, a simple
    P(+D)-on-angle→rate loop converges (angle' = gain·(desired−angle)) instead of chasing
    the corrupted accel-only tilt that made `ScriptedController` limit-cycle.

    Stateful (holds the AHRS) — call `reset()` between episodes. Signs/gains are tuned live
    via `collect_demos.py` flags; if the AHRS diverges (`ahrs.divergence` grows), flip a
    `gyro_sign_*`. Produces flyable trajectories (stable flight, gate approach) — not
    guaranteed full completions, but far better replay seed than random crashes.
    """

    def __init__(self, cfg: ActionConfig, forward_lean: float = 0.10, kp_att: float = 0.6,
                 kd_att: float = 0.03, kp_yaw: float = 0.8, bank_gain: float = 0.3,
                 kp_vert: float = 0.4, gate_v_target: float = 0.58, climb_bias: float = 0.0,
                 rate_cap: float = 0.35, ahrs_cfg: Optional[AHRSConfig] = None,
                 detect_gate: Optional[Callable] = None):
        self.cfg = cfg
        self.forward_lean = forward_lean
        self.kp_att = kp_att
        self.kd_att = kd_att
        self.kp_yaw = kp_yaw
        self.bank_gain = bank_gain
        self.kp_vert = kp_vert                 # gate vertical-pixel error -> thrust (thread the hole)
        self.gate_v_target = gate_v_target     # where the gate should sit vertically (frac of H).
                                               # Camera tilts UP 20°, so an aligned gate sits LOW in
                                               # frame => target >0.5. THE key knob to tune live.
        self.climb_bias = climb_bias
        self.rate_cap = rate_cap
        self.ahrs = ComplementaryAHRS(ahrs_cfg)
        self._detect = detect_gate or _try_load_detector()

    def reset(self) -> None:
        self.ahrs.reset()

    def __call__(self, obs: dict) -> np.ndarray:
        v = obs["vector"]
        gx, gy, gz = float(v[_I_GX]), float(v[_I_GY]), float(v[_I_GZ])
        ax, ay, az = float(v[_I_AX]), float(v[_I_AY]), float(v[_I_AZ])
        dt = float(v[_I_DT])
        roll_est, pitch_est, _ = self.ahrs.update((gx, gy, gz), (ax, ay, az), dt)
        cap = self.rate_cap

        # vision servo: bank + yaw toward gate center (horizontal), thrust toward it (vertical)
        desired_roll, a_yaw, a_vert = 0.0, 0.0, 0.0
        img = obs.get("image")
        if self._detect is not None and img is not None and img.shape[-1] == 3:
            det = self._detect(img[..., ::-1])  # RGB->BGR
            if det is not None:
                h, w = img.shape[0], img.shape[1]
                err_x = (det.center_px[0] - w / 2.0) / (w / 2.0)          # [-1,1] left/right
                err_y = (det.center_px[1] - self.gate_v_target * h) / (h / 2.0)  # up/down
                desired_roll = float(np.clip(self.bank_gain * err_x, -0.3, 0.3))
                a_yaw = float(np.clip(self.kp_yaw * err_x, -cap, cap))
                # gate low in frame (err_y>0) => descend (less thrust); sign is tunable via kp_vert
                a_vert = float(np.clip(-self.kp_vert * err_y, -0.5, 0.5))

        # P(+D)-on-angle -> normalized rate command (sim executes the rate)
        a_roll = float(np.clip(self.kp_att * (desired_roll - roll_est) - self.kd_att * gx,
                               -cap, cap))
        a_pitch = float(np.clip(self.kp_att * (self.forward_lean - pitch_est) - self.kd_att * gy,
                                -cap, cap))
        a_thrust = float(np.clip(self.climb_bias + a_vert, -1.0, 1.0))
        return np.array([a_thrust, a_roll, a_pitch, a_yaw], dtype=np.float32)


def _try_load_detector():
    try:
        repo_root = Path(__file__).resolve().parents[4]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from vision.gate_detector import detect_gate  # type: ignore
        return detect_gate
    except Exception:
        return None
