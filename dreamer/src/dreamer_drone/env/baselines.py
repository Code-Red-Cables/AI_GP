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
        self._det_cfg: Optional[dict] = None
        self._det_cfg_hw: Optional[tuple] = None
        # target-switch rejection: hold the last accepted center through brief detector
        # flips (e.g. to the glow pool under the gate, which sits ~0.15 frame lower and
        # made the vertical servo cut thrust into the ground — measured 2026-07-25)
        self._last_center: Optional[tuple] = None
        self._hold_frames = 0

    def _detector_cfg(self, h: int, w: int) -> dict:
        """Resolution-aware overrides: the controller sees the small obs image, but the
        detector defaults are tuned for the ~640x360 sim frame. min_area=400 px is ~10%
        of a 64x64 thumbnail (gate undetectable until point-blank), the 5px morphology
        kernel erases the opening, and the hole rule then rejects what remains. Scale
        the area floor, shrink the kernel, and require the hole only for blobs big
        enough (>6% of frame) that a real opening survives the downsample."""
        if self._det_cfg_hw != (h, w):
            scale = (h * w) / (640.0 * 360.0)
            small = min(h, w) <= 128
            self._det_cfg = {
                "min_area": max(8.0, 400.0 * scale),
                # kernel 1 = no morphology: at 64x64 the gate is a ~9x8 ring 1-2 px
                # thick; even a 3x3 open shreds it into fragments (measured 2026-07-25)
                "kernel_size": 1 if small else 5,
                "hole_min_bbox_frac": 0.06 if small else 0.025,
            }
            self._det_cfg_hw = (h, w)
        return self._det_cfg

    def reset(self) -> None:
        self.ahrs.reset()
        self._last_center = None
        self._hold_frames = 0

    def _servo_center(self, det, h: int, w: int, trusted: bool) -> Optional[tuple]:
        """Accept/reject a detection center with a short hold across flips.

        A jump of >25% of the frame in one step means the detector switched targets
        (glow pool, sign, next gate). Hold the previous center for a few frames; if
        the jump persists, accept it as a genuine new target. `trusted` (close-range,
        big blob) detections bypass the hold: near the gate large frame-to-frame
        motion is genuine."""
        c = (det.center_px[0], det.center_px[1]) if det is not None else None
        if c is None:
            self._hold_frames += 1
            if self._hold_frames > 5:
                self._last_center = None
            return self._last_center if self._hold_frames <= 5 else None
        if not trusted and self._last_center is not None:
            jump = np.hypot((c[0] - self._last_center[0]) / w,
                            (c[1] - self._last_center[1]) / h)
            if jump > 0.25 and self._hold_frames <= 5:
                self._hold_frames += 1
                return self._last_center
        self._last_center = c
        self._hold_frames = 0
        return c

    def __call__(self, obs: dict) -> np.ndarray:
        v = obs["vector"]
        gx, gy, gz = float(v[_I_GX]), float(v[_I_GY]), float(v[_I_GZ])
        ax, ay, az = float(v[_I_AX]), float(v[_I_AY]), float(v[_I_AZ])
        dt = float(v[_I_DT])
        roll_est, pitch_est, _ = self.ahrs.update((gx, gy, gz), (ax, ay, az), dt)
        cap = self.rate_cap

        # vision servo: bank + yaw toward gate center always; vertical (thrust) only at
        # close range. At long range the gate's vertical pixel position is dominated by
        # the drone's own pitch attitude, not altitude error — servoing thrust on it
        # dove the drone into the ground from spawn (measured 2026-07-25). The original
        # working demos flew the approach with hover thrust and only used the vertical
        # servo as a close-range climb burst through the gate (v_target 0.58).
        desired_roll, a_yaw, a_vert = 0.0, 0.0, 0.0
        img = obs.get("image")
        if self._detect is not None and img is not None and img.shape[-1] == 3:
            h, w = img.shape[0], img.shape[1]
            det = self._detect(img[..., ::-1], self._detector_cfg(h, w))  # RGB->BGR
            # 40 px on the 64x64 obs = any solid detection. The working demo profile is
            # a SUSTAINED gentle climb servoing the centroid to v_target~0.58 through
            # the whole visible approach; late engagement (120-300 px triggers) turned
            # it into a violent last-second burst that clipped the gate bar
            close = det is not None and det.area_px >= 40.0 * (h * w) / 4096.0
            center = self._servo_center(det, h, w, trusted=close)
            if center is not None:
                err_x = (center[0] - w / 2.0) / (w / 2.0)          # [-1,1] left/right
                desired_roll = float(np.clip(self.bank_gain * err_x, -0.3, 0.3))
                a_yaw = float(np.clip(self.kp_yaw * err_x, -cap, cap))
            if close:
                err_y = (det.center_px[1] - self.gate_v_target * h) / (h / 2.0)  # up/down
                # Deadband: the spawn-aligned ballistic path (hover thrust, forward
                # lean) threads gate 1 on its own; proportional correction from spawn
                # (err ~-0.2) lifted the drone off that rail and it missed. Only
                # correct gross vertical deviations.
                dead = 0.30
                if err_y > dead:
                    err_y -= dead
                elif err_y < -dead:
                    err_y += dead
                else:
                    err_y = 0.0
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
