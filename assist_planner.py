"""Assist planner — always track current primary gate (gate1).

When gate2 is promoted it becomes gate1; same policy continues.

Policy (user-defined):
- Yaw: primarily gate1 PnP bearing atan2(body_y, body_x), blended with image
  nx (093229: image nx stayed ~0 while drifting into gate-2 left edge).
  Do *not* use pitch to keep the gate in frame (092525).
- Altitude: gate1 PnP geometric height vs us (body→NED z) only — no fixed
  sink/aim bias. Thrust scales in proportion to |pose_dz|. Never pitch-from-ny.
- Forward/lateral lean: gate1 PnP body (range / body-y).
- Prefer live gate2 latch; on course 2 after gate 1, if no live latch,
  seed memorized right+slightly-up aim (ASSIST_POST_G1_*). Blind L/R
  scan is a last resort (it shook the craft on 124804).
- Seek may look up to TWO gates ahead (latch + live within
  ASSIST_SEEK_MAX_AHEAD_M). Never chase end-course / gate-3+ far boxes.

Plant: manual teleop attitude+thrust (KALMAN_KP_ATT / HOVER_THRUST).
"""

from __future__ import annotations

import math
import time

import numpy as np

import camera_model as cm
import config
from control.pid import PIDConfig, PIDController


def _f(value, default=None):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def image_gate_norm(shared_data):
    """Best image-normalized gate centre (nx, ny, source) or Nones.

    Prefer the YOLO identity-locked box over dual_pnp.gate1 (nearest-solved
    can steal identity onto a far gate).
    """
    det = shared_data.get('gate_detection') or {}
    center = det.get('center_px') if isinstance(det, dict) else None
    if center is not None and len(center) >= 2:
        width, height = 640.0, 360.0
        cx, cy = float(center[0]), float(center[1])
        return (
            (cx - width * 0.5) / (width * 0.5),
            (cy - height * 0.5) / (height * 0.5),
            'yolo',
        )
    dual = shared_data.get('dual_gate_pnp') or {}
    nx = _f(dual.get('gate1_norm_x'))
    ny = _f(dual.get('gate1_norm_y'))
    if nx is not None and ny is not None and int(dual.get('n_solved') or 0) >= 1:
        return nx, ny, 'dual_pnp'
    return None, None, 'none'


def next_gate_hint(shared_data):
    """Live gate2 pose only — never memorized course_bearing / default_right.

    Caps at ASSIST_SEEK_MAX_AHEAD_M (two gates ahead). Farther = ignore.
    Also ignores nearer than ASSIST_LATCH_MIN_AHEAD_M and aims above
    ASSIST_LATCH_NY_MIN (120804/122209: cleared-gate residual poisoned seek).

    Returns (nx, ny, source, range_m) — any field may be None.
    """
    dual = shared_data.get('dual_gate_pnp') or {}
    g2 = dual.get('gate2_body')
    if g2 is not None and len(g2) >= 3:
        try:
            x, y, z = float(g2[0]), float(g2[1]), float(g2[2])
        except (TypeError, ValueError):
            x = y = z = 0.0
        if x > 0.5 and math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
            rng = float(math.sqrt(x * x + y * y + z * z))
            max_ahead = float(getattr(config, 'ASSIST_SEEK_MAX_AHEAD_M', 28.0))
            min_ahead = float(getattr(config, 'ASSIST_LATCH_MIN_AHEAD_M', 14.0))
            if rng > max_ahead or rng < min_ahead:
                return None, None, 'none', None
            nx = float(np.clip(y / x, -1.2, 1.2))
            ny = float(np.clip(z / x, -1.2, 1.2))
            ny_min = float(getattr(config, 'ASSIST_LATCH_NY_MIN', -0.05))
            if ny < ny_min:
                return None, None, 'none', None
            return nx, ny, 'gate2_body', rng
    return None, None, 'none', None


def pose_aim_y_m() -> float:
    """Body-right aim offset (m); residual ey−aim is the proportional nudge."""
    return float(getattr(config, 'ASSIST_POSE_AIM_Y_M', 0.0))


def pose_bearing_yaw_rad(body, aim_y_m: float | None = None) -> float | None:
    """Yaw error (rad) to face gate aim. + = aim point right of nose.

    aim_y_m shifts the target in body-y so lateral nudge stays ∝ pose
    (atan2((ey−aim)/ex) shrinks with range — not a fixed image offset).
    """
    if body is None:
        return None
    try:
        ex = float(body[0])
        ey = float(body[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not (math.isfinite(ex) and math.isfinite(ey)):
        return None
    if aim_y_m is None:
        aim_y_m = pose_aim_y_m()
    return float(math.atan2(ey - float(aim_y_m), max(ex, 0.4)))


def forward_speed_mps(shared_data, yaw: float, pitch: float, max_lean: float) -> float:
    """Body-forward speed (m/s). Prefer NED velocity; else pitch-lean proxy."""
    pos = shared_data.get('local_position_ned') or {}
    vn = _f(pos.get('vx'))
    ve = _f(pos.get('vy'))
    if vn is not None and ve is not None:
        # Project ground velocity onto body-forward (yaw).
        return max(0.0, vn * math.cos(yaw) + ve * math.sin(yaw))
    # VQ2 often has no LOCAL_POSITION_NED — lean stands in for speed.
    v_ref = float(getattr(config, 'ASSIST_CAM_TILT_SPEED_MPS', 6.0))
    lean = max(1e-3, abs(float(max_lean)))
    fwd_sign = float(getattr(config, 'FORWARD_PITCH_SIGN', 1.0))
    pitch_fwd = max(0.0, fwd_sign * float(pitch))
    return float(np.clip(pitch_fwd / lean, 0.0, 1.2) * v_ref)


def lateral_speed_mps(shared_data, yaw: float, roll: float, max_lean: float) -> float:
    """Body-right speed (m/s). + = moving right. Prefer NED; else roll proxy."""
    pos = shared_data.get('local_position_ned') or {}
    vn = _f(pos.get('vx'))
    ve = _f(pos.get('vy'))
    if vn is not None and ve is not None:
        # Body-right = (-sin yaw, cos yaw) · (vn, ve).
        return float(-vn * math.sin(yaw) + ve * math.cos(yaw))
    # No velocity: bank stands in ( +roll ≈ right wing down ≈ fly right ).
    v_ref = float(getattr(config, 'ASSIST_CAM_TILT_SPEED_MPS', 6.0))
    lean = max(1e-3, abs(float(max_lean)))
    return float(np.clip(float(roll) / lean, -1.2, 1.2) * v_ref)


def cam_bank_lateral_bias_nx(
    roll: float,
    lat_speed_mps: float,
    fwd_speed_mps: float,
    max_lean: float,
) -> float:
    """nx units to ADD to measured image nx (cancel bank/strafe cam coupling).

    Fly left → body banks left → camera look shifts right → gate appears too
    far left in the frame (nx too negative). Add a positive bias so yaw/roll
    do not chase the false left. Symmetric for fly-right. Bias grows with
    |bank| and speed (same idea as cam_tilt_height_bias_m).
    """
    gain = float(getattr(config, 'ASSIST_CAM_ROLL_BIAS', 0.40))
    if gain <= 0.0:
        return 0.0
    lean = max(1e-3, abs(float(max_lean)))
    v_ref = max(0.5, float(getattr(config, 'ASSIST_CAM_TILT_SPEED_MPS', 6.0)))
    bank = float(roll)
    lat_v = float(lat_speed_mps)
    # +rightward flight from bank and/or lateral velocity.
    rightward = float(
        np.clip(bank / lean, -1.5, 1.5)
        + np.clip(lat_v / v_ref, -1.5, 1.5)
    )
    if abs(rightward) < 0.05:
        return 0.0
    speed_mag = max(abs(lat_v), 0.45 * abs(float(fwd_speed_mps)))
    speed_scale = float(np.clip(speed_mag / v_ref, 0.0, 1.5))
    bank_scale = float(np.clip(abs(bank) / lean, 0.0, 1.5))
    tilt_scale = float(np.clip(1.0 + max(speed_scale, bank_scale), 1.0, 2.5))
    # Fly right → gate looks too right → subtract from nx (negative bias).
    unit = float(np.clip(0.5 * rightward, -1.2, 1.2))
    bias = -gain * tilt_scale * unit
    return float(np.clip(bias, -0.55, 0.55))


def cam_tilt_height_bias_m(
    pose_dz: float,
    horiz_m: float,
    ny: float,
    pitch: float,
    speed_mps: float,
) -> float:
    """Metres (NED-down) to add to pose_dz to cancel cam look-up.

    Fixed 20° cam-up couples into gate height. Faster flight ⇒ more forward
    body tilt ⇒ *more* bias. 090736: centre-weight went to 0 as ny rose
    (gate low in frame) and residual dz kept climbing — keep a floor on the
    weight so bias does not vanish exactly when we are already high.
    """
    gain = float(getattr(config, 'ASSIST_CAM_TILT_BIAS', 1.0))
    if gain <= 0.0 or horiz_m <= 0.2:
        return 0.0
    # Only disable look-up cancel when the gate is clearly below the
    # same-height cam-tilt band (≈0.65). 092019: ny≈0.5 is normal on
    # gate-1 approach — still need bias so we do not false-sink.
    if float(ny) > 0.70:
        return 0.0
    cam_tilt = float(
        getattr(config, 'CAMERA_TILT_RAD', math.radians(cm.CAMERA_TILT_UP_DEG))
    )
    fwd_sign = float(getattr(config, 'FORWARD_PITCH_SIGN', 1.0))
    pitch_fwd = max(0.0, fwd_sign * float(pitch))
    v_ref = max(0.5, float(getattr(config, 'ASSIST_CAM_TILT_SPEED_MPS', 6.0)))
    # More speed / more nose-down lean → stronger bias.
    speed_scale = float(np.clip(float(speed_mps) / v_ref, 0.0, 1.5))
    pitch_scale = float(
        np.clip(pitch_fwd / max(cam_tilt, 1e-3), 0.0, 1.5)
    )
    tilt_scale = float(np.clip(1.0 + max(speed_scale, pitch_scale), 1.0, 2.5))
    # Near image centre → full look-up cancel (partial left residual climb).
    if abs(float(ny)) < 0.15:
        center_w = 1.0
    else:
        center_w = float(np.clip(1.0 - abs(float(ny)) / 0.55, 0.0, 1.0))
    bias = (
        gain
        * tilt_scale
        * center_w
        * float(horiz_m)
        * math.sin(cam_tilt)
    )
    # Never push a "gate below" reading further down; only cancel look-up loft.
    if pose_dz >= 0.0:
        return 0.0
    return float(min(bias, -pose_dz))


def gate1_body_m(shared_data):
    """Primary gate centre in body frame (x fwd, y right, z down), or None."""
    dual = shared_data.get('dual_gate_pnp') or {}
    g1 = dual.get('gate1_body')
    if g1 is None or len(g1) < 3:
        return None
    try:
        body = np.array(
            [float(g1[0]), float(g1[1]), float(g1[2])], dtype=np.float64
        )
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(body)) or float(body[0]) < 0.5:
        return None
    return body


class AssistImagePlanner:
    """Auto image IBVS on the manual attitude+thrust plant."""

    name = 'assist_image'

    def __init__(self):
        # 093229: 30°/s + 120°/s slew could not catch gate-2 left drift.
        max_yaw = min(
            config.YAW_RATE_MAX_RAD_S,
            math.radians(
                float(getattr(config, 'ASSIST_YAW_MAX_DEG', 55.0))
            ),
        )
        max_rate = float(config.KALMAN_MAX_RATE_RAD_S)
        self._max_yaw = max_yaw
        self._max_lean = math.radians(
            float(getattr(config, 'ASSIST_LEAN_DEG', config.KALMAN_MAX_LEAN_DEG))
        )
        self._fwd_sign = float(getattr(config, 'FORWARD_PITCH_SIGN', 1.0))
        self._lat_sign = float(getattr(config, 'LATERAL_LEAN_SIGN', 1.0))
        self._yaw_kp = float(getattr(config, 'ASSIST_KP_YAW', 2.4))
        self._yaw_pid = PIDController(
            PIDConfig(
                kp=self._yaw_kp,
                kd=0.0,
                output_min=-max_yaw,
                output_max=max_yaw,
            )
        )
        self._roll_pid = PIDController(
            PIDConfig(
                kp=float(config.KALMAN_KP_ATT),
                kd=float(config.KALMAN_KD_ATT),
                output_min=-max_rate,
                output_max=max_rate,
            )
        )
        self._pitch_pid = PIDController(
            PIDConfig(
                kp=float(config.KALMAN_KP_ATT),
                kd=float(config.KALMAN_KD_ATT),
                output_min=-max_rate,
                output_max=max_rate,
            )
        )
        self._yaw_slew = math.radians(
            float(getattr(config, 'ASSIST_YAW_SLEW_DEG', 360.0))
        )
        self._last_yaw_cmd = 0.0
        self._last_t = None
        self._nx_f = 0.0
        self._ny_f = 0.0
        self._have_filt = False
        self._arm_z = None
        self._peak_climbed = 0.0
        self._last_see_t = 0.0
        self._active_gate = None
        self._coast_until = 0.0
        self._seek_until = 0.0
        self._pass_t = None
        self._last_status_t = 0.0
        self._last_range_m = None
        self._climb_f = None
        self._climb_rate = 0.0  # m/s, + = ascending
        self._climb_rate_t = None
        self._climb_rate_z = None
        self._lift_start_t = None
        self._left_pad = False
        self._airborne_t = None
        self._last_area_px = None
        self._body_f = None
        # Post-pass: lock next gate/pose before yawing (094827 haywire).
        self._gate_lock = True
        self._lock_count = 0
        self._lock_nx = None
        self._lock_ok_t = None
        self._seek_seen = False  # first next-gate glimpse after a pass
        # Last good next-gate (gate2) aim — kept across the pass.
        self._next_nx = None
        self._next_ny = None
        self._next_rng = None
        self._next_body = None
        self._next_t = None
        # Last live gate2 refresh (not a pass-seed echo).
        self._next_live_t = None
        # Course-2 one-shot: only g1→g2 (never after later gates).
        self._course_mem = False
        self._course_mem_yaw_tgt = None
        self._course_mem_spent = False
        self._course_mem_done = False
        self._course_mem_yaw_integ = 0.0
        self._course_mem_yaw_budget = 0.0
        self._next_course_mem = False
        self._des_pitch_f = None
        # Approach snapshot of next-next (survives dual dropout in the slot).
        self._snap_next_nx = None
        self._snap_next_ny = None
        self._snap_next_rng = None
        self._snap_next_body = None
        self._snap_next_t = None

    def reset_episode(self):
        self._yaw_pid.reset()
        self._roll_pid.reset()
        self._pitch_pid.reset()
        self._last_yaw_cmd = 0.0
        self._last_t = None
        self._nx_f = 0.0
        self._ny_f = 0.0
        self._have_filt = False
        self._arm_z = None
        self._peak_climbed = 0.0
        self._last_see_t = 0.0
        self._active_gate = None
        self._coast_until = 0.0
        self._seek_until = 0.0
        self._pass_t = None
        self._last_range_m = None
        self._climb_f = None
        self._climb_rate = 0.0
        self._climb_rate_t = None
        self._climb_rate_z = None
        self._lift_start_t = None
        self._left_pad = False
        self._airborne_t = None
        self._last_area_px = None
        self._body_f = None
        self._gate_lock = True
        self._lock_count = 0
        self._lock_nx = None
        self._lock_ok_t = None
        self._seek_seen = False
        self._next_nx = None
        self._next_ny = None
        self._next_rng = None
        self._next_body = None
        self._next_t = None
        self._next_live_t = None
        self._snap_next_nx = None
        self._snap_next_ny = None
        self._snap_next_rng = None
        self._snap_next_body = None
        self._snap_next_t = None
        self._course_mem = False
        self._course_mem_yaw_tgt = None
        self._course_mem_spent = False
        self._course_mem_done = False
        self._course_mem_yaw_integ = 0.0
        self._course_mem_yaw_budget = 0.0
        self._next_course_mem = False
        self._des_pitch_f = None

    def _reset_gate_lock(self) -> None:
        self._gate_lock = False
        self._lock_count = 0
        self._lock_nx = None
        self._lock_ok_t = None

    def _clear_next_latch(self) -> None:
        self._next_nx = None
        self._next_ny = None
        self._next_rng = None
        self._next_body = None
        self._next_t = None
        self._next_live_t = None
        self._course_mem = False
        self._course_mem_yaw_tgt = None
        self._next_course_mem = False
        # Keep spent/done/integ — one-shot turn must not restart mid-episode.

    def _clear_snap_latch(self) -> None:
        self._snap_next_nx = None
        self._snap_next_ny = None
        self._snap_next_rng = None
        self._snap_next_body = None
        self._snap_next_t = None

    @staticmethod
    def _latch_ahead_ok(rng, ny=None) -> bool:
        """True if range/ny look like a real next gate, not the cleared slot."""
        if rng is None:
            return False
        try:
            r = float(rng)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(r):
            return False
        min_ahead = float(getattr(config, 'ASSIST_LATCH_MIN_AHEAD_M', 14.0))
        max_ahead = float(getattr(config, 'ASSIST_SEEK_MAX_AHEAD_M', 28.0))
        if not (min_ahead <= r <= max_ahead):
            return False
        if ny is not None:
            try:
                ny_f = float(ny)
            except (TypeError, ValueError):
                return False
            if ny_f < float(getattr(config, 'ASSIST_LATCH_NY_MIN', -0.05)):
                return False
        return True

    def _maybe_freeze_approach_latch(
        self, range_m, area_px, *, seeking: bool, coasting: bool, now: float
    ) -> None:
        """Snapshot live gate2 while closing so pass still has a next aim.

        115959: during gate-2 commit dual often drops to n_solved=1, latch
        ages out (>6 s), and GATE_PASSED seeds nothing for gate 3.
        120804/122209: must be clearly behind + not above-frame residual.
        """
        if seeking or coasting:
            return
        if self._next_nx is None or self._next_ny is None or self._next_t is None:
            return
        if not self._latch_ahead_ok(self._next_rng, self._next_ny):
            return
        freeze_m = float(
            getattr(config, 'ASSIST_LATCH_APPROACH_FREEZE_M', 12.0)
        )
        close = (
            range_m is not None and float(range_m) <= freeze_m
        ) or (area_px is not None and float(area_px) >= 4000.0)
        if not close:
            return
        margin = float(getattr(config, 'ASSIST_LATCH_BEHIND_MARGIN_M', 6.0))
        if range_m is not None and float(self._next_rng) < float(range_m) + margin:
            # Not behind the active gate — same slot / residual.
            return
        latch_age = float(now) - float(self._next_t)
        # Only refresh snap from a fresh live latch; keep last snap if stale.
        if latch_age > 3.0:
            return
        self._snap_next_nx = float(self._next_nx)
        self._snap_next_ny = float(self._next_ny)
        self._snap_next_rng = self._next_rng
        self._snap_next_body = (
            None if self._next_body is None else self._next_body.copy()
        )
        self._snap_next_t = float(now)

    def _latch_still_valid(self, now: float, *, max_age: float) -> bool:
        if self._next_nx is None or self._next_ny is None or self._next_t is None:
            return False
        if not self._latch_ahead_ok(self._next_rng, self._next_ny):
            return False
        return (float(now) - float(self._next_t)) < float(max_age)

    def _seed_course_memory(
        self,
        now: float,
        ag_i: int,
        yaw: float,
        hint_nx: float | None = None,
    ) -> bool:
        """Course-2 one-shot: after gate 1 only, turn right then hold.

        Never runs after later gates (`ag_i != 1`) and never twice
        (`_course_mem_spent`). Live latch/snap is preferred when present.
        """
        if not bool(getattr(config, 'ASSIST_COURSE_MEMORY', True)):
            return False
        if int(getattr(config, 'ASSIST_COURSE', 2)) != 2:
            return False
        # One-shot: only the g1→g2 hunt (active_gate==1 after clearing g0/g1).
        if int(ag_i) != 1 or bool(getattr(self, '_course_mem_spent', False)):
            return False
        nx = float(getattr(config, 'ASSIST_POST_G1_NX', 0.35))
        ny = float(getattr(config, 'ASSIST_POST_G1_NY', 0.05))
        rng = float(getattr(config, 'ASSIST_POST_G1_RANGE_M', 16.0))
        yaw_ofs = math.radians(
            float(getattr(config, 'ASSIST_POST_G1_YAW_DEG', 28.0))
        )
        # Extra right if we already saw gate2 far right before the pass.
        if hint_nx is not None and math.isfinite(float(hint_nx)):
            hx = max(0.0, float(hint_nx))
            extra_per = math.radians(
                float(getattr(config, 'ASSIST_POST_G1_YAW_EXTRA_PER_NX', 40.0))
            )
            yaw_ofs += extra_per * hx
            nx = max(nx, hx)
        self._next_nx = nx
        self._next_ny = ny
        self._next_rng = rng
        self._next_body = None
        self._next_t = now
        self._next_live_t = now
        self._next_course_mem = True
        self._course_mem = True
        self._course_mem_spent = True
        self._course_mem_done = False
        self._course_mem_yaw_integ = 0.0
        self._course_mem_yaw_budget = abs(float(yaw_ofs))
        # Heading target — yaw-rate → 0 when reached (no perpetual nx hunt).
        self._course_mem_yaw_tgt = float(yaw) + yaw_ofs
        return True

    def _course_mem_heading_err(self, yaw: float) -> float | None:
        """Wrapped heading error to post-g1 target, or None if unset."""
        tgt = getattr(self, '_course_mem_yaw_tgt', None)
        if tgt is None:
            return None
        err = float(tgt) - float(yaw)
        while err > math.pi:
            err -= 2.0 * math.pi
        while err < -math.pi:
            err += 2.0 * math.pi
        return err

    def _course_mem_mark_done(self) -> None:
        self._course_mem_done = True
        self._course_mem = False
        self._last_yaw_cmd = 0.0
        # Drop synthetic g1→g2 aim so we do not forever chase nx=+0.55 / floor
        # after the bounded turn (133934).
        if bool(getattr(self, '_next_course_mem', False)):
            self._clear_next_latch()
            self._have_filt = False
            self._nx_f = 0.0
            self._ny_f = 0.0
            self._body_f = None

    def _course_mem_heading_done(self, yaw: float, now: float | None = None) -> bool:
        """True once the post-g1 right turn is finished (sticky).

        133354: attitude yaw jumped backward after overshoot and re-armed a
        perpetual right turn — once done, stay done. Also finish on commanded
        angle budget / max time (plant yaw sign can disagree with attitude).
        """
        if bool(getattr(self, '_course_mem_done', False)):
            return True
        # Commanded-angle budget (robust if attitude drifts the wrong way).
        budget = float(getattr(self, '_course_mem_yaw_budget', 0.0) or 0.0)
        integ = abs(float(getattr(self, '_course_mem_yaw_integ', 0.0) or 0.0))
        if budget > 1e-3 and integ >= 0.90 * budget:
            self._course_mem_mark_done()
            return True
        # Hard time cap — 133354 kept yawing right for many seconds.
        if (
            now is not None
            and self._pass_t is not None
            and (float(now) - float(self._pass_t))
            >= float(getattr(config, 'ASSIST_POST_G1_YAW_MAX_S', 1.8))
        ):
            self._course_mem_mark_done()
            return True
        err = self._course_mem_heading_err(yaw)
        if err is None:
            self._course_mem_mark_done()
            return True
        dead = math.radians(
            float(getattr(config, 'ASSIST_POST_G1_YAW_DEAD_DEG', 5.0))
        )
        if abs(err) < dead:
            self._course_mem_mark_done()
            return True
        mem_nx = float(
            self._next_nx
            if self._next_nx is not None
            else getattr(config, 'ASSIST_POST_G1_NX', 0.55)
        )
        # Overshot past target on the memorized side → done (no reverse).
        if mem_nx >= 0.0 and err < 0.0:
            self._course_mem_mark_done()
            return True
        if mem_nx < 0.0 and err > 0.0:
            self._course_mem_mark_done()
            return True
        return False

    def _course_mem_yaw_rate(
        self, yaw: float, dt: float, now: float | None = None
    ) -> float:
        """Yaw to post-g1 heading for a bounded turn, then stop.

        132813: live cleared mem too early. 133354: never released → spun right.
        """
        if self._course_mem_heading_done(yaw, now=now):
            return 0.0
        err = self._course_mem_heading_err(yaw)
        if err is None:
            self._course_mem_mark_done()
            return 0.0
        # Prefer budget direction (right on course 2) over noisy attitude err.
        budget = float(getattr(self, '_course_mem_yaw_budget', 0.0) or 0.0)
        mem_nx = float(
            self._next_nx
            if self._next_nx is not None
            else getattr(config, 'ASSIST_POST_G1_NX', 0.55)
        )
        turn_sign = 1.0 if mem_nx >= 0.0 else -1.0
        amp = math.radians(
            float(getattr(config, 'ASSIST_POST_G1_YAW_RATE_MAX_DEG', 45.0))
        )
        amp = min(amp, float(self._max_yaw))
        floor = math.radians(
            float(getattr(config, 'ASSIST_POST_G1_YAW_FLOOR_DEG', 22.0))
        )
        # Rate from remaining budget (smooth taper near the end).
        remain = max(0.0, budget - abs(float(self._course_mem_yaw_integ)))
        if remain < math.radians(8.0):
            yaw_rate = turn_sign * max(remain * 2.5, floor * 0.35)
        else:
            yaw_rate = turn_sign * max(floor, min(amp, abs(float(err)) * 1.2))
        yaw_rate = float(np.clip(yaw_rate, -amp, amp))
        max_dps = float(getattr(config, 'ASSIST_YAW_SLEW_DEG', 180.0))
        max_step = math.radians(max_dps) * max(1e-3, float(dt))
        prev = float(self._last_yaw_cmd)
        yaw_rate = float(np.clip(yaw_rate, prev - max_step, prev + max_step))
        self._course_mem_yaw_integ = float(self._course_mem_yaw_integ) + (
            float(yaw_rate) * max(1e-3, float(dt))
        )
        # Finish if this tick completes the budget.
        if abs(float(self._course_mem_yaw_integ)) >= 0.90 * max(budget, 1e-3):
            self._course_mem_mark_done()
            return 0.0
        self._last_yaw_cmd = yaw_rate
        return yaw_rate

    def _seek_blind_scan_yaw(
        self, now: float, yaw: float = 0.0, dt: float = 0.02
    ) -> float:
        """Seek yaw with no live box.

        Course-2 memory: heading hold (125233 perpetual nx=+0.35 shook yaw).
        Fallback: small damped L/R scan.
        """
        t0 = float(self._pass_t) if self._pass_t is not None else float(now)
        age = max(0.0, float(now) - t0)
        ease = float(np.clip(age / 0.6, 0.0, 1.0))
        if getattr(self, '_course_mem', False):
            return self._course_mem_yaw_rate(yaw, dt, now=now)
        # If course-mem already finished, never fall into perpetual scan spin.
        amp = float(getattr(config, 'ASSIST_SEEK_SCAN_YAW_RAD', 0.22))
        hz = float(getattr(config, 'ASSIST_SEEK_SCAN_HZ', 0.12))
        yaw_cmd = amp * ease * math.sin(2.0 * math.pi * hz * age)
        return float(np.clip(yaw_cmd, -amp, amp))

    def _update_next_latch(self, shared_data, now: float) -> None:
        """Remember live gate2 pose so we can soft-aim after the pass.

        Never replace a good near-ahead latch with a much farther hint
        (111515: 8→44 m mid-frame steal). Nearer than MIN_AHEAD is junk
        (120804) — always allow a real ahead hint to overwrite it.
        """
        hx, hy, _src, hrng = next_gate_hint(shared_data)
        if hx is None or hy is None:
            return
        if not self._latch_ahead_ok(hrng, hy):
            return
        jump = float(getattr(config, 'ASSIST_LATCH_MAX_RANGE_JUMP_M', 6.0))
        freeze_s = float(getattr(config, 'ASSIST_LATCH_FREEZE_S', 5.0))
        post_pass = (
            self._pass_t is not None
            and (now - float(self._pass_t)) < freeze_s
        )
        have_good = self._latch_ahead_ok(self._next_rng, self._next_ny)
        if (
            have_good
            and self._next_rng is not None
            and hrng is not None
            and float(hrng) > float(self._next_rng) + jump
        ):
            # Farther than a good latch — keep the nearer ahead aim.
            return
        if (
            post_pass
            and have_good
            and self._next_rng is not None
            and hrng is not None
            and float(hrng) > float(self._next_rng) + 1.5
        ):
            # Freeze: only equal/nearer refreshes right after a pass.
            return
        self._next_nx = float(hx)
        self._next_ny = float(hy)
        self._next_rng = float(hrng) if hrng is not None else self._next_rng
        self._next_t = now
        self._next_live_t = now
        self._next_course_mem = False
        g2 = (shared_data.get('dual_gate_pnp') or {}).get('gate2_body')
        if g2 is not None and len(g2) >= 3:
            try:
                body = np.array(
                    [float(g2[0]), float(g2[1]), float(g2[2])],
                    dtype=np.float64,
                )
            except (TypeError, ValueError):
                return
            if math.isfinite(body[0]) and body[0] > 0.5:
                body_rng = float(np.linalg.norm(body[:3]))
                if not self._latch_ahead_ok(body_rng, hy):
                    return
                if (
                    have_good
                    and self._next_body is not None
                    and body_rng
                    > float(np.linalg.norm(self._next_body[:3])) + jump
                ):
                    return
                self._next_body = body

    def _update_gate_lock(self, chaseable, nx, body, seeking: bool) -> bool:
        """Require a stable gate sighting before enabling full chase."""
        if not seeking:
            self._gate_lock = True
            self._lock_count = 0
            self._lock_nx = None
            return True
        need = int(getattr(config, 'ASSIST_LOCK_FRAMES', 12))
        max_jump = float(getattr(config, 'ASSIST_LOCK_NX_JUMP', 0.18))
        if not chaseable or nx is None:
            # 102331: sticky lock + flicker → seek_scan yaw=0. Drop immediately;
            # ghost hold keeps soft-yawing on the last aim.
            self._gate_lock = False
            self._lock_count = 0
            self._lock_nx = None
            self._lock_ok_t = None
            return False
        nx_f = float(nx)
        if self._gate_lock:
            # Stay locked through small jitter — drop on a huge identity jump.
            if (
                self._lock_nx is not None
                and abs(nx_f - self._lock_nx) > max_jump * 2.8
            ):
                self._gate_lock = False
                self._lock_count = 1
                self._lock_ok_t = None
                self._lock_nx = nx_f
                self._last_yaw_cmd = 0.0
            else:
                if self._lock_nx is None:
                    self._lock_nx = nx_f
                else:
                    self._lock_nx = 0.75 * float(self._lock_nx) + 0.25 * nx_f
                self._lock_count += 1
            return self._gate_lock
        if self._lock_nx is not None and abs(nx_f - self._lock_nx) > max_jump:
            # Still acquiring — restart if the box jumps around.
            self._lock_count = 1
            self._lock_nx = nx_f
        else:
            self._lock_count += 1
            if body is not None:
                self._lock_count += 1  # pose confirms lock faster
            self._lock_nx = nx_f
        if self._lock_count >= need:
            self._gate_lock = True
            self._lock_ok_t = time.monotonic()
            self._last_yaw_cmd = 0.0
            print(
                f'[ASSIST] gate lock ok nx={nx_f:+.3f} frames={self._lock_count}',
                flush=True,
            )
        return self._gate_lock

    def _seek_look_ahead_pitch(self) -> float:
        """Nose-down so the fixed cam-up looks ahead at the next gate.

        Flat hover + 20° cam mount ⇒ sky. Tip more than cruise lean (~10°)
        so the readjustment is visible and the track fills the frame.
        """
        cam = float(
            getattr(
                config,
                'CAMERA_TILT_RAD',
                math.radians(cm.CAMERA_TILT_UP_DEG),
            )
        )
        frac = float(getattr(config, 'ASSIST_SEEK_CAM_LEVEL_FRAC', 0.80))
        max_seek = math.radians(
            float(getattr(config, 'ASSIST_SEEK_PITCH_MAX_DEG', 16.0))
        )
        pitch = self._fwd_sign * cam * max(0.0, frac)
        return float(np.clip(pitch, -max_seek, max_seek))

    def _seek_soft_pitch(self) -> float:
        """Cam tip + optional crawl lean after the first next-gate glimpse."""
        tip = self._seek_look_ahead_pitch()
        if not self._seek_seen:
            return tip
        crawl = self._fwd_sign * math.radians(
            float(getattr(config, 'ASSIST_SEEK_CRAWL_DEG', 5.5))
        )
        if self._fwd_sign >= 0.0:
            return float(max(tip, crawl))
        return float(min(tip, crawl))

    def _limit_forward_pitch_for_speed(
        self, des_pitch: float, v_fwd: float
    ) -> float:
        """Brake forward lean when over ASSIST_SPEED_CAP_MPS (all phases).

        Scale tip down only — never reverse-lean. 125233 reverse brake
        flipped tip↔level every tick (porpoise / shake).
        """
        v_cap = float(getattr(config, 'ASSIST_SPEED_CAP_MPS', 4.0))
        if v_cap <= 0.0:
            return float(des_pitch)
        v = max(0.0, float(v_fwd))
        if v <= v_cap:
            return float(des_pitch)
        # Soft scale with a tip floor so we never slam to reverse.
        scale = float(np.clip(v_cap / max(v, 0.1), 0.20, 1.0))
        limited = float(des_pitch) * scale
        # Keep a small same-sign crawl so the cam doesn't nod.
        crawl = self._fwd_sign * math.radians(
            float(getattr(config, 'ASSIST_SEEK_CRAWL_DEG', 5.5))
        )
        if self._fwd_sign >= 0.0:
            limited = max(limited, crawl * 0.5)
        else:
            limited = min(limited, crawl * 0.5)
        return float(
            np.clip(limited, -self._max_lean, self._max_lean)
        )

    def _slew_pitch(self, des_pitch: float, dt: float) -> float:
        """Rate-limit pitch cmd so tip can't flip every control tick."""
        max_dps = float(getattr(config, 'ASSIST_PITCH_SLEW_DEG', 60.0))
        max_step = math.radians(max_dps) * max(1e-3, float(dt))
        prev = getattr(self, '_des_pitch_f', None)
        if prev is None:
            self._des_pitch_f = float(des_pitch)
            return float(des_pitch)
        delta = float(np.clip(float(des_pitch) - float(prev), -max_step, max_step))
        self._des_pitch_f = float(prev) + delta
        return float(self._des_pitch_f)

    def _seek_forward_pitch(
        self, nx: float, locked: bool, v_fwd: float
    ) -> float:
        """Forward lean while seeking — keep cam leveled after a pass.

        Fixed cam is 20° up. Tip (~16°) cancels that so the view looks
        forward at the track. 110826: locked path dropped to crawl-only
        (~5°) so the camera still looked at sky after gate 1.
        Speed cap still brakes when too fast.
        """
        tip = self._seek_look_ahead_pitch()
        crawl = self._fwd_sign * math.radians(
            float(getattr(config, 'ASSIST_SEEK_CRAWL_DEG', 5.5))
        )
        brake_nx = float(getattr(config, 'ASSIST_ALIGN_BRAKE_NX', 0.12))
        align = float(
            np.clip(1.0 - abs(float(nx)) / max(brake_nx, 1e-3), 0.25, 1.0)
        )
        # Always keep most of the cam-level tip (never crawl-only).
        tip_keep = 0.80 if locked else 0.70
        pitch = tip * (tip_keep + (1.0 - tip_keep) * align)
        if locked and abs(float(nx)) < brake_nx:
            # Aligned: tip + crawl for closing range.
            if self._fwd_sign >= 0.0:
                pitch = max(pitch, crawl)
            else:
                pitch = min(pitch, crawl)
        pitch = self._limit_forward_pitch_for_speed(pitch, v_fwd)
        max_seek = math.radians(
            float(getattr(config, 'ASSIST_SEEK_PITCH_MAX_DEG', 16.0))
        )
        return float(np.clip(pitch, -max_seek, max_seek))

    def _seek_hold_thrust(self, hover: float) -> float:
        """Bleed collective while seeking unlocked — pitched tip + HT climbs."""
        bleed = float(getattr(config, 'ASSIST_SEEK_THRUST_BLEED', 0.014))
        return float(hover - max(0.0, bleed))

    def _seek_ny_thrust(
        self,
        base_thrust: float,
        hover: float,
        ny,
        ny_aim: float,
        climbed: float = 99.0,
        range_m=None,
        allow_climb: bool = True,
        pose_dz=None,
    ):
        """Seek altitude from image height: sink if gate low, rise if high.

        Stop sink from live gate pose/YOLO — not the pad floor (112750:
        pose_dz≈0 but tipped ny kept digging). Sink is range-gated for
        far low boxes. Climb only with a live sighting.
        """
        bleed = float(getattr(config, 'ASSIST_SEEK_THRUST_BLEED', 0.014))
        thrust = float(base_thrust) - max(0.0, bleed)
        if ny is None:
            return thrust, 'seek_hold'
        err = float(ny) - float(ny_aim)
        dead = float(getattr(config, 'ASSIST_SEEK_NY_DEAD', 0.10))
        if abs(err) < dead:
            return thrust, 'seek_hold'
        # Stale latch / hold: never climb — wait for live reacquire.
        if err < 0.0 and not allow_climb:
            return thrust, 'seek_hold'
        # Pose near level: stop tip dig near cruise/floor. Still allow a
        # controlled dig when we are clearly ABOVE cruise with tip ny
        # (123610: held at ~3 m then lofted over gate 2).
        if err > 0.0 and pose_dz is not None and math.isfinite(float(pose_dz)):
            pose_dead = float(
                getattr(config, 'ASSIST_SEEK_POSE_STOP_SINK_M', 0.25)
            )
            cruise_alt = float(
                getattr(config, 'ASSIST_SEEK_CRUISE_ALT_M', 1.55)
            )
            clearly_high = float(climbed) > cruise_alt + 0.45
            tip_low = err > float(
                getattr(config, 'ASSIST_SEEK_POSE_STOP_NY_ERR', 0.30)
            )
            if float(pose_dz) <= pose_dead and not (clearly_high and tip_low):
                return thrust, 'seek_hold'
        # No pose: image-proxy height (ny−aim)×range. Stop when small.
        if (
            err > 0.0
            and pose_dz is None
            and range_m is not None
            and float(range_m) > 0.5
        ):
            k = float(getattr(config, 'ASSIST_SEEK_NY_TO_DZ', 0.55))
            img_dz = float(err) * float(range_m) * k
            pose_dead = float(
                getattr(config, 'ASSIST_SEEK_POSE_STOP_SINK_M', 0.25)
            )
            if img_dz <= pose_dead:
                return thrust, 'seek_hold'
        # Close punch: tip exaggerates low — hold altitude and fly through
        # (112750: ny≈0.65 @ ~10 m kept digging into the pad).
        punch_r = float(getattr(config, 'ASSIST_SEEK_PUNCH_RANGE_M', 9.0))
        punch_ny = float(getattr(config, 'ASSIST_SEEK_PUNCH_NY_MAX', 0.80))
        if (
            err > 0.0
            and range_m is not None
            and float(range_m) <= punch_r
            and float(ny) < punch_ny
        ):
            return thrust, 'seek_hold'
        gain = float(getattr(config, 'ASSIST_SEEK_NY_THRUST_GAIN', 0.035))
        sink_cap = float(getattr(config, 'ASSIST_SEEK_NY_SINK_CAP', 0.035))
        climb_cap = float(getattr(config, 'ASSIST_SEEK_NY_CLIMB_CAP', 0.022))
        min_alt = float(getattr(config, 'ASSIST_SEEK_MIN_ALT_M', 0.55))
        floor_boost = float(
            getattr(config, 'ASSIST_SEEK_FLOOR_THRUST', 0.022)
        )
        if float(climbed) < min_alt and err > 0.0:
            return hover + floor_boost, 'seek_floor'
        # Far low box → no dig (gate-3 steal). Scale sink ∝ nearness.
        max_rng = float(getattr(config, 'ASSIST_SEEK_SINK_MAX_RANGE_M', 24.0))
        near_w = 1.0
        if err > 0.0 and range_m is not None and float(range_m) > 0.5:
            near_w = float(np.clip(max_rng / float(range_m), 0.0, 1.0))
            if float(range_m) > max_rng * 1.25:
                return thrust, 'seek_hold'
        raw = -gain * err
        if err > 0.0:
            lim = sink_cap * min(abs(err), 1.0) * near_w
            delta = float(np.clip(raw * near_w, -lim, 0.0))
        else:
            lim = climb_cap * min(abs(err), 1.0)
            delta = float(np.clip(raw, 0.0, lim))
        thrust = float(thrust + delta)
        vert = 'seek_sink' if err > 0.0 else 'seek_climb'
        return thrust, vert

    @staticmethod
    def _nx_extreme_yaw_boost(nx: float) -> tuple[float, float]:
        """Far L/R in frame → (kp_mult, max_yaw_mult). Near centre → (1, 1)."""
        abs_nx = abs(float(nx))
        start = float(getattr(config, 'ASSIST_YAW_EXTREME_NX', 0.20))
        if abs_nx <= start:
            return 1.0, 1.0
        t = float(
            np.clip((abs_nx - start) / max(1e-3, 1.0 - start), 0.0, 1.0)
        )
        t = t * t * (3.0 - 2.0 * t)  # smoothstep
        kp_hi = float(getattr(config, 'ASSIST_YAW_EXTREME_KP_MULT', 2.4))
        max_hi = float(getattr(config, 'ASSIST_YAW_EXTREME_MAX_MULT', 1.7))
        return 1.0 + (kp_hi - 1.0) * t, 1.0 + (max_hi - 1.0) * t

    def _seek_glimpse_yaw(
        self, nx: float, dt: float, *, live: bool = False, body=None
    ) -> float:
        """Seek left/right yaw toward the next gate (image nx + optional pose).

        Same role as seek sink/climb: correct off-center gates before chase.
        Far |nx| yaws harder (extreme boost).
        """
        half_fov = math.radians(
            0.5 * float(getattr(config, 'ASSIST_HFOV_DEG', 70.0))
        )
        yaw_img = float(nx) * half_fov
        yaw_pose = pose_bearing_yaw_rad(body)
        if yaw_pose is not None and abs(float(nx)) >= 0.03:
            w = float(getattr(config, 'ASSIST_SEEK_YAW_POSE_WEIGHT', 0.45))
            if yaw_img * float(yaw_pose) < 0.0:
                # Image wins on sign conflict (pose can be stale latch).
                w = 0.15
            yaw_err = w * float(yaw_pose) + (1.0 - w) * yaw_img
        else:
            yaw_err = yaw_img
        dead = float(getattr(config, 'ASSIST_YAW_ALIGN_DEAD_RAD', 0.035)) * 0.6
        if abs(yaw_err) < dead and abs(float(nx)) < 0.025:
            yaw_rate = 0.15 * self._last_yaw_cmd
            self._last_yaw_cmd = yaw_rate
            return yaw_rate
        if live:
            kp = float(getattr(config, 'ASSIST_SEEK_LIVE_YAW_KP', 1.90))
            max_yaw = math.radians(
                float(getattr(config, 'ASSIST_SEEK_LIVE_YAW_MAX_DEG', 42.0))
            )
        else:
            kp = float(getattr(config, 'ASSIST_SEEK_YAW_KP', 1.25))
            max_yaw = math.radians(
                float(getattr(config, 'ASSIST_SEEK_YAW_MAX_DEG', 28.0))
            )
        kp_m, max_m = self._nx_extreme_yaw_boost(nx)
        kp *= kp_m
        max_yaw = min(float(self._max_yaw), max_yaw * max_m)
        # Far L/R: bang-bang saturate (132028 proportional only ~20°/s on
        # nx=+0.51 while left ghosts stole the turn).
        bang_nx = float(getattr(config, 'ASSIST_YAW_BANG_NX', 0.28))
        if abs(float(nx)) >= bang_nx:
            sign = 1.0 if float(yaw_err) >= 0.0 else -1.0
            if abs(float(yaw_err)) < 1e-6:
                sign = 1.0 if float(nx) >= 0.0 else -1.0
            yaw_rate = sign * max_yaw
        else:
            yaw_rate = float(np.clip(kp * yaw_err, -max_yaw, max_yaw))
        # Faster slew when far off-centre so the boost is usable in-flight.
        base_slew = 220.0 if live else 180.0
        max_step = math.radians(base_slew * max(1.0, 0.55 + 0.45 * kp_m)) * max(
            1e-3, float(dt)
        )
        yaw_rate = float(
            np.clip(
                yaw_rate,
                self._last_yaw_cmd - max_step,
                self._last_yaw_cmd + max_step,
            )
        )
        self._last_yaw_cmd = yaw_rate
        return yaw_rate

    def _yaw_from_pose_and_image(
        self, nx: float, body, dt: float, range_m=None, soft_start: bool = False
    ) -> float:
        """Image-led yaw — mild near centre, hard when far L/R in frame.

        Pose only fills in when the gate is near image-centre.
        """
        half_fov = math.radians(
            0.5 * float(getattr(config, 'ASSIST_HFOV_DEG', 70.0))
        )
        yaw_img = float(nx) * half_fov
        yaw_pose = pose_bearing_yaw_rad(body)
        # Image owns yaw once visibly off-centre; pose only fills near centre.
        if yaw_pose is not None and abs(float(nx)) < 0.06:
            w = float(getattr(config, 'ASSIST_YAW_POSE_WEIGHT', 0.25))
            if range_m is not None and float(range_m) < 12.0:
                w = min(w, 0.15)
            if yaw_img * float(yaw_pose) < 0.0:
                w = 0.0
            w = float(np.clip(w, 0.0, 1.0))
            yaw_err = w * float(yaw_pose) + (1.0 - w) * yaw_img
        else:
            yaw_err = yaw_img

        mag = abs(float(yaw_err))
        dead = float(getattr(config, 'ASSIST_YAW_ALIGN_DEAD_RAD', 0.035))
        # Tiny image offset alone → hold; pose fill can still command a nudge.
        if mag < dead:
            yaw_rate = 0.20 * self._last_yaw_cmd
            self._last_yaw_cmd = yaw_rate
            return yaw_rate
        if abs(float(nx)) < 0.05 and (
            yaw_pose is None or abs(float(yaw_pose)) < dead
        ):
            yaw_rate = 0.20 * self._last_yaw_cmd
            self._last_yaw_cmd = yaw_rate
            return yaw_rate

        kp = float(getattr(config, 'ASSIST_KP_YAW_FINE', 1.35))
        coarse = float(getattr(config, 'ASSIST_YAW_COARSE_RAD', 0.16))
        if mag >= coarse:
            kp = float(getattr(config, 'ASSIST_KP_YAW_COARSE', 2.0))
        kp_m, _max_m = self._nx_extreme_yaw_boost(nx)
        kp *= kp_m
        bang_nx = float(getattr(config, 'ASSIST_YAW_BANG_NX', 0.28))
        if abs(float(nx)) >= bang_nx:
            sign = 1.0 if float(yaw_err) >= 0.0 else -1.0
            if abs(float(yaw_err)) < 1e-6:
                sign = 1.0 if float(nx) >= 0.0 else -1.0
            yaw_rate = sign * float(self._max_yaw)
        else:
            yaw_rate = float(
                np.clip(kp * yaw_err, -self._max_yaw, self._max_yaw)
            )

        max_step = self._yaw_slew * dt * max(1.0, 0.55 + 0.45 * kp_m)
        if soft_start and self._lock_ok_t is not None:
            age = time.monotonic() - float(self._lock_ok_t)
            ramp = float(np.clip(age / 0.60, 0.0, 1.0))
            yaw_rate *= ramp
            max_step *= 0.25 + 0.75 * ramp
        yaw_rate = float(
            np.clip(
                yaw_rate,
                self._last_yaw_cmd - max_step,
                self._last_yaw_cmd + max_step,
            )
        )
        self._last_yaw_cmd = yaw_rate
        return yaw_rate

    def _climb_m(self, shared_data) -> float:
        if self._arm_z is None:
            return 0.0
        climbs = []
        for key in ('local_position_ned', 'position_ned'):
            z = (shared_data.get(key) or {}).get('z')
            if z is None:
                continue
            try:
                c = float(self._arm_z) - float(z)
            except (TypeError, ValueError):
                continue
            if -1.0 <= c <= 20.0:
                climbs.append(c)
        return max(climbs) if climbs else 0.0

    def _climb_filtered(self, shared_data) -> float:
        """Low-pass climb; reject NED spikes that abort pad_lift (assist_d)."""
        raw = self._climb_m(shared_data)
        if self._climb_f is None:
            self._climb_f = raw
            return raw
        # One-tick jumps of ~1 m are odometry glitches, not flight.
        if abs(raw - self._climb_f) > 0.45:
            raw = self._climb_f + (0.45 if raw > self._climb_f else -0.45)
        self._climb_f = 0.85 * self._climb_f + 0.15 * raw
        return float(self._climb_f)

    def _update_climb_rate(self, climbed: float, now: float) -> None:
        """Track vertical speed (m/s, + = up) for descent braking."""
        if self._climb_rate_t is not None and self._climb_rate_z is not None:
            dt = max(1e-3, float(now) - float(self._climb_rate_t))
            inst = (float(climbed) - float(self._climb_rate_z)) / dt
            self._climb_rate = 0.65 * float(self._climb_rate) + 0.35 * inst
        else:
            self._climb_rate = 0.0
        self._climb_rate_z = float(climbed)
        self._climb_rate_t = float(now)

    def _brake_descent(
        self,
        thrust: float,
        hover: float,
        vert_src: str,
        *,
        ny=None,
        ny_aim=None,
        climbed=None,
    ):
        """Arrest fast sink near gate height — not while still digging.

        114816: thr≈0.23 at vz≈−1.1 m/s punched the pad (need brake).
        115153: brake at climb≈2 m with ny≈0.62 raised thr to hover and
        we never got low enough for gate 2 — only brake hard when YOLO is
        near aim or altitude is already low.
        """
        descent = -float(self._climb_rate)  # + when falling
        start = float(getattr(config, 'ASSIST_SEEK_DESCENT_START_MPS', 0.70))
        full = float(getattr(config, 'ASSIST_SEEK_DESCENT_FULL_MPS', 1.20))
        if descent < start:
            return float(thrust), vert_src
        w = float(
            np.clip(
                (descent - start) / max(1e-3, full - start),
                0.0,
                1.0,
            )
        )
        still_low = False
        if ny is not None and ny_aim is not None:
            still_low = (float(ny) - float(ny_aim)) > float(
                getattr(config, 'ASSIST_SEEK_POSE_STOP_NY_ERR', 0.30)
            )
        low_enough = (
            climbed is not None
            and float(climbed)
            <= float(getattr(config, 'ASSIST_SEEK_BRAKE_ALT_M', 1.15))
        )
        # Still high + gate still low in frame: only rate-limit extreme dig,
        # do not cancel sink to hover (115153).
        if still_low and not low_enough:
            floor_thr = float(hover) - float(
                getattr(config, 'ASSIST_SEEK_DESCENT_MIN_SINK', 0.028)
            )
            if descent >= full and float(thrust) < floor_thr:
                return max(float(thrust), floor_thr), vert_src
            return float(thrust), vert_src
        boost = float(getattr(config, 'ASSIST_SEEK_DESCENT_BRAKE_THRUST', 0.014))
        target = float(hover) + boost * w
        out = float(thrust) + (target - float(thrust)) * max(w, 0.55)
        if w >= 0.50:
            out = max(out, float(hover))
        if w >= 0.85:
            out = max(out, float(hover) + 0.5 * boost)
        if w >= 0.35:
            return out, 'seek_descent_brake'
        return out, vert_src

    def _note_pass(self, shared_data, now: float) -> None:
        race = shared_data.get('race_status') or {}
        ag = race.get('active_gate')
        try:
            ag_i = int(ag) if ag is not None else None
        except (TypeError, ValueError):
            ag_i = None
        if ag_i is None:
            return
        if self._active_gate is not None and ag_i > self._active_gate:
            prev_pass = self._pass_t
            self._pass_t = now
            self._coast_until = now + float(
                getattr(config, 'ASSIST_COAST_S', 1.5)
            )
            self._seek_until = now + float(
                getattr(config, 'ASSIST_SEEK_S', 14.0)
            )
            self._yaw_pid.reset()
            self._last_yaw_cmd = 0.0
            self._last_area_px = None
            self._peak_climbed = 0.0
            self._reset_gate_lock()
            # Prefer approach snapshot (next-next) over a latch that aged
            # out while dual dropped during the slot (115959).
            # 120804: reject near residual (~8.7 m) — not a real next gate.
            snap_age = (
                None
                if self._snap_next_t is None
                else (now - float(self._snap_next_t))
            )
            snap_max = float(
                getattr(config, 'ASSIST_LATCH_SNAP_MAX_AGE_S', 15.0)
            )
            snap_ok = (
                self._snap_next_nx is not None
                and self._snap_next_ny is not None
                and snap_age is not None
                and snap_age < snap_max
                and self._latch_ahead_ok(
                    self._snap_next_rng, self._snap_next_ny
                )
            )
            if snap_ok:
                self._next_nx = float(self._snap_next_nx)
                self._next_ny = float(self._snap_next_ny)
                self._next_rng = self._snap_next_rng
                self._next_body = (
                    None
                    if self._snap_next_body is None
                    else self._snap_next_body.copy()
                )
                self._next_t = now
                self._next_live_t = now
                self._next_course_mem = False
            # One-shot: latch must be a live refresh after the previous pass
            # (or this snap) — do not reuse the aim seeded on the last gate.
            live_fresh = (
                self._next_live_t is not None
                and (
                    prev_pass is None
                    or float(self._next_live_t) > float(prev_pass) + 0.05
                )
            )
            latch_age = (
                None if self._next_t is None else (now - float(self._next_t))
            )
            latch_ok = (
                self._next_nx is not None
                and self._next_ny is not None
                and latch_age is not None
                and latch_age < 6.0
                and self._latch_ahead_ok(self._next_rng, self._next_ny)
                and live_fresh
            )
            seed_src = 'snap' if snap_ok else ('latch' if latch_ok else 'none')
            # Past g1→g2: never keep a memorized right yaw.
            if int(ag_i) > 1:
                self._course_mem = False
                self._course_mem_yaw_tgt = None
                self._course_mem_spent = True
            if latch_ok:
                self._have_filt = True
                self._nx_f = float(self._next_nx)
                self._ny_f = float(self._next_ny)
                self._last_see_t = now
                self._seek_seen = True
                self._body_f = (
                    None
                    if self._next_body is None
                    else self._next_body.copy()
                )
                self._last_range_m = self._next_rng
                # Live next-gate after g1 — memory not needed / not allowed later.
                if int(ag_i) == 1:
                    self._course_mem = False
                    self._course_mem_yaw_tgt = None
                    self._course_mem_spent = True
                print(
                    f'[ASSIST] GATE_PASSED → seed next-gate {seed_src} '
                    f'nx={self._nx_f:+.3f} ny={self._ny_f:+.3f} '
                    f'rng={self._next_rng}',
                    flush=True,
                )
            else:
                # Keep far-right glimpse only to scale the one-shot g1→g2 yaw.
                hint_nx = None
                if int(ag_i) == 1 and not getattr(
                    self, '_course_mem_spent', False
                ):
                    for cand in (self._next_nx, self._snap_next_nx):
                        if cand is not None and math.isfinite(float(cand)):
                            hint_nx = float(cand)
                            break
                # Drop near/stale poison so the next cycle cannot reuse the
                # just-cleared gate as aim (120804 g1→g2 same 8.7 m).
                self._clear_next_latch()
                self._have_filt = False
                self._body_f = None
                self._last_range_m = None
                self._seek_seen = False
                self._course_mem = False
                att = shared_data.get('attitude') or {}
                yaw_now = float(att.get('yaw', 0.0) or 0.0)
                if not math.isfinite(yaw_now):
                    yaw_now = 0.0
                # Course memory: ONLY g1→g2 on course 2 (one-shot).
                if self._seed_course_memory(
                    now, ag_i, yaw_now, hint_nx=hint_nx
                ):
                    self._have_filt = True
                    self._nx_f = float(self._next_nx)
                    self._ny_f = float(self._next_ny)
                    self._last_see_t = now
                    self._seek_seen = True
                    self._last_range_m = self._next_rng
                    seed_src = 'course_mem'
                    print(
                        f'[ASSIST] GATE_PASSED → seed course memory '
                        f'(g1→g2 only) nx={self._nx_f:+.3f} '
                        f'ny={self._ny_f:+.3f} rng={self._next_rng}',
                        flush=True,
                    )
                else:
                    print(
                        f'[ASSIST] GATE_PASSED → coast/seek gate={ag_i} '
                        f'(no next-gate latch)',
                        flush=True,
                    )
            self._clear_snap_latch()
            if self._arm_z is not None:
                climbed = self._climb_m(shared_data)
                self._peak_climbed = max(0.0, climbed)
            log = shared_data.get('log_event')
            if log:
                log(
                    'ASSIST_PASS',
                    f'active_gate={ag_i} '
                    f'latch={seed_src}',
                )
        self._active_gate = ag_i

    def _chaseable(self, nx, ny, area_px, seeking: bool) -> bool:
        if nx is None or ny is None:
            return False
        ny_r = float(ny)
        if area_px is not None and area_px > 90000.0:
            return False
        # Seeking: next gate often sits low/small (102028 ny≈0.77 wiped).
        # Still reject true floor-band junk above ASSIST_SEEK_NY_MAX.
        if seeking:
            ny_max = float(getattr(config, 'ASSIST_SEEK_NY_MAX', 0.92))
            # Post-g1: tighter floor reject — 133934 latched ny≈0.86.
            if (
                getattr(self, '_course_mem_spent', False)
                and int(self._active_gate or 0) == 1
            ):
                ny_max = min(
                    ny_max,
                    float(getattr(config, 'ASSIST_POST_G1_LIVE_NY_MAX', 0.62)),
                )
            min_area = float(getattr(config, 'ASSIST_SEEK_MIN_AREA', 180.0))
            if ny_r > ny_max:
                return False
            if area_px is not None and area_px < min_area:
                return False
            return True
        if ny_r > 0.95:
            return False
        return True

    def compute_target(self, shared_data):
        shared_data['planner_mode'] = self.name
        shared_data['post_pass_hunt'] = False
        now = time.monotonic()
        dt = 0.02 if self._last_t is None else max(1e-3, now - self._last_t)
        self._last_t = now

        attitude = shared_data.get('attitude') or {}
        roll = float(attitude.get('roll', 0.0) or 0.0)
        pitch = float(attitude.get('pitch', 0.0) or 0.0)
        yaw = float(attitude.get('yaw', 0.0) or 0.0)
        if not math.isfinite(yaw):
            yaw = 0.0

        if self._arm_z is None:
            for key in ('local_position_ned', 'position_ned'):
                z = (shared_data.get(key) or {}).get('z')
                if z is not None and math.isfinite(float(z)):
                    self._arm_z = float(z)
                    break
            if self._arm_z is None:
                self._arm_z = 0.0

        self._update_next_latch(shared_data, now)

        nx_raw, ny_raw, src = image_gate_norm(shared_data)
        det = shared_data.get('gate_detection') or {}
        area_px = _f(det.get('area_px')) if isinstance(det, dict) else None
        dual = shared_data.get('dual_gate_pnp') or {}
        dual_rng = _f(dual.get('gate1_range_m'))
        max_ahead = float(getattr(config, 'ASSIST_SEEK_MAX_AHEAD_M', 28.0))
        range_m = dual_rng
        if range_m is None and area_px is not None and area_px > 50.0:
            range_m = float((320.0 * 1.5) / math.sqrt(area_px))
        # Reject identity flips (assist_a: 8 m → 25 m in one tick).
        # Do NOT demote a far dual (> two gates) into a near area_range —
        # that let 44 m look like ~21 m and steal the latch (112030/111515).
        far_dual = bool(dual_rng is not None and float(dual_rng) > max_ahead)
        if (
            not far_dual
            and range_m is not None
            and self._last_range_m is not None
            and abs(range_m - self._last_range_m) > 8.0
            and area_px is not None
            and area_px > 50.0
        ):
            range_m = float((320.0 * 1.5) / math.sqrt(area_px))
            src = 'area_range'
        if range_m is not None and not far_dual:
            self._last_range_m = range_m

        # Snapshot next-next before pass edge — then consume on GATE_PASSED.
        pre_seeking = now < self._seek_until
        pre_coasting = now < self._coast_until
        self._maybe_freeze_approach_latch(
            range_m,
            area_px,
            seeking=pre_seeking,
            coasting=pre_coasting,
            now=now,
        )
        self._note_pass(shared_data, now)

        # Visual commit before race_status updates (assist_a lost YOLO at
        # ~9 m with bbox exploding / ny ← −0.6, then dual_pnp stole ID).
        # Require real altitude — assist_j committed while climb≈−0.4 on floor.
        # 092927: require tight lateral align; keep aim through coast (do NOT
        # clear filt / YOLO lock here — that flew blind into gate-2 left edge).
        # Next-gate acquire is vision_rx on active_gate++ / GATE_PASSED.
        climb_for_commit = (
            self._climb_f if self._climb_f is not None else self._climb_m(shared_data)
        )
        commit_nx_max = float(
            getattr(config, 'ASSIST_COMMIT_NX_MAX', 0.12)
        )
        area_rng = None
        if area_px is not None and area_px > 50.0:
            area_rng = float((320.0 * 1.5) / math.sqrt(area_px))
        # Prefer bbox-implied range — dual range can stick stale (094154: 18 m).
        close_for_commit = (
            (area_rng is not None and area_rng < 9.0 and area_px > 5000.0)
            or (
                range_m is not None
                and range_m < 8.0
                and (area_rng is None or area_rng < 12.0)
            )
            or (
                area_px is not None
                and area_px > 4500.0
                and ny_raw is not None
                and float(ny_raw) < -0.40
                and area_rng is not None
                and area_rng < 10.0
            )
        )
        if (
            now >= self._coast_until
            and self._left_pad
            and climb_for_commit > 0.45
            and nx_raw is not None
            and abs(float(nx_raw)) < commit_nx_max
            and close_for_commit
        ):
            self._coast_until = now + float(
                getattr(config, 'ASSIST_COAST_S', 1.5)
            )
            self._seek_until = max(
                self._seek_until,
                now + float(getattr(config, 'ASSIST_SEEK_S', 14.0)),
            )
            print('[ASSIST] VISUAL_COMMIT → coast', flush=True)
            log = shared_data.get('log_event')
            if log:
                log(
                    'ASSIST_COMMIT',
                    f'area={area_px} range={range_m} '
                    f'area_r={area_rng} nx={float(nx_raw):+.3f}',
                )

        coasting = now < self._coast_until
        seeking = now < self._seek_until
        # Empty primary image: live gate2 first, else latched pre-pass gate2.
        hx, hy, hsrc, hrng = next_gate_hint(shared_data)
        if (coasting or seeking) and nx_raw is None:
            if hx is not None:
                nx_raw, ny_raw, src = hx, hy, hsrc
                if range_m is None:
                    range_m = hrng
            elif self._latch_still_valid(
                now,
                max_age=float(getattr(config, 'ASSIST_LATCH_HOLD_S', 12.0)),
            ):
                nx_raw = float(self._next_nx)
                ny_raw = float(self._next_ny)
                src = 'next_latch'
                if range_m is None:
                    range_m = self._next_rng
        shared_data['post_pass_hunt'] = bool(seeking and not coasting)

        chaseable = self._chaseable(nx_raw, ny_raw, area_px, seeking)
        # assist_g: area collapsed while range jumped 18→34 (far-gate steal).
        identity_steal = bool(
            chaseable
            and not seeking
            and not coasting
            and area_px is not None
            and self._last_area_px is not None
            and area_px < 0.70 * self._last_area_px
            and self._last_range_m is not None
            and range_m is not None
            and range_m > self._last_range_m + 3.0
        )
        # Seek look-ahead: up to TWO gates (ASSIST_SEEK_MAX_AHEAD_M).
        # 112030: rejecting 18–22 m live as "jump past latch" kept a bad
        # ny=-0.22 latch and lofted to 14 m. Only reject *beyond* two gates.
        # Sink stay tighter via ASSIST_SEEK_SINK_MAX_RANGE_M (not steal).
        beyond_two = bool(
            far_dual
            or (range_m is not None and float(range_m) > max_ahead)
        )
        # 115959: post-pass small off-center box with no range (~31 m left
        # ghost) stole yaw — treat as far when range unknown.
        post_pass_recent = (
            self._pass_t is not None
            and (now - float(self._pass_t))
            < float(getattr(config, 'ASSIST_LATCH_FREEZE_S', 5.0))
        )
        suspect_far_norange = bool(
            post_pass_recent
            and chaseable
            and range_m is None
            and area_px is not None
            and float(area_px) < 900.0
            and nx_raw is not None
            and abs(float(nx_raw)) > 0.15
        )
        far_next_steal = bool(
            (seeking or coasting)
            and chaseable
            and (beyond_two or suspect_far_norange)
        )
        # Latch/hold fills are not live — 112030 treated next_latch as a
        # glimpse and climbed on ny=-0.22.
        live_src = str(src) not in ('next_latch', 'hold_id', 'none', '')
        live_glimpse = bool(
            chaseable
            and not identity_steal
            and not far_next_steal
            and live_src
        )
        # 132028/132343/132813: while course_mem heading unfinished, do not
        # treat live boxes as chase handoff (right glimpse cleared mem in 0.4s
        # and killed the turn; left ghosts spun). After turn done (133934):
        # accept near-center / slight-left live g2; reject floor + far-left.
        post_g1_hunt = (
            getattr(self, '_course_mem_spent', False)
            and int(self._active_gate or 0) == 1
        )
        if (
            live_glimpse
            and post_g1_hunt
            and not self._course_mem_heading_done(yaw, now=now)
        ):
            live_glimpse = False
        elif live_glimpse and post_g1_hunt and nx_raw is not None:
            left_lim = float(
                getattr(config, 'ASSIST_POST_G1_LIVE_NX_LEFT', -0.18)
            )
            ny_max = float(
                getattr(config, 'ASSIST_POST_G1_LIVE_NY_MAX', 0.62)
            )
            min_area = float(
                getattr(config, 'ASSIST_POST_G1_LIVE_AREA_MIN', 700.0)
            )
            area_ok = area_px is None or float(area_px) >= min_area
            ny_ok = ny_raw is not None and float(ny_raw) <= ny_max
            if (
                (not area_ok)
                or (not ny_ok)
                or float(nx_raw) < left_lim
            ):
                live_glimpse = False
        ghost_s = float(getattr(config, 'ASSIST_SEEK_GHOST_S', 1.20))
        ghost_hold = False
        if live_glimpse:
            self._last_see_t = now
            if seeking:
                self._seek_seen = True
            # Live g2 replaces synthetic course-mem latch.
            if self._course_mem and str(src) not in ('next_latch', 'hold_id'):
                if self._course_mem_heading_done(yaw, now=now):
                    self._course_mem = False
                    if bool(getattr(self, '_next_course_mem', False)):
                        self._next_course_mem = False
            # Snap on large vertical jump (bad sky latch → real low gate).
            snap = (
                self._have_filt
                and abs(float(ny_raw) - float(self._ny_f)) > 0.35
            )
            if not self._have_filt or snap:
                self._nx_f = float(nx_raw)
                self._ny_f = float(ny_raw)
                self._have_filt = True
            else:
                self._nx_f = 0.65 * self._nx_f + 0.35 * float(nx_raw)
                self._ny_f = 0.65 * self._ny_f + 0.35 * float(ny_raw)
            if area_px is not None:
                self._last_area_px = float(area_px)
            if range_m is not None:
                self._last_range_m = float(range_m)
            # Promote live g2 into the latch (real, not synthetic).
            if post_g1_hunt and str(src) not in ('next_latch', 'hold_id'):
                self._next_nx = float(nx_raw)
                self._next_ny = float(ny_raw)
                self._next_t = now
                self._next_live_t = now
                self._next_course_mem = False
                if range_m is not None:
                    self._next_rng = float(range_m)
        elif chaseable and (identity_steal or far_next_steal):
            # Keep last filter / latch; do not adopt far gate-3 box.
            self._last_see_t = now
            if far_next_steal and self._next_nx is not None:
                self._have_filt = True
                self._nx_f = float(self._next_nx)
                self._ny_f = float(
                    self._next_ny if self._next_ny is not None else self._ny_f
                )
                if self._next_body is not None:
                    self._body_f = self._next_body.copy()
                if self._next_rng is not None:
                    range_m = float(self._next_rng)
                src = 'next_latch'
            elif range_m is not None and self._last_range_m is not None:
                range_m = self._last_range_m
                src = 'hold_id'
            else:
                src = 'hold_id'
        elif seeking and self._have_filt:
            # Soft-hold last aim briefly (102028), then fall back to latch.
            # Drop floor-band filt after g1 (133934 ny≈0.86 scrape).
            post_ny_max = float(
                getattr(config, 'ASSIST_POST_G1_LIVE_NY_MAX', 0.62)
            )
            if (
                post_g1_hunt
                and self._course_mem_heading_done(yaw, now=now)
                and float(self._ny_f) > post_ny_max
            ):
                self._have_filt = False
                self._nx_f = 0.0
                self._ny_f = 0.0
                self._last_yaw_cmd = 0.0
                self._body_f = None
                if (
                    self._next_ny is not None
                    and float(self._next_ny) > post_ny_max
                ):
                    self._clear_next_latch()
            elif (now - self._last_see_t) <= ghost_s:
                ghost_hold = True
            elif self._latch_still_valid(
                now,
                max_age=float(getattr(config, 'ASSIST_LATCH_HOLD_S', 12.0)),
            ):
                # 122209: do not replace a just-lost LOW dig with a HIGH
                # residual latch (ny≈−0.21) — clear and scan instead.
                ny_aim_g = float(
                    getattr(config, 'ASSIST_NY_AIM', config.KALMAN_NY_AIM)
                )
                had_low = float(self._ny_f) > ny_aim_g + 0.20
                latch_high = float(self._next_ny) < ny_aim_g - 0.15
                latch_floor = (
                    post_g1_hunt
                    and float(self._next_ny) > post_ny_max
                )
                if had_low and latch_high:
                    # Drop poison so the empty-filt reinject path cannot
                    # re-seed it next tick.
                    self._clear_next_latch()
                    self._have_filt = False
                    self._nx_f = 0.0
                    self._ny_f = 0.0
                    self._last_yaw_cmd = 0.0
                    self._body_f = None
                elif latch_floor or (
                    post_g1_hunt
                    and bool(getattr(self, '_next_course_mem', False))
                    and self._course_mem_heading_done(yaw, now=now)
                ):
                    # No synthetic/floor reinject after the turn finishes.
                    if latch_floor or getattr(self, '_next_course_mem', False):
                        self._clear_next_latch()
                    self._have_filt = False
                    self._nx_f = 0.0
                    self._ny_f = 0.0
                    self._last_yaw_cmd = 0.0
                    self._body_f = None
                else:
                    self._have_filt = True
                    self._nx_f = float(self._next_nx)
                    self._ny_f = float(self._next_ny)
                    self._body_f = (
                        None
                        if self._next_body is None
                        else self._next_body.copy()
                    )
                    if self._next_rng is not None:
                        self._last_range_m = float(self._next_rng)
                        range_m = float(self._next_rng)
                    src = 'next_latch'
                    ghost_hold = True
                    self._last_see_t = now
            else:
                self._have_filt = False
                self._nx_f = 0.0
                self._ny_f = 0.0
                self._last_yaw_cmd = 0.0
                self._body_f = None
        elif seeking and not self._have_filt and self._latch_still_valid(
            now,
            max_age=float(getattr(config, 'ASSIST_LATCH_HOLD_S', 12.0)),
        ):
            post_ny_max = float(
                getattr(config, 'ASSIST_POST_G1_LIVE_NY_MAX', 0.62)
            )
            # After turn done: do not reinject synthetic/floor course-mem aim.
            if (
                post_g1_hunt
                and self._course_mem_heading_done(yaw, now=now)
                and (
                    bool(getattr(self, '_next_course_mem', False))
                    or float(self._next_ny) > post_ny_max
                )
            ):
                self._clear_next_latch()
            else:
                # Ghost already cleared — reinject latch so seek_lock is not mute.
                self._have_filt = True
                self._nx_f = float(self._next_nx)
                self._ny_f = float(self._next_ny)
                self._body_f = (
                    None if self._next_body is None else self._next_body.copy()
                )
                if self._next_rng is not None:
                    self._last_range_m = float(self._next_rng)
                    range_m = float(self._next_rng)
                src = 'next_latch'
                ghost_hold = True
                self._last_see_t = now
        lost = (now - self._last_see_t) > float(
            getattr(config, 'ASSIST_LOST_TIMEOUT_S', 0.8)
        )

        hover = float(config.HOVER_THRUST)
        ny_aim = float(getattr(config, 'ASSIST_NY_AIM', config.KALMAN_NY_AIM))
        climbed_raw = self._climb_m(shared_data)
        climbed = self._climb_filtered(shared_data)
        self._update_climb_rate(climbed, now)
        self._peak_climbed = max(self._peak_climbed, climbed)
        # Brake / pad-exit on filtered climb (assist_d: raw NED jumped 1 m/tick).
        alt = climbed
        if self._lift_start_t is None and shared_data.get('flight_started'):
            self._lift_start_t = now
        lift_age = (
            0.0 if self._lift_start_t is None else (now - self._lift_start_t)
        )
        # Leave pad sooner so we lean before lofting past the gate (024550).
        airborne = alt >= 0.40 and lift_age >= 0.70
        if airborne:
            self._left_pad = True
            if self._airborne_t is None:
                self._airborne_t = now
        # assist_e: after a floor bounce, alt dropped and pad_lift locked
        # pitch=0 forever. Pad lift is one-shot per episode.
        use_pad_lift = (not self._left_pad) and (not airborne)

        # Post-pass floor-speck reject — same floor as _chaseable (was hard 400
        # and wiped real gate-2 boxes ~250 px in 102028).
        min_seek_area = float(getattr(config, 'ASSIST_SEEK_MIN_AREA', 180.0))
        if seeking and chaseable and area_px is not None and area_px < min_seek_area:
            chaseable = False

        # Only abort coast toward a far next gate *after* a real pass.
        # Visual-commit coast must finish the slot (094441 still aborted early).
        if coasting and chaseable and not identity_steal:
            recently_passed = (
                self._pass_t is not None and (now - self._pass_t) < 8.0
            )
            big_box = area_px is not None and area_px > 3500.0
            far_next = (
                range_m is not None
                and range_m > 14.0
                and (area_px is None or area_px < 3000.0)
                and (area_rng is None or area_rng > 10.0)
            )
            # Don't abort coast into a floor-band dual_pnp ghost (100134).
            low_junk = ny_raw is not None and float(ny_raw) > 0.70
            if recently_passed and far_next and not big_box and not low_junk:
                self._coast_until = now
                coasting = False
                seeking = now < self._seek_until
                shared_data['post_pass_hunt'] = bool(seeking)
                print(
                    f'[ASSIST] coast→seek_chase src={src} '
                    f'nx={nx_raw} ny={ny_raw} r={range_m}',
                    flush=True,
                )

        # Fresh PnP body (preferred identity via vision_rx).
        # Reject far-gate body while seeking (110826 gate-3 steal).
        body_raw = gate1_body_m(shared_data)
        if (
            body_raw is not None
            and chaseable
            and not identity_steal
            and not far_next_steal
        ):
            # Drop stale body if it fights the live image lateral (094154).
            if (
                nx_raw is not None
                and abs(float(nx_raw)) > 0.04
                and self._body_f is not None
            ):
                prev_yaw = pose_bearing_yaw_rad(self._body_f)
                if (
                    prev_yaw is not None
                    and prev_yaw * float(nx_raw) < 0.0
                ):
                    self._body_f = None
            if self._body_f is None:
                self._body_f = body_raw.copy()
            else:
                jump = float(np.linalg.norm(body_raw - self._body_f))
                a = 0.45 if jump > 6.0 else 0.30
                self._body_f = (1.0 - a) * self._body_f + a * body_raw
        elif (
            far_next_steal
            and self._next_body is not None
            and seeking
        ):
            self._body_f = self._next_body.copy()
        body = self._body_f

        phase = 'hover'
        pose_dz = None  # NED-down gate height vs us (+ = gate lower)
        pose_dz_raw = None
        tilt_bias = 0.0
        nx_roll_bias = 0.0
        v_fwd = forward_speed_mps(
            shared_data, yaw, pitch, self._max_lean
        )
        v_lat = lateral_speed_mps(
            shared_data, yaw, roll, self._max_lean
        )
        nx_roll_bias = cam_bank_lateral_bias_nx(
            roll, v_lat, v_fwd, self._max_lean
        )
        # Bias aim left in image (−) ⇒ path right through the hole (104123).
        # Image nx aim (trim) + pose aim_y (∝ ey/range) — both left-bias path.
        nx_aim_lat = float(getattr(config, 'ASSIST_NX_AIM', 0.03))
        pose_aim_y = pose_aim_y_m()
        # Build/refresh lock while seeking (before any yaw adjust).
        locked = self._update_gate_lock(
            chaseable and self._have_filt,
            self._nx_f if self._have_filt else None,
            body,
            seeking,
        )

        if coasting:
            phase = 'coast'
            lean_scale = float(
                getattr(
                    config,
                    'KALMAN_BODY_Y_LEAN_SCALE',
                    getattr(config, 'ASSIST_ROLL_SCALE', 0.45),
                )
            )
            recently_passed = (
                self._pass_t is not None and (now - self._pass_t) < 8.0
            )
            # After a pass before next-gate lock: tip/crawl + soft yaw on
            # live or ghost-held box (no full cruise).
            if recently_passed and not locked and self._have_filt and (
                live_glimpse or ghost_hold
            ):
                nx = float(np.clip(self._nx_f, -1.2, 1.2))
                ny = float(self._ny_f) if self._have_filt else ny_aim
                nx_cmd = float(
                    np.clip(nx + nx_roll_bias - nx_aim_lat, -1.2, 1.2)
                )
                yaw_rate = self._seek_glimpse_yaw(
                    nx_cmd,
                    dt,
                    live=bool(live_glimpse),
                    body=body,
                )
                des_roll = 0.0
                des_pitch = self._seek_forward_pitch(nx_cmd, False, v_fwd)
                # Sink toward next gate during post-pass coast (not hover).
                # Latch-only: no climb (112030 lofted on ny=-0.22).
                thrust, vert_src = self._seek_ny_thrust(
                    hover,
                    hover,
                    ny,
                    ny_aim,
                    climbed=float(climbed),
                    range_m=range_m if range_m is not None else self._next_rng,
                    allow_climb=bool(live_glimpse),
                )
                if vert_src == 'seek_hold':
                    vert_src = 'coast_yaw'
            elif recently_passed and not locked:
                nx = 0.0
                ny = ny_aim
                yaw_rate = 0.0
                self._last_yaw_cmd = 0.0
                des_roll = 0.0
                des_pitch = self._seek_forward_pitch(0.0, False, v_fwd)
                thrust = self._seek_hold_thrust(hover)
                vert_src = 'coast_lock'
            elif self._have_filt:
                nx = float(np.clip(self._nx_f, -1.2, 1.2))
                ny = float(np.clip(self._ny_f, -1.2, 1.2))
                nx_cmd = float(np.clip(nx + nx_roll_bias, -1.2, 1.2))
                coast_rng = area_rng if area_rng is not None else range_m
                yaw_rate = self._yaw_from_pose_and_image(
                    nx_cmd,
                    body,
                    dt,
                    coast_rng,
                    soft_start=bool(recently_passed),
                )
                lat_coast = float(nx_cmd)
                if lat_coast > 0.0:
                    lat_coast *= float(
                        getattr(config, 'ASSIST_ROLL_LEFT_MISS_BOOST', 1.6)
                    )
                des_roll = float(
                    np.clip(
                        -self._lat_sign
                        * min(1.25, lean_scale * 2.6)
                        * lat_coast
                        * self._max_lean,
                        -self._max_lean,
                        self._max_lean,
                    )
                )
                des_pitch = self._fwd_sign * self._max_lean * 0.95
                # Through-slot: tip-high (we're low) → lift; tip-low/high alt
                # → settle. 124438: ny→−0.16 on coast_lift still hit bottom.
                tip_low = float(ny) > float(ny_aim) + 0.12
                tip_high = float(ny) < float(ny_aim) - 0.04
                if tip_high:
                    thrust = hover + 0.012
                    vert_src = 'coast_lift'
                elif tip_low or float(climbed) > 1.85:
                    bleed = 0.012 if tip_low else 0.008
                    thrust = hover - bleed
                    vert_src = 'coast_settle'
                else:
                    thrust = hover + 0.006
                    vert_src = 'coast_lift'
            else:
                nx = 0.0
                ny = ny_aim
                yaw_rate = 0.0
                self._last_yaw_cmd = 0.0
                des_roll = 0.0
                des_pitch = self._fwd_sign * self._max_lean * 0.95
                if float(climbed) > 1.85:
                    thrust = hover - 0.008
                    vert_src = 'coast_settle'
                else:
                    thrust = hover + 0.006
                    vert_src = 'coast_lift'
        elif chaseable and not lost:
            hold_lock = bool(seeking and not locked)
            phase = (
                'seek_yaw'
                if hold_lock
                else ('chase' if not seeking else 'seek_chase')
            )
            nx = float(np.clip(self._nx_f, -1.2, 1.2))
            ny = float(np.clip(self._ny_f, -1.2, 1.2))
            nx_cmd = float(
                np.clip(nx + nx_roll_bias - nx_aim_lat, -1.2, 1.2)
            )
            # Seeking: left/right yaw from image nx + pose (like sink/climb).
            # Course memory uses a heading target — perpetual nx=+0.35 hunted.
            if seeking and getattr(self, '_course_mem', False) and not live_glimpse:
                yaw_rate = self._course_mem_yaw_rate(yaw, dt, now=now)
            elif hold_lock or seeking:
                yaw_rate = self._seek_glimpse_yaw(
                    nx_cmd,
                    dt,
                    live=bool(live_glimpse or locked),
                    body=body,
                )
            else:
                yaw_rate = self._yaw_from_pose_and_image(
                    nx_cmd,
                    body,
                    dt,
                    range_m if range_m is not None else area_rng,
                    soft_start=False,
                )
            if hold_lock:
                des_roll = 0.0
                des_pitch = self._seek_forward_pitch(nx_cmd, False, v_fwd)
                thrust = hover
                vert_src = 'seek_yaw'

            if hold_lock:
                pass  # tip+yaw+bleed already set
            elif use_pad_lift:
                phase = 'pad_lift'
                des_pitch = 0.0
                des_roll = 0.0
                # 131746: ±8.6°/s pad yaw left |nx|~0.9 almost uncorrected.
                pad_yaw = math.radians(
                    float(getattr(config, 'ASSIST_PAD_YAW_MAX_DEG', 45.0))
                )
                pad_yaw = min(pad_yaw, float(self._max_yaw))
                yaw_rate = float(np.clip(yaw_rate, -pad_yaw, pad_yaw))
                self._last_yaw_cmd = yaw_rate
                thrust = hover
                vert_src = f'pad_lift:{src}'
            else:
                lean_scale = float(
                    getattr(
                        config,
                        'KALMAN_BODY_Y_LEAN_SCALE',
                        getattr(config, 'ASSIST_ROLL_SCALE', 0.45),
                    )
                )
                z_gain = float(
                    getattr(config, 'KALMAN_BODY_Z_THRUST_GAIN', 0.028)
                )
                # Forward lean from pose range only (no image-ny pitch).
                # Lateral roll from pose bearing + image nx.
                nx_pose = None
                if body is not None:
                    ex, ey = float(body[0]), float(body[1])
                    # Geometric height: rotate gate1 body into NED. +z = down,
                    # so +pose_dz means the gate is *lower* than we are.
                    gate_ned = cm.body_to_ned(body, roll, pitch, yaw)
                    pose_dz_raw = float(gate_ned[2])
                    horiz = float(math.hypot(gate_ned[0], gate_ned[1]))
                    tilt_bias = cam_tilt_height_bias_m(
                        pose_dz_raw, horiz, ny, pitch, v_fwd
                    )
                    # No fixed aim bias — altitude ∝ live pose height only.
                    pose_dz = pose_dz_raw + tilt_bias
                    fwd_den = max(1.0, abs(ex))
                    # Pose-proportional lateral: (ey − aim_y)/x, not fixed nx_aim.
                    nx_pose = float(
                        np.clip((ey - pose_aim_y) / fwd_den, -1.2, 1.2)
                    )
                    pose_range = float(np.linalg.norm(body))
                    if range_m is None:
                        range_m = pose_range
                    fwd = float(
                        np.clip(0.40 + 0.50 * (pose_range / 12.0), 0.40, 0.90)
                    )
                    align = float(
                        np.clip(1.0 - abs(nx_pose) / 0.45, 0.30, 1.0)
                    )
                    fwd *= align
                else:
                    pose_dz = None
                    fwd = 0.55 * float(
                        np.clip(1.0 - abs(nx_cmd) / 0.55, 0.35, 1.0)
                    )

                if nx_pose is None:
                    lat_err = float(nx_cmd)
                else:
                    # Pose owns lateral (∝ ey); image trims, especially close.
                    img_w = 0.35
                    if range_m is not None and range_m < 12.0:
                        img_w = 0.55
                    if nx_pose * float(nx_cmd) < 0.0 and abs(float(nx_cmd)) > 0.06:
                        img_w = 0.85
                    lat_err = float(
                        np.clip(
                            (1.0 - img_w) * nx_pose + img_w * float(nx_cmd),
                            -1.2,
                            1.2,
                        )
                    )
                # Prefer body-right residual when it confirms a left miss.
                if nx_pose is not None and nx_pose > 0.05:
                    lat_err = max(lat_err, 0.65 * float(nx_pose))
                if lat_err > 0.0:
                    lat_err *= float(
                        getattr(config, 'ASSIST_ROLL_LEFT_MISS_BOOST', 1.6)
                    )
                # Roll ∝ |lat_err| (pose-proportional strength).
                close_boost = 1.0 + 1.6 * min(abs(float(lat_err)) / 0.20, 1.8)
                if range_m is not None and range_m < 12.0:
                    close_boost *= 1.0 + 1.0 * (1.0 - range_m / 12.0)
                if hold_lock:
                    # Pose/image may still jump — no lateral whip while locking.
                    des_roll = 0.0
                else:
                    des_roll = float(
                        np.clip(
                            -self._lat_sign
                            * lean_scale
                            * close_boost
                            * lat_err
                            * self._max_lean,
                            -self._max_lean,
                            self._max_lean,
                        )
                    )

                # Pitch = forward lean from range / alignment only.
                des_pitch = self._fwd_sign * self._max_lean * fwd
                if self._fwd_sign >= 0.0:
                    des_pitch = float(
                        np.clip(des_pitch, 0.0, self._max_lean)
                    )
                else:
                    des_pitch = float(
                        np.clip(des_pitch, -self._max_lean, 0.0)
                    )
                # Seeking: brake speed when off-center — do not force 16° tip.
                if seeking:
                    des_pitch = self._seek_forward_pitch(
                        float(nx_cmd), bool(locked), float(v_fwd)
                    )

                # --- Altitude: live gate height (PnP); strength ∝ |dz| ---
                rng = float(range_m) if range_m is not None else 12.0
                if pose_dz is None:
                    thrust = hover
                    vert_src = f'no_pose:{src}'
                else:
                    # Optional approach-high (default 0 — 105106 lofted on this).
                    if not seeking:
                        pose_dz = float(pose_dz) - float(
                            getattr(config, 'ASSIST_APPROACH_HIGH_M', 0.0)
                        )
                    # 090736 / 105106: never climb when gate is already at or
                    # below image aim — cam-tilt residual dz≈-2.8 lofted to 4 m.
                    if float(ny) > float(ny_aim) - 0.02 and pose_dz < 0.0:
                        pose_dz = 0.0
                    # 112030: latch-only pose_dz≈-0.9 + ny=-0.22 lofted to 14 m.
                    if (
                        seeking
                        and pose_dz < 0.0
                        and not live_glimpse
                        and str(src) in ('next_latch', 'hold_id')
                    ):
                        pose_dz = 0.0
                    # Approach: ignore mild pose-below only when image is
                    # near aim. Tighter than +0.25 — course-2 hit the top
                    # rail at climb≈1.8 m with ny≈0.35 held as "fine".
                    approach_ok = float(
                        getattr(config, 'ASSIST_APPROACH_NY_OK', 0.12)
                    )
                    if (
                        (not seeking)
                        and pose_dz > 0.0
                        and float(ny) < float(ny_aim) + approach_ok
                    ):
                        pose_dz = 0.0
                    # Approach tip sink: mild dig when gate is low in frame.
                    # 124213 overshot top; 124438 forced dead+0.10 and scraped
                    # the bottom rail at ~1.3 m (ny went negative on coast).
                    tip_min_alt = float(
                        getattr(config, 'ASSIST_APPROACH_TIP_MIN_ALT_M', 1.20)
                    )
                    tip_min_rng = float(
                        getattr(config, 'ASSIST_APPROACH_TIP_MIN_RANGE_M', 8.0)
                    )
                    if (
                        (not seeking)
                        and float(ny) > float(ny_aim) + approach_ok
                        and float(climbed) >= tip_min_alt
                        and float(rng) >= tip_min_rng
                    ):
                        k = float(
                            getattr(config, 'ASSIST_APPROACH_TIP_SINK', 0.10)
                        )
                        tip_dz = (
                            (float(ny) - float(ny_aim))
                            * float(np.clip(rng, 6.0, 12.0))
                            * k
                        )
                        tip_dz = float(np.clip(tip_dz, 0.0, 0.40))
                        dead_est = max(0.25, 0.020 * float(rng))
                        if tip_dz > dead_est:
                            pose_dz = max(float(pose_dz), tip_dz)
                    seek_hold_alt = (
                        seeking
                        and self._pass_t is not None
                        and (now - float(self._pass_t))
                        < float(getattr(config, 'ASSIST_SEEK_HOLD_S', 0.0))
                    )
                    # Seeking: scale pose error (default 1.0 — keep ∝ |dz|).
                    if seeking and (not seek_hold_alt) and pose_dz > 0.0:
                        pose_dz = float(pose_dz) * float(
                            getattr(
                                config,
                                'ASSIST_SEEK_POSE_SINK_SCALE',
                                getattr(
                                    config, 'ASSIST_SEEK_POSE_SINK_BOOST', 1.0
                                ),
                            )
                        )
                    # Live-below: only for a *near* gate (far low = gate 3).
                    # Cap so tip cannot dig us under cruise (123610 → floor).
                    same_height_ny = 0.62
                    max_sink_rng = float(
                        getattr(config, 'ASSIST_SEEK_SINK_MAX_RANGE_M', 24.0)
                    )
                    cruise_alt = float(
                        getattr(config, 'ASSIST_SEEK_CRUISE_ALT_M', 1.55)
                    )
                    if (
                        (not seek_hold_alt)
                        and climbed > cruise_alt + 0.35
                        and float(ny) > same_height_ny + 0.12
                        and float(ny) > float(ny_aim) + 0.10
                        and (not seeking or rng <= max_sink_rng)
                    ):
                        live_below = (
                            (float(ny) - same_height_ny)
                            * float(np.clip(rng, 8.0, max_sink_rng))
                            * (0.10 if seeking else 0.18)
                        )
                        # Only sink toward cruise, not through the pad.
                        max_dz = max(0.0, float(climbed) - cruise_alt)
                        live_below = min(float(live_below), max_dz + 0.15)
                        if live_below > 0.0:
                            pose_dz = max(float(pose_dz), live_below)
                    # Seeking: scale/kill pose sink using *body* range (not a
                    # latched range override that can mask a far gate1_body).
                    body_rng = (
                        float(np.linalg.norm(body))
                        if body is not None
                        else float(rng)
                    )
                    if seeking and pose_dz > 0.0 and body_rng > max_sink_rng:
                        pose_dz = float(pose_dz) * float(
                            np.clip(
                                max_sink_rng / max(body_rng, 1.0), 0.0, 1.0
                            )
                        )
                        if body_rng > max_sink_rng * 1.25:
                            pose_dz = 0.0
                    # Hold near zero error; outside that, thrust ∝ |dz|
                    # (gentle when close, much harder when far — no fixed bias).
                    dead_climb = max(0.35, 0.028 * rng)
                    dead_sink = max(
                        0.12 if seeking else 0.25,
                        (0.012 if seeking else 0.020) * rng,
                    )
                    if pose_dz >= 0.0 and pose_dz < dead_sink:
                        thrust = hover
                        vert_src = 'pose_g1:hold'
                    elif pose_dz < 0.0 and pose_dz > -dead_climb:
                        thrust = hover
                        vert_src = 'pose_g1:hold'
                    else:
                        ez = float(np.clip(pose_dz, -4.0, 5.0))
                        mag = abs(ez)
                        dead = dead_sink if ez > 0.0 else dead_climb
                        over = max(0.0, mag - dead)
                        # Linear + quadratic: small over → mild; large → extreme.
                        shape = over + 0.70 * over * over
                        sink_g = 2.2 * float(
                            getattr(config, 'ASSIST_SEEK_SINK_GAIN_SCALE', 1.25)
                        ) if seeking else 2.2
                        climb_g = 0.9 if seeking else 1.1
                        gain = z_gain * (sink_g if ez > 0.0 else climb_g)
                        thrust = hover - math.copysign(gain * shape, ez)
                        # Cap ∝ over; seeking allows a quicker drop to gate height.
                        sink_cap = (
                            0.014 + 0.040 * float(np.clip(over / 1.0, 0.0, 2.5))
                            if seeking
                            else 0.010 + 0.028 * float(
                                np.clip(over / 1.2, 0.0, 2.5)
                            )
                        )
                        climb_cap = 0.004 + 0.014 * float(
                            np.clip(over / 1.5, 0.0, 2.0)
                        )
                        thrust = float(
                            np.clip(thrust, hover - sink_cap, hover + climb_cap)
                        )
                        vert_src = (
                            'pose_g1:climb'
                            if pose_dz < 0.0
                            else 'pose_g1:sink'
                        )
                    # Enforce seek floor on pose path — climb a bit to arrest.
                    min_alt = float(
                        getattr(config, 'ASSIST_SEEK_MIN_ALT_M', 0.55)
                    )
                    floor_boost = float(
                        getattr(config, 'ASSIST_SEEK_FLOOR_THRUST', 0.022)
                    )
                    if seeking and float(climbed) < min_alt:
                        if float(thrust) < hover + floor_boost:
                            thrust = hover + floor_boost
                        vert_src = 'seek_floor'
                        if pose_dz is not None and pose_dz > 0.0:
                            pose_dz = 0.0
        elif seeking and self._have_filt and ghost_hold:
            # Vision flickered — unlock already cleared; keep soft track.
            phase = 'seek_yaw'
            nx = float(np.clip(self._nx_f, -1.2, 1.2))
            ny = float(self._ny_f)
            nx_cmd = float(
                np.clip(nx + nx_roll_bias - nx_aim_lat, -1.2, 1.2)
            )
            if getattr(self, '_course_mem', False):
                yaw_rate = self._course_mem_yaw_rate(yaw, dt, now=now)
            else:
                yaw_rate = self._seek_glimpse_yaw(
                    nx_cmd, dt, live=False, body=body
                )
            des_roll = 0.0
            des_pitch = self._seek_forward_pitch(nx_cmd, False, v_fwd)
            thrust = hover
            vert_src = 'seek_yaw'
        elif seeking and not locked:
            # No live box — tip/crawl; latch yaw or blind scan (121451 mute).
            phase = 'seek_lock'
            nx = float(self._nx_f) if self._have_filt else 0.0
            ny = float(self._ny_f) if self._have_filt else ny_aim
            des_roll = 0.0
            if getattr(self, '_course_mem', False):
                nx_cmd = float(
                    np.clip(nx + nx_roll_bias - nx_aim_lat, -1.2, 1.2)
                )
                yaw_rate = self._course_mem_yaw_rate(yaw, dt, now=now)
                des_pitch = self._seek_forward_pitch(nx_cmd, False, v_fwd)
                vert_src = 'course_mem'
            elif self._have_filt or self._latch_still_valid(
                now,
                max_age=float(getattr(config, 'ASSIST_LATCH_HOLD_S', 12.0)),
            ):
                if not self._have_filt:
                    nx = float(self._next_nx)
                    ny = float(self._next_ny)
                nx_cmd = float(
                    np.clip(nx + nx_roll_bias - nx_aim_lat, -1.2, 1.2)
                )
                yaw_body = body if body is not None else self._next_body
                yaw_rate = self._seek_glimpse_yaw(
                    nx_cmd, dt, live=False, body=yaw_body
                )
                des_pitch = self._seek_forward_pitch(nx_cmd, False, v_fwd)
                vert_src = 'seek_lock'
            else:
                yaw_rate = self._seek_blind_scan_yaw(now, yaw, dt)
                self._last_yaw_cmd = yaw_rate
                des_pitch = self._seek_forward_pitch(0.0, False, v_fwd)
                vert_src = 'seek_scan'
                phase = 'seek_scan'
            thrust = hover
        else:
            phase = 'lost' if lost else 'search'
            nx = float(self._nx_f) if self._have_filt else 0.0
            ny = float(self._ny_f) if self._have_filt else ny_aim
            if seeking:
                phase = 'seek_scan'
                # Prefer latch yaw; else blind L/R scan (not mute tip-crawl).
                if getattr(self, '_course_mem', False):
                    yaw_rate = self._course_mem_yaw_rate(yaw, dt, now=now)
                    des_pitch = self._seek_forward_pitch(0.0, False, v_fwd)
                    vert_src = 'course_mem'
                elif self._latch_still_valid(
                    now,
                    max_age=float(getattr(config, 'ASSIST_LATCH_HOLD_S', 12.0)),
                ):
                    nx = float(self._next_nx)
                    ny = float(self._next_ny)
                    nx_cmd = float(
                        np.clip(nx + nx_roll_bias - nx_aim_lat, -1.2, 1.2)
                    )
                    yaw_rate = self._seek_glimpse_yaw(
                        nx_cmd, dt, live=False, body=self._next_body
                    )
                    des_pitch = self._seek_forward_pitch(nx_cmd, False, v_fwd)
                    vert_src = 'seek_scan'
                else:
                    yaw_rate = self._seek_blind_scan_yaw(now, yaw, dt)
                    des_pitch = self._seek_forward_pitch(0.0, False, v_fwd)
                    vert_src = 'seek_scan'
                thrust = hover
            else:
                yaw_rate = 0.0
                des_pitch = self._fwd_sign * self._max_lean * 0.35
                thrust = hover
                vert_src = 'hover'
            des_roll = 0.0
            self._last_yaw_cmd = yaw_rate
            # 085654: after losing the gate mid-loft, hover kept climbing to
            # ~19 m. Bleed a little collective once we are clearly airborne.
            if climbed > 2.5 and vert_src == 'hover':
                thrust = hover - 0.010
                vert_src = 'lost_settle'

        roll_rate = float(self._roll_pid.update(des_roll - roll, dt))
        pitch_rate = float(self._pitch_pid.update(des_pitch - pitch, dt))

        tilt = max(
            0.88,
            math.cos(abs(float(des_pitch))) * math.cos(abs(float(des_roll))),
        )
        delta = float(thrust) - hover
        thrust = (hover / tilt) + delta
        # Seeking: blend pose thrust with stronger image-ny sink/climb.
        # Not while coasting — 104433 seek bleed scraped gate-1 bottom rail
        # on the way through the opening.
        if seeking and not coasting:
            if self._have_filt:
                pose_thr = float(thrust)
                sink_rng = range_m
                if sink_rng is None:
                    sink_rng = self._next_rng
                if sink_rng is None:
                    sink_rng = self._last_range_m
                # Prefer raw pose height for stop-sink (before hold deadband).
                pose_dz_for_ny = pose_dz_raw if pose_dz_raw is not None else pose_dz
                ny_thr, ny_vert = self._seek_ny_thrust(
                    hover,
                    hover,
                    float(self._ny_f),
                    ny_aim,
                    climbed=float(climbed),
                    range_m=sink_rng,
                    allow_climb=bool(live_glimpse),
                    pose_dz=pose_dz_for_ny,
                )
                # Pose owns when it has a height command; ny is a proportional
                # trim (not min-thrust winner — that broke ∝ pose).
                pose_active = vert_src in (
                    'pose_g1:sink',
                    'pose_g1:climb',
                    'pose_g1:hold',
                ) or str(vert_src).startswith('pose_g1')
                if ny_vert == 'seek_floor' or vert_src == 'seek_floor':
                    thrust = max(pose_thr, hover)
                    vert_src = 'seek_floor'
                elif pose_active and vert_src == 'pose_g1:sink':
                    ny_delta = float(ny_thr) - hover
                    thrust = float(pose_thr + 0.45 * ny_delta)
                    vert_src = 'seek_pose_sink'
                elif pose_active and vert_src == 'pose_g1:hold':
                    # Pose level: dig only while clearly above cruise with tip
                    # ny (123610 held at 3 m). Near cruise/floor → hold.
                    cruise_alt = float(
                        getattr(config, 'ASSIST_SEEK_CRUISE_ALT_M', 1.55)
                    )
                    clearly_high = float(climbed) > cruise_alt + 0.45
                    ny_err = float(self._ny_f) - float(ny_aim)
                    tip_low = ny_err > float(
                        getattr(config, 'ASSIST_SEEK_POSE_STOP_NY_ERR', 0.30)
                    )
                    if ny_vert == 'seek_sink' and clearly_high and tip_low:
                        thrust = float(ny_thr)
                        vert_src = ny_vert
                    elif ny_vert == 'seek_sink' or ny_vert == 'seek_hold':
                        thrust = float(pose_thr)
                        vert_src = 'seek_hold'
                    else:
                        thrust = max(float(pose_thr), float(ny_thr))
                        vert_src = ny_vert
                elif pose_active and vert_src == 'pose_g1:climb':
                    # Below gate — climb; ignore ny sink.
                    if ny_vert == 'seek_sink':
                        thrust = float(pose_thr)
                        vert_src = vert_src
                    else:
                        thrust = max(pose_thr, float(ny_thr))
                        vert_src = (
                            'seek_climb' if ny_vert == 'seek_climb' else vert_src
                        )
                else:
                    thrust = float(ny_thr)
                    vert_src = ny_vert
            elif vert_src in (
                'seek_yaw',
                'seek_lock',
                'seek_scan',
                'coast_yaw',
                'coast_lock',
            ):
                thrust = self._seek_hold_thrust(thrust)
        # Descent brake while seeking/coasting post-pass (before pad floor).
        if seeking or (
            self._pass_t is not None and (now - self._pass_t) < 8.0
        ):
            if 'sink' in str(vert_src) or str(vert_src) in (
                'seek_hold',
                'coast_yaw',
            ):
                ny_brake = float(self._ny_f) if self._have_filt else None
                thrust, vert_src = self._brake_descent(
                    thrust,
                    hover,
                    vert_src,
                    ny=ny_brake,
                    ny_aim=ny_aim,
                    climbed=climbed,
                )
        # Hard floor on every seek/post-pass path — climb a touch to arrest.
        min_alt = float(getattr(config, 'ASSIST_SEEK_MIN_ALT_M', 0.55))
        floor_boost = float(
            getattr(config, 'ASSIST_SEEK_FLOOR_THRUST', 0.022)
        )
        if (
            (seeking or (self._pass_t is not None and (now - self._pass_t) < 8.0))
            and float(climbed) < min_alt
        ):
            # 115626: hover+0.01 could not arrest ~2.5 m/s into the rail.
            descent = -float(self._climb_rate)
            boost = floor_boost
            if descent >= float(
                getattr(config, 'ASSIST_SEEK_DESCENT_FULL_MPS', 1.20)
            ):
                boost = max(
                    boost,
                    float(
                        getattr(config, 'ASSIST_SEEK_DESCENT_BRAKE_THRUST', 0.028)
                    )
                    * 1.5,
                )
            thrust = max(float(thrust), hover + boost)
            vert_src = 'seek_floor'
        # Post-pass cruise: get eyes up before tip-crawl when low/blind
        # (115959: climb≈−0.2 after gate 2 → never acquired gate 3).
        cruise_alt = float(getattr(config, 'ASSIST_SEEK_CRUISE_ALT_M', 1.55))
        cruise_boost = float(
            getattr(config, 'ASSIST_SEEK_CRUISE_THRUST', 0.016)
        )
        post_pass_window = (
            self._pass_t is not None and (now - float(self._pass_t)) < 8.0
        )
        ny_for_cruise = float(self._ny_f) if self._have_filt else None
        still_digging = bool(
            live_glimpse
            and ny_for_cruise is not None
            and (ny_for_cruise - float(ny_aim))
            > float(getattr(config, 'ASSIST_SEEK_POSE_STOP_NY_ERR', 0.30))
            and float(climbed) >= min_alt
        )
        if (
            (seeking or coasting)
            and post_pass_window
            and float(climbed) >= min_alt
            and float(climbed) < cruise_alt
            and not still_digging
            and str(vert_src)
            in (
                'seek_lock',
                'seek_hold',
                'seek_yaw',
                'seek_scan',
                'seek_floor',
                'coast_lift',
                'coast_yaw',
                'coast_lock',
                'hover',
            )
        ):
            thrust = max(float(thrust), hover + cruise_boost)
            vert_src = 'seek_cruise'
        # Course-2 memory: after g1, nudge slightly UP while soft-yawing right.
        if (
            seeking
            and getattr(self, '_course_mem', False)
            and not live_glimpse
            and post_pass_window
            and float(climbed) < cruise_alt + 0.45
            and 'sink' not in str(vert_src)
            and vert_src != 'seek_floor'
        ):
            climb_b = float(
                getattr(config, 'ASSIST_POST_G1_CLIMB_THRUST', 0.010)
            )
            thrust = max(float(thrust), hover + climb_b)
            if str(vert_src) in ('seek_hold', 'seek_lock', 'seek_yaw', 'seek_scan'):
                vert_src = 'course_mem_climb'
        # Blind scan loft cap — MUST outlive the 8 s post-pass window
        # (123610: window ended → thr≈hover tip-climbed to ~6 m over gate 2).
        if (
            seeking
            and str(phase) == 'seek_scan'
            and not live_glimpse
            and float(climbed) > cruise_alt + 0.10
        ):
            base_cap = float(
                getattr(config, 'ASSIST_SEEK_SCAN_CAP_THRUST', 0.022)
            )
            excess = max(0.0, float(climbed) - cruise_alt)
            cap = base_cap + 0.018 * min(excess, 4.0)
            thrust = min(float(thrust), hover - cap)
            vert_src = 'seek_scan_cap'
        # Hard ceiling while BLIND seeking — stop the 3–6 m tip loft
        # (123610). Do not fight a live climb/sink toward a seen gate.
        seek_ceil = float(getattr(config, 'ASSIST_SEEK_CEILING_M', 2.35))
        if (
            seeking
            and not live_glimpse
            and str(phase) in ('seek_scan', 'seek_lock', 'lost', 'search')
            and float(climbed) > seek_ceil
            and 'sink' not in str(vert_src)
            and 'climb' not in str(vert_src)
        ):
            excess = float(climbed) - seek_ceil
            ceil_bleed = 0.016 + 0.020 * min(excess, 3.0)
            thrust = min(float(thrust), hover - ceil_bleed)
            vert_src = 'seek_ceiling'
        # Post-g1: bounded right turn (132813 too short; 133354 never stopped).
        post_g1 = (
            getattr(self, '_course_mem_spent', False)
            and not bool(getattr(self, '_course_mem_done', False))
            and self._pass_t is not None
            and (now - float(self._pass_t))
            < float(getattr(config, 'ASSIST_POST_G1_YAW_MAX_S', 1.8))
            and int(self._active_gate or 0) == 1
        )
        if post_g1 and not self._course_mem_heading_done(yaw, now=now):
            self._course_mem = True
            yaw_rate = self._course_mem_yaw_rate(yaw, dt, now=now)
        elif getattr(self, '_course_mem_spent', False) and int(
            self._active_gate or 0
        ) == 1:
            # Turn finished: never re-arm. Block far-left ghost yaw; allow
            # fine left so a centered g2 can be chased (133934).
            self._course_mem = False
            mem_nx = float(
                self._next_nx
                if self._next_nx is not None
                else getattr(config, 'ASSIST_POST_G1_NX', 0.55)
            )
            lock_s = float(getattr(config, 'ASSIST_POST_G1_YAW_LOCK_S', 2.5))
            if (
                self._pass_t is not None
                and (now - float(self._pass_t)) < lock_s
            ):
                left_lim = float(
                    getattr(config, 'ASSIST_POST_G1_LIVE_NX_LEFT', -0.18)
                )
                fine = math.radians(
                    float(getattr(config, 'ASSIST_POST_G1_FINE_YAW_DEG', 22.0))
                )
                nx_ref = (
                    float(self._nx_f) if self._have_filt else 0.0
                )
                if mem_nx >= 0.0 and float(yaw_rate) < 0.0:
                    if nx_ref < left_lim:
                        yaw_rate = 0.0
                        self._last_yaw_cmd = 0.0
                    else:
                        yaw_rate = max(float(yaw_rate), -fine)
                elif mem_nx < 0.0 and float(yaw_rate) > 0.0:
                    if nx_ref > -left_lim:
                        yaw_rate = 0.0
                        self._last_yaw_cmd = 0.0
                    else:
                        yaw_rate = min(float(yaw_rate), fine)
        # Stronger lateral/vertical response when tracking a gate — scales
        # commands only; aims / floors / ranges stay as tuned.
        # 114108: do NOT amplify coast_lift / hold (lofted into gate 1).
        tracking = bool(
            phase
            in (
                'chase',
                'coast',
                'seek_chase',
                'seek_yaw',
                'seek_scan',
            )
            and vert_src != 'seek_floor'
            and vert_src != 'seek_scan_cap'
            and vert_src != 'seek_ceiling'
            and vert_src != 'coast_settle'
        )
        if tracking:
            lat_a = float(getattr(config, 'ASSIST_LATERAL_AUTH', 1.0))
            vert_a = float(getattr(config, 'ASSIST_VERTICAL_AUTH', 1.0))
            if lat_a > 1.0 and not getattr(self, '_course_mem', False):
                des_roll = float(
                    np.clip(
                        float(des_roll) * lat_a,
                        -self._max_lean,
                        self._max_lean,
                    )
                )
                yaw_lim = max(
                    self._max_yaw,
                    math.radians(
                        float(
                            getattr(config, 'ASSIST_SEEK_LIVE_YAW_MAX_DEG', 22.0)
                        )
                    ),
                )
                yaw_rate = float(
                    np.clip(float(yaw_rate) * lat_a, -yaw_lim, yaw_lim)
                )
                self._last_yaw_cmd = yaw_rate
            vert_correcting = any(
                s in str(vert_src)
                for s in ('sink', 'climb', 'seek_pose_sink')
            )
            if vert_a > 1.0 and vert_correcting:
                thrust = hover + (float(thrust) - hover) * vert_a
        # Global speed cap — brake forward lean in chase/coast/seek.
        des_pitch = self._limit_forward_pitch_for_speed(des_pitch, v_fwd)
        # Slew pitch so tip can't flip every tick (125233 shake).
        des_pitch = self._slew_pitch(des_pitch, dt)
        # Wide rail — do not flatten pose-proportional sink with a tight clamp.
        thrust = float(np.clip(thrust, hover - 0.075, hover + 0.035))
        thrust = float(np.clip(thrust, 0.200, 0.33))

        path = {
            'phase': phase,
            'source': 'assist',
            'norm_x': float(nx) if chaseable or self._have_filt else None,
            'norm_y': float(ny) if chaseable or self._have_filt else None,
            'norm_src': src,
            'range_m': range_m,
            'area_px': area_px,
            'align': None,
            'climbed': climbed,
            'climbed_raw': climbed_raw,
            'peak_climbed': self._peak_climbed,
            'thrust': thrust,
            'des_roll': des_roll,
            'des_pitch': des_pitch,
            'yaw_rate': yaw_rate,
            'vert_src': vert_src,
            'pose_dz': pose_dz,
            'pose_dz_raw': pose_dz_raw,
            'tilt_bias': float(tilt_bias),
            'nx_roll_bias': float(nx_roll_bias),
            'v_lat': float(v_lat),
            'v_fwd': float(v_fwd),
            'chaseable': bool(chaseable),
            'coasting': bool(coasting),
            'seeking': bool(seeking),
            'gate_lock': bool(locked),
            'lock_count': int(self._lock_count),
        }
        if chaseable:
            path['align'] = float(np.clip(1.0 - abs(float(nx)) / 0.55, 0.0, 1.0))
        shared_data['kalman_path'] = path
        shared_data['assist'] = path

        if now - self._last_status_t >= 1.0:
            self._last_status_t = now
            dz_s = (
                f'{pose_dz:.2f}' if pose_dz is not None else '-'
            )
            print(
                f'[ASSIST] phase={phase} src={src} '
                f'nx={path["norm_x"]} ny={path["norm_y"]} '
                f'rng={range_m} climb={climbed:.2f} thr={thrust:.3f} '
                f'pitch={math.degrees(des_pitch):.1f}° '
                f'vert={vert_src} dz={dz_s} '
                f'lock={int(locked)}/{self._lock_count}',
                flush=True,
            )
            log = shared_data.get('log_event')
            if log:
                log(
                    'ASSIST',
                    f'phase={phase} src={src} climb={climbed:.2f} '
                    f'thr={thrust:.3f} vert={vert_src} dz={dz_s}',
                )

        shared_data['planner_target'] = {
            'vn': 0.0,
            've': 0.0,
            'vd': 0.0,
            'yaw_rate': yaw_rate,
            'kalman': True,
            'roll_rate': roll_rate,
            'pitch_rate': pitch_rate,
            'thrust': thrust,
            'desired_roll': des_roll,
            'desired_pitch': des_pitch,
            'desired_yaw': yaw + (float(nx) * 0.5 if chaseable else 0.0),
        }
        return shared_data['planner_target']
