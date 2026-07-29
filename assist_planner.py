"""Assist planner — always track current primary gate (gate1).

When gate2 is promoted it becomes gate1; same policy continues.

Policy (user-defined):
- Pitch + yaw: keep gate1 in the image frame (YOLO / dual norms).
- Altitude: only from gate1 PnP pose (camera-optical Y vs boresight) —
  climb if we are below the gate, sink if we are above. Never from image-ny.
- Forward/lateral lean: gate1 PnP body (range / body-y).

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
    """Post-pass aim when primary lock is empty: course_bearing or gate2 body.

    Returns (nx, ny, source, range_m) — any field may be None.
    """
    now = time.monotonic()
    cb = shared_data.get('course_bearing') or {}
    if isinstance(cb, dict):
        ts = _f(cb.get('ts'))
        nx = _f(cb.get('nx'))
        ny = _f(cb.get('ny'))
        if (
            ts is not None
            and (now - ts) < 0.8
            and nx is not None
            and ny is not None
        ):
            return nx, ny, f"bearing:{cb.get('source', '?')}", _f(cb.get('range_m'))
    dual = shared_data.get('dual_gate_pnp') or {}
    g2 = dual.get('gate2_body')
    if g2 is not None and len(g2) >= 3:
        try:
            x, y, z = float(g2[0]), float(g2[1]), float(g2[2])
        except (TypeError, ValueError):
            x = y = z = 0.0
        if x > 0.5 and math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
            nx = float(np.clip(y / x, -1.2, 1.2))
            ny = float(np.clip(z / x, -1.2, 1.2))
            rng = float(math.sqrt(x * x + y * y + z * z))
            return nx, ny, 'gate2_body', rng
    return None, None, 'none', None


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
        max_yaw = min(config.YAW_RATE_MAX_RAD_S, math.radians(30.0))
        max_rate = float(config.KALMAN_MAX_RATE_RAD_S)
        self._max_lean = math.radians(
            float(getattr(config, 'ASSIST_LEAN_DEG', config.KALMAN_MAX_LEAN_DEG))
        )
        self._fwd_sign = float(getattr(config, 'FORWARD_PITCH_SIGN', 1.0))
        self._lat_sign = float(getattr(config, 'LATERAL_LEAN_SIGN', 1.0))
        self._yaw_pid = PIDController(
            PIDConfig(
                kp=float(getattr(config, 'KALMAN_KP_YAW', 0.9)),
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
        self._yaw_slew = math.radians(120.0)
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
        self._lift_start_t = None
        self._left_pad = False
        self._airborne_t = None
        self._last_area_px = None
        self._body_f = None

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
        self._lift_start_t = None
        self._left_pad = False
        self._airborne_t = None
        self._last_area_px = None
        self._body_f = None

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
            self._pass_t = now
            self._coast_until = now + float(
                getattr(config, 'ASSIST_COAST_S', 1.5)
            )
            self._seek_until = now + float(
                getattr(config, 'ASSIST_SEEK_S', 14.0)
            )
            self._yaw_pid.reset()
            self._last_yaw_cmd = 0.0
            self._have_filt = False
            self._peak_climbed = 0.0
            if self._arm_z is not None:
                climbed = self._climb_m(shared_data)
                self._peak_climbed = max(0.0, climbed)
            log = shared_data.get('log_event')
            if log:
                log('ASSIST_PASS', f'active_gate={ag_i}')
            print(f'[ASSIST] GATE_PASSED → coast/seek gate={ag_i}', flush=True)
        self._active_gate = ag_i

    def _chaseable(self, nx, ny, area_px, seeking: bool) -> bool:
        if nx is None or ny is None:
            return False
        ny_r, nx_r = float(ny), float(nx)
        if area_px is not None and area_px > 90000.0:
            return False
        if ny_r > 0.92:
            return False
        if seeking and ny_r > 0.78 and abs(nx_r) < 0.22:
            return False
        if seeking and area_px is not None and area_px < 700.0:
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

        self._note_pass(shared_data, now)

        nx_raw, ny_raw, src = image_gate_norm(shared_data)
        det = shared_data.get('gate_detection') or {}
        area_px = _f(det.get('area_px')) if isinstance(det, dict) else None
        dual = shared_data.get('dual_gate_pnp') or {}
        range_m = _f(dual.get('gate1_range_m'))
        if range_m is None and area_px is not None and area_px > 50.0:
            range_m = float((320.0 * 1.5) / math.sqrt(area_px))
        # Reject identity flips (assist_a: 8 m → 25 m in one tick).
        if (
            range_m is not None
            and self._last_range_m is not None
            and abs(range_m - self._last_range_m) > 8.0
            and area_px is not None
            and area_px > 50.0
        ):
            range_m = float((320.0 * 1.5) / math.sqrt(area_px))
            src = 'area_range'
        if range_m is not None:
            self._last_range_m = range_m

        # Visual commit before race_status updates (assist_a lost YOLO at
        # ~9 m with bbox exploding / ny ← −0.6, then dual_pnp stole ID).
        # Require real altitude — assist_j committed while climb≈−0.4 on floor.
        climb_for_commit = (
            self._climb_f if self._climb_f is not None else self._climb_m(shared_data)
        )
        if (
            now >= self._coast_until
            and self._left_pad
            and climb_for_commit > 0.45
            and nx_raw is not None
            and abs(float(nx_raw)) < 0.35
            and (
                (area_px is not None and area_px > 9000.0)
                or (range_m is not None and range_m < 8.0)
                or (
                    area_px is not None
                    and area_px > 4500.0
                    and ny_raw is not None
                    and float(ny_raw) < -0.40
                )
            )
        ):
            self._coast_until = now + float(
                getattr(config, 'ASSIST_COAST_S', 1.5)
            )
            self._seek_until = max(
                self._seek_until,
                now + float(getattr(config, 'ASSIST_SEEK_S', 14.0)),
            )
            self._have_filt = False
            print('[ASSIST] VISUAL_COMMIT → coast', flush=True)
            # Drop YOLO sticky identity so post-coast seek acquires NEXT gate
            # (not the expanding remnant / far steal of the same lock).
            shared_data['vision_begin_next_gate'] = True
            log = shared_data.get('log_event')
            if log:
                log('ASSIST_COMMIT', f'area={area_px} range={range_m}')

        coasting = now < self._coast_until
        seeking = now < self._seek_until
        # Blind coast ignored gate-2 glimpses (031742: DUAL_PNP during coast
        # with yaw=0). Prefer course_bearing / gate2 when primary is empty.
        if (coasting or seeking) and nx_raw is None:
            hx, hy, hsrc, hrng = next_gate_hint(shared_data)
            if hx is not None and hy is not None:
                nx_raw, ny_raw, src = hx, hy, hsrc
                if range_m is None:
                    range_m = hrng
        shared_data['post_pass_hunt'] = bool(seeking and not coasting)

        chaseable = self._chaseable(nx_raw, ny_raw, area_px, seeking)
        # assist_g: area collapsed while range jumped 18→34 (far-gate steal).
        # After a pass, a far jump is the NEXT gate — do not hold old id.
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
        if chaseable and not identity_steal:
            self._last_see_t = now
            if not self._have_filt:
                self._nx_f = float(nx_raw)
                self._ny_f = float(ny_raw)
                self._have_filt = True
            else:
                self._nx_f = 0.65 * self._nx_f + 0.35 * float(nx_raw)
                self._ny_f = 0.65 * self._ny_f + 0.35 * float(ny_raw)
            if area_px is not None:
                self._last_area_px = float(area_px)
        elif chaseable and identity_steal:
            # Keep last filter / last_see; coast on prior aim briefly.
            self._last_see_t = now
            if range_m is not None and self._last_range_m is not None:
                range_m = self._last_range_m
            src = 'hold_id'
        lost = (now - self._last_see_t) > float(
            getattr(config, 'ASSIST_LOST_TIMEOUT_S', 0.8)
        )

        hover = float(config.HOVER_THRUST)
        ny_aim = float(getattr(config, 'ASSIST_NY_AIM', config.KALMAN_NY_AIM))
        climbed_raw = self._climb_m(shared_data)
        climbed = self._climb_filtered(shared_data)
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

        # Post-pass: only reject tiny / floor-speck boxes. Old range>18 m and
        # ny>0.65 filters dropped real gate-2 at 20–25 m (031742).
        if seeking and chaseable and (
            (area_px is not None and area_px < 500.0)
            or (ny_raw is not None and float(ny_raw) > 0.88)
        ):
            chaseable = False

        # Abort blind coast as soon as a plausible next gate appears
        # (sideways → yaw on nx, up → climb on ny). Still require >9 m so we
        # do not re-lock the gate we just flew through.
        if coasting and chaseable and not identity_steal:
            remnant = area_px is not None and area_px > 25000.0
            still_inside = range_m is not None and range_m < 9.0
            if not remnant and not still_inside:
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
        body_raw = gate1_body_m(shared_data)
        if body_raw is not None and chaseable and not identity_steal:
            if self._body_f is None:
                self._body_f = body_raw.copy()
            else:
                jump = float(np.linalg.norm(body_raw - self._body_f))
                a = 0.45 if jump > 6.0 else 0.30
                self._body_f = (1.0 - a) * self._body_f + a * body_raw
        body = self._body_f

        phase = 'hover'
        pose_ez = None
        if coasting:
            phase = 'coast'
            nx = 0.0
            ny = ny_aim
            des_roll = 0.0
            des_pitch = self._fwd_sign * self._max_lean * 0.95
            yaw_rate = 0.0
            thrust = hover
            vert_src = 'coast_hold'
        elif chaseable and not lost:
            phase = 'chase' if not seeking else 'seek_chase'
            nx = float(np.clip(self._nx_f, -1.2, 1.2))
            ny = float(np.clip(self._ny_f, -1.2, 1.2))
            # --- Image: yaw + pitch keep gate1 in frame ---
            yaw_rate = float(self._yaw_pid.update(nx, dt))
            max_step = self._yaw_slew * dt
            yaw_rate = float(
                np.clip(
                    yaw_rate,
                    self._last_yaw_cmd - max_step,
                    self._last_yaw_cmd + max_step,
                )
            )
            self._last_yaw_cmd = yaw_rate
            look = float(np.clip(ny - ny_aim, -1.0, 1.0))

            if use_pad_lift:
                phase = 'pad_lift'
                des_pitch = 0.0
                des_roll = 0.0
                yaw_rate = float(np.clip(yaw_rate, -0.15, 0.15))
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
                # Baseline forward from pose range (translation); image owns
                # the pitch correction that keeps the gate in frame.
                if body is not None:
                    ex, ey = float(body[0]), float(body[1])
                    pose_ez = float(cm.body_to_cam(body)[1])
                    fwd_den = max(1.0, abs(ex))
                    nx_pose = float(np.clip(ey / fwd_den, -1.2, 1.2))
                    pose_range = float(np.linalg.norm(body))
                    if range_m is None:
                        range_m = pose_range
                    fwd = float(
                        np.clip(0.40 + 0.50 * (pose_range / 12.0), 0.40, 0.90)
                    )
                    align = float(
                        np.clip(1.0 - abs(nx_pose) / 0.50, 0.35, 1.0)
                    )
                    fwd *= align
                    des_roll = float(
                        np.clip(
                            -self._lat_sign
                            * lean_scale
                            * nx_pose
                            * self._max_lean,
                            -self._max_lean,
                            self._max_lean,
                        )
                    )
                else:
                    pose_ez = None
                    fwd = 0.55 * float(np.clip(1.0 - abs(nx) / 0.55, 0.35, 1.0))
                    des_roll = float(
                        np.clip(
                            -self._lat_sign * 0.35 * nx * self._max_lean,
                            -self._max_lean,
                            self._max_lean,
                        )
                    )

                # Pitch = pose forward + image look (keep gate1 in frame).
                # Keep pitch forward-biased so look-up never flips to nose-up
                # loft (040724: des_pitch≈−0.05 + thr 0.32).
                des_pitch = self._fwd_sign * self._max_lean * (
                    fwd + 0.55 * look
                )
                if self._fwd_sign >= 0.0:
                    des_pitch = float(
                        np.clip(des_pitch, 0.0, self._max_lean)
                    )
                else:
                    des_pitch = float(
                        np.clip(des_pitch, -self._max_lean, 0.0)
                    )

                # --- Altitude: gate1 PnP only (cam Y, +down) ---
                # ez<0 ⇒ gate above boresight ⇒ we are below ⇒ climb
                # ez>0 ⇒ gate below boresight ⇒ we are above ⇒ sink
                if pose_ez is None:
                    thrust = hover
                    vert_src = f'no_pose:{src}'
                else:
                    dead = 0.40  # metres of cam-Y deadzone around boresight
                    if abs(pose_ez) < dead:
                        thrust = hover
                        vert_src = 'pose_g1:hold'
                    else:
                        thrust = hover - z_gain * pose_ez
                        vert_src = (
                            'pose_g1:climb'
                            if pose_ez < 0.0
                            else 'pose_g1:sink'
                        )
        else:
            phase = 'lost' if lost else 'search'
            nx = float(self._nx_f) if self._have_filt else 0.0
            ny = float(self._ny_f) if self._have_filt else ny_aim
            if seeking:
                phase = 'seek_scan'
                scan = 0.20 * math.sin(2.0 * math.pi * 0.12 * now)
                yaw_rate = float(scan)
                des_pitch = self._fwd_sign * self._max_lean * 0.50
                thrust = hover
            else:
                yaw_rate = 0.0
                des_pitch = self._fwd_sign * self._max_lean * 0.35
                thrust = hover
            des_roll = 0.0
            self._last_yaw_cmd = yaw_rate
            vert_src = 'hover'

        roll_rate = float(self._roll_pid.update(des_roll - roll, dt))
        pitch_rate = float(self._pitch_pid.update(des_pitch - pitch, dt))

        tilt = max(
            0.88,
            math.cos(abs(float(des_pitch))) * math.cos(abs(float(des_roll))),
        )
        delta = float(thrust) - hover
        thrust = (hover / tilt) + delta
        # Modest thrust band — altitude from pose, not runaway climb/dive.
        thrust = float(np.clip(thrust, hover - 0.020, hover + 0.030))
        thrust = float(np.clip(thrust, 0.210, 0.32))

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
            'pose_ez': pose_ez,
            'chaseable': bool(chaseable),
            'coasting': bool(coasting),
            'seeking': bool(seeking),
        }
        if chaseable:
            path['align'] = float(np.clip(1.0 - abs(float(nx)) / 0.55, 0.0, 1.0))
        shared_data['kalman_path'] = path
        shared_data['assist'] = path

        if now - self._last_status_t >= 1.0:
            self._last_status_t = now
            print(
                f'[ASSIST] phase={phase} src={src} '
                f'nx={path["norm_x"]} ny={path["norm_y"]} '
                f'rng={range_m} climb={climbed:.2f} thr={thrust:.3f} '
                f'pitch={math.degrees(des_pitch):.1f}°',
                flush=True,
            )
            log = shared_data.get('log_event')
            if log:
                log(
                    'ASSIST',
                    f'phase={phase} src={src} climb={climbed:.2f} '
                    f'thr={thrust:.3f}',
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
