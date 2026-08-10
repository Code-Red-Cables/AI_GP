"""Spline waypoint follower on DERIVED position (no vision in the loop).

Flies a fixed, pre-captured path. Position and heading come from
``shared_data['position_ned']`` / ``['attitude']``, which the dual-gate EKF
publishes from IMU dead reckoning (optionally corrected by PnP — see the
warning below). Nothing here reads the camera.

Why this can work despite drift
-------------------------------
The derived frame drifts, so the captured waypoints are wrong in absolute
terms. But if the *same* estimator produced them, the error is largely
**common mode**: the follower is wrong in the same direction the capture was,
and the two substantially cancel. The quantity that matters is therefore
run-to-run *repeatability*, not absolute accuracy.

  CRITICAL: capture and replay must use the same ``EKF_USE_PNP`` setting.
  PnP corrections pull the frame toward a landmark whose world pose was
  itself initialised from the drone's own belief, so their timing and size
  vary run to run. That is exactly the non-repeatable component. Pure dead
  reckoning (``EKF_USE_PNP=0``) is absolutely worse but more deterministic,
  which is usually the better trade here. Mismatch the two and there is no
  cancellation at all.

Output contract
---------------
Emits the same rate+thrust dict the assist / kalman planners use, so it runs
on the plant that has actually been tuned (``KALMAN_KP_ATT``,
``KALMAN_KD_ATT``, ``HOVER_THRUST``). It deliberately does *not* use the
controller's velocity-fallback branch, whose ``KP_ATT`` / ``KP_ROLL_ATT``
gains are untuned on this branch.

Guidance chain per tick:
  project onto path -> carrot at lookahead -> desired NED velocity
  -> desired lean (horizontal) + vertical rate -> body rates + thrust
"""
from __future__ import annotations

import math
import time

import numpy as np

import config
from control.pid import PIDConfig, PIDController
from mission import Mission, load_mission
from planning.spline_path import (
    BACK_WINDOW,
    FWD_WINDOW,
    build_spline_path,
    path_curvature,
    speed_profile,
)


def _f(value, default=None):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class SplinePlanner:
    """Follow a captured spline using the EKF's derived position."""

    name = 'spline_derived'

    def __init__(self, mission_path: str | Mission | None = None):
        if isinstance(mission_path, Mission):
            self.mission = mission_path
            path = getattr(mission_path, 'name', 'mission') or 'mission'
        else:
            path = mission_path or config.SPLINE_MISSION_PATH
            # load_mission returns None for a missing or unreadable file.
            self.mission = load_mission(path)
            if self.mission is None:
                raise ValueError(
                    f'no usable mission at {path!r}. Capture one first: '
                    'tools/tune_flight.py pilot --capture'
                )
        positions = np.array(
            [w.pos for w in self.mission.waypoints], dtype=np.float64
        )
        if len(positions) < 2:
            raise ValueError(
                f'{path}: need at least 2 waypoints, got {len(positions)}'
            )
        self._pts, self._cum_s, self._wp_s = build_spline_path(positions)
        self._curv = path_curvature(self._pts, self._cum_s)
        self._speed = speed_profile(
            self._curv,
            self._cum_s,
            cruise=config.SPLINE_CRUISE_MPS,
            a_lat=config.SPLINE_A_LAT,
            a_lon=config.SPLINE_A_LON,
            end_speed=config.SPLINE_FINISH_MPS,
        )
        self._s_end = float(self._cum_s[-1])
        self._max_lean = math.radians(config.SPLINE_MAX_LEAN_DEG)
        max_rate = config.KALMAN_MAX_RATE_RAD_S
        self._roll_pid = PIDController(PIDConfig(
            kp=config.KALMAN_KP_ATT, kd=config.KALMAN_KD_ATT,
            output_min=-max_rate, output_max=max_rate,
        ))
        self._pitch_pid = PIDController(PIDConfig(
            kp=config.KALMAN_KP_ATT, kd=config.KALMAN_KD_ATT,
            output_min=-max_rate, output_max=max_rate,
        ))
        self._vert_pid = PIDController(PIDConfig(
            kp=config.KP_THRUST_VEL, ki=config.KI_THRUST_VEL,
            output_min=-config.SPLINE_VERT_AUTH,
            output_max=config.SPLINE_VERT_AUTH,
            integral_min=-config.THRUST_INTEGRAL_LIMIT,
            integral_max=config.THRUST_INTEGRAL_LIMIT,
        ))
        self.reset_episode()
        print(
            f'[SPLINE] {path}: {len(positions)} waypoints, '
            f'{self._s_end:.1f} m path, cruise {config.SPLINE_CRUISE_MPS:.1f} m/s, '
            f'EKF_USE_PNP={int(bool(config.EKF_USE_PNP))}',
            flush=True,
        )

    # ------------------------------------------------------------------
    def reset_episode(self) -> None:
        self._i = 0
        self._last_t = None
        self._armed_at = None
        self._finished = False
        self._roll_pid.reset()
        self._pitch_pid.reset()
        self._vert_pid.reset()

    # ------------------------------------------------------------------
    def _project(self, pos):
        """Closest path sample searched in a window around the last progress."""
        lo = max(0, self._i - BACK_WINDOW)
        hi = min(len(self._pts), self._i + FWD_WINDOW + 1)
        d = np.linalg.norm(self._pts[lo:hi] - pos, axis=1)
        k = lo + int(np.argmin(d))
        self._i = k
        return float(self._cum_s[k]), k, float(d[k - lo])

    def _point_at_s(self, s: float):
        s = min(max(s, 0.0), self._s_end)
        j = int(np.searchsorted(self._cum_s, s))
        if j <= 0:
            return self._pts[0]
        if j >= len(self._pts):
            return self._pts[-1]
        s0, s1 = self._cum_s[j - 1], self._cum_s[j]
        span = max(s1 - s0, 1e-9)
        w = (s - s0) / span
        return self._pts[j - 1] * (1.0 - w) + self._pts[j] * w

    def _hover(self, shared_data, reason: str, yaw_rate: float = 0.0):
        self._roll_pid.reset()
        self._pitch_pid.reset()
        shared_data['spline'] = {'phase': 'hover', 'reason': reason}
        return {
            'kalman': True,
            'roll_rate': 0.0,
            'pitch_rate': 0.0,
            'yaw_rate': float(yaw_rate),
            'thrust': float(config.HOVER_THRUST),
            'desired_roll': 0.0,
            'desired_pitch': 0.0,
            'source': 'spline_hover',
        }

    # ------------------------------------------------------------------
    def compute_target(self, shared_data, dt=None):
        """One guidance tick.

        ``dt`` defaults to the wall-clock interval since the last call, which
        is what main.py wants. Pass it explicitly to drive the loop from a
        simulated clock — without that the derivative term sees a real-time
        interval while the plant advances by a fixed step, which saturates the
        rate clamp and makes the loop untestable offline.
        """
        now = time.monotonic()
        if dt is None:
            dt = (
                1.0 / config.CONTROL_HZ
                if self._last_t is None
                else max(1e-3, now - self._last_t)
            )
        dt = max(1e-3, float(dt))
        self._last_t = now
        if self._armed_at is None:
            self._armed_at = now

        pos_entry = shared_data.get('position_ned') or {}
        att = shared_data.get('attitude') or {}
        px, py, pz = (_f(pos_entry.get(k)) for k in ('x', 'y', 'z'))
        roll = _f(att.get('roll'), 0.0)
        pitch = _f(att.get('pitch'), 0.0)
        yaw = _f(att.get('yaw'))
        if px is None or py is None or pz is None or yaw is None:
            return self._hover(shared_data, 'no_derived_pose')

        pos = np.array([px, py, pz], dtype=np.float64)

        # Altitude guard on the derived estimate — it is all we have.
        if -pz > config.SPLINE_MAX_ALT_M:
            return self._hover(shared_data, 'alt_guard')

        s_proj, idx, cross_track = self._project(pos)

        # Finish first: running off the end of the path is a normal terminal
        # condition and must take precedence over the geometry guard. Checking
        # xte first let an overshoot past the last sample trip xte_guard
        # instead of finishing.
        at_last_sample = idx >= len(self._pts) - 1
        if at_last_sample or s_proj >= self._s_end - config.SPLINE_FINISH_TOL_M:
            self._finished = True
            return self._hover(shared_data, 'finished')

        if cross_track > config.SPLINE_MAX_XTE_M:
            return self._hover(shared_data, 'xte_guard')

        target_speed = float(self._speed[idx])
        lookahead = float(
            np.clip(
                config.SPLINE_LOOKAHEAD_M
                + config.SPLINE_LOOKAHEAD_TIME_S * target_speed,
                config.SPLINE_LOOKAHEAD_M,
                config.SPLINE_LOOKAHEAD_MAX_M,
            )
        )
        s_carrot = s_proj + lookahead

        carrot = self._point_at_s(s_carrot)
        to_carrot = carrot - pos
        dist = float(np.linalg.norm(to_carrot))
        if dist < 1e-6:
            return self._hover(shared_data, 'degenerate_carrot')
        unit = to_carrot / dist

        # ---- yaw first: face along the PATH tangent (not carrot-from-pos,
        # which oscillates when you are off to the side). ----
        tangent = (
            self._point_at_s(s_proj + config.SPLINE_YAW_LOOKAHEAD_M)
            - self._point_at_s(s_proj)
        )
        if float(np.hypot(tangent[0], tangent[1])) > 0.25:
            yaw_want = math.atan2(tangent[1], tangent[0])
        else:
            # Degenerate short segment — fall back to carrot bearing.
            yaw_want = math.atan2(to_carrot[1], to_carrot[0])
        yaw_err = _wrap(yaw_want - yaw)
        yaw_rate = float(np.clip(
            config.SPLINE_KP_YAW * yaw_err,
            -config.YAW_RATE_MAX_RAD_S, config.YAW_RATE_MAX_RAD_S,
        ))

        # Nose-first only. Large heading error → turn in place (no pitch/roll),
        # otherwise a carrot behind the nose commands reverse tip ("backward").
        align_lim = math.radians(
            float(getattr(config, 'SPLINE_YAW_ALIGN_DEG', 35.0))
        )
        align = float(np.clip(
            1.0 - abs(yaw_err) / max(align_lim, 1e-3),
            0.0, 1.0,
        ))
        speed_cmd = target_speed * align
        vel_cmd = unit * speed_cmd

        # ---- horizontal: NED velocity error -> desired lean ----
        vel_entry = shared_data.get('ekf_state') or {}
        vel = vel_entry.get('velocity_ned') or [0.0, 0.0, 0.0]
        v_ned = np.array([_f(v, 0.0) for v in vel[:3]], dtype=np.float64)

        cy, sy = math.cos(yaw), math.sin(yaw)
        err_n = vel_cmd[0] - v_ned[0]
        err_e = vel_cmd[1] - v_ned[1]
        err_fwd = err_n * cy + err_e * sy
        err_right = -err_n * sy + err_e * cy

        des_pitch = float(np.clip(
            config.SPLINE_KP_VEL_LEAN * err_fwd * config.FORWARD_PITCH_SIGN,
            -self._max_lean, self._max_lean,
        )) * align
        des_roll = float(np.clip(
            config.SPLINE_KP_VEL_LEAN * err_right * config.LATERAL_LEAN_SIGN,
            -self._max_lean, self._max_lean,
        )) * align
        if align < 1e-3:
            # Hard turn-in-place: level while yaw catches up.
            des_pitch = 0.0
            des_roll = 0.0
            roll_rate = float(self._roll_pid.update(0.0 - roll, dt))
            pitch_rate = float(self._pitch_pid.update(0.0 - pitch, dt))
        else:
            roll_rate = float(self._roll_pid.update(des_roll - roll, dt))
            pitch_rate = float(self._pitch_pid.update(des_pitch - pitch, dt))

        # ---- vertical: rate loop on the path's descent rate ----
        vd_cmd = float(vel_cmd[2])             # NED down-positive
        vz_meas = _f(v_ned[2], 0.0)
        tilt = max(0.88, math.cos(abs(des_pitch)) * math.cos(abs(des_roll)))
        thrust = config.HOVER_THRUST / tilt
        thrust += float(self._vert_pid.update(vz_meas - vd_cmd, dt))
        thrust = float(np.clip(thrust, config.MIN_THRUST, config.MAX_THRUST))

        shared_data['spline'] = {
            'phase': 'yaw_align' if align < 1e-3 else 'track',
            's': s_proj,
            's_end': self._s_end,
            'progress': s_proj / max(self._s_end, 1e-9),
            'idx': idx,
            'cross_track_m': cross_track,
            'target_speed_mps': speed_cmd,
            'lookahead_m': lookahead,
            'des_pitch': des_pitch,
            'des_roll': des_roll,
            'yaw_err_rad': yaw_err,
            'yaw_align': align,
            'vd_cmd': vd_cmd,
            'vz_meas': vz_meas,
            'thrust': thrust,
        }
        return {
            'kalman': True,
            'roll_rate': roll_rate,
            'pitch_rate': pitch_rate,
            'yaw_rate': yaw_rate,
            'thrust': thrust,
            'desired_roll': des_roll,
            'desired_pitch': des_pitch,
            'source': 'spline',
        }
