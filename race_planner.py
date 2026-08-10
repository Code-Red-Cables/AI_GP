"""Li & de Croon classical race planner: PD align + feed-forward arc.

``FLIGHT_MODE=race``. When a gate is in view, bank to null the lateral LS
offset (paper eq. 22). When the gate leaves the frame or has just been
passed, fly a coordinated feed-forward arc (paper eq. 29) for up to
``RACE_ARC_MAX_S`` using drag-model forward speed.

Drop-in for AssistImagePlanner: exposes ``.name`` and
``compute_target(shared_data)``, writes the same ``planner_target`` contract.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

import config
from ekf.drag_ekf import DragEKF
from vision.gate_ls_pose import solve_keypoints_ls

G = 9.80665
DEFAULT_COURSE_MAP = os.environ.get('RACE_COURSE_MAP', 'course_map.json')


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class RacePlanner:
    """PD gate alignment with a short feed-forward arc between gates."""

    def __init__(self, course_map_path: str | None = None):
        self.name = 'race(li-decroon)'
        self._ekf = DragEKF(
            k_x=float(getattr(config, 'DRAG_KX', -0.5)),
            k_y=float(getattr(config, 'DRAG_KY', -0.5)),
        )
        self._course = self._load_course(course_map_path or DEFAULT_COURSE_MAP)
        self._mode = 'align'  # align | arc
        self._arc_t0: Optional[float] = None
        self._arc_psi = 0.0
        self._arc_heading0 = 0.0
        self._arc_radius = float(getattr(config, 'RACE_ARC_RADIUS_M', 1.5))
        self._arc_turn_rad = float(getattr(config, 'RACE_ARC_TURN_RAD', math.radians(90.0)))
        self._last_gate: Optional[int] = None
        self._last_imu_t: Optional[float] = None
        self._last_pose_t: Optional[float] = None
        self._roll_i = 0.0
        print(f'[RACE] course_map={len(self._course)} gates, '
              f'k_x={self._ekf.k_x:.3f} k_y={self._ekf.k_y:.3f}', flush=True)

    @staticmethod
    def _load_course(path: str) -> dict[int, dict]:
        p = Path(path)
        if not p.is_file():
            return {}
        try:
            raw = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        out: dict[int, dict] = {}
        gates = raw.get('gates', raw) if isinstance(raw, dict) else {}
        if isinstance(gates, list):
            for item in gates:
                try:
                    gid = int(item['id'])
                except (KeyError, TypeError, ValueError):
                    continue
                out[gid] = item
        elif isinstance(gates, dict):
            for key, item in gates.items():
                try:
                    out[int(key)] = item
                except (TypeError, ValueError):
                    continue
        return out

    def reset_episode(self) -> None:
        self._ekf.reset()
        self._mode = 'align'
        self._arc_t0 = None
        self._arc_psi = 0.0
        self._last_gate = None
        self._last_imu_t = None
        self._last_pose_t = None
        self._roll_i = 0.0

    def _attitude(self, shared_data: dict) -> tuple[float, float, float]:
        att = shared_data.get('attitude') or {}
        return (
            _num(att.get('roll')),
            _num(att.get('pitch')),
            _num(att.get('yaw')),
        )

    def _imu(self, shared_data: dict) -> tuple[np.ndarray, np.ndarray]:
        imu = shared_data.get('highres_imu') or {}
        accel = np.array([
            _num(imu.get('xacc')),
            _num(imu.get('yacc')),
            _num(imu.get('zacc'), -G),
        ], dtype=np.float64)
        gyro = np.array([
            _num(imu.get('xgyro')),
            _num(imu.get('ygyro')),
            _num(imu.get('zgyro')),
        ], dtype=np.float64)
        return accel, gyro

    def _predict_ekf(self, shared_data: dict, roll: float, pitch: float, yaw: float) -> float:
        accel, gyro = self._imu(shared_data)
        now = time.monotonic()
        dt = 0.01 if self._last_imu_t is None else max(1e-3, min(0.05, now - self._last_imu_t))
        self._last_imu_t = now
        self._ekf.predict(accel, gyro, roll, pitch, yaw, dt)
        return self._ekf.velocity_body_forward(accel)

    def _solve_pose(self, shared_data: dict, roll: float, pitch: float, yaw: float):
        gate = shared_data.get('gate_detection') or {}
        kps = gate.get('keypoints') or gate.get('keypoints_px')
        if kps is None:
            # Fall back to dual-gate / PnP payload if present.
            dual = shared_data.get('dual_gate_pnp') or {}
            kps = dual.get('keypoints_px')
            conf = dual.get('keypoint_confidences')
        else:
            conf = gate.get('keypoint_confidences') or gate.get('keypoints_conf')
        if kps is None:
            return None
        try:
            pts = np.asarray(kps, dtype=np.float64).reshape(-1, 2)
        except (TypeError, ValueError):
            return None
        if pts.shape[0] != 8:
            return None
        pose = solve_keypoints_ls(
            pts,
            conf,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            min_keypoint_confidence=float(
                getattr(config, 'RACE_MIN_KP_CONF', 0.25)
            ),
        )
        if pose is not None:
            self._last_pose_t = time.monotonic()
        return pose

    def _gate_visible(self, shared_data: dict, pose) -> bool:
        if pose is None:
            return False
        max_age = float(getattr(config, 'RACE_POSE_MAX_AGE_S', 0.25))
        if self._last_pose_t is None:
            return False
        if time.monotonic() - self._last_pose_t > max_age:
            return False
        # Range only as a coarse gate: ignore absurd solves.
        if pose.range_m > float(getattr(config, 'RACE_MAX_RANGE_M', 40.0)):
            return False
        if pose.residual_m > float(getattr(config, 'RACE_MAX_RESIDUAL_M', 0.6)):
            return False
        return True

    def _begin_arc(self, shared_data: dict, yaw: float) -> None:
        race = shared_data.get('race_status') or {}
        try:
            gid = int(race.get('active_gate') or 0)
        except (TypeError, ValueError):
            gid = 0
        spec = self._course.get(gid) or self._course.get(gid - 1) or {}
        self._arc_radius = float(spec.get(
            'turn_radius_m', getattr(config, 'RACE_ARC_RADIUS_M', 1.5)
        ))
        turn_deg = float(spec.get(
            'turn_deg', math.degrees(getattr(config, 'RACE_ARC_TURN_RAD', math.pi / 2))
        ))
        # Positive turn_deg = right (positive yaw in NED).
        self._arc_turn_rad = math.radians(turn_deg)
        self._arc_heading0 = yaw
        self._arc_psi = 0.0
        self._arc_t0 = time.monotonic()
        self._mode = 'arc'
        log = shared_data.get('log_event')
        if callable(log):
            log('RACE', f'arc_start gate={gid} r={self._arc_radius:.2f} '
                f'turn_deg={turn_deg:.1f}')

    def _align_command(
        self,
        pose,
        roll: float,
        pitch: float,
        yaw: float,
        v_fwd: float,
        dt: float,
    ) -> dict:
        # Paper eq. 22: phi_c = -kp y - kd ydot, theta fixed, psi hold.
        y = float(pose.lateral_m)
        # Finite-difference lateral rate in gate frame ≈ body-right when
        # roughly aligned; use drag lateral velocity as a stand-in.
        kp = float(getattr(config, 'RACE_KP_LAT', 0.35))
        kd = float(getattr(config, 'RACE_KD_LAT', 0.15))
        # No direct ydot; damp with current roll as a proxy for lateral rate.
        phi_c = -kp * y - kd * roll
        max_lean = math.radians(float(getattr(config, 'RACE_MAX_LEAN_DEG', 12.0)))
        phi_c = _clamp(phi_c, -max_lean, max_lean)

        theta_c = math.radians(float(getattr(config, 'RACE_PITCH_DEG', -5.0)))
        # Vertical: push thrust from vertical offset (Y down → positive means
        # camera below gate centre → climb).
        hover = float(config.HOVER_THRUST)
        vert_gain = float(getattr(config, 'RACE_VERT_THRUST_GAIN', 0.04))
        thrust = hover + vert_gain * float(pose.vertical_m)
        thrust = _clamp(
            thrust,
            float(getattr(config, 'RACE_THRUST_MIN', 0.20)),
            float(getattr(config, 'RACE_THRUST_MAX', 0.55)),
        )

        # Hold heading; small yaw toward bearing if configured.
        yaw_kp = float(getattr(config, 'RACE_YAW_KP', 0.8))
        yaw_rate = _clamp(yaw_kp * float(pose.bearing_rad), -0.6, 0.6)

        return self._angles_to_target(phi_c, theta_c, yaw_rate, thrust, roll, pitch, dt)

    def _arc_command(
        self,
        roll: float,
        pitch: float,
        yaw: float,
        v_fwd: float,
        accel: np.ndarray,
        dt: float,
    ) -> dict:
        # Paper eq. 29.
        r = max(0.5, abs(self._arc_radius))
        sign = 1.0 if self._arc_turn_rad >= 0.0 else -1.0
        v = max(0.3, abs(v_fwd))
        # Integrate heading change along the arc.
        self._arc_psi += sign * (v / r) * dt

        theta_c = math.radians(float(getattr(config, 'RACE_PITCH_DEG', -5.0)))
        # Specific force leftovers in the body-fixed earth frame: approximate
        # with body accel (drag already embedded).
        a_y = float(accel[1])
        a_z = float(accel[2])
        num = (a_y - sign * (v * v) / r) * math.cos(theta_c)
        den = -G - a_z
        if abs(den) < 1e-3:
            phi_c = sign * math.radians(8.0)
        else:
            phi_c = math.atan2(num, den)
        max_lean = math.radians(float(getattr(config, 'RACE_MAX_LEAN_DEG', 18.0)))
        phi_c = _clamp(phi_c, -max_lean, max_lean)

        # Yaw rate command tracks the arc rate.
        yaw_rate = sign * (v / r)
        yaw_rate = _clamp(yaw_rate, -1.2, 1.2)

        thrust = float(config.HOVER_THRUST) + 0.02
        # End the arc once heading change is done or time budget expires.
        max_s = float(getattr(config, 'RACE_ARC_MAX_S', 2.0))
        elapsed = 0.0 if self._arc_t0 is None else time.monotonic() - self._arc_t0
        if abs(self._arc_psi) >= abs(self._arc_turn_rad) or elapsed >= max_s:
            self._mode = 'align'
            self._arc_t0 = None

        return self._angles_to_target(phi_c, theta_c, yaw_rate, thrust, roll, pitch, dt)

    def _angles_to_target(
        self,
        phi_c: float,
        theta_c: float,
        yaw_rate: float,
        thrust: float,
        roll: float,
        pitch: float,
        dt: float,
    ) -> dict:
        kp = float(getattr(config, 'KP_ATT', 1.8))
        roll_rate = kp * (phi_c - roll)
        pitch_rate = kp * (theta_c - pitch)
        max_rate = float(config.MAX_RATE_RAD_S)
        roll_rate = _clamp(roll_rate, -max_rate, max_rate)
        pitch_rate = _clamp(pitch_rate, -max_rate, max_rate)
        return {
            'vn': 0.0,
            've': 0.0,
            'vd': 0.0,
            'kalman': True,
            'roll_rate': roll_rate,
            'pitch_rate': pitch_rate,
            'yaw_rate': yaw_rate,
            'thrust': thrust,
            'desired_roll': phi_c,
            'desired_pitch': theta_c,
        }

    def compute_target(self, shared_data: dict) -> dict:
        shared_data['planner_mode'] = self.name
        roll, pitch, yaw = self._attitude(shared_data)
        accel, _gyro = self._imu(shared_data)
        v_fwd = self._predict_ekf(shared_data, roll, pitch, yaw)
        dt = 0.01

        race = shared_data.get('race_status') or {}
        try:
            active = int(race.get('active_gate') or 0)
        except (TypeError, ValueError):
            active = 0
        if self._last_gate is not None and active > self._last_gate:
            # Gate just passed — start the feed-forward arc immediately.
            self._begin_arc(shared_data, yaw)
        self._last_gate = active

        pose = self._solve_pose(shared_data, roll, pitch, yaw)
        visible = self._gate_visible(shared_data, pose)

        if self._mode == 'arc':
            target = self._arc_command(roll, pitch, yaw, v_fwd, accel, dt)
        elif visible:
            target = self._align_command(pose, roll, pitch, yaw, v_fwd, dt)
            # Commit to arc when very close (histogram regime in the paper).
            commit_m = float(getattr(config, 'RACE_COMMIT_RANGE_M', 1.2))
            if pose is not None and pose.body_forward_range < commit_m:
                self._begin_arc(shared_data, yaw)
                target = self._arc_command(roll, pitch, yaw, v_fwd, accel, dt)
        else:
            # Blind without an active arc: gentle pitch hold, seek with yaw 0.
            if self._arc_t0 is None:
                self._begin_arc(shared_data, yaw)
            target = self._arc_command(roll, pitch, yaw, v_fwd, accel, dt)

        shared_data['planner_target'] = target
        shared_data['race_pose'] = None if pose is None else {
            'range_m': pose.range_m,
            'lateral_m': pose.lateral_m,
            'vertical_m': pose.vertical_m,
            'bearing_rad': pose.bearing_rad,
            'residual_m': pose.residual_m,
            'ring_disagree_m': pose.ring_disagree_m,
            'mode': self._mode,
        }
        return target
