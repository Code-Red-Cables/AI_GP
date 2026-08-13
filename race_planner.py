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


def _tilt_compensate(thrust: float, roll: float, pitch: float) -> float:
    """Keep vertical lift constant while leaned.

    Only the component of thrust along the vertical holds the drone up, so a
    level-flight collective sinks as soon as the craft pitches to drive forward:
    at 35 degrees only 82% of it is lifting. Without this the planner trades
    altitude for speed and settles onto the floor.
    """
    if not bool(getattr(config, 'RACE_TILT_COMPENSATE', True)):
        return float(thrust)
    cos_tilt = math.cos(float(roll)) * math.cos(float(pitch))
    floor = float(getattr(config, 'MIN_TILT_COMPENSATION_COSINE', 0.70) or 0.70)
    cos_tilt = max(max(0.55, min(0.95, floor)), cos_tilt)
    return float(thrust) / cos_tilt


class _HeldPose:
    """A dead-reckoned gate pose, shaped like GateLSPose for the controller."""

    __slots__ = ('lateral_m', 'vertical_m', 'through_m', 'range_m',
                 'bearing_rad', 'residual_m', 'ring_disagree_m',
                 'body_forward_range', 'age_s', 'held')

    def __init__(self, d: dict):
        self.lateral_m = float(d['lateral_m'])
        self.vertical_m = float(d['vertical_m'])
        self.through_m = float(d['through_m'])
        self.range_m = float(d['range_m'])
        self.bearing_rad = math.atan2(
            self.lateral_m, max(-self.through_m, 1e-6)
        )
        self.residual_m = 0.0
        self.ring_disagree_m = 0.0
        self.body_forward_range = abs(self.through_m)
        self.age_s = float(d.get('age_s', 0.0))
        self.held = True


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
        # Last solved gate pose, dead-reckoned forward while vision is blind.
        self._held_pose: Optional[dict] = None
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
        self._held_pose = None
        self._roll_i = 0.0

    def _attitude(self, shared_data: dict) -> tuple[float, float, float]:
        """Roll/pitch/yaw, gravity-referenced sources first.

        ``shared_data['attitude']`` is the EKF's integrated-gyro belief, which
        this repo measured drifting +4 deg to -23 deg over 50 s. The LS gate
        pose de-rotates the corner rays by this attitude, so feeding it drift
        tilts every solved gate position. Prefer the controller's AHRS, then the
        sim's own ATTITUDE, and fall back to the EKF only if neither exists.
        """
        ctrl = shared_data.get('control_output') or {}
        raw = shared_data.get('attitude_raw') or {}
        att = shared_data.get('attitude') or {}
        roll = _num(ctrl.get('ahrs_roll'), math.nan)
        pitch = _num(ctrl.get('ahrs_pitch'), math.nan)
        if not (math.isfinite(roll) and math.isfinite(pitch)):
            roll = _num(raw.get('roll'), math.nan)
            pitch = _num(raw.get('pitch'), math.nan)
        if not (math.isfinite(roll) and math.isfinite(pitch)):
            roll, pitch = _num(att.get('roll')), _num(att.get('pitch'))
        # Yaw has no gravity reference; the sim's own value beats the EKF's.
        yaw = _num(raw.get('yaw'), math.nan)
        if not math.isfinite(yaw):
            yaw = _num(att.get('yaw'))
        return roll, pitch, yaw

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

    def _predict_ekf(
        self, shared_data: dict, roll: float, pitch: float, yaw: float
    ) -> tuple[float, float]:
        """Advance the drag EKF; return body (forward, right) velocity.

        Lateral velocity is the paper's damping term in eq. 22, recovered from
        specific force via the drag model (eq. 14) rather than differentiated
        from a noisy position.
        """
        accel, gyro = self._imu(shared_data)
        now = time.monotonic()
        dt = 0.01 if self._last_imu_t is None else max(1e-3, min(0.05, now - self._last_imu_t))
        self._last_imu_t = now
        self._ekf.predict(accel, gyro, roll, pitch, yaw, dt)
        v_xy = self._ekf.body_velocity_xy(accel)
        return float(v_xy[0]), float(v_xy[1])

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

    def _propagate_pose(self, v_fwd: float, v_lat: float, dt: float):
        """Dead-reckon the last gate pose while the gate is out of sight.

        Detection is strongly pitch-dependent -- measured 65% at level falling
        to 31% beyond 40 degrees of pitch, because the camera is tilted up 20
        degrees and hard forward flight aims it below the horizon. So the fast
        flight this planner is trying to achieve is exactly when it goes blind,
        for stretches of several seconds.

        The paper's answer is its Kalman filter: it propagates the drone's
        position on drag-model velocity and lets a new detection correct the
        drift (their Figure 22 shows the jump when vision returns). This does the
        same on the gate-relative pose, so control continues on a decaying
        estimate rather than dropping to a hold the instant a corner is lost.
        """
        held = self._held_pose
        if held is None:
            return None
        # Camera moves forward and right; the gate's relative position moves the
        # opposite way in the gate frame.
        held['lateral_m'] -= float(v_lat) * dt
        held['through_m'] += float(v_fwd) * dt
        held['range_m'] = max(
            0.1, math.hypot(held['lateral_m'], abs(held['through_m']))
        )
        held['age_s'] += dt
        return held

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

    def _begin_arc(self, shared_data: dict, yaw: float) -> bool:
        """Start a feed-forward arc. Returns False when the map cannot say how.

        Without a surveyed course map every gate defaults to a 90 degree right
        turn, and committing to that blind is worse than flying straight: on a
        left-hand gate it drives into the scenery. The caller falls back to a
        hold-and-search when this returns False.
        """
        race = shared_data.get('race_status') or {}
        try:
            gid = int(race.get('active_gate') or 0)
        except (TypeError, ValueError):
            gid = 0
        spec = self._course.get(gid) or self._course.get(gid - 1) or {}
        if not spec or str(spec.get('note', '')).startswith('stub'):
            return False
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
        return True

    def _align_command(
        self,
        pose,
        roll: float,
        pitch: float,
        yaw: float,
        v_fwd: float,
        dt: float,
        v_lat: float = 0.0,
    ) -> dict:
        """Straight-part control, paper eq. 22:

            phi_c = -kp * y - kd * ydot     roll nulls the lateral offset
            theta_c = theta_0               pitch is fixed; it sets the speed
            psi_c = 0                       heading held on the gate

        Three departures from the paper have been removed here.

        The damping term is the *lateral velocity*, taken from the drag EKF
        (paper eq. 14), not the roll angle. Roll is where the controller is
        already pushing, not how fast the drone is sliding, so damping on it fed
        the output back into itself.

        There is no yaw term. The paper fixes heading to the gate's direction
        and steers purely with roll; adding a bearing-driven yaw loop puts a
        second controller on the same error, competing with the roll loop.

        There is no altitude term -- see ``_hold_thrust``.
        """
        y = float(pose.lateral_m)
        kp = float(getattr(config, 'RACE_KP_LAT', 1.0))
        kd = float(getattr(config, 'RACE_KD_LAT', 2.0))
        phi_c = -kp * y - kd * float(v_lat)
        max_lean = math.radians(float(getattr(config, 'RACE_MAX_LEAN_DEG', 12.0)))
        phi_c = _clamp(phi_c, -max_lean, max_lean)

        theta_c = math.radians(float(getattr(config, 'RACE_PITCH_DEG', 20.0)))
        thrust = self._hold_thrust(phi_c, theta_c)
        return self._angles_to_target(phi_c, theta_c, 0.0, thrust, roll, pitch, dt)

    def _hold_thrust(self, phi_c: float, theta_c: float) -> float:
        """Collective for level flight while leaned (paper eq. 27).

            T = (-g - a_z) / (cos(theta) cos(phi))

        The paper never derives altitude from the gate: "we neglect z because in
        the real-world flight, the altitude is controlled by a separate
        controller which can keep the altitude to be a constant" -- a sonar, in
        their Figure 16c. Driving thrust from the gate's apparent vertical offset
        couples height to a quantity that grows with both range and attitude
        error, and in flight it pinned the collective at maximum for half the run
        and climbed away.
        """
        return _clamp(
            _tilt_compensate(float(config.HOVER_THRUST), phi_c, theta_c),
            float(getattr(config, 'RACE_THRUST_MIN', 0.20)),
            float(getattr(config, 'RACE_THRUST_MAX', 0.40)),
        )

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

    def _search_command(self, roll: float, pitch: float, dt: float) -> dict:
        """Wings level, gentle forward, slow yaw scan — reacquire, do not guess.

        Used when no gate is solved and the course map cannot say which way the
        next one lies.
        """
        # Positive is forward on this plant; keep it gentle while blind.
        theta_c = math.radians(
            float(getattr(config, 'RACE_SEARCH_PITCH_DEG', 6.0))
        )
        yaw_rate = float(getattr(config, 'RACE_SEARCH_YAW_RATE', 0.0))
        thrust = _tilt_compensate(float(config.HOVER_THRUST), 0.0, theta_c)
        return self._angles_to_target(
            0.0, theta_c, yaw_rate, thrust, roll, pitch, dt,
        )

    def compute_target(self, shared_data: dict) -> dict:
        shared_data['planner_mode'] = self.name
        roll, pitch, yaw = self._attitude(shared_data)
        accel, _gyro = self._imu(shared_data)
        v_fwd, v_lat = self._predict_ekf(shared_data, roll, pitch, yaw)
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

        if visible:
            # Fresh solve: reset the dead-reckoned estimate to it.
            self._held_pose = {
                'lateral_m': float(pose.lateral_m),
                'vertical_m': float(pose.vertical_m),
                'through_m': float(pose.through_m),
                'range_m': float(pose.range_m),
                'age_s': 0.0,
            }
        else:
            held = self._propagate_pose(v_fwd, v_lat, dt)
            hold_s = float(getattr(config, 'RACE_POSE_HOLD_S', 2.0))
            if held is not None and held['age_s'] <= hold_s:
                pose = _HeldPose(held)
                visible = True
            else:
                self._held_pose = None

        if self._mode == 'arc':
            target = self._arc_command(roll, pitch, yaw, v_fwd, accel, dt)
        elif visible:
            target = self._align_command(
                pose, roll, pitch, yaw, v_fwd, dt, v_lat=v_lat
            )
            # Commit to arc when very close (histogram regime in the paper).
            commit_m = float(getattr(config, 'RACE_COMMIT_RANGE_M', 1.2))
            if pose is not None and pose.body_forward_range < commit_m:
                if self._begin_arc(shared_data, yaw):
                    target = self._arc_command(
                        roll, pitch, yaw, v_fwd, accel, dt
                    )
        else:
            # No gate solved and no arc running. With a surveyed map the arc
            # carries us to the next gate; without one, guessing a 90 degree
            # turn is how you fly into scenery, so hold and let vision reacquire.
            if self._arc_t0 is None and not self._begin_arc(shared_data, yaw):
                target = self._search_command(roll, pitch, dt)
            else:
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
