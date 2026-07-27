"""Visual-inertial state estimator: IMU dead-reckoning + gate-PnP corrections.

WHY THIS EXISTS: the VQ2 sim sends NO ATTITUDE and NO LOCAL_POSITION_NED — the
only telemetry is HIGHRES_IMU (~114 Hz gyro/accel; baro is NaN). The planner and
controller were written against ``shared_data['attitude']`` and
``shared_data['position_ned']``; without them the planner sat in watchdog_hover
forever. This module MANUFACTURES those blackboard entries:

  * Attitude: quaternion complementary filter — gyro integration, accelerometer
    gravity correction for roll/pitch, and absolute roll/pitch + yaw/position
    corrections from gate PnP fixes (vision/yolo_pnp.py) whenever a gate is in
    view. Gates hang upright, so a gate observation fixes roll/pitch absolutely
    and yaw/position relative to that gate.
  * Position/velocity (world NED, origin = arm point, yaw 0 = boot heading):
    accelerometer dead-reckoning between PnP fixes, snapped/blended to the
    vision solution when a gate is visible. Pure-inertial drift is unbounded,
    so a velocity leak keeps DR bounded during (short) blind stretches.

Gate anchoring: the sim broadcasts no gate poses (nulled in VQ2), so the FIRST
good PnP of each gate index anchors that gate's world pose using the current
belief; later fixes of the same gate correct the drone against that anchor.
Gate index comes from race_status.active_gate. Anchors are persisted alongside
the course map so later runs can preplan in a consistent frame.

Frames (matching camera_model.py):
  body = FRD (x fwd, y right, z down);  world = NED;
  gate-NED = N through the gate, E right along the plane, D down, origin centre.
"""
from __future__ import annotations

import json
import math
import threading
import time
from typing import Optional

import numpy as np

import camera_model as cm

G = 9.80665
G_NED = np.array([0.0, 0.0, G])

# gate frame (X right, Y down, Z through) -> gate-NED (N through, E right, D down)
P_GATE2NED = np.array([[0.0, 0.0, 1.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0]])

# ---- complementary-filter / fusion gains -------------------------------------
ACC_TILT_GAIN = 0.02       # per-sample accel roll/pitch pull (only when |a|~g)
ACC_NORM_TOL = 2.5         # m/s^2 window around g where accel tilt is trusted
FIX_RP_GAIN = 0.5          # per-fix absolute roll/pitch blend from PnP
FIX_YAW_GAIN = 0.3         # per-fix yaw blend toward the gate-anchored yaw
FIX_POS_GAIN = 0.35        # per-fix position blend
FIX_VEL_GAIN = 0.25        # per-fix velocity blend (finite-diff of fixes)
# Finite-diff velocity needs a real time baseline: at 33 ms between fixes, a
# 0.45 m corner-jitter wobble reads as 13.6 m/s (measured in run_1785180116 —
# the belief kicked to -3.4 m/s while the drone was PARKED, and the controller
# fought that phantom from the first armed tick). Difference over >=100 ms and
# clamp the implied speed to something a racing quad can actually do.
FIX_VEL_MIN_DT_S = 0.10    # min baseline between the fixes being differenced
FIX_VEL_MAX_DT_S = 0.50    # beyond this the pair spans too much maneuvering
FIX_VEL_CLAMP = 8.0        # m/s cap on any fix-derived velocity component
VEL_LEAK = 0.4             # 1/s exponential decay of DR velocity (bounds drift)
FIX_STALE_S = 0.5          # fixes older than this are ignored
# Blind-flight damping (added after run_1785180116: the drone tumbled, vision
# never re-acquired, and pure-IMU integration walked the position belief to
# z=+1170 m over 95 s). With no fix recently, the velocity belief decays hard
# toward zero so the controller degrades to a calm attitude-hold hover instead
# of chasing integration ghosts at max thrust.
BLIND_AFTER_S = 1.0        # no fix for this long -> "blind" regime
BLIND_VEL_LEAK = 2.5       # 1/s velocity decay while blind (tau = 0.4 s)
POS_SNAP_M = 3.0           # 2 consecutive self-consistent fixes farther than
FIX_AGREE_M = 1.0          #   this from belief -> snap (DR was lost)
INNOV_REJECT_M = 8.0       # single-fix innovation beyond this is rejected
RP_REJECT_RAD = math.radians(60.0)   # PnP fix implying >60deg tilt is a bad solve
# Anchor hygiene (added after run 3: a phantom "gate 1" was anchored from a
# blind-drifted belief the moment the flickery race counter ticked over, and
# every later REAL fix failed the innovation gate against it -> permanent
# blindness while the drone sat crashed).
GATE_IDX_SETTLE_S = 0.5    # a NEW race gate index must persist this long first
REANCHOR_AFTER_REJECTS = 12   # consecutive rejected fixes -> drop + re-anchor
                              # the gate from the current belief (the WORLD frame
                              # absorbs the accumulated drift; guidance is
                              # relative to the gate, so local consistency wins)
REANCHOR_AGREE_M = 1.5        # ...but ONLY if the last two rejected gate-relative
                              # measurements agree with each other (run 5: churning
                              # re-anchors while tumbling created a fresh garbage
                              # anchor every 0.4 s and dragged the frame around)


def _rz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rpy_from_R(R: np.ndarray) -> tuple:
    """ZYX euler (roll, pitch, yaw) from a world<-body rotation matrix."""
    pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class StateEstimator:
    """Owns shared_data['attitude'] and shared_data['position_ned']."""

    def __init__(self, data: dict, anchors_path: str = "gate_anchors.json"):
        self.data = data
        self.anchors_path = anchors_path
        # attitude belief (world<-body) as roll/pitch/yaw + last gyro rates
        self._roll = 0.0
        self._pitch = 0.0
        self._yaw = 0.0
        self._rates = np.zeros(3)
        # translational belief
        self._pos = np.zeros(3)
        self._vel = np.zeros(3)
        # gate anchors: idx -> {'yaw': psi_g (world yaw of through-axis), 'pos': [n,e,d]}
        self.anchors: dict[int, dict] = {}
        self._load_anchors()
        # bookkeeping
        self._last_imu_us: Optional[int] = None
        self._last_fix_ts = 0
        self._last_fix_wall = 0.0   # monotonic time of last ACCEPTED fix
        self._vel_ref: Optional[dict] = None   # older fix used as the velocity baseline
        self._prev_fix: Optional[dict] = None   # for velocity + snap logic
        self._n_imu = 0
        self._n_fix = 0
        self._n_fix_rej = 0
        # settled race gate index (see GATE_IDX_SETTLE_S); hwm ignores downward flicker
        self._gate_idx = 0
        self._gate_idx_raw = 0
        self._gate_idx_since = 0.0
        self._consec_rej = 0
        self._last_rej_pgb: Optional[np.ndarray] = None   # gate-relative pos of last reject
        self.is_running = True
        self.thread = threading.Thread(target=self._loop, daemon=False)
        self.thread.start()

    # ------------------------------------------------------------------
    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    def _load_anchors(self):
        try:
            with open(self.anchors_path) as f:
                raw = json.load(f)
            self.anchors = {int(k): v for k, v in raw.items()}
            print(f"[vio] gate anchors loaded: {sorted(self.anchors)}", flush=True)
        except (OSError, ValueError):
            self.anchors = {}

    def save_anchors(self):
        try:
            with open(self.anchors_path, "w") as f:
                json.dump(self.anchors, f, indent=1)
        except OSError:
            pass

    # ------------------------------------------------------------------
    def _loop(self):
        last_pub = 0.0
        while self.is_running:
            with self.data['lock']:
                imu = self.data.get('highres_imu')
                fix = self.data.get('pnp_fix')
                race = self.data.get('race_status')
            if imu is not None and imu.get('ts_us') != self._last_imu_us:
                self._propagate(imu)
            self._settle_gate_idx(race)
            if fix is not None and fix['ts'] != self._last_fix_ts:
                self._last_fix_ts = fix['ts']
                if time.time_ns() - fix['ts'] < FIX_STALE_S * 1e9:
                    self._apply_fix(fix, self._gate_idx)
            now = time.monotonic()
            if now - last_pub >= 0.005:
                self._publish()
                last_pub = now
            time.sleep(0.002)

    # ------------------------------------------------------------------
    # IMU propagation (complementary filter + accel dead-reckoning)
    # ------------------------------------------------------------------
    def _propagate(self, imu: dict):
        ts_us = imu['ts_us']
        if self._last_imu_us is None:
            self._last_imu_us = ts_us
            return
        dt = (ts_us - self._last_imu_us) * 1e-6
        self._last_imu_us = ts_us
        if not (0.0 < dt < 0.1):        # reset/rollover glitch — skip this sample
            return
        self._n_imu += 1

        p, q, r = imu['xgyro'], imu['ygyro'], imu['zgyro']    # body rates, rad/s
        self._rates = np.array([p, q, r])
        # body rates -> euler-angle rates (standard aerospace kinematics)
        sr, cr = math.sin(self._roll), math.cos(self._roll)
        tp = math.tan(self._pitch)
        cp = max(0.2, math.cos(self._pitch))    # keep kinematics sane near +/-90
        self._roll += (p + sr * tp * q + cr * tp * r) * dt
        self._pitch += (cr * q - sr * r) * dt
        self._yaw = _wrap(self._yaw + ((sr / cp) * q + (cr / cp) * r) * dt)

        acc = np.array([imu['xacc'], imu['yacc'], imu['zacc']])   # specific force
        anorm = float(np.linalg.norm(acc))
        if abs(anorm - G) < ACC_NORM_TOL:
            # low dynamics: accel ~ -gravity in body frame -> absolute tilt
            roll_acc = math.atan2(-acc[1], -acc[2])
            pitch_acc = math.asin(max(-1.0, min(1.0, acc[0] / anorm)))
            self._roll += ACC_TILT_GAIN * _wrap(roll_acc - self._roll)
            self._pitch += ACC_TILT_GAIN * _wrap(pitch_acc - self._pitch)

        # Pre-flight ZUPT: until the client actually arms, the drone is parked on
        # its spawn point — hold velocity/position at the origin and let only the
        # attitude converge. (run_1785180116: DR ran through the ~40 s YOLO model
        # load, so the first flight tick already believed a 2 m/s phantom fall and
        # slammed the thrust to 0.9.)
        if not self.data.get('flight_started'):
            self._vel[:] = 0.0
            self._pos[:] = 0.0
            return

        # dead-reckon velocity/position in NED; leak bounds pure-inertial drift.
        # While BLIND (no recent PnP fix) the leak is much stronger — a wrong
        # velocity belief is worse than a conservative one (see BLIND_VEL_LEAK).
        R_wb = cm.rot_world_body(self._roll, self._pitch, self._yaw)
        a_ned = R_wb @ acc + G_NED
        leak = (VEL_LEAK if time.monotonic() - self._last_fix_wall < BLIND_AFTER_S
                else BLIND_VEL_LEAK)
        self._vel += a_ned * dt
        self._vel *= max(0.0, 1.0 - leak * dt)
        self._pos += self._vel * dt

    # ------------------------------------------------------------------
    def _settle_gate_idx(self, race: Optional[dict]):
        """Adopt a new race gate index only after it persists GATE_IDX_SETTLE_S.

        The sim's active_gate telemetry flickers (measured on the Dreamer
        branch); a single flickered reading must not anchor a phantom gate.
        Downward readings are ignored outright (the race counter is monotonic
        within a run).
        """
        if race is None:
            return
        raw = int(race.get('active_gate', 0))
        now = time.monotonic()
        if raw != self._gate_idx_raw:
            self._gate_idx_raw = raw
            self._gate_idx_since = now
            return
        if (raw > self._gate_idx
                and now - self._gate_idx_since >= GATE_IDX_SETTLE_S):
            self._gate_idx = raw
            self._consec_rej = 0
            with self.data['lock']:
                self.data['gate_idx_settled'] = raw
            print(f"[vio] active gate -> {raw}", flush=True)

    # ------------------------------------------------------------------
    # PnP fix fusion
    # ------------------------------------------------------------------
    def _apply_fix(self, fix: dict, gate_idx: int):
        """fix = {'ts', 'R_cg' (3x3 list), 't_cg' (3,), 'reproj_err_px'}"""
        R_cg = np.asarray(fix['R_cg'], float)
        t_cg = np.asarray(fix['t_cg'], float)
        # body pose in gate-NED
        R_gb = P_GATE2NED @ R_cg.T @ cm.R_CB        # gate-NED <- body
        p_gb = P_GATE2NED @ (-R_cg.T @ t_cg)        # body origin in gate-NED
        roll_m, pitch_m, yaw_g = _rpy_from_R(R_gb)  # yaw_g: heading rel. to gate
        if abs(roll_m) > RP_REJECT_RAD or abs(pitch_m) > RP_REJECT_RAD:
            self._n_fix_rej += 1
            return

        # A poisoned anchor (created from a drifted belief) rejects every real
        # fix forever. If that happens — AND the rejected measurements are
        # SELF-consistent (a stable view of a real gate, not tumbling noise) —
        # drop the anchor and re-anchor from the current belief: the world
        # frame absorbs the drift, local guidance recovers.
        if self._consec_rej >= REANCHOR_AFTER_REJECTS and gate_idx in self.anchors:
            stable = (self._last_rej_pgb is not None
                      and float(np.linalg.norm(p_gb - self._last_rej_pgb)) < REANCHOR_AGREE_M)
            if stable:
                print(f"[vio] gate {gate_idx} anchor rejected {self._consec_rej} fixes"
                      f" in a row — re-anchoring", flush=True)
                del self.anchors[gate_idx]
                self._consec_rej = 0

        anchor = self.anchors.get(gate_idx)
        if anchor is None:
            # First sight of this gate: anchor it in the world using the belief.
            psi_g = _wrap(self._yaw - yaw_g)
            p_wg = self._pos - _rz(psi_g) @ p_gb
            self.anchors[gate_idx] = {'yaw': psi_g, 'pos': p_wg.tolist()}
            print(f"[vio] anchored gate {gate_idx} at "
                  f"({p_wg[0]:+.1f},{p_wg[1]:+.1f},{p_wg[2]:+.1f}) "
                  f"yaw={math.degrees(psi_g):+.0f}deg", flush=True)
            anchor = self.anchors[gate_idx]

        psi_g = float(anchor['yaw'])
        p_wg = np.asarray(anchor['pos'], float)
        yaw_m = _wrap(psi_g + yaw_g)
        pos_m = p_wg + _rz(psi_g) @ p_gb

        innov = float(np.linalg.norm(pos_m - self._pos))
        if innov > INNOV_REJECT_M:
            self._n_fix_rej += 1
            self._consec_rej += 1
            self._last_rej_pgb = p_gb.copy()
            self._prev_fix = None
            return

        # DR-lost recovery: two consecutive fixes that agree with each other but
        # not with the belief mean vision is right and dead-reckoning walked off.
        snap = False
        if innov > POS_SNAP_M:
            if (self._prev_fix is not None
                    and np.linalg.norm(pos_m - self._prev_fix['pos']) < FIX_AGREE_M
                    and fix['ts'] - self._prev_fix['ts'] < 0.3e9):
                snap = True
            else:
                self._prev_fix = {'ts': fix['ts'], 'pos': pos_m}
                return

        # attitude correction (roll/pitch absolute — gates hang upright)
        self._roll += FIX_RP_GAIN * _wrap(roll_m - self._roll)
        self._pitch += FIX_RP_GAIN * _wrap(pitch_m - self._pitch)
        self._yaw = _wrap(self._yaw + FIX_YAW_GAIN * _wrap(yaw_m - self._yaw))

        if snap:
            self._pos = pos_m.copy()
            self._vel[:] = 0.0
            self._vel_ref = None
        else:
            self._pos += FIX_POS_GAIN * (pos_m - self._pos)
            # Fix-derived velocity over a >=FIX_VEL_MIN_DT_S baseline (see above:
            # consecutive-frame differencing amplified corner jitter into m/s).
            if self._vel_ref is not None:
                dt = (fix['ts'] - self._vel_ref['ts']) * 1e-9
                if dt > FIX_VEL_MAX_DT_S:
                    self._vel_ref = None
                elif dt >= FIX_VEL_MIN_DT_S:
                    v_m = (pos_m - self._vel_ref['pos']) / dt
                    v_m = np.clip(v_m, -FIX_VEL_CLAMP, FIX_VEL_CLAMP)
                    self._vel += FIX_VEL_GAIN * (v_m - self._vel)
                    self._vel_ref = {'ts': fix['ts'], 'pos': pos_m}
            if self._vel_ref is None:
                self._vel_ref = {'ts': fix['ts'], 'pos': pos_m}
        self._prev_fix = {'ts': fix['ts'], 'pos': pos_m}
        self._last_fix_wall = time.monotonic()
        self._n_fix += 1
        self._consec_rej = 0

    # ------------------------------------------------------------------
    def _publish(self):
        ts = time.time_ns()
        with self.data['lock']:
            self.data['attitude'] = {
                'roll': self._roll, 'pitch': self._pitch, 'yaw': self._yaw,
                'rollspeed': float(self._rates[0]),
                'pitchspeed': float(self._rates[1]),
                'yawspeed': float(self._rates[2]),
                'ts': ts, 'source': 'vio',
            }
            self.data['position_ned'] = {
                'x': float(self._pos[0]), 'y': float(self._pos[1]),
                'z': float(self._pos[2]),
                'vx': float(self._vel[0]), 'vy': float(self._vel[1]),
                'vz': float(self._vel[2]),
                'ts': ts, 'source': 'vio',
            }
            self.data['vio_stats'] = {
                'imu_samples': self._n_imu, 'fixes': self._n_fix,
                'fixes_rejected': self._n_fix_rej,
                'anchored_gates': sorted(self.anchors),
            }
