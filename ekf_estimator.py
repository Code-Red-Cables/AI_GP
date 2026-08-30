"""Background thread: IMU predict (~sample rate) + dual-gate PnP correct (~30 Hz).

Publishes ``shared_data['attitude']`` and ``shared_data['position_ned']`` from
the EKF, and ``shared_data['ekf_state']`` for the dual-gate planner.

When Gate 2 leaves the FOV the correction step simply omits it; the EKF
keeps propagating with commanded thrust + attitude so Gate 2's last NED
fix remains a dead-reckoning look-at target.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Optional

import numpy as np

import config
from ekf.commanded_accel import observe_hover_trim
from ekf.drone_ekf import DroneEKF


class EKFEstimator:
    def __init__(self, data: dict):
        self.data = data
        self.ekf = DroneEKF(
            use_commanded_accel=bool(
                getattr(config, 'EKF_COMMANDED_ACCEL', True)
            ),
            hover_trim=float(getattr(config, 'HOVER_THRUST', 0.255)),
            drag_k_body=np.array(
                [
                    float(getattr(config, 'DRAG_KX', -0.50)),
                    float(getattr(config, 'DRAG_KY', -0.50)),
                    float(getattr(config, 'DRAG_KZ', -0.15)),
                ],
                dtype=np.float64,
            ),
            commanded_accel_noise=float(
                getattr(config, 'EKF_COMMANDED_ACCEL_NOISE', 0.14)
            ),
        )
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._last_imu_ts = None
        self._last_pnp_ts = None
        self._hover_trim = float(getattr(config, 'HOVER_THRUST', 0.255))
        self._hover_trim_tau_s = float(
            getattr(config, 'EKF_HOVER_TRIM_TAU_S', 2.0)
        )
        self._gate_horizon_fixes = 0
        self._gate_yaw_fixes = 0
        self._gate_att_rejects = 0
        self._gate_att_skips = 0
        self._use_pnp = bool(getattr(config, 'EKF_USE_PNP', True))
        self.ekf.accel_tilt_gain = float(
            getattr(config, 'EKF_ACCEL_TILT_GAIN', 0.0)
        )
        self.ekf.accel_tilt_max_acc = float(
            getattr(config, 'EKF_ACCEL_TILT_MAX_ACC', 1.5)
        )
        self.ekf.accel_tilt_max_rate = math.radians(
            float(getattr(config, 'EKF_ACCEL_TILT_MAX_RATE_DEG', 25.0))
        )
        self.ekf.gate_horizon_gain = float(
            getattr(config, 'EKF_GATE_HORIZON_GAIN', 0.0)
        )
        self.ekf.gate_horizon_max_step = math.radians(
            float(getattr(config, 'EKF_GATE_HORIZON_MAX_STEP_DEG', 1.0))
        )
        self.ekf.gate_horizon_bias_gain = float(
            getattr(config, 'EKF_GATE_HORIZON_BIAS_GAIN', 0.30)
        )
        self.ekf.gate_horizon_pitch_scale = float(
            getattr(config, 'EKF_GATE_HORIZON_PITCH_SCALE', 0.25)
        )
        self.ekf.gate_bias_innov_max = math.radians(
            float(getattr(config, 'EKF_GATE_BIAS_INNOV_MAX_DEG', 8.0))
        )
        self.ekf.gyro_bias_limit = math.radians(
            float(getattr(config, 'EKF_GYRO_BIAS_LIMIT_DPS', 1.5))
        )
        self.ekf.gate_yaw_gain = float(
            getattr(config, 'EKF_GATE_YAW_GAIN', 0.0)
        )
        self.ekf.gate_yaw_max_step = math.radians(
            float(getattr(config, 'EKF_GATE_YAW_MAX_STEP_DEG', 1.0))
        )
        self.ekf.gate_yaw_bias_gain = float(
            getattr(config, 'EKF_GATE_YAW_BIAS_GAIN', 0.20)
        )
        self._gate_att_max_range = float(
            getattr(config, 'EKF_GATE_ATT_MAX_RANGE_M', 30.0)
        )
        self._gate_att_max_reproj = float(
            getattr(config, 'EKF_GATE_ATT_MAX_REPROJ_PX', 6.0)
        )
        if self.ekf.use_commanded_accel:
            print(
                '[EKF] velocity from commanded thrust+attitude '
                f'(hover_trim={self._hover_trim:.3f}, not accelerometer)',
                flush=True,
            )
        if not self._use_pnp:
            print(
                '[EKF] PnP corrections OFF — process-model dead reckoning only '
                '(EKF_USE_PNP=0)',
                flush=True,
            )

    @classmethod
    def create_ekf_estimator(cls, data: dict) -> 'EKFEstimator':
        est = cls(data)
        est.is_running = True
        est._thread = threading.Thread(
            target=est._loop, name='EKFEstimator', daemon=True
        )
        est._thread.start()
        return est

    def get_thread_for_join(self) -> threading.Thread:
        self.is_running = False
        assert self._thread is not None
        return self._thread

    def reset_episode(self) -> None:
        """Zero the filter after a sim reset so pos_d does not stay crashed."""
        self.ekf.reset(timestamp=0.0)
        self._last_imu_ts = None
        self._last_pnp_ts = None
        self._hover_trim = float(getattr(config, 'HOVER_THRUST', 0.255))
        lock = self.data.get('lock')
        cleared = {
            'x': 0.0,
            'y': 0.0,
            'z': 0.0,
            'vx': 0.0,
            'vy': 0.0,
            'vz': 0.0,
            'ts': time.time_ns(),
            'source': 'ekf_reset',
        }
        if lock is not None:
            with lock:
                self.data['position_ned'] = cleared
        else:
            self.data['position_ned'] = cleared

    def realign_gravity_from_imu(self) -> bool:
        """Snap EKF roll/pitch to current IMU accel (keep yaw). Quiet hover only."""
        imu = self.data.get('highres_imu') or {}
        if not imu:
            return False
        accel = np.array(
            [
                float(imu.get('xacc', 0.0)),
                float(imu.get('yacc', 0.0)),
                float(imu.get('zacc', 0.0)),
            ],
            dtype=np.float64,
        )
        return bool(self.ekf.realign_gravity(accel))

    def zero_tilt(self) -> tuple[float, float, float]:
        """Declare current attitude as level (roll/pitch=0, keep yaw).

        Publishes attitude immediately so the next control tick sees it.
        """
        roll, pitch, yaw = self.ekf.zero_tilt()
        attitude = {
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'rollspeed': 0.0,
            'pitchspeed': 0.0,
            'yawspeed': 0.0,
            'ts': time.time_ns(),
            'source': 'ekf_zero_tilt',
        }
        lock = self.data.get('lock')
        if lock is not None:
            with lock:
                prev = dict(self.data.get('ekf_state') or {})
                prev.update({
                    'roll': roll,
                    'pitch': pitch,
                    'yaw': yaw,
                    'gyro_bias': self.ekf.state().gyro_bias.tolist(),
                })
                self.data['attitude'] = attitude
                self.data['ekf_state'] = prev
        else:
            self.data['attitude'] = attitude
        return roll, pitch, yaw

    def _attitude_frame_usable(self, dual: dict) -> bool:
        """Is this PnP frame good enough to take *attitude* from?

        A stricter bar than position. Gate rotation degrades with range long
        before the centre does, and a 'held' frame is last frame's pose
        re-stamped — applying it again at 30 Hz would drag attitude toward a
        stale horizon while the craft keeps rotating.
        """
        if dual.get('held'):
            self._gate_att_skips += 1
            return False
        rng = dual.get('gate1_range_m')
        if rng is None or float(rng) > self._gate_att_max_range:
            self._gate_att_skips += 1
            return False
        reproj = dual.get('gate1_reproj_px')
        if reproj is not None and float(reproj) > self._gate_att_max_reproj:
            self._gate_att_skips += 1
            return False
        return True

    def _loop(self) -> None:
        # Target up to 500 Hz wait; actual rate follows IMU sample arrival.
        period = 1.0 / 500.0
        while self.is_running:
            t0 = time.monotonic()
            self._step()
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, period - elapsed))

    def _step(self) -> None:
        lock = self.data.get('lock')
        imu = self.data.get('highres_imu')
        if not imu:
            return
        ts = imu.get('ts')
        if ts is None:
            return
        # Wall-clock arrival time. (Sim ``time_usec`` often does not advance
        # on this link — switching to it froze EKF at 0° and the pilot
        # somersaulted under max rate, 141128. Re-timing arrivals into CE
        # slow-mo seconds fails the same way: under CE the IMU stream is
        # already wall-referenced, so scaling dt halved reported lean —
        # ekf/des 0.49 with rate pinned at 100°/s, 143405.)
        if ts > 1e12:
            t_s = ts * 1e-9
        else:
            t_s = float(ts)
        if self._last_imu_ts is not None and t_s <= self._last_imu_ts:
            # Still process PnP if newer.
            pass
        else:
            # Negate xgyro: raw IMU roll-rate is opposite truth ATTITUDE
            # (step_pitch_20260728_174223: truth roll +0.40 while EKF went
            # -0.38). RATE_SIGN_ROLL was -1 to compensate the inverted EKF;
            # with the EKF matching truth, RATE_SIGN_ROLL is +1.
            gyro = np.array(
                [
                    -float(imu.get('xgyro', 0.0)),
                    float(imu.get('ygyro', 0.0)),
                    float(imu.get('zgyro', 0.0)),
                ],
                dtype=np.float64,
            )
            accel = np.array(
                [
                    float(imu.get('xacc', 0.0)),
                    float(imu.get('yacc', 0.0)),
                    float(imu.get('zacc', 0.0)),
                ],
                dtype=np.float64,
            )
            if self.data.get('flight_started'):
                ctrl = self.data.get('control_output') or {}
                thrust = ctrl.get('thrust')
                cfg_hover = ctrl.get('hover_thrust')
                if cfg_hover is not None and math.isfinite(float(cfg_hover)):
                    # Controller trim is the prior; the observer may walk it.
                    if abs(self._hover_trim - float(cfg_hover)) > 0.08:
                        self._hover_trim = float(cfg_hover)
                thr = None if thrust is None else float(thrust)
                if thr is not None and math.isfinite(thr):
                    dt_obs = (
                        0.0
                        if self._last_imu_ts is None
                        else max(0.0, min(0.05, t_s - self._last_imu_ts))
                    )
                    st_prev = self.ekf.state()
                    self._hover_trim = observe_hover_trim(
                        self._hover_trim,
                        thr,
                        roll=float(st_prev.roll_pitch_yaw[0]),
                        pitch=float(st_prev.roll_pitch_yaw[1]),
                        rates_rad_s=gyro,
                        vel_d=float(st_prev.velocity_ned[2]),
                        dt=dt_obs,
                        tau_s=self._hover_trim_tau_s,
                    )
                else:
                    thr = None
                self.ekf.predict(
                    gyro,
                    accel,
                    t_s,
                    thrust=thr,
                    hover_trim=self._hover_trim,
                )
            else:
                self.ekf._last_t = t_s
            self._last_imu_ts = t_s

        if self._use_pnp:
            dual = self.data.get('dual_gate_pnp')
            if dual and dual.get('ts') != self._last_pnp_ts:
                self._last_pnp_ts = dual.get('ts')
                g1 = dual.get('gate1_body')
                g2 = dual.get('gate2_body')
                if g1 is not None and self.data.get('flight_started'):
                    self.ekf.correct_dual_gate_body(
                        np.asarray(g1, dtype=np.float64),
                        None
                        if g2 is None
                        else np.asarray(g2, dtype=np.float64),
                        t_s,
                    )
                    # Absolute attitude from the gate itself — the one
                    # reference that does not integrate, so "level" stops
                    # walking over a lap (141532: +4° → −23°). Far gates are
                    # only a few dozen pixels wide, where corner noise and
                    # IPPE's planar ambiguity make the *rotation* unreliable
                    # long before the centre is, so take attitude from near
                    # gates only.
                    if self._attitude_frame_usable(dual):
                        down = dual.get('gate1_down_body')
                        normal = dual.get('gate1_normal_body')
                        if down is not None:
                            if self.ekf.correct_gate_horizon(
                                np.asarray(down, dtype=np.float64)
                            ):
                                self._gate_horizon_fixes += 1
                            else:
                                self._gate_att_rejects += 1
                        if normal is not None:
                            if self.ekf.correct_gate_yaw(
                                np.asarray(normal, dtype=np.float64)
                            ):
                                self._gate_yaw_fixes += 1

        st = self.ekf.state()
        roll, pitch, yaw = st.roll_pitch_yaw
        attitude = {
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'rollspeed': 0.0,
            'pitchspeed': 0.0,
            'yawspeed': 0.0,
            'ts': time.time_ns(),
            'source': 'ekf',
        }
        position = {
            'x': float(st.position_ned[0]),
            'y': float(st.position_ned[1]),
            'z': float(st.position_ned[2]),
            'vx': float(st.velocity_ned[0]),
            'vy': float(st.velocity_ned[1]),
            'vz': float(st.velocity_ned[2]),
            'ts': time.time_ns(),
            'source': 'ekf',
        }
        ekf_pub = {
            'position_ned': st.position_ned.tolist(),
            'velocity_ned': st.velocity_ned.tolist(),
            'quaternion': st.quaternion.tolist(),
            'gate1_ned': None
            if st.gate1_ned is None
            else st.gate1_ned.tolist(),
            'gate2_ned': None
            if st.gate2_ned is None
            else st.gate2_ned.tolist(),
            'gate2_fresh': st.gate2_fresh,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            # Vision attitude aid — without these the only way to tell the aid
            # is doing nothing is to notice the drift it was meant to remove.
            'gate_horizon_fixes': self._gate_horizon_fixes,
            'gate_yaw_fixes': self._gate_yaw_fixes,
            'gate_att_rejects': self._gate_att_rejects,
            'gate_att_skips': self._gate_att_skips,
            'horizon_innov_roll': float(self.ekf.last_horizon_innov[0]),
            'horizon_innov_pitch': float(self.ekf.last_horizon_innov[1]),
            'yaw_innov': float(self.ekf.last_yaw_innov),
            'gyro_bias': st.gyro_bias.tolist(),
            'hover_trim': float(self._hover_trim),
            'accel_source': (
                'commanded' if self.ekf.use_commanded_accel else 'imu'
            ),
            'a_cmd_n': float(self.ekf.last_accel_ned[0]),
            'a_cmd_e': float(self.ekf.last_accel_ned[1]),
            'a_cmd_d': float(self.ekf.last_accel_ned[2]),
        }
        if lock is not None:
            with lock:
                self.data['attitude'] = attitude
                self.data['position_ned'] = position
                self.data['ekf_state'] = ekf_pub
        else:
            self.data['attitude'] = attitude
            self.data['position_ned'] = position
            self.data['ekf_state'] = ekf_pub
