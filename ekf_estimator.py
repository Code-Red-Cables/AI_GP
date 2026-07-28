"""Background thread: IMU predict (~sample rate) + dual-gate PnP correct (~30 Hz).

Publishes ``shared_data['attitude']`` and ``shared_data['position_ned']`` from
the EKF, and ``shared_data['ekf_state']`` for the dual-gate planner.

When Gate 2 leaves the FOV the correction step simply omits it; the EKF
prediction continues from IMU so Gate 2's last NED fix remains a
dead-reckoning look-at target.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from ekf.drone_ekf import DroneEKF


class EKFEstimator:
    def __init__(self, data: dict):
        self.data = data
        self.ekf = DroneEKF()
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._last_imu_ts = None
        self._last_pnp_ts = None

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
        # Convert ns → seconds for filter timebase when needed.
        if ts > 1e12:
            t_s = ts * 1e-9
        else:
            t_s = float(ts)
        if self._last_imu_ts is not None and t_s <= self._last_imu_ts:
            # Still process PnP if newer.
            pass
        else:
            gyro = np.array(
                [
                    float(imu.get('xgyro', 0.0)),
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
                self.ekf.predict(gyro, accel, t_s)
            else:
                self.ekf._last_t = t_s
            self._last_imu_ts = t_s

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
