import math
import sys
import time
from pathlib import Path

from pymavlink import mavutil

import config

_DREAMER_SRC = Path(__file__).resolve().parent / 'dreamer' / 'src'
if str(_DREAMER_SRC) not in sys.path:
    sys.path.insert(0, str(_DREAMER_SRC))
from dreamer_drone.env.ahrs import AHRSConfig, ComplementaryAHRS

MAVLINK_CMD_SIM_RESET = 31000

# Ignore attitude quaternion, use body rates + thrust only
_RATES_MASK = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE


class Controller:

    def __init__(self, sim_conn, data, system_boot_ms):
        self.conn     = sim_conn
        self.data     = data
        self.boot_ms  = system_boot_ms
        self._interval = 1.0 / config.CONTROL_HZ
        self._ahrs = ComplementaryAHRS(AHRSConfig(alpha=0.95))
        self._last_imu_ts_us = None
        self._ahrs_ready = False

    def _demo_attitude(self):
        """Update and return collect_demos.py's legal-telemetry AHRS state."""
        imu = self.data.get('highres_imu') or {}
        wall_ts = imu.get('ts')
        fresh = bool(
            wall_ts is not None
            and (time.time_ns() - wall_ts) / 1e9
            <= config.TELEMETRY_TIMEOUT_S
        )
        imu_ts_us = imu.get('ts_us')
        if fresh and imu_ts_us is not None and imu_ts_us != self._last_imu_ts_us:
            if self._last_imu_ts_us is None or imu_ts_us <= self._last_imu_ts_us:
                dt = self._interval
            else:
                dt = (imu_ts_us - self._last_imu_ts_us) / 1e6
            self._last_imu_ts_us = imu_ts_us
            self._ahrs.update(
                (
                    float(imu.get('xgyro', 0.0)),
                    float(imu.get('ygyro', 0.0)),
                    float(imu.get('zgyro', 0.0)),
                ),
                (
                    float(imu.get('xacc', 0.0)),
                    float(imu.get('yacc', 0.0)),
                    float(imu.get('zacc', -9.81)),
                ),
                dt,
            )
            self._ahrs_ready = True
        return (
            self._ahrs.roll,
            self._ahrs.pitch,
            float(imu.get('xgyro', 0.0)),
            float(imu.get('ygyro', 0.0)),
            fresh and self._ahrs_ready,
        )

    # ------------------------------------------------------------------
    def update(self):
        t0 = time.time()

        att = self.data.get('attitude') or {}
        roll, pitch, rollspeed, pitchspeed, telemetry_ok = (
            self._demo_attitude()
        )
        yaw   = att.get('yaw',   0.0)

        tgt = self.data.get('planner_target') or {}
        vn       = tgt.get('vn',       0.0) if telemetry_ok else 0.0
        ve       = tgt.get('ve',       0.0) if telemetry_ok else 0.0
        vd       = tgt.get('vd',       0.0) if telemetry_ok else 0.0
        yaw_rate = tgt.get('yaw_rate', 0.0) if telemetry_ok else 0.0

        # NED velocity → body-frame velocity
        v_fwd   =  vn * math.cos(yaw) + ve * math.sin(yaw)
        v_right = -vn * math.sin(yaw) + ve * math.cos(yaw)

        # Body velocity → desired lean angles
        # collect_demos.py AHRS convention: +pitch leans forward; a gate to
        # the right requests -roll. The simulator then inverts both rate axes.
        d_pitch =  v_fwd   * config.KP_LEAN
        d_roll  = -v_right * config.KP_LEAN
        d_pitch = max(-config.MAX_LEAN_RAD, min(config.MAX_LEAN_RAD, d_pitch))
        d_roll  = max(-config.MAX_LEAN_RAD, min(config.MAX_LEAN_RAD, d_roll))

        # Lean error → rate commands (sign corrections from empirical sim tuning)
        pitch_rate = (
            (d_pitch - pitch) * config.KP_ATT
            - config.KD_ATT * pitchspeed
        ) * config.RATE_SIGN_PITCH
        roll_rate = (
            (d_roll - roll) * config.KP_ATT
            - config.KD_ATT * rollspeed
        ) * config.RATE_SIGN_ROLL
        pitch_rate = max(
            -config.MAX_RATE_RAD_S,
            min(config.MAX_RATE_RAD_S, pitch_rate),
        )
        roll_rate = max(
            -config.MAX_RATE_RAD_S,
            min(config.MAX_RATE_RAD_S, roll_rate),
        )
        yaw_rate = max(
            -config.MAX_RATE_RAD_S,
            min(config.MAX_RATE_RAD_S, yaw_rate),
        )

        # Thrust: open-loop (no baro in VQ2, vz_now=0 always)
        # vd < 0 means climb → add thrust; vd > 0 means descend → reduce thrust
        thrust = config.HOVER_THRUST - vd * config.KP_THRUST
        thrust = max(config.MIN_THRUST, min(config.MAX_THRUST, thrust))

        self.data['control_output'] = {
            'thrust':     thrust,
            'roll_rate':  roll_rate,
            'pitch_rate': pitch_rate,
            'yaw_rate':   yaw_rate,
            'telemetry_ok': telemetry_ok,
            'ahrs_roll': roll,
            'ahrs_pitch': pitch,
            'ahrs_divergence': self._ahrs.divergence,
        }

        now_ms = int(time.time() * 1000)
        self.conn.mav.set_attitude_target_send(
            now_ms - self.boot_ms,
            self.conn.target_system,
            self.conn.target_component,
            _RATES_MASK,
            [1, 0, 0, 0],   # dummy quaternion (ignored by mask)
            roll_rate,
            pitch_rate,
            yaw_rate,
            thrust,
        )

        elapsed = time.time() - t0
        rem = self._interval - elapsed
        if rem > 0:
            time.sleep(rem)

    # ------------------------------------------------------------------
    def arm(self):
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0,
        )

    def send_sim_reset(self):
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            MAVLINK_CMD_SIM_RESET,
            0, 0, 0, 0, 0, 0, 0, 0,
        )
