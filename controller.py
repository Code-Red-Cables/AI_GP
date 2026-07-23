import math
import time

from pymavlink import mavutil

import config

MAVLINK_CMD_SIM_RESET = 31000

# Ignore attitude quaternion, use body rates + thrust only
_RATES_MASK = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE


class Controller:

    def __init__(self, sim_conn, data, system_boot_ms):
        self.conn     = sim_conn
        self.data     = data
        self.boot_ms  = system_boot_ms
        self._interval = 1.0 / config.CONTROL_HZ

    # ------------------------------------------------------------------
    def update(self):
        t0 = time.time()

        att = self.data.get('attitude') or {}
        roll  = att.get('roll',  0.0)
        pitch = att.get('pitch', 0.0)
        yaw   = att.get('yaw',   0.0)

        tgt = self.data.get('planner_target') or {}
        vn       = tgt.get('vn',       0.0)
        ve       = tgt.get('ve',       0.0)
        vd       = tgt.get('vd',       0.0)
        yaw_rate = tgt.get('yaw_rate', 0.0)

        # NED velocity → body-frame velocity
        v_fwd   =  vn * math.cos(yaw) + ve * math.sin(yaw)
        v_right = -vn * math.sin(yaw) + ve * math.cos(yaw)

        # Body velocity → desired lean angles
        d_pitch = -v_fwd   * config.KP_LEAN    # negative pitch = nose down = forward
        d_roll  =  v_right * config.KP_LEAN
        d_pitch = max(-config.MAX_LEAN_RAD, min(config.MAX_LEAN_RAD, d_pitch))
        d_roll  = max(-config.MAX_LEAN_RAD, min(config.MAX_LEAN_RAD, d_roll))

        # Lean error → rate commands (sign corrections from empirical sim tuning)
        pitch_rate = (d_pitch - pitch) * config.KP_ATT * config.RATE_SIGN_PITCH
        roll_rate  = (d_roll  - roll)  * config.KP_ATT * config.RATE_SIGN_ROLL

        # Thrust: open-loop (no baro in VQ2, vz_now=0 always)
        # vd < 0 means climb → add thrust; vd > 0 means descend → reduce thrust
        thrust = config.HOVER_THRUST - vd * config.KP_THRUST
        thrust = max(config.MIN_THRUST, min(config.MAX_THRUST, thrust))

        self.data['control_output'] = {
            'thrust':     thrust,
            'roll_rate':  roll_rate,
            'pitch_rate': pitch_rate,
            'yaw_rate':   yaw_rate,
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
