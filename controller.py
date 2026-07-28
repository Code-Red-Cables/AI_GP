"""Body-rate + thrust controller for keyboard teleop."""
import math
import time

from pymavlink import mavutil

import config
from ahrs import AHRSConfig, ComplementaryAHRS
from control.pid import PIDConfig, PIDController

MAVLINK_CMD_SIM_RESET = 31000
_RATES_MASK = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE


class Controller:

    def __init__(self, sim_conn, data, system_boot_ms):
        self.conn = sim_conn
        self.data = data
        self.boot_ms = system_boot_ms
        self._interval = 1.0 / config.CONTROL_HZ
        self._ahrs = ComplementaryAHRS(
            AHRSConfig(alpha=config.AHRS_GYRO_WEIGHT)
        )
        self._last_imu_ts_us = None
        self._ahrs_ready = False
        self._armed_at = None
        self._last_control_at = None
        self._last_safety_reason = None
        self._pitch_pid = PIDController(
            PIDConfig(
                kp=config.KP_ATT,
                ki=config.KI_ATT,
                kd=config.KD_ATT,
                output_min=-config.MAX_RATE_RAD_S,
                output_max=config.MAX_RATE_RAD_S,
                integral_min=-config.ATTITUDE_INTEGRAL_LIMIT,
                integral_max=config.ATTITUDE_INTEGRAL_LIMIT,
                derivative_filter_tau_s=config.ATTITUDE_DERIVATIVE_FILTER_TAU_S,
                minimum_dt_s=config.CONTROL_MIN_DT_S,
                maximum_dt_s=config.CONTROL_MAX_DT_S,
            )
        )
        self._roll_pid = PIDController(
            PIDConfig(
                kp=config.KP_ROLL_ATT,
                ki=config.KI_ROLL_ATT,
                kd=config.KD_ROLL_ATT,
                output_min=-config.MAX_RATE_RAD_S,
                output_max=config.MAX_RATE_RAD_S,
                integral_min=-config.ATTITUDE_INTEGRAL_LIMIT,
                integral_max=config.ATTITUDE_INTEGRAL_LIMIT,
                derivative_filter_tau_s=config.ATTITUDE_DERIVATIVE_FILTER_TAU_S,
                minimum_dt_s=config.CONTROL_MIN_DT_S,
                maximum_dt_s=config.CONTROL_MAX_DT_S,
            )
        )

    def _log_safety(self, reason):
        if reason == self._last_safety_reason:
            return
        log_event = self.data.get('log_event')
        if log_event:
            log_event('CONTROL_SAFETY', reason or 'clear')
        self._last_safety_reason = reason

    def _reset_control_state(self):
        self._pitch_pid.reset()
        self._roll_pid.reset()
        self._last_control_at = None

    def _update_ahrs(self):
        imu = self.data.get('highres_imu') or {}
        wall_ts = imu.get('ts')
        age_s = (
            (time.time_ns() - wall_ts) / 1e9
            if wall_ts is not None
            else float('inf')
        )
        imu_values = tuple(
            float(imu.get(name, default))
            for name, default in (
                ('xgyro', 0.0),
                ('ygyro', 0.0),
                ('zgyro', 0.0),
                ('xacc', 0.0),
                ('yacc', 0.0),
                ('zacc', -9.81),
            )
        )
        fresh = bool(
            wall_ts is not None
            and -config.SENSOR_FUTURE_TOLERANCE_S
            <= age_s
            <= config.TELEMETRY_TIMEOUT_S
            and all(math.isfinite(value) for value in imu_values)
        )
        imu_ts_us = imu.get('ts_us')
        if fresh and imu_ts_us is not None and imu_ts_us != self._last_imu_ts_us:
            if self._last_imu_ts_us is None or imu_ts_us <= self._last_imu_ts_us:
                dt = self._interval
            else:
                dt = (imu_ts_us - self._last_imu_ts_us) / 1e6
            self._last_imu_ts_us = imu_ts_us
            self._ahrs.update(imu_values[:3], imu_values[3:], dt)
            self._ahrs_ready = True
        return (
            self._ahrs.roll,
            self._ahrs.pitch,
            float(imu.get('xgyro', 0.0)),
            float(imu.get('ygyro', 0.0)),
            float(imu.get('zgyro', 0.0)),
            fresh and self._ahrs_ready,
        )

    def update(self):
        t0 = time.time()
        monotonic_now = time.monotonic()
        dt = (
            self._interval
            if self._last_control_at is None
            else monotonic_now - self._last_control_at
        )
        self._last_control_at = monotonic_now

        att = self.data.get('attitude') or {}
        roll, pitch, rollspeed, pitchspeed, _yawspeed, telemetry_ok = (
            self._update_ahrs()
        )
        yaw = float(att.get('yaw', 0.0) or 0.0)
        if not math.isfinite(yaw):
            yaw = 0.0

        tgt = self.data.get('planner_target') or {}
        raw_target = tuple(
            float(tgt.get(name, 0.0))
            for name in ('vn', 've', 'vd', 'yaw_rate')
        )
        target_ok = all(math.isfinite(value) for value in raw_target)
        if telemetry_ok and target_ok:
            vn, ve, vd, yaw_rate = raw_target
            self._log_safety(None)
        else:
            vn = ve = vd = yaw_rate = 0.0
            self._reset_control_state()
            self._log_safety(
                'stale_or_invalid_imu'
                if not telemetry_ok
                else 'nonfinite_planner_target'
            )

        v_fwd = vn * math.cos(yaw) + ve * math.sin(yaw)
        v_right = -vn * math.sin(yaw) + ve * math.cos(yaw)

        d_pitch = max(
            -config.MAX_LEAN_RAD,
            min(config.MAX_LEAN_RAD, v_fwd * config.KP_LEAN),
        )
        d_roll = max(
            -config.MAX_LEAN_RAD,
            min(
                config.MAX_LEAN_RAD,
                v_right * config.LATERAL_LEAN_GAIN,
            ),
        )

        pitch_rate = self._pitch_pid.update(
            d_pitch - pitch, dt, measurement_rate=pitchspeed,
        ) * config.RATE_SIGN_PITCH
        roll_rate = self._roll_pid.update(
            d_roll - roll, dt, measurement_rate=rollspeed,
        ) * config.RATE_SIGN_ROLL
        pitch_rate = max(
            -config.MAX_RATE_RAD_S, min(config.MAX_RATE_RAD_S, pitch_rate)
        )
        roll_rate = max(
            -config.MAX_RATE_RAD_S, min(config.MAX_RATE_RAD_S, roll_rate)
        )
        yaw_rate = yaw_rate * config.RATE_SIGN_YAW
        yaw_rate = max(
            -config.MAX_RATE_RAD_S, min(config.MAX_RATE_RAD_S, yaw_rate)
        )

        thrust_adjustment = min(
            config.MAX_ASCENT_THRUST_INCREASE,
            max(
                -config.MAX_DESCENT_THRUST_REDUCTION,
                -vd * config.KP_THRUST,
            ),
        )
        level_thrust = config.HOVER_THRUST + thrust_adjustment
        if (
            telemetry_ok
            and self._armed_at is not None
            and time.monotonic() - self._armed_at < config.TAKEOFF_DURATION_S
        ):
            level_thrust = max(level_thrust, config.TAKEOFF_THRUST)
        vertical_lift_fraction = max(
            config.MIN_TILT_COMPENSATION_COSINE,
            math.cos(roll) * math.cos(pitch),
        )
        thrust = level_thrust / vertical_lift_fraction
        thrust = max(config.MIN_THRUST, min(config.MAX_THRUST, thrust))

        if not all(
            math.isfinite(value)
            for value in (roll_rate, pitch_rate, yaw_rate, thrust)
        ):
            roll_rate = pitch_rate = yaw_rate = 0.0
            thrust = config.HOVER_THRUST
            self._reset_control_state()
            self._log_safety('nonfinite_final_command')

        self.data['control_output'] = {
            'thrust': thrust,
            'roll_rate': roll_rate,
            'pitch_rate': pitch_rate,
            'yaw_rate': yaw_rate,
            'telemetry_ok': telemetry_ok,
            'ahrs_roll': roll,
            'ahrs_pitch': pitch,
            'desired_roll': d_roll,
            'desired_pitch': d_pitch,
            'safety_reason': self._last_safety_reason,
        }

        now_ms = int(time.time() * 1000)
        self.conn.mav.set_attitude_target_send(
            now_ms - self.boot_ms,
            self.conn.target_system,
            self.conn.target_component,
            _RATES_MASK,
            [1, 0, 0, 0],
            roll_rate,
            pitch_rate,
            yaw_rate,
            thrust,
        )

        rem = self._interval - (time.time() - t0)
        if rem > 0:
            time.sleep(rem)

    def arm(self):
        self._reset_control_state()
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0,
        )
        self._armed_at = time.monotonic()

    def disarm(self):
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0,
        )
        self._armed_at = None
        self._reset_control_state()

    def send_sim_reset(self):
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            MAVLINK_CMD_SIM_RESET,
            0, 0, 0, 0, 0, 0, 0, 0,
        )
