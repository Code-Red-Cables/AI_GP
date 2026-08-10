import math
import time

from pymavlink import mavutil

import config
from ahrs import AHRSConfig, ComplementaryAHRS
from control.pid import PIDConfig, PIDController

MAVLINK_CMD_SIM_RESET = 31000

# Ignore attitude quaternion, use body rates + thrust only
_RATES_MASK = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
# Hover hold: command level attitude, ignore body rates (ANGLE self-level).
# Zero rates + ATTITUDE_IGNORE freezes any tip → sustained forward drift.
_ATTITUDE_HOLD_MASK = (
    mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE
    | mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_PITCH_RATE_IGNORE
    | mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_YAW_RATE_IGNORE
)

# The Q2 course needs materially more bank than pitch while capturing an
# off-axis gate. Keeping this separate from config.KP_LEAN preserves the
# demonstrated forward-speed mapping while making right_mps produce actual
# lateral translation instead of being masked by the yaw response.
_LATERAL_LEAN_GAIN = 0.24


class Controller:

    def __init__(self, sim_conn, data, system_boot_ms):
        self.conn     = sim_conn
        self.data     = data
        self.boot_ms  = system_boot_ms
        self._interval = 1.0 / config.CONTROL_HZ
        # Gyro-dominant estimator. Roll gyro sign matches EKF (-xgyro); the
        # default +1 made ahrs_roll opposite ekf roll (033644 death spiral).
        self._ahrs = ComplementaryAHRS(
            AHRSConfig(alpha=0.95, gyro_sign_roll=-1.0)
        )
        self._active_ahrs = self._ahrs
        self._last_imu_ts_us = None
        self._ahrs_ready = False
        self._armed_at = None
        self._takeoff_z0 = None
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
                derivative_filter_tau_s=(
                    config.ATTITUDE_DERIVATIVE_FILTER_TAU_S
                ),
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
                derivative_filter_tau_s=(
                    config.ATTITUDE_DERIVATIVE_FILTER_TAU_S
                ),
                minimum_dt_s=config.CONTROL_MIN_DT_S,
                maximum_dt_s=config.CONTROL_MAX_DT_S,
            )
        )
        # VIO-only loops. Heading hold engages when the planner requests ~zero
        # yaw rate and a fresh VIO yaw exists; the thrust PI closes the
        # vertical-velocity loop the open-loop path cannot (VQ2 has no baro).
        self._thrust_pid = PIDController(
            PIDConfig(
                kp=config.KP_THRUST_VEL,
                ki=config.KI_THRUST_VEL,
                output_min=config.VIO_THRUST_MIN - config.MAX_THRUST,
                output_max=config.VIO_THRUST_MAX,
                integral_min=-config.THRUST_INTEGRAL_LIMIT,
                integral_max=config.THRUST_INTEGRAL_LIMIT,
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
        self._thrust_pid.reset()
        self._last_control_at = None

    def reset_ahrs(self):
        """Clear complementary-filter attitude (call on arm / sim reset)."""
        self._ahrs.reset()
        self._active_ahrs = self._ahrs
        self._last_imu_ts_us = None
        self._ahrs_ready = False

    def _demo_attitude(self):
        """Update and return collect_demos.py's legal-telemetry AHRS state."""
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
            self._ahrs.update(
                imu_values[:3],
                imu_values[3:],
                dt,
            )
            self._ahrs_ready = True
        self._active_ahrs = self._ahrs
        return (
            self._active_ahrs.roll,
            self._active_ahrs.pitch,
            float(imu.get('xgyro', 0.0)),
            float(imu.get('ygyro', 0.0)),
            float(imu.get('zgyro', 0.0)),
            fresh and self._ahrs_ready,
        )

    # ------------------------------------------------------------------
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
        roll, pitch, rollspeed, pitchspeed, yawspeed, telemetry_ok = (
            self._demo_attitude()
        )
        yaw = float(att.get('yaw', 0.0))
        if not math.isfinite(yaw):
            yaw = 0.0

        tgt = self.data.get('planner_target') or {}
        kalman_direct = bool(tgt.get('kalman'))
        # Unused: this sim ignores attitude quaternions (132257 froze tip).
        attitude_hold_q = None

        if kalman_direct:
            raw_rates = (
                float(tgt.get('roll_rate', 0.0)),
                float(tgt.get('pitch_rate', 0.0)),
                float(tgt.get('yaw_rate', 0.0)),
                float(tgt.get('thrust', config.HOVER_THRUST)),
            )
            target_ok = all(math.isfinite(value) for value in raw_rates)
            if telemetry_ok and target_ok:
                roll_rate, pitch_rate, yaw_rate, thrust = raw_rates
                roll_rate *= config.RATE_SIGN_ROLL
                pitch_rate *= config.RATE_SIGN_PITCH
                # Kalman yaw is NED-positive-right (same as RATE_SIGN_YAW);
                # Run 043812: +cmd produced left turns and walked the
                # gate off-frame.
                yaw_rate = yaw_rate * config.RATE_SIGN_YAW
                # Acro / unrestricted: do not clip commanded body rates.
                if bool(tgt.get('acro')) or bool(tgt.get('unrestricted_rates')):
                    yaw_lim = None
                else:
                    # Yaw uses its own ceiling — MAX_RATE_RAD_S (~60°/s) was
                    # clipping assist extreme yaw (132028 planner asked 70°/s).
                    yaw_lim = float(
                        getattr(
                            config,
                            'YAW_RATE_MAX_RAD_S',
                            config.MAX_RATE_RAD_S,
                        )
                    )
                    yaw_rate = max(-yaw_lim, min(yaw_lim, yaw_rate))
                # Kalman skips the attitude vertical loop — boost only until
                # clear of the ground. 063921: 0.32×2s rocketed to −4 m.
                if (
                    self._armed_at is not None
                    and time.monotonic() - self._armed_at
                    < config.TAKEOFF_DURATION_S
                ):
                    pos = (
                        self.data.get('position_ned')
                        or self.data.get('local_position_ned')
                        or {}
                    )
                    z = pos.get('z')
                    climbed = None
                    if z is not None and math.isfinite(float(z)):
                        if self._takeoff_z0 is None:
                            self._takeoff_z0 = float(z)
                        climbed = self._takeoff_z0 - float(z)
                    if climbed is None or climbed < 0.55:
                        thrust = max(thrust, config.TAKEOFF_THRUST)
                vd = 0.0
                d_roll = float(tgt.get('desired_roll', 0.0))
                d_pitch = float(tgt.get('desired_pitch', 0.0))
                requested_yaw_rate = float(tgt.get('yaw_rate', 0.0))
                self._log_safety(None)
            else:
                roll_rate = pitch_rate = yaw_rate = 0.0
                thrust = config.HOVER_THRUST
                vd = 0.0
                d_roll = d_pitch = 0.0
                requested_yaw_rate = 0.0
                self._reset_control_state()
                self._log_safety(
                    'stale_or_invalid_imu'
                    if not telemetry_ok
                    else 'nonfinite_planner_target'
                )
            # Cascaded PID already produced rates + thrust; skip lean/yaw remap.
            vn = ve = 0.0
        else:
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

            # NED velocity → body-frame velocity
            v_fwd   =  vn * math.cos(yaw) + ve * math.sin(yaw)
            v_right = -vn * math.sin(yaw) + ve * math.cos(yaw)

            # Body velocity → desired lean angles.
            d_pitch = v_fwd * config.KP_LEAN
            d_roll  = (
                v_right
                * _LATERAL_LEAN_GAIN
                * config.LATERAL_LEAN_SIGN
            )
            d_pitch = max(-config.MAX_LEAN_RAD, min(config.MAX_LEAN_RAD, d_pitch))
            d_roll  = max(-config.MAX_LEAN_RAD, min(config.MAX_LEAN_RAD, d_roll))

            # Lean error → rate commands (sign corrections from empirical sim tuning)
            pitch_rate = self._pitch_pid.update(
                d_pitch - pitch,
                dt,
                measurement_rate=pitchspeed,
            ) * config.RATE_SIGN_PITCH
            roll_rate = self._roll_pid.update(
                d_roll - roll,
                dt,
                measurement_rate=rollspeed,
            ) * config.RATE_SIGN_ROLL
            pitch_rate = max(
                -config.MAX_RATE_RAD_S,
                min(config.MAX_RATE_RAD_S, pitch_rate),
            )
            roll_rate = max(
                -config.MAX_RATE_RAD_S,
                min(config.MAX_RATE_RAD_S, roll_rate),
            )
            requested_yaw_rate = yaw_rate
            thrust = None  # computed by vertical loop below

        measured_yaw_rate = 0.0
        yaw_rate_feedback = 0.0
        if not kalman_direct:
            yaw_rate = requested_yaw_rate * config.RATE_SIGN_YAW
            yaw_lim = float(
                getattr(config, 'YAW_RATE_MAX_RAD_S', config.MAX_RATE_RAD_S)
            )
            yaw_rate = max(-yaw_lim, min(yaw_lim, yaw_rate))

            # Thrust: with a fresh VIO velocity the vertical loop is CLOSED — the
            # PI tracks the commanded NED down-velocity ``vd`` against the VIO's
            # vz belief. Without VIO (or during a vision-starved dropout) thrust
            # falls back to the open-loop lean-compensated hover baseline.
            # vd < 0 means climb → add thrust; vd > 0 means descend → reduce thrust.
            vio_vz = None
            thrust_closed_loop = (
                telemetry_ok and target_ok and vio_vz is not None
            )
            if thrust_closed_loop:
                # Positive error = descending faster than commanded → more thrust.
                thrust_adjustment = self._thrust_pid.update(vio_vz - vd, dt)
            else:
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
                and time.monotonic() - self._armed_at
                < config.TAKEOFF_DURATION_S
            ):
                # Lift off the deck, then hold HOVER_THRUST.
                level_thrust = max(level_thrust, config.TAKEOFF_THRUST)
            vertical_lift_fraction = max(
                config.MIN_TILT_COMPENSATION_COSINE,
                math.cos(roll) * math.cos(pitch),
            )
            thrust = level_thrust / vertical_lift_fraction
            if thrust_closed_loop:
                # Closed-loop floor keeps prop wash / attitude authority while
                # descending; the ceiling matches the flight-tested VIO limits.
                thrust = max(
                    config.VIO_THRUST_MIN, min(config.VIO_THRUST_MAX, thrust)
                )
            else:
                thrust = max(config.MIN_THRUST, min(config.MAX_THRUST, thrust))
        else:
            # kalman_direct path: yaw was already signed above. Skip clip for
            # acro / unrestricted rate targets (full-stick body rates).
            if not (
                bool(tgt.get('acro')) or bool(tgt.get('unrestricted_rates'))
            ):
                yaw_lim = float(
                    getattr(
                        config, 'YAW_RATE_MAX_RAD_S', config.MAX_RATE_RAD_S
                    )
                )
                yaw_rate = max(-yaw_lim, min(yaw_lim, yaw_rate))
            thrust = max(config.MIN_THRUST, min(config.MAX_THRUST, thrust))
            vertical_lift_fraction = max(
                config.MIN_TILT_COMPENSATION_COSINE,
                math.cos(roll) * math.cos(pitch),
            )
            thrust_adjustment = 0.0
            vio_vz = None
            thrust_closed_loop = False
            vd = 0.0

        if not all(
            math.isfinite(value)
            for value in (roll_rate, pitch_rate, yaw_rate, thrust)
        ):
            roll_rate = pitch_rate = yaw_rate = 0.0
            thrust = config.HOVER_THRUST
            self._reset_control_state()
            self._log_safety('nonfinite_final_command')

        self.data['control_output'] = {
            'thrust':     thrust,
            'roll_rate':  roll_rate,
            'pitch_rate': pitch_rate,
            'yaw_rate':   yaw_rate,
            'telemetry_ok': telemetry_ok,
            'ahrs_roll': roll,
            'ahrs_pitch': pitch,
            'ahrs_divergence': self._active_ahrs.divergence,
            'desired_roll': d_roll,
            'desired_pitch': d_pitch,
            'requested_yaw_rate': requested_yaw_rate,
            'measured_yaw_rate': measured_yaw_rate,
            'yaw_rate_feedback': yaw_rate_feedback,
            'vertical_lift_fraction': vertical_lift_fraction,
            'hover_thrust': config.HOVER_THRUST,
            'thrust_adjustment': thrust_adjustment,
            'vertical_command': vd,
            'vertical_velocity': (
                vio_vz
                if vio_vz is not None
                else (self.data.get('local_position_ned') or {}).get('vz')
            ),
            'thrust_closed_loop': thrust_closed_loop,
            'kalman_direct': kalman_direct,
            'safety_reason': self._last_safety_reason,
        }

        now_ms = int(time.time() * 1000)
        if attitude_hold_q is not None:
            # ANGLE tracks level quaternion; rates ignored. Required for
            # pad hover — zero rates + ATTITUDE_IGNORE freezes tip.
            self.conn.mav.set_attitude_target_send(
                now_ms - self.boot_ms,
                self.conn.target_system,
                self.conn.target_component,
                _ATTITUDE_HOLD_MASK,
                attitude_hold_q,
                0.0,
                0.0,
                0.0,
                thrust,
            )
        else:
            self.conn.mav.set_attitude_target_send(
                now_ms - self.boot_ms,
                self.conn.target_system,
                self.conn.target_component,
                _RATES_MASK,
                [1.0, 0.0, 0.0, 0.0],  # ignored by rates mask
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
        self._reset_control_state()
        self.reset_ahrs()
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0,
        )
        self._armed_at = time.monotonic()
        self._takeoff_z0 = None

    def disarm(self):
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0,
        )
        self._armed_at = None
        self._takeoff_z0 = None
        self._reset_control_state()

    def send_sim_reset(self):
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            MAVLINK_CMD_SIM_RESET,
            0, 0, 0, 0, 0, 0, 0, 0,
        )
