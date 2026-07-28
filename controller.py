import math
import sys
import time
from pathlib import Path

from pymavlink import mavutil

import config
from control.pid import PIDConfig, PIDController

_DREAMER_SRC = Path(__file__).resolve().parent / 'dreamer' / 'src'
if str(_DREAMER_SRC) not in sys.path:
    sys.path.insert(0, str(_DREAMER_SRC))
from dreamer_drone.env.ahrs import AHRSConfig, ComplementaryAHRS

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
        # Preserve exact demonstration replay behavior outside OpenCV racing,
        # while giving Q2 a gyro-dominant estimator that can actually observe
        # the ±25-30 degree banks visible in the camera and raw gyro trace.
        self._ahrs = ComplementaryAHRS(AHRSConfig(alpha=0.95))
        self._opencv_ahrs = ComplementaryAHRS(
            AHRSConfig(alpha=config.OPENCV_AHRS_GYRO_WEIGHT)
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
        self._yaw_pid = PIDController(
            PIDConfig(
                kp=config.KP_YAW_ATT,
                ki=config.KI_YAW_ATT,
                kd=config.KD_YAW_ATT,
                output_min=-config.YAW_RATE_MAX_RAD_S,
                output_max=config.YAW_RATE_MAX_RAD_S,
                integral_min=-config.ATTITUDE_INTEGRAL_LIMIT,
                integral_max=config.ATTITUDE_INTEGRAL_LIMIT,
                minimum_dt_s=config.CONTROL_MIN_DT_S,
                maximum_dt_s=config.CONTROL_MAX_DT_S,
            )
        )
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
        self._yaw_hold_target = None
        # Pose-debug station-keep state (accel → vz, vision range lock).
        self._pose_vz = 0.0
        self._pose_az_bias = None
        self._pose_vz_i = 0.0
        self._pose_locked_range = None
        self._pose_max_range = None
        self._pose_locked_nx = None
        self._pose_locked_ny = None
        self._pose_yaw_hold = None
        self._pose_v_fwd = 0.0
        self._pose_v_right = 0.0
        self._pose_ax_lp = 0.0
        self._pose_ay_lp = 0.0
        self._pose_range_filt = None
        self._pose_last_range_err = None

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
        self._yaw_pid.reset()
        self._thrust_pid.reset()
        self._yaw_hold_target = None
        self._last_control_at = None
        self._pose_vz = 0.0
        self._pose_az_bias = None
        self._pose_vz_i = 0.0
        self._pose_locked_range = None
        self._pose_max_range = None
        self._pose_locked_nx = None
        self._pose_locked_ny = None
        self._pose_yaw_hold = None
        self._pose_v_fwd = 0.0
        self._pose_v_right = 0.0
        self._pose_ax_lp = 0.0
        self._pose_ay_lp = 0.0
        self._pose_range_filt = None
        self._pose_last_range_err = None

    def _vio_state(self):
        """Return fresh VIO-owned (attitude, position) dicts, or Nones.

        The StateEstimator stamps its blackboard entries with source='vio'.
        A stale belief (thread died, vision starved at boot) must not be
        trusted, so anything older than VIO_STATE_TIMEOUT_S falls back to the
        AHRS / open-loop paths.
        """
        if not config.USE_VIO:
            return None, None
        now_ns = time.time_ns()

        def fresh(entry):
            return (
                entry is not None
                and entry.get('source') == 'vio'
                and now_ns - entry.get('ts', 0)
                < config.VIO_STATE_TIMEOUT_S * 1e9
            )

        attitude = self.data.get('attitude')
        position = self.data.get('position_ned')
        return (
            attitude if fresh(attitude) else None,
            position if fresh(position) else None,
        )

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
            for estimator in (self._ahrs, self._opencv_ahrs):
                estimator.update(
                    imu_values[:3],
                    imu_values[3:],
                    dt,
                )
            self._ahrs_ready = True
        planner_mode = str(self.data.get('planner_mode', ''))
        self._active_ahrs = (
            self._opencv_ahrs
            if planner_mode.startswith('opencv_')
            else self._ahrs
        )
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
        vio_att, vio_pos = self._vio_state()
        if vio_att is not None:
            # The VIO integrates the same raw gyro as the AHRS (identical
            # sign conventions) but adds gravity + gate-PnP corrections, so
            # its roll/pitch are absolute rather than drift-prone.
            roll = float(vio_att['roll'])
            pitch = float(vio_att['pitch'])
            rollspeed = float(vio_att['rollspeed'])
            pitchspeed = float(vio_att['pitchspeed'])
            yawspeed = float(vio_att['yawspeed'])
        yaw = float(att.get('yaw', 0.0))
        if not math.isfinite(yaw):
            yaw = 0.0

        tgt = self.data.get('planner_target') or {}
        planner_mode = str(self.data.get('planner_mode', ''))
        kalman_direct = bool(tgt.get('kalman'))
        # Unused: this sim ignores attitude quaternions (132257 froze tip).
        attitude_hold_q = None

        # pose_debug: start-pad hover via continuous body-rate leveling.
        # Evidence 132257: ANGLE attitude-hold + rate-ignore → tip froze at
        # ~4.4° / ax≈-0.75 and cruised forward. Rates are the only actuator.
        if planner_mode.startswith('pose_debug'):
            armed_age = (
                0.0
                if self._armed_at is None
                else monotonic_now - self._armed_at
            )
            in_takeoff = armed_age < config.POSE_DEBUG_TAKEOFF_S
            roll_rate = yaw_rate = pitch_rate = 0.0
            d_roll = d_pitch = 0.0
            phase = 'takeoff' if in_takeoff else 'hover'
            range_m = None
            nx = ny = 0.0

            imu = self.data.get('highres_imu') or {}
            try:
                ax = float(imu.get('xacc', 0.0))
            except (TypeError, ValueError):
                ax = 0.0
            if not math.isfinite(ax):
                ax = 0.0
            try:
                ay = float(imu.get('yacc', 0.0))
            except (TypeError, ValueError):
                ay = 0.0
            if not math.isfinite(ay):
                ay = 0.0
            try:
                az = float(imu.get('zacc', -9.81))
            except (TypeError, ValueError):
                az = -9.81
            if not math.isfinite(az):
                az = -9.81

            dual = self.data.get('dual_gate_pnp') or {}
            try:
                if dual.get('gate1_norm_x') is not None:
                    nx = float(dual['gate1_norm_x'])
                if dual.get('gate1_norm_y') is not None:
                    ny = float(dual['gate1_norm_y'])
            except (TypeError, ValueError):
                nx = ny = 0.0
            try:
                raw_r = dual.get('gate1_range_m')
                if raw_r is not None:
                    range_m = float(raw_r)
            except (TypeError, ValueError):
                range_m = None
            if range_m is not None and not (1.0 < range_m < 40.0):
                range_m = None

            det = self.data.get('gate_detection') or {}
            bbox = det.get('bbox_px')
            center = det.get('center_px')
            try:
                from camera_model import FX, CX, CY, WIDTH, HEIGHT
            except Exception:
                FX, CX, CY, WIDTH, HEIGHT = 320.0, 320.0, 180.0, 640.0, 360.0
            if center is not None and (nx == 0.0 and ny == 0.0):
                try:
                    cx, cy = float(center[0]), float(center[1])
                    nx = (cx - CX) / (WIDTH * 0.5)
                    ny = (cy - CY) / (HEIGHT * 0.5)
                except (TypeError, ValueError, IndexError):
                    pass
            if bbox is not None and len(bbox) >= 4:
                try:
                    _x, _y, bw, bh = (float(v) for v in bbox[:4])
                    side = max(bw, bh, 1.0)
                    if side >= 20.0:
                        size_r = float(FX) * 2.7 / side
                        if 1.0 < size_r < 40.0:
                            range_m = (
                                size_r
                                if range_m is None
                                else max(range_m, size_r)
                            )
                except (TypeError, ValueError):
                    pass

            if range_m is not None:
                if (
                    self._pose_max_range is None
                    or range_m > self._pose_max_range
                ):
                    self._pose_max_range = range_m

            # Jump-rejected range EMA before lock/brake. Raw PnP flips
            # 2↔22m — ignore |Δ|>3m spikes (1325xx).
            range_trust = None
            if range_m is not None:
                if self._pose_range_filt is None:
                    self._pose_range_filt = float(range_m)
                    range_trust = self._pose_range_filt
                elif abs(range_m - self._pose_range_filt) <= 3.0:
                    self._pose_range_filt = (
                        0.85 * self._pose_range_filt + 0.15 * float(range_m)
                    )
                    range_trust = self._pose_range_filt

            if (
                self._pose_locked_range is None
                and self._pose_range_filt is not None
            ):
                self._pose_locked_range = float(self._pose_range_filt)
                self._pose_locked_nx = float(nx)
                self._pose_locked_ny = float(ny)
                print(
                    f'[POSE_DEBUG] lock start pose range='
                    f'{self._pose_locked_range:.2f}m '
                    f'nx={self._pose_locked_nx:+.2f} '
                    f'ny={self._pose_locked_ny:+.2f}',
                    flush=True,
                )

            # Hold level when range is trusted near lock; slight nose-back
            # only when PnP is blind. Allow a small forward tip when clearly
            # too far (never-forward clamp caused reverse drift at 14m).
            lim = config.POSE_DEBUG_MAX_LEVEL_RATE
            des_pitch = -config.POSE_DEBUG_HOVER_BIAS_RAD
            des_roll = 0.0
            if (
                self._pose_locked_range is not None
                and range_trust is not None
            ):
                range_err = range_trust - self._pose_locked_range
                self._pose_last_range_err = range_err
                if abs(range_err) <= 0.35:
                    des_pitch = 0.0
                else:
                    des_pitch = config.POSE_DEBUG_KP_RANGE * range_err
                    des_pitch = max(
                        -config.POSE_DEBUG_MAX_BRAKE_RAD,
                        min(config.POSE_DEBUG_MAX_FWD_RAD, des_pitch),
                    )
            # Brief nose-back after boost only (was 1.2s×2° — crawled).
            if 0.3 <= armed_age < 0.9:
                des_pitch = min(des_pitch, -math.radians(1.2))
            d_pitch = des_pitch
            d_roll = des_roll
            self._pose_v_fwd = (
                0.0
                if self._pose_last_range_err is None
                else -float(self._pose_last_range_err)
            )

            if not telemetry_ok:
                phase = 'no_telem'
            elif abs(pitch) > math.radians(40.0) or abs(roll) > math.radians(40.0):
                self._pitch_pid.reset()
                self._roll_pid.reset()
                phase = 'tumble_hold'
            else:
                # No KP_AX: gravity projects into ax when holding a brake tip
                # and fights des_pitch (133xxx couldn't hold −1.5°).
                pitch_rate = (
                    config.POSE_DEBUG_KP_LEVEL
                    * (des_pitch - pitch)
                    * config.RATE_SIGN_PITCH
                )
                roll_rate = (
                    config.POSE_DEBUG_KP_LEVEL
                    * (des_roll - roll)
                    * config.RATE_SIGN_ROLL
                )
                pitch_rate = max(-lim, min(lim, pitch_rate))
                roll_rate = max(-lim, min(lim, roll_rate))
                if in_takeoff:
                    phase = 'takeoff'
                elif abs(pitch - des_pitch) > math.radians(0.5) or abs(
                    roll - des_roll
                ) > math.radians(0.5):
                    phase = 'righting'
                else:
                    phase = 'hover'

            # Altitude: hold clear of the deck. 141553 scraped at 0.26 with
            # thr_min 0.245 and no tilt compensation.
            g_ref = -9.81
            accel_up = -(az - g_ref)
            if in_takeoff:
                self._pose_vz = 0.0
                self._pose_vz_i = 0.0
            else:
                self._pose_vz = 0.97 * self._pose_vz + accel_up * dt
                self._pose_vz_i = max(
                    -0.02,
                    min(0.02, self._pose_vz_i + self._pose_vz * dt),
                )
            thrust = (
                config.POSE_DEBUG_HOVER_THRUST
                - config.POSE_DEBUG_KP_VZ * self._pose_vz
                - config.POSE_DEBUG_KI_VZ * self._pose_vz_i
            )
            if (
                self._pose_locked_ny is not None
                and math.isfinite(ny)
                and not in_takeoff
            ):
                thrust += config.POSE_DEBUG_KP_NY * (
                    self._pose_locked_ny - ny
                )
            # Collective must cover tilt or a −1° brake tip bleeds altitude.
            tilt_cos = max(
                config.MIN_TILT_COMPENSATION_COSINE,
                math.cos(roll) * math.cos(pitch),
            )
            thrust = thrust / tilt_cos
            if in_takeoff:
                thrust = max(thrust, config.POSE_DEBUG_TAKEOFF_THRUST)
            thrust = max(
                config.POSE_DEBUG_THRUST_MIN,
                min(0.34, thrust),
            )

            vd = 0.0
            requested_yaw_rate = 0.0
            vn = ve = 0.0
            kalman_direct = True
            self.data.setdefault('kalman_path', {}).update(
                {
                    'phase': phase,
                    'vz_est': self._pose_vz,
                    'des_pitch': des_pitch,
                    'des_roll': des_roll,
                    'pitch_deg': (
                        math.degrees(pitch) if math.isfinite(pitch) else None
                    ),
                    'roll_deg': (
                        math.degrees(roll) if math.isfinite(roll) else None
                    ),
                    'range_m': range_m,
                    'locked_range_m': self._pose_locked_range,
                    'nx': nx,
                    'ny': ny,
                    'locked_nx': self._pose_locked_nx,
                    'locked_ny': self._pose_locked_ny,
                    'ax': ax,
                    'ay': ay,
                    'v_fwd': self._pose_v_fwd,
                    'v_right': self._pose_v_right,
                    'pitch_rate': pitch_rate,
                    'roll_rate': roll_rate,
                    'attitude_hold': False,
                    'thrust': thrust,
                    'source': 'pose_debug',
                }
            )
            self._log_safety(None)
        elif kalman_direct:
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
                # OPENCV_RATE_SIGN_YAW=+1 is for IBVS planners that already
                # bake the sim inversion. Run 043812: +cmd produced left
                # turns and walked the gate off-frame.
                yaw_rate = yaw_rate * config.RATE_SIGN_YAW
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

            # Body velocity → desired lean angles.  OpenCV uses its independently
            # calibrated lateral sign because the live VQ2 rate path does not
            # share the demonstration replay convention.
            forward_lean_gain = (
                config.OPENCV_KP_LEAN
                if planner_mode.startswith('opencv_')
                else config.KP_LEAN
            )
            d_pitch = v_fwd * forward_lean_gain
            d_roll  = (
                v_right
                * _LATERAL_LEAN_GAIN
                * config.OPENCV_LATERAL_LEAN_SIGN
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
        yaw_hold_active = False
        if not kalman_direct:
            if planner_mode.startswith('opencv_'):
                measured_yaw_rate = (
                    yawspeed * config.OPENCV_YAW_GYRO_SIGN
                )
                braking_to_hold_heading = (
                    abs(requested_yaw_rate)
                    <= config.OPENCV_YAW_BRAKE_REQUEST_DEADBAND
                )
                if not braking_to_hold_heading:
                    self._yaw_hold_target = None
                if braking_to_hold_heading and vio_att is not None:
                    # Heading hold: capture the VIO yaw at the moment the planner
                    # stops turning and actively hold it, instead of the passive
                    # brake gains (which default to zero, i.e. free drift).
                    if self._yaw_hold_target is None:
                        self._yaw_hold_target = yaw
                        self._yaw_pid.reset()
                    yaw_error = math.atan2(
                        math.sin(self._yaw_hold_target - yaw),
                        math.cos(self._yaw_hold_target - yaw),
                    )
                    physical_yaw_rate = self._yaw_pid.update(
                        yaw_error,
                        dt,
                        measurement_rate=yawspeed,
                    )
                    # Positive sent yaw produces negative raw body-z gyro — the
                    # same axis flip OPENCV_YAW_GYRO_SIGN measures — so a desired
                    # physical yaw rate maps to the sent command through it too.
                    yaw_rate = physical_yaw_rate * config.OPENCV_YAW_GYRO_SIGN
                    yaw_hold_active = True
            if planner_mode.startswith('opencv_') and not yaw_hold_active:
                feedback_kp = (
                    config.OPENCV_YAW_BRAKE_FEEDBACK_KP
                    if braking_to_hold_heading
                    else config.OPENCV_YAW_RATE_FEEDBACK_KP
                )
                feedback_limit = (
                    config.OPENCV_YAW_BRAKE_FEEDBACK_LIMIT
                    if braking_to_hold_heading
                    else config.OPENCV_YAW_RATE_FEEDBACK_LIMIT
                )
                yaw_rate_feedback = max(
                    -feedback_limit,
                    min(
                        feedback_limit,
                        feedback_kp
                        * (requested_yaw_rate - measured_yaw_rate),
                    ),
                )
                corrected_yaw_rate = requested_yaw_rate + yaw_rate_feedback
                # Rate feedback may reduce a requested turn to zero, but it must
                # never command a turn in the opposite direction.  Heading-capture
                # braking is handled separately above when the request is zero.
                if (
                    not braking_to_hold_heading
                    and requested_yaw_rate * corrected_yaw_rate < 0.0
                ):
                    corrected_yaw_rate = 0.0
                    yaw_rate_feedback = -requested_yaw_rate
                yaw_rate = corrected_yaw_rate * config.OPENCV_RATE_SIGN_YAW
            elif not planner_mode.startswith('opencv_'):
                yaw_rate = requested_yaw_rate * config.RATE_SIGN_YAW
            yaw_rate = max(
                -config.MAX_RATE_RAD_S,
                min(config.MAX_RATE_RAD_S, yaw_rate),
            )

            # Thrust: with a fresh VIO velocity the vertical loop is CLOSED — the
            # PI tracks the commanded NED down-velocity ``vd`` against the VIO's
            # vz belief. Without VIO (or during a vision-starved dropout) thrust
            # falls back to the open-loop lean-compensated hover baseline.
            # vd < 0 means climb → add thrust; vd > 0 means descend → reduce thrust.
            vio_vz = None
            if vio_pos is not None:
                candidate_vz = vio_pos.get('vz')
                if candidate_vz is not None and math.isfinite(candidate_vz):
                    vio_vz = float(candidate_vz)
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
            # pose_debug needs a higher open-loop hover (0.24 drops the craft).
            hover_base = (
                config.POSE_DEBUG_HOVER_THRUST
                if planner_mode.startswith('pose_debug')
                else config.HOVER_THRUST
            )
            level_thrust = hover_base + thrust_adjustment
            takeoff_s = (
                1.0
                if planner_mode.startswith('pose_debug')
                else config.TAKEOFF_DURATION_S
            )
            if (
                telemetry_ok
                and self._armed_at is not None
                and time.monotonic() - self._armed_at < takeoff_s
            ):
                # Lift off the deck, then hold hover_base (0.28 for pose_debug).
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
            yaw_rate = max(
                -config.MAX_RATE_RAD_S,
                min(config.MAX_RATE_RAD_S, yaw_rate),
            )
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
            'vio_attitude_active': vio_att is not None,
            'thrust_closed_loop': thrust_closed_loop,
            'yaw_hold_active': yaw_hold_active,
            'yaw_hold_target': self._yaw_hold_target,
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
