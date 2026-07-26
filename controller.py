import math
import time

from pymavlink import mavutil

# --------------------------------------------------------------------------------------
# RESET COMMAND
MAVLINK_CMD_SIM_RESET = 31000

# --------------------------------------------------------------------------------------
# Control loop rate. Spec caps Client->Sim commands at < 100 Hz (was 250 in the
# example — bug #3). 60 Hz leaves margin while staying well above the 2 Hz heartbeat min.
# --------------------------------------------------------------------------------------
CONTROL_HZ = 60

# --------------------------------------------------------------------------------------
# VELOCITY CONTROL  (SET_POSITION_TARGET_LOCAL_NED)  -- NOT used by this sim.
#
# The spec lists SET_POSITION_TARGET_LOCAL_NED as a supported message, but the DCL sim
# is a Betaflight-style FPV racer running in ACRO/ANGLE flight modes, which have NO
# velocity or position loop — these setpoints are silently ignored and the armed quad
# climbs away (logs/run_1780516557.jsonl). Kept for reference / the offline send-path
# test; actual flight uses ATTITUDE control below.
# --------------------------------------------------------------------------------------
VELOCITY_NED_MASK = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
)


def send_velocity_ned(mavlink_conn, system_boot_ms, vn, ve, vd, yaw):
    """Command a NED velocity (m/s) and yaw setpoint (rad, 0=North, +East)."""
    now_ms = int(time.time() * 1000)
    mavlink_conn.mav.set_position_target_local_ned_send(
        now_ms - system_boot_ms,
        mavlink_conn.target_system,
        mavlink_conn.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        VELOCITY_NED_MASK,
        0.0, 0.0, 0.0,      # position (ignored)
        vn, ve, vd,         # velocity NED
        0.0, 0.0, 0.0,      # acceleration (ignored)
        yaw,                # yaw setpoint
        0.0                 # yaw rate (ignored)
    )


# --------------------------------------------------------------------------------------
# ATTITUDE CONTROL  (SET_ATTITUDE_TARGET)  -- the control path the sim actually flies.
#
# The sim's ANGLE mode is self-levelling: we command a desired attitude (roll/pitch/yaw)
# + collective thrust and the flight controller holds that attitude. We turn the
# planner's desired NED velocity into a small forward/lateral LEAN (pitch/roll) and a
# thrust that tracks the desired vertical velocity. Mask uses the quaternion + thrust
# and ignores the body-rate fields.
#
# These gains MUST be tuned against the sim. HOVER_THRUST is the critical one: the
# first observed flight climbed away steadily (~5.5 m/s, to 100 m+) because 0.5 was
# well above this racing quad's true hover throttle, and the vertical loop was too weak
# to pull it back (logs/run_1780521287.jsonl). Tuning order: find the HOVER_THRUST that
# holds altitude (level, zero lean) — if it climbs lower it, if it sinks raise it — then
# raise KP_LEAN for responsiveness. KP_THRUST sets how hard we fight vertical error.
# --------------------------------------------------------------------------------------
# Vertical loop, tuned against logs/ (2026-06-05). With the body-rate leveling loop now
# stable, a clean stationary-flight fit of vertical accel vs. commanded thrust gives
# accel_down ~= -58.2*thrust + 15.57, i.e. zero-accel HOVER ~= 0.268 (much lower than the
# 0.33 used before stabilisation). The old 0.33 added an upward kick every tick: the loop
# only clawed thrust back to ~0.26 AFTER the quad had already ballooned to ~8 m, which also
# pushed the gate out of the up-tilted camera's view (vision dropped at t~=4s and never
# recovered) — i.e. the climb caused BOTH the fly-over and the vision loss. Set to the
# measured hover and roughly double KP_THRUST so altitude-rate error is arrested quickly.
# Re-tune from the next log: climbs -> lower HOVER_THRUST; sinks -> raise it.
HOVER_THRUST = 0.27        # collective thrust (0..1) that roughly holds altitude — TUNE FIRST
KP_THRUST = 0.25           # extra thrust per (m/s) of vertical-velocity error (more authority)
THRUST_MIN, THRUST_MAX = 0.2, 0.9   # floor keeps prop wash / attitude authority while descending
KP_LEAN = 0.3           # rad of lean per (m/s) of desired horizontal velocity
MAX_LEAN_RAD = math.radians(20.0)   # cap on commanded pitch/roll angle

# --------------------------------------------------------------------------------------
# Control is BODY-RATE (ACRO), with our own outer attitude->rate leveling loop.
#
# Every prior strategy that relied on the sim holding an ATTITUDE setpoint failed the
# same way (logs 2026-06-05): commanding perfectly level (roll=pitch=0, yaw_rate=0) left
# the drone pinned at ~75deg pitch with near-zero body rates — i.e. it never self-levels.
# Thrust IS honoured (it climbs/descends on command) but the quaternion is ignored. That
# is ACRO (raw-rate) behaviour: the FC only obeys body RATES. The ANGLE-mode switch is
# not taking effect, so depending on it is hopeless.
# Fix: stop sending an attitude setpoint entirely. We close the angle loop ourselves and
# command body rates the ACRO FC will track:
#   roll_rate  = KP_ATT * (roll_des  - roll_meas)
#   pitch_rate = KP_ATT * (pitch_des - pitch_meas)
#   yaw_rate   = KP_YAW * (yaw_des   - yaw_meas)        (heading hold / turn-to-gate)
# all clamped to RATE_MAX. roll_des/pitch_des still come from velocity_to_attitude (lean
# to translate) and are low-passed first to damp vision jitter.
# --------------------------------------------------------------------------------------
KP_ATT = 3.0                        # body roll/pitch rate (rad/s) per rad of angle error
KP_YAW = 2.0                        # body yaw rate (rad/s) per rad of heading error
RATE_MAX = math.radians(220.0)      # cap on any commanded body rate
# Yaw-rate cap kept low on purpose: the camera FOV is narrow, so a fast yaw sweeps the
# gate out of view and kills tracking. A 104 deg/s slew (chasing a garbage detection) did
# exactly that (log 2026-06-05). 70 deg/s still turns the nose toward gates briskly while
# keeping them in frame; the planner's plausibility gate removes the bad-frame swings.
YAW_RATE_MAX = math.radians(70.0)   # tighter cap on the yaw rate specifically
LEAN_LPF_ALPHA = 0.35               # roll/pitch desired-angle smoothing (1.0 = off)
AI_ACTION_TIMEOUT_NS = 300_000_000
AI_RATE_SLEW_PER_TICK = math.radians(18.0)
AI_THRUST_SLEW_PER_TICK = 0.035

# Per-axis sign of the body-rate command needed for NEGATIVE feedback. This sim's
# SET_ATTITUDE_TARGET rate axes do NOT share a consistent convention with its reported
# ATTITUDE Euler angles, and roll differs from pitch (logs 2026-06-05):
#   * PITCH angle responds OPPOSITE to a commanded pitch rate -> negate (-1) to stabilise.
#     (verified: cmdPitch->measured-pitch correlation -0.97, att_err pitch ~0.30 = clean.)
#   * ROLL angle responds with the SAME sign as the commanded roll rate -> do NOT negate
#     (+1). With -1 the loop was positive feedback and roll ramped to a tumble every run.
#   * YAW is stable with -1 (heading-hold error ~0.04).
# If an axis ever diverges from level again, flip its sign here first.
RATE_SIGN_ROLL = +1.0
RATE_SIGN_PITCH = -1.0
RATE_SIGN_YAW = -1.0

# Ignore the attitude quaternion; command the three body rates + thrust.
ATTITUDE_CONTROL_MASK = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE

_IDENTITY_QUAT = [1.0, 0.0, 0.0, 0.0]   # sent but ignored (ATTITUDE_IGNORE set)


def _clamp(x, lim):
    """Clamp x to [-lim, +lim]."""
    return max(-lim, min(lim, x))


def send_rate_target(mavlink_conn, system_boot_ms, roll_rate, pitch_rate, yaw_rate, thrust):
    """Command body angular RATES (rad/s) + collective thrust (0..1) — ACRO control."""
    now_ms = int(time.time() * 1000)
    mavlink_conn.mav.set_attitude_target_send(
        now_ms - system_boot_ms,
        mavlink_conn.target_system,
        mavlink_conn.target_component,
        ATTITUDE_CONTROL_MASK,
        _IDENTITY_QUAT,
        roll_rate, pitch_rate, yaw_rate,
        thrust,
    )


def velocity_to_attitude(vel_ned, yaw_cmd, yaw_now, vel_now):
    """Map a desired NED velocity + yaw into (roll, pitch, yaw, thrust) for ANGLE mode.

    - thrust tracks the desired vertical velocity ``vd`` (NED, +down): more thrust when
      we are sinking relative to the command.
    - horizontal control is a VELOCITY loop: lean is proportional to the velocity ERROR
      (desired - measured), rotated into the heading frame. A fixed lean is a fixed
      ACCELERATION, so the old open-loop "lean = KP*desired_velocity" never stopped
      accelerating — log 2026-06-05 shows a 2 m/s command building to ~6 m/s, then
      coasting through the gate because a zero-lean "hover" does not brake. Closing the
      loop on the error makes lean fall to zero as we reach target speed and lean
      BACKWARD to brake when overspeeding or commanded to hover.

    ``vel_now`` is the measured NED velocity (vn, ve, vz).
    """
    vn, ve, vd = vel_ned
    vn_now, ve_now, vz_now = vel_now

    thrust = HOVER_THRUST + KP_THRUST * (vz_now - vd)
    thrust = max(THRUST_MIN, min(THRUST_MAX, thrust))

    # Horizontal velocity error in world NED, then rotate into the body heading frame.
    en, ee = (vn - vn_now), (ve - ve_now)
    c, s = math.cos(yaw_now), math.sin(yaw_now)
    e_fwd = c * en + s * ee          # body-forward velocity error
    e_lat = -s * en + c * ee         # body-right velocity error
    # Pitch sign is INVERTED on this sim (same asymmetry as RATE_SIGN_PITCH=-1 vs
    # RATE_SIGN_ROLL=+1): the rate loop faithfully holds the commanded pitch angle, but
    # the textbook "nose-down to go forward" sign drove the quad BACKWARD (verified
    # 2026-06-05), so forward uses +KP_LEAN. Roll/lateral was already correct.
    pitch = max(-MAX_LEAN_RAD, min(MAX_LEAN_RAD, KP_LEAN * e_fwd))   # forward translation
    roll = max(-MAX_LEAN_RAD, min(MAX_LEAN_RAD, KP_LEAN * e_lat))    # roll right to go right
    return roll, pitch, yaw_cmd, thrust


# --------------------------------------------------------------------------------------
# Control Loop
# --------------------------------------------------------------------------------------
class Controller:
    """Drives the drone toward the planner's target each tick.

    Reads ``shared_data['dry_run']``: when True (the default until perception is
    validated), it computes and logs the command but does NOT send flight setpoints,
    so the pipeline can be validated safely. Set it False to actually fly.
    """

    def __init__(self, sim_conn, data, system_boot_ms, planner=None):
        self.sim_conn = sim_conn
        self.data = data
        self.system_boot_ms = system_boot_ms
        self.planner = planner
        self._dt = 1.0 / CONTROL_HZ
        self._last_log = 0.0
        # Command-conditioning state (see KP_YAW / LEAN_LPF above).
        self._roll_cmd = 0.0        # last low-passed roll command (rad)
        self._pitch_cmd = 0.0       # last low-passed pitch command (rad)
        self._last_ai_command = [HOVER_THRUST, 0.0, 0.0, 0.0]

    @staticmethod
    def _wrap(a):
        """Wrap an angle to (-pi, pi]."""
        return math.atan2(math.sin(a), math.cos(a))

    def _yaw_rate_cmd(self, yaw_target, yaw_now):
        """Body yaw RATE (rad/s) to turn the nose toward yaw_target (shortest path)."""
        err = self._wrap(yaw_target - yaw_now)
        return max(-YAW_RATE_MAX, min(YAW_RATE_MAX, KP_YAW * err))

    def _lpf_lean(self, roll, pitch):
        """First-order low-pass on the roll/pitch commands to damp per-frame jitter."""
        a = LEAN_LPF_ALPHA
        self._roll_cmd = a * roll + (1.0 - a) * self._roll_cmd
        self._pitch_cmd = a * pitch + (1.0 - a) * self._pitch_cmd
        return self._roll_cmd, self._pitch_cmd

    def _telemetry(self):
        """Measured attitude (roll, pitch, yaw, rad) and NED velocity (vn, ve, vz, m/s).

        Velocity feeds the horizontal velocity loop (and vz the thrust loop); +down.

        CRITICAL: use LOCAL_POSITION_NED velocity (``position_ned`` vx/vy/vz), which the
        MAVLink spec defines in WORLD NED. The previous code preferred ODOMETRY ``vel``,
        but ODOMETRY velocity is in the BODY (child) frame — feeding it into the loop
        (which rotates by yaw as if it were NED) turned the velocity feedback into the
        wrong frame, so the loop ran effectively OPEN: it leaned proportional to the
        DESIRED velocity, never braked, and saturated roll+pitch at the cap while the
        drone accelerated away (16 m/s, +7 m/s climb, ignoring hover/descend — log
        2026-06-06). The logger reads position_ned and showed the true NED velocity all
        along; the controller just wasn't using it. Prefer it here too.
        """
        with self.data['lock']:
            att = self.data.get('attitude')
            pos = self.data.get('position_ned')
            odo = self.data.get('odometry')
        roll_now = att['roll'] if att else 0.0
        pitch_now = att['pitch'] if att else 0.0
        yaw_now = att['yaw'] if att else 0.0
        vn_now = ve_now = vz_now = 0.0
        if pos is not None and pos.get('vx') is not None:
            vn_now = pos.get('vx', 0.0)
            ve_now = pos.get('vy', 0.0)
            vz_now = pos.get('vz', 0.0)
        elif odo is not None and odo.get('vel') is not None:
            # Fallback only: ODOMETRY vel is body-frame, so rotate it into NED with the
            # measured yaw before the loop consumes it (better than feeding raw body vel).
            vx_b, vy_b, vz_b = odo['vel']
            c, s = math.cos(yaw_now), math.sin(yaw_now)
            vn_now = c * vx_b - s * vy_b
            ve_now = s * vx_b + c * vy_b
            vz_now = vz_b
        return (float(roll_now), float(pitch_now), float(yaw_now),
                float(vn_now), float(ve_now), float(vz_now))

    def update(self):
        target = self.planner.compute_target() if self.planner is not None else None
        dry_run = self.data.get('dry_run', True)

        if target is not None and target.get('source') in ('ai_active', 'ai_safe_hover'):
            self._apply_ai_or_safe(target, dry_run)
            time.sleep(self._dt)
            return

        if target is not None and target['mode'] == 'velocity':
            roll_now, pitch_now, yaw_now, vn_now, ve_now, vz_now = self._telemetry()
            roll_des, pitch_des, yaw_target, thrust = velocity_to_attitude(
                target['vel_ned'], target['yaw'], yaw_now, (vn_now, ve_now, vz_now))
            # Smooth the desired lean angles, then close the angle loop OURSELVES and
            # command body rates (the sim is ACRO: it tracks rates, ignores attitude).
            roll_des, pitch_des = self._lpf_lean(roll_des, pitch_des)
            # Body rates with the PER-AXIS sign correction (see RATE_SIGN_* above); each
            # axis is tuned so its angle loop is negative feedback (roll != pitch here).
            roll_rate = RATE_SIGN_ROLL * _clamp(KP_ATT * self._wrap(roll_des - roll_now), RATE_MAX)
            pitch_rate = RATE_SIGN_PITCH * _clamp(KP_ATT * self._wrap(pitch_des - pitch_now), RATE_MAX)
            yaw_rate = RATE_SIGN_YAW * self._yaw_rate_cmd(yaw_target, yaw_now)
            if not dry_run:
                send_rate_target(self.sim_conn, self.system_boot_ms,
                                 roll_rate, pitch_rate, yaw_rate, thrust)
            self._publish_control(
                target, (roll_des, pitch_des, yaw_target, thrust),
                (roll_rate, pitch_rate, yaw_rate),
                (roll_now, pitch_now, yaw_now, vz_now), dry_run)
            self._maybe_log(target, (roll_des, pitch_des, yaw_target, thrust), dry_run)

        time.sleep(self._dt)

    def _apply_ai_or_safe(self, target, dry_run):
        """Send one bounded learned-policy action, or an explicit safe hover.

        The AI provider contract uses physical body rates and thrust. This path
        is mutually exclusive with the OpenCV velocity target, so two systems
        can never issue commands on the same control tick.
        """
        now = time.time_ns()
        with self.data['lock']:
            action = self.data.get('ai_action')
        fresh = (
            target.get('source') == 'ai_active'
            and action is not None
            and action.get('ts')
            and now - action['ts'] <= AI_ACTION_TIMEOUT_NS
        )
        desired = [HOVER_THRUST, 0.0, 0.0, 0.0]
        source = 'ai_safe_hover'
        if fresh:
            desired = [
                max(THRUST_MIN, min(THRUST_MAX, float(action['thrust']))),
                _clamp(float(action['roll_rate']), RATE_MAX),
                _clamp(float(action['pitch_rate']), RATE_MAX),
                _clamp(float(action['yaw_rate']), YAW_RATE_MAX),
            ]
            source = 'ai'

        conditioned = []
        for index, (old, new) in enumerate(zip(self._last_ai_command, desired)):
            limit = AI_THRUST_SLEW_PER_TICK if index == 0 else AI_RATE_SLEW_PER_TICK
            conditioned.append(old + _clamp(new - old, limit))
        self._last_ai_command = conditioned
        thrust, roll_rate, pitch_rate, yaw_rate = conditioned
        if not dry_run:
            send_rate_target(
                self.sim_conn,
                self.system_boot_ms,
                roll_rate,
                pitch_rate,
                yaw_rate,
                thrust,
            )
        with self.data['lock']:
            self.data['control'] = {
                'ts': now,
                'dry_run': dry_run,
                'source': source,
                'vel_cmd_ned': (0.0, 0.0, 0.0),
                'cmd': {
                    'roll_rate': roll_rate,
                    'pitch_rate': pitch_rate,
                    'yaw_rate': yaw_rate,
                    'thrust': thrust,
                },
            }

    def _publish_control(self, target, attitude_des, rate_cmd, meas, dry_run):
        """Expose the actual command (and the telemetry it used) for the logger."""
        roll_des, pitch_des, yaw_des, thrust = attitude_des
        roll_rate, pitch_rate, yaw_rate = rate_cmd
        roll_now, pitch_now, yaw_now, vz_now = meas
        with self.data['lock']:
            self.data['control'] = {
                'ts': time.time_ns(),
                'dry_run': dry_run,
                'source': target.get('source'),
                'range_m': target.get('range_m'),
                'vel_cmd_ned': tuple(float(v) for v in target['vel_ned']),
                'yaw_target': float(target.get('yaw', 0.0)),
                'meas_yaw': float(yaw_now),
                'meas_vz': float(vz_now),
                # 'cmd' carries the DESIRED attitude (what we want the airframe to hold)
                # plus the body RATES we actually sent to get there.
                'cmd': {'roll': float(roll_des), 'pitch': float(pitch_des),
                        'yaw': float(yaw_des), 'thrust': float(thrust),
                        'roll_rate': float(roll_rate), 'pitch_rate': float(pitch_rate),
                        'yaw_rate': float(yaw_rate)},
            }

    def _maybe_log(self, target, attitude_cmd, dry_run):
        """Throttled, human-readable status (~2 Hz) — fine outside timed runs."""
        now = time.time()
        if now - self._last_log < 0.5:
            return
        self._last_log = now
        vn, ve, vd = target['vel_ned']
        roll, pitch, yaw, thrust = attitude_cmd
        tag = "DRY" if dry_run else "FLY"
        rng = target.get('range_m')
        rng_s = f"{rng:5.1f}m" if rng is not None else "  ?  "
        print(f"[{tag}] src={target['source']:>14} range={rng_s} "
              f"vel_ned=({vn:+.2f},{ve:+.2f},{vd:+.2f}) "
              f"att=(r{roll:+.2f},p{pitch:+.2f},y{yaw:+.2f}) thr={thrust:.2f}",
              flush=True)

    # -------------------------------
    # Mode entry (see PLAN.md §8.8). The sim is a Betaflight-style FPV racer; it starts
    # in ACRO and we switch it to ANGLE (self-levelling attitude mode) so a stable
    # attitude+thrust controller can fly it. The mode switch is accepted more reliably
    # with a setpoint stream already flowing, so we prime first. Order: stream holds ->
    # set mode -> arm. (In ACRO, with no valid command, the armed quad climbs away —
    # logs/run_1780516557.jsonl.)
    # -------------------------------
    def hold(self):
        """Send one zero-rate, hover-thrust command (primes the stream; inert pre-arm)."""
        send_rate_target(self.sim_conn, self.system_boot_ms, 0.0, 0.0, 0.0, HOVER_THRUST)

    def prime_setpoint_stream(self, seconds=1.0, hz=50.0):
        """Stream level-attitude holds so the sim will accept the mode switch."""
        n = max(1, int(seconds * hz))
        for _ in range(n):
            self.hold()
            time.sleep(1.0 / hz)

    def request_offboard_mode(self):
        """Ask the sim to enter a self-levelling attitude mode (ANGLE / STABILIZE).

        Resolves the autopilot's own mode map when available and prefers an
        attitude-stabilised mode; falls back to the PX4 ALTCTL/STABILIZED custom mode.
        Returns the mode name/string requested (for logging).
        """
        conn = self.sim_conn
        try:
            mode_map = conn.mode_mapping() or {}
        except Exception:
            mode_map = {}
        print(f"Sim mode map: {sorted(mode_map) if mode_map else 'unknown (no mode_mapping)'}", flush=True)
        for name in ('ANGLE', 'STABILIZE', 'STABILIZED', 'ALTCTL', 'GUIDED'):
            if name in mode_map:
                conn.mav.command_long_send(
                    conn.target_system, conn.target_component,
                    mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                    mode_map[name], 0, 0, 0, 0, 0)
                return name
        # Fallback: PX4 custom main-mode 6 (used by the example; landed the sim in ANGLE).
        conn.mav.command_long_send(
            conn.target_system, conn.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6, 0, 0, 0, 0, 0)
        return 'fallback(custom-mode 6)'

    # -------------------------------
    # Arm the drone
    # -------------------------------
    def arm(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,  # arm
            0, 0, 0, 0, 0, 0
        )

    def send_sim_reset_command(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            MAVLINK_CMD_SIM_RESET,
            0,  # confirmation
            0, 0, 0, 0, 0, 0, 0
        )
