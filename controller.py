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
HOVER_THRUST = 0.15        # collective thrust (0..1) that roughly holds altitude — TUNE FIRST
KP_THRUST = 0.05           # extra thrust per (m/s) of vertical-velocity error (more authority)
THRUST_MIN, THRUST_MAX = 0.05, 0.9  # allow near-zero thrust so we can actually descend
KP_LEAN = 0.3           # rad of lean per (m/s) of desired horizontal velocity
MAX_LEAN_RAD = math.radians(20.0)   # cap on commanded pitch/roll angle

ATTITUDE_CONTROL_MASK = (
    mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE |
    mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_PITCH_RATE_IGNORE |
    mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_YAW_RATE_IGNORE
)


def _euler_to_quat(roll, pitch, yaw):
    """Aerospace ZYX (yaw-pitch-roll) Euler angles -> (w, x, y, z) attitude quaternion."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return [
        cr * cp * cy + sr * sp * sy,   # w
        sr * cp * cy - cr * sp * sy,   # x
        cr * sp * cy + sr * cp * sy,   # y
        cr * cp * sy - sr * sp * cy,   # z
    ]


def send_attitude_target(mavlink_conn, system_boot_ms, roll, pitch, yaw, thrust):
    """Command a desired attitude (rad) + collective thrust (0..1) for ANGLE mode."""
    now_ms = int(time.time() * 1000)
    mavlink_conn.mav.set_attitude_target_send(
        now_ms - system_boot_ms,
        mavlink_conn.target_system,
        mavlink_conn.target_component,
        ATTITUDE_CONTROL_MASK,
        _euler_to_quat(roll, pitch, yaw),
        0.0, 0.0, 0.0,      # body rates (ignored)
        thrust,
    )


def velocity_to_attitude(vel_ned, yaw_cmd, yaw_now, vz_now):
    """Map a desired NED velocity + yaw into (roll, pitch, yaw, thrust) for ANGLE mode.

    - thrust tracks the desired vertical velocity ``vd`` (NED, +down): more thrust when
      we are sinking relative to the command.
    - horizontal velocity is rotated into the current heading frame, then forward speed
      -> nose-down pitch and rightward speed -> right roll, each capped at MAX_LEAN_RAD.
    """
    vn, ve, vd = vel_ned

    thrust = HOVER_THRUST + KP_THRUST * (vz_now - vd)
    thrust = max(THRUST_MIN, min(THRUST_MAX, thrust))

    c, s = math.cos(yaw_now), math.sin(yaw_now)
    v_fwd = c * vn + s * ve          # body-forward component of the desired velocity
    v_lat = -s * vn + c * ve         # body-right component
    pitch = max(-MAX_LEAN_RAD, min(MAX_LEAN_RAD, -KP_LEAN * v_fwd))  # nose-down to go fwd
    roll = max(-MAX_LEAN_RAD, min(MAX_LEAN_RAD, KP_LEAN * v_lat))    # roll right to go right
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

    def _telemetry(self):
        """Current heading (rad) and vertical velocity (m/s, NED +down) for the loop."""
        with self.data['lock']:
            att = self.data.get('attitude')
            pos = self.data.get('position_ned')
            odo = self.data.get('odometry')
        yaw_now = att['yaw'] if att else 0.0
        if pos is not None:
            vz_now = pos.get('vz', 0.0)
        elif odo is not None:
            vz_now = odo['vel'][2]
        else:
            vz_now = 0.0
        return float(yaw_now), float(vz_now)

    def update(self):
        target = self.planner.compute_target() if self.planner is not None else None
        dry_run = self.data.get('dry_run', True)

        if target is not None and target['mode'] == 'velocity':
            yaw_now, vz_now = self._telemetry()
            roll, pitch, yaw, thrust = velocity_to_attitude(
                target['vel_ned'], target['yaw'], yaw_now, vz_now)
            if not dry_run:
                send_attitude_target(self.sim_conn, self.system_boot_ms,
                                     roll, pitch, yaw, thrust)
            self._maybe_log(target, (roll, pitch, yaw, thrust), dry_run)

        time.sleep(self._dt)

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
        """Send one level-attitude hold (primes/maintains the stream; inert pre-arm)."""
        send_attitude_target(self.sim_conn, self.system_boot_ms, 0.0, 0.0, 0.0, HOVER_THRUST)

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
