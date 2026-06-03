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
# VELOCITY CONTROL  (SET_POSITION_TARGET_LOCAL_NED)
#
# The spec's supported Client->Sim control messages are SET_POSITION_TARGET_LOCAL_NED
# and SET_ATTITUDE_TARGET (NOT actuator control — bug #4). We use NED velocity + a yaw
# setpoint: ignore position, acceleration and yaw-rate; command velocity and yaw.
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

    def update(self):
        target = self.planner.compute_target() if self.planner is not None else None
        dry_run = self.data.get('dry_run', True)

        if target is not None and target['mode'] == 'velocity':
            vn, ve, vd = target['vel_ned']
            yaw = target['yaw']
            if not dry_run:
                send_velocity_ned(self.sim_conn, self.system_boot_ms, vn, ve, vd, yaw)
            self._maybe_log(target, dry_run)

        time.sleep(self._dt)

    def _maybe_log(self, target, dry_run):
        """Throttled, human-readable status (~2 Hz) — fine outside timed runs."""
        now = time.time()
        if now - self._last_log < 0.5:
            return
        self._last_log = now
        vn, ve, vd = target['vel_ned']
        tag = "DRY" if dry_run else "FLY"
        rng = target.get('range_m')
        rng_s = f"{rng:5.1f}m" if rng is not None else "  ?  "
        print(f"[{tag}] src={target['source']:>14} range={rng_s} "
              f"vel_ned=({vn:+.2f},{ve:+.2f},{vd:+.2f}) yaw={target['yaw']:+.2f}",
              flush=True)

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
