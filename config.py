"""Central configuration for the AI-GP client (this branch).

Every run setting and tunable knob lives here as a plain Python constant -- edit this
file to configure a run. Nothing is read from OS environment variables anymore; main.py,
planner.py and teleop.py import what they need from here.

(Deep flight-control gains -- hover thrust, lean/rate gains, sign corrections -- still
live in controller.py, next to the tuning notes that explain them.)
"""

# ======================================================================================
# Connection -- the DCL sim's MAVLink UDP endpoint. Change these for a remote sim.
# ======================================================================================
SIM_SERVER_UDP_IP = "127.0.0.1"
SIM_SERVER_UDP_PORT = 14550

# ======================================================================================
# Run flags (main.py)
#   DRY_RUN     : True  = compute & log guidance but send NO flight setpoints (safe ground
#                         check; the planner runs but the drone stays put).
#                 False = actually fly -- the drone arms on startup. CAUTION.
#   DEBUG_VISION: write detection overlays (no effect here -- vision is off).
#   LOGGING     : write per-run JSONL logs under logs/ for offline tuning.
#   USE_VISION  : run the camera/detector pipeline (off on this branch).
#   USE_TELEOP  : True  = manual keyboard control (teleop, this branch's default).
#                 False = autonomous waypoint mission (loads MISSION_PATH below).
# ======================================================================================
DRY_RUN = False
DEBUG_VISION = False
LOGGING = True
USE_VISION = False
USE_TELEOP = True

# ======================================================================================
# Mission / paths (main.py)
#   CAPTURE_PATH : where teleop's B key writes captured waypoints.
#   MISSION_PATH : mission JSON the autonomous planner loads; if the file is missing it
#                  builds the SQUARE_* mission below instead.
#   SQUARE_*     : built-in square mission, used only when MISSION_PATH is absent.
#                  SQUARE_CCW False = clockwise (right turns).
# ======================================================================================
CAPTURE_PATH = 'captured_waypoints.json'
MISSION_PATH = 'mission.json'
SQUARE_SIDE_M = 5.0
SQUARE_ALT_M = 2.0
SQUARE_CCW = False

# ======================================================================================
# Autonomous planner (planner.py)
#   MAX_SPEED    : m/s cap on commanded velocity magnitude (speed scales with distance to
#                  the waypoint, then is capped here). If the drone can't reach this it has
#                  saturated its lean cap -- raise controller.MAX_LEAN_RAD / KP_LEAN.
#   MAX_VSPEED   : m/s cap on the vertical (climb/descend) component.
#   MAX_WP_DIST_M: runaway guard -- if the drone is farther than this from its ACTIVE
#                  waypoint it hovers (the velocity loop brakes back). MUST exceed the
#                  mission's LONGEST leg or it stalls at a waypoint (the captured course's
#                  longest leg is ~39 m, so 60 clears it while still catching a 100 m+
#                  flyaway).
# ======================================================================================
MAX_SPEED = 3.0
MAX_VSPEED = 1.5
MAX_WP_DIST_M = 60.0

# ======================================================================================
# Manual teleop (teleop.py)
#   TELEOP_SPEED      : m/s horizontal at full stick (one WASD key held).
#   TELEOP_VSPEED     : m/s climb/descend at full stick (Space / C).
#   TELEOP_YAWRATE_DPS: deg/s yaw rate while Q/E are held.
#   TELEOP_YAW_SIGN   : flip to -1.0 if Q/E turn the nose the wrong way.
# ======================================================================================
TELEOP_SPEED = 3.0
TELEOP_VSPEED = 1.5
TELEOP_YAWRATE_DPS = 60.0
TELEOP_YAW_SIGN = 1.0
