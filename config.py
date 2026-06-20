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
#   USE_TELEOP  : True  = manual keyboard control (teleop).
#                 False = autonomous flight (the spline / waypoint planner below).
#   USE_SPLINE  : autonomous mode only. True  = follow a smooth Catmull-Rom SPLINE through
#                 the mission waypoints at constant cruise speed (continuous flight, this
#                 branch's default -- see spline_planner.py). False = the stop-at-each
#                 waypoint planner (planner.py).
# ======================================================================================
DRY_RUN = False
DEBUG_VISION = False
LOGGING = True
USE_VISION = False
USE_TELEOP = False
USE_SPLINE = True

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
#claude --resume 48ec61fe-f15b-4b9c-b0ab-6f8a5368fc07
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
MAX_SPEED = 60.0
MAX_VSPEED = 25.5
MAX_WP_DIST_M = 60.0

# ======================================================================================
# Spline planner (spline_planner.py) -- continuous flight through the waypoints.
#   CRUISE_SPEED: m/s flown along the path (constant; tapers only on the final approach).
#                 Must be <= MAX_SPEED (the velocity magnitude is still capped there).
#   LOOKAHEAD_* : pure-pursuit carrot distance, SPEED-SCALED -- lookahead = clamp(
#                 LOOKAHEAD_TIME * speed, LOOKAHEAD_M, LOOKAHEAD_MAX). Short when slow (dense
#                 gates / the climb-out off the line -> tight tracking, incl. altitude), long
#                 when fast (smooth straights, no wobble). A single fixed value can't do both
#                 when the gate spacing (a few m) is smaller than the carrot needed at speed.
#                 LOOKAHEAD_M is the FLOOR (slow/dense), LOOKAHEAD_MAX the cap (fast).
# When MAX_WP_DIST_M is used here it bounds CROSS-TRACK error off the path (not distance to
# a single waypoint): farther off the path than this -> hover and brake back toward it.
# ======================================================================================
CRUISE_SPEED = 60.0
LOOKAHEAD_M = 2.0      # floor: carrot distance (m) when slow / at the dense early gates
LOOKAHEAD_TIME = 0.5   # seconds of travel ahead: lookahead grows as 0.5 * current speed
LOOKAHEAD_MAX = 7.0    # cap: carrot distance (m) at high speed (smooths the fast straights)

# KP_VERT_PATH: vertical PATH-altitude correction gain (1/s). The controller's vertical loop
# tracks vertical VELOCITY only, and the carrot's vertical pull is diluted at speed (the carrot
# is mostly horizontal), so on climbs/descents the drone settles BELOW the path and clips gate
# BOTTOMS. This adds vd += KP_VERT_PATH * (planned_alt_error), a position term that does NOT
# shrink with horizontal speed, pulling the drone back onto the path's height. ~1/s means a 1 m
# altitude error commands ~1 m/s of climb/descend (capped at MAX_VSPEED). 0.0 = old behaviour
# (carrot-only vertical). Raise if it still flies low through gates; lower if altitude hunts.
KP_VERT_PATH = 1.2

# ======================================================================================
# Curvature-aware speed (spline_planner.py) -- so the drone SLOWS for corners instead of
# overshooting them at high CRUISE_SPEED, then re-accelerates on the straights. The planner
# caps speed at sqrt(A_LAT_MAX / path_curvature) and brakes ahead of corners / the final
# waypoint at up to A_LON_MAX. These are a no-op when the path is gentle enough to hold
# cruise everywhere (e.g. at low CRUISE_SPEED), so slow runs are unchanged. Set them to the
# airframe's real limits: HIGHER = carries more speed through bends but risks overshoot;
# LOWER = slower, safer racing lines.
#
# CRITICAL COUPLING: A_LAT_MAX is the cornering accel the AIRFRAME must actually produce,
# and a leaning quad can only make g*tan(roll). So A_LAT_MAX must stay <= g*tan(MAX_LEAN_RAD)
# (the ROLL cap in controller.py) or the planner commands corner speeds the drone physically
# can't turn at -> it drifts WIDE and clips the gate. With the roll cap now at 52deg the
# ceiling is g*tan52 = ~12.5 m/s^2, so A_LAT_MAX=11 keeps margin (needs ~48deg roll). If you
# drop the roll cap back to 45deg, drop A_LAT_MAX back to ~9 (g*tan45 = 9.8) to match.
# The tightest corner on the captured course (R=8.9m, right at the start / the wp0-wp3
# climb-out) is what limits the FIRST segment's speed -- raising A_LAT_MAX speeds that bend
# too (that's why the drone only mildly pitches there: it's the sharpest turn on the course).
# A_LON_MAX is the SAFE speed lever: it doesn't raise any corner speed, just lets the drone
# accelerate onto the straights and brake later into corners (more time spent at top speed).
# ======================================================================================
A_LAT_MAX = 6.0     # m/s^2 max lateral (cornering) accel -- keep <= g*tan(roll cap); see above
A_LON_MAX = 9.0      # m/s^2 max longitudinal (accel/brake) -- safe to raise; no corner-speed risk

# FINISH_SPEED: speed (m/s) allowed AT the last waypoint. A race doesn't need to stop -- the
# timer ends when you cross the final gate -- so braking to a halt there wastes the run.
# 0.0 = brake to a full stop (settle on the last point). A positive value CROSSES the finish
# at speed (much faster overall): the drone keeps the last gate's corner/cruise speed instead
# of decelerating for ~CRUISE^2/(2*A_LON) metres. It will overshoot ~FINISH_SPEED^2/(2*A_LON)
# m PAST the finish before the completion-hover brakes it, so only raise it toward CRUISE if
# there is open space beyond the last gate. (Looping missions ignore this -- they never stop.)
FINISH_SPEED = 8.0

# ======================================================================================
# Manual teleop (teleop.py)
#   TELEOP_SPEED      : m/s horizontal at full stick (one WASD key held).
#   TELEOP_VSPEED     : m/s climb/descend at full stick (Space / C).
#   TELEOP_YAWRATE_DPS: deg/s yaw rate while Q/E are held.
#   TELEOP_YAW_SIGN   : flip to -1.0 if Q/E turn the nose the wrong way.
# ======================================================================================
TELEOP_SPEED = 10.0
TELEOP_VSPEED = 1.5
TELEOP_YAWRATE_DPS = 60.0
TELEOP_YAW_SIGN = 1.0
