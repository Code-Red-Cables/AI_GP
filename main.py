#
# Sample Python client for the AI GP controller
#

import os
import threading
import time

from setup import setup_components
from mission import square_mission, load_mission
from teleop import print_controls

# Modify these properties if you want to run the server remotely for example
SIM_SERVER_UDP_IP = "127.0.0.1"
SIM_SERVER_UDP_PORT = 14550

# --------------------------------------------------------------------------------------
# Safety / debug flags.
#   DRY_RUN:  compute & log guidance but DO NOT send flight setpoints. Set True for a
#             safe ground check (the planner runs but nothing is sent); False to fly.
#   LOGGING:  write per-run JSONL telemetry/command logs under logs/ for offline tuning.
# --------------------------------------------------------------------------------------
DRY_RUN = False
DEBUG_VISION = False
LOGGING = True

# --------------------------------------------------------------------------------------
# MANUAL CONTROL (this branch). Vision and the autonomous waypoint planner are OFF; you
# fly the drone by keyboard (WASD = translate, Space/Ctrl = climb/descend, Q/E = yaw) and
# press B to capture the drone's current pose as a waypoint. Captures are written to
# CAPTURE_PATH in the same JSON schema the preplanning branch flies (mission.save_mission),
# so you can map the whole course by hand and replay it there. See teleop.py.
#   Set USE_TELEOP=0 to fall back to the autonomous square/mission flight instead.
# --------------------------------------------------------------------------------------
USE_VISION = False
USE_TELEOP = os.environ.get('USE_TELEOP', '1') == '1'   # this branch defaults to manual control
CAPTURE_PATH = os.environ.get('CAPTURE_PATH', 'captured_waypoints.json')
MISSION_PATH = os.environ.get('MISSION_PATH', 'mission.json')
SQUARE_SIDE_M = float(os.environ.get('SQUARE_SIDE_M', '5.0'))
SQUARE_ALT_M = float(os.environ.get('SQUARE_ALT_M', '2.0'))
SQUARE_CCW = os.environ.get('SQUARE_CCW', '0') == '1'   # default clockwise (right turns)

# In autonomous mode, load a custom mission.json if present, else build the default square.
# Manual control needs no mission.
mission = None
if not USE_TELEOP:
    mission = load_mission(MISSION_PATH) or square_mission(
        SQUARE_SIDE_M, SQUARE_ALT_M, counter_clockwise=SQUARE_CCW)

# time since sim started ms
system_boot_ms = int(time.time() * 1000)

# arbitrary shared data between the various components (the cross-thread blackboard)
shared_data = {
    'lock': threading.RLock(),
    'dry_run': DRY_RUN,
    'debug_vision': DEBUG_VISION,
    'logging': LOGGING,
    'use_vision': USE_VISION,
    'use_teleop': USE_TELEOP,
    'capture_path': CAPTURE_PATH,
    'mission': mission,
}

if USE_TELEOP:
    print_controls(CAPTURE_PATH)
else:
    print(f"Mission '{mission.name}': {len(mission.waypoints)} waypoints, loop={mission.loop}", flush=True)
    for i, wp in enumerate(mission.waypoints):
        n, e, d = wp.pos
        yaw_s = f"{wp.yaw_deg:+.0f}deg" if wp.yaw is not None else "hold   "
        print(f"  [{i}] {wp.name or '?':<10} n={n:+.2f} e={e:+.2f} d={d:+.2f}  yaw={yaw_s}", flush=True)

# setup components
components = setup_components(shared_data, system_boot_ms, SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT)
controller = components['controller']
ts_loop = components['ts_loop']
mavlink_rx = components['mavlink_rx']
vision_rx = components['vision_rx']
teleop = components.get('teleop')
logger = components.get('logger')

# Flight-mode entry. The sim is a Betaflight-style FPV racer that boots in ACRO (raw
# rate control) — which has no velocity/position loop, so the armed quad ignored our
# commands and climbed away. We stream level-attitude holds, switch to a self-levelling
# ANGLE mode, then arm; the control loop then flies it with attitude+thrust commands.
# Order: stream holds -> set mode -> arm. (DRY_RUN skips this — nothing is sent.)
if not DRY_RUN:
    print("Priming attitude-hold stream...", flush=True)
    controller.prime_setpoint_stream(seconds=1.0)
    mode = controller.request_offboard_mode()
    print(f"Requested control mode: {mode}", flush=True)
    controller.prime_setpoint_stream(seconds=0.3)  # keep stream alive across the switch

print("Arming drone...", flush=True)
controller.arm()
print(f"Starting control loop... (DRY_RUN={DRY_RUN})", flush=True)

try:
    while True:
        controller.update()
except KeyboardInterrupt:
    print("\nInterrupted — shutting down...", flush=True)

# exit: stop each RX/loop thread and join (guard against threads that never started)
for component in (ts_loop, mavlink_rx, vision_rx, teleop, logger):
    if component is None:
        continue
    thread = component.get_thread_for_join()
    if thread is not None:
        thread.join(timeout=1.0)

print("Client exited!", flush=True)
