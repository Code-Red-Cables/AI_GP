#
# Sample Python client for the AI GP controller
#

import os
import threading
import time

from setup import setup_components

# Modify these properties if you want to run the server remotely for example
SIM_SERVER_UDP_IP = "127.0.0.1"
SIM_SERVER_UDP_PORT = 14550

# --------------------------------------------------------------------------------------
# Safety / debug flags.
#   DRY_RUN:      compute & log guidance but DO NOT send flight setpoints. Keep True
#                 until perception is validated against the sim, then flip to False to fly.
#   DEBUG_VISION: write annotated gate-detection overlays to disk (tuning only — keep
#                 OFF for compliant timed runs; no human interaction is allowed there).
# --------------------------------------------------------------------------------------
DRY_RUN = False
DEBUG_VISION = False
LOGGING = True

# Gate-targeting owner.
#   "pnp"        — YOLO gate corners + solvePnP feed the VIO state estimator and
#                  the planner's world-waypoint guidance (the default: the VQ2 sim
#                  sends no attitude/position telemetry, so VIO is the only pose).
#   "opencv"     — HSV detector + image-space navigator (no world pose needed).
#   "existing_ai" — delegates to a learned policy through AI_POLICY_FACTORY.
GATE_NAVIGATION_MODE = os.environ.get(
    "GATE_NAVIGATION_MODE",
    os.environ.get("VISION_MODE", "pnp"),  # compatibility with older launch scripts
).lower()
AI_POLICY_FACTORY = os.environ.get("AI_POLICY_FACTORY")
if GATE_NAVIGATION_MODE not in {"pnp", "opencv", "existing_ai"}:
    raise ValueError(
        "GATE_NAVIGATION_MODE must be one of: pnp, opencv, existing_ai"
    )

# --------------------------------------------------------------------------------------
# Preplanning (learn-then-replay the deterministic course; spec 3.5). The sim sends no
# track map (spec 4.3), so we LEARN each gate's world position as we pass it and PREPLAN
# (fly to the known next-gate position) on later runs — fixing the "see gate 1, then
# hover because gate 2 is out of the FOV" failure.
#   LEARN:   record gate world positions into COURSE_MAP_PATH as they are passed.
#   PREPLAN: when vision has no fresh lock, fly to the mapped gate instead of coasting.
# Both default ON on this branch (env-overridable: PREPLAN=0 / LEARN=0). First run with an
# empty map behaves like the reactive planner but learns gate 0+; later runs replay it.
# --------------------------------------------------------------------------------------
PREPLAN = os.environ.get('PREPLAN', '1') != '0'
LEARN = os.environ.get('LEARN', '1') != '0'
COURSE_MAP_PATH = os.environ.get('COURSE_MAP_PATH', 'course_map.json')

# time since sim started ms
system_boot_ms = int(time.time() * 1000)

# arbitrary shared data between the various components (the cross-thread blackboard)
shared_data = {
    'lock': threading.RLock(),
    'dry_run': DRY_RUN,
    'debug_vision': DEBUG_VISION,
    'logging': LOGGING,
    'gate_navigation_mode': GATE_NAVIGATION_MODE,
    'ai_policy_factory': AI_POLICY_FACTORY,
    'control_source': (
        'ai' if GATE_NAVIGATION_MODE == 'existing_ai' else GATE_NAVIGATION_MODE
    ),
    'preplan': PREPLAN,
    'learn': LEARN,
    'course_map_path': COURSE_MAP_PATH,
}

# setup components
components = setup_components(shared_data, system_boot_ms, SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT)
controller = components['controller']
ts_loop = components['ts_loop']
mavlink_rx = components['mavlink_rx']
vision_rx = components['vision_rx']
logger = components.get('logger')
state_estimator = components.get('state_estimator')

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
# Release the VIO's pre-flight ZUPT: from here on the drone can actually move,
# so the estimator may dead-reckon. Before this it pins vel/pos to the origin.
with shared_data['lock']:
    shared_data['flight_started'] = True
controller.arm()
print(f"Starting control loop... (DRY_RUN={DRY_RUN})", flush=True)

try:
    while True:
        controller.update()
except KeyboardInterrupt:
    print("\nInterrupted — shutting down...", flush=True)

# exit: stop each RX/loop thread and join (guard against threads that never started)
for component in (ts_loop, mavlink_rx, vision_rx, logger, state_estimator):
    if component is None:
        continue
    thread = component.get_thread_for_join()
    if thread is not None:
        thread.join(timeout=1.0)

# Persist the VIO gate anchors so later runs share one world frame.
if state_estimator is not None:
    state_estimator.save_anchors()
    print(f"Gate anchors saved: {sorted(state_estimator.anchors)}", flush=True)

# Persist the learned course map so the next run can preplan from it.
course_map = shared_data.get('course_map')
if course_map is not None:
    course_map.save()
    print(f"Course map saved: gates {course_map.indices()} -> {COURSE_MAP_PATH}", flush=True)

print("Client exited!", flush=True)
