#
# Sample Python client for the AI GP controller
#

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
DEBUG_VISION = True
LOGGING = True

# time since sim started ms
system_boot_ms = int(time.time() * 1000)

# arbitrary shared data between the various components (the cross-thread blackboard)
shared_data = {
    'lock': threading.RLock(),
    'dry_run': DRY_RUN,
    'debug_vision': DEBUG_VISION,
    'logging': LOGGING,
    'preplan': True,
    'learn': True,
}

# setup components
components = setup_components(shared_data, system_boot_ms, SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT)
controller = components['controller']
ts_loop = components['ts_loop']
mavlink_rx = components['mavlink_rx']
vision_rx = components['vision_rx']
logger = components.get('logger')

# Flight-mode entry. The sim is a Betaflight-style FPV racer that boots in ACRO (raw
# rate control) — which has no velocity/position loop, so the armed quad ignored our
# commands and climbed away. We stream level-attitude holds, switch to a self-levelling
# ANGLE mode, then arm; the control loop then flies it with attitude+thrust commands.
# Order: stream holds -> set mode -> arm. (DRY_RUN skips this — nothing is sent.)
if not DRY_RUN:
    print("Resetting simulator...", flush=True)
    controller.send_sim_reset_command()
    time.sleep(1.0)
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
for component in (ts_loop, mavlink_rx, vision_rx, logger):
    if component is None:
        continue
    thread = component.get_thread_for_join()
    if thread is not None:
        thread.join(timeout=1.0)

if shared_data.get('course_map'):
    print("Saving learned course map...", flush=True)
    shared_data['course_map'].save()

print("Client exited!", flush=True)
