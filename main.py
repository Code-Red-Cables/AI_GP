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
DRY_RUN = True
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
}

# setup components
components = setup_components(shared_data, system_boot_ms, SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT)
controller = components['controller']
ts_loop = components['ts_loop']
mavlink_rx = components['mavlink_rx']
vision_rx = components['vision_rx']
logger = components.get('logger')

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

print("Client exited!", flush=True)
