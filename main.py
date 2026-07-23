import time

from setup import setup_components

SIM_SERVER_UDP_IP   = '127.0.0.1'
SIM_SERVER_UDP_PORT = 14550

system_boot_ms = int(time.time() * 1000)
shared_data    = {}

components = setup_components(shared_data, system_boot_ms, SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT)
controller = components['controller']
ts_loop    = components['ts_loop']
mavlink_rx = components['mavlink_rx']
vision_rx  = components['vision_rx']
planner    = components['planner']
logger     = components['logger']

print('Arming drone...', flush=True)
controller.arm()
print('Control loop running -- Ctrl+C to exit', flush=True)

try:
    while True:
        target = planner.compute_target(shared_data)
        shared_data['planner_target'] = target
        controller.update()
except KeyboardInterrupt:
    pass

logger.stop()
ts_loop.get_thread_for_join().join(timeout=1.0)
mavlink_rx.get_thread_for_join().join(timeout=1.0)
vision_rx.get_thread_for_join().join(timeout=1.0)
print('Client exited!', flush=True)
