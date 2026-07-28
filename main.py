"""Keyboard teleop client — fly the sim with WASD / Space / QE."""
import time

import config
from setup import setup_components

SIM_SERVER_UDP_IP = '127.0.0.1'
SIM_SERVER_UDP_PORT = 14550


def main():
    system_boot_ms = int(time.time() * 1000)
    shared_data = {}
    components = setup_components(
        shared_data,
        system_boot_ms,
        SIM_SERVER_UDP_IP,
        SIM_SERVER_UDP_PORT,
    )
    controller = components['controller']
    planner = components['planner']
    logger = components['logger']

    print('[MODE] keyboard teleop', flush=True)
    print('Arming drone...', flush=True)
    controller.arm()
    print('Control loop running — Ctrl+C to exit', flush=True)

    try:
        while True:
            shared_data['planner_target'] = planner.compute_target(shared_data)
            controller.update()
    except KeyboardInterrupt:
        pass
    finally:
        controller.disarm()
        logger.stop()
        for name in ('ts_loop', 'mavlink_rx'):
            component = components.get(name)
            if component is not None:
                component.get_thread_for_join().join(timeout=1.0)
        print('Client exited!', flush=True)


if __name__ == '__main__':
    main()
