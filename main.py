import sys
import time
from pathlib import Path

import config

SIM_SERVER_UDP_IP   = '127.0.0.1'
SIM_SERVER_UDP_PORT = 14550


def run_existing_ai():
    """Delegate exclusively to Q2_new's unchanged Dreamer deploy controller."""
    if not config.DREAMER_CHECKPOINT:
        raise RuntimeError(
            'existing_ai mode requires DREAMER_CHECKPOINT=/path/to/deploy_*.pt'
        )

    dreamer_src = Path(__file__).resolve().parent / 'dreamer' / 'src'
    sys.path.insert(0, str(dreamer_src))
    from dreamer_drone.config import load_config
    from dreamer_drone.deploy.controller import DeployController

    dreamer_config = load_config(config.DREAMER_CONFIG)
    controller = DeployController(dreamer_config, config.DREAMER_CHECKPOINT)
    print('[MODE] existing_ai (Dreamer owns all flight commands)', flush=True)
    try:
        controller.run(arm=True, max_seconds=config.DREAMER_MAX_SECONDS)
    except KeyboardInterrupt:
        pass
    finally:
        controller.close()


def run_opencv():
    """Run the Q2 rate controller with OpenCV navigation as its planner."""
    from setup import setup_components

    system_boot_ms = int(time.time() * 1000)
    shared_data = {'gate_navigation_mode': 'opencv'}
    components = setup_components(
        shared_data,
        system_boot_ms,
        SIM_SERVER_UDP_IP,
        SIM_SERVER_UDP_PORT,
    )
    controller = components['controller']
    planner = components['planner']
    logger = components['logger']

    print('[MODE] opencv (Q2 rate controller)', flush=True)
    if config.PERCEPTION_ONLY:
        print(
            '[SAFE] perception-only: not arming and not sending flight commands',
            flush=True,
        )
    else:
        print('Arming drone...', flush=True)
        controller.arm()
    print('Control loop running -- Ctrl+C to exit', flush=True)

    try:
        while True:
            shared_data['planner_target'] = planner.compute_target(shared_data)
            if config.PERCEPTION_ONLY:
                time.sleep(1.0 / config.CONTROL_HZ)
            else:
                controller.update()
    except KeyboardInterrupt:
        pass
    finally:
        logger.stop()
        for name in ('ts_loop', 'mavlink_rx', 'vision_rx'):
            components[name].get_thread_for_join().join(timeout=1.0)
        print('Client exited!', flush=True)


def main():
    if config.GATE_NAVIGATION_MODE == 'existing_ai':
        run_existing_ai()
    else:
        run_opencv()


if __name__ == '__main__':
    main()
