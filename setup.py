import threading

from pymavlink import mavutil

import config
from controller import Controller
from logger import Logger
from mavlink_rx import MAVLinkRX
from timesync import TimeSync
from vision_rx import VisionRX


def setup_components(shared_data, system_boot_ms, server_ip, server_udp_port):
    # The VIO state estimator reads/writes the blackboard under this lock;
    # the other components keep their existing atomic-replace access.
    shared_data.setdefault('lock', threading.Lock())
    # Dead-reckoning stays parked (ZUPT) until main.py arms the drone.
    shared_data.setdefault('flight_started', False)

    # Logger first — all other components call shared_data['log_event']
    logger = Logger(shared_data)
    logger.log_event('BOOT', f'{server_ip}:{server_udp_port}')

    sim_conn = mavutil.mavlink_connection(f'udpin:{server_ip}:{server_udp_port}')
    print(
        f'Waiting for heartbeat on udpin:{server_ip}:{server_udp_port} ...',
        flush=True,
    )
    print(
        '(Sim must be logged in AND a race started — menu alone sends nothing.)',
        flush=True,
    )
    # Progress every 2s so a slow sim doesn't look frozen (062920 waited ~56s).
    waited = 0.0
    while True:
        hb = sim_conn.wait_heartbeat(blocking=True, timeout=2)
        if hb is not None and getattr(sim_conn, 'target_system', 0):
            break
        waited += 2.0
        print(f'  still waiting... {waited:.0f}s', flush=True)
        if waited >= 120.0:
            raise TimeoutError(
                f'No MAVLink heartbeat on {server_ip}:{server_udp_port} '
                f'after {waited:.0f}s. Start FlightSim, log in, and enter a race. '
                f'Also ensure no other main.py is already bound to that port.'
            )
    print(f'Connected to system {sim_conn.target_system}', flush=True)
    logger.log_event('HEARTBEAT', f'system={sim_conn.target_system}')

    mavlink_rx = MAVLinkRX.create_mavlink_rx(sim_conn, shared_data)
    ts_loop    = TimeSync.create_timesync(sim_conn, shared_data)
    controller = Controller(sim_conn, shared_data, system_boot_ms)
    if config.RESET_SIM_ON_START:
        print('[SIM] resetting race/drone with MAVLink command 31000', flush=True)
        controller.send_sim_reset()
        logger.log_event('SIM_RESET', 'command=31000')
        import time
        time.sleep(max(0.0, config.SIM_RESET_SETTLE_S))
    vision_rx  = VisionRX(shared_data)

    # State estimation: kalman EKF (default on Q2_kalman) or legacy VIO.
    state_estimator = None
    if config.GATE_NAVIGATION_MODE == 'kalman' and config.USE_KALMAN_EKF:
        from ekf_estimator import EKFEstimator
        state_estimator = EKFEstimator.create_ekf_estimator(shared_data)
        logger.log_event('EKF', 'dual_gate_pnp+imu')
    elif config.USE_VIO:
        from state_estimator import StateEstimator
        state_estimator = StateEstimator(
            shared_data, anchors_path=config.VIO_ANCHORS_PATH
        )
        logger.log_event('VIO', f'anchors={config.VIO_ANCHORS_PATH}')

    # Planner selection. Racing modes are exclusive; OpenCV never blends with
    # Dreamer output. existing_ai is handled before this function in main.py.
    if config.GATE_NAVIGATION_MODE == 'kalman':
        from kalman_planner import KalmanDualGatePlanner
        planner = KalmanDualGatePlanner()
    elif config.GATE_NAVIGATION_MODE == 'pose_debug':
        from pose_debug_planner import PoseDebugPlanner
        planner = PoseDebugPlanner()
    elif config.GATE_NAVIGATION_MODE == 'opencv':
        from opencv_gate_planner import OpenCVGatePlanner
        planner = OpenCVGatePlanner()
    elif config.USE_TELEOP:
        from teleop import TeleopPlanner
        planner = TeleopPlanner(shared_data)
    else:
        from planner import Planner
        planner = Planner()

    logger.log_event('PLANNER', planner.name)
    shared_data['planner'] = planner

    return {
        'logger':     logger,
        'vision_rx':  vision_rx,
        'mavlink_rx': mavlink_rx,
        'ts_loop':    ts_loop,
        'sim_conn':   sim_conn,
        'controller': controller,
        'planner':    planner,
        'state_estimator': state_estimator,
    }
