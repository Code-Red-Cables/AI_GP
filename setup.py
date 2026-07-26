from pymavlink import mavutil

import config
from controller import Controller
from logger import Logger
from mavlink_rx import MAVLinkRX
from timesync import TimeSync
from vision_rx import VisionRX


def setup_components(shared_data, system_boot_ms, server_ip, server_udp_port):
    # Logger first — all other components call shared_data['log_event']
    logger = Logger(shared_data)
    logger.log_event('BOOT', f'{server_ip}:{server_udp_port}')

    sim_conn = mavutil.mavlink_connection(f'udpin:{server_ip}:{server_udp_port}')
    print('Waiting for heartbeat...', flush=True)
    sim_conn.wait_heartbeat()
    print(f'Connected to system {sim_conn.target_system}', flush=True)
    logger.log_event('HEARTBEAT', f'system={sim_conn.target_system}')

    mavlink_rx = MAVLinkRX.create_mavlink_rx(sim_conn, shared_data)
    ts_loop    = TimeSync.create_timesync(sim_conn, shared_data)
    vision_rx  = VisionRX(shared_data)

    # Planner selection. Racing modes are exclusive; OpenCV never blends with
    # Dreamer output. existing_ai is handled before this function in main.py.
    if config.GATE_NAVIGATION_MODE == 'opencv':
        from opencv_gate_planner import OpenCVGatePlanner
        planner = OpenCVGatePlanner()
    elif config.USE_GATE_CHASER:
        from gate_chaser import GateChaserPlanner
        planner = GateChaserPlanner()
    elif config.USE_TELEOP:
        from teleop import TeleopPlanner
        planner = TeleopPlanner(shared_data)
    else:
        from planner import Planner
        planner = Planner()

    logger.log_event('PLANNER', planner.name)
    shared_data['planner'] = planner

    controller = Controller(sim_conn, shared_data, system_boot_ms)

    return {
        'logger':     logger,
        'vision_rx':  vision_rx,
        'mavlink_rx': mavlink_rx,
        'ts_loop':    ts_loop,
        'sim_conn':   sim_conn,
        'controller': controller,
        'planner':    planner,
    }
