"""Wire the keyboard-teleop flight stack (no vision / YOLO / VIO)."""
import time

from pymavlink import mavutil

import config
from controller import Controller
from logger import Logger
from mavlink_rx import MAVLinkRX
from teleop import TeleopPlanner
from timesync import TimeSync


def setup_components(shared_data, system_boot_ms, server_ip, server_udp_port):
    logger = Logger(shared_data)
    logger.log_event('BOOT', f'{server_ip}:{server_udp_port}')

    sim_conn = mavutil.mavlink_connection(
        f'udpin:{server_ip}:{server_udp_port}'
    )
    print('Waiting for heartbeat...', flush=True)
    sim_conn.wait_heartbeat()
    print(f'Connected to system {sim_conn.target_system}', flush=True)
    logger.log_event('HEARTBEAT', f'system={sim_conn.target_system}')

    mavlink_rx = MAVLinkRX.create_mavlink_rx(sim_conn, shared_data)
    ts_loop = TimeSync.create_timesync(sim_conn, shared_data)
    controller = Controller(sim_conn, shared_data, system_boot_ms)

    if config.RESET_SIM_ON_START:
        print('[SIM] resetting race/drone with MAVLink command 31000', flush=True)
        controller.send_sim_reset()
        logger.log_event('SIM_RESET', 'command=31000')
        time.sleep(max(0.0, config.SIM_RESET_SETTLE_S))

    planner = TeleopPlanner(shared_data)
    logger.log_event('PLANNER', planner.name)
    shared_data['planner'] = planner

    return {
        'logger': logger,
        'mavlink_rx': mavlink_rx,
        'ts_loop': ts_loop,
        'sim_conn': sim_conn,
        'controller': controller,
        'planner': planner,
    }
