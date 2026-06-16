from pymavlink import mavutil
from timesync import TimeSync
from mavlink_rx import MAVLinkRX
from controller import Controller
from planner import Planner
from logger import Logger

def setup_components(shared_data, system_boot_ms, server_ip, server_udp_port):
    # -------------------------------
    # Mavlink Connection
    # -------------------------------
    # Start a connection listening on a UDP port
    sim_conn = mavutil.mavlink_connection('udpin:%s:%s' % (server_ip, server_udp_port,))
    print("Waiting for heartbeat...", flush=True)
    sim_conn.wait_heartbeat()
    print(f"Connected to system: {sim_conn.target_system}", flush=True)

    # -------------------------------
    # Setup Mavlink msg receiver
    # -------------------------------
    print("Setting up MAVLink rx...", flush=True)
    mavlink_rx = MAVLinkRX.create_mavlink_rx(sim_conn, shared_data)

    # -------------------------------
    # Timesync request Loop
    # -------------------------------
    print("Setting up Timesync loop...", flush=True)
    ts_loop = TimeSync.create_timesync(sim_conn, shared_data)

    # -------------------------------
    # Connect Vision receiver (OFF on this preplanning branch -- the path is flown purely
    # from the waypoint mission, so we don't bind the camera socket or run detection).
    # -------------------------------
    vision_rx = None
    if shared_data.get('use_vision', False):
        from vision_rx import VisionRX
        vision_rx = VisionRX(shared_data)

    # -------------------------------
    # Planner + main control loop. The planner follows shared_data['mission'] (an ordered
    # list of position+yaw waypoints, see mission.py), already injected by main.py.
    # -------------------------------
    print("Setting up planner...", flush=True)
    planner = Planner(shared_data)
    controller = Controller(sim_conn, shared_data, system_boot_ms, planner)

    # -------------------------------
    # Run logger (optional, for offline tuning)
    # -------------------------------
    logger = None
    if shared_data.get('logging', True):
        logger = Logger.create_logger(shared_data)

    return {
        'vision_rx': vision_rx,
        'mavlink_rx': mavlink_rx,
        'ts_loop': ts_loop,
        'sim_conn': sim_conn,
        'controller': controller,
        'planner': planner,
        'logger': logger
    }