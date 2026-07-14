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
    # VQ2 state estimator. ATTITUDE / LOCAL_POSITION_NED / ODOMETRY are blocked (spec 9.3),
    # so we rebuild attitude (IMU AHRS) + altitude (baro) ourselves and publish them into
    # shared_data with the old schema -> the controller is unchanged. Must start BEFORE the
    # controller so a state estimate exists when the loop begins.
    # -------------------------------
    state_estimator = None
    if shared_data.get('use_state_estimator', False):
        print("Setting up VQ2 state estimator (IMU/baro)...", flush=True)
        from state_estimator import StateEstimator
        state_estimator = StateEstimator.create_state_estimator(shared_data)

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
    # Planner + main control loop. Three planners share the controller's velocity contract:
    #   * manual (use_teleop): TeleopPlanner is driven by a KeyboardTeleop input thread, and
    #     B captures the live pose to a waypoint file (see teleop.py).
    #   * autonomous + use_spline (default): SplinePlanner flies a smooth Catmull-Rom spline
    #     through shared_data['mission'] at constant cruise speed (continuous flight).
    #   * autonomous, no spline: Planner stops at each shared_data['mission'] waypoint.
    # -------------------------------
    teleop = None
    if shared_data.get('use_gate_chaser', False):
        # VQ2 Phase 1: reactive visual servo toward the detected gate (the camera is the only
        # absolute reference). Needs vision + the state estimator (attitude).
        print("Setting up VQ2 gate chaser (reactive visual servo)...", flush=True)
        from gate_chaser import GateChaser
        planner = GateChaser(shared_data)
    elif shared_data.get('use_teleop', False):
        print("Setting up teleop (manual keyboard control)...", flush=True)
        from teleop import TeleopPlanner, KeyboardTeleop
        planner = TeleopPlanner(shared_data)
        teleop = KeyboardTeleop.create_keyboard_teleop(
            shared_data, shared_data.get('capture_path', 'captured_waypoints.json'))
    elif shared_data.get('use_spline', False):
        print("Setting up spline planner (continuous waypoint following)...", flush=True)
        from spline_planner import SplinePlanner
        planner = SplinePlanner(shared_data)
    elif shared_data.get('use_state_estimator', False):
        # VQ2 Phase 0: no navigation -- hover on pure estimated state to validate the IMU pipeline.
        print("Setting up VQ2 hover planner (Phase-0 estimator check)...", flush=True)
        from hover_planner import HoverPlanner
        planner = HoverPlanner(shared_data)
    else:
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
        'state_estimator': state_estimator,
        'ts_loop': ts_loop,
        'sim_conn': sim_conn,
        'controller': controller,
        'planner': planner,
        'teleop': teleop,
        'logger': logger
    }