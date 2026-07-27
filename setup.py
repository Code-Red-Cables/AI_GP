from pymavlink import mavutil
from timesync import TimeSync
from vision_rx import VisionRX
from mavlink_rx import MAVLinkRX
from controller import Controller
from planner import Planner
from logger import Logger
from course_map import CourseMap
from vision.ai_adapter import load_policy_factory
from state_estimator import StateEstimator

def setup_components(shared_data, system_boot_ms, server_ip, server_udp_port):
    # Optional and lazy: OpenCV-only mode never imports the learned policy's
    # framework. The checked-in main branch has no model weights, so callers
    # provide their existing policy through this narrow adapter boundary.
    shared_data['ai_policy'] = load_policy_factory(
        shared_data.get('ai_policy_factory'), shared_data
    )
    shared_data.setdefault(
        'control_source',
        'ai'
        if shared_data.get('gate_navigation_mode') == 'existing_ai'
        else 'opencv'
    )
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
    # VIO state estimator — MUST exist on this sim build: VQ2 sends no ATTITUDE
    # and no LOCAL_POSITION_NED, so this thread manufactures shared_data
    # ['attitude'] and ['position_ned'] from HIGHRES_IMU + gate-PnP fixes.
    # Started before VisionRX so the first PnP fix finds a live consumer.
    # -------------------------------
    print("Setting up VIO state estimator...", flush=True)
    state_estimator = StateEstimator(shared_data)

    # -------------------------------
    # Connect Vision receiver
    # -------------------------------
    vision_rx = VisionRX(shared_data)

    # -------------------------------
    # Preplanning course map (learn-then-replay). The sim broadcasts no track geometry
    # (spec 4.3), but the course is deterministic (spec 3.5), so we learn each gate's
    # world position on one run and replay it on the next. Loaded BEFORE the planner so
    # it can read it in __init__. Only created when preplan/learn is enabled, so a
    # default run is unaffected.
    # -------------------------------
    if shared_data.get('preplan') or shared_data.get('learn'):
        course_map = CourseMap(shared_data.get('course_map_path', 'course_map.json')).load()
        shared_data['course_map'] = course_map
        print(f"Course map loaded: {course_map.indices()} "
              f"(preplan={shared_data.get('preplan')}, learn={shared_data.get('learn')})",
              flush=True)

    # -------------------------------
    # Planner + main control loop
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
        'logger': logger,
        'state_estimator': state_estimator,
    }
