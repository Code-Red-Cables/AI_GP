import threading

from pymavlink import mavutil

import config
from controller import Controller
from logger import Logger
from mavlink_rx import MAVLinkRX
from timesync import TimeSync
from vision_rx import VisionRX


# Tuning that changes where the drone flies. Recorded at boot because it was
# not: after a batch regressed from gate 3 to gate 1, the settings behind the
# earlier good runs could not be recovered from the logs, and the simulator
# became unavailable before the difference could be bisected by re-flying.
LOGGED_CONFIG_KEYS = (
    'FLIGHT_MODE', 'CONTROL_HZ', 'HOVER_THRUST',
    'GATE_DETECTOR_BACKEND', 'YOLO_DEVICE', 'YOLO_INFERENCE_SIZE',
    'ASSIST_LEAN_DEG', 'ASSIST_ROLL_SCALE', 'ASSIST_FWD_FRAC',
    'ASSIST_COAST_ROLL_NX', 'ASSIST_COAST_ROLL_FRAC',
    'ASSIST_NY_AIM', 'ASSIST_COMMIT_HOLD_S',
    'ASSIST_YAW_ALIGN_MAX_DEG', 'ASSIST_CENTERLINE_YAW_KP',
    'ASSIST_YAW_FRAME_NX', 'ASSIST_YAW_FRAME_MAX_DEG',
    'ASSIST_SIM_CLOCK', 'MAX_RATE_RAD_S',
    'RATE_SIGN_ROLL', 'RATE_SIGN_PITCH', 'RATE_SIGN_YAW',
)


def _log_flight_config(logger) -> None:
    """Snapshot the knobs that decide the trajectory, so a run is reproducible."""
    for key in LOGGED_CONFIG_KEYS:
        value = getattr(config, key, None)
        if value is not None:
            logger.log_event('CONFIG', f'{key}={value}')
    try:
        import torch
        device = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
        )
        logger.log_event('CONFIG', f'torch={torch.__version__} device={device}')
    except Exception:  # noqa: BLE001 - a missing torch is the detector's problem
        pass


def setup_components(
    shared_data,
    system_boot_ms,
    server_ip,
    server_udp_port,
    *,
    enable_vision: bool = True,
):
    # The VIO state estimator reads/writes the blackboard under this lock;
    # the other components keep their existing atomic-replace access.
    shared_data.setdefault('lock', threading.Lock())
    # Dead-reckoning stays parked (ZUPT) until main.py arms the drone.
    shared_data.setdefault('flight_started', False)

    # Logger first — all other components call shared_data['log_event']
    logger = Logger(shared_data)
    logger.log_event('BOOT', f'{server_ip}:{server_udp_port}')
    _log_flight_config(logger)

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
    vision_rx = None
    if enable_vision:
        vision_rx = VisionRX(shared_data)
    else:
        print('[VISION] disabled (pure stick / fly mode)', flush=True)
        logger.log_event('VISION', 'disabled')

    # State estimation: dual-gate PnP fixes + IMU dead reckoning.
    from ekf_estimator import EKFEstimator
    state_estimator = EKFEstimator.create_ekf_estimator(shared_data)
    logger.log_event(
        'EKF',
        'dual_gate_pnp+imu' if config.EKF_USE_PNP else 'imu_only',
    )

    if config.FLIGHT_MODE == 'spline':
        from spline_planner import SplinePlanner
        planner = SplinePlanner()
    elif config.FLIGHT_MODE == 'assist':
        from assist_planner import AssistImagePlanner
        planner = AssistImagePlanner()
    elif config.FLIGHT_MODE == 'bc':
        from bc_planner import BCPlanner
        planner = BCPlanner()
    else:
        from kalman_planner import KalmanDualGatePlanner
        planner = KalmanDualGatePlanner()

    logger.log_event('PLANNER', f'{planner.name} mode={config.FLIGHT_MODE}')
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
