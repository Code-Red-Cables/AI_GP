"""Keyboard teleop flight config — no vision / YOLO / VIO."""
import math
import os


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


# ---- Flight controller (body rates + collective thrust) ----
HOVER_THRUST = 0.27
TAKEOFF_THRUST = 0.27
TAKEOFF_DURATION_S = 1.0
KP_THRUST = 0.25
MAX_DESCENT_THRUST_REDUCTION = 0.025
MAX_ASCENT_THRUST_INCREASE = 0.010
MIN_TILT_COMPENSATION_COSINE = 0.70
KP_LEAN = 0.10
MAX_LEAN_RAD = math.radians(25.0)
AHRS_GYRO_WEIGHT = float(os.environ.get('AHRS_GYRO_WEIGHT', '0.95'))

KP_ATT = 1.8
KD_ATT = 0.09
KI_ATT = float(os.environ.get('KI_ATT', '0.0'))
KP_ROLL_ATT = 2.6
KD_ROLL_ATT = 0.22
KI_ROLL_ATT = float(os.environ.get('KI_ROLL_ATT', '0.0'))
ATTITUDE_INTEGRAL_LIMIT = float(os.environ.get('ATTITUDE_INTEGRAL_LIMIT', '0.20'))
ATTITUDE_DERIVATIVE_FILTER_TAU_S = float(
    os.environ.get('ATTITUDE_DERIVATIVE_FILTER_TAU_S', '0.0')
)

RATE_SIGN_PITCH = -1.0
RATE_SIGN_ROLL = -1.0
RATE_SIGN_YAW = -1.0
MAX_RATE_RAD_S = 1.05
MAX_THRUST = 0.90
MIN_THRUST = 0.05
CONTROL_HZ = 100
TELEMETRY_TIMEOUT_S = 1.25
SENSOR_FUTURE_TOLERANCE_S = 0.05
CONTROL_MIN_DT_S = 1.0 / 500.0
CONTROL_MAX_DT_S = 0.05

# ---- Teleop ----
TELEOP_SPEED = float(os.environ.get('TELEOP_SPEED', '2.0'))       # m/s
TELEOP_VSPEED = float(os.environ.get('TELEOP_VSPEED', '1.5'))     # m/s
TELEOP_YAW_RATE_DPS = float(os.environ.get('TELEOP_YAW_RATE_DPS', '60.0'))
LATERAL_LEAN_GAIN = 0.24

# ---- Sim ----
RESET_SIM_ON_START = _env_bool('RESET_SIM_ON_START', False)
SIM_RESET_SETTLE_S = float(os.environ.get('SIM_RESET_SETTLE_S', '1.5'))
