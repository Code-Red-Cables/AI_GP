import math
import os


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}

# ---- Flight controller ----
HOVER_THRUST    = 0.27      # open-loop hover baseline (no baro in VQ2)
KP_THRUST       = 0.25      # vd cmd (m/s) → thrust delta
KP_LEAN         = 0.10      # body velocity (m/s) → desired lean angle (rad)
MAX_LEAN_RAD    = math.radians(25.0)
KP_ATT          = 1.8       # demo: 0.6 normalized gain * 3.0 max rate
KD_ATT          = 0.09      # demo: 0.03 normalized damping * 3.0 max rate
RATE_SIGN_PITCH = -1.0      # sim pitch rate axis is inverted
RATE_SIGN_ROLL  = -1.0      # measured demo ActionConfig: all rate axes inverted
MAX_RATE_RAD_S  = 1.05      # demo rate_cap 0.35 * 3.0 max rate
MAX_THRUST      = 0.90
MIN_THRUST      = 0.05
CONTROL_HZ      = 100
TELEMETRY_TIMEOUT_S = 0.35

# ---- Mode selection ----
# Exactly one racing command owner is selected at process start.  ``opencv``
# uses the Q2 rate controller below; ``existing_ai`` delegates to the unchanged
# Dreamer deployment controller under dreamer/.
GATE_NAVIGATION_MODE = os.environ.get(
    'GATE_NAVIGATION_MODE', 'opencv'
).strip().lower()
if GATE_NAVIGATION_MODE not in {'opencv', 'existing_ai'}:
    raise ValueError(
        'GATE_NAVIGATION_MODE must be "opencv" or "existing_ai", '
        f'not {GATE_NAVIGATION_MODE!r}'
    )

DREAMER_CHECKPOINT = os.environ.get('DREAMER_CHECKPOINT', '')
DREAMER_CONFIG     = os.environ.get('DREAMER_CONFIG') or None
DREAMER_MAX_SECONDS = float(os.environ.get('DREAMER_MAX_SECONDS', '480'))

# Development-only fallbacks used when setup_components() is called with a
# non-racing mode.  They do not override GATE_NAVIGATION_MODE.
USE_TELEOP      = False
USE_GATE_CHASER = False

# ---- Teleop ----
TELEOP_SPEED        = 2.0   # m/s forward / strafe
TELEOP_VSPEED       = 1.5   # m/s climb / descend
TELEOP_YAW_RATE_DPS = 60.0  # deg/s yaw

# ---- Gate chaser ----
GATE_KP_LAT       = 0.40
GATE_MAX_STRAFE   = 0.65    # m/s clamp
GATE_KP_VERT      = 0.60
GATE_MAX_VERT     = 0.65    # m/s clamp
GATE_APPROACH_SPD = 1.2     # m/s constant forward approach
GATE_CLOSE_RANGE  = 5.0     # m — switch to commit (blast through) mode

# ---- Camera / geometry ----
GATE_INNER_M    = 1.5       # gate inner side (m) for range estimation
CAMERA_FOCAL_PX = 320.0     # fx = fy from spec
CAMERA_CX       = 320.0
CAMERA_CY       = 180.0
CAMERA_TILT_RAD = math.radians(20.0)   # camera tilted 20° UP from body forward

# ---- OpenCV vision runtime ----
VISION_UDP_IP             = '0.0.0.0'
VISION_UDP_PORT           = 5600
VISION_COMMAND_TIMEOUT_S  = 0.35
VISION_DEBUG              = _env_bool('VISION_DEBUG', False)
VISION_DISPLAY            = _env_bool('VISION_DISPLAY', False)
PERCEPTION_ONLY           = _env_bool('PERCEPTION_ONLY', False)
VISION_DEBUG_DIR          = os.environ.get('VISION_DEBUG_DIR', '_vision_debug')
VISION_DEBUG_INTERVAL_S   = float(
    os.environ.get('VISION_DEBUG_INTERVAL_S', '5.0')
)
