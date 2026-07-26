import math
import os


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def _env_int_tuple(name, default):
    value = os.environ.get(name)
    if value is None:
        return tuple(default)
    parsed = tuple(int(part.strip()) for part in value.split(','))
    if len(parsed) != len(default):
        raise ValueError(
            f'{name} must contain {len(default)} comma-separated integers'
        )
    return parsed

# ---- Flight controller ----
HOVER_THRUST    = 0.25      # open-loop hover baseline (no baro in VQ2)
TAKEOFF_THRUST  = 0.31      # short launch boost before settling to hover
TAKEOFF_DURATION_S = 1.0
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

# ``auto`` uses the custom YOLO model when it exists and otherwise prints an
# explicit warning before preserving the established HSV detector. Set
# ``yolo_hybrid`` to require the model and fail fast if weights/dependencies
# are missing; set ``hsv`` to intentionally use the legacy global detector.
GATE_DETECTOR_BACKEND = os.environ.get(
    'GATE_DETECTOR_BACKEND', 'auto'
).strip().lower()
if GATE_DETECTOR_BACKEND not in {'auto', 'yolo_hybrid', 'hsv'}:
    raise ValueError(
        'GATE_DETECTOR_BACKEND must be "auto", "yolo_hybrid", or "hsv"'
    )

YOLO_MODEL_PATH = os.environ.get(
    'YOLO_MODEL_PATH', 'models/gate_detector.pt'
)
YOLO_GATE_CLASS_NAME = os.environ.get('YOLO_GATE_CLASS_NAME', 'gate')
YOLO_CONFIDENCE_THRESHOLD = float(
    os.environ.get('YOLO_CONFIDENCE_THRESHOLD', '0.35')
)
YOLO_NMS_IOU_THRESHOLD = float(
    os.environ.get('YOLO_NMS_IOU_THRESHOLD', '0.70')
)
YOLO_TARGET_LOCK_SECONDS = float(
    os.environ.get('YOLO_TARGET_LOCK_SECONDS', '0.75')
)
YOLO_CROP_PADDING_PX = int(os.environ.get('YOLO_CROP_PADDING_PX', '14'))
YOLO_MIN_GATE_AREA_PX = float(
    os.environ.get('YOLO_MIN_GATE_AREA_PX', '400')
)
YOLO_MAX_OUTSIDE_FRACTION = float(
    os.environ.get('YOLO_MAX_OUTSIDE_FRACTION', '0.35')
)
YOLO_PREVIOUS_CENTER_FRAMES = int(
    os.environ.get('YOLO_PREVIOUS_CENTER_FRAMES', '5')
)
YOLO_INFERENCE_SIZE = int(os.environ.get('YOLO_INFERENCE_SIZE', '640'))
YOLO_DEVICE = os.environ.get('YOLO_DEVICE', '').strip() or None
YOLO_LOG_INTERVAL_S = float(os.environ.get('YOLO_LOG_INTERVAL_S', '1.0'))

# Initial crop-local segmentation uses the calibrated Q2 values. These remain
# environment-configurable without modifying detector code.
GATE_HSV_LOWER = _env_int_tuple('GATE_HSV_LOWER', (3, 105, 180))
GATE_HSV_UPPER = _env_int_tuple('GATE_HSV_UPPER', (17, 255, 255))
GATE_MIN_CONTOUR_AREA = float(
    os.environ.get('GATE_MIN_CONTOUR_AREA', '45')
)
