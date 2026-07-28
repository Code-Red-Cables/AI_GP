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
HOVER_THRUST    = 0.27      # open-loop hover baseline (no baro in VQ2)
TAKEOFF_THRUST  = 0.27      # short launch boost before settling to hover
TAKEOFF_DURATION_S = 1.0
KP_THRUST       = 0.25      # vd cmd (m/s) → thrust delta
# Gate-two telemetry saturated the optical descent command for 1.6 seconds at
# ~0.263 tilt-compensated thrust while its image speed still accelerated to
# 0.476 normalized/s toward the bottom edge. Permit a 0.256-ish tilted level
# only while vision explicitly requests descent; hover/takeoff remain 0.27.
# Gate-two telemetry held the previous 0.014 reduction continuously while the
# target still fell from y=152 to y=209 in 0.9 s. Preserve the calibrated
# 0.27 hover/takeoff baseline, but give a confirmed low gate enough downward
# authority to arrest that climb.
MAX_DESCENT_THRUST_REDUCTION = 0.025
MAX_ASCENT_THRUST_INCREASE = 0.010
MIN_TILT_COMPENSATION_COSINE = 0.70
KP_LEAN         = 0.10      # demonstrated forward command → pitch mapping
OPENCV_KP_LEAN  = 0.16      # stronger gate-racing forward pitch mapping
OPENCV_LATERAL_LEAN_SIGN = float(
    # Direct A/B runs show negative desired roll moves a right-side gate
    # farther inward than positive desired roll in the live VQ2 rate path.
    os.environ.get('OPENCV_LATERAL_LEAN_SIGN', '-1.0')
)
MAX_LEAN_RAD    = math.radians(25.0)
# VQ2's accelerometer magnitude looks gravity-like while its lateral component
# does not rotate with the visibly banked camera. The demonstration filter's
# 0.95 gyro weight therefore erases real Q2 bank in roughly 0.2 seconds.
# Retain slow drift correction, but let the racing controller observe the
# 1-2 second roll motion that determines lateral momentum.
OPENCV_AHRS_GYRO_WEIGHT = float(
    os.environ.get('OPENCV_AHRS_GYRO_WEIGHT', '0.995')
)
KP_ATT          = 1.8       # demo: 0.6 normalized gain * 3.0 max rate
KD_ATT          = 0.09      # demo: 0.03 normalized damping * 3.0 max rate
KI_ATT          = float(os.environ.get('KI_ATT', '0.0'))
KP_ROLL_ATT     = 2.6       # faster bank reversal for gate counter-steering
# The gate-two trace reverses desired bank before centre crossing, but the
# measured roll rate keeps carrying the vehicle through the old direction.
# Add rate damping without weakening the lateral position command.
KD_ROLL_ATT     = 0.22
KI_ROLL_ATT     = float(os.environ.get('KI_ROLL_ATT', '0.0'))
ATTITUDE_INTEGRAL_LIMIT = float(
    os.environ.get('ATTITUDE_INTEGRAL_LIMIT', '0.20')
)
ATTITUDE_DERIVATIVE_FILTER_TAU_S = float(
    os.environ.get('ATTITUDE_DERIVATIVE_FILTER_TAU_S', '0.0')
)
RATE_SIGN_PITCH = -1.0      # sim pitch rate axis is inverted
RATE_SIGN_ROLL  = -1.0      # measured demo ActionConfig: all rate axes inverted
RATE_SIGN_YAW   = -1.0      # preserve the recorded demonstration convention
# Keep OpenCV explicit so the live axis can be calibrated independently.
# Yaw-only A/B traces isolate this from lateral bank: positive sent yaw moved
# a right-side gate inward from x=524 to x=492, whereas negative sent yaw
# moved it outward. Positive is therefore the rightward navigation mapping.
OPENCV_RATE_SIGN_YAW = float(
    os.environ.get('OPENCV_RATE_SIGN_YAW', '1.0')
)
# Positive sent yaw produces negative raw body-z gyro in VQ2. Convert that
# sensor axis into the positive/right navigation convention before applying
# rate feedback.
OPENCV_YAW_GYRO_SIGN = float(
    os.environ.get('OPENCV_YAW_GYRO_SIGN', '-1.0')
)
OPENCV_YAW_RATE_FEEDBACK_KP = float(
    os.environ.get('OPENCV_YAW_RATE_FEEDBACK_KP', '0.45')
)
OPENCV_YAW_RATE_FEEDBACK_LIMIT = float(
    os.environ.get('OPENCV_YAW_RATE_FEEDBACK_LIMIT', '0.30')
)
# A zero yaw request means "capture the current heading", not "snap to an
# equal-and-opposite turn".  The simulator responds strongly enough that the
# normal tracking gain can reverse a +0.58 rad/s turn in one vision interval.
# Use a deliberately softer brake while the planner requests zero yaw.
OPENCV_YAW_BRAKE_FEEDBACK_KP = float(
    os.environ.get('OPENCV_YAW_BRAKE_FEEDBACK_KP', '0.0')
)
OPENCV_YAW_BRAKE_FEEDBACK_LIMIT = float(
    os.environ.get('OPENCV_YAW_BRAKE_FEEDBACK_LIMIT', '0.0')
)
OPENCV_YAW_BRAKE_REQUEST_DEADBAND = float(
    os.environ.get('OPENCV_YAW_BRAKE_REQUEST_DEADBAND', '0.02')
)
MAX_RATE_RAD_S  = 1.05      # demo rate_cap 0.35 * 3.0 max rate
MAX_THRUST      = 0.90
MIN_THRUST      = 0.05
CONTROL_HZ      = 100
TELEMETRY_TIMEOUT_S = 1.25
SENSOR_FUTURE_TOLERANCE_S = 0.05
CONTROL_MIN_DT_S = 1.0 / 500.0
CONTROL_MAX_DT_S = 0.05

# ---- VIO / PnP state estimation ----
# The VQ2 sim sends no attitude (deprecated) and no LOCAL_POSITION_NED. The
# StateEstimator (state_estimator.py) manufactures shared_data['attitude'] and
# shared_data['position_ned'] from HIGHRES_IMU dead-reckoning corrected by
# YOLO-corner PnP gate fixes (vision/yolo_pnp.py). When enabled it owns the
# 'attitude' key and MAVLinkRX publishes the sim message as 'attitude_raw'.
#
# PID tuning order (matches the procedure proven on the Q2_pnp branch):
#   1. HOVER_THRUST first — stationary hover, zero commanded velocity.
#   2. Thrust PI (KP_THRUST_VEL / KI_THRUST_VEL) against vertical steps.
#   3. Attitude PDs (KP_ATT / KD_ATT, KP_ROLL_ATT / KD_ROLL_ATT).
#   4. Yaw hold last (KP_YAW_ATT) — the camera FOV is narrow, so keep
#      YAW_RATE_MAX_DEG_S low or a fast yaw sweeps the gate out of frame.
USE_VIO = _env_bool('USE_VIO', True)
VIO_ANCHORS_PATH = os.environ.get('VIO_ANCHORS_PATH', 'gate_anchors.json')
# VIO attitude/velocity older than this falls back to the AHRS / open-loop
# thrust paths instead of trusting a stale belief.
VIO_STATE_TIMEOUT_S = float(os.environ.get('VIO_STATE_TIMEOUT_S', '0.5'))
# Heading-hold yaw PID (active when the planner requests ~zero yaw rate and a
# fresh VIO yaw exists). Seeded from the flight-tested Q2_pnp KP_YAW.
KP_YAW_ATT = float(os.environ.get('KP_YAW_ATT', '2.0'))
KI_YAW_ATT = float(os.environ.get('KI_YAW_ATT', '0.0'))
KD_YAW_ATT = float(os.environ.get('KD_YAW_ATT', '0.0'))
YAW_RATE_MAX_RAD_S = math.radians(
    float(os.environ.get('YAW_RATE_MAX_DEG_S', '70.0'))
)
# Vertical-velocity thrust PI (closes the loop the open-loop path cannot:
# thrust = HOVER_THRUST + PID(vd_measured - vd_target)). Gains seeded from the
# Q2_pnp branch fit of vertical accel vs commanded thrust.
KP_THRUST_VEL = float(os.environ.get('KP_THRUST_VEL', '0.25'))
KI_THRUST_VEL = float(os.environ.get('KI_THRUST_VEL', '0.06'))
# Integral state bound in (m/s)*s: contribution bound = KI_THRUST_VEL * limit
# (2.5 * 0.06 = 0.15 thrust — enough to absorb a hover-thrust misfit without
# letting a blind stretch wind the collective to the ceiling).
THRUST_INTEGRAL_LIMIT = float(os.environ.get('THRUST_INTEGRAL_LIMIT', '2.5'))
# Closed-loop thrust floor keeps prop wash / attitude authority during descent
# (the open-loop path keeps its own MIN_THRUST).
VIO_THRUST_MIN = float(os.environ.get('VIO_THRUST_MIN', '0.20'))
VIO_THRUST_MAX = float(os.environ.get('VIO_THRUST_MAX', '0.90'))

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
VISION_COMMAND_TIMEOUT_S  = 1.25
VISION_DEBUG              = _env_bool('VISION_DEBUG', False)
VISION_DISPLAY            = _env_bool('VISION_DISPLAY', True)
PERCEPTION_ONLY           = _env_bool('PERCEPTION_ONLY', False)
RESET_SIM_ON_START        = _env_bool('RESET_SIM_ON_START', False)
SIM_RESET_SETTLE_S        = float(
    os.environ.get('SIM_RESET_SETTLE_S', '1.5')
)
# Zero preserves the normal race client, which runs until Ctrl+C. A positive
# value is useful for bounded simulator test attempts and exits through the
# regular cleanup/disarm path.
OPENCV_MAX_SECONDS        = float(
    os.environ.get('OPENCV_MAX_SECONDS', '0')
)
VISION_DEBUG_DIR          = os.environ.get('VISION_DEBUG_DIR', '_vision_debug')
VISION_DEBUG_INTERVAL_S   = float(
    os.environ.get('VISION_DEBUG_INTERVAL_S', '5.0')
)
GATE_FRAME_CAPTURE        = _env_bool('GATE_FRAME_CAPTURE', True)
GATE_FRAME_CAPTURE_DIR    = os.environ.get(
    'GATE_FRAME_CAPTURE_DIR', 'frames'
)
# Zero saves every real detector hit. Increase this only if a lower capture
# rate is desired, for example 0.2 for at most five images per second.
GATE_FRAME_CAPTURE_INTERVAL_S = float(
    os.environ.get('GATE_FRAME_CAPTURE_INTERVAL_S', '0.0')
)

# ``auto`` prefers the four-keypoint pose model, then the YOLO-box/HSV hybrid,
# and finally prints an explicit warning before preserving the legacy HSV
# detector. Explicit backends fail fast when their weights are missing.
GATE_DETECTOR_BACKEND = os.environ.get(
    'GATE_DETECTOR_BACKEND', 'yolo_pose'
).strip().lower()
if GATE_DETECTOR_BACKEND not in {
    'auto', 'yolo_pose', 'yolo_hybrid', 'hsv'
}:
    raise ValueError(
        'GATE_DETECTOR_BACKEND must be "auto", "yolo_pose", '
        '"yolo_hybrid", or "hsv"'
    )

YOLO_POSE_MODEL_PATH = os.environ.get(
    'YOLO_POSE_MODEL_PATH', 'models/gate_pose.pt'
)
YOLO_MODEL_PATH = os.environ.get(
    'YOLO_MODEL_PATH', 'models/gate_detector.pt'
)
YOLO_GATE_CLASS_NAME = os.environ.get('YOLO_GATE_CLASS_NAME', 'gate')
YOLO_CONFIDENCE_THRESHOLD = float(
    os.environ.get('YOLO_CONFIDENCE_THRESHOLD', '0.45')
)
YOLO_KEYPOINT_CONFIDENCE_THRESHOLD = float(
    os.environ.get('YOLO_KEYPOINT_CONFIDENCE_THRESHOLD', '0.25')
)
YOLO_NMS_IOU_THRESHOLD = float(
    os.environ.get('YOLO_NMS_IOU_THRESHOLD', '0.70')
)
YOLO_TARGET_LOCK_SECONDS = float(
    os.environ.get('YOLO_TARGET_LOCK_SECONDS', '0.75')
)
YOLO_PERSISTENT_TARGET_LOCK = _env_bool(
    'YOLO_PERSISTENT_TARGET_LOCK', True
)
YOLO_TARGET_ASSOCIATION_CENTER_SPAN = float(
    os.environ.get('YOLO_TARGET_ASSOCIATION_CENTER_SPAN', '1.85')
)
YOLO_TARGET_ASSOCIATION_MIN_AREA_RATIO = float(
    os.environ.get('YOLO_TARGET_ASSOCIATION_MIN_AREA_RATIO', '0.45')
)
YOLO_TARGET_ASSOCIATION_MAX_AREA_RATIO = float(
    os.environ.get('YOLO_TARGET_ASSOCIATION_MAX_AREA_RATIO', '2.20')
)
YOLO_ACQUISITION_CONFIRMATION_FRAMES = int(
    # YOLO pose detections are already required to have orange HSV support.
    # At the model's ~5 Hz inference rate, a second acquisition frame leaves
    # roughly 0.2-0.4 s of uncontrolled motion after a gate pass.  Accept the
    # first jointly confirmed instance so an edge gate can be captured before
    # it exits the camera.
    os.environ.get('YOLO_ACQUISITION_CONFIRMATION_FRAMES', '1')
)
# Immediately after a confirmed pass the gate behind the drone can remain as a
# very large partial YOLO box for a few frames.  It must not win the normal
# "largest visible gate" acquisition policy over the smaller next gate.
YOLO_POST_PASS_REJECTION_SECONDS = float(
    os.environ.get('YOLO_POST_PASS_REJECTION_SECONDS', '0.80')
)
YOLO_POST_PASS_MAX_AREA_RATIO = float(
    os.environ.get('YOLO_POST_PASS_MAX_AREA_RATIO', '0.18')
)
YOLO_REQUIRE_HSV_CONFIRMATION = _env_bool(
    'YOLO_REQUIRE_HSV_CONFIRMATION', True
)
YOLO_HSV_MIN_ORANGE_RATIO = float(
    os.environ.get('YOLO_HSV_MIN_ORANGE_RATIO', '0.08')
)
YOLO_HSV_MAX_ORANGE_RATIO = float(
    os.environ.get('YOLO_HSV_MAX_ORANGE_RATIO', '0.85')
)
YOLO_HSV_SIDE_BAND_FRACTION = float(
    os.environ.get('YOLO_HSV_SIDE_BAND_FRACTION', '0.28')
)
YOLO_HSV_MIN_SIDE_DENSITY = float(
    os.environ.get('YOLO_HSV_MIN_SIDE_DENSITY', '0.06')
)
YOLO_HSV_MIN_SUPPORTED_SIDES = int(
    os.environ.get('YOLO_HSV_MIN_SUPPORTED_SIDES', '2')
)
YOLO_CROP_PADDING_PX = int(os.environ.get('YOLO_CROP_PADDING_PX', '14'))
YOLO_MIN_GATE_AREA_PX = float(
    os.environ.get('YOLO_MIN_GATE_AREA_PX', '250')
)
YOLO_MAX_OUTSIDE_FRACTION = float(
    os.environ.get('YOLO_MAX_OUTSIDE_FRACTION', '0.35')
)
YOLO_PREVIOUS_CENTER_FRAMES = int(
    os.environ.get('YOLO_PREVIOUS_CENTER_FRAMES', '5')
)
YOLO_ESTIMATED_OPENING_SCALE = float(
    os.environ.get('YOLO_ESTIMATED_OPENING_SCALE', '0.72')
)
YOLO_INFERENCE_SIZE = int(os.environ.get('YOLO_INFERENCE_SIZE', '640'))
YOLO_DEVICE = os.environ.get('YOLO_DEVICE', '').strip() or None
YOLO_LOG_INTERVAL_S = float(os.environ.get('YOLO_LOG_INTERVAL_S', '1.0'))
YOLO_SCORE_CONFIDENCE_WEIGHT = float(
    os.environ.get('YOLO_SCORE_CONFIDENCE_WEIGHT', '0.40')
)
YOLO_SCORE_CENTER_WEIGHT = float(
    os.environ.get('YOLO_SCORE_CENTER_WEIGHT', '0.30')
)
YOLO_SCORE_AREA_WEIGHT = float(
    os.environ.get('YOLO_SCORE_AREA_WEIGHT', '0.30')
)
YOLO_SCORE_REFERENCE_AREA_RATIO = float(
    os.environ.get('YOLO_SCORE_REFERENCE_AREA_RATIO', '0.08')
)
YOLO_HSV_BLUR_KERNEL = int(os.environ.get('YOLO_HSV_BLUR_KERNEL', '5'))
YOLO_HSV_OPENING_KERNEL = int(
    os.environ.get('YOLO_HSV_OPENING_KERNEL', '3')
)
YOLO_HSV_CLOSING_KERNEL = int(
    os.environ.get('YOLO_HSV_CLOSING_KERNEL', '5')
)
YOLO_HSV_CENTER_BLEND = float(
    os.environ.get('YOLO_HSV_CENTER_BLEND', '0.25')
)
YOLO_HSV_CENTER_MAX_SHIFT_FRACTION = float(
    os.environ.get('YOLO_HSV_CENTER_MAX_SHIFT_FRACTION', '0.12')
)
# Strict YOLO+HSV is the safe default used for racing. This optional, lower
# confidence fallback exists for offline evaluation or a deliberately enabled
# recovery experiment.
GLOBAL_HSV_FALLBACK_ENABLED = _env_bool(
    'GLOBAL_HSV_FALLBACK_ENABLED', False
)
GLOBAL_HSV_FALLBACK_CONFIDENCE_SCALE = float(
    os.environ.get('GLOBAL_HSV_FALLBACK_CONFIDENCE_SCALE', '0.55')
)

# Initial crop-local segmentation uses the calibrated Q2 values. These remain
# environment-configurable without modifying detector code.
GATE_HSV_LOWER = _env_int_tuple('GATE_HSV_LOWER', (0, 75, 140))
GATE_HSV_UPPER = _env_int_tuple('GATE_HSV_UPPER', (23, 255, 255))
GATE_MIN_CONTOUR_AREA = float(
    os.environ.get('GATE_MIN_CONTOUR_AREA', '30')
)
