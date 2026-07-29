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
HOVER_THRUST    = float(os.environ.get('HOVER_THRUST', '0.264'))
# Phase 5: 0.30×2.5s → ~6 m; 0.272×1.0s still peaked ~6 m EKF. Keep pad
# clear soft — descent is handled by image-ny thrust once framed.
# Soft pad clear — 0.268×0.8s + hard pitch step was the Phase-5 jitter/loft.
TAKEOFF_THRUST  = float(os.environ.get('TAKEOFF_THRUST', '0.266'))
TAKEOFF_DURATION_S = float(os.environ.get('TAKEOFF_DURATION_S', '0.50'))
KP_THRUST       = 0.25      # vd cmd (m/s) → thrust delta
# Gate-two telemetry saturated the optical descent command for 1.6 seconds at
# ~0.263 tilt-compensated thrust while its image speed still accelerated to
# 0.476 normalized/s toward the bottom edge. Permit a modest tilted descent
# only while vision explicitly requests descent; hover/takeoff stay at 0.28.
MAX_DESCENT_THRUST_REDUCTION = 0.020
# Open-loop climb headroom was only +0.01 and could not hold altitude under
# forward lean when VIO thrust PI was not yet tracking a climb command.
MAX_ASCENT_THRUST_INCREASE = 0.030
MIN_TILT_COMPENSATION_COSINE = 0.70
# Phase 4.5 lean-hover (8° roll) holds at HT=0.264 with lean_boost=0.
# Forward crawl (4.6): 0.000 sank ~0.4, 0.005 climbed ~0.4 → 0.002 ≈ +0.05 m/s.
# Assist pose path: keep 0 so altitude is owned by PnP cam-Y, not lean add-on.
LEAN_THRUST_BOOST = float(os.environ.get('LEAN_THRUST_BOOST', '0.0'))
KP_LEAN         = 0.10      # forward command → pitch mapping
LATERAL_LEAN_SIGN = float(
    # Direct A/B runs show negative desired roll moves a right-side gate
    # farther inward than positive desired roll in the live VQ2 rate path.
    os.environ.get('LATERAL_LEAN_SIGN', '-1.0')
)
MAX_LEAN_RAD    = math.radians(25.0)
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
RATE_SIGN_ROLL  = 1.0       # +cmd → +truth roll (EKF roll sign fixed to match)
# Phase 4.7 yaw-align (2026-07-28): with -1.0, closed-loop yaw drove a
# right-offset gate farther off-frame (nx 0.54→0.96). +1.0 recentres.
RATE_SIGN_YAW   = float(os.environ.get('RATE_SIGN_YAW', '1.0'))
# Desired-pitch sign for *forward* translation. Old code used des_pitch<0
# ("nose-down"). drive_e (2026-07-28): des_pitch=-10° tracked in ATTITUDE
# but the craft translated backward away from gate 1 → flip to +1.
FORWARD_PITCH_SIGN = float(os.environ.get('FORWARD_PITCH_SIGN', '1.0'))
MAX_RATE_RAD_S  = 1.05      # demo rate_cap 0.35 * 3.0 max rate
MAX_THRUST      = 0.90
MIN_THRUST      = 0.05
CONTROL_HZ      = 100
TELEMETRY_TIMEOUT_S = 1.25
SENSOR_FUTURE_TOLERANCE_S = 0.05
CONTROL_MIN_DT_S = 1.0 / 500.0
CONTROL_MAX_DT_S = 0.05

# ---- PnP + IMU state estimation ----
# The VQ2 sim sends no attitude (deprecated) and no LOCAL_POSITION_NED.
# ekf_estimator.py / ekf/drone_ekf.py dead-reckon HIGHRES_IMU and correct with
# dual-gate PnP fixes (vision/dual_gate_pnp.py + vision/yolo_pnp.py),
# publishing shared_data['position_ned'] for the planner.
#
# PID tuning order (matches the procedure proven on the Q2_pnp branch):
#   1. HOVER_THRUST first — stationary hover, zero commanded velocity.
#   2. Thrust PI (KP_THRUST_VEL / KI_THRUST_VEL) against vertical steps.
#   3. Attitude PDs (KP_ATT / KD_ATT, KP_ROLL_ATT / KD_ROLL_ATT).
# The camera FOV is narrow, so keep YAW_RATE_MAX_DEG_S low or a fast yaw
# sweeps the gate out of frame.
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

# ---- Flight mode for main.py ----
# assist  = image-chase on the manual attitude+hover plant (default).
# kalman  = dual-gate PnP body-path / EKF geometric planner.
FLIGHT_MODE = os.environ.get('FLIGHT_MODE', 'assist').strip().lower()
if FLIGHT_MODE not in {'assist', 'kalman'}:
    raise ValueError('FLIGHT_MODE must be "assist" or "kalman"')

# Assist (image IBVS) knobs — no EKF position in the loop.
# 024550: yaw locked but loft→pitch-cap→ny=-0.7 with full lean (camera 20° up).
ASSIST_LEAN_DEG = float(os.environ.get('ASSIST_LEAN_DEG', '10.0'))
ASSIST_FWD_FRAC = float(os.environ.get('ASSIST_FWD_FRAC', '0.90'))
ASSIST_ROLL_SCALE = float(os.environ.get('ASSIST_ROLL_SCALE', '0.45'))
# Aim gate slightly below center — camera tilts 20° up from body forward.
ASSIST_NY_AIM = float(os.environ.get('ASSIST_NY_AIM', '0.22'))
ASSIST_NY_THRUST_GAIN = float(os.environ.get('ASSIST_NY_THRUST_GAIN', '0.050'))
ASSIST_LOST_TIMEOUT_S = float(os.environ.get('ASSIST_LOST_TIMEOUT_S', '0.8'))
# Through-opening only; abort early when next gate is visible (was 2.5 s
# blind with yaw=0 — lost gate-2 glimpse in 031742).
ASSIST_COAST_S = float(os.environ.get('ASSIST_COAST_S', '1.2'))
ASSIST_SEEK_S = float(os.environ.get('ASSIST_SEEK_S', '14.0'))
# Skip controller takeoff boost — assist clears the pad with soft lean only.
if FLIGHT_MODE == 'assist' and 'TAKEOFF_DURATION_S' not in os.environ:
    TAKEOFF_DURATION_S = 0.0
# LEAN_THRUST_BOOST default is 0 (pose/cam-Y owns altitude in assist).

# ---- Dual-gate EKF path planner knobs ----
# Geometric path (body-frame PnP): aim at gate_center - approach*through,
# then through/exit. Image IBVS is fallback when PnP drops.
KALMAN_USE_BODY_PATH = os.environ.get('KALMAN_USE_BODY_PATH', '1').strip() not in (
    '0', 'false', 'False', 'no', 'NO',
)
KALMAN_APPROACH_DISTANCE_M = float(
    os.environ.get('KALMAN_APPROACH_DISTANCE_M', '3.5')
)
KALMAN_EXIT_DISTANCE_M = float(
    os.environ.get('KALMAN_EXIT_DISTANCE_M', '1.5')
)
# Camera-optical Y of aim (down, metres) → thrust:
# thrust = hover - gain * ez_cam. Must NOT use raw body-z — 20° cam tilt
# puts an on-boresight gate at body_z ≈ -range*sin(20°) and lofted the pad.
KALMAN_BODY_Z_THRUST_GAIN = float(
    os.environ.get('KALMAN_BODY_Z_THRUST_GAIN', '0.028')
)
# Lateral body-y → roll lean scale (des_roll = -scale * ey/ex * max_lean).
KALMAN_BODY_Y_LEAN_SCALE = float(
    os.environ.get('KALMAN_BODY_Y_LEAN_SCALE', '0.55')
)
# 14° + hard far-pitch floor jittered the pad; 12° matches drive lean authority.
KALMAN_MAX_LEAN_DEG = float(os.environ.get('KALMAN_MAX_LEAN_DEG', '12.0'))
# Image-IBVS yaw (norm_x). 043043 saturated from lock jumps — keep moderate.
KALMAN_KP_YAW = float(os.environ.get('KALMAN_KP_YAW', '0.9'))
# Inner attitude loop actually flown on this branch: desired lean - measured
# lean -> body rate. These live in kalman_planner.py, NOT in the KP_ATT /
# KP_ROLL_ATT gains above (those only serve the velocity fallback path, which
# the kalman planner never takes). Tune these with tools/tune_flight.py.
KALMAN_KP_ATT = float(os.environ.get('KALMAN_KP_ATT', '2.2'))
KALMAN_KD_ATT = float(os.environ.get('KALMAN_KD_ATT', '0.10'))
KALMAN_MAX_RATE_RAD_S = float(
    os.environ.get('KALMAN_MAX_RATE_RAD_S', '0.9')
)
# Far-gate min forward pitch as fraction of max lean (0.42 ≈ 5° @ 12° max).
# Ramped by climbed so the pad is not slammed on arm.
KALMAN_FAR_PITCH_FRAC = float(os.environ.get('KALMAN_FAR_PITCH_FRAC', '0.42'))
# Image-ny descent gain (error relative to KALMAN_NY_AIM).
KALMAN_NY_DESC_GAIN = float(os.environ.get('KALMAN_NY_DESC_GAIN', '0.035'))
# Desired gate image-ny (cam tilts up ~20° → co-height gate sits low).
# phase5_n: aim=0 dug under a framed commit; crawl uses +0.08.
KALMAN_NY_AIM = float(os.environ.get('KALMAN_NY_AIM', '0.18'))
# Altitude brake ladder (metres above arm). Gate centre ≈ 1.5–2.5 m.
# dualgate_t: brake@1.0/hard@1.8 dumped thr≈0.253 mid-approach (sink
# before gate1). Hold until above typical hole height, then loft-cap.
KALMAN_ALT_BRAKE_M = float(os.environ.get('KALMAN_ALT_BRAKE_M', '1.6'))
KALMAN_ALT_HARD_M = float(os.environ.get('KALMAN_ALT_HARD_M', '2.4'))
KALMAN_ALT_MAX_M = float(os.environ.get('KALMAN_ALT_MAX_M', '3.0'))
KALMAN_ALT_EMERGENCY_M = float(os.environ.get('KALMAN_ALT_EMERGENCY_M', '3.8'))

# ---- Camera / geometry ----
GATE_INNER_M    = 1.5       # gate inner side (m) for range estimation
CAMERA_FOCAL_PX = 320.0     # fx = fy from spec
CAMERA_CX       = 320.0
CAMERA_CY       = 180.0
CAMERA_TILT_RAD = math.radians(20.0)   # camera tilted 20° UP from body forward

# ---- Vision runtime ----
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
# OLD-sim early-start DQ: arming/thrusting immediately after race start or
# cmd-31000 pad reset tips the craft off the pad. Hold still this long before
# every arm (main + crash re-arm). Tune scripts use the same default.
EARLY_START_HOLD_S        = float(
    os.environ.get('EARLY_START_HOLD_S', '3.5')
)
# After a floor crash (pos_d below spawn / Environment slam), send cmd 31000
# and re-arm so the client can keep practicing without a manual restart.
AUTO_RESET_ON_CRASH       = _env_bool('AUTO_RESET_ON_CRASH', True)
CRASH_FLOOR_D_M           = float(os.environ.get('CRASH_FLOOR_D_M', '0.45'))
CRASH_CONFIRM_S           = float(os.environ.get('CRASH_CONFIRM_S', '0.35'))
CRASH_RESET_COOLDOWN_S    = float(os.environ.get('CRASH_RESET_COOLDOWN_S', '4.0'))
CRASH_ENV_IMPULSE_MIN     = float(os.environ.get('CRASH_ENV_IMPULSE_MIN', '0.15'))
# VQ2 publishes no LOCAL_POSITION_NED, so crash logic there runs on the EKF.
# The VQ1 tuning build does publish it; set this to 0 on VQ1 to rehearse the
# exact crash behaviour VQ2 will give you.
CRASH_USE_SIM_ODOMETRY    = _env_bool('CRASH_USE_SIM_ODOMETRY', True)
# When False, EKF only integrates IMU (dead reckoning) — no dual-gate PnP
# corrections. Use on VQ1 to baseline IMU drift vs truth without vision.
EKF_USE_PNP               = _env_bool('EKF_USE_PNP', True)
# Zero preserves the normal race client, which runs until Ctrl+C. A positive
# value is useful for bounded simulator test attempts and exits through the
# regular cleanup/disarm path.
RUN_MAX_SECONDS           = float(
    os.environ.get('RUN_MAX_SECONDS', '0')
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
    # Backup window if persistent lock glitches; persistent owns approach.
    os.environ.get('YOLO_TARGET_LOCK_SECONDS', '2.0')
)
YOLO_PERSISTENT_TARGET_LOCK = _env_bool(
    'YOLO_PERSISTENT_TARGET_LOCK', True
)
YOLO_TARGET_ASSOCIATION_CENTER_SPAN = float(
    os.environ.get('YOLO_TARGET_ASSOCIATION_CENTER_SPAN', '1.85')
)
YOLO_TARGET_ASSOCIATION_MIN_AREA_RATIO = float(
    # Reject far/small steals while locked (was only checked at low IoU).
    os.environ.get('YOLO_TARGET_ASSOCIATION_MIN_AREA_RATIO', '0.45')
)
YOLO_TARGET_ASSOCIATION_MAX_AREA_RATIO = float(
    # Closing on a gate grows area fast; 2.2 dropped real matches → reacquire.
    os.environ.get('YOLO_TARGET_ASSOCIATION_MAX_AREA_RATIO', '4.0')
)
YOLO_ACQUISITION_CONFIRMATION_FRAMES = int(
    # YOLO-only mode accepts the first pose instance above the confidence
    # threshold. At ~5 Hz inference, a second confirmation frame leaves
    # roughly 0.2-0.4 s of uncontrolled motion after a gate pass.
    os.environ.get('YOLO_ACQUISITION_CONFIRMATION_FRAMES', '1')
)
# Immediately after a confirmed pass the gate behind the drone can remain as a
# very large partial YOLO box for a few frames.  It must not win the normal
# "largest visible gate" acquisition policy over the smaller next gate.
YOLO_POST_PASS_REJECTION_SECONDS = float(
    # 0826: 1.6 s let the passed-gate remnant (area~110k) win right after
    # the window closed. Keep rejecting oversized boxes longer.
    os.environ.get('YOLO_POST_PASS_REJECTION_SECONDS', '6.00')
)
YOLO_POST_PASS_MAX_AREA_RATIO = float(
    os.environ.get('YOLO_POST_PASS_MAX_AREA_RATIO', '0.12')
)
# Vision steering uses the YOLO pose model only — no HSV gate filter / fallback.
YOLO_REQUIRE_HSV_CONFIRMATION = _env_bool(
    'YOLO_REQUIRE_HSV_CONFIRMATION', False
)
# Gate-two/distant YOLO boxes often fill only a few percent orange after the
# opening is large in-frame. 0.08 rejected real detections (overlay showed the
# YOLO box while steering published NO GATE). Keep a small floor against sky.
YOLO_HSV_MIN_ORANGE_RATIO = float(
    os.environ.get('YOLO_HSV_MIN_ORANGE_RATIO', '0.03')
)
YOLO_HSV_MAX_ORANGE_RATIO = float(
    os.environ.get('YOLO_HSV_MAX_ORANGE_RATIO', '0.85')
)
YOLO_HSV_SIDE_BAND_FRACTION = float(
    os.environ.get('YOLO_HSV_SIDE_BAND_FRACTION', '0.28')
)
YOLO_HSV_MIN_SIDE_DENSITY = float(
    os.environ.get('YOLO_HSV_MIN_SIDE_DENSITY', '0.03')
)
YOLO_HSV_MIN_SUPPORTED_SIDES = int(
    os.environ.get('YOLO_HSV_MIN_SUPPORTED_SIDES', '1')
)
YOLO_CROP_PADDING_PX = int(os.environ.get('YOLO_CROP_PADDING_PX', '14'))
YOLO_MIN_GATE_AREA_PX = float(
    os.environ.get('YOLO_MIN_GATE_AREA_PX', '100')
)
YOLO_MAX_OUTSIDE_FRACTION = float(
    os.environ.get('YOLO_MAX_OUTSIDE_FRACTION', '0.35')
)
YOLO_PREVIOUS_CENTER_FRAMES = int(
    # Hold YOLO aim across brief miss / bad PnP (~pose rate) without
    # falling through to nearest-range dual_pnp.
    os.environ.get('YOLO_PREVIOUS_CENTER_FRAMES', '8')
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
    os.environ.get('YOLO_SCORE_CENTER_WEIGHT', '0.55')
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
    os.environ.get('YOLO_HSV_CENTER_BLEND', '0.0')
)
YOLO_HSV_CENTER_MAX_SHIFT_FRACTION = float(
    os.environ.get('YOLO_HSV_CENTER_MAX_SHIFT_FRACTION', '0.12')
)
# Kept off: racing uses YOLO pose only. Enable only for offline HSV recovery
# experiments.
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
    os.environ.get('GATE_MIN_CONTOUR_AREA', '12')
)
