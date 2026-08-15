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
# Trimmed down: 0.264 + vertical-rate hold was lofting in pilot (215823).
HOVER_THRUST    = float(os.environ.get('HOVER_THRUST', '0.255'))
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
# Hard attitude clamp in controller (pilot lean must stay ≤ this).
MAX_LEAN_RAD    = math.radians(
    float(os.environ.get('MAX_LEAN_DEG', '62.0'))
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
RATE_SIGN_ROLL  = 1.0       # +cmd → +truth roll (EKF roll sign fixed to match)
# Yaw command polarity into the sim rate path. Default +1.0 (user/live).
RATE_SIGN_YAW   = float(os.environ.get('RATE_SIGN_YAW', '1.0'))
# Desired-pitch sign for *forward* translation. Old code used des_pitch<0
# ("nose-down"). drive_e (2026-07-28): des_pitch=-10° tracked in ATTITUDE
# but the craft translated backward away from gate 1 → flip to +1.
FORWARD_PITCH_SIGN = float(os.environ.get('FORWARD_PITCH_SIGN', '1.0'))
MAX_RATE_RAD_S  = 1.05      # demo rate_cap 0.35 * 3.0 max rate
MAX_THRUST      = 0.90
MIN_THRUST      = 0.05
# VADR-TS-001 4.4 requires a command rate *below* 100 Hz, so 100 -- the rate
# every recorded result was flown at -- sits exactly on the boundary and is not
# strictly compliant. 99 is, and at a 1% difference in loop period it cannot
# plausibly change the flight the way an untested 90 might; the simulator went
# away before anything could be re-flown, so the smallest compliant step is the
# only one worth taking.
CONTROL_HZ      = int(os.environ.get('CONTROL_HZ', '99'))
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
    float(os.environ.get('YAW_RATE_MAX_DEG_S', '95.0'))
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
# race    = Li & de Croon classical: LS gate pose + drag EKF + PD/arc.
# policy  = HG-DAgger student (policy_planner.py); timed runs only — no gamepad.
FLIGHT_MODE = os.environ.get('FLIGHT_MODE', 'assist').strip().lower()
if FLIGHT_MODE not in {'assist', 'kalman', 'spline', 'race', 'policy'}:
    raise ValueError(
        'FLIGHT_MODE must be "assist", "kalman", "spline", "race" or "policy"'
    )
POLICY_WEIGHTS = os.environ.get('POLICY_WEIGHTS', 'models/policy.pt')

# ---- Spline waypoint following on DERIVED position (FLIGHT_MODE=spline) ----
# Capture/replay use the same EKF_USE_PNP so drift stays common-mode.
SPLINE_MISSION_PATH = os.environ.get(
    'SPLINE_MISSION_PATH', 'captured_waypoints.json'
)
SPLINE_CRUISE_MPS = float(os.environ.get('SPLINE_CRUISE_MPS', '2.0'))
SPLINE_FINISH_MPS = float(os.environ.get('SPLINE_FINISH_MPS', '0.0'))
SPLINE_A_LAT = float(os.environ.get('SPLINE_A_LAT', '4.0'))
SPLINE_A_LON = float(os.environ.get('SPLINE_A_LON', '2.5'))
SPLINE_LOOKAHEAD_M = float(os.environ.get('SPLINE_LOOKAHEAD_M', '1.5'))
SPLINE_LOOKAHEAD_TIME_S = float(
    os.environ.get('SPLINE_LOOKAHEAD_TIME_S', '0.6')
)
SPLINE_LOOKAHEAD_MAX_M = float(
    os.environ.get('SPLINE_LOOKAHEAD_MAX_M', '4.0')
)
SPLINE_YAW_LOOKAHEAD_M = float(
    os.environ.get('SPLINE_YAW_LOOKAHEAD_M', '2.5')
)
SPLINE_KP_YAW = float(os.environ.get('SPLINE_KP_YAW', '1.2'))
SPLINE_KP_VEL_LEAN = float(os.environ.get('SPLINE_KP_VEL_LEAN', '0.09'))
SPLINE_MAX_LEAN_DEG = float(os.environ.get('SPLINE_MAX_LEAN_DEG', '12.0'))
# When |yaw_err| exceeds this, kill pitch/roll and turn in place so we do not
# reverse-tip toward a carrot behind the nose (125956 flew "backward").
SPLINE_YAW_ALIGN_DEG = float(os.environ.get('SPLINE_YAW_ALIGN_DEG', '35.0'))
SPLINE_VERT_AUTH = float(os.environ.get('SPLINE_VERT_AUTH', '0.08'))
SPLINE_MAX_ALT_M = float(os.environ.get('SPLINE_MAX_ALT_M', '8.0'))
SPLINE_MAX_XTE_M = float(os.environ.get('SPLINE_MAX_XTE_M', '6.0'))
SPLINE_FINISH_TOL_M = float(os.environ.get('SPLINE_FINISH_TOL_M', '0.6'))
SPLINE_CAPTURE_PATH = os.environ.get(
    'SPLINE_CAPTURE_PATH', 'captured_waypoints.json'
)
# Continuous capture sample rate while recording a remember-path (Hz).
SPLINE_CAPTURE_HZ = float(os.environ.get('SPLINE_CAPTURE_HZ', '5.0'))
# Time-synced stick/command remember file (pilot --capture / --replay).
REMEMBER_PATH = os.environ.get('REMEMBER_PATH', 'captured_controls.json')
# Practice checkpoints (best through-gate pad attitude tapes). Sim cannot
# teleport, so --practice-from-gate N replays best attitude through gate N.
PRACTICE_DIR = os.environ.get('PRACTICE_DIR', 'practice')
PRACTICE_AUTO_SAVE = int(
    float(os.environ.get('PRACTICE_AUTO_SAVE', '1') or 0)
)

# Assist (image IBVS) knobs — no EKF position in the loop.
# 024550: yaw locked but loft→pitch-cap→ny=-0.7 with full lean (camera 20° up).
ASSIST_LEAN_DEG = float(os.environ.get('ASSIST_LEAN_DEG', '10.0'))
ASSIST_FWD_FRAC = float(os.environ.get('ASSIST_FWD_FRAC', '0.90'))
ASSIST_ROLL_SCALE = float(os.environ.get('ASSIST_ROLL_SCALE', '0.45'))
# Seek (pre-lock / ghost): fraction of ASSIST_ROLL_SCALE when not using roll plan.
ASSIST_SEEK_ROLL_FRAC = float(os.environ.get('ASSIST_SEEK_ROLL_FRAC', '0.85'))
# Lateral by roll: L/R translation from bank pulses; yaw only keeps gate in frame.
ASSIST_LATERAL_BY_ROLL = os.environ.get(
    'ASSIST_LATERAL_BY_ROLL', '1'
).strip() not in ('0', 'false', 'False', 'no', 'NO')
# One bank pulse hold time (s) → metres via ½·g·tan(φ)·T²·eff.
ASSIST_ROLL_PULSE_S = float(os.environ.get('ASSIST_ROLL_PULSE_S', '0.80'))
ASSIST_ROLL_PULSE_EFF = float(os.environ.get('ASSIST_ROLL_PULSE_EFF', '1.0'))
ASSIST_ROLL_PLAN_LEAN_FRAC = float(
    os.environ.get('ASSIST_ROLL_PLAN_LEAN_FRAC', '0.95')
)
ASSIST_ROLL_PLAN_DEAD_M = float(os.environ.get('ASSIST_ROLL_PLAN_DEAD_M', '0.20'))
# Far L/R in image (|nx|) → push lean toward full.
# Mid between 035017 (roll=0 right clip) and 035647 (over-bank right hit).
ASSIST_ROLL_EXTREME_NX = float(os.environ.get('ASSIST_ROLL_EXTREME_NX', '0.16'))
ASSIST_ROLL_EXTREME_BOOST = float(
    os.environ.get('ASSIST_ROLL_EXTREME_BOOST', '0.28')
)
# Coast punch: soft centre-seeking bank when |nx| grows (not full chase).
#
# Restored to the 0.50 that coast_center_roll() itself defaults to, after
# measuring what the 2026-07-30 runs that actually scored gate 1 commanded
# (042441/044140/044830/044304/042525/041807/043504/045233/045645, coast-phase
# rows binned by |nx|, |desired_roll| in degrees):
#
#   |nx|        flown median   flown p90   frac 0.30   frac 0.50
#   0.05-0.10        0.0          1.7         0.0         0.0
#   0.10-0.15        1.0          2.5         0.9         1.5
#   0.15-0.25        1.9          7.0         2.2         3.7
#   0.25+            5.6          7.7         3.5         5.9
#
# 0.30 could only produce 3.5 deg where those runs banked 5.6 -- a 37% authority
# deficit precisely when the gate is far off centre and the drone has to recover
# -- and total error against the flown medians is 4.55 deg at 0.30 against 2.84
# at 0.50, with every 0.50 value still inside the flown median..p90 envelope.
# The deadband stays at 0.10 because that matches the flown 0.0 deg median below
# |nx| 0.10. Unflown: the simulator was unavailable, so this is a restoration
# toward measured behaviour rather than new tuning. Override to revert.
ASSIST_COAST_ROLL_NX = float(os.environ.get('ASSIST_COAST_ROLL_NX', '0.10'))
ASSIST_COAST_ROLL_FRAC = float(os.environ.get('ASSIST_COAST_ROLL_FRAC', '0.50'))
# Forward speed in roll plan: faster → harder bank (less time to close ey).
ASSIST_ROLL_SPEED_REF_MPS = float(
    os.environ.get('ASSIST_ROLL_SPEED_REF_MPS', '3.5')
)
ASSIST_ROLL_SPEED_BOOST = float(
    os.environ.get('ASSIST_ROLL_SPEED_BOOST', '0.25')
)
# Extra lean when time-to-gate < time needed to close lateral gap.
ASSIST_ROLL_TTG_BOOST = float(os.environ.get('ASSIST_ROLL_TTG_BOOST', '0.35'))
# Floor on |cos α| so skewed heading still plans finite rolls (α = face-on angle).
ASSIST_ROLL_ALIGN_COS_MIN = float(
    os.environ.get('ASSIST_ROLL_ALIGN_COS_MIN', '0.25')
)
# Below this face-on cos, add soft yaw to re-square the nose for efficient rolls.
ASSIST_ROLL_ALIGN_YAW_COS = float(
    os.environ.get('ASSIST_ROLL_ALIGN_YAW_COS', '0.70')
)
# Both squaring yaws default off. They close a loop on the PnP through-axis,
# but gate_normal_* does not swing when the drone yaws -- 203406 held it at
# (0.94, 0.13) through a 0 -> -41.6 deg turn -- so the bearing error never
# shrinks and the nose winds off ~40 deg with the gate sitting dead centre
# (norm_x 0.005) at 10 m, carrying the drone 6 m wide of gate 1. Disabling
# both took a 1.0x batch from 0/8 gates and a collision every run to 2/6
# through gate 1 with four clean runs. yaw_keep_in_frame still yaws when the
# gate nears the frame edge, which is the only yaw the approach needs.
ASSIST_YAW_ALIGN_MAX_DEG = float(
    os.environ.get('ASSIST_YAW_ALIGN_MAX_DEG', '0.0')
)
ASSIST_YAW_ALIGN_KP = float(os.environ.get('ASSIST_YAW_ALIGN_KP', '1.4'))
# On the gate perpendicular / approach line → yaw to face gate centre + through.
ASSIST_CENTERLINE_EY_M = float(os.environ.get('ASSIST_CENTERLINE_EY_M', '0.55'))
ASSIST_CENTERLINE_NX = float(os.environ.get('ASSIST_CENTERLINE_NX', '0.16'))
ASSIST_CENTERLINE_YAW_DEAD_RAD = float(
    os.environ.get('ASSIST_CENTERLINE_YAW_DEAD_RAD', '0.04')
)
ASSIST_CENTERLINE_YAW_KP = float(
    os.environ.get('ASSIST_CENTERLINE_YAW_KP', '0.0')
)
ASSIST_CENTERLINE_YAW_MAX_DEG = float(
    os.environ.get('ASSIST_CENTERLINE_YAW_MAX_DEG', '45.0')
)
# Yaw when |nx| exceeds this (frame edge keep).
ASSIST_YAW_FRAME_NX = float(os.environ.get('ASSIST_YAW_FRAME_NX', '0.55'))
ASSIST_YAW_FRAME_MAX_DEG = float(
    os.environ.get('ASSIST_YAW_FRAME_MAX_DEG', '40.0')
)
ASSIST_YAW_FRAME_KP = float(os.environ.get('ASSIST_YAW_FRAME_KP', '2.2'))
# Legacy roll-first blend (used only when ASSIST_LATERAL_BY_ROLL=0).
ASSIST_ROLL_FIRST_NX = float(os.environ.get('ASSIST_ROLL_FIRST_NX', '0.18'))
ASSIST_ROLL_FIRST_NX_FULL = float(
    os.environ.get('ASSIST_ROLL_FIRST_NX_FULL', '0.38')
)
ASSIST_ROLL_FIRST_EY_M = float(os.environ.get('ASSIST_ROLL_FIRST_EY_M', '1.2'))
ASSIST_ROLL_FIRST_EY_FULL_M = float(
    os.environ.get('ASSIST_ROLL_FIRST_EY_FULL_M', '2.8')
)
ASSIST_ROLL_FIRST_YAW_FRAC = float(
    os.environ.get('ASSIST_ROLL_FIRST_YAW_FRAC', '0.28')
)
ASSIST_ROLL_FIRST_LEAN_FRAC = float(
    os.environ.get('ASSIST_ROLL_FIRST_LEAN_FRAC', '0.95')
)
# Scale gate-tracking roll/yaw and thrust deltas only — does NOT change aims,
# floors, punch/sink ranges, speed cap, or other hard-fought setpoints.
# 114108 @ 1.40: coast_lift thr→0.278 lofted to 4.9 m + gate collision;
# yaw sat 22°/s. Soften; vertical auth only applies on sink/climb (code).
ASSIST_LATERAL_AUTH = float(os.environ.get('ASSIST_LATERAL_AUTH', '1.20'))
ASSIST_VERTICAL_AUTH = float(os.environ.get('ASSIST_VERTICAL_AUTH', '1.25'))
# Aim gate slightly below center — camera tilts 20° up from body forward.
# 034220: 0.20 + seek/coast dig scraped bottom (climb 2.4→1.4, ny→−0.2).
ASSIST_NY_AIM = float(os.environ.get('ASSIST_NY_AIM', '0.12'))
# Image nx aim (normalized). 0 = fly the gate centre (033736: +0.03/+0.15 m
# left-bias + coast roll chase hit the right pillar at contact_h=+0.37).
ASSIST_NX_AIM = float(os.environ.get('ASSIST_NX_AIM', '0.0'))
# Body-right aim offset through the gate (m). 0 = centre of the opening.
ASSIST_POSE_AIM_Y_M = float(os.environ.get('ASSIST_POSE_AIM_Y_M', '0.0'))
# Global forward speed cap (m/s) — brakes forward lean in chase/coast/seek.
# Over cap → scale pitch down; well over → small reverse lean to scrub speed.
ASSIST_SPEED_CAP_MPS = float(os.environ.get('ASSIST_SPEED_CAP_MPS', '4.0'))
ASSIST_ALIGN_BRAKE_NX = float(os.environ.get('ASSIST_ALIGN_BRAKE_NX', '0.12'))
ASSIST_NY_THRUST_GAIN = float(os.environ.get('ASSIST_NY_THRUST_GAIN', '0.050'))
# Extra metres above pose-matched height on approach (NED-up). Off by default —
# 105106 approach-high + false dz=-2.8 climb lofted over gate 1.
ASSIST_APPROACH_HIGH_M = float(os.environ.get('ASSIST_APPROACH_HIGH_M', '0.0'))
# Approach tip sink (mild): 124213 top-rail / 124438 bottom-rail.
ASSIST_APPROACH_NY_OK = float(os.environ.get('ASSIST_APPROACH_NY_OK', '0.18'))
ASSIST_APPROACH_TIP_SINK = float(
    os.environ.get('ASSIST_APPROACH_TIP_SINK', '0.08')
)
ASSIST_APPROACH_TIP_MIN_ALT_M = float(
    os.environ.get('ASSIST_APPROACH_TIP_MIN_ALT_M', '1.20')
)
ASSIST_APPROACH_TIP_MIN_RANGE_M = float(
    os.environ.get('ASSIST_APPROACH_TIP_MIN_RANGE_M', '8.0')
)
# Cancel cam-tilt look-up in gate1 height when near image centre (085654 loft).
# Bias *grows* with forward speed / body pitch (more lean ⇒ more cam coupling).
# ASSIST_CAM_TILT_SPEED_MPS = speed that reaches +1 on the speed scale.
ASSIST_CAM_TILT_BIAS = float(os.environ.get('ASSIST_CAM_TILT_BIAS', '1.0'))
ASSIST_CAM_TILT_SPEED_MPS = float(
    os.environ.get('ASSIST_CAM_TILT_SPEED_MPS', '6.0')
)
# Cancel bank/strafe lateral cam coupling in image nx (both left and right).
# Fly left → cam looks right → gate appears too left; add +bias to nx.
# Grows with |roll| and lateral/forward speed (same speed scale as height bias).
ASSIST_CAM_ROLL_BIAS = float(os.environ.get('ASSIST_CAM_ROLL_BIAS', '0.40'))
# Persist online-learn gains across process restarts. Off by default —
# set ASSIST_LEARN_PERSIST=1 only when you explicitly want disk warm-start.
ASSIST_LEARN_PERSIST = os.environ.get(
    'ASSIST_LEARN_PERSIST', '0'
).strip() not in ('0', 'false', 'False', 'no', 'NO')
ASSIST_LEARN_PATH = os.environ.get(
    'ASSIST_LEARN_PATH',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'assist_learn.json'),
)
ASSIST_LEARN_SAVE_EVERY_N = int(os.environ.get('ASSIST_LEARN_SAVE_EVERY_N', '5'))
# Online learner: fit ASSIST_CAM_ROLL_BIAS from pose vs image residuals while banking.
ASSIST_CAM_ROLL_BIAS_LEARN = os.environ.get(
    'ASSIST_CAM_ROLL_BIAS_LEARN', '0'
).strip() not in ('0', 'false', 'False', 'no', 'NO')
ASSIST_CAM_ROLL_BIAS_LEARN_ALPHA = float(
    os.environ.get('ASSIST_CAM_ROLL_BIAS_LEARN_ALPHA', '0.08')
)
# Apply learned k from the first samples; ready flag for diagnostics.
ASSIST_CAM_ROLL_BIAS_LEARN_N = int(
    os.environ.get('ASSIST_CAM_ROLL_BIAS_LEARN_N', '12')
)
ASSIST_CAM_ROLL_BIAS_LEARN_MIN = float(
    os.environ.get('ASSIST_CAM_ROLL_BIAS_LEARN_MIN', '0.05')
)
ASSIST_CAM_ROLL_BIAS_LEARN_MAX = float(
    os.environ.get('ASSIST_CAM_ROLL_BIAS_LEARN_MAX', '1.20')
)
ASSIST_CAM_ROLL_BIAS_LEARN_ROLL_DEAD = float(
    os.environ.get('ASSIST_CAM_ROLL_BIAS_LEARN_ROLL_DEAD', '0.08')
)
ASSIST_CAM_ROLL_BIAS_LEARN_POSE_MAX = float(
    os.environ.get('ASSIST_CAM_ROLL_BIAS_LEARN_POSE_MAX', '0.85')
)
ASSIST_LOST_TIMEOUT_S = float(os.environ.get('ASSIST_LOST_TIMEOUT_S', '0.8'))
# Through-opening only; abort early when next gate is visible (was 2.5 s
# blind with yaw=0 — lost gate-2 glimpse in 031742).
ASSIST_COAST_S = float(os.environ.get('ASSIST_COAST_S', '0.8'))
# Keep punching the committed slot until GATE_PASSED (or this cap).
# 034542: 0.8 s coast ended → seek_chase on gate-2 nx≈+0.34 → edge hit;
# race pass arrived ~3 s later.
ASSIST_COMMIT_HOLD_S = float(os.environ.get('ASSIST_COMMIT_HOLD_S', '4.0'))
ASSIST_SEEK_S = float(os.environ.get('ASSIST_SEEK_S', '14.0'))
# Every duration above is read off a wall clock, which is correct at the 1.0x
# the drone actually races at. Cheat Engine's 0.2x exists only to make the
# course flyable by hand, and it silently compresses all of them 5x in sim
# terms -- the 4.0 s commit hold becomes 0.8 s of simulated flight. Enable
# this to retime them against the simulator so a slowed session behaves like
# a full-speed one; it is a no-op at 1.0x, where the two clocks agree.
ASSIST_SIM_CLOCK = _env_bool('ASSIST_SIM_CLOCK', False)
# Visual commit only when |nx| below this (035647: +0.107 already off-centre
# then coast over-banked right into the pillar).
ASSIST_COMMIT_NX_MAX = float(os.environ.get('ASSIST_COMMIT_NX_MAX', '0.09'))
# Yaw from PnP bearing atan2(ey,ex) (093229 image-nx under-yawed into left edge).
ASSIST_KP_YAW = float(os.environ.get('ASSIST_KP_YAW', '1.4'))  # legacy alias
# Proportional yaw — mild near centre; extreme |nx| boosts below.
ASSIST_KP_YAW_COARSE = float(os.environ.get('ASSIST_KP_YAW_COARSE', '2.6'))
ASSIST_KP_YAW_FINE = float(os.environ.get('ASSIST_KP_YAW_FINE', '1.55'))
ASSIST_YAW_COARSE_RAD = float(os.environ.get('ASSIST_YAW_COARSE_RAD', '0.12'))
ASSIST_YAW_MAX_DEG = float(os.environ.get('ASSIST_YAW_MAX_DEG', '90.0'))
ASSIST_YAW_SLEW_DEG = float(os.environ.get('ASSIST_YAW_SLEW_DEG', '280.0'))
# Far-left / far-right in frame (|nx| above start) → harder yaw (g2 acquire).
ASSIST_YAW_EXTREME_NX = float(os.environ.get('ASSIST_YAW_EXTREME_NX', '0.12'))
ASSIST_YAW_EXTREME_KP_MULT = float(
    os.environ.get('ASSIST_YAW_EXTREME_KP_MULT', '3.2')
)
ASSIST_YAW_EXTREME_MAX_MULT = float(
    os.environ.get('ASSIST_YAW_EXTREME_MAX_MULT', '2.0')
)
# |nx| above this → saturate yaw rate (same sign as offset).
# Raised after 132343: bang at 0.28 + left ghost = full circle.
ASSIST_YAW_BANG_NX = float(os.environ.get('ASSIST_YAW_BANG_NX', '0.45'))
# Seek left/right yaw — correct nx hard enough to center the next gate.
ASSIST_SEEK_YAW_MAX_DEG = float(os.environ.get('ASSIST_SEEK_YAW_MAX_DEG', '55.0'))
ASSIST_SEEK_YAW_KP = float(os.environ.get('ASSIST_SEEK_YAW_KP', '1.80'))
ASSIST_SEEK_LIVE_YAW_MAX_DEG = float(
    os.environ.get('ASSIST_SEEK_LIVE_YAW_MAX_DEG', '70.0')
)
ASSIST_SEEK_LIVE_YAW_KP = float(
    os.environ.get('ASSIST_SEEK_LIVE_YAW_KP', '2.40')
)
# Pad lift used to hard-cap ±8.6°/s (131746 far-right uncorrected).
ASSIST_PAD_YAW_MAX_DEG = float(os.environ.get('ASSIST_PAD_YAW_MAX_DEG', '45.0'))
# Blend latched/live pose bearing into seek yaw (pose owns lateral metres).
ASSIST_SEEK_YAW_POSE_WEIGHT = float(
    os.environ.get('ASSIST_SEEK_YAW_POSE_WEIGHT', '0.70')
)
# Lateral separation (body-right metres) scales yaw/roll both L and R.
ASSIST_LAT_SEP_REF_M = float(os.environ.get('ASSIST_LAT_SEP_REF_M', '2.0'))
ASSIST_LAT_SEP_DEAD_M = float(os.environ.get('ASSIST_LAT_SEP_DEAD_M', '0.35'))
ASSIST_YAW_LAT_SEP_KP_MULT = float(
    os.environ.get('ASSIST_YAW_LAT_SEP_KP_MULT', '2.8')
)
ASSIST_YAW_LAT_SEP_MAX_MULT = float(
    os.environ.get('ASSIST_YAW_LAT_SEP_MAX_MULT', '1.9')
)
ASSIST_YAW_BANG_EY_M = float(os.environ.get('ASSIST_YAW_BANG_EY_M', '2.2'))
ASSIST_ROLL_LAT_SEP_MULT = float(
    os.environ.get('ASSIST_ROLL_LAT_SEP_MULT', '2.0')
)
# Lateral→yaw learner (imaginary gates farther left/right → harder yaw).
ASSIST_LAT_YAW_LEARN = os.environ.get(
    'ASSIST_LAT_YAW_LEARN', '0'
).strip() not in ('0', 'false', 'False', 'no', 'NO')
ASSIST_LAT_YAW_LEARN_ALPHA = float(
    os.environ.get('ASSIST_LAT_YAW_LEARN_ALPHA', '0.14')
)
ASSIST_LAT_YAW_LEARN_N = int(os.environ.get('ASSIST_LAT_YAW_LEARN_N', '24'))
ASSIST_LAT_YAW_MILD_MULT = float(
    os.environ.get('ASSIST_LAT_YAW_MILD_MULT', '1.0')
)
ASSIST_LAT_YAW_HARD_MULT = float(
    os.environ.get('ASSIST_LAT_YAW_HARD_MULT', '2.6')
)
ASSIST_LAT_YAW_MULT_MIN = float(
    os.environ.get('ASSIST_LAT_YAW_MULT_MIN', '0.80')
)
ASSIST_LAT_YAW_MULT_MAX = float(
    os.environ.get('ASSIST_LAT_YAW_MULT_MAX', '3.20')
)
ASSIST_LAT_YAW_TAU_S = float(os.environ.get('ASSIST_LAT_YAW_TAU_S', '0.55'))
ASSIST_LAT_YAW_EY_KICK = float(os.environ.get('ASSIST_LAT_YAW_EY_KICK', '0.55'))
ASSIST_LAT_YAW_PRETRAIN = os.environ.get(
    'ASSIST_LAT_YAW_PRETRAIN', '0'
).strip() not in ('0', 'false', 'False', 'no', 'NO')
# Combined lateral+height envelope for through-gate punch.
ASSIST_GATE_OPENING_M = float(os.environ.get('ASSIST_GATE_OPENING_M', '1.5'))
ASSIST_HEIGHT_SEP_REF_M = float(
    os.environ.get('ASSIST_HEIGHT_SEP_REF_M', '1.2')
)
ASSIST_GATE_PUNCH_RANGE_M = float(
    os.environ.get('ASSIST_GATE_PUNCH_RANGE_M', '9.0')
)
# Nose-down while seeking to cancel 20° cam-up (level the view forward).
# Must stay on after lock too — 110826 crawl-only ~5° left the cam looking up.
ASSIST_SEEK_CAM_LEVEL_FRAC = float(
    os.environ.get('ASSIST_SEEK_CAM_LEVEL_FRAC', '0.80')
)
ASSIST_SEEK_PITCH_MAX_DEG = float(
    os.environ.get('ASSIST_SEEK_PITCH_MAX_DEG', '16.0')
)
# Pitched seek + HT still climbs (101541: 3→15 m). Base bleed while seeking.
ASSIST_SEEK_THRUST_BLEED = float(
    os.environ.get('ASSIST_SEEK_THRUST_BLEED', '0.014')
)
# Image-height trim while seeking — proportional to (ny - aim), not a flat dig.
ASSIST_SEEK_NY_DEAD = float(os.environ.get('ASSIST_SEEK_NY_DEAD', '0.10'))
ASSIST_SEEK_NY_THRUST_GAIN = float(
    os.environ.get('ASSIST_SEEK_NY_THRUST_GAIN', '0.035')
)
# Caps scale with |ny err| in code; these are the |err|=1 endpoints.
ASSIST_SEEK_NY_SINK_CAP = float(
    os.environ.get('ASSIST_SEEK_NY_SINK_CAP', '0.024')
)
ASSIST_SEEK_NY_CLIMB_CAP = float(
    os.environ.get('ASSIST_SEEK_NY_CLIMB_CAP', '0.022')
)
# Hard floor while seeking — pad bury only. 112536: 1.80 m arrested a good
# gate-2 sink (good runs 103411/095531 continued seek_sink down to ~0.5 m).
ASSIST_SEEK_MIN_ALT_M = float(os.environ.get('ASSIST_SEEK_MIN_ALT_M', '0.55'))
# Descent-rate brake (114816 overshot pad; 115153 braked too early at ~2 m).
# Full cancel only when YOLO near aim or climb ≤ BRAKE_ALT; while still low,
# only floor thrust at hover−MIN_SINK if falling ≥ FULL.
ASSIST_SEEK_DESCENT_START_MPS = float(
    os.environ.get('ASSIST_SEEK_DESCENT_START_MPS', '0.70')
)
ASSIST_SEEK_DESCENT_FULL_MPS = float(
    os.environ.get('ASSIST_SEEK_DESCENT_FULL_MPS', '1.20')
)
# 115626: +0.014 barely moved thr while falling ~2.5 m/s into gate-2 rail.
ASSIST_SEEK_DESCENT_BRAKE_THRUST = float(
    os.environ.get('ASSIST_SEEK_DESCENT_BRAKE_THRUST', '0.028')
)
ASSIST_SEEK_BRAKE_ALT_M = float(os.environ.get('ASSIST_SEEK_BRAKE_ALT_M', '1.15'))
ASSIST_SEEK_DESCENT_MIN_SINK = float(
    os.environ.get('ASSIST_SEEK_DESCENT_MIN_SINK', '0.028')
)
# Floor collective when climbed < MIN_ALT (pad / bottom-rail arrest).
ASSIST_SEEK_FLOOR_THRUST = float(
    os.environ.get('ASSIST_SEEK_FLOOR_THRUST', '0.022')
)
# Pose sink scale while seeking (1.0 = true |pose_dz|).
ASSIST_SEEK_POSE_SINK_SCALE = float(
    os.environ.get('ASSIST_SEEK_POSE_SINK_SCALE', '1.0')
)
# Seek sink gain vs approach (same ∝|dz| shape).
ASSIST_SEEK_SINK_GAIN_SCALE = float(
    os.environ.get('ASSIST_SEEK_SINK_GAIN_SCALE', '0.90')
)
# Post-pass alt freeze — off; sink is range-gated instead.
ASSIST_SEEK_HOLD_S = float(os.environ.get('ASSIST_SEEK_HOLD_S', '0.0'))
# Allow sink out to ~two-gates (103411 sank from ~22 m). Farther = hold.
ASSIST_SEEK_SINK_MAX_RANGE_M = float(
    os.environ.get('ASSIST_SEEK_SINK_MAX_RANGE_M', '24.0')
)
# Stop YOLO seek-sink when pose_dz is at/below this (gate height reached),
# but only if YOLO error is also small (see ASSIST_SEEK_POSE_STOP_NY_ERR).
ASSIST_SEEK_POSE_STOP_SINK_M = float(
    os.environ.get('ASSIST_SEEK_POSE_STOP_SINK_M', '0.25')
)
# 114438: pose_dz=0 with ny−aim≈0.5 held sink — require YOLO near aim too.
ASSIST_SEEK_POSE_STOP_NY_ERR = float(
    os.environ.get('ASSIST_SEEK_POSE_STOP_NY_ERR', '0.30')
)
# When pose missing: img_dz ≈ (ny−aim) * range * this → stop if ≤ STOP_SINK.
ASSIST_SEEK_NY_TO_DZ = float(os.environ.get('ASSIST_SEEK_NY_TO_DZ', '0.55'))
# Within this range, stop sinking if ny is not extreme — punch through.
ASSIST_SEEK_PUNCH_RANGE_M = float(
    os.environ.get('ASSIST_SEEK_PUNCH_RANGE_M', '9.0')
)
ASSIST_SEEK_PUNCH_NY_MAX = float(
    os.environ.get('ASSIST_SEEK_PUNCH_NY_MAX', '0.80')
)
# Seek may look up to TWO gates ahead for yaw/latch/chase — never farther
# (111515: 40–44 m end-course box).
ASSIST_SEEK_MAX_AHEAD_M = float(
    os.environ.get('ASSIST_SEEK_MAX_AHEAD_M', '28.0')
)
# After a pass, freeze the near next-gate latch so a mid-frame far box
# (111515: 8 m latch → 44 m) cannot overwrite aim/yaw.
ASSIST_LATCH_FREEZE_S = float(os.environ.get('ASSIST_LATCH_FREEZE_S', '5.0'))
# Reject any live chase whose range jumps this far past the latch / last near.
ASSIST_LATCH_MAX_RANGE_JUMP_M = float(
    os.environ.get('ASSIST_LATCH_MAX_RANGE_JUMP_M', '6.0')
)
# While closing on the active gate, snapshot gate2 as next-next latch so
# dual dropout at the slot does not age out the seed (115959 gate-3 miss).
ASSIST_LATCH_APPROACH_FREEZE_M = float(
    os.environ.get('ASSIST_LATCH_APPROACH_FREEZE_M', '12.0')
)
ASSIST_LATCH_SNAP_MAX_AGE_S = float(
    os.environ.get('ASSIST_LATCH_SNAP_MAX_AGE_S', '15.0')
)
# seek_lock may soft-yaw from latch this long after last refresh.
ASSIST_LATCH_HOLD_S = float(os.environ.get('ASSIST_LATCH_HOLD_S', '12.0'))
# 120804: ~8.7 m residual; 122209: ~12.3 m / ny≈−0.21 still stole gate-2 dig.
ASSIST_LATCH_MIN_AHEAD_M = float(
    os.environ.get('ASSIST_LATCH_MIN_AHEAD_M', '14.0')
)
# Approach snap only if next_rng > primary_rng + this (truly behind).
ASSIST_LATCH_BEHIND_MARGIN_M = float(
    os.environ.get('ASSIST_LATCH_BEHIND_MARGIN_M', '6.0')
)
# Next-gate latch must not sit above the image aim (cleared-gate / sky).
ASSIST_LATCH_NY_MIN = float(os.environ.get('ASSIST_LATCH_NY_MIN', '-0.05'))
# Post-pass: climb to this height before tip-crawl / dig when low/blind
# (115959 exited gate 2 at climb≈−0.2 m and never saw gate 3).
ASSIST_SEEK_CRUISE_ALT_M = float(
    os.environ.get('ASSIST_SEEK_CRUISE_ALT_M', '1.55')
)
ASSIST_SEEK_CRUISE_THRUST = float(
    os.environ.get('ASSIST_SEEK_CRUISE_THRUST', '0.016')
)
# Blind seek (no latch): slow L/R yaw scan instead of mute tip-crawl
# (121451: yaw=0 lofted to ~8 m and never found gate 3).
# Smaller/slower scan — 124804 ±0.32 @ 0.20 Hz shook the craft.
ASSIST_SEEK_SCAN_YAW_RAD = float(
    os.environ.get('ASSIST_SEEK_SCAN_YAW_RAD', '0.22')
)
ASSIST_SEEK_SCAN_HZ = float(os.environ.get('ASSIST_SEEK_SCAN_HZ', '0.12'))
# Course-2 memory after gate 1: yaw right + slight climb (user-allowed).
ASSIST_COURSE = int(os.environ.get('ASSIST_COURSE', '2'))
ASSIST_COURSE_MEMORY = os.environ.get('ASSIST_COURSE_MEMORY', '1') not in (
    '0', 'false', 'False',
)
# One-shot g1→g2 on course 2 only (not after later gates).
ASSIST_POST_G1_NX = float(os.environ.get('ASSIST_POST_G1_NX', '0.55'))
ASSIST_POST_G1_NY = float(os.environ.get('ASSIST_POST_G1_NY', '0.05'))
ASSIST_POST_G1_RANGE_M = float(os.environ.get('ASSIST_POST_G1_RANGE_M', '16.0'))
# Right turn after g1 — bounded by angle budget + max time (133354 spun).
ASSIST_POST_G1_YAW_DEG = float(os.environ.get('ASSIST_POST_G1_YAW_DEG', '40.0'))
# Pilot remember-path: synthetic yaw keys around gate tags.
PILOT_POST_G1_YAW_KEY = os.environ.get('PILOT_POST_G1_YAW_KEY', 'e')  # right
PILOT_POST_G1_YAW_DEG = float(os.environ.get('PILOT_POST_G1_YAW_DEG', '16.0'))
PILOT_POST_G1_YAW_RATE_DEG = float(
    os.environ.get('PILOT_POST_G1_YAW_RATE_DEG', '35.0')
)
PILOT_POST_G1_YAW_LEAD_S = float(
    os.environ.get('PILOT_POST_G1_YAW_LEAD_S', '0.45')
)
# Auto yaw-right after GATE 1 during live pilot. Off by default so manual
# W/A/S/D is sticks-only (no surprise turn). Remember --capture still bakes
# E via ensure_pilot_gate_yaws; set 1 to restore the live auto yaw.
PILOT_LIVE_POST_G1_YAW = int(
    float(os.environ.get('PILOT_LIVE_POST_G1_YAW', '0') or 0)
)
# Before GATE 2: yaw left to face gate 3.
PILOT_POST_G2_YAW_KEY = os.environ.get('PILOT_POST_G2_YAW_KEY', 'q')  # left
# Base turn onto gate 3, plus ~pitch-tilt compensation (nose ~14° up → path
# drifts; need extra left yaw or we miss the gate on the right).
PILOT_POST_G2_YAW_DEG = float(os.environ.get('PILOT_POST_G2_YAW_DEG', '120.0'))
# Negative = start AFTER gate_pass tag. Tag is early (~6.78); real GATE 2
# clears ~12.3s — delay so sharp Q begins only after the physical clear.
PILOT_POST_G2_YAW_LEAD_S = float(
    os.environ.get('PILOT_POST_G2_YAW_LEAD_S', '-6.07')
)
# Extra KEY Q after the main G2 yaw, roughly the forward lean angle.
PILOT_G3_TILT_YAW_DEG = float(os.environ.get('PILOT_G3_TILT_YAW_DEG', '8.0'))
# F sink (m/s): mild by default so GATE 1 dips stay gentle. After GATE 1
# latches, F uses the faster gate-2 dive rate instead.
PILOT_SINK_RATE = float(os.environ.get('PILOT_SINK_RATE', '2.5'))
PILOT_G2_SINK_RATE = float(os.environ.get('PILOT_G2_SINK_RATE', '2.5'))
# When trimming --keep-until-gate N, keep this many seconds of keys after the
# gate_pass tag so post-gate F/Q are not cut off at the latch instant.
# Gate-2 remember tag is early (~6.78); physical clear ~12.3. Keep enough
# post-tag keys (finish-W) when trimming --keep-until-gate 2.
PILOT_TRIM_AFTER_GATE_S = float(
    os.environ.get('PILOT_TRIM_AFTER_GATE_S', '6.8')
)
# After keep-until gate latches (and post-gate yaw finishes), wait this long
# before HUMAN handoff so the next-gate approach keys can play.
PILOT_HANDOFF_AFTER_GATE_S = float(
    os.environ.get('PILOT_HANDOFF_AFTER_GATE_S', '5.0')
)
# Hybrid remember: keys through GATE N, then ASSIST closed-loop for the next
# gate(s). 0/None = disabled. Typical: 1 → assist aims G2 after G1 clears.
PILOT_ASSIST_AFTER_GATE = int(
    float(os.environ.get('PILOT_ASSIST_AFTER_GATE', '0') or 0)
)
# After ASSIST clears this gate, hand sticks back to HUMAN (0 = assist-after+1).
PILOT_HUMAN_AFTER_GATE = int(
    float(os.environ.get('PILOT_HUMAN_AFTER_GATE', '0') or 0)
)
# Short settle after the assist-after gate before enabling ASSIST.
PILOT_ASSIST_AFTER_GATE_DELAY_S = float(
    os.environ.get('PILOT_ASSIST_AFTER_GATE_DELAY_S', '0.35')
)
# Human teleop: boost collective by 1/cos(tilt) while leaned so hard W/pitch
# does not drop altitude. 0 = flat HOVER (old feel; sinks on hard forward).
PILOT_TILT_COMPENSATE = int(
    float(os.environ.get('PILOT_TILT_COMPENSATE', '1') or 1)
)

# PlayStation / Xbox gamepad teleop (pygame). 1 = try to open a pad.
PILOT_GAMEPAD = int(float(os.environ.get('PILOT_GAMEPAD', '1') or 1))
# Stick feel. Higher SMOOTH = snappier (1.0 = no filter). Lower EXPO =
# more linear near center (faster reverse through zero).
PILOT_PAD_DEADZONE = float(os.environ.get('PILOT_PAD_DEADZONE', '0.12'))
PILOT_PAD_EXPO = float(os.environ.get('PILOT_PAD_EXPO', '0.22'))
PILOT_PAD_SMOOTH = float(os.environ.get('PILOT_PAD_SMOOTH', '0.85'))
# Fraction of full stick/trigger authority required before pilot arms. The old
# 1e-4 gate let decaying pad filters and resting trigger noise spool motors
# with "no input", which is an early-start DQ the moment hover thrust lifts
# the craft. 0.12 ≈ a deliberate nudge; keyboard full-presses still pass.
PILOT_ENGAGE_FRAC = float(os.environ.get('PILOT_ENGAGE_FRAC', '0.12'))
# Consecutive control ticks that must clear PILOT_ENGAGE_FRAC before arm.
PILOT_ENGAGE_TICKS = int(float(os.environ.get('PILOT_ENGAGE_TICKS', '3') or 3))
# Fraction of lean/yaw/climb when NOT holding boost (R2). 1.0 = always full.
# Full stick authority to the lean/yaw caps (triggers already own vertical).
PILOT_PAD_SOFT_GAIN = float(os.environ.get('PILOT_PAD_SOFT_GAIN', '1.0'))
PILOT_PAD_SOFT_YAW = float(os.environ.get('PILOT_PAD_SOFT_YAW', '1.0'))
# Full trigger climb/sink authority (triggers own vertical now).
PILOT_PAD_SOFT_THRUST = float(os.environ.get('PILOT_PAD_SOFT_THRUST', '1.0'))
# Right bumper: extra collective while held (on top of hover / rate climb).
PILOT_PAD_THRUST_BUMP = float(os.environ.get('PILOT_PAD_THRUST_BUMP', '0.07'))
# Stick axis indices if auto-detect is wrong (pygame order).
# Default: left stick roll/pitch (0/1), right stick X yaw (2).
PILOT_PAD_AXIS_ROLL = int(
    float(os.environ.get('PILOT_PAD_AXIS_ROLL', '0') or 0)
)
PILOT_PAD_AXIS_PITCH = int(
    float(os.environ.get('PILOT_PAD_AXIS_PITCH', '1') or 1)
)
PILOT_PAD_AXIS_YAW = int(float(os.environ.get('PILOT_PAD_AXIS_YAW', '2') or 2))
# Proven manual teleop stick (manual_20260730_005007 cleared gates 1–4).
# coach / localize --teleop / seed-lap flying prefer these over PILOT_*.
MANUAL_LEAN_DEG = float(os.environ.get('MANUAL_LEAN_DEG', '14.0'))
MANUAL_YAW_RATE_DEG = float(os.environ.get('MANUAL_YAW_RATE_DEG', '40.0'))
MANUAL_THRUST_STEP = float(os.environ.get('MANUAL_THRUST_STEP', '0.028'))
MANUAL_SINK_STEP = float(os.environ.get('MANUAL_SINK_STEP', '0.040'))

# Soft keyboard defaults for pilot/manual (full keys still hit these caps).
# Lean capped by MAX_LEAN_RAD. Climb/sink are rate setpoints (m/s).
PILOT_LEAN_DEG = float(os.environ.get('PILOT_LEAN_DEG', '52.0'))
# Forward/back pitch can be hotter than roll (W / left-stick Y).
PILOT_PITCH_LEAN_DEG = float(os.environ.get('PILOT_PITCH_LEAN_DEG', '60.0'))
PILOT_YAW_RATE_DEG = float(os.environ.get('PILOT_YAW_RATE_DEG', '130.0'))
# Pilot attitude rate ceiling (°/s). Separate from KALMAN_MAX_RATE so assist
# stays tame; 0.9 rad/s (~52°/s) made left↔right reverse feel sluggish.
PILOT_MAX_RATE_DEG = float(os.environ.get('PILOT_MAX_RATE_DEG', '160.0'))
# Acro (rate mode): sticks command body rates. No angle self-level, no lean
# cap — center stick = 0 rate (attitude stays where you left it).
ACRO_ROLL_RATE_DEG = float(os.environ.get('ACRO_ROLL_RATE_DEG', '180.0'))
ACRO_PITCH_RATE_DEG = float(os.environ.get('ACRO_PITCH_RATE_DEG', '180.0'))
ACRO_YAW_RATE_DEG = float(os.environ.get('ACRO_YAW_RATE_DEG', '160.0'))
# Acro trigger/R-F collective offsets (hotter than PILOT_*_AUTH).
ACRO_CLIMB_AUTH = float(os.environ.get('ACRO_CLIMB_AUTH', '0.55'))
ACRO_SINK_AUTH = float(os.environ.get('ACRO_SINK_AUTH', '0.55'))
# Acro thrust clamp — wider than fly/pilot so full RT can actually punch.
ACRO_THRUST_MIN = float(os.environ.get('ACRO_THRUST_MIN', '0.05'))
ACRO_THRUST_MAX = float(os.environ.get('ACRO_THRUST_MAX', '0.70'))
# Pad feel in acro only (mid-stick softer than full deflection).
ACRO_PAD_SOFT_GAIN = float(os.environ.get('ACRO_PAD_SOFT_GAIN', '0.55'))
ACRO_PAD_SOFT_YAW = float(os.environ.get('ACRO_PAD_SOFT_YAW', '0.45'))
ACRO_PAD_EXPO = float(os.environ.get('ACRO_PAD_EXPO', '0.45'))
ACRO_PAD_SMOOTH = float(os.environ.get('ACRO_PAD_SMOOTH', '0.55'))
# Client slow-mo (pair with Cheat Engine / DxWnd at the SAME factor).
# Toggle in pilot with O / D-pad ↓. Scale < 1 slows control rate + tape playhead.
PILOT_SLOW_MO = int(float(os.environ.get('PILOT_SLOW_MO', '0') or 0))
PILOT_SLOW_MO_SCALE = float(os.environ.get('PILOT_SLOW_MO_SCALE', '0.77'))
# Neutral-stick AHRS blend (roll gyro sign now matches EKF). Lets "level"
# track gravity mid-race without a full hover pause. Disagreement / tumble
# gates in read_pilot_attitude fall back to EKF. Set 0 to disable.
PILOT_LEVEL_AHRS = int(float(os.environ.get('PILOT_LEVEL_AHRS', '1') or 0))
PILOT_LEVEL_AHRS_BLEND_DEG = float(
    os.environ.get('PILOT_LEVEL_AHRS_BLEND_DEG', '12.0')
)
# EKF↔AHRS disagreement grows as the EKF drifts, so a flat "they differ, keep
# EKF" gate muted the correction exactly when it mattered (eligible on 11.6%
# of 141532). Past DRIFT_DEG, if the accelerometer can vouch for AHRS, widen
# the blend window instead of backing off. MAX_DISAGREE stays a hard bail.
PILOT_LEVEL_AHRS_DRIFT_DEG = float(
    os.environ.get('PILOT_LEVEL_AHRS_DRIFT_DEG', '8.0')
)
PILOT_LEVEL_AHRS_DRIFT_WIDEN = float(
    os.environ.get('PILOT_LEVEL_AHRS_DRIFT_WIDEN', '2.5')
)
PILOT_LEVEL_AHRS_YAW_GATE_DEG = float(
    os.environ.get('PILOT_LEVEL_AHRS_YAW_GATE_DEG', '35.0')
)
PILOT_LEVEL_AHRS_MAX_DISAGREE_DEG = float(
    os.environ.get('PILOT_LEVEL_AHRS_MAX_DISAGREE_DEG', '60.0')
)
# Brief near-level window → snap EKF roll/pitch to accel (keep yaw).
PILOT_LEVEL_REALIGN_S = float(os.environ.get('PILOT_LEVEL_REALIGN_S', '0.20'))

PILOT_CLIMB_RATE = float(os.environ.get('PILOT_CLIMB_RATE', '2.8'))
PILOT_CLIMB_AUTH = float(os.environ.get('PILOT_CLIMB_AUTH', '0.24'))
# Extra collective cut for LT/F vs RT/R (must beat a hot hover).
PILOT_SINK_AUTH = float(os.environ.get('PILOT_SINK_AUTH', '0.30'))
# 0 = triggers/keys map straight to collective (default). 1 = close a
# climb-rate loop on EKF vz — disabled because bad vz pegs +auth at
# "hover" and LT cannot overcome the runaway climb (run 215823).
PILOT_RATE_HOLD = int(float(os.environ.get('PILOT_RATE_HOLD', '0') or 0))
ASSIST_POST_G1_YAW_EXTRA_PER_NX = float(
    os.environ.get('ASSIST_POST_G1_YAW_EXTRA_PER_NX', '15.0')
)
ASSIST_POST_G1_YAW_RATE_MAX_DEG = float(
    os.environ.get('ASSIST_POST_G1_YAW_RATE_MAX_DEG', '45.0')
)
ASSIST_POST_G1_YAW_FLOOR_DEG = float(
    os.environ.get('ASSIST_POST_G1_YAW_FLOOR_DEG', '22.0')
)
ASSIST_POST_G1_YAW_DEAD_DEG = float(
    os.environ.get('ASSIST_POST_G1_YAW_DEAD_DEG', '5.0')
)
ASSIST_POST_G1_YAW_MAX_S = float(
    os.environ.get('ASSIST_POST_G1_YAW_MAX_S', '1.8')
)
# Live handoff after turn done: reject floor + far-left ghosts (133934).
# Near-center / slight-left is OK once the right turn finished.
ASSIST_POST_G1_LIVE_NX_MIN = float(
    os.environ.get('ASSIST_POST_G1_LIVE_NX_MIN', '0.22')
)
ASSIST_POST_G1_LIVE_NX_LEFT = float(
    os.environ.get('ASSIST_POST_G1_LIVE_NX_LEFT', '-0.18')
)
ASSIST_POST_G1_LIVE_NY_MAX = float(
    os.environ.get('ASSIST_POST_G1_LIVE_NY_MAX', '0.62')
)
ASSIST_POST_G1_LIVE_AREA_MIN = float(
    os.environ.get('ASSIST_POST_G1_LIVE_AREA_MIN', '700.0')
)
# After turn done: block far-left ghost yaw; allow fine left to center g2.
ASSIST_POST_G1_YAW_LOCK_S = float(
    os.environ.get('ASSIST_POST_G1_YAW_LOCK_S', '2.5')
)
ASSIST_POST_G1_FINE_YAW_DEG = float(
    os.environ.get('ASSIST_POST_G1_FINE_YAW_DEG', '22.0')
)
ASSIST_POST_G1_CLIMB_THRUST = float(
    os.environ.get('ASSIST_POST_G1_CLIMB_THRUST', '0.010')
)
# Pitch slew — 125233 tip↔brake every tick shook the craft.
ASSIST_PITCH_SLEW_DEG = float(os.environ.get('ASSIST_PITCH_SLEW_DEG', '60.0'))
# Bleed collective when blind-seeking above cruise (stop the loft).
ASSIST_SEEK_SCAN_CAP_THRUST = float(
    os.environ.get('ASSIST_SEEK_SCAN_CAP_THRUST', '0.022')
)
# Hard alt while seeking — 123610 lofted to ~6 m after scan-cap window died.
ASSIST_SEEK_CEILING_M = float(os.environ.get('ASSIST_SEEK_CEILING_M', '2.35'))
# Legacy alias (tests / env); prefer ASSIST_SEEK_POSE_SINK_SCALE.
ASSIST_SEEK_POSE_SINK_BOOST = float(
    os.environ.get(
        'ASSIST_SEEK_POSE_SINK_BOOST',
        os.environ.get('ASSIST_SEEK_POSE_SINK_SCALE', '1.0'),
    )
)
# Keep last box through SEARCH blips. Short — long ghost + edge nx spun
# circles in 031737. Soft yaw only (no bang) while ghosting.
ASSIST_SEEK_GHOST_S = float(os.environ.get('ASSIST_SEEK_GHOST_S', '0.55'))
ASSIST_SEEK_GHOST_YAW_MAX_DEG = float(
    os.environ.get('ASSIST_SEEK_GHOST_YAW_MAX_DEG', '28.0')
)
# Seeking chaseable: allow lower-in-frame / smaller next-gate boxes.
ASSIST_SEEK_NY_MAX = float(os.environ.get('ASSIST_SEEK_NY_MAX', '0.92'))
ASSIST_SEEK_MIN_AREA = float(os.environ.get('ASSIST_SEEK_MIN_AREA', '180.0'))
# After first post-pass glimpse: keep at least this forward lean (closes range).
ASSIST_SEEK_CRAWL_DEG = float(os.environ.get('ASSIST_SEEK_CRAWL_DEG', '5.5'))
# Post-pass: stable frames before full chase (was 12 — gate-2 flash too short).
ASSIST_LOCK_FRAMES = int(os.environ.get('ASSIST_LOCK_FRAMES', '5'))
ASSIST_LOCK_NX_JUMP = float(os.environ.get('ASSIST_LOCK_NX_JUMP', '0.18'))
# Legacy pose blend weight (geometry path prefers ASSIST_SEEK_YAW_POSE_WEIGHT).
ASSIST_YAW_POSE_WEIGHT = float(os.environ.get('ASSIST_YAW_POSE_WEIGHT', '0.55'))
ASSIST_HFOV_DEG = float(os.environ.get('ASSIST_HFOV_DEG', '70.0'))
# Hold yaw near zero inside this bearing error (rad).
ASSIST_YAW_ALIGN_DEAD_RAD = float(
    os.environ.get('ASSIST_YAW_ALIGN_DEAD_RAD', '0.035')
)
# Legacy aliases — lateral sep gain is now symmetric via ASSIST_*_LAT_SEP_*.
ASSIST_YAW_LEFT_MISS_BOOST = float(
    os.environ.get('ASSIST_YAW_LEFT_MISS_BOOST', '1.0')
)
ASSIST_ROLL_LEFT_MISS_BOOST = float(
    os.environ.get('ASSIST_ROLL_LEFT_MISS_BOOST', '1.0')
)
# Skip controller takeoff boost — assist clears the pad with soft lean only.
if FLIGHT_MODE in ('assist', 'race') and 'TAKEOFF_DURATION_S' not in os.environ:
    TAKEOFF_DURATION_S = 0.0
# LEAN_THRUST_BOOST default is 0 (geometric gate1 height owns altitude in assist).

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

# ---- Race mode (Li & de Croon classical) ----
# Drag: a_body = k * v_body. Negative k (force opposes motion). Fitted offline
# by tools/identify_drag.py when telem returns; these defaults are order-of-
# magnitude from earlier pitched-flight ax measurements.
DRAG_KX = float(os.environ.get('DRAG_KX', '-0.50'))
DRAG_KY = float(os.environ.get('DRAG_KY', '-0.50'))
RACE_COURSE_MAP = os.environ.get('RACE_COURSE_MAP', 'course_map.json')
# Forward drive for FLIGHT_MODE=race. POSITIVE is forward on this plant:
# FORWARD_PITCH_SIGN is +1 and the manual stick maps W to +lean. The paper's
# eq. 22 writes a small negative theta_c in the aerospace convention (nose-down
# negative), and transcribing that literally gave -5 deg, which pitches *back*
# by five degrees -- the drone aligned with gates beautifully and hovered.
# Measured reference: human race laps hold +0.61 to +0.65 rad (35-37 deg).
RACE_PITCH_DEG = float(os.environ.get('RACE_PITCH_DEG', '20.0'))
# Collective must rise with lean or the drone sinks while driving forward:
# at 35 deg, cos(tilt) = 0.82, so level thrust leaves 18% of lift missing.
RACE_TILT_COMPENSATE = _env_bool('RACE_TILT_COMPENSATE', True)
RACE_MAX_LEAN_DEG = float(os.environ.get('RACE_MAX_LEAN_DEG', '12.0'))
# Paper section 4: "A PD controller is employed to steer the drone to y = 0 by
# phi = kv(kp(0 - y) - vy), where kp = 1 and kv = 2".
RACE_KP_LAT = float(os.environ.get('RACE_KP_LAT', '1.0'))
RACE_KD_LAT = float(os.environ.get('RACE_KD_LAT', '2.0'))
# Unused in the straight part: the paper fixes psi_c = 0 and steers with roll
# alone. Kept only for the arc, where heading tracks the turn (eq. 29).
RACE_YAW_KP = float(os.environ.get('RACE_YAW_KP', '0.8'))
# Vertical authority, deliberately timid. Raising the gain to 0.09 and the
# ceiling to 0.62 made the drone climb away at full collective, because the
# vertical the planner steers on is not trustworthy: it reads +5.5 m (drone
# below the gate) whenever the craft pitches 20 deg forward, which is the size
# of error a mishandled camera tilt produces at 10 m range. Do not re-open this
# authority until the vertical estimate itself is fixed.
RACE_VERT_THRUST_GAIN = float(os.environ.get('RACE_VERT_THRUST_GAIN', '0.04'))
RACE_THRUST_MIN = float(os.environ.get('RACE_THRUST_MIN', '0.20'))
RACE_THRUST_MAX = float(os.environ.get('RACE_THRUST_MAX', '0.40'))
RACE_ARC_RADIUS_M = float(os.environ.get('RACE_ARC_RADIUS_M', '1.5'))
RACE_ARC_TURN_RAD = float(os.environ.get('RACE_ARC_TURN_RAD', str(math.radians(90.0))))
RACE_ARC_MAX_S = float(os.environ.get('RACE_ARC_MAX_S', '2.0'))
RACE_COMMIT_RANGE_M = float(os.environ.get('RACE_COMMIT_RANGE_M', '1.2'))
RACE_POSE_MAX_AGE_S = float(os.environ.get('RACE_POSE_MAX_AGE_S', '0.25'))
# How long to keep steering on a dead-reckoned gate pose after vision drops.
# Detection falls from 65% at level to 31% past 40 deg of pitch (the camera is
# tilted up 20 deg, so fast forward flight aims it at the ground), and blind
# stretches on real laps ran 5-6 s of wall time. Discarding the gate after
# RACE_POSE_MAX_AGE_S left the planner holding level through all of it. The
# paper instead propagates on drag-model velocity and lets the next detection
# correct the drift. Beyond a couple of seconds the estimate is not worth having.
RACE_POSE_HOLD_S = float(os.environ.get('RACE_POSE_HOLD_S', '2.0'))
RACE_MAX_RANGE_M = float(os.environ.get('RACE_MAX_RANGE_M', '40.0'))
RACE_MAX_RESIDUAL_M = float(os.environ.get('RACE_MAX_RESIDUAL_M', '0.6'))
RACE_MIN_KP_CONF = float(os.environ.get('RACE_MIN_KP_CONF', '0.25'))

# ---- Vision runtime ----
VISION_UDP_IP             = '0.0.0.0'
VISION_UDP_PORT           = 5600
VISION_COMMAND_TIMEOUT_S  = 1.25
VISION_DEBUG              = _env_bool('VISION_DEBUG', False)
VISION_DISPLAY            = _env_bool('VISION_DISPLAY', True)
PERCEPTION_ONLY           = _env_bool('PERCEPTION_ONLY', False)
RESET_SIM_ON_START        = _env_bool('RESET_SIM_ON_START', False)
SIM_RESET_SETTLE_S        = float(
    os.environ.get('SIM_RESET_SETTLE_S', '1.0')
)
# Total seconds from sim-reset until arm (single aligned wait — do not stack
# settle + hold). Sim countdown is ~3s; 3.0 was −440ms early-start DQ, 3.45
# was −50ms. 3.40 keeps a thin margin without a post-GO dead stick feel.
EARLY_START_HOLD_S        = float(
    os.environ.get('EARLY_START_HOLD_S', '3.40')
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
# ---- Bounded attitude references (stop roll/pitch/yaw integrating away) ----
# Gyro integration has no absolute reference, so "level" walks a few tenths of
# a degree per second (141532 drifted +4° → −23° over 50 s). Two aids bound it.
#
# 1. Accelerometer tilt while coasting. OFF by default: "leaned under thrust"
#    and "level but the estimate is wrong" produce a byte-identical
#    accelerometer signal (both |acc_ned| = g·sin θ), so no gate built on
#    accel alone can separate them. Loose enough to fix a 12° drift is loose
#    enough to drag a held 30° lean flat again — which is exactly what
#    step_pitch_20260728_174353 did. Needs an independent velocity or
#    attitude reference, i.e. aid 2. Left wired for experiments.
EKF_ACCEL_TILT_GAIN       = float(os.environ.get('EKF_ACCEL_TILT_GAIN', '0.0'))
EKF_ACCEL_TILT_MAX_ACC    = float(
    os.environ.get('EKF_ACCEL_TILT_MAX_ACC', '1.5')
)
EKF_ACCEL_TILT_MAX_RATE_DEG = float(
    os.environ.get('EKF_ACCEL_TILT_MAX_RATE_DEG', '25.0')
)
# 2. The gate itself. An upright gate's vertical axis IS gravity, and PnP
#    recovers it exactly at any tilt — unlike accel it is immune to
#    acceleration. Gates are not perfectly vertical, so pull slowly and drop
#    large disagreements. Set the gains to 0 to disable either aid.
#    Runs as a Mahony filter: proportional pull on attitude (_GAIN) plus an
#    integral term into gyro bias (_BIAS_GAIN). The bias term is the one that
#    matters — it removes the cause, so attitude still holds through the
#    0.6–2.5 s stretches with no gate in view (152912 had a near gate only
#    64% of the time). _MAX_STEP_DEG clamps any single correction so one bad
#    pose can nudge but never yank.
#    DEFAULT OFF. Measured on run 155234 the live gate horizon arrives at only
#    ~2 Hz with ~19° spread and a ~12° pitch offset, and at that quality the
#    aid is worth exactly nothing: mean attitude error 26.7° with it versus
#    26.8° without, while its bias integral railed and made real flights worse.
#    It is not the idea that fails, it is the input — the same filter reaches
#    3.7° at 5 Hz / 10° spread and 1.5° at 15 Hz / 5°. Turn it back on once
#    vision clears roughly 5 Hz and 10°; `gh_fixes` and `gh_innov_*` in the
#    telemetry CSV are how you check.
EKF_GATE_HORIZON_GAIN     = float(
    os.environ.get('EKF_GATE_HORIZON_GAIN', '0.0')
)
EKF_GATE_HORIZON_MAX_STEP_DEG = float(
    os.environ.get('EKF_GATE_HORIZON_MAX_STEP_DEG', '1.0')
)
EKF_GATE_HORIZON_BIAS_GAIN = float(
    os.environ.get('EKF_GATE_HORIZON_BIAS_GAIN', '0.30')
)
# Roll and pitch from a planar square are NOT equally trustworthy. A gate seen
# near head-on barely constrains its own out-of-plane tilt, so corner jitter
# lands almost entirely in pitch — measured at 15 m, 1 px of corner error is
# 0.9° of roll but 7.6° of pitch. Live runs bear that out: gate roll matched
# the accelerometer to +0.6° median while pitch sat ~12° off with ~19° spread.
EKF_GATE_HORIZON_PITCH_SCALE = float(
    os.environ.get('EKF_GATE_HORIZON_PITCH_SCALE', '0.25')
)
# Anti-windup. The bias integral is only meaningful once the proportional term
# has pulled the innovation small; integrating a large one just drives bias to
# the rail (155234/155700 both pinned it at the old 8°/s limit and then flew on
# that fiction). Nothing real exceeds ~1.5°/s.
EKF_GATE_BIAS_INNOV_MAX_DEG = float(
    os.environ.get('EKF_GATE_BIAS_INNOV_MAX_DEG', '8.0')
)
EKF_GYRO_BIAS_LIMIT_DPS   = float(
    os.environ.get('EKF_GYRO_BIAS_LIMIT_DPS', '1.5')
)
EKF_GATE_YAW_GAIN         = float(os.environ.get('EKF_GATE_YAW_GAIN', '0.0'))
EKF_GATE_YAW_MAX_STEP_DEG = float(
    os.environ.get('EKF_GATE_YAW_MAX_STEP_DEG', '1.0')
)
EKF_GATE_YAW_BIAS_GAIN    = float(
    os.environ.get('EKF_GATE_YAW_BIAS_GAIN', '0.20')
)
# Outliers are screened on solve quality, never on distance from the filter's
# own belief — a drifted filter would reject the evidence that it drifted.
# Gate rotation degrades with range long before its centre does, so attitude
# (not position) is taken from near, well-reprojecting, non-held frames only.
EKF_GATE_ATT_MAX_RANGE_M  = float(
    os.environ.get('EKF_GATE_ATT_MAX_RANGE_M', '30.0')
)
EKF_GATE_ATT_MAX_REPROJ_PX = float(
    os.environ.get('EKF_GATE_ATT_MAX_REPROJ_PX', '6.0')
)
# Zero preserves the normal race client, which runs until Ctrl+C. A positive
# value is useful for bounded simulator test attempts and exits through the
# regular cleanup/disarm path.
RUN_MAX_SECONDS           = float(
    os.environ.get('RUN_MAX_SECONDS', '0')
)
VISION_DEBUG_DIR          = os.environ.get('VISION_DEBUG_DIR', '_vision_debug')
VISION_STATS_INTERVAL_S   = float(
    os.environ.get('VISION_STATS_INTERVAL_S', '5.0')
)
VISION_DEBUG_INTERVAL_S   = float(
    os.environ.get('VISION_DEBUG_INTERVAL_S', '5.0')
)
GATE_FRAME_CAPTURE        = _env_bool('GATE_FRAME_CAPTURE', True)
GATE_FRAME_CAPTURE_DIR    = os.environ.get(
    'GATE_FRAME_CAPTURE_DIR', 'frames'
)
# Minimum spacing between saved frames. Zero saves every real detector hit,
# which is what filled the old flat frames/ directory with 282k images; 1/3 s
# caps it at three per second, which is plenty for building a label set.
GATE_FRAME_CAPTURE_INTERVAL_S = float(
    os.environ.get('GATE_FRAME_CAPTURE_INTERVAL_S', '0.3333')
)

# ``auto`` prefers the four-keypoint pose model, then the YOLO-box/HSV hybrid,
# and finally prints an explicit warning before preserving the legacy HSV
# detector. Explicit backends fail fast when their weights are missing.
GATE_DETECTOR_BACKEND = os.environ.get(
    'GATE_DETECTOR_BACKEND', 'yolo_pose'
).strip().lower()
if GATE_DETECTOR_BACKEND not in {
    'auto', 'yolo_pose', 'yolo_hybrid', 'hsv', 'gatenet'
}:
    raise ValueError(
        'GATE_DETECTOR_BACKEND must be "auto", "yolo_pose", '
        '"yolo_hybrid", "hsv", or "gatenet"'
    )

YOLO_POSE_MODEL_PATH = os.environ.get(
    'YOLO_POSE_MODEL_PATH', 'models/gate_pose.pt'
)
YOLO_MODEL_PATH = os.environ.get(
    'YOLO_MODEL_PATH', 'models/gate_detector.pt'
)
YOLO_GATE_CLASS_NAME = os.environ.get('YOLO_GATE_CLASS_NAME', 'gate')
YOLO_CONFIDENCE_THRESHOLD = float(
    os.environ.get('YOLO_CONFIDENCE_THRESHOLD', '0.25')
)
YOLO_KEYPOINT_CONFIDENCE_THRESHOLD = float(
    os.environ.get('YOLO_KEYPOINT_CONFIDENCE_THRESHOLD', '0.25')
)
YOLO_NMS_IOU_THRESHOLD = float(
    os.environ.get('YOLO_NMS_IOU_THRESHOLD', '0.70')
)
YOLO_TARGET_LOCK_SECONDS = float(
    # Backup window if persistent lock glitches; persistent owns approach.
    os.environ.get('YOLO_TARGET_LOCK_SECONDS', '1.2')
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
# While locked, a different box this many times larger is treated as a
# nearer gate and may steal identity. Same-gate growth is protected by
# IoU (steal requires IoU < 0.20), so 1.6 is enough to take a closer
# instance without waiting until it fills the frame.
YOLO_TARGET_STEAL_AREA_RATIO = float(
    os.environ.get('YOLO_TARGET_STEAL_AREA_RATIO', '1.6')
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
    # Only used to prefer a smaller next gate over a huge remnant when
    # both are visible. Sole boxes are never dropped. 2 s is enough for
    # the remnant to leave; 6 s kept rejecting the approaching next gate
    # once it grew past 12% of the frame.
    os.environ.get('YOLO_POST_PASS_REJECTION_SECONDS', '2.00')
)
YOLO_POST_PASS_MAX_AREA_RATIO = float(
    os.environ.get('YOLO_POST_PASS_MAX_AREA_RATIO', '0.12')
)
# YOLO boxes are not colour-gated. A leftover YOLO_REQUIRE_HSV_CONFIRMATION=1
# in the environment used to turn this back on and drop real pose hits.
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
    os.environ.get('YOLO_SCORE_CONFIDENCE_WEIGHT', '0.15')
)
YOLO_SCORE_CENTER_WEIGHT = float(
    os.environ.get('YOLO_SCORE_CENTER_WEIGHT', '0.0')
)
YOLO_SCORE_AREA_WEIGHT = float(
    os.environ.get('YOLO_SCORE_AREA_WEIGHT', '0.85')
)
YOLO_SCORE_REFERENCE_AREA_RATIO = float(
    # 0.08 saturated every mid-size box, so a 0.95 far speck beat a
    # 0.50 close gate. Keep area meaningful up to ~20% of the frame.
    os.environ.get('YOLO_SCORE_REFERENCE_AREA_RATIO', '0.20')
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
# Colour recovery when YOLO pose comes up empty. Measured on
# telem_20260813_025559: beyond 45 deg of roll the pose model delivered a gate
# on 51-61% of frames against 74% while level, and the flight logs show the
# policy going fully blind through exactly those moments. Li & de Croon
# (reference/paper0.pdf) raced a 15-gate track on colour alone, so colour is a
# reasonable second opinion even though its standalone true positive rate (0.46)
# is below this detector's.
GLOBAL_HSV_FALLBACK_ENABLED = _env_bool(
    'GLOBAL_HSV_FALLBACK_ENABLED', False
)
GLOBAL_HSV_FALLBACK_CONFIDENCE_SCALE = float(
    os.environ.get('GLOBAL_HSV_FALLBACK_CONFIDENCE_SCALE', '0.55')
)
# The fallback previously required that no target had ever been locked, so it
# only ever fired before acquisition and never mid-race.
GLOBAL_HSV_FALLBACK_DURING_LOCK = _env_bool(
    'GLOBAL_HSV_FALLBACK_DURING_LOCK', True
)

# How many consecutive missed frames a held ("predicted") detection may span
# before the policy is given nothing instead. At ~5 Hz vision this is about
# 0.6 s of coasting on the last seen gate; beyond that the geometry is stale
# enough that an all-sentinel observation is more honest.
GATE_HELD_MAX_FRAMES = int(os.environ.get('GATE_HELD_MAX_FRAMES', '3'))

# Snake gate detection (Li & de Croon, reference/paper0.pdf section 3.1) run
# observe-only next to the YOLO pose model, so its hit rate on this course can be
# compared before committing to it. Nothing in the control path reads it. Off by
# default because it costs a few ms per frame.
SNAKE_GATE_ENABLED = _env_bool('SNAKE_GATE_ENABLED', False)

# GateNet (gatenet_handoff): four inner-aperture corners. Observe-only by
# default so YOLO still feeds the policy and the panel can compare the two on
# the same frame. Set GATE_DETECTOR_BACKEND=gatenet to actually fly on it.
GATENET_ENABLED = _env_bool('GATENET_ENABLED', False)
GATENET_MODEL_PATH = os.environ.get(
    'GATENET_MODEL_PATH', 'gatenet_handoff/gatenet.onnx'
)
# Author's instruction: gate on the weakest per-corner peak, not the collapsed
# confidence head. Their README recommends 0.80 (48.6% recall); we fly at 0.45
# so more distant / partial gates survive.
GATENET_SCORE_THRESHOLD = float(
    os.environ.get('GATENET_SCORE_THRESHOLD', '0.45')
)
# Two visible inner corners still give a bearing; PnP needs all four.
GATENET_MIN_CORNERS = int(os.environ.get('GATENET_MIN_CORNERS', '2'))
# sigma_L. The paper's ROC sweep puts the knee at 25 px: under 15 the false
# positives climb, over 35 the true positive rate drops sharply.
SNAKE_MIN_LENGTH_PX = int(os.environ.get('SNAKE_MIN_LENGTH_PX', '25'))
# sigma_cf, the colour-fitness threshold from eq. 1.
SNAKE_MIN_COLOR_FITNESS = float(
    os.environ.get('SNAKE_MIN_COLOR_FITNESS', '0.35')
)
SNAKE_MAX_SAMPLES = int(os.environ.get('SNAKE_MAX_SAMPLES', '1500'))
# Square the snake points off with a rotated rectangle instead of the paper's
# axis-aligned one. Measured on synthetic gates, the paper's version scores
# colour fitness 1.00 / 0.34 / 0.19 / 0.09 at 0 / 5 / 10 / 20 deg of roll and so
# goes blind past about 5 deg; the rotated version holds 1.00 out to 60 deg.
# Set SNAKE_USE_ROTATED_RECT=0 to measure the published behaviour instead.
SNAKE_USE_ROTATED_RECT = _env_bool('SNAKE_USE_ROTATED_RECT', True)

# Initial crop-local segmentation uses the calibrated Q2 values. These remain
# environment-configurable without modifying detector code.
GATE_HSV_LOWER = _env_int_tuple('GATE_HSV_LOWER', (0, 75, 140))
GATE_HSV_UPPER = _env_int_tuple('GATE_HSV_UPPER', (23, 255, 255))
GATE_MIN_CONTOUR_AREA = float(
    os.environ.get('GATE_MIN_CONTOUR_AREA', '12')
)
