import math

# ---- Flight controller ----
HOVER_THRUST    = 0.27      # open-loop hover baseline (no baro in VQ2)
KP_THRUST       = 0.25      # vd cmd (m/s) → thrust delta
KP_LEAN         = 0.10      # body velocity (m/s) → desired lean angle (rad)
MAX_LEAN_RAD    = math.radians(25.0)
KP_ATT          = 3.0       # lean error (rad) → body rate command (rad/s)
RATE_SIGN_PITCH = -1.0      # sim pitch rate axis is inverted
RATE_SIGN_ROLL  =  1.0
MAX_THRUST      = 0.90
MIN_THRUST      = 0.05
CONTROL_HZ      = 100

# ---- Mode selection ----
USE_TELEOP      = True
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
