"""Localization check + PID tuning harness for the dual-gate PnP + IMU path.

Subcommands, smallest blast radius first:

  localize    Localization report. Default is read-only (never arms). Pass
              --teleop to arm and tip the craft from the keyboard so the
              gyro sign check gets real roll/pitch motion (Phase 1).

  hover       Arms and holds level at a fixed collective. This is the
              HOVER_THRUST trim test: if the drone climbs or sinks, the number
              is wrong and every downstream gain is fighting it. Keep a gate in
              view — the verdict then rests on the gate's image-space drift,
              which needs no integration.

  step        Arms, levels, then injects a desired-lean step and measures the
              inner attitude loop's rise / overshoot / settle. Tune
              KALMAN_KP_ATT and KALMAN_KD_ATT here.

  lean-hover  Hold a constant forward pitch with tilt-compensated thrust
              (Phase 4.5). Confirms altitude holds while leaned.

  crawl       Small image-space lean toward the visible gate (Phase 4.6).
              No punch / no takeoff boost — vision→lean wiring + lean-boost
              altitude trim.

  yaw-align   Hold level and yaw on image nx (Phase 4.7). Tune KALMAN_KP_YAW.

  authority   Step to KALMAN_MAX_LEAN_DEG (Phase 4.8). Confirms max lean /
              max rate are usable, not saturating uselessly.

  climb       Level takeoff-thrust pulse (Phase 2.5). Trims TAKEOFF_THRUST.

  acquire     Short armed run of the real Kalman planner (Phase 5.0). Pass if
              DUAL_PNP or yolo_fallback plus nonzero desired pitch arrives
              within ~2 s of arm.

  drive       Hard image chase (Phase 4.9). Always tip toward the gate when
              it is in view — no planner safety / hover. Pass if range closes
              or the craft moves forward in local NED. Prove "go to target"
              before Phase 5.

  manual      Auto-stabilize teleop: ANGLE self-level + hover thrust + the
              tuned attitude loop. You command lean / yaw / thrust; vision
              does not steer. Fly the course yourself.

  assist      Bounded live check of the image-chase planner (same as
              main.py FLIGHT_MODE=assist). Prefer `python main.py` for
              full races; use this for short gain tweaks.

Gains are passed as flags and exported to the environment *before* config is
imported, so the values under test are the ones the live planner would use:

  python tools/tune_flight.py localize --seconds 30
  python tools/tune_flight.py hover --hover-thrust 0.245 --seconds 12
  python tools/tune_flight.py step --axis pitch --amplitude-deg 8 \
      --kp-att 2.2 --kd-att 0.10
  python tools/tune_flight.py lean-hover --amplitude-deg 8 --seconds 14
  python tools/tune_flight.py crawl --seconds 12 --lean-deg 4
  python tools/tune_flight.py drive --seconds 16 --lean-deg 10
  python tools/tune_flight.py acquire --seconds 8
  python tools/tune_flight.py manual --seconds 0
  python tools/tune_flight.py assist --seconds 30
  python main.py   # default FLIGHT_MODE=assist

Every run appends a CSV under logs/tuning/ for offline comparison.

On the OLD (VQ1) sim, ATTITUDE / LOCAL_POSITION_NED / ODOMETRY are scoring
only — never fed into the control loop. VQ2 omits them; gains still transfer.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

SIM_SERVER_UDP_IP = '127.0.0.1'
SIM_SERVER_UDP_PORT = 14550

# Hard aborts for the arming subcommands. These are deliberately tight: a
# tuning run should stop long before it becomes a crash.
ABORT_ALTITUDE_M = 3.0
ABORT_LEAN_RAD = math.radians(35.0)

# A lean step slower than this leaves the drone still rolling into the gate
# when it arrives. Used only for tuning guidance, not as an abort.
SLOW_RISE_S = 0.60

FEEDBACK_HELP = (
    "which attitude signal the loop closes on. 'ekf' (default) is what races. "
    "'truth' uses the sim's ATTITUDE message and is a DIAGNOSTIC ONLY: it "
    "shows the controller's ceiling with perfect sensing, so the gap between "
    "the two runs is the cost of the estimator. Never ship gains tuned with "
    "--feedback truth -- VQ2 does not publish ATTITUDE."
)


# --------------------------------------------------------------------------
# argument parsing (must run before `import config`)
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='mode', required=True)

    def common(p, default_seconds):
        p.add_argument('--seconds', type=float, default=default_seconds,
                       help='run duration (0 = until Ctrl+C)')
        p.add_argument('--hz', type=float, default=20.0,
                       help='console/CSV sample rate')
        p.add_argument('--csv', default=None,
                       help='CSV path (default: logs/tuning/<mode>_<ts>.csv)')
        p.add_argument('--quiet', action='store_true',
                       help='CSV only, no console table')
        p.add_argument(
            '--no-sim-reset', action='store_true',
            help='skip MAVLink 31000 pad reset before arm (default: always reset)',
        )
        p.add_argument(
            '--early-start-hold-s', type=float, default=3.5,
            help='seconds to wait after reset/pad-ready before arm '
                 '(avoids OLD-sim early-start DQ; default 3.5)',
        )

    p_loc = sub.add_parser(
        'localize',
        help='localization report (add --teleop to tip via keyboard)',
    )
    common(p_loc, 30.0)
    p_loc.add_argument('--park-ekf', action='store_true',
                       help='leave the EKF in ZUPT (PnP only). By default '
                            'localize sets flight_started so the EKF actually '
                            'integrates — otherwise it reports a frozen zero '
                            'state.')
    p_loc.add_argument(
        '--teleop', action='store_true',
        help='arm and fly with the same hold-to-fly keys as `manual` '
             '(WASD lean, Q/E yaw, R/F thrust, Space=level, Esc/X=quit). '
             'Still writes the full localize / EKF-vs-truth report.',
    )
    p_loc.add_argument(
        '--teleop-lean-deg', type=float, default=14.0,
        help='lean amplitude while a key is held (default 14, same as manual)',
    )
    p_loc.add_argument(
        '--yaw-rate-deg', type=float, default=40.0,
        help='yaw rate while Q/E held (default 40, same as manual)',
    )
    p_loc.add_argument(
        '--thrust-step', type=float, default=0.022,
        help='collective offset while R/F held (default 0.022, same as manual)',
    )
    p_loc.add_argument('--lean-boost', type=float, default=None)
    p_loc.add_argument('--hover-thrust', type=float, default=None)
    p_loc.add_argument('--kp-att', type=float, default=None)
    p_loc.add_argument('--kd-att', type=float, default=None)
    p_loc.add_argument('--max-rate', type=float, default=None)

    p_hov = sub.add_parser('hover', help='HOVER_THRUST trim test (arms)')
    common(p_hov, 12.0)
    p_hov.add_argument('--hover-thrust', type=float, default=None)
    p_hov.add_argument('--kp-att', type=float, default=None)
    p_hov.add_argument('--kd-att', type=float, default=None)
    p_hov.add_argument('--feedback', choices=('ekf', 'truth'), default='ekf',
                       help=FEEDBACK_HELP)

    p_step = sub.add_parser('step', help='attitude step response (arms)')
    common(p_step, 14.0)
    p_step.add_argument('--axis', choices=('pitch', 'roll'), default='pitch')
    p_step.add_argument('--amplitude-deg', type=float, default=8.0)
    p_step.add_argument('--settle-s', type=float, default=4.0,
                        help='level-hold time before the step is injected')
    p_step.add_argument('--hover-thrust', type=float, default=None)
    p_step.add_argument('--kp-att', type=float, default=None)
    p_step.add_argument('--kd-att', type=float, default=None)
    p_step.add_argument('--max-rate', type=float, default=None)
    p_step.add_argument('--feedback', choices=('ekf', 'truth'), default='ekf',
                        help=FEEDBACK_HELP)

    p_lean = sub.add_parser(
        'lean-hover',
        help='hold a lean with tilt-compensated thrust (Phase 4.5)',
    )
    common(p_lean, 14.0)
    p_lean.add_argument(
        '--axis', choices=('roll', 'pitch'), default='roll',
        help='lean axis. Default roll — forward pitch leaves the start '
             'pad and the OLD sim early-start DQs the run (looks like a '
             'fake altitude crash). Use pitch only after a clean race start.',
    )
    p_lean.add_argument('--amplitude-deg', type=float, default=8.0,
                        help='lean amplitude to hold after settle')
    p_lean.add_argument('--settle-s', type=float, default=3.0,
                        help='level-hold time before leaning')
    p_lean.add_argument(
        '--lean-boost', type=float, default=0.0,
        help='extra collective while leaned (default 0 — measure pure '
             'HOVER/cos(tilt); flight uses config.LEAN_THRUST_BOOST=0.008)',
    )
    p_lean.add_argument('--hover-thrust', type=float, default=None)
    p_lean.add_argument('--kp-att', type=float, default=None)
    p_lean.add_argument('--kd-att', type=float, default=None)
    p_lean.add_argument('--max-rate', type=float, default=None)
    p_lean.add_argument('--feedback', choices=('ekf', 'truth'), default='ekf',
                        help=FEEDBACK_HELP)

    p_crawl = sub.add_parser(
        'crawl',
        help='gentle image-space crawl toward a visible gate (Phase 4.6)',
    )
    common(p_crawl, 12.0)
    p_crawl.add_argument('--lean-deg', type=float, default=4.0,
                         help='max forward / lateral lean while crawling')
    p_crawl.add_argument('--settle-s', type=float, default=2.0,
                         help='level-hold before enabling image lean')
    p_crawl.add_argument(
        '--lean-boost', type=float, default=0.002,
        help='extra collective while crawled (exports LEAN_THRUST_BOOST; '
             '0.002 holds ~±0.05 m/s on OLD sim crawl)',
    )
    p_crawl.add_argument('--hover-thrust', type=float, default=None)
    p_crawl.add_argument('--kp-att', type=float, default=None)
    p_crawl.add_argument('--kd-att', type=float, default=None)
    p_crawl.add_argument('--max-rate', type=float, default=None)
    p_crawl.add_argument('--feedback', choices=('ekf', 'truth'), default='ekf',
                         help=FEEDBACK_HELP)

    p_yaw = sub.add_parser(
        'yaw-align',
        help='level hover + image yaw align (Phase 4.7 / KALMAN_KP_YAW)',
    )
    common(p_yaw, 12.0)
    p_yaw.add_argument('--settle-s', type=float, default=2.0,
                       help='level-hold before enabling yaw-on-nx')
    p_yaw.add_argument('--kp-yaw', type=float, default=None,
                       help='override KALMAN_KP_YAW')
    p_yaw.add_argument('--hover-thrust', type=float, default=None)
    p_yaw.add_argument('--kp-att', type=float, default=None)
    p_yaw.add_argument('--kd-att', type=float, default=None)
    p_yaw.add_argument('--max-rate', type=float, default=None)
    p_yaw.add_argument('--feedback', choices=('ekf', 'truth'), default='ekf',
                       help=FEEDBACK_HELP)

    p_auth = sub.add_parser(
        'authority',
        help='step to KALMAN_MAX_LEAN_DEG (Phase 4.8)',
    )
    common(p_auth, 14.0)
    p_auth.add_argument('--axis', choices=('roll', 'pitch'), default='roll')
    p_auth.add_argument('--settle-s', type=float, default=3.0)
    p_auth.add_argument('--max-lean-deg', type=float, default=None,
                        help='override KALMAN_MAX_LEAN_DEG (step amplitude)')
    p_auth.add_argument('--hover-thrust', type=float, default=None)
    p_auth.add_argument('--kp-att', type=float, default=None)
    p_auth.add_argument('--kd-att', type=float, default=None)
    p_auth.add_argument('--max-rate', type=float, default=None)
    p_auth.add_argument(
        '--lean-boost', type=float, default=0.0,
        help='extra collective while leaned (default 0 for clean HT/cos)',
    )
    p_auth.add_argument('--feedback', choices=('ekf', 'truth'), default='ekf',
                        help=FEEDBACK_HELP)

    p_climb = sub.add_parser(
        'climb',
        help='level TAKEOFF_THRUST pulse (Phase 2.5)',
    )
    common(p_climb, 8.0)
    p_climb.add_argument('--takeoff-thrust', type=float, default=None,
                         help='override TAKEOFF_THRUST during the pulse')
    p_climb.add_argument('--pulse-s', type=float, default=2.5,
                         help='seconds of takeoff thrust before returning to HT')
    p_climb.add_argument('--hover-thrust', type=float, default=None)
    p_climb.add_argument('--kp-att', type=float, default=None)
    p_climb.add_argument('--kd-att', type=float, default=None)
    p_climb.add_argument('--feedback', choices=('ekf', 'truth'), default='ekf',
                         help=FEEDBACK_HELP)

    p_drive = sub.add_parser(
        'drive',
        help='hard image chase toward gate 1 (Phase 4.9)',
    )
    common(p_drive, 16.0)
    p_drive.add_argument('--lean-deg', type=float, default=10.0,
                         help='forward pitch when a gate is visible')
    p_drive.add_argument(
        '--settle-s', type=float, default=0.4,
        help='brief level hold before chase (keep short — long settle '
             'just climbs the gate out of frame)',
    )
    p_drive.add_argument(
        '--lean-boost', type=float, default=0.0,
        help='extra collective while leaned (default 0 — boost lofted '
             'the pad climb)',
    )
    p_drive.add_argument('--kp-yaw', type=float, default=None,
                         help='override KALMAN_KP_YAW for image yaw')
    p_drive.add_argument('--hover-thrust', type=float, default=None)
    p_drive.add_argument('--kp-att', type=float, default=None)
    p_drive.add_argument('--kd-att', type=float, default=None)
    p_drive.add_argument('--max-rate', type=float, default=None)
    p_drive.add_argument('--feedback', choices=('ekf', 'truth'), default='ekf',
                         help=FEEDBACK_HELP)

    p_acq = sub.add_parser(
        'acquire',
        help='short real-planner gate-acquire check (Phase 5.0)',
    )
    common(p_acq, 8.0)
    p_acq.add_argument('--deadline-s', type=float, default=2.0,
                       help='seconds after arm to get vision + pitch')
    p_acq.add_argument('--hover-thrust', type=float, default=None)
    p_acq.add_argument('--kp-att', type=float, default=None)
    p_acq.add_argument('--kd-att', type=float, default=None)

    p_man = sub.add_parser(
        'manual',
        help='auto-stabilize teleop (you fly; attitude loop + hover trim)',
    )
    common(p_man, 0.0)  # 0 = until Esc / Ctrl+C
    p_man.add_argument(
        '--lean-deg', type=float, default=14.0,
        help='roll/pitch lean amplitude while a key is held (default 14)',
    )
    p_man.add_argument(
        '--yaw-rate-deg', type=float, default=40.0,
        help='yaw rate while Q/E held (default 40 deg/s)',
    )
    p_man.add_argument(
        '--thrust-step', type=float, default=0.022,
        help='collective offset while R/F held (default 0.022; release=hover)',
    )
    p_man.add_argument(
        '--wait-pad', action='store_true',
        help='block until dual-gate PnP sees gate 1 before arm',
    )
    p_man.add_argument('--hover-thrust', type=float, default=None)
    p_man.add_argument('--kp-att', type=float, default=None)
    p_man.add_argument('--kd-att', type=float, default=None)
    p_man.add_argument('--max-rate', type=float, default=None)
    p_man.add_argument('--lean-boost', type=float, default=None)

    p_assist = sub.add_parser(
        'assist',
        help='bounded image-chase assist (same planner as main.py default)',
    )
    common(p_assist, 30.0)
    p_assist.add_argument('--hover-thrust', type=float, default=None)
    p_assist.add_argument('--kp-att', type=float, default=None)
    p_assist.add_argument('--kd-att', type=float, default=None)
    p_assist.add_argument('--kp-yaw', type=float, default=None)
    p_assist.add_argument('--lean-boost', type=float, default=None)
    return parser


def export_gain_overrides(args) -> dict:
    """Push --flags into the environment so `import config` picks them up."""
    mapping = {
        'hover_thrust': 'HOVER_THRUST',
        'kp_att': 'KALMAN_KP_ATT',
        'kd_att': 'KALMAN_KD_ATT',
        'max_rate': 'KALMAN_MAX_RATE_RAD_S',
        'kp_yaw': 'KALMAN_KP_YAW',
        'max_lean_deg': 'KALMAN_MAX_LEAN_DEG',
        'takeoff_thrust': 'TAKEOFF_THRUST',
        'lean_boost': 'LEAN_THRUST_BOOST',
    }
    applied = {}
    for attr, env in mapping.items():
        value = getattr(args, attr, None)
        if value is not None:
            os.environ[env] = repr(float(value))
            applied[env] = float(value)
    # Never let a tuning run auto-reset or bound itself via the race client.
    os.environ.setdefault('AUTO_RESET_ON_CRASH', '0')
    os.environ.setdefault('RUN_MAX_SECONDS', '0')
    # Takeoff boost would dominate hover/lean/crawl/yaw/authority trim.
    # `acquire` and `climb` keep the real takeoff path.
    if getattr(args, 'mode', None) in (
            'hover', 'step', 'lean-hover', 'crawl', 'drive',
            'yaw-align', 'authority', 'manual', 'assist',
    ) or (
        getattr(args, 'mode', None) == 'localize'
        and bool(getattr(args, 'teleop', False))
    ):
        os.environ['TAKEOFF_DURATION_S'] = '0'
        applied['TAKEOFF_DURATION_S'] = 0.0
    if getattr(args, 'mode', None) in ('acquire', 'manual', 'assist'):
        os.environ.setdefault('CRASH_USE_SIM_ODOMETRY', '0')
        applied.setdefault('CRASH_USE_SIM_ODOMETRY', 0.0)
    if getattr(args, 'mode', None) == 'assist':
        os.environ['FLIGHT_MODE'] = 'assist'
        applied['FLIGHT_MODE'] = 'assist'
    return applied


def _poll_teleop_lean(des_roll, des_pitch, lean_rad):
    """Non-blocking keyboard → desired lean. Windows console only (msvcrt).

    Returns (des_roll, des_pitch, quit). Keys: A/D or arrows = roll,
    W/S or arrows = pitch (W = forward), Space = level, Esc/Q = quit.
    Tap a key to hold that lean until Space or an opposing key.
    """
    import config as _cfg
    fwd = float(getattr(_cfg, 'FORWARD_PITCH_SIGN', 1.0))
    quit_req = False
    try:
        import msvcrt
    except ImportError:
        return des_roll, des_pitch, False

    while msvcrt.kbhit():
        ch = msvcrt.getch()
        if ch in (b'\x00', b'\xe0'):
            # Arrow / function prefix
            if not msvcrt.kbhit():
                break
            code = msvcrt.getch()
            if code == b'K':      # left
                des_roll = -lean_rad
            elif code == b'M':    # right
                des_roll = lean_rad
            elif code == b'H':    # up arrow = forward
                des_pitch = fwd * lean_rad
            elif code == b'P':    # down arrow = reverse
                des_pitch = -fwd * lean_rad
            continue
        try:
            key = ch.decode('ascii', errors='ignore').lower()
        except Exception:
            continue
        if key == 'a':
            des_roll = -lean_rad
        elif key == 'd':
            des_roll = lean_rad
        elif key == 'w':
            des_pitch = fwd * lean_rad
        elif key == 's':
            des_pitch = -fwd * lean_rad
        elif key in (' ',):
            des_roll = des_pitch = 0.0
        elif key in ('q', '\x1b'):
            quit_req = True
    return des_roll, des_pitch, quit_req


# msvcrt has no key-up events — while a key is held the console autorepeats.
# Treat an axis as released when it has not been refreshed for this long.
_MANUAL_HOLD_RELEASE_S = 0.18


def _poll_manual_controls(
    hold_state: dict,
    *,
    lean_rad,
    yaw_rate_cmd,
    thrust_step,
    now: float,
):
    """Hold-to-fly keyboard (Windows msvcrt). Release → level / hover.

    ``hold_state`` is mutated across ticks (stores last-seen times + values).
    Returns (des_roll, des_pitch, yaw_rate, thrust_delta, quit).

      W/S or ↑/↓   pitch while held
      A/D or ←/→   roll while held
      Q/E          yaw while held
      R/F          climb / sink thrust while held (back to hover on release)
      Space        force level now
      Esc / X      quit
    """
    import config as _cfg
    fwd = float(getattr(_cfg, 'FORWARD_PITCH_SIGN', 1.0))
    yaw_sign = float(getattr(_cfg, 'RATE_SIGN_YAW', 1.0))
    quit_req = False

    def _press(axis: str, value: float) -> None:
        hold_state[axis] = value
        hold_state[f'{axis}_t'] = now

    try:
        import msvcrt
    except ImportError:
        return 0.0, 0.0, 0.0, 0.0, False

    while msvcrt.kbhit():
        ch = msvcrt.getch()
        if ch in (b'\x00', b'\xe0'):
            if not msvcrt.kbhit():
                break
            code = msvcrt.getch()
            if code == b'K':
                _press('roll', -lean_rad)
            elif code == b'M':
                _press('roll', lean_rad)
            elif code == b'H':
                _press('pitch', fwd * lean_rad)
            elif code == b'P':
                _press('pitch', -fwd * lean_rad)
            continue
        try:
            key = ch.decode('ascii', errors='ignore').lower()
        except Exception:
            continue
        if key == 'a':
            _press('roll', -lean_rad)
        elif key == 'd':
            _press('roll', lean_rad)
        elif key == 'w':
            _press('pitch', fwd * lean_rad)
        elif key == 's':
            _press('pitch', -fwd * lean_rad)
        elif key == 'q':
            _press('yaw', -yaw_sign * yaw_rate_cmd)
        elif key == 'e':
            _press('yaw', yaw_sign * yaw_rate_cmd)
        elif key == 'r':
            _press('thrust', thrust_step)
        elif key == 'f':
            _press('thrust', -thrust_step)
        elif key == ' ':
            for axis in ('roll', 'pitch', 'yaw', 'thrust'):
                hold_state[axis] = 0.0
                hold_state[f'{axis}_t'] = 0.0
        elif key in ('x', '\x1b'):
            quit_req = True

    def _active(axis: str) -> float:
        t = float(hold_state.get(f'{axis}_t') or 0.0)
        if t <= 0.0 or (now - t) > _MANUAL_HOLD_RELEASE_S:
            return 0.0
        return float(hold_state.get(axis) or 0.0)

    return (
        _active('roll'),
        _active('pitch'),
        _active('yaw'),
        _active('thrust'),
        quit_req,
    )


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _f(value, default=None):
    """Coerce to float, returning `default` for None/garbage."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _fmt(value, spec='7.2f'):
    return format(value, spec) if isinstance(value, float) else '     --'


class Recorder:
    """Append-only CSV writer that takes its header from the first row."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = path.open('w', newline='', encoding='utf-8')
        self._writer = None
        self.rows = []

    def write(self, row: dict) -> None:
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)
        self.rows.append(row)

    def close(self) -> None:
        self._file.close()


def default_csv_path(mode: str) -> Path:
    stamp = time.strftime('%Y%m%d_%H%M%S')
    return REPOSITORY_ROOT / 'logs' / 'tuning' / f'{mode}_{stamp}.csv'


def truth_state(shared_data):
    """Sim-provided ground truth, or None on the VQ2 build that omits it.

    VQ1 publishes ATTITUDE, LOCAL_POSITION_NED and ODOMETRY; VQ2 publishes
    none of them. ODOMETRY is the richest source — pose quaternion plus body
    rates — so it fills in truth velocity and truth angular rates that
    LOCAL_POSITION_NED cannot provide.

    SCORING ONLY. Never feed this into a control loop: VQ2 does not publish
    it, so any gain tuned against it would not transfer.
    """
    pos = shared_data.get('local_position_ned') or {}
    odo = shared_data.get('odometry') or {}
    att = shared_data.get('attitude_raw') or {}
    if not (pos or odo or att):
        return None

    def pick(key, *sources):
        for src in sources:
            value = _f(src.get(key))
            if value is not None:
                return value
        return None

    state = {
        # position: LOCAL_POSITION_NED first, ODOMETRY as backup
        'x': pick('x', pos, odo),
        'y': pick('y', pos, odo),
        'z': pick('z', pos, odo),
        # velocity: either source carries it
        'vx': pick('vx', pos, odo),
        'vy': pick('vy', pos, odo),
        'vz': pick('vz', pos, odo),
        # attitude: ATTITUDE first, else derived from the ODOMETRY quaternion
        'roll': pick('roll', att, odo),
        'pitch': pick('pitch', att, odo),
        'yaw': pick('yaw', att, odo),
        # body rates: the only external check on gyro sign conventions
        'rollspeed': pick('rollspeed', att, odo),
        'pitchspeed': pick('pitchspeed', att, odo),
        'yawspeed': pick('yawspeed', att, odo),
    }
    if all(v is None for v in state.values()):
        return None
    return state


def gyro_sign_report(gyro_pairs, *, motion_threshold=0.05, min_samples=10):
    """Compare truth body rates against the raw gyro the AHRS integrates.

    `gyro_pairs` are (truth_rollspeed, xgyro, truth_pitchspeed, ygyro) tuples.
    Returns {axis: (agree_fraction, verdict, n)} or {} if there was too little
    motion to judge. An INVERTED axis is an estimator bug (flip the matching
    gyro_sign_* in ahrs.py), not something gains can compensate for.
    """
    moving = [g for g in gyro_pairs
              if abs(g[0]) > motion_threshold or abs(g[2]) > motion_threshold]
    if len(moving) < min_samples:
        return {}
    out = {}
    for axis, truth_i, raw_i in (('roll/xgyro', 0, 1), ('pitch/ygyro', 2, 3)):
        rows = [g for g in moving if abs(g[truth_i]) > motion_threshold]
        if not rows:
            continue
        agree = sum(1 for g in rows if g[truth_i] * g[raw_i] > 0)
        frac = agree / len(rows)
        verdict = ('same sign' if frac > 0.8 else
                   'INVERTED' if frac < 0.2 else 'inconsistent')
        out[axis] = (frac, verdict, len(rows))
    return out


def blind_drift(rows):
    """EKF position travel during stretches with no PnP correction.

    With no odometry and no sim attitude, this is the only direct read on how
    fast the IMU-only solution runs away between gate fixes.
    """
    worst = 0.0
    run_start = None
    for row in rows:
        solved = (row.get('n_solved') or 0) >= 1
        pos = (row.get('ekf_x'), row.get('ekf_y'), row.get('ekf_z'))
        if not all(isinstance(v, float) for v in pos):
            continue
        if solved:
            run_start = None
            continue
        if run_start is None:
            run_start = pos
        else:
            worst = max(worst, math.dist(pos, run_start))
    return worst


# --------------------------------------------------------------------------
# localize
# --------------------------------------------------------------------------
def run_localize(args) -> int:
    import config
    from control.pid import PIDConfig, PIDController
    from setup import setup_components

    teleop = bool(getattr(args, 'teleop', False))
    shared_data = {}
    components = setup_components(
        shared_data, int(time.time() * 1000),
        SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT,
    )
    controller = components['controller']
    recorder = Recorder(Path(args.csv) if args.csv
                        else default_csv_path('localize'))

    des_roll = 0.0
    des_pitch = 0.0
    yaw_rate = 0.0
    thrust_delta = 0.0
    hold_state: dict = {}
    roll_pid = pitch_pid = None
    lean_rad = math.radians(float(getattr(args, 'teleop_lean_deg', 14.0)))
    yaw_rate_cmd = math.radians(float(getattr(args, 'yaw_rate_deg', 40.0)))
    thrust_step = float(getattr(args, 'thrust_step', 0.022))
    last_t = None

    if teleop:
        print('[TELEOP] localize will ARM — same hold-to-fly keys as manual.',
              flush=True)
        print('         Focus THIS console window for keys:', flush=True)
        print('           W/S or ↑/↓   pitch    A/D or ←/→   roll',
              flush=True)
        print('           Q/E          yaw      R/F          climb/sink',
              flush=True)
        print('           Space        level    Esc / X      quit',
              flush=True)
        print(
            f'         lean={math.degrees(lean_rad):.0f}°  '
            f'yaw={math.degrees(yaw_rate_cmd):.0f}°/s  '
            f'thrust_step={thrust_step:.3f}  '
            f'(release = level / hover)',
            flush=True,
        )
        max_rate = config.KALMAN_MAX_RATE_RAD_S
        roll_pid = PIDController(PIDConfig(
            kp=config.KALMAN_KP_ATT, kd=config.KALMAN_KD_ATT,
            output_min=-max_rate, output_max=max_rate,
        ))
        pitch_pid = PIDController(PIDConfig(
            kp=config.KALMAN_KP_ATT, kd=config.KALMAN_KD_ATT,
            output_min=-max_rate, output_max=max_rate,
        ))
        if bool(getattr(args, 'no_sim_reset', False)):
            print('[SIM] skip reset (--no-sim-reset)', flush=True)
            time.sleep(1.0)
        else:
            print('[SIM] reset (command 31000) before arm...', flush=True)
            controller.send_sim_reset()
            time.sleep(max(0.5, float(getattr(config, 'SIM_RESET_SETTLE_S', 1.5))))
        hold_s = max(0.0, float(getattr(args, 'early_start_hold_s', 3.5) or 3.5))
        print(f'[SIM] early-start hold {hold_s:.1f}s before arm...', flush=True)
        time.sleep(hold_s)
        print('Arming...', flush=True)
        controller.arm()
        shared_data['flight_started'] = True
    else:
        print('[SAFE] localize: never arms, never sends a flight command',
              flush=True)
        # The EKF only integrates while 'flight_started' is set — otherwise it
        # is held in ZUPT and would report a frozen zero state.
        if args.park_ekf:
            print('       EKF parked (--park-ekf): PnP only, EKF will read '
                  'zeros', flush=True)
        else:
            shared_data['flight_started'] = True
            print('       EKF integrating. On a stationary pad the drift you '
                  'see IS the IMU-only error;', flush=True)
            print('       use --teleop to tip the craft from the keyboard for '
                  'the gyro sign check.', flush=True)

    if not args.quiet:
        print('\n    t   n  g1_rng  g2_rng  g2f    ekf_x   ekf_y   ekf_z'
              '   |v|   norm_x  norm_y   roll   pitch   det_hz', flush=True)

    period = 1.0 / max(args.hz, 1.0)
    started = time.monotonic()
    solved_frames = 0
    total_frames = 0
    last_frame_id = None
    frame_count = 0
    frame_window_t = started
    gate2_seen = 0
    gate2_retained = 0
    truth_available = False
    pos_err_samples = []
    alt_err_samples = []
    att_err_samples = []
    vel_err_samples = []
    gyro_pairs = []      # (truth_rollspeed, raw_xgyro) for sign validation

    try:
        while True:
            now = time.monotonic()
            elapsed = now - started
            if args.seconds > 0 and elapsed >= args.seconds:
                break

            if teleop:
                (
                    des_roll, des_pitch, yaw_rate, thrust_delta, quit_req
                ) = _poll_manual_controls(
                    hold_state,
                    lean_rad=lean_rad,
                    yaw_rate_cmd=yaw_rate_cmd,
                    thrust_step=thrust_step,
                    now=now,
                )
                if quit_req:
                    print('\n[STOP] teleop quit', flush=True)
                    break
                dt = period if last_t is None else max(1e-3, now - last_t)
                last_t = now
                roll, pitch, _, _ = read_attitude(shared_data)
                roll_rate = roll_pid.update(des_roll - roll, dt)
                pitch_rate = pitch_pid.update(des_pitch - pitch, dt)
                lean_boost = float(
                    getattr(config, 'LEAN_THRUST_BOOST', 0.0) or 0.0
                )
                thrust = _tilt_compensated_thrust(
                    config.HOVER_THRUST, des_roll, des_pitch,
                    lean_boost=lean_boost,
                )
                thrust = float(max(0.18, min(0.36, thrust + thrust_delta)))
                shared_data['planner_target'] = {
                    'kalman': True,
                    'roll_rate': roll_rate,
                    'pitch_rate': pitch_rate,
                    'yaw_rate': yaw_rate,
                    'thrust': thrust,
                    'desired_roll': des_roll,
                    'desired_pitch': des_pitch,
                }
                shared_data['planner_mode'] = 'kalman_dual_gate'
                controller.update()

            dual = shared_data.get('dual_gate_pnp') or {}
            ekf = shared_data.get('ekf_state') or {}
            nav = shared_data.get('navigation') or {}
            att = shared_data.get('attitude') or {}
            pos = ekf.get('position_ned') or [None, None, None]
            vel = ekf.get('velocity_ned') or [None, None, None]

            n_solved = int(dual.get('n_solved') or 0)
            frame_id = nav.get('frame_id')
            if frame_id is not None and frame_id != last_frame_id:
                last_frame_id = frame_id
                frame_count += 1
                total_frames += 1
                if n_solved >= 1:
                    solved_frames += 1
                if n_solved >= 2:
                    gate2_seen += 1
                elif ekf.get('gate2_ned') is not None:
                    gate2_retained += 1
            det_hz = frame_count / max(now - frame_window_t, 1e-6)
            if now - frame_window_t >= 2.0:
                frame_window_t, frame_count = now, 0

            ekf_x, ekf_y, ekf_z = (_f(v) for v in pos)
            speed = None
            if all(_f(v) is not None for v in vel):
                speed = math.sqrt(sum(_f(v) ** 2 for v in vel))

            truth = truth_state(shared_data)
            if truth is not None:
                truth_available = True
            err_xy = err_z = err_roll = err_pitch = None
            if truth is not None:
                if truth['z'] is not None and ekf_x is not None:
                    err_xy = math.hypot(ekf_x - (truth['x'] or 0.0),
                                        ekf_y - (truth['y'] or 0.0))
                    err_z = abs(ekf_z - truth['z'])
                    pos_err_samples.append(err_xy)
                    alt_err_samples.append(err_z)
                if truth['roll'] is not None and _f(att.get('roll')) is not None:
                    err_roll = _f(att.get('roll')) - truth['roll']
                    err_pitch = _f(att.get('pitch')) - truth['pitch']
                    att_err_samples.append(max(abs(err_roll), abs(err_pitch)))
                if truth['vx'] is not None and speed is not None:
                    vel_err_samples.append(math.dist(
                        [_f(v, 0.0) for v in vel],
                        [truth['vx'], truth['vy'] or 0.0, truth['vz'] or 0.0],
                    ))
                imu = shared_data.get('highres_imu') or {}
                if truth['rollspeed'] is not None:
                    gyro_pairs.append((
                        truth['rollspeed'], _f(imu.get('xgyro'), 0.0),
                        truth['pitchspeed'], _f(imu.get('ygyro'), 0.0),
                    ))

            row = {
                't': round(elapsed, 3),
                'n_solved': n_solved,
                'gate1_range_m': _f(dual.get('gate1_range_m')),
                'gate2_range_m': _f(dual.get('gate2_range_m')),
                'gate1_norm_x': _f(dual.get('gate1_norm_x')),
                'gate1_norm_y': _f(dual.get('gate1_norm_y')),
                'gate2_fresh': ekf.get('gate2_fresh'),
                'gate2_retained': ekf.get('gate2_ned') is not None,
                'ekf_x': ekf_x, 'ekf_y': ekf_y, 'ekf_z': ekf_z,
                'ekf_speed': speed,
                'roll': _f(att.get('roll')),
                'pitch': _f(att.get('pitch')),
                'yaw': _f(att.get('yaw')),
                'des_roll': des_roll if teleop else None,
                'des_pitch': des_pitch if teleop else None,
                'truth_x': truth['x'] if truth else None,
                'truth_y': truth['y'] if truth else None,
                'truth_z': truth['z'] if truth else None,
                'truth_roll': truth['roll'] if truth else None,
                'truth_pitch': truth['pitch'] if truth else None,
                'err_xy_m': err_xy, 'err_z_m': err_z,
                'err_roll_rad': err_roll, 'err_pitch_rad': err_pitch,
                'vision_state': nav.get('state'),
                'det_hz': round(det_hz, 2),
            }
            recorder.write(row)

            if not args.quiet:
                print(
                    f"{elapsed:5.1f} {n_solved:3d}"
                    f" {_fmt(row['gate1_range_m'])}"
                    f" {_fmt(row['gate2_range_m'])}"
                    f"  {str(ekf.get('gate2_fresh'))[:1]:>3}"
                    f" {_fmt(ekf_x)} {_fmt(ekf_y)} {_fmt(ekf_z)}"
                    f" {_fmt(speed, '5.2f')}"
                    f"  {_fmt(row['gate1_norm_x'], '6.2f')}"
                    f" {_fmt(row['gate1_norm_y'], '6.2f')}"
                    f" {_fmt(row['roll'], '6.3f')}"
                    f" {_fmt(row['pitch'], '6.3f')}"
                    f"   {det_hz:5.1f}",
                    flush=True,
                )
            time.sleep(period)
    except KeyboardInterrupt:
        print('\n[STOP] interrupted', flush=True)
    finally:
        if teleop:
            try:
                controller.disarm()
                print('Disarmed.', flush=True)
            except Exception:
                pass
        recorder.close()
        shutdown(components)

    # ---- report ----
    rows = recorder.rows
    print('\n=== localization report ===')
    print(f'samples            {len(rows)}')
    print(f'camera frames      {total_frames}')
    solve_rate = solved_frames / total_frames if total_frames else 0.0
    print(f'PnP solve rate     {solve_rate * 100:.1f}%  '
          f'({solved_frames}/{total_frames} frames with >=1 gate solved)')
    print(f'two-gate frames    {gate2_seen}   '
          f'gate2 retained after leaving FOV: {gate2_retained} frames')

    ranges = [r['gate1_range_m'] for r in rows
              if isinstance(r['gate1_range_m'], float)]
    if len(ranges) > 2:
        jumps = [abs(b - a) for a, b in zip(ranges, ranges[1:])]
        print(f'gate1 range        {min(ranges):.2f} - {max(ranges):.2f} m   '
              f'median step {statistics.median(jumps):.3f} m  '
              f'max step {max(jumps):.2f} m')
        print('                   (large max step = PnP flipping between '
              'gates or a bad corner solve)')
    if not args.park_ekf:
        print(f'EKF blind drift    {blind_drift(rows):.2f} m worst-case '
              'position travel across a no-PnP stretch')

    if truth_available:
        print('\n--- ground truth available (sim publishes ATTITUDE / '
              'LOCAL_POSITION_NED) ---')
        if pos_err_samples:
            print(f'EKF horizontal err mean {statistics.fmean(pos_err_samples):.2f} m'
                  f'   max {max(pos_err_samples):.2f} m')
            print(f'EKF vertical   err mean {statistics.fmean(alt_err_samples):.2f} m'
                  f'   max {max(alt_err_samples):.2f} m')
        if vel_err_samples:
            print(f'EKF velocity   err mean {statistics.fmean(vel_err_samples):.2f} m/s'
                  f' max {max(vel_err_samples):.2f} m/s')
            print('                   (velocity feeds every kd term — a bad '
                  'velocity estimate mistunes damping)')
        if att_err_samples:
            worst = math.degrees(max(att_err_samples))
            mean = math.degrees(statistics.fmean(att_err_samples))
            print(f'EKF attitude   err mean {mean:.2f} deg  max {worst:.2f} deg')
            if worst > 15.0:
                print('                   [WARN] attitude error >15 deg points '
                      'at a gyro sign or axis-mapping bug, not tuning — check '
                      'RATE_SIGN_* and the AHRS gyro_sign_* fields')
        signs = gyro_sign_report(gyro_pairs)
        if signs:
            print('\ngyro sign check (truth body rate vs raw HIGHRES_IMU gyro, '
                  'moving samples only)')
            for axis, (frac, verdict, n) in signs.items():
                print(f'  {axis:12s} {frac * 100:5.1f}% agree over {n:5d} '
                      f'samples  -> {verdict}')
            if any(v == 'INVERTED' for _, v, _ in signs.values()):
                print('  [!] An inverted axis means the AHRS integrates the '
                      'wrong way. Flip the matching gyro_sign_* in ahrs.py '
                      'AHRSConfig. This is an estimator bug, not a gain.')
        elif gyro_pairs:
            print('\ngyro sign check: not enough motion to judge — rotate the '
                  'drone during the run')

        print('\nScoring only: none of this reaches the control loop, and the '
              'race build does not publish it.')
    else:
        print('ground truth       not published by this build — EKF error can '
              'only be inferred from blind drift and PnP continuity')

    if total_frames == 0:
        print('\n[WARN] no camera frames arrived — is the sim streaming on '
              'UDP 5600 with a race running?')
    elif solve_rate < 0.5:
        print('\n[WARN] fewer than half the frames produced a PnP solve. '
              'Localization will be jumpy and the EKF will lean on IMU '
              'dead reckoning. Check the gate is in view and that '
              'models/gate_pose.pt is the trained pose model.')
    print(f'\nCSV: {recorder.path}')
    return 0


# --------------------------------------------------------------------------
# shared flight scaffolding for hover / step
# --------------------------------------------------------------------------
def shutdown(components) -> None:
    logger = components.get('logger')
    if logger is not None:
        logger.stop()
    for name in ('ts_loop', 'mavlink_rx', 'vision_rx', 'state_estimator'):
        component = components.get(name)
        joiner = getattr(component, 'get_thread_for_join', None)
        if joiner is not None:
            joiner().join(timeout=1.0)


def read_attitude(shared_data):
    """Roll/pitch from the EKF; body rates from raw gyro.

    The EKF publishes rollspeed/pitchspeed/yawspeed as literal 0.0, so rates
    must come from HIGHRES_IMU directly — the same source controller.py uses.
    """
    att = shared_data.get('attitude') or {}
    imu = shared_data.get('highres_imu') or {}
    return (
        _f(att.get('roll'), 0.0), _f(att.get('pitch'), 0.0),
        _f(imu.get('xgyro'), 0.0), _f(imu.get('ygyro'), 0.0),
    )


def vertical_observables(shared_data, z0, norm_y0):
    """Vertical motion estimates, best first.

    This sim publishes neither ATTITUDE nor LOCAL_POSITION_NED, so there is no
    odometry to trust. Two options remain:

    ``d_norm_y``  image-space vertical offset of gate 1. A *direct* per-frame
                  measurement with no integration, and exactly the signal the
                  planner's thrust law uses. Needs a gate in view. Positive =
                  gate has moved down the frame = drone rose.
    ``d_ekf_z``   EKF altitude change. Between PnP fixes this is pure IMU
                  dead reckoning and will drift, so it is a cross-check only.
    """
    dual = shared_data.get('dual_gate_pnp') or {}
    norm_y = _f(dual.get('gate1_norm_y'))
    d_norm_y = (
        norm_y - norm_y0
        if norm_y is not None and norm_y0 is not None else None
    )
    z = _f((shared_data.get('position_ned') or {}).get('z'))
    d_ekf_z = z0 - z if (z is not None and z0 is not None) else None
    return d_norm_y, d_ekf_z, _f(dual.get('gate1_range_m'))


def _tilt_compensated_thrust(hover_thrust, des_roll, des_pitch, *, lean_boost=None):
    """Match kalman_planner: keep vertical lift while pitched/rolled.

    Phase 4.5 lean-hover defaults lean_boost=0 so HT / cos(tilt) is measured
    cleanly. Flight still uses config.LEAN_THRUST_BOOST via kalman_planner.
    """
    import config
    tilt = max(
        0.88,
        math.cos(abs(float(des_pitch))) * math.cos(abs(float(des_roll))),
    )
    thrust = float(hover_thrust) / tilt
    if lean_boost is None:
        lean_boost = float(getattr(config, 'LEAN_THRUST_BOOST', 0.008))
    lean = max(abs(float(des_pitch)), abs(float(des_roll)))
    # Apply as soon as a commanded lean is present. Crawl uses ~1.4° pitch
    # (ny-halved); the old 3°→8° ramp left that path unboosted and sinking.
    if lean_boost and lean > math.radians(0.5):
        thrust += float(lean_boost)
    return thrust


def _image_gate_norm(shared_data):
    """Best available image-normalized gate centre (nx, ny, source) or Nones.

    Prefer the YOLO identity-locked box over dual_pnp.gate1. Dual-PnP
    re-sorts nearest-solved each frame, so a brief gate-1 PnP miss can
    hand the chase to gate 2 before GATE_PASSED (drive_m: 8 m → 27 m).
    YOLO keeps the locked instance until a real pass resets it.
    """
    det = shared_data.get('gate_detection') or {}
    center = det.get('center_px') if isinstance(det, dict) else None
    if center is not None and len(center) >= 2:
        # Match vision_rx / kalman_planner normalization (640x360).
        width, height = 640.0, 360.0
        cx, cy = float(center[0]), float(center[1])
        return (
            (cx - width * 0.5) / (width * 0.5),
            (cy - height * 0.5) / (height * 0.5),
            str(det.get('method') or 'yolo'),
        )
    dual = shared_data.get('dual_gate_pnp') or {}
    nx = _f(dual.get('gate1_norm_x'))
    ny = _f(dual.get('gate1_norm_y'))
    if nx is not None and ny is not None and int(dual.get('n_solved') or 0) >= 1:
        return nx, ny, 'dual_pnp'
    return None, None, None


def run_attitude_loop(args, desired_at, label, *, tilt_compensate=False):
    """Arm, close the inner attitude loop toward desired_at(t), log, disarm.

    `desired_at(elapsed, shared_data)` returns
    (desired_roll_rad, desired_pitch_rad). This mirrors kalman_planner's inner
    loop exactly — same PIDController, same gains, same rate clamp — so
    numbers found here transfer to real flight.
    """
    import config
    from control.pid import PIDConfig, PIDController
    from setup import setup_components

    shared_data = {}
    components = setup_components(
        shared_data, int(time.time() * 1000),
        SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT,
    )
    controller = components['controller']
    recorder = Recorder(Path(args.csv) if args.csv
                        else default_csv_path(label))

    max_rate = config.KALMAN_MAX_RATE_RAD_S
    make_pid = lambda: PIDController(PIDConfig(
        kp=config.KALMAN_KP_ATT, kd=config.KALMAN_KD_ATT,
        output_min=-max_rate, output_max=max_rate,
    ))
    roll_pid, pitch_pid = make_pid(), make_pid()

    print(f'[GAINS] KALMAN_KP_ATT={config.KALMAN_KP_ATT} '
          f'KALMAN_KD_ATT={config.KALMAN_KD_ATT} '
          f'max_rate={max_rate} HOVER_THRUST={config.HOVER_THRUST}',
          flush=True)
    lean_boost = float(getattr(args, 'lean_boost', 0.0) or 0.0)
    if tilt_compensate:
        print(f'[THRUST] tilt-compensated (HOVER / cos(tilt) + '
              f'lean_boost={lean_boost:.4f})', flush=True)
    print(f'[ABORT] |climb| > {ABORT_ALTITUDE_M} m or |lean| > '
          f'{math.degrees(ABORT_LEAN_RAD):.0f} deg disarms immediately',
          flush=True)
    # Always pad-reset before arm so each tune starts from a clean spawn
    # (unless --no-sim-reset). Still needs an active race for heartbeat/video.
    if bool(getattr(args, 'no_sim_reset', False)):
        print('[SIM] skip reset (--no-sim-reset)', flush=True)
        time.sleep(1.0)
    else:
        print('[SIM] reset (command 31000) before arm...', flush=True)
        controller.send_sim_reset()
        time.sleep(max(0.5, float(getattr(config, 'SIM_RESET_SETTLE_S', 1.5))))
    # Refuse to arm mid-course / underground: need a gate in view first.
    if getattr(args, 'mode', None) in (
            'lean-hover', 'crawl', 'drive', 'yaw-align', 'authority', 'climb'):
        ready_deadline = time.monotonic() + 45.0
        print('[PAD] waiting for gate in view (DUAL_PNP / YOLO) before arm — '
              'reset onto the start pad if this hangs...', flush=True)
        while time.monotonic() < ready_deadline:
            with shared_data['lock']:
                dual = shared_data.get('dual_gate_pnp') or {}
                det = shared_data.get('gate_detection') or {}
                vstate = str(shared_data.get('vision_state') or '')
            n_solved = int(dual.get('n_solved') or 0)
            has_det = bool(
                isinstance(det, dict)
                and det.get('center_px') is not None
            )
            if n_solved >= 1 or has_det or vstate == 'DUAL_PNP':
                print(f'[PAD] ready (vision_state={vstate or "det"} '
                      f'n_solved={n_solved})', flush=True)
                break
            time.sleep(0.2)
        else:
            print('[FAIL] no gate in view after 45s — put craft on the pad '
                  'facing gate 1, then re-run', flush=True)
            shutdown(components)
            return None, 'no gate in view (not on pad)', 0.0
    # Arming / thrusting immediately after reset tips off the pad → early-start
    # DQ on the OLD sim. Hold still once the race/spawn is live.
    hold_s = max(0.0, float(getattr(args, 'early_start_hold_s', 3.5) or 3.5))
    print(f'[SIM] early-start hold {hold_s:.1f}s before arm...', flush=True)
    time.sleep(hold_s)
    print('Arming...', flush=True)
    use_truth_feedback = getattr(args, 'feedback', 'ekf') == 'truth'
    if use_truth_feedback:
        if truth_state(shared_data) is None:
            print('[FAIL] --feedback truth needs the sim to publish ATTITUDE, '
                  'and none has arrived. This is the VQ2 build.', flush=True)
            shutdown(components)
            return None, 'no truth attitude', 0.0
        print('[LOOP]  *** DIAGNOSTIC: closing on sim ATTITUDE (perfect '
              'sensing) ***', flush=True)
        print('        This measures the controller ceiling only. VQ2 has no '
              'ATTITUDE, so do NOT ship gains found this way — re-run with '
              '--feedback ekf and compare.', flush=True)
    else:
        print('[LOOP]  closing on the EKF attitude estimate (what races). Any '
              'ground truth is recorded alongside, never fed back.', flush=True)
    controller.arm()
    shared_data['flight_started'] = True

    period = 1.0 / max(args.hz, 1.0)
    started = time.monotonic()
    last_t = started
    z0 = None
    norm_y0 = None
    truth_z0 = None
    abort = None
    solved_samples = 0
    if not args.quiet:
        print('\n    t   des_roll  roll   des_pitch  pitch   r_rate  p_rate'
              '  thrust  d_ny   d_ekfz  g1_rng', flush=True)
    try:
        while True:
            now = time.monotonic()
            elapsed = now - started
            dt = max(now - last_t, 1e-3)
            last_t = now
            if args.seconds > 0 and elapsed >= args.seconds:
                break

            roll, pitch, rollspeed, pitchspeed = read_attitude(shared_data)
            dual = shared_data.get('dual_gate_pnp') or {}
            n_solved = int(dual.get('n_solved') or 0)
            if n_solved >= 1:
                solved_samples += 1
            if z0 is None:
                z0 = _f((shared_data.get('position_ned') or {}).get('z'))
            if norm_y0 is None:
                norm_y0 = _f(dual.get('gate1_norm_y'))
            d_norm_y, d_ekf_z, g1_range = vertical_observables(
                shared_data, z0, norm_y0
            )
            truth = truth_state(shared_data)
            if truth is not None and truth_z0 is None:
                truth_z0 = truth['z']
            d_truth_z = (
                truth_z0 - truth['z']
                if truth is not None and truth['z'] is not None
                and truth_z0 is not None else None
            )
            local = shared_data.get('local_position_ned') or {}
            local_x = _f(local.get('x'))
            local_y = _f(local.get('y'))
            local_z = _f(local.get('z'))
            det = shared_data.get('gate_detection') or {}
            area_px = _f(det.get('area_px'))

            if max(abs(roll), abs(pitch)) > ABORT_LEAN_RAD:
                abort = f'lean {math.degrees(max(abs(roll), abs(pitch))):.0f} deg'
                break
            # Pad-tip settle can swing before the attitude loop is quiet.
            # During forward lean, EKF position drifts hard — abort on truth
            # only when the OLD sim provides it; otherwise use EKF.
            abort_after = float(getattr(args, 'settle_s', 0.0) or 0.0) + 1.0
            if getattr(args, 'mode', None) == 'drive':
                alt_limit = 8.0
            elif tilt_compensate:
                alt_limit = 5.0
            else:
                alt_limit = ABORT_ALTITUDE_M
            if tilt_compensate and d_truth_z is not None:
                alt_for_abort, alt_src = d_truth_z, 'truth'
            elif d_truth_z is not None:
                alt_for_abort, alt_src = d_truth_z, 'truth'
            else:
                alt_for_abort, alt_src = d_ekf_z, 'EKF'
            if (elapsed >= abort_after
                    and alt_for_abort is not None
                    and abs(alt_for_abort) > alt_limit):
                abort = f'{alt_src} altitude {alt_for_abort:+.1f} m'
                break

            desired = desired_at(elapsed, shared_data)
            yaw_rate = 0.0
            thrust_override = None
            if not isinstance(desired, (list, tuple)):
                raise TypeError('desired_at must return a tuple')
            if len(desired) == 2:
                des_roll, des_pitch = desired
            elif len(desired) == 3:
                des_roll, des_pitch, yaw_rate = desired
            else:
                des_roll, des_pitch, yaw_rate, thrust_override = desired[:4]
            gate_nx, gate_ny, gate_src = _image_gate_norm(shared_data)
            # Feedback source: the EKF estimate (what races) or, as a
            # diagnostic, the sim's own attitude.
            fb_roll, fb_pitch = roll, pitch
            if use_truth_feedback and truth is not None:
                if truth['roll'] is not None:
                    fb_roll, fb_pitch = truth['roll'], truth['pitch']
            # kalman_planner calls update(err, dt) with no measurement_rate,
            # i.e. derivative-on-error. Mirror that exactly or a kd tuned here
            # would not transfer to real flight.
            roll_rate = roll_pid.update(des_roll - fb_roll, dt)
            pitch_rate = pitch_pid.update(des_pitch - fb_pitch, dt)
            if thrust_override is not None:
                thrust = float(thrust_override)
            else:
                thrust = config.HOVER_THRUST
                # Only boost when we are *commanding* a lean. Using measured tip
                # attitude during level settle over-thrusts, climbs off the pad,
                # and the OLD sim early-start DQs the run. Threshold is 0.5° so
                # Phase 4.6 crawl (~1.4° pitch) still gets lean_boost.
                if tilt_compensate and (
                        abs(des_roll) > math.radians(0.5)
                        or abs(des_pitch) > math.radians(0.5)):
                    thrust = _tilt_compensated_thrust(
                        config.HOVER_THRUST,
                        fb_roll if abs(fb_roll) >= abs(des_roll) else des_roll,
                        fb_pitch if abs(fb_pitch) >= abs(des_pitch) else des_pitch,
                        lean_boost=lean_boost,
                    )

            # Same contract the kalman planner uses, so the controller takes
            # its direct rate path rather than the velocity fallback.
            shared_data['planner_target'] = {
                'kalman': True,
                'roll_rate': roll_rate,
                'pitch_rate': pitch_rate,
                'yaw_rate': float(yaw_rate or 0.0),
                'thrust': thrust,
            }
            shared_data['planner_mode'] = 'kalman_dual_gate'
            controller.update()

            race = shared_data.get('race_status') or {}
            race_finish_ns = race.get('race_finish_ns')
            try:
                race_finish_ns = int(race_finish_ns) if race_finish_ns else 0
            except (TypeError, ValueError):
                race_finish_ns = 0
            active_gate = race.get('active_gate')
            collision = shared_data.get('collision') or {}
            recorder.write({
                't': round(elapsed, 4),
                'des_roll': des_roll, 'roll': roll,
                'des_pitch': des_pitch, 'pitch': pitch,
                'roll_err': des_roll - fb_roll,
                'pitch_err': des_pitch - fb_pitch,
                'feedback': 'truth' if use_truth_feedback else 'ekf',
                'gyro_x': rollspeed, 'gyro_y': pitchspeed,
                'roll_rate_cmd': roll_rate, 'pitch_rate_cmd': pitch_rate,
                'thrust': thrust,
                'n_solved': n_solved,
                'd_norm_y': d_norm_y,
                'd_ekf_z': d_ekf_z,
                'd_truth_z': d_truth_z,
                'truth_roll': truth['roll'] if truth else None,
                'truth_pitch': truth['pitch'] if truth else None,
                'gate1_range_m': g1_range,
                'kp_att': config.KALMAN_KP_ATT,
                'kd_att': config.KALMAN_KD_ATT,
                'hover_thrust': config.HOVER_THRUST,
                'gate_nx': gate_nx,
                'gate_ny': gate_ny,
                'gate_src': gate_src,
                'area_px': area_px,
                'local_x': local_x,
                'local_y': local_y,
                'local_z': local_z,
                'yaw_rate_cmd': float(yaw_rate or 0.0),
                'race_finish_ns': race_finish_ns,
                'active_gate': active_gate,
                'collision_id': collision.get('id') or collision.get('collision_id'),
            })
            # race_finish_ns is -1/0 while racing; a positive value means the
            # sim ended the run (finish or early-start DQ).
            if race_finish_ns > 0 and abort is None:
                abort = (
                    f'race finished / early-start DQ '
                    f'(race_finish_ns={race_finish_ns})'
                )
                break
            if not args.quiet:
                print(
                    f'{elapsed:5.1f}  {des_roll:+7.3f} {roll:+7.3f}'
                    f'   {des_pitch:+7.3f} {pitch:+7.3f}'
                    f'  {roll_rate:+6.3f} {pitch_rate:+6.3f}'
                    f'  {thrust:6.3f} {_fmt(d_norm_y, "+5.2f")}'
                    f' {_fmt(d_ekf_z, "+7.2f")} {_fmt(g1_range, "6.2f")}',
                    flush=True,
                )
            time.sleep(period)
    except KeyboardInterrupt:
        print('\n[STOP] interrupted', flush=True)
    finally:
        controller.disarm()
        print('Disarmed.', flush=True)
        recorder.close()
        shutdown(components)

    if abort:
        print(f'\n[ABORT] {abort} — disarmed. Back the gains off.', flush=True)
    coverage = solved_samples / len(recorder.rows) if recorder.rows else 0.0
    return recorder, abort, coverage


def run_hover(args) -> int:
    level = lambda _elapsed, _sd: (0.0, 0.0)
    recorder, abort, coverage = run_attitude_loop(args, level, 'hover')
    if recorder is None:
        print(f'[FAIL] {abort}', flush=True)
        return 2
    rows = recorder.rows

    print('\n=== hover / HOVER_THRUST trim ===')
    if not rows:
        print('no samples collected')
        return 1
    span_s = rows[-1]['t'] - rows[0]['t'] if len(rows) > 1 else 0.0
    print(f'PnP coverage         {coverage * 100:.0f}% of samples had a gate '
          'solved')

    ny = [r['d_norm_y'] for r in rows if isinstance(r['d_norm_y'], float)]
    ez = [r['d_ekf_z'] for r in rows if isinstance(r['d_ekf_z'], float)]
    tz = [r.get('d_truth_z') for r in rows
          if isinstance(r.get('d_truth_z'), float)]

    verdict_rate = None
    if tz and span_s > 0:
        print(f'truth altitude       {tz[-1]:+.2f} m over {span_s:.1f} s'
              '   <-- exact, sim odometry')
        # Score post-settle rate: the first ~3 s include a pad-tip transient
        # (~17 deg lean) that biases the full-run average even when trim is
        # correct afterward.
        settle_s = 3.0
        settled = [r for r in rows
                   if isinstance(r.get('d_truth_z'), float)
                   and r['t'] >= settle_s]
        if len(settled) >= 2:
            dt = settled[-1]['t'] - settled[0]['t']
            verdict_rate = (
                (settled[-1]['d_truth_z'] - settled[0]['d_truth_z']) / dt
                if dt > 0 else tz[-1] / span_s
            )
            print(f'post-settle rate     {verdict_rate:+.3f} m/s '
                  f'(t>={settle_s:.0f}s; ignores pad-tip transient)')
        else:
            verdict_rate = tz[-1] / span_s
        source = 'sim odometry (exact)'
        threshold = 0.05           # m/s
    elif ny and coverage >= 0.5 and span_s > 0:
        # Image-space: gate drifts DOWN the frame (norm_y up) as the drone rises.
        drift = ny[-1]
        print(f'gate image drift     {drift:+.3f} normalized over {span_s:.1f} s'
              '   <-- drift-free, needs a gate in view')
        verdict_rate = drift / span_s
        source = 'image-space gate drift'
        threshold = 0.010          # normalized units per second
    elif ez and span_s > 0:
        print(f'EKF altitude change  {ez[-1]:+.2f} m over {span_s:.1f} s'
              '   <-- IMU dead reckoning, treat as indicative only')
        verdict_rate = ez[-1] / span_s
        source = 'EKF altitude (drifty)'
        threshold = 0.05           # m/s
    if verdict_rate is None:
        print('no usable vertical observable — keep a gate in view for this '
              'test, or check the EKF got IMU')
        print(f'\nCSV: {recorder.path}')
        return 1 if abort else 0

    hover_thrust = rows[0]['hover_thrust']
    print(f'basis                {source}')
    if abs(verdict_rate) < threshold:
        print(f'VERDICT: HOVER_THRUST={hover_thrust} holds altitude. Keep it.')
    elif verdict_rate > 0:
        print(f'VERDICT: climbing — reduce HOVER_THRUST '
              f'(try {hover_thrust - 0.005:.3f})')
    else:
        print(f'VERDICT: sinking — raise HOVER_THRUST '
              f'(try {hover_thrust + 0.005:.3f})')
    if coverage < 0.5:
        print('\n[WARN] gate was visible for under half the run, so the '
              'verdict rests on IMU dead reckoning. Re-run with the drone '
              'facing a gate for a trustworthy trim.')

    leans = [max(abs(r['roll']), abs(r['pitch'])) for r in rows]
    if leans:
        print(f'peak lean            {math.degrees(max(leans)):.1f} deg '
              '(should stay small; large means the attitude loop is fighting)')
    print(f'\nCSV: {recorder.path}')
    return 1 if abort else 0


def _post_settle_alt_rate(rows, settle_s=3.0):
    """Return (rate_m_s, source_label) from truth/EKF climb after settle.

    LOCAL_POSITION_NED on this sim occasionally repeats the pad altitude
    (d_truth_z≈0) between real samples. Drop near-zero echoes, then take the
    early-third vs late-third mean slope on the remaining points.
    """
    settled = [r for r in rows if r['t'] >= settle_s]
    if len(settled) < 5:
        settled = rows

    def robust_rate(key):
        pts = [(float(r['t']), float(r[key])) for r in settled
               if isinstance(r.get(key), float)]
        if len(pts) < 5:
            return None
        vals = [v for _, v in pts]
        span_alt = max(vals) - min(vals)
        # When altitude clearly moved, drop pad-echo zeros.
        if span_alt > 0.4:
            kept = [(t, v) for t, v in pts if abs(v) > 0.08]
            if len(kept) >= 5:
                pts = kept
        n = len(pts)
        early = pts[: max(2, n // 3)]
        late = pts[max(0, n - n // 3):]
        dt = late[-1][0] - early[0][0]
        if dt <= 0.2:
            return None
        v_early = statistics.fmean(v for _, v in early)
        v_late = statistics.fmean(v for _, v in late)
        return (v_late - v_early) / dt

    # Prefer EKF when truth is dominated by pad echoes (many ~0 samples).
    tz = [r.get('d_truth_z') for r in settled
          if isinstance(r.get('d_truth_z'), float)]
    truth_nonzero = sum(1 for v in tz if abs(v) > 0.08)
    if tz and truth_nonzero >= max(5, len(tz) // 4):
        rate = robust_rate('d_truth_z')
        if rate is not None:
            return rate, 'truth'
    rate = robust_rate('d_ekf_z')
    if rate is not None:
        return rate, 'ekf'
    rate = robust_rate('d_truth_z')
    if rate is not None:
        return rate, 'truth'
    return None, 'none'


def run_lean_hover(args) -> int:
    amplitude = math.radians(args.amplitude_deg)
    settle = args.settle_s
    axis = getattr(args, 'axis', 'roll')

    def desired(elapsed, _sd):
        if elapsed < settle:
            return 0.0, 0.0
        # Roll keeps the craft near the pad. Pitch flies through the start
        # line and the OLD sim early-start DQs — altitude then looks like a
        # crash even though the attitude loop was fine.
        if axis == 'roll':
            return amplitude, 0.0
        import config as _cfg
        return 0.0, float(_cfg.FORWARD_PITCH_SIGN) * amplitude

    if axis == 'pitch':
        print('[WARN] pitch lean can early-start DQ on the pad — prefer '
              '--axis roll for Phase 4.5', flush=True)

    recorder, abort, coverage = run_attitude_loop(
        args, desired, f'lean_hover_{axis}', tilt_compensate=True,
    )
    if recorder is None:
        print(f'[FAIL] {abort}', flush=True)
        return 2
    rows = recorder.rows
    print(f'\n=== lean-hover {axis} ({args.amplitude_deg:.1f} deg after '
          f't={settle:.1f}s, tilt-compensated) ===')
    if not rows:
        print('no samples collected')
        return 1

    if abort and 'early-start' in str(abort):
        print(f'[INVALID] {abort}')
        print('VERDICT: inconclusive — race DQ, not an attitude/trim fail. '
              'Re-run with --axis roll (default).')
        print(f'\nCSV: {recorder.path}')
        return 1

    lean_rows = [r for r in rows if r['t'] >= settle]
    # Score only samples before any race_finish flag if present mid-file.
    clean = [r for r in lean_rows
             if not (isinstance(r.get('race_finish_ns'), (int, float))
                     and r['race_finish_ns'] > 0)]
    if len(clean) >= 5:
        lean_rows = clean
    thrusts = [r['thrust'] for r in lean_rows
               if isinstance(r.get('thrust'), float)]
    ht = rows[0]['hover_thrust']
    if thrusts:
        print(f'thrust while leaned  mean={statistics.fmean(thrusts):.4f}  '
              f'min={min(thrusts):.4f}  max={max(thrusts):.4f}  '
              f'(HOVER_THRUST={ht})')
        if max(thrusts) <= ht + 1e-6:
            print('  [!] thrust never rose above HT — tilt compensation '
                  'inactive?')
    rate, src = _post_settle_alt_rate(lean_rows, settle_s=settle + 1.0)
    print(f'PnP coverage         {coverage * 100:.0f}%')
    if rate is None:
        print('altitude rate        unknown (no truth/EKF samples)')
        print('VERDICT: inconclusive')
        print(f'\nCSV: {recorder.path}')
        return 1 if abort else 0
    print(f'altitude rate        {rate:+.3f} m/s ({src}, post-lean settle)')
    threshold = 0.05
    if abs(rate) < threshold:
        print(f'VERDICT: lean-hover holds altitude at HT={ht}. Pass.')
        rc = 0
    elif rate > 0:
        print(f'VERDICT: climbing under lean — reduce HT or tilt boost '
              f'(try {ht - 0.005:.3f})')
        rc = 1
    else:
        print(f'VERDICT: sinking under lean — raise HT '
              f'(try {ht + 0.005:.3f}) or check tilt compensation')
        rc = 1
    print(f'\nCSV: {recorder.path}')
    return 1 if abort else rc


def run_crawl(args) -> int:
    import config
    lean = math.radians(args.lean_deg)
    settle = args.settle_s
    lat_sign = float(config.LATERAL_LEAN_SIGN)

    def desired(elapsed, shared_data):
        if elapsed < settle:
            return 0.0, 0.0
        nx, ny, _src = _image_gate_norm(shared_data)
        if nx is None:
            # No gate: hold level (do not invent a punch).
            return 0.0, 0.0
        # Gentle forward always when framed; lateral proportional to nx.
        des_pitch = float(config.FORWARD_PITCH_SIGN) * 0.70 * lean
        des_roll = lat_sign * float(max(-1.0, min(1.0, nx))) * lean
        # Light vertical nudge from ny (image down → slight less lean).
        if ny is not None and ny > 0.15:
            des_pitch *= 0.5
        return des_roll, des_pitch

    recorder, abort, coverage = run_attitude_loop(
        args, desired, 'crawl', tilt_compensate=True,
    )
    if recorder is None:
        print(f'[FAIL] {abort}', flush=True)
        return 2
    rows = recorder.rows
    print(f'\n=== image crawl (lean≤{args.lean_deg:.1f} deg after '
          f't={settle:.1f}s) ===')
    if not rows:
        print('no samples collected')
        return 1

    post = [r for r in rows if r['t'] >= settle]
    seen = [r for r in post if r.get('gate_nx') is not None]
    pitched = [r for r in seen
               if isinstance(r.get('des_pitch'), float)
               and abs(r['des_pitch']) > 1e-4]
    print(f'gate visible         {len(seen)}/{len(post)} post-settle samples '
          f'({(100.0 * len(seen) / len(post)) if post else 0:.0f}%)')
    print(f'PnP coverage         {coverage * 100:.0f}%')
    print(f'nonzero des_pitch    {len(pitched)}/{max(len(seen), 1)} while '
          'gate seen')
    rate, src = _post_settle_alt_rate(post, settle_s=settle + 1.0)
    if rate is not None:
        print(f'altitude rate        {rate:+.3f} m/s ({src})')
    if not seen:
        print('VERDICT: FAIL — no gate in image. Face a gate and re-run.')
        rc = 1
    elif len(pitched) < 0.5 * len(seen):
        print('VERDICT: FAIL — gate seen but crawl rarely commanded pitch.')
        rc = 1
    elif rate is not None and rate < -0.08:
        print('VERDICT: FAIL — altitude collapsing while crawling '
              f'(rate {rate:+.3f}). Raise --lean-boost.')
        rc = 1
    elif rate is not None and rate > 0.08:
        print('VERDICT: FAIL — climbing while crawling '
              f'(rate {rate:+.3f}). Reduce --lean-boost '
              f'(try {max(0.0, lean_boost_used(args) - 0.001):.3f}).')
        rc = 1
    elif rate is not None and abs(rate) <= 0.05:
        print('VERDICT: Pass — vision→lean wiring + altitude hold '
              f'(±0.05 m/s, lean_boost={lean_boost_used(args):.3f}).')
        rc = 0
    else:
        print('VERDICT: Pass — wiring alive; altitude marginal '
              f'(|rate|={abs(rate):.3f} ≤0.08). Prefer |rate|≤0.05.')
        rc = 0
    print(f'\nCSV: {recorder.path}')
    return 1 if abort else rc


def lean_boost_used(args) -> float:
    return float(getattr(args, 'lean_boost', 0.0) or 0.0)


def run_drive(args) -> int:
    """Phase 4.9 — chase the gate from relative pose (learn from the gate).

    Every correction comes from where the gate is relative to the drone
    (PnP body offset when available, else image×range). No absolute
    "climb to 2 m on takeoff" — that lofted drive_t the moment it armed.
    """
    import config
    from control.pid import PIDConfig, PIDController

    lean = math.radians(float(args.lean_deg))
    settle = float(args.settle_s)
    lat_sign = float(config.LATERAL_LEAN_SIGN)
    hover = float(config.HOVER_THRUST)
    max_yaw = min(config.YAW_RATE_MAX_RAD_S, math.radians(30.0))
    yaw_pid = PIDController(PIDConfig(
        kp=config.KALMAN_KP_YAW, kd=0.0,
        output_min=-max_yaw, output_max=max_yaw,
    ))
    last_yaw = 0.0
    yaw_slew = math.radians(120.0)
    last_des = (0.0, 0.0, 0.0, hover)
    last_active_gate = {'v': None}
    # Post-pass: short forward coast through the hole, then hunt gate 2.
    pass_t = {'t': None}
    coast_until = {'t': -1e9}
    seek_until = {'t': -1e9}
    bearing_latched = {
        'nx': 0.28, 'ny': -0.06, 'ok': False,  # default right bias
    }
    # Once gate 2 is chaseable, keep committing even across YOLO flicker
    # (drive_ak: chaseable→seek→yaw-scan→bbox shrink).
    g2_lock = {
        'on': False, 'nx': 0.0, 'ny': 0.0, 'area': 0.0, 't': -1e9,
    }
    from camera_model import attitude_compensated_gate_norm

    filt = {
        'nx': None, 'ny': None, 'ny_lvl': None, 'ny_dot': 0.0, 'thr': hover,
    }
    dt_cmd = 1.0 / max(float(args.hz), 1.0)

    def _note_gate_pass(elapsed, shared_data) -> None:
        nonlocal last_yaw
        race = shared_data.get('race_status') or {}
        ag = race.get('active_gate')
        try:
            ag_i = int(ag) if ag is not None else None
        except (TypeError, ValueError):
            ag_i = None
        if ag_i is None:
            return
        prev = last_active_gate['v']
        if prev is not None and ag_i > prev:
            pass_t['t'] = elapsed
            coast_until['t'] = elapsed + 1.2
            seek_until['t'] = elapsed + 16.0
            yaw_pid.reset()
            last_yaw = 0.0
            filt['nx'] = filt['ny'] = filt['ny_lvl'] = None
            filt['ny_dot'] = 0.0
            g2_lock['on'] = False
            g2_lock['t'] = -1e9
            g2_lock['_announced'] = False
            # Keep current thrust — don't slam to hover mid-pass.
            bearing_latched['ok'] = False
            bearing_latched['nx'] = 0.28
            bearing_latched['ny'] = -0.06
            print(
                f'[DRIVE] post-pass hunt: coast→{coast_until["t"]:.1f}s '
                f'seek→{seek_until["t"]:.1f}s',
                flush=True,
            )
        last_active_gate['v'] = ag_i

    def _latch_bearing(shared_data) -> None:
        bearing = shared_data.get('course_bearing') or {}
        if not isinstance(bearing, dict):
            return
        bx = _f(bearing.get('nx'))
        by = _f(bearing.get('ny'))
        if bx is None:
            return
        # Prefer a real vision bearing over the synthetic default once.
        src = str(bearing.get('source') or '')
        if bearing_latched['ok'] and src == 'default_right':
            return
        bearing_latched['nx'] = float(max(-0.70, min(0.70, bx)))
        if by is not None:
            bearing_latched['ny'] = float(max(-0.25, min(0.10, by)))
        bearing_latched['ok'] = True

    def _gate_range_m(shared_data, *, prefer_area: bool = False) -> float:
        # After gate 1, dual.gate1_range_m stays on the *old* gate (~25 m)
        # and makes chase think we are still far while YOLO area is the
        # real target. Prefer bbox range once we are hunting gate 2.
        det = shared_data.get('gate_detection') or {}
        area = _f(det.get('area_px')) if isinstance(det, dict) else None
        area_rng = None
        if area is not None and area > 50.0:
            area_rng = float((320.0 * 1.5) / math.sqrt(area))
        if prefer_area and area_rng is not None:
            return area_rng
        dual = shared_data.get('dual_gate_pnp') or {}
        rng = _f(dual.get('gate1_range_m'))
        if rng is not None and rng >= 0.5:
            return float(rng)
        if area_rng is not None:
            return area_rng
        return 15.0

    def _slew_yaw(yaw_rate):
        nonlocal last_yaw
        max_step = yaw_slew * dt_cmd
        yaw_rate = float(max(last_yaw - max_step,
                             min(last_yaw + max_step, yaw_rate)))
        last_yaw = yaw_rate
        return yaw_rate

    def _slew_thrust(thr_tgt, rate=0.030):
        thr_prev = float(filt['thr'])
        max_dthr = rate * dt_cmd
        thrust = float(max(thr_prev - max_dthr,
                           min(thr_prev + max_dthr, thr_tgt)))
        filt['thr'] = thrust
        return thrust

    def desired(elapsed, shared_data):
        nonlocal last_yaw, last_des
        _note_gate_pass(elapsed, shared_data)
        coasting = bool(elapsed < coast_until['t'])
        seeking = bool(elapsed < seek_until['t'])
        if seeking:
            _latch_bearing(shared_data)
            shared_data['post_pass_hunt'] = True
        else:
            shared_data['post_pass_hunt'] = False

        if elapsed < settle:
            yaw_pid.reset()
            last_yaw = 0.0
            filt['nx'] = filt['ny'] = filt['ny_lvl'] = None
            filt['ny_dot'] = 0.0
            filt['thr'] = hover
            shared_data['gate_aim'] = None
            return 0.0, 0.0, 0.0, hover

        nx, ny, _src = _image_gate_norm(shared_data)
        att = shared_data.get('attitude') or {}
        roll_meas = _f(att.get('roll')) or 0.0
        pitch_meas = _f(att.get('pitch')) or 0.0

        # Post-pass quality: "visible" ≠ lock. drive_ae counted ny≈0.85 /
        # area≈700 centered floor-specks as gate 2 while des_pitch crawled
        # at 0.07 and range never closed. Only chase solid candidates;
        # weak ones are yaw hints while we keep sweeping.
        det = shared_data.get('gate_detection') or {}
        area_px = _f(det.get('area_px')) if isinstance(det, dict) else None
        hint_nx = None
        chaseable = bool(nx is not None and ny is not None)
        if chaseable:
            ny_r, nx_r = float(ny), float(nx)
            if area_px is not None and area_px > 90000.0:
                chaseable = False
            elif ny_r > 0.92:
                chaseable = False
            elif seeking and ny_r > 0.78 and abs(nx_r) < 0.22:
                # Dead-center floor band — ignore.
                chaseable = False
            elif seeking and ny_r > 0.72:
                # drive_ai: area≈2500@ny≈0.86/nx≈0.7 was a real next-gate
                # peek — chase side/sized ones; only tiny lows stay hints.
                side = abs(nx_r) >= 0.28
                big = area_px is not None and area_px >= 1800.0
                if big or side:
                    chaseable = True
                else:
                    hint_nx = nx_r
                    chaseable = False
            elif seeking and area_px is not None and area_px < 900.0:
                hint_nx = nx_r
                chaseable = False
        if chaseable and seeking:
            g2_lock['on'] = True
            g2_lock['nx'] = float(nx)
            g2_lock['ny'] = float(ny) if ny is not None else float(g2_lock['ny'])
            if area_px is not None:
                g2_lock['area'] = float(area_px)
            g2_lock['t'] = float(elapsed)
            if not g2_lock.get('_announced'):
                print(
                    f'[DRIVE] gate2 lock latched nx={g2_lock["nx"]:+.2f} '
                    f'ny={g2_lock["ny"]:+.2f} area={g2_lock["area"]:.0f}',
                    flush=True,
                )
                g2_lock['_announced'] = True
        elif (
            seeking
            and g2_lock['on']
            and (elapsed - float(g2_lock['t'])) <= 2.5
        ):
            # Hold last chase through brief flicker — don't yaw-scan away.
            # Refresh TTL if YOLO still sees *something* near the latch.
            if nx is not None and ny is not None:
                g2_lock['t'] = float(elapsed)
                g2_lock['nx'] = 0.7 * float(g2_lock['nx']) + 0.3 * float(nx)
                g2_lock['ny'] = 0.7 * float(g2_lock['ny']) + 0.3 * float(ny)
                if area_px is not None:
                    g2_lock['area'] = 0.7 * float(g2_lock['area']) + 0.3 * float(
                        area_px
                    )
            nx = float(g2_lock['nx'])
            ny = float(g2_lock['ny'])
            if area_px is None:
                area_px = float(g2_lock['area']) or None
            chaseable = True
            hint_nx = None
        elif seeking and g2_lock['on'] and (elapsed - float(g2_lock['t'])) > 2.5:
            g2_lock['on'] = False
            g2_lock['_announced'] = False
        if not chaseable:
            nx, ny = None, None

        # --- Blind / hint post-pass coast / seek (no chaseable gate). ---
        # Body-zero hover (drive_ah/ai) stopped sky-climb but left the
        # 20°-up camera high — gate 2 stuck at ny≈0.85 while we yaw-scanned
        # in place. Mild forward tip keeps the view on course and closes;
        # tip harder once chaseable.
        if nx is None:
            filt['nx'] = filt['ny'] = filt['ny_lvl'] = None
            filt['ny_dot'] = 0.0
            shared_data['gate_aim'] = None
            bx = float(bearing_latched['nx'])
            by = float(bearing_latched['ny'])
            # Widen the hunt if still blind — drive_ac sat at yaw≈0 with
            # default_right and never swept onto gate 2.
            blind_age = 0.0
            if pass_t['t'] is not None and seeking:
                blind_age = max(0.0, elapsed - max(pass_t['t'],
                                                   coast_until['t']))
            if coasting:
                # Short exit coast — mostly level, tiny forward to clear rim.
                pitch_frac = 0.15
                yaw_cmd = 0.25 * bx
            elif seeking:
                # Gentle close while hunting (not hover-spin).
                pitch_frac = 0.32
                if hint_nx is not None:
                    pitch_frac = 0.48
                    yaw_cmd = float(max(
                        -max_yaw, min(max_yaw, 0.85 * hint_nx + 0.15 * bx)
                    ))
                else:
                    # Open-loop yaw onto bearing, then keep sweeping right.
                    yaw_cmd = float(max(
                        0.20, min(max_yaw, 0.55 + 0.08 * blind_age)
                    ))
                    if bx < 0.0:
                        yaw_cmd = -yaw_cmd
                    # After 3s still blind: slow back-and-forth scan ±.
                    if blind_age > 3.0:
                        phase = math.sin(1.2 * (blind_age - 3.0))
                        yaw_cmd = (
                            0.45 * phase if bx >= 0 else -0.45 * phase
                        )
            else:
                # Pre-pass blind (through-hole blackout) — keep crawling.
                # drive_ag set this to 0 and scraped gate 1's rim.
                pitch_frac = 0.45
                yaw_cmd = 0.25 * bx
            yaw_rate = float(max(-max_yaw, min(max_yaw, yaw_cmd)))
            yaw_rate = _slew_yaw(yaw_rate)
            des_pitch = float(config.FORWARD_PITCH_SIGN) * lean * pitch_frac
            if coasting:
                thr_tgt = float(hover)
                thr_tgt = float(max(hover - 0.004, min(hover + 0.004, thr_tgt)))
            elif seeking:
                tilt = max(0.94, math.cos(abs(des_pitch)))
                thr_tgt = hover / tilt
                thr_tgt = float(max(hover - 0.004, min(hover + 0.006, thr_tgt)))
            else:
                tilt = max(0.92, math.cos(abs(des_pitch)))
                thr_tgt = hover / tilt + 0.5 * lean_boost_used(args)
                thr_tgt = float(max(hover - 0.006, min(hover + 0.008, thr_tgt)))
            thrust = _slew_thrust(thr_tgt, rate=0.020)
            last_des = (0.0, des_pitch, yaw_rate, thrust)
            return 0.0, des_pitch, yaw_rate, thrust

        # Fresh lock after pass — accept the new gate, don't keep old EMA.
        if seeking and filt['nx'] is None:
            yaw_pid.reset()

        nx_f = float(nx)
        ny_f = float(ny) if ny is not None else 0.0
        a = 0.28
        if filt['nx'] is None:
            filt['nx'], filt['ny'] = nx_f, ny_f
        else:
            if abs(nx_f - filt['nx']) > 0.55:
                filt['nx'] = 0.6 * filt['nx'] + 0.4 * nx_f
                yaw_pid.reset()
            else:
                filt['nx'] = (1 - a) * filt['nx'] + a * nx_f
            if abs(ny_f - float(filt['ny'])) > 0.55:
                ny_f = float(filt['ny'])
            filt['ny'] = (1 - a) * float(filt['ny']) + a * ny_f
        nx_f, ny_f = float(filt['nx']), float(filt['ny'])

        # Lean-aware through-aim: where the gate is after removing pitch/roll.
        # This is the variable that says "fly through here" while leaned.
        gate_nx_level, gate_ny_level = attitude_compensated_gate_norm(
            nx_f, ny_f, roll_meas, pitch_meas,
        )
        if filt['ny_lvl'] is None:
            filt['ny_lvl'] = gate_ny_level
            filt['ny_dot'] = 0.0
        else:
            prev = float(filt['ny_lvl'])
            filt['ny_lvl'] = 0.70 * prev + 0.30 * gate_ny_level
            filt['ny_dot'] = (
                0.75 * float(filt['ny_dot'])
                + 0.25 * (float(filt['ny_lvl']) - prev) / dt_cmd
            )
        gate_ny_level = float(filt['ny_lvl'])
        ny_dot = float(filt['ny_dot'])
        rng = _gate_range_m(shared_data, prefer_area=bool(seeking))

        shared_data['gate_aim'] = {
            'nx_raw': nx_f,
            'ny_raw': ny_f,
            'nx_level': float(gate_nx_level),
            'ny_level': float(gate_ny_level),
            'roll_rad': float(roll_meas),
            'pitch_rad': float(pitch_meas),
            'range_m': float(rng),
        }

        # Through-aim = camera boresight (ny_level≈0). Lean already removed.
        # aim_bias > 0 ⇒ setpoint is slightly BELOW image centre ⇒ climb a
        # bit when the gate is centred (flies higher through the hole).
        # drive_aa used −0.05 by mistake and sank harder.
        aim_bias = 0.08
        e_y = gate_ny_level - aim_bias
        if abs(e_y) < 0.10:
            e_y = 0.0
        # Soft height loop — prefer climb over sink on approach.
        # drive_am: gate2 stayed at ny≈0.7 while closing — cam tilt puts a
        # co-height gate low; tip forward + allow sink, don't loft.
        k_up = 0.016 if rng < 7.0 else 0.013
        k_dn = 0.008 if rng < 7.0 else 0.006
        if seeking and g2_lock['on']:
            # drive_an locked gate2 then lofted +8 m (abort). Prefer sink /
            # hold; only tiny climb when the gate is clearly above aim.
            k_up = 0.003
            k_dn = 0.012 if e_y > 0.0 else 0.005
        elif seeking and rng > 10.0:
            k_up *= 0.55
            k_dn *= 0.55
        k_y = k_dn if e_y > 0.0 else k_up
        k_d = 0.010
        u_z = -k_y * math.tanh(e_y / 0.40) - k_d * math.tanh(ny_dot / 1.2)
        if seeking and g2_lock['on'] and u_z > 0.0:
            u_z *= 0.35

        align_x = float(max(0.0, min(1.0, 1.0 - abs(gate_nx_level) / 0.45)))
        align_y = float(
            max(0.0, min(1.0, 1.0 - abs(gate_ny_level - aim_bias) / 0.40))
        )
        align = align_x * align_y
        if e_y == 0.0:
            pitch_frac = 0.50 + 0.50 * align
        else:
            pitch_frac = 0.32 + 0.40 * align
        if coasting:
            # Still punching out of gate 1 — stay nearly level on rim noise.
            pitch_frac = min(max(pitch_frac, 0.10), 0.25)
        elif seeking:
            # Forward tip raises a low gate in the FOV (20° cam tilt) and
            # closes. drive_ai briefly locked then dropped to pitch=0.
            pitch_frac = max(pitch_frac, 0.55 + 0.25 * align_x)
            if ny_f > 0.55 or rng > 10.0 or (
                area_px is not None and area_px < 5000.0
            ):
                pitch_frac = max(pitch_frac, 0.72)
            if g2_lock['on']:
                pitch_frac = max(pitch_frac, 0.80)
        des_pitch = float(config.FORWARD_PITCH_SIGN) * lean * pitch_frac
        des_roll = (
            lat_sign * float(max(-1.2, min(1.2, gate_nx_level))) * 0.50 * lean
        )
        yaw_rate = float(yaw_pid.update(
            float(max(-1.2, min(1.2, gate_nx_level))), dt_cmd
        ))
        yaw_rate = _slew_yaw(yaw_rate)

        tilt = max(
            0.90,
            math.cos(abs(des_pitch)) * math.cos(abs(des_roll)),
        )
        tilt_hover = hover / tilt + lean_boost_used(args)
        thr_tgt = tilt_hover + u_z
        # Floor/ceiling: allow more sink when hunting a low gate2; cap climb.
        thr_lo = hover - (0.014 if (seeking and g2_lock['on']) else 0.010)
        thr_hi = hover + (0.008 if (seeking and g2_lock['on']) else 0.016)
        thr_tgt = float(max(thr_lo, min(thr_hi, thr_tgt)))
        thrust = _slew_thrust(thr_tgt, rate=0.028)
        last_des = (des_roll, des_pitch, yaw_rate, thrust)
        return des_roll, des_pitch, yaw_rate, thrust

    recorder, abort, coverage = run_attitude_loop(
        args, desired, 'drive', tilt_compensate=True,
    )
    if recorder is None:
        print(f'[FAIL] {abort}', flush=True)
        return 2
    rows = recorder.rows
    print(f'\n=== drive / hard chase (lean={args.lean_deg:.1f} deg after '
          f't={settle:.1f}s) ===')
    if not rows:
        print('no samples collected')
        return 1

    post = [r for r in rows if r['t'] >= settle]
    seen = [r for r in post if r.get('gate_nx') is not None]
    pitched = [
        r for r in seen
        if isinstance(r.get('des_pitch'), float) and abs(r['des_pitch']) > 1e-3
    ]
    ranges = [
        float(r['gate1_range_m']) for r in post
        if isinstance(r.get('gate1_range_m'), float)
        and math.isfinite(float(r['gate1_range_m']))
        and float(r['gate1_range_m']) > 0.5
    ]
    xs = [
        float(r['local_x']) for r in post
        if isinstance(r.get('local_x'), float)
        and math.isfinite(float(r['local_x']))
    ]
    areas = [
        float(r['area_px']) for r in seen
        if isinstance(r.get('area_px'), float)
        and math.isfinite(float(r['area_px']))
        and float(r['area_px']) > 0.0
    ]

    print(f'gate visible         {len(seen)}/{len(post)} post-settle '
          f'({(100.0 * len(seen) / len(post)) if post else 0:.0f}%)')
    print(f'PnP coverage         {coverage * 100:.0f}%')
    print(f'nonzero des_pitch    {len(pitched)}/{max(len(seen), 1)} while '
          'gate seen')

    range_delta = None
    if len(ranges) >= 4:
        # Median of first/last ~1s windows — ignore single-frame spikes.
        n_win = max(3, len(ranges) // 8)
        r0 = statistics.median(ranges[:n_win])
        r1 = statistics.median(ranges[-n_win:])
        range_delta = r0 - r1
        print(f'range close          {r0:.2f} → {r1:.2f} m '
              f'(Δ {range_delta:+.2f} m)')

    x_delta = None
    if len(xs) >= 4:
        n_win = max(3, len(xs) // 8)
        x0 = statistics.median(xs[:n_win])
        x1 = statistics.median(xs[-n_win:])
        x_delta = x1 - x0
        print(f'local N advance      {x0:+.2f} → {x1:+.2f} m '
              f'(Δ {x_delta:+.2f} m)')

    area_delta = None
    if len(areas) >= 4:
        n_win = max(3, len(areas) // 8)
        a0 = statistics.median(areas[:n_win])
        a1 = statistics.median(areas[-n_win:])
        area_delta = a1 - a0
        print(f'bbox area            {a0:.0f} → {a1:.0f} px '
              f'(Δ {area_delta:+.0f})')

    rate, src = _post_settle_alt_rate(post, settle_s=settle + 1.0)
    if rate is not None:
        print(f'altitude rate        {rate:+.3f} m/s ({src})')

    # Post-pass gate-2 reacquire: active_gate advanced and vision returned.
    # "Visible" alone lied on drive_ae (tiny ny≈0.85 blobs, range never
    # closed). Lock = chaseable samples that also grow in area / close.
    passed_gate1 = False
    t_pass = None
    for r in post:
        try:
            ag = int(r.get('active_gate')) if r.get('active_gate') not in (
                None, ''
            ) else None
        except (TypeError, ValueError):
            ag = None
        if ag is not None and ag >= 1:
            passed_gate1 = True
            t_pass = float(r['t'])
            break
    post_seen = 0
    post_chase = 0
    gate2_locked = False
    if t_pass is not None:
        after = [r for r in post if float(r['t']) >= t_pass + 0.5]
        post_seen = sum(1 for r in after if r.get('gate_nx') is not None)

        def _chaseable_row(r):
            nx = r.get('gate_nx')
            ny = r.get('gate_ny')
            if nx is None or ny is None:
                return False
            try:
                ny_f = float(ny)
                nx_f = float(nx)
                area = r.get('area_px')
                area_f = float(area) if area not in (None, '') else None
            except (TypeError, ValueError):
                return False
            if ny_f > 0.92:
                return False
            if ny_f > 0.78 and abs(nx_f) < 0.22:
                return False
            if ny_f > 0.72 and (area_f is None or area_f < 2200.0):
                return False
            if area_f is not None and area_f < 900.0:
                return False
            return True

        chase_rows = [r for r in after if _chaseable_row(r)]
        post_chase = len(chase_rows)
        area_after = [
            float(r['area_px']) for r in chase_rows
            if r.get('area_px') not in (None, '')
            and math.isfinite(float(r['area_px']))
            and float(r['area_px']) > 0.0
        ]
        area_grow = None
        if len(area_after) >= 6:
            n_win = max(3, len(area_after) // 5)
            a0 = float(statistics.median(area_after[:n_win]))
            a1 = float(statistics.median(area_after[-n_win:]))
            area_grow = a1 - a0
            print(f'gate2 bbox after     {a0:.0f} → {a1:.0f} px '
                  f'(Δ {area_grow:+.0f})')
        print(f'gate1 passed         yes (t≈{t_pass:.1f}s)')
        print(f'gate2 visible after  {post_seen}/{len(after)} samples '
              f'(t>{t_pass + 0.5:.1f}s)')
        print(f'gate2 chaseable      {post_chase}/{len(after)} samples')
        gate2_locked = bool(
            post_chase >= 12
            and area_grow is not None
            and area_grow >= 400.0
        )
        print(f'gate2 locked         '
              f'{"yes" if gate2_locked else "no"} '
              f'(need chaseable≥12 and area Δ≥+400)')
    else:
        print('gate1 passed         no')

    # Only range/bbox count. local-N lied on drive_b (+59 m while the
    # craft just climbed on the pad).
    closed = bool(
        (range_delta is not None and range_delta >= 1.5)
        or (area_delta is not None and area_delta >= 400.0)
    )
    climbed_away = bool(
        rate is not None and rate > 0.15
        and not closed
        and (range_delta is None or range_delta < 0.5)
    )
    if not seen:
        print('VERDICT: FAIL — no gate in image. Face gate 1 on the pad.')
        rc = 1
    elif len(pitched) < 0.6 * len(seen):
        print('VERDICT: FAIL — gate seen but drive rarely commanded pitch.')
        rc = 1
    elif climbed_away:
        print('VERDICT: FAIL — climbed away without closing range '
              '(lower HT / settle thrust, tip forward sooner).')
        rc = 1
    elif closed and passed_gate1 and gate2_locked:
        print('VERDICT: Pass — gate 1 cleared and gate 2 locked (closing).')
        rc = 0
    elif closed and passed_gate1:
        print('VERDICT: Pass — gate 1 cleared; gate 2 not yet locked '
              '(visible/chaseable without area growth — keep hunting).')
        rc = 0
    elif closed:
        print('VERDICT: Pass — range/bbox closing toward the gate.')
        rc = 0
    else:
        print('VERDICT: FAIL — not closing on the gate. Raise --lean-deg '
              f'(try {min(14.0, float(args.lean_deg) + 2.0):.0f}), lower '
              'HOVER_THRUST if still lofting, check RATE_SIGN_PITCH.')
        rc = 1
    print(f'\nCSV: {recorder.path}')
    return 1 if abort else rc


def run_yaw_align(args) -> int:
    """Phase 4.7 — level hover, yaw only from image nx (KALMAN_KP_YAW).

    Pad starts are usually centred, so the harness yaws open-loop briefly to
    create an offset, then closes the same yaw PID the planner uses.
    """
    import config
    from control.pid import PIDConfig, PIDController

    settle = float(args.settle_s)
    # Mild offset — 1.5s×0.35 rad/s yaws the gate clean off the FOV on
    # this airframe. Stay short so closed-loop still has an image lock.
    offset_s = 0.7
    close_t = settle + offset_s
    max_yaw = min(config.YAW_RATE_MAX_RAD_S, math.radians(25.0))
    yaw_pid = PIDController(PIDConfig(
        kp=config.KALMAN_KP_YAW, kd=0.0,
        output_min=-max_yaw, output_max=max_yaw,
    ))
    last_yaw = 0.0
    yaw_slew = math.radians(90.0)
    # Open-loop nudge. Do NOT pre-multiply RATE_SIGN_YAW — the controller
    # applies that sign on the way out (same as kalman_planner cmds).
    open_loop = 0.22

    def desired(elapsed, shared_data):
        nonlocal last_yaw
        if elapsed < settle:
            yaw_pid.reset()
            last_yaw = 0.0
            return 0.0, 0.0, 0.0
        if elapsed < close_t:
            yaw_pid.reset()
            last_yaw = open_loop
            return 0.0, 0.0, open_loop
        nx, _ny, _src = _image_gate_norm(shared_data)
        if nx is None:
            yaw_pid.reset()
            last_yaw = 0.0
            return 0.0, 0.0, 0.0
        # Same convention as kalman_planner: update(nx) → yaw_rate.
        yaw_rate = float(yaw_pid.update(float(nx), 1.0 / max(args.hz, 1.0)))
        max_step = yaw_slew * (1.0 / max(args.hz, 1.0))
        yaw_rate = float(max(last_yaw - max_step,
                             min(last_yaw + max_step, yaw_rate)))
        last_yaw = yaw_rate
        return 0.0, 0.0, yaw_rate

    recorder, abort, coverage = run_attitude_loop(
        args, desired, 'yaw_align', tilt_compensate=False,
    )
    if recorder is None:
        print(f'[FAIL] {abort}', flush=True)
        return 2
    rows = recorder.rows
    print(f'\n=== yaw align (KALMAN_KP_YAW={config.KALMAN_KP_YAW}; '
          f'open-loop {offset_s:.1f}s then close) ===')
    closed = [r for r in rows if r['t'] >= close_t]
    seen = [r for r in closed if r.get('gate_nx') is not None]
    if not seen:
        print('VERDICT: FAIL — no gate after open-loop offset. Re-run facing '
              'gate 1 on the pad.')
        print(f'\nCSV: {recorder.path}')
        return 1
    nx0 = abs(float(seen[0]['gate_nx']))
    nx_end = abs(float(seen[-1]['gate_nx']))
    # Worst |nx| in the first 0.5 s of closed loop (the offset we must kill).
    early = [r for r in seen if float(r['t']) <= close_t + 0.6]
    nx_peak = max((abs(float(r['gate_nx'])) for r in early), default=nx0)
    half_t = None
    for r in seen:
        if abs(float(r['gate_nx'])) <= 0.5 * max(nx_peak, 1e-3):
            half_t = float(r['t']) - close_t
            break
    yaw_cmds = [abs(float(r['yaw_rate_cmd'])) for r in seen
                if isinstance(r.get('yaw_rate_cmd'), float)]
    peak_yaw = max(yaw_cmds) if yaw_cmds else 0.0
    print(f'gate visible         {len(seen)}/{len(closed)} '
          f'(|nx| peak {nx_peak:.2f} → end {nx_end:.2f})')
    print(f'PnP coverage         {coverage * 100:.0f}%')
    print(f'half-error time      '
          f'{half_t:.2f}s' if half_t is not None else 'half-error time      —')
    print(f'peak |yaw_rate|      {math.degrees(peak_yaw):.1f} deg/s')
    if nx_peak < 0.10:
        print('VERDICT: FAIL — open-loop yaw did not offset the gate. Check '
              'RATE_SIGN_YAW / sim yaw authority.')
        rc = 1
    elif nx_end >= max(0.55, nx_peak + 0.15):
        print('VERDICT: FAIL — yaw drove the gate farther off. Flip '
              'RATE_SIGN_YAW or invert the yaw PID sign in kalman_planner.')
        rc = 1
    elif nx_end <= 0.15 or (half_t is not None and half_t <= 4.0):
        print('VERDICT: Pass — yaw closed image error. Keep KALMAN_KP_YAW.')
        rc = 0
    elif nx_end >= nx_peak * 0.85:
        print('VERDICT: FAIL — nx barely recovered. Raise KALMAN_KP_YAW '
              f'(try {config.KALMAN_KP_YAW + 0.2:.1f}).')
        rc = 1
    else:
        print('VERDICT: marginal — error reduced slowly. Consider raising '
              f'KALMAN_KP_YAW (try {config.KALMAN_KP_YAW + 0.15:.1f}).')
        rc = 0
    if abort and rc == 0:
        print(f'[NOTE] loop aborted ({abort}) — yaw verdict still stands.')
    print(f'\nCSV: {recorder.path}')
    return rc if rc == 0 else (1 if abort else rc)


def run_authority(args) -> int:
    """Phase 4.8 — step to max lean; confirm authority / rate ceiling."""
    import config
    settle = float(args.settle_s)
    amp_deg = float(
        args.max_lean_deg
        if args.max_lean_deg is not None
        else config.KALMAN_MAX_LEAN_DEG
    )
    amplitude = math.radians(amp_deg)

    def desired(elapsed, _sd):
        value = amplitude if elapsed >= settle else 0.0
        return (value, 0.0) if args.axis == 'roll' else (0.0, value)

    recorder, abort, _coverage = run_attitude_loop(
        args, desired, 'authority', tilt_compensate=True,
    )
    if recorder is None:
        print(f'[FAIL] {abort}', flush=True)
        return 2
    rows = recorder.rows
    print(f'\n=== authority step ({args.axis} {amp_deg:.1f} deg, '
          f'max_rate={config.KALMAN_MAX_RATE_RAD_S:.2f}) ===')
    post = [r for r in rows if r['t'] >= settle]
    key = 'roll' if args.axis == 'roll' else 'pitch'
    rate_key = 'roll_rate_cmd' if args.axis == 'roll' else 'pitch_rate_cmd'
    if not post:
        print('no samples')
        return 1
    # Rise 10–90% of commanded amplitude.
    target = amplitude
    lo, hi = 0.1 * target, 0.9 * target
    t10 = t90 = None
    peak = 0.0
    for r in post:
        meas = abs(float(r[key]))
        peak = max(peak, meas)
        if t10 is None and meas >= lo:
            t10 = float(r['t'])
        if t10 is not None and t90 is None and meas >= hi:
            t90 = float(r['t'])
    rise = (t90 - t10) if (t10 is not None and t90 is not None) else None
    sat_frac = 0.0
    rate_rows = [r for r in post if isinstance(r.get(rate_key), float)]
    if rate_rows:
        sat = sum(
            1 for r in rate_rows
            if abs(float(r[rate_key])) >= 0.95 * config.KALMAN_MAX_RATE_RAD_S
        )
        sat_frac = sat / len(rate_rows)
    final = abs(float(post[-1][key]))
    print(f'steady |{key}|         {math.degrees(final):.2f} deg '
          f'(cmd {amp_deg:.1f})')
    print(f'rise 10-90%          '
          f'{rise:.3f}s' if rise is not None else 'rise 10-90%          —')
    print(f'peak                 {math.degrees(peak):.2f} deg')
    print(f'rate saturation      {sat_frac * 100:.0f}% of post-settle samples')
    if final < 0.70 * amplitude:
        print('VERDICT: FAIL — never reached 70% of max lean. Raise '
              'KALMAN_MAX_RATE_RAD_S or KALMAN_KP_ATT.')
        rc = 1
    elif rise is not None and rise > 1.2:
        print('VERDICT: FAIL — max-lean rise >1.2s. Raise '
              'KALMAN_MAX_RATE_RAD_S (or KP_ATT).')
        rc = 1
    elif sat_frac > 0.85:
        print('VERDICT: Pass with note — rate pegged >85% of the time. '
              'Authority is rate-limited; OK if Phase 4 step was clean.')
        rc = 0
    else:
        print('VERDICT: Pass — max lean reachable with current rate ceiling.')
        rc = 0
    if abort and rc == 0:
        print(f'[NOTE] loop aborted ({abort}) after attitude measurement — '
              'authority verdict still stands.')
    print(f'\nCSV: {recorder.path}')
    return rc if rc == 0 else (1 if abort else rc)


def run_climb(args) -> int:
    """Phase 2.5 — level TAKEOFF_THRUST pulse; trim climb rate."""
    import config
    pulse = float(args.pulse_s)
    takeoff = float(config.TAKEOFF_THRUST)
    hover = float(config.HOVER_THRUST)

    def desired(elapsed, _sd):
        thrust = takeoff if elapsed < pulse else hover
        return 0.0, 0.0, 0.0, thrust

    recorder, abort, _coverage = run_attitude_loop(
        args, desired, 'climb', tilt_compensate=False,
    )
    if recorder is None:
        print(f'[FAIL] {abort}', flush=True)
        return 2
    rows = recorder.rows
    print(f'\n=== takeoff climb pulse ({pulse:.1f}s @ {takeoff:.3f}) ===')
    pulse_rows = [r for r in rows if 0.3 <= r['t'] <= pulse]
    rate, src = _post_settle_alt_rate(pulse_rows, settle_s=0.3)
    # _post_settle_alt_rate expects full rows with t relative to arm; reuse
    # a direct truth/ekf fit if helper returns None on short window.
    if rate is None and pulse_rows:
        zs = []
        for r in pulse_rows:
            z = r.get('d_truth_z')
            if z is None:
                z = r.get('d_ekf_z')
            if isinstance(z, float):
                zs.append((float(r['t']), float(z)))
        if len(zs) >= 4:
            dt = zs[-1][0] - zs[0][0]
            if dt > 0.2:
                # d_*_z is climb-up positive in the recorder (z0 - z).
                rate = (zs[-1][1] - zs[0][1]) / dt
                src = 'truth/ekf delta'
    print(f'climb rate           '
          f'{rate:+.3f} m/s ({src})' if rate is not None else
          'climb rate           —')
    if rate is None:
        print('VERDICT: FAIL — no altitude samples.')
        rc = 1
    elif rate < 0.20:
        print('VERDICT: FAIL — too weak. Raise TAKEOFF_THRUST '
              f'(try {takeoff + 0.01:.3f}).')
        rc = 1
    elif rate > 1.8:
        print('VERDICT: FAIL — too hot. Lower TAKEOFF_THRUST '
              f'(try {takeoff - 0.01:.3f}).')
        rc = 1
    else:
        print('VERDICT: Pass — takeoff climb in band (0.2–1.8 m/s).')
        rc = 0
    if abort and rc == 0:
        print(f'[NOTE] loop aborted ({abort}) after a usable pulse — '
              'rate verdict still stands.')
    print(f'\nCSV: {recorder.path}')
    # Prefer the pulse measurement over a late altitude abort.
    return rc if rc == 0 else (1 if abort else rc)


def run_acquire(args) -> int:
    """Short real-planner run: must acquire a gate and command pitch soon."""
    import config
    from setup import setup_components

    shared_data = {}
    components = setup_components(
        shared_data, int(time.time() * 1000),
        SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT,
    )
    controller = components['controller']
    planner = components['planner']
    recorder = Recorder(Path(args.csv) if args.csv
                        else default_csv_path('acquire'))
    deadline = float(args.deadline_s)

    print(f'[GAINS] KALMAN_KP_ATT={config.KALMAN_KP_ATT} '
          f'KALMAN_KD_ATT={config.KALMAN_KD_ATT} '
          f'HOVER_THRUST={config.HOVER_THRUST}', flush=True)
    print(f'[ACQUIRE] deadline={deadline:.1f}s for DUAL_PNP|yolo_fallback '
          f'+ nonzero desired_pitch', flush=True)
    print('[SIM] reset (command 31000) before arm...', flush=True)
    controller.send_sim_reset()
    time.sleep(max(0.5, float(getattr(config, 'SIM_RESET_SETTLE_S', 1.5))))
    hold_s = max(0.0, float(getattr(args, 'early_start_hold_s', 3.5) or 3.5))
    print(f'[SIM] early-start hold {hold_s:.1f}s before arm...', flush=True)
    time.sleep(hold_s)
    print('Arming...', flush=True)
    controller.arm()
    shared_data['flight_started'] = True

    period = 1.0 / max(args.hz, 1.0)
    started = time.monotonic()
    acquire_t = None
    abort = None
    if not args.quiet:
        print('\n    t   phase          source           n_sol  des_p   '
              'nx      range', flush=True)
    try:
        while True:
            now = time.monotonic()
            elapsed = now - started
            if args.seconds > 0 and elapsed >= args.seconds:
                break

            target = planner.compute_target(shared_data)
            shared_data['planner_target'] = target
            controller.update()

            path = shared_data.get('kalman_path') or {}
            dual = shared_data.get('dual_gate_pnp') or {}
            n_solved = int(dual.get('n_solved') or 0)
            source = str(path.get('source') or '')
            phase = str(path.get('phase') or '')
            des_pitch = _f(path.get('des_pitch'))
            if des_pitch is None:
                des_pitch = _f((target or {}).get('desired_pitch'))
            nx = _f(path.get('norm_x'))
            range_m = _f(path.get('range_m'))

            vision_ok = (
                n_solved >= 1
                or source == 'yolo_fallback'
                or str(
                    (shared_data.get('navigation') or {}).get('state') or ''
                ) == 'DUAL_PNP'
            )
            pitch_ok = des_pitch is not None and abs(des_pitch) > 1e-4
            if acquire_t is None and vision_ok and pitch_ok:
                acquire_t = elapsed

            roll, pitch, _, _ = read_attitude(shared_data)
            if max(abs(roll), abs(pitch)) > ABORT_LEAN_RAD:
                abort = f'lean {math.degrees(max(abs(roll), abs(pitch))):.0f} deg'
                break
            truth = truth_state(shared_data)
            z = None
            if truth is not None:
                z = truth.get('z')
            if z is None:
                z = _f((shared_data.get('position_ned') or {}).get('z'))
            # NED z negative-up; large negative climb → abort.
            if z is not None and z < -ABORT_ALTITUDE_M:
                abort = f'altitude z={z:.1f} m'
                break

            recorder.write({
                't': round(elapsed, 4),
                'phase': phase,
                'source': source,
                'n_solved': n_solved,
                'des_pitch': des_pitch,
                'norm_x': nx,
                'range_m': range_m,
                'vision_ok': int(vision_ok),
                'pitch_ok': int(pitch_ok),
                'acquired': int(acquire_t is not None),
                'hover_thrust': config.HOVER_THRUST,
                'kp_att': config.KALMAN_KP_ATT,
                'kd_att': config.KALMAN_KD_ATT,
            })
            if not args.quiet:
                print(
                    f'{elapsed:5.1f}  {phase:14s} {source:16s} '
                    f'{n_solved:5d}  {_fmt(des_pitch, "+6.3f")} '
                    f'{_fmt(nx, "+6.2f")} {_fmt(range_m, "6.2f")}',
                    flush=True,
                )
            if acquire_t is not None and elapsed >= max(deadline + 1.0, 4.0):
                # Acquired early — no need to keep flying the full window.
                if elapsed >= 5.0:
                    break
            time.sleep(period)
    except KeyboardInterrupt:
        print('\n[STOP] interrupted', flush=True)
    finally:
        controller.disarm()
        print('Disarmed.', flush=True)
        recorder.close()
        shutdown(components)

    print('\n=== gate acquire (real planner) ===')
    if abort:
        print(f'[ABORT] {abort}')
    if acquire_t is None:
        print(f'VERDICT: FAIL — no DUAL_PNP/yolo_fallback + pitch within '
              f'{args.seconds:.0f}s. Fix vision/planner before Phase 5.')
        rc = 1
    elif acquire_t <= deadline:
        print(f'VERDICT: Pass — acquired at t={acquire_t:.2f}s '
              f'(deadline {deadline:.1f}s).')
        rc = 0
    else:
        print(f'VERDICT: FAIL — acquired late at t={acquire_t:.2f}s '
              f'(deadline {deadline:.1f}s).')
        rc = 1
    print(f'\nCSV: {recorder.path}')
    return 1 if abort else rc


def run_step(args) -> int:
    amplitude = math.radians(args.amplitude_deg)
    settle = args.settle_s

    def desired(elapsed, _sd):
        value = amplitude if elapsed >= settle else 0.0
        return (value, 0.0) if args.axis == 'roll' else (0.0, value)

    recorder, abort, _coverage = run_attitude_loop(
        args, desired, f'step_{args.axis}'
    )
    if recorder is None:
        print(f'[FAIL] {abort}', flush=True)
        return 2
    rows = recorder.rows

    print(f'\n=== {args.axis} step response '
          f'({args.amplitude_deg:.1f} deg at t={settle:.1f}s) ===')
    key = 'roll' if args.axis == 'roll' else 'pitch'
    fb = rows[0].get('feedback', 'ekf') if rows else 'ekf'
    # Score the signal the loop actually closed on. With --feedback truth the
    # EKF can lag badly mid-step and inflate rise time even when the airframe
    # (truth) already meets the 0.6 s / 25% targets.
    measure_key = f'truth_{key}' if fb == 'truth' else key
    post = [r for r in rows if r['t'] >= settle]
    if len(post) < 5:
        print('not enough post-step samples — increase --seconds')
        print(f'\nCSV: {recorder.path}')
        return 1 if abort else 0

    t0 = post[0]['t']
    measured = [
        (r['t'] - t0, r[measure_key])
        for r in post
        if isinstance(r.get(measure_key), float)
    ]
    if len(measured) < 5:
        measured = [(r['t'] - t0, r[key]) for r in post]
        measure_key = key
    target = amplitude

    def crossing(fraction):
        want = fraction * target
        for t, v in measured:
            if (v >= want) if target > 0 else (v <= want):
                return t
        return None

    t10, t90 = crossing(0.10), crossing(0.90)
    peak = max((v for _, v in measured), key=abs)
    overshoot = (peak - target) / target * 100.0 if target else 0.0
    tail = measured[max(0, len(measured) - int(len(measured) * 0.25)):]
    steady = statistics.fmean(v for _, v in tail)
    steady_err = math.degrees(target - steady)

    print(f'commanded            {math.degrees(target):+.2f} deg')
    print(f'steady state         {math.degrees(steady):+.2f} deg  '
          f'(error {steady_err:+.2f} deg)')
    print(f'rise time 10-90%     '
          f'{f"{t90 - t10:.3f} s" if t10 is not None and t90 is not None else "never reached 90%"}')
    print(f'peak                 {math.degrees(peak):+.2f} deg  '
          f'(overshoot {overshoot:+.1f}%)')
    print(f'gains                kp={rows[0]["kp_att"]} kd={rows[0]["kd_att"]}')
    print(f'measured on          {measure_key}')
    print(f'feedback source       {fb}'
          + ('   <-- DIAGNOSTIC, not shippable' if fb == 'truth' else
             '    (what races)'))

    # If the sim gave truth, the same response measured externally separates
    # "my controller is mistuned" from "my attitude estimate is wrong".
    truth_key = f'truth_{key}'
    truth_post = [r[truth_key] for r in post
                  if isinstance(r.get(truth_key), float)]
    truth_warning = False
    if len(truth_post) >= 5:
        t_peak = max(truth_post, key=abs)
        t_tail = truth_post[max(0, len(truth_post) - len(truth_post) // 4):]
        t_steady = statistics.fmean(t_tail)
        t_overshoot = (t_peak - target) / target * 100.0 if target else 0.0
        bias = math.degrees(t_steady - steady)
        print('\n--- same step measured against ground truth ---')
        print(f'truth steady state   {math.degrees(t_steady):+.2f} deg  '
              f'(overshoot {t_overshoot:+.1f}%)')
        print(f'estimator bias       {bias:+.2f} deg '
              '(truth minus EKF at steady state)')
        if abs(bias) > 2.0:
            truth_warning = True
            print('  [!] The EKF attitude estimate is off by more than 2 deg. '
                  'The loop is holding the *estimate* at the setpoint, so the '
                  'airframe is actually leaning by this much extra. Fix the '
                  'estimator before chasing gains — no kp/kd can remove a '
                  'bias in the feedback signal.')
        elif abs(t_overshoot - overshoot) > 15.0:
            truth_warning = True
            print('  [!] Truth and estimate disagree on overshoot, so the '
                  'estimator is lagging or filtering the real motion. Tuned '
                  'gains will look better here than they fly.')
        else:
            print('  estimate tracks truth closely — gains found here are '
                  'trustworthy.')
    rise = (t90 - t10) if (t10 is not None and t90 is not None) else None
    print('\nguidance:')
    if truth_warning:
        print('  NOTE: the numbers below are measured on the EKF estimate, '
              'which ground truth just flagged as unreliable. Fix the '
              'estimator first — gains tuned against a bad feedback signal '
              'will not fly as measured.')
    if t90 is None:
        print('  never reached 90% — raise KALMAN_KP_ATT, or the rate clamp '
              'KALMAN_MAX_RATE_RAD_S is binding')
    elif overshoot > 25.0:
        print(f'  overshoot {overshoot:.0f}% is high — raise KALMAN_KD_ATT '
              f'(from {rows[0]["kd_att"]}), or lower KALMAN_KP_ATT')
    elif rise is not None and rise > SLOW_RISE_S:
        print(f'  rise {rise:.2f}s is sluggish for a {math.degrees(target):.0f} deg '
              f'step — raise KALMAN_KP_ATT (from {rows[0]["kp_att"]}). '
              'A slow attitude loop shows up as late gate alignment.')
    elif abs(steady_err) > 1.5:
        print('  steady-state offset >1.5 deg — this loop has no integrator '
              'by design; check HOVER_THRUST trim and lean-sign conventions '
              'before adding one')
    else:
        print(f'  rise {rise:.2f}s, overshoot {overshoot:+.0f}% — responsive '
              'with acceptable damping. Good operating point.')
    print(f'\nCSV: {recorder.path}')
    return 1 if abort else 0


# --------------------------------------------------------------------------
# assist — bounded image-chase (main.py default planner)
# --------------------------------------------------------------------------
def run_assist(args) -> int:
    """Short armed run of AssistImagePlanner (same as main.py FLIGHT_MODE=assist)."""
    import config
    from assist_planner import AssistImagePlanner
    from setup import setup_components

    shared_data = {}
    components = setup_components(
        shared_data, int(time.time() * 1000),
        SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT,
    )
    controller = components['controller']
    planner = AssistImagePlanner()
    shared_data['planner'] = planner
    recorder = Recorder(
        Path(args.csv) if args.csv else default_csv_path('assist')
    )

    print('', flush=True)
    print('=== ASSIST (image chase) ===', flush=True)
    print(
        f'  lean={getattr(config, "ASSIST_LEAN_DEG", 10)}°  '
        f'ny_aim={getattr(config, "ASSIST_NY_AIM", 0.12)}  '
        f'HT={config.HOVER_THRUST:.3f}  kp_att={config.KALMAN_KP_ATT}',
        flush=True,
    )
    print('  Full race client: python main.py  (FLIGHT_MODE=assist default)',
          flush=True)

    if bool(getattr(args, 'no_sim_reset', False)):
        time.sleep(1.0)
    else:
        print('[SIM] reset (command 31000) before arm...', flush=True)
        controller.send_sim_reset()
        time.sleep(max(0.5, float(getattr(config, 'SIM_RESET_SETTLE_S', 1.5))))

    ready_deadline = time.monotonic() + 45.0
    print('[PAD] waiting for YOLO/DUAL_PNP before arm...', flush=True)
    while time.monotonic() < ready_deadline:
        dual = shared_data.get('dual_gate_pnp') or {}
        det = shared_data.get('gate_detection') or {}
        if dual.get('gate1_body') is not None or (
            isinstance(det, dict) and det.get('center_px') is not None
        ):
            print('[PAD] ready', flush=True)
            break
        time.sleep(0.2)
    else:
        print('[FAIL] no gate in view', flush=True)
        shutdown(components)
        return 1

    hold_s = max(0.0, float(getattr(args, 'early_start_hold_s', 3.5) or 3.5))
    print(f'[SIM] early-start hold {hold_s:.1f}s before arm...', flush=True)
    time.sleep(hold_s)
    controller.arm()
    shared_data['flight_started'] = True

    period = 1.0 / max(args.hz, 1.0)
    started = time.monotonic()
    last_floor_reset = 0.0
    try:
        while True:
            now = time.monotonic()
            elapsed = now - started
            if args.seconds > 0 and elapsed >= args.seconds:
                break
            tgt = planner.compute_target(shared_data)
            controller.update()
            path = shared_data.get('kalman_path') or {}
            climbed = path.get('climbed')
            try:
                climbed_f = float(climbed) if climbed is not None else 0.0
            except (TypeError, ValueError):
                climbed_f = 0.0
            # Only after we actually left the pad (pad NED noise is ~±0.2 m).
            left_pad = bool(getattr(planner, '_left_pad', False))
            if (
                shared_data.get('flight_started')
                and left_pad
                and climbed_f < -0.45
                and (now - last_floor_reset) > 5.0
            ):
                print('[SIM] assist floor — reset/re-arm', flush=True)
                last_floor_reset = now
                shared_data['flight_started'] = False
                shared_data['vision_reset_episode'] = True
                try:
                    controller.disarm()
                except Exception:
                    pass
                controller.send_sim_reset()
                time.sleep(max(0.5, float(getattr(config, 'SIM_RESET_SETTLE_S', 1.5))))
                planner.reset_episode()
                shared_data['local_position_ned'] = None
                ready_deadline = time.monotonic() + 20.0
                while time.monotonic() < ready_deadline:
                    dual = shared_data.get('dual_gate_pnp') or {}
                    det = shared_data.get('gate_detection') or {}
                    if dual.get('gate1_body') is not None or (
                        isinstance(det, dict) and det.get('center_px') is not None
                    ):
                        break
                    time.sleep(0.2)
                time.sleep(1.0)
                controller.arm()
                shared_data['flight_started'] = True
                print('[SIM] assist re-armed', flush=True)
                continue
            row = {
                't': round(elapsed, 3),
                'phase': path.get('phase'),
                'source': path.get('source'),
                'nx': path.get('norm_x'),
                'ny': path.get('norm_y'),
                'thrust': tgt.get('thrust'),
                'des_pitch': tgt.get('desired_pitch'),
                'yaw_rate': tgt.get('yaw_rate'),
                'climbed': path.get('climbed'),
                'range_m': path.get('range_m'),
            }
            recorder.write(row)
            if not args.quiet and int(elapsed * args.hz) % max(1, int(args.hz)) == 0:
                print(
                    f"{elapsed:5.1f} {path.get('phase')} "
                    f"nx={path.get('norm_x')} thr={tgt.get('thrust')}",
                    flush=True,
                )
            time.sleep(period)
    except KeyboardInterrupt:
        print('\n[STOP] interrupted', flush=True)
    finally:
        try:
            controller.disarm()
        except Exception:
            pass
        recorder.close()
        shutdown(components)
    print(f'\nCSV: {recorder.path}')
    print('Events/telem also under logs/ if Logger started via setup.')
    return 0


# --------------------------------------------------------------------------
# manual — auto-stabilize teleop
# --------------------------------------------------------------------------
def run_manual(args) -> int:
    """ANGLE self-level + hover trim; human commands lean / yaw / thrust."""
    import config
    from control.pid import PIDConfig, PIDController
    from setup import setup_components

    shared_data = {}
    components = setup_components(
        shared_data, int(time.time() * 1000),
        SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT,
    )
    controller = components['controller']
    recorder = Recorder(
        Path(args.csv) if args.csv else default_csv_path('manual')
    )

    lean_rad = math.radians(float(getattr(args, 'lean_deg', 10.0)))
    yaw_rate_cmd = math.radians(float(getattr(args, 'yaw_rate_deg', 25.0)))
    thrust_step = float(getattr(args, 'thrust_step', 0.004))
    max_rate = config.KALMAN_MAX_RATE_RAD_S
    roll_pid = PIDController(PIDConfig(
        kp=config.KALMAN_KP_ATT, kd=config.KALMAN_KD_ATT,
        output_min=-max_rate, output_max=max_rate,
    ))
    pitch_pid = PIDController(PIDConfig(
        kp=config.KALMAN_KP_ATT, kd=config.KALMAN_KD_ATT,
        output_min=-max_rate, output_max=max_rate,
    ))

    print('', flush=True)
    print('=== MANUAL STABILIZE ===', flush=True)
    print('  Focus THIS console for keys (sim window can stay visible).',
          flush=True)
    print('  Hold-to-fly (release = level / hover):', flush=True)
    print('  W/S or ↑/↓   pitch forward / back', flush=True)
    print('  A/D or ←/→   roll left / right', flush=True)
    print('  Q/E          yaw left / right', flush=True)
    print('  R/F          climb / sink (while held)', flush=True)
    print('  Space        force level now', flush=True)
    print('  Esc / X      disarm and quit', flush=True)
    print(
        f'  lean={math.degrees(lean_rad):.0f}°  '
        f'yaw={math.degrees(yaw_rate_cmd):.0f}°/s  '
        f'HT={config.HOVER_THRUST:.3f}  '
        f'kp_att={config.KALMAN_KP_ATT}  '
        f'lean_boost={getattr(config, "LEAN_THRUST_BOOST", 0.0)}',
        flush=True,
    )
    print('', flush=True)

    if bool(getattr(args, 'no_sim_reset', False)):
        print('[SIM] skip reset (--no-sim-reset)', flush=True)
        time.sleep(1.0)
    else:
        print('[SIM] reset (command 31000) before arm...', flush=True)
        controller.send_sim_reset()
        time.sleep(max(0.5, float(getattr(config, 'SIM_RESET_SETTLE_S', 1.5))))

    if bool(getattr(args, 'wait_pad', False)):
        print('[PAD] waiting for DUAL_PNP gate1...', flush=True)
        pad_deadline = time.monotonic() + 45.0
        while time.monotonic() < pad_deadline:
            dual = shared_data.get('dual_gate_pnp') or {}
            body = dual.get('gate1_body')
            n = int(dual.get('n_solved') or 0)
            if body is not None and n >= 1:
                rng = dual.get('gate1_range_m')
                print(f'[PAD] ready n_solved={n} range={rng}', flush=True)
                break
            time.sleep(0.05)
        else:
            print('[PAD] timeout — arming anyway', flush=True)

    hold_s = max(0.0, float(getattr(args, 'early_start_hold_s', 3.5) or 3.5))
    print(f'[SIM] early-start hold {hold_s:.1f}s before arm...', flush=True)
    time.sleep(hold_s)
    print('Arming...', flush=True)
    controller.arm()
    shared_data['flight_started'] = True

    hold_state: dict = {}
    last_t = None
    period = 1.0 / max(args.hz, 1.0)
    started = time.monotonic()
    if not args.quiet:
        print(
            '\n    t   climb  roll  pitch   yaw_r   thr   '
            'des_r  des_p   g1_rng  n',
            flush=True,
        )

    try:
        while True:
            now = time.monotonic()
            elapsed = now - started
            if args.seconds > 0 and elapsed >= args.seconds:
                print('\n[STOP] time limit', flush=True)
                break

            (
                des_roll, des_pitch, yaw_rate, thrust_delta, quit_req
            ) = _poll_manual_controls(
                hold_state,
                lean_rad=lean_rad,
                yaw_rate_cmd=yaw_rate_cmd,
                thrust_step=thrust_step,
                now=now,
            )
            if quit_req:
                print('\n[STOP] quit key', flush=True)
                break

            dt = period if last_t is None else max(1e-3, now - last_t)
            last_t = now
            roll, pitch, _, _ = read_attitude(shared_data)
            roll_rate = roll_pid.update(des_roll - roll, dt)
            pitch_rate = pitch_pid.update(des_pitch - pitch, dt)
            lean_boost = float(getattr(config, 'LEAN_THRUST_BOOST', 0.0) or 0.0)
            # Optional CLI override already exported via LEAN_THRUST_BOOST.
            thrust = _tilt_compensated_thrust(
                config.HOVER_THRUST, des_roll, des_pitch, lean_boost=lean_boost,
            )
            # Wider clamp than race planner — manual R/F needs authority.
            thrust = float(max(0.18, min(0.36, thrust + thrust_delta)))

            shared_data['planner_target'] = {
                'kalman': True,
                'roll_rate': roll_rate,
                'pitch_rate': pitch_rate,
                'yaw_rate': yaw_rate,
                'thrust': thrust,
                'desired_roll': des_roll,
                'desired_pitch': des_pitch,
            }
            shared_data['planner_mode'] = 'kalman_dual_gate'
            controller.update()

            dual = shared_data.get('dual_gate_pnp') or {}
            climb = _climb_estimate(shared_data)
            row = {
                't': round(elapsed, 3),
                'climb_m': climb,
                'roll': roll,
                'pitch': pitch,
                'des_roll': des_roll,
                'des_pitch': des_pitch,
                'yaw_rate': yaw_rate,
                'thrust': thrust,
                'thrust_delta': thrust_delta,
                'gate1_range_m': _f(dual.get('gate1_range_m')),
                'n_solved': int(dual.get('n_solved') or 0),
            }
            recorder.write(row)
            if not args.quiet:
                print(
                    f"{elapsed:5.1f} {_fmt(climb, '6.2f')}"
                    f" {_fmt(roll, '6.3f')} {_fmt(pitch, '6.3f')}"
                    f" {_fmt(yaw_rate, '6.3f')} {_fmt(thrust, '5.3f')}"
                    f" {_fmt(des_roll, '6.3f')} {_fmt(des_pitch, '6.3f')}"
                    f" {_fmt(row['gate1_range_m'])} {row['n_solved']:2d}",
                    flush=True,
                )
            time.sleep(period)
    except KeyboardInterrupt:
        print('\n[STOP] interrupted', flush=True)
    finally:
        try:
            # Level + hover before disarm so the craft doesn't drop armed.
            shared_data['planner_target'] = {
                'kalman': True,
                'roll_rate': 0.0,
                'pitch_rate': 0.0,
                'yaw_rate': 0.0,
                'thrust': float(config.HOVER_THRUST),
                'desired_roll': 0.0,
                'desired_pitch': 0.0,
            }
            for _ in range(8):
                controller.update()
                time.sleep(0.02)
            controller.disarm()
            print('Disarmed.', flush=True)
        except Exception:
            pass
        recorder.close()
        shutdown(components)

    print(f'\nCSV: {recorder.path}')
    return 0


def _climb_estimate(shared_data) -> float | None:
    """Best-effort metres above arm for the manual HUD."""
    ekf = shared_data.get('ekf_state') or {}
    pos = ekf.get('position_ned') or [None, None, None]
    z = _f(pos[2] if len(pos) > 2 else None)
    if z is not None:
        # NED z down — climb is -z relative to arm≈0.
        return max(0.0, -z)
    local = shared_data.get('local_position_ned') or {}
    z = _f(local.get('z'))
    if z is not None:
        return max(0.0, -z)
    return None


def _prewarm_yolo() -> None:
    """Load + warm YOLO before waiting on the sim heartbeat.

    `_run_tune.ps1` launches FlightSim first; this warm window is when you
    should log in and start a fresh race so arming does not early-start DQ.
    """
    print('', flush=True)
    print('>>> YOLO pre-warm starting. In the FlightSim window NOW:', flush=True)
    print('>>>   log in and START A RACE (stay on the pad).', flush=True)
    print('', flush=True)
    try:
        from vision_rx import create_gate_detector
        create_gate_detector()
        print('[VISION] pre-warm detector constructed', flush=True)
    except Exception as exc:
        print(f'[VISION] pre-warm skipped: {exc}', flush=True)


def main() -> int:
    args = build_parser().parse_args()
    applied = export_gain_overrides(args)
    if applied:
        print('[OVERRIDE] ' + '  '.join(f'{k}={v}' for k, v in applied.items()),
              flush=True)
    # Armed modes: warm YOLO before heartbeat so the sim can be started
    # during that load (fresh race → no early-start DQ).
    if args.mode in (
            'hover', 'step', 'lean-hover', 'crawl', 'drive', 'yaw-align',
            'authority', 'climb', 'acquire', 'manual', 'assist',
    ) or (args.mode == 'localize' and bool(getattr(args, 'teleop', False))):
        _prewarm_yolo()
    if args.mode == 'localize':
        return run_localize(args)
    if args.mode == 'hover':
        return run_hover(args)
    if args.mode == 'step':
        return run_step(args)
    if args.mode == 'lean-hover':
        return run_lean_hover(args)
    if args.mode == 'crawl':
        return run_crawl(args)
    if args.mode == 'drive':
        return run_drive(args)
    if args.mode == 'yaw-align':
        return run_yaw_align(args)
    if args.mode == 'authority':
        return run_authority(args)
    if args.mode == 'climb':
        return run_climb(args)
    if args.mode == 'acquire':
        return run_acquire(args)
    if args.mode == 'manual':
        return run_manual(args)
    if args.mode == 'assist':
        return run_assist(args)
    return 2


if __name__ == '__main__':
    sys.exit(main())
