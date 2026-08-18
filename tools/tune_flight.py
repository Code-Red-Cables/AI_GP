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

  policy      Autonomous timed flyer. Policy owns the sticks from arm to
              quit. No gamepad, no H/T. Same path as tools/run_policy.py.
              Esc / Ctrl+C disarms.

  coach       HG-DAgger intervention harness. Policy flies; press H the
              instant it looks wrong (human takes the proven MANUAL stick);
              press T to return control. Every telem row is tagged with
              control_authority=policy|human and an intervention_id.
              Training rounds only — never use for timed evaluation.

  pilot       Manual-first: you fly with YOLO still tracking. Console shows
              LOCK when a gate is locked. Press T to hand that gate to
              assist; press H to take the sticks again. Esc/X quits.
              Use --pure for stick-only (same as `fly`).

              Remember-path (hybrid):
                --capture [PATH]           record EXACT stick cmds vs time
                --replay PATH              play that timeline open-loop
                --keep-until-gate N        replay keys through gate N
                --assist-after-gate N      then ASSIST (closed-loop next gate)
                --human-after-gate M       then HUMAN after gate M (default N+1)

  fly         Pure stick flying — no YOLO/pose, no assist, no AHRS/level
              attitude corrections. Same pad plant as pilot (engage on stick,
              Y reset, practice tapes). ANGLE mode (self-levels toward stick
              lean).

  acro        True rate-mode stick flying (like a normal FPV acro drone).
              Sticks command body rates; center stick stops rotation but does
              NOT self-level. No lean/angle caps. Vision records observe-only
              reference data; it never changes the controls or EKF.

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
  python tools/tune_flight.py policy              # autonomous, no human input
  python tools/tune_flight.py policy --panel
  python tools/tune_flight.py pilot
  python tools/tune_flight.py fly                 # stick only, no vision/assist
  python tools/tune_flight.py acro                # rate mode, no angle limits
  python tools/tune_flight.py pilot --pure        # same as fly
  python tools/tune_flight.py pilot --capture
  python tools/tune_flight.py pilot --replay captured_controls.json --keep-until-gate 1
  python tools/tune_flight.py pilot --replay captured_controls_g2_locked.json \\
      --keep-until-gate 1 --assist-after-gate 1   # keys→G1, assist G2, human G3+
  python main.py   # default FLIGHT_MODE=assist

Every run appends a CSV under logs/tuning/ for offline comparison.

On the OLD (VQ1) sim, ATTITUDE / LOCAL_POSITION_NED / ODOMETRY are scoring
only — never fed into the control loop. VQ2 omits them; gains still transfer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

# Status text uses arrows and degree symbols. A legacy Windows stdout encoding
# must not turn a help/banner print into a pre-flight crash.
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, 'reconfigure', None)
    if callable(_reconfigure):
        try:
            _reconfigure(errors='replace')
        except (OSError, ValueError):
            pass

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
            '--early-start-hold-s', type=float, default=None,
            help='seconds from sim-reset until arm (match ~3s countdown; '
                 'default EARLY_START_HOLD_S=3.55). Not stacked on settle.',
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
        '--yaw-rate-deg', type=float, default=35.0,
        help='yaw rate while Q/E held (default 35, same as manual)',
    )
    p_loc.add_argument(
        '--climb-rate', type=float, default=0.6,
        help='climb/sink RATE in m/s while R/F held (default 0.6, same as manual)',
    )
    p_loc.add_argument('--climb-auth', type=float, default=0.08)
    p_loc.add_argument('--climb-kp', type=float, default=None)
    p_loc.add_argument('--climb-ki', type=float, default=None)
    p_loc.add_argument('--open-loop-thrust', action='store_true')
    p_loc.add_argument(
        '--thrust-step', type=float, default=0.022,
        help='open-loop collective offset (only with --open-loop-thrust)',
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
        '--lean-deg', type=float, default=None,
        help='roll/pitch lean while held (default MANUAL_LEAN_DEG=14)',
    )
    p_man.add_argument(
        '--yaw-rate-deg', type=float, default=None,
        help='yaw rate while Q/E held (default MANUAL_YAW_RATE_DEG=40)',
    )
    p_man.add_argument(
        '--climb-rate', type=float, default=None,
        help='climb/sink RATE m/s (opts into rate-hold; default is open-loop MANUAL stick)',
    )
    p_man.add_argument(
        '--climb-auth', type=float, default=None,
        help='max thrust offset the rate loop may command (default PILOT_CLIMB_AUTH=0.18)',
    )
    p_man.add_argument('--climb-kp', type=float, default=None)
    p_man.add_argument('--climb-ki', type=float, default=None)
    p_man.add_argument(
        '--open-loop-thrust', action='store_true',
        help='force open-loop R/F thrust offsets (default for seed laps)',
    )
    p_man.add_argument(
        '--thrust-step', type=float, default=None,
        help='open-loop climb thrust offset (default MANUAL_THRUST_STEP=0.028)',
    )
    p_man.add_argument(
        '--capture', nargs='?', const='', default=None, metavar='PATH',
        help='record waypoints: M marks pose; also samples at SPLINE_CAPTURE_HZ. '
             'Saved on exit (default SPLINE_CAPTURE_PATH).',
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
    p_man.add_argument(
        '--slow-mo', action='store_true',
        help='start with client slow-mo ON (match Cheat Engine / DxWnd). '
             'Also scales telem LOG_HZ so H=32 still spans ~0.64 s of sim time.',
    )
    p_man.add_argument(
        '--slow-mo-scale', type=float, default=None,
        help='client/CE slow-mo factor (default PILOT_SLOW_MO_SCALE; use 0.2)',
    )

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

    p_coach = sub.add_parser(
        'coach',
        help='HG-DAgger: policy flies, H=intervene, T=return (authority logged)',
    )
    common(p_coach, 0.0)
    p_coach.add_argument(
        '--weights', default=None,
        help='policy checkpoint (default POLICY_WEIGHTS / models/policy.pt)',
    )
    p_coach.add_argument(
        '--planner', choices=('policy', 'race', 'assist', 'kalman'),
        default='policy',
        help='which autopilot the harness hands control to. Use this to test '
             'an untried controller with a human takeover available.',
    )
    p_coach.add_argument(
        '--angle', action='store_true',
        help='take over in ANGLE (self-levelling) mode instead of ACRO. '
             'Default is ACRO, matching how the seed laps were flown.',
    )
    p_coach.add_argument(
        '--roll-rate-deg', type=float, default=None,
        help='full-stick roll rate °/s in acro (default ACRO_ROLL_RATE_DEG)',
    )
    p_coach.add_argument(
        '--pitch-rate-deg', type=float, default=None,
        help='full-stick pitch rate °/s in acro (default ACRO_PITCH_RATE_DEG)',
    )
    p_coach.add_argument(
        '--yaw-rate-deg', type=float, default=None,
        help='full-stick yaw rate °/s (default ACRO_YAW_RATE_DEG)',
    )
    p_coach.add_argument('--hover-thrust', type=float, default=None)
    p_coach.add_argument('--kp-att', type=float, default=None)
    p_coach.add_argument('--kd-att', type=float, default=None)
    p_coach.add_argument('--lean-boost', type=float, default=None)
    p_coach.add_argument(
        '--start-human', action='store_true',
        help='begin in human mode (seed a takeoff, then T to hand to policy)',
    )
    p_coach.add_argument(
        '--panel', action='store_true',
        help='live window: camera frame plus the exact observation vector fed '
             'to the network, the attitude source in use, and the command out',
    )
    p_coach.add_argument(
        '--panel-scale', type=float, default=1.0,
        help='scale factor for the --panel window',
    )
    p_coach.add_argument(
        '--gatenet', action='store_true',
        help='show GateNet 4 inner corners on the panel instead of YOLO '
             '(observe-only; YOLO still feeds the policy unless --detector gatenet)',
    )
    p_coach.add_argument(
        '--detector', choices=('yolo_pose', 'gatenet', 'hsv', 'yolo_hybrid'),
        default=None,
        help='which detector feeds the policy. Default yolo_pose. '
             '"gatenet" flies on the 4 inner corners only.',
    )
    p_coach.add_argument(
        '--save-frames', nargs='?', const=1.0, default=None, type=float,
        metavar='SEC',
        help='save a raw camera JPEG every SEC seconds (default 1), including '
             'misses, under frames/run_<timestamp>/ for labelling',
    )

    p_policy = sub.add_parser(
        'policy',
        help='autonomous policy flyer — no gamepad, no H/T (timed path)',
    )
    common(p_policy, 0.0)
    from tools.run_policy import add_policy_flags
    add_policy_flags(p_policy)

    p_pilot = sub.add_parser(
        'pilot',
        help='manual-first: T=auto on LOCK, H=human again (YOLO stays on)',
    )
    common(p_pilot, 0.0)
    p_pilot.add_argument(
        '--lean-deg', type=float, default=None,
        help='roll/pitch lean while a key/stick is held (default PILOT_LEAN_DEG=38)',
    )
    p_pilot.add_argument(
        '--yaw-rate-deg', type=float, default=None,
        help='yaw rate while Q/E / R-stick (default PILOT_YAW_RATE_DEG=85)',
    )
    p_pilot.add_argument(
        '--climb-rate', type=float, default=None,
        help='climb/sink RATE in m/s (default PILOT_CLIMB_RATE=2.2)',
    )
    p_pilot.add_argument(
        '--climb-auth', type=float, default=None,
        help='max thrust offset the rate loop may command (default PILOT_CLIMB_AUTH=0.18)',
    )
    p_pilot.add_argument('--climb-kp', type=float, default=None)
    p_pilot.add_argument('--climb-ki', type=float, default=None)
    p_pilot.add_argument(
        '--open-loop-thrust', action='store_true',
        help='R/F is a raw thrust offset (coasts on release)',
    )
    p_pilot.add_argument(
        '--thrust-step', type=float, default=0.022,
        help='open-loop collective offset while R/F held (with --open-loop-thrust)',
    )
    p_pilot.add_argument(
        '--lock-area', type=float, default=1200.0,
        help='min YOLO area_px for LOCK (default 1200)',
    )
    p_pilot.add_argument('--hover-thrust', type=float, default=None)
    p_pilot.add_argument('--kp-att', type=float, default=None)
    p_pilot.add_argument('--kd-att', type=float, default=None)
    p_pilot.add_argument('--max-rate', type=float, default=None)
    p_pilot.add_argument('--lean-boost', type=float, default=None)
    p_pilot.add_argument(
        '--capture', nargs='?', const='', default=None, metavar='PATH',
        help='remember-path: record which flight keys are held and for how long. '
             'Default REMEMBER_PATH (captured_controls.json). Press K to keep.',
    )
    p_pilot.add_argument(
        '--replay', default=None, metavar='PATH',
        help='replay recorded key presses, then hand off sticks',
    )
    p_pilot.add_argument(
        '--replay-attitude', default=None, metavar='PATH',
        help='replay an attitude tape (des_roll/pitch + yaw_rate + thrust) '
             'exported from telem via tools/export_attitude_tape.py',
    )
    p_pilot.add_argument(
        '--keep-until-gate', type=int, default=None, metavar='N',
        help='with --replay: play timeline through gate N, then HUMAN '
             '(or ASSIST if --assist-after-gate is set). '
             'Raise N later to extend the remembered segment.',
    )
    p_pilot.add_argument(
        '--assist-after-gate', type=int, default=None, metavar='N',
        help='with --replay or --replay-attitude: after real GATE N clears, '
             'hand REPLAY→ASSIST (closed-loop vision flight). Typical: '
             '--assist-after-gate 1 --human-after-gate 17.',
    )
    p_pilot.add_argument(
        '--human-after-gate', type=int, default=None, metavar='M',
        help='with --assist-after-gate: after real GATE M clears, ASSIST→HUMAN '
             'so you can fly/append keys (default: assist-after-gate+1; '
             'use 17 to keep assist through the whole course).',
    )
    p_pilot.add_argument(
        '--practice-from-gate', type=int, default=None, metavar='N',
        help='replay your best saved PAD attitude through GATE N, then '
             'HUMAN (you fly gate N+1 onward). Example: after saving '
             'through gate 3, use --practice-from-gate 3. '
             'Auto-saves under practice/ (no mid-course teleport).',
    )
    p_pilot.add_argument(
        '--list-practice', action='store_true',
        help='print saved practice checkpoints and exit',
    )
    p_pilot.add_argument(
        '--no-practice-save', action='store_true',
        help='disable auto-saving faster through-gate checkpoints',
    )
    p_pilot.add_argument(
        '--slow-mo', action='store_true',
        help='start with client slow-mo ON (PILOT_SLOW_MO_SCALE, default 0.77). '
             'Toggle anytime with O or D-pad ↓. Match Cheat Engine / DxWnd '
             'to the same factor or tapes desync.',
    )
    p_pilot.add_argument(
        '--pure', action='store_true',
        help='stick-only: no YOLO/pose, no assist, no AHRS/level corrections '
             '(same as the `fly` mode)',
    )
    p_fly = sub.add_parser(
        'fly',
        help='pure stick flying — no vision/pose/assist/attitude corrections',
    )
    common(p_fly, 0.0)
    p_fly.add_argument(
        '--lean-deg', type=float, default=None,
        help='roll/pitch lean while a key/stick is held (default PILOT_LEAN_DEG)',
    )
    p_fly.add_argument(
        '--yaw-rate-deg', type=float, default=None,
        help='yaw rate while Q/E / R-stick (default PILOT_YAW_RATE_DEG)',
    )
    p_fly.add_argument(
        '--climb-rate', type=float, default=None,
        help='climb/sink RATE in m/s (default PILOT_CLIMB_RATE)',
    )
    p_fly.add_argument(
        '--climb-auth', type=float, default=None,
        help='max thrust offset the rate loop may command',
    )
    p_fly.add_argument('--climb-kp', type=float, default=None)
    p_fly.add_argument('--climb-ki', type=float, default=None)
    p_fly.add_argument(
        '--open-loop-thrust', action='store_true',
        help='R/F is a raw thrust offset (coasts on release)',
    )
    p_fly.add_argument(
        '--thrust-step', type=float, default=0.022,
        help='open-loop collective offset while R/F held (with --open-loop-thrust)',
    )
    p_fly.add_argument('--hover-thrust', type=float, default=None)
    p_fly.add_argument('--kp-att', type=float, default=None)
    p_fly.add_argument('--kd-att', type=float, default=None)
    p_fly.add_argument('--max-rate', type=float, default=None)
    p_fly.add_argument('--lean-boost', type=float, default=None)
    p_fly.add_argument(
        '--no-practice-save', action='store_true',
        help='disable auto-saving practice checkpoints / run archives',
    )
    p_fly.add_argument(
        '--replay-attitude', default=None, metavar='PATH',
        help='replay a saved attitude tape (prefer tools/replay_attitude.py)',
    )
    p_fly.add_argument(
        '--slow-mo', action='store_true',
        help='start with client slow-mo ON (toggle with O / D-pad ↓)',
    )

    p_acro = sub.add_parser(
        'acro',
        help='rate-mode stick flying — no angle limits / no self-level '
             '(like a normal acro drone)',
    )
    common(p_acro, 0.0)
    p_acro.add_argument(
        '--roll-rate-deg', type=float, default=None,
        help='full-stick roll rate °/s (default ACRO_ROLL_RATE_DEG)',
    )
    p_acro.add_argument(
        '--pitch-rate-deg', type=float, default=None,
        help='full-stick pitch rate °/s (default ACRO_PITCH_RATE_DEG)',
    )
    p_acro.add_argument(
        '--yaw-rate-deg', type=float, default=None,
        help='full-stick yaw rate °/s (default ACRO_YAW_RATE_DEG)',
    )
    p_acro.add_argument(
        '--climb-rate', type=float, default=None,
        help='climb/sink RATE in m/s (default PILOT_CLIMB_RATE)',
    )
    p_acro.add_argument(
        '--climb-auth', type=float, default=None,
        help='max thrust offset the collective stick may command',
    )
    p_acro.add_argument('--climb-kp', type=float, default=None)
    p_acro.add_argument('--climb-ki', type=float, default=None)
    p_acro.add_argument(
        '--open-loop-thrust', action='store_true',
        help='R/F is a raw thrust offset (coasts on release)',
    )
    p_acro.add_argument(
        '--thrust-step', type=float, default=0.022,
        help='open-loop collective offset while R/F held (with --open-loop-thrust)',
    )
    p_acro.add_argument('--hover-thrust', type=float, default=None)
    p_acro.add_argument(
        '--no-practice-save', action='store_true',
        help='disable auto-saving practice checkpoints / run archives',
    )
    p_acro.add_argument(
        '--no-vision', action='store_true',
        help='disable acro observe-only YOLO/PnP recording (vision is on by '
             'default and never feeds the controls or EKF)',
    )
    p_acro.add_argument(
        '--slow-mo', action='store_true',
        help='start with client slow-mo ON (toggle with O / D-pad ↓)',
    )
    p_acro.add_argument(
        '--slow-mo-scale', type=float, default=None,
        help='client/CE slow-mo factor (default PILOT_SLOW_MO_SCALE; use 0.2)',
    )

    p_prac = sub.add_parser(
        'practice',
        help='list / manage practice gate checkpoints',
    )
    p_prac.add_argument(
        'action', nargs='?', default='list', choices=('list',),
        help='list saved through-gate checkpoints (default)',
    )
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
            'yaw-align', 'authority', 'manual', 'assist', 'coach',
            'policy', 'pilot', 'fly',
    ) or (
        getattr(args, 'mode', None) == 'localize'
        and bool(getattr(args, 'teleop', False))
    ):
        os.environ['TAKEOFF_DURATION_S'] = '0'
        applied['TAKEOFF_DURATION_S'] = 0.0
    if getattr(args, 'mode', None) in (
            'acquire', 'manual', 'assist', 'coach', 'policy', 'pilot', 'fly',
    ):
        os.environ.setdefault('CRASH_USE_SIM_ODOMETRY', '0')
        applied.setdefault('CRASH_USE_SIM_ODOMETRY', 0.0)
    if getattr(args, 'mode', None) in ('assist', 'pilot'):
        # `fly` / `pilot --pure` stay off assist planner defaults.
        if not bool(getattr(args, 'pure', False)):
            os.environ['FLIGHT_MODE'] = 'assist'
            applied['FLIGHT_MODE'] = 'assist'
    if getattr(args, 'mode', None) == 'policy':
        from tools.run_policy import apply_flight_env
        applied.update(apply_flight_env(args))
    if getattr(args, 'mode', None) == 'coach':
        which = str(getattr(args, 'planner', 'policy') or 'policy')
        os.environ['FLIGHT_MODE'] = which
        applied['FLIGHT_MODE'] = which
        weights = getattr(args, 'weights', None)
        if weights:
            os.environ['POLICY_WEIGHTS'] = str(weights)
            applied['POLICY_WEIGHTS'] = str(weights)
        if which == 'race':
            # The classical stack solves gate pose itself from the keypoints;
            # it wants the detector, not the PnP/EKF path.
            os.environ.setdefault('EKF_USE_PNP', '0')
        # Match the acro plant the seed laps were flown on, and strip
        # everything the policy does not consume. It reads raw keypoints, so
        # the dual-gate PnP solve, the PnP-fed EKF and the debug window are
        # pure overhead that also steal time from the control loop.
        os.environ['EKF_USE_PNP'] = '0'
        os.environ['VISION_DISPLAY'] = '0'
        os.environ['TAKEOFF_DURATION_S'] = '0'
        applied['EKF_USE_PNP'] = 0.0
        applied['VISION_DISPLAY'] = 0.0
        os.environ.setdefault('PILOT_PAD_SOFT_GAIN', '1.0')
        os.environ.setdefault('PILOT_PAD_SOFT_YAW', '1.0')
        os.environ.setdefault('PILOT_PAD_SOFT_THRUST', '1.0')
        if bool(getattr(args, 'gatenet', False)):
            os.environ['GATENET_ENABLED'] = '1'
            applied['GATENET_ENABLED'] = 1.0
        detector = getattr(args, 'detector', None)
        if detector:
            os.environ['GATE_DETECTOR_BACKEND'] = str(detector)
            applied['GATE_DETECTOR_BACKEND'] = str(detector)
            if detector == 'gatenet':
                os.environ['GATENET_ENABLED'] = '1'
                applied['GATENET_ENABLED'] = 1.0
        save_frames = getattr(args, 'save_frames', None)
        if save_frames is not None:
            interval = max(0.05, float(save_frames))
            os.environ['GATE_FRAME_CAPTURE'] = '1'
            os.environ['GATE_FRAME_CAPTURE_INTERVAL_S'] = repr(interval)
            applied['GATE_FRAME_CAPTURE'] = 1.0
            applied['GATE_FRAME_CAPTURE_INTERVAL_S'] = interval
    if getattr(args, 'mode', None) in ('manual', 'acro'):
        # Seed laps need YOLO keypoints in the full telem logger.
        os.environ.setdefault('GATE_DETECTOR_BACKEND', 'yolo_pose')
        scale = getattr(args, 'slow_mo_scale', None)
        if scale is not None:
            os.environ['PILOT_SLOW_MO_SCALE'] = repr(float(scale))
            applied['PILOT_SLOW_MO_SCALE'] = float(scale)
        if bool(getattr(args, 'slow_mo', False)) or scale is not None:
            os.environ['PILOT_SLOW_MO'] = '1'
            applied['PILOT_SLOW_MO'] = 1.0
            try:
                s = float(os.environ.get('PILOT_SLOW_MO_SCALE', '0.77') or 0.77)
            except ValueError:
                s = 0.77
            s = max(0.05, min(1.0, s))
            # Keep ~50 Hz of *sim* samples under CE slow-mo.
            log_hz = max(5.0, 50.0 * s)
            os.environ['LOG_HZ'] = repr(log_hz)
            applied['LOG_HZ'] = log_hz
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


# Fallback when GetAsyncKeyState is unavailable: msvcrt autorepeat timeout.
_MANUAL_HOLD_RELEASE_S = 0.45

# Opposing flight keys — pressing one releases the other in the chord.
_MANUAL_KEY_OPPOSITE = {
    'w': 's', 's': 'w', 'up': 'down', 'down': 'up',
    'a': 'd', 'd': 'a', 'left': 'right', 'right': 'left',
    'q': 'e', 'e': 'q',
    'r': 'f', 'f': 'r',
}
_MANUAL_FLIGHT_KEYS = frozenset(_MANUAL_KEY_OPPOSITE)

# Win32 VK codes for true key-down (hold matches physical press duration).
_MANUAL_VK = {
    'w': 0x57, 'a': 0x41, 's': 0x53, 'd': 0x44,
    'q': 0x51, 'e': 0x45, 'r': 0x52, 'f': 0x46,
    'left': 0x25, 'up': 0x26, 'right': 0x27, 'down': 0x28,
}


def _async_flight_keys_down():
    """Return currently held flight keys via Win32, or None if unavailable."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        down = set()
        for name, vk in _MANUAL_VK.items():
            if user32.GetAsyncKeyState(vk) & 0x8000:
                down.add(name)
        return down
    except Exception:
        return None


def _poll_manual_controls(
    hold_state: dict,
    *,
    lean_rad,
    yaw_rate_cmd,
    thrust_step,
    now: float,
    ignore_sticks: bool = False,
    sink_step=None,
    pitch_rad=None,
):
    """Hold-to-fly keyboard + optional gamepad (Windows).

    Flight holds use ``GetAsyncKeyState`` so Q/E/W/A… stay active for the
    whole time the key is physically down (msvcrt autorepeat is not enough).
    msvcrt still drains the console buffer and edges T/H/K/Esc.

    Gamepad (PlayStation / Xbox via pygame) uses Mode-2 sticks; analog
    deflection overrides the matching keyboard axis while outside deadzone.

    When ``ignore_sticks`` is True (REPLAY / AUTO), flight keys/sticks are
    noted for handoff but do not move axes.

      W/S or ↑/↓   pitch while held
      A/D or ←/→   roll while held
      Q/E          yaw while held
      R/F          climb / sink thrust while held (back to hover on release)
      Space        force level now
      M            mark waypoint (capture mode)
      K            keep / commit remembered path to disk
      T            pilot: request auto (assist) on LOCK
      H            pilot: return sticks to human (letter H; not arrow-up)
      O            toggle client slow-mo (match CE / DxWnd scale)
      Esc / X      quit

      Pad L-stick  yaw / climb    R-stick roll / pitch
      ✕/A level   ○/B quit   □/X human   △/Y reset   LB auto   Options keep
      D-pad ↓      toggle client slow-mo
    """
    import config as _cfg
    fwd = float(getattr(_cfg, 'FORWARD_PITCH_SIGN', 1.0))
    yaw_sign = float(getattr(_cfg, 'RATE_SIGN_YAW', 1.0))
    pitch_lim = float(lean_rad) if pitch_rad is None else float(pitch_rad)
    quit_req = False

    def _apply_chord_axes(chord: set) -> None:
        """Set roll/pitch/yaw/thrust from the full held chord."""
        if ignore_sticks:
            return
        for axis in ('roll', 'pitch', 'yaw', 'thrust'):
            hold_state[axis] = 0.0
            hold_state[f'{axis}_t'] = 0.0
        if chord & {'a', 'left'}:
            hold_state['roll'] = -lean_rad
            hold_state['roll_t'] = now
        if chord & {'d', 'right'}:
            hold_state['roll'] = lean_rad
            hold_state['roll_t'] = now
        if chord & {'w', 'up'}:
            hold_state['pitch'] = fwd * pitch_lim
            hold_state['pitch_t'] = now
        if chord & {'s', 'down'}:
            hold_state['pitch'] = -fwd * pitch_lim
            hold_state['pitch_t'] = now
        if 'q' in chord:
            hold_state['yaw'] = -yaw_sign * yaw_rate_cmd
            hold_state['yaw_t'] = now
        if 'e' in chord:
            hold_state['yaw'] = yaw_sign * yaw_rate_cmd
            hold_state['yaw_t'] = now
        if 'r' in chord:
            hold_state['thrust'] = thrust_step
            hold_state['thrust_t'] = now
        if 'f' in chord:
            # Mild F by default; caller passes PILOT_G2_SINK_RATE after GATE 1.
            sink = (
                abs(float(sink_step))
                if sink_step is not None
                else abs(float(
                    getattr(_cfg, 'PILOT_SINK_RATE', abs(thrust_step))
                ))
            )
            hold_state['thrust'] = -sink
            hold_state['thrust_t'] = now

    def _note_meta(name: str) -> None:
        """msvcrt edge for non-hold keys (still used if async unavailable)."""
        keys = hold_state.setdefault('keys_held', {})
        opp = _MANUAL_KEY_OPPOSITE.get(name)
        if opp:
            keys.pop(opp, None)
        keys[name] = now
        for k in list(keys.keys()):
            if k in _MANUAL_FLIGHT_KEYS:
                keys[k] = now

    try:
        import msvcrt
    except ImportError:
        msvcrt = None

    # Drain console buffer + edge-detect meta keys (T/H/K/Space/quit).
    if msvcrt is not None:
        while msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch in (b'\x00', b'\xe0'):
                if not msvcrt.kbhit():
                    break
                code = msvcrt.getch()
                # Arrows still noted as fallback when async is missing.
                if code == b'K':
                    _note_meta('left')
                elif code == b'M':
                    _note_meta('right')
                elif code == b'H':
                    _note_meta('up')
                elif code == b'P':
                    _note_meta('down')
                continue
            try:
                key = ch.decode('ascii', errors='ignore').lower()
            except Exception:
                continue
            if key in _MANUAL_FLIGHT_KEYS:
                _note_meta(key)
            elif key == 'm':
                if not ignore_sticks:
                    hold_state['marks'] = int(hold_state.get('marks', 0)) + 1
            elif key == 'k':
                hold_state['keeps'] = int(hold_state.get('keeps', 0)) + 1
            elif key == 't':
                hold_state['auto'] = int(hold_state.get('auto', 0)) + 1
            elif key == 'h':
                hold_state['human'] = int(hold_state.get('human', 0)) + 1
            elif key == 'z':
                # Declare-level / zero EKF tilt (fly mode; harmless elsewhere).
                hold_state['zero_att'] = int(hold_state.get('zero_att', 0)) + 1
            elif key == 'y':
                hold_state['resets'] = int(hold_state.get('resets', 0)) + 1
            elif key == 'o':
                hold_state['slowmo'] = int(hold_state.get('slowmo', 0)) + 1
            elif key == ' ':
                for axis in ('roll', 'pitch', 'yaw', 'thrust'):
                    hold_state[axis] = 0.0
                    hold_state[f'{axis}_t'] = 0.0
                hold_state['keys_held'] = {}
            elif key in ('x', '\x1b'):
                quit_req = True

    # True held state (Q stays yawing the whole time you hold it).
    async_down = _async_flight_keys_down()
    keys = hold_state.setdefault('keys_held', {})
    if async_down is not None:
        chord = set(async_down)
        # Resolve opposites: keep both letter and arrow aliases; drop pairs.
        for a, b in (('w', 's'), ('a', 'd'), ('q', 'e'), ('r', 'f'),
                     ('up', 'down'), ('left', 'right')):
            if a in chord and b in chord:
                chord.discard(b)
        for k in list(keys.keys()):
            if k in _MANUAL_FLIGHT_KEYS and k not in chord:
                keys.pop(k, None)
        for k in chord:
            keys[k] = now
        hold_state['keys_down'] = set(chord)
    else:
        hold_state['keys_down'] = {
            k for k, t in keys.items()
            if k in _MANUAL_FLIGHT_KEYS
            and t > 0.0
            and (now - float(t)) <= _MANUAL_HOLD_RELEASE_S
        }

    _apply_chord_axes(hold_state['keys_down'])

    def _active(axis: str) -> float:
        if ignore_sticks:
            return 0.0
        # With async holds, axes are rewritten every tick while down.
        if axis not in hold_state or not hold_state.get(f'{axis}_t'):
            return 0.0
        if async_down is None:
            t = float(hold_state.get(f'{axis}_t') or 0.0)
            if t <= 0.0 or (now - t) > _MANUAL_HOLD_RELEASE_S:
                return 0.0
        return float(hold_state.get(axis) or 0.0)

    roll = _active('roll')
    pitch = _active('pitch')
    yaw = _active('yaw')
    thrust = _active('thrust')

    # Gamepad: always poll for meta buttons (Y/H/O/…); sticks only when flying.
    pad = None
    try:
        from gamepad_input import read_pad_axes
        pad = read_pad_axes(
            deadzone=float(
                getattr(_cfg, 'PILOT_PAD_DEADZONE', 0.16) or 0.16
            ),
            expo=float(getattr(_cfg, 'PILOT_PAD_EXPO', 0.55) or 0.0),
            smooth=float(getattr(_cfg, 'PILOT_PAD_SMOOTH', 0.28) or 0.28),
        )
    except Exception:
        pad = None
    if pad is not None:
        if not ignore_sticks:
            sink = abs(
                float(sink_step)
                if sink_step is not None
                else float(getattr(_cfg, 'PILOT_SINK_RATE', abs(thrust_step)))
            )
            soft = float(getattr(_cfg, 'PILOT_PAD_SOFT_GAIN', 0.70) or 0.70)
            soft_yaw = float(
                getattr(_cfg, 'PILOT_PAD_SOFT_YAW', 0.55) or 0.55
            )
            soft_thr = float(
                getattr(_cfg, 'PILOT_PAD_SOFT_THRUST', 1.0) or 1.0
            )
            bump = float(
                getattr(_cfg, 'PILOT_PAD_THRUST_BUMP', 0.035) or 0.0
            )
            g_att = soft
            g_yaw = soft_yaw
            g_thr = soft_thr
            if pad.level:
                roll = pitch = yaw = thrust = 0.0
                for axis in ('roll', 'pitch', 'yaw', 'thrust'):
                    hold_state[axis] = 0.0
                    hold_state[f'{axis}_t'] = 0.0
                hold_state['keys_held'] = {}
                hold_state.pop('thrust_bump', None)
            else:
                if abs(pad.roll) > 1e-4:
                    roll = float(pad.roll) * float(lean_rad) * g_att
                    hold_state['roll'] = roll
                    hold_state['roll_t'] = now
                if abs(pad.pitch) > 1e-4:
                    pitch = (
                        float(fwd) * float(pad.pitch) * float(pitch_lim) * g_att
                    )
                    hold_state['pitch'] = pitch
                    hold_state['pitch_t'] = now
                if abs(pad.yaw) > 1e-4:
                    yaw = (
                        float(yaw_sign)
                        * float(pad.yaw)
                        * float(yaw_rate_cmd)
                        * g_yaw
                    )
                    hold_state['yaw'] = yaw
                    hold_state['yaw_t'] = now
                if abs(pad.thrust) > 1e-4:
                    # RT = +climb rate, LT = -sink rate.
                    if pad.thrust >= 0.0:
                        thrust = (
                            float(pad.thrust) * float(thrust_step) * g_thr
                        )
                    else:
                        thrust = float(pad.thrust) * float(sink) * g_thr
                    hold_state['thrust'] = thrust
                    hold_state['thrust_t'] = now
                # RB: raw collective bump (applied in pilot tick via hold_state).
                if pad.thrust_bump and bump > 0.0:
                    hold_state['thrust_bump'] = bump
                    hold_state['thrust_bump_t'] = now
                else:
                    hold_state.pop('thrust_bump', None)
                    hold_state.pop('thrust_bump_t', None)
            hold_state['pad'] = {
                'name': pad.name,
                'roll': pad.roll,
                'pitch': pad.pitch,
                'yaw': pad.yaw,
                'thrust': pad.thrust,
                'thrust_bump': bool(pad.thrust_bump),
            }
            moved = max(
                abs(pad.roll), abs(pad.pitch), abs(pad.yaw), abs(pad.thrust)
            )
            last_log = float(hold_state.get('_pad_log_t') or 0.0)
            if (moved > 0.08 or pad.thrust_bump) and (now - last_log) > 1.0:
                hold_state['_pad_log_t'] = now
                print(
                    f'[PAD] R={pad.roll:+.2f} P={pad.pitch:+.2f} '
                    f'Y={pad.yaw:+.2f} T={pad.thrust:+.2f}'
                    f'{" RB+" if pad.thrust_bump else ""}',
                    flush=True,
                )
            elif (
                moved <= 0.08
                and not pad.thrust_bump
                and (now - last_log) > 8.0
            ):
                hold_state['_pad_log_t'] = now
                if not hold_state.get('_pad_silent_warned'):
                    hold_state['_pad_silent_warned'] = True
                    print(
                        '[PAD] sticks idle — if you are moving them, disable '
                        'the controller inside FlightSim (it is eating '
                        'XInput).',
                        flush=True,
                    )
        if pad.auto:
            hold_state['auto'] = int(hold_state.get('auto', 0)) + 1
        if pad.human:
            hold_state['human'] = int(hold_state.get('human', 0)) + 1
        if pad.keep:
            hold_state['keeps'] = int(hold_state.get('keeps', 0)) + 1
        if getattr(pad, 'reset', False):
            hold_state['resets'] = int(hold_state.get('resets', 0)) + 1
        if getattr(pad, 'slowmo', False):
            hold_state['slowmo'] = int(hold_state.get('slowmo', 0)) + 1
        if pad.quit:
            quit_req = True

    return (roll, pitch, yaw, thrust, quit_req)


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


class WaypointCapture:
    """Append derived poses to a mission JSON for spline replay.

    Records ``shared_data['position_ned']`` (EKF-derived) plus yaw. Absolute
    accuracy does not matter — capture and replay must share ``EKF_USE_PNP``.

    Use ``mark()`` for M-key / GATE_PASSED tags, and ``maybe_sample()`` for
    continuous sampling at ``SPLINE_CAPTURE_HZ``.
    """

    def __init__(self, path, *, continuous_hz: float = 0.0):
        self.path = Path(path)
        self.waypoints = []
        self.continuous_hz = float(continuous_hz or 0.0)
        self._last_sample_t = None
        self._last_active_gate = None
        self.saved_once = False

    def seed_from_mission(self, mission) -> int:
        """Copy existing waypoints (replay prefix) so KEEP extends that path."""
        if mission is None:
            return 0
        n0 = len(self.waypoints)
        for w in mission.waypoints:
            row = {
                'n': float(w.pos[0]),
                'e': float(w.pos[1]),
                'd': float(w.pos[2]),
                'yaw_deg': (
                    float(w.yaw_deg) if w.yaw is not None else 0.0
                ),
                'name': w.name or f'wp{len(self.waypoints)}',
            }
            if w.active_gate is not None:
                row['active_gate'] = int(w.active_gate)
                prev = self._last_active_gate
                self._last_active_gate = (
                    int(w.active_gate)
                    if prev is None
                    else max(prev, int(w.active_gate))
                )
            if w.event:
                row['event'] = w.event
            if w.t is not None:
                row['t'] = float(w.t)
            self.waypoints.append(row)
        return len(self.waypoints) - n0

    def max_gate(self) -> int | None:
        gates = [
            int(w['active_gate'])
            for w in self.waypoints
            if w.get('active_gate') is not None
        ]
        for w in self.waypoints:
            name = str(w.get('name') or '')
            if name.lower().startswith('gate'):
                try:
                    gates.append(int(''.join(ch for ch in name if ch.isdigit())))
                except ValueError:
                    pass
        return max(gates) if gates else None

    def mark(
        self,
        shared_data,
        *,
        name=None,
        event=None,
        active_gate=None,
        t=None,
    ):
        """Capture one waypoint. Returns a status string for the console."""
        pos = shared_data.get('position_ned') or {}
        att = shared_data.get('attitude') or {}
        n, e, d = (_f(pos.get(k)) for k in ('x', 'y', 'z'))
        yaw = _f(att.get('yaw'))
        if n is None or e is None or d is None or yaw is None:
            return 'MARK REJECTED — no derived pose yet (is the EKF running?)'
        idx = len(self.waypoints)
        row = {
            'n': n, 'e': e, 'd': d,
            'yaw_deg': math.degrees(yaw),
            'name': name or f'wp{idx}',
        }
        if active_gate is not None:
            row['active_gate'] = int(active_gate)
        if event:
            row['event'] = event
        if t is not None:
            row['t'] = float(t)
        self.waypoints.append(row)
        tag = ''
        if event:
            tag = f'  event={event}'
        if active_gate is not None:
            tag += f'  gate={int(active_gate)}'
        return (
            f'MARK {row["name"]}  n={n:+.2f} e={e:+.2f} d={d:+.2f} '
            f'yaw={math.degrees(yaw):+.1f}deg{tag}'
        )

    def maybe_sample(self, shared_data, t: float):
        """Periodic sample when continuous_hz > 0. Returns status or None."""
        if self.continuous_hz <= 0.0:
            return None
        period = 1.0 / self.continuous_hz
        if (
            self._last_sample_t is not None
            and (t - self._last_sample_t) < period
        ):
            return None
        msg = self.mark(shared_data, t=t)
        if 'REJECTED' in msg:
            return None
        self._last_sample_t = t
        return msg

    def watch_gate_pass(self, shared_data, t: float):
        """If race ``active_gate`` advanced, tag a gate_pass mark. Else None."""
        race = shared_data.get('race_status') or {}
        ag = race.get('active_gate')
        if ag is None:
            return None
        try:
            ag_i = int(ag)
        except (TypeError, ValueError):
            return None
        prev = self._last_active_gate
        self._last_active_gate = ag_i if prev is None else max(prev, ag_i)
        if prev is None or ag_i <= prev:
            return None
        # active_gate became ag_i after clearing that gate (or the next index —
        # we tag with the new value so keep_until_gate can find it).
        return self.mark(
            shared_data,
            name=f'gate{ag_i}',
            event='gate_pass',
            active_gate=ag_i,
            t=t,
        )

    def save(self):
        """Write the mission file. Returns the path, or None if nothing marked."""
        if len(self.waypoints) < 2:
            return None
        import config
        from mission import Mission, Waypoint, save_mission

        self.path.parent.mkdir(parents=True, exist_ok=True)
        wps = [
            Waypoint(
                w['n'], w['e'], w['d'], w.get('yaw_deg', 0.0),
                name=w.get('name', ''),
                active_gate=w.get('active_gate'),
                event=w.get('event'),
                t=w.get('t'),
            )
            for w in self.waypoints
        ]
        mission = Mission(
            wps,
            loop=False,
            name='captured',
            ekf_use_pnp=int(bool(getattr(config, 'EKF_USE_PNP', True))),
        )
        save_mission(mission, str(self.path))
        self.saved_once = True
        return self.path


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
    yaw_rate_cmd = math.radians(float(getattr(args, 'yaw_rate_deg', 35.0)))
    open_loop = bool(getattr(args, 'open_loop_thrust', False))
    climb_rate_cmd = float(getattr(args, 'climb_rate', 0.6) or 0.6)
    # Key returns a raw thrust offset in open loop, else a rate setpoint.
    thrust_step = (float(getattr(args, 'thrust_step', 0.022))
                   if open_loop else climb_rate_cmd)
    vrate = None if open_loop else VerticalRateHold(
        kp=getattr(args, 'climb_kp', None),
        ki=getattr(args, 'climb_ki', None),
        authority=float(getattr(args, 'climb_auth', 0.08) or 0.08),
    )
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
        hold_s = _countdown_hold_s(args)
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
                if vrate is None:
                    climb_meas, climb_src = None, None
                else:
                    thrust_delta, climb_meas, climb_src = vrate.update(
                        shared_data, thrust_delta, dt
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


def _gate_aid_info(shared_data) -> dict:
    """Live state of the vision attitude aid, for the HUD and pilot CSV.

    Without this the only symptom of the aid doing nothing is the drift it
    was added to remove — which is indistinguishable from it being too weak.
    ``gh`` counts applied horizon fixes; ``b`` is the learned gyro bias in
    deg/s, which should settle to a small non-zero constant.
    """
    ekf = shared_data.get('ekf_state') or {}
    fixes = int(ekf.get('gate_horizon_fixes') or 0)
    bias = ekf.get('gyro_bias') or (0.0, 0.0, 0.0)
    try:
        bx, by = math.degrees(float(bias[0])), math.degrees(float(bias[1]))
    except (TypeError, ValueError, IndexError):
        bx = by = 0.0
    return {
        'fixes': fixes,
        'hud': f'gh{fixes:<5d} b{bx:+5.2f}/{by:+5.2f}',
    }


def accel_is_gravity_reference(shared_data) -> bool:
    """True when the raw IMU can currently vouch for which way is down.

    The accelerometer only reads gravity while the craft is not being slung
    around: near-1g magnitude and modest body rates. When both hold, an AHRS
    built on it is trustworthy no matter how far the EKF has wandered.
    """
    imu = shared_data.get('highres_imu') or {}
    if not imu:
        return False
    ax = _f(imu.get('xacc'))
    ay = _f(imu.get('yacc'))
    az = _f(imu.get('zacc'))
    if ax is None or ay is None or az is None:
        return False
    amag = math.sqrt(ax * ax + ay * ay + az * az)
    if abs(amag - 9.80665) > 0.18 * 9.80665:
        return False
    gyro_mag = math.hypot(
        _f(imu.get('xgyro'), 0.0),
        math.hypot(_f(imu.get('ygyro'), 0.0), _f(imu.get('zgyro'), 0.0)),
    )
    return gyro_mag <= math.radians(60.0)


def read_pilot_attitude(
    shared_data,
    des_roll=0.0,
    des_pitch=0.0,
    yaw_rate=0.0,
):
    """Attitude feedback for human teleop.

    EKF roll/pitch drift mid-race (no continuous gravity aid). When the stick
    lean command is small, blend in controller AHRS (accel-aided) so "neutral"
    tracks gravity without needing a full hover pause. Hard lean stays on EKF.
    """
    import config as _cfg
    ekf_r, ekf_p, rs, ps = read_attitude(shared_data)
    # Note: `x or 1` would treat 0 as missing and keep the blend ON (033644).
    if int(getattr(_cfg, 'PILOT_LEVEL_AHRS', 0) or 0) == 0:
        return ekf_r, ekf_p, rs, ps
    co = shared_data.get('control_output') or {}
    ar = co.get('ahrs_roll')
    ap = co.get('ahrs_pitch')
    try:
        ar_f = float(ar)
        ap_f = float(ap)
    except (TypeError, ValueError):
        return ekf_r, ekf_p, rs, ps
    if not (math.isfinite(ar_f) and math.isfinite(ap_f)):
        return ekf_r, ekf_p, rs, ps
    # Reject tumbled / wild AHRS.
    if abs(ar_f) > math.radians(45.0) or abs(ap_f) > math.radians(45.0):
        return ekf_r, ekf_p, rs, ps
    disagree = max(abs(ar_f - float(ekf_r)), abs(ap_f - float(ekf_p)))
    trusted = accel_is_gravity_reference(shared_data)
    # Disagreement GROWS as the EKF drifts (141532: +4° → −23° over 50 s), so
    # a plain "they differ, stay on EKF" gate silences the fix exactly when it
    # is needed. Only bail when the accelerometer cannot vouch for AHRS.
    max_disagree = math.radians(
        float(getattr(_cfg, 'PILOT_LEVEL_AHRS_MAX_DISAGREE_DEG', 60.0) or 60.0)
    )
    if disagree > max_disagree:
        return ekf_r, ekf_p, rs, ps
    if disagree > math.radians(22.0) and not trusted:
        return ekf_r, ekf_p, rs, ps
    # While yawing hard, AHRS euler is still noisy — stay on EKF.
    yaw_gate = math.radians(
        float(getattr(_cfg, 'PILOT_LEVEL_AHRS_YAW_GATE_DEG', 35.0) or 35.0)
    )
    if abs(float(yaw_rate)) > yaw_gate:
        return ekf_r, ekf_p, rs, ps
    lean_cmd = math.hypot(float(des_roll), float(des_pitch))
    blend_rad = math.radians(
        float(getattr(_cfg, 'PILOT_LEVEL_AHRS_BLEND_DEG', 14.0) or 14.0)
    )
    # Confirmed drift: widen the window so the correction survives real stick
    # input instead of only firing on a near-neutral stick.
    drift_rad = math.radians(
        float(getattr(_cfg, 'PILOT_LEVEL_AHRS_DRIFT_DEG', 8.0) or 8.0)
    )
    if trusted and disagree > drift_rad:
        blend_rad *= float(
            getattr(_cfg, 'PILOT_LEVEL_AHRS_DRIFT_WIDEN', 2.5) or 1.0
        )
    w_ekf = (
        0.0 if blend_rad <= 1e-6
        else max(0.0, min(1.0, lean_cmd / blend_rad))
    )
    roll = (1.0 - w_ekf) * ar_f + w_ekf * float(ekf_r)
    pitch = (1.0 - w_ekf) * ap_f + w_ekf * float(ekf_p)
    return roll, pitch, rs, ps


def read_pilot_attitude(shared_data):
    """Roll/pitch for pilot teleop: real AHRS, not the dual-gate EKF.

    ``shared_data['attitude']`` is EKF-owned and on VQ2 often tracks the
    *commanded* lean (telem 023415: des=pitch=62° while ahrs≈20°). Using that
    for tilt-comp commanded HT/cos(floor)=0.364 and for the pitch PID made
    error≈0 so forward tip felt weak while the craft lofted.
    Prefer controller AHRS, then MAVLink ``attitude_raw``, else EKF.
    """
    imu = shared_data.get('highres_imu') or {}
    gx = _f(imu.get('xgyro'), 0.0)
    gy = _f(imu.get('ygyro'), 0.0)
    ctrl = shared_data.get('control_output') or {}
    ar = ctrl.get('ahrs_roll')
    ap = ctrl.get('ahrs_pitch')
    if ar is not None and ap is not None:
        try:
            ar_f, ap_f = float(ar), float(ap)
            if math.isfinite(ar_f) and math.isfinite(ap_f):
                return ar_f, ap_f, gx, gy
        except (TypeError, ValueError):
            pass
    raw = shared_data.get('attitude_raw') or {}
    if raw.get('roll') is not None and raw.get('pitch') is not None:
        try:
            ar_f, ap_f = float(raw['roll']), float(raw['pitch'])
            if math.isfinite(ar_f) and math.isfinite(ap_f):
                return ar_f, ap_f, gx, gy
        except (TypeError, ValueError):
            pass
    return read_attitude(shared_data)


def _gate_aid_info(shared_data) -> dict:
    """Live state of the vision attitude aid, for the HUD and pilot CSV.

    Without this the only symptom of the aid doing nothing is the drift it
    was added to remove — which is indistinguishable from it being too weak.
    ``gh`` counts applied horizon fixes; ``b`` is the learned gyro bias in
    deg/s, which should settle to a small non-zero constant.
    """
    ekf = shared_data.get('ekf_state') or {}
    fixes = int(ekf.get('gate_horizon_fixes') or 0)
    bias = ekf.get('gyro_bias') or (0.0, 0.0, 0.0)
    try:
        bx, by = math.degrees(float(bias[0])), math.degrees(float(bias[1]))
    except (TypeError, ValueError, IndexError):
        bx = by = 0.0
    return {
        'fixes': fixes,
        'hud': f'gh{fixes:<5d} b{bx:+5.2f}/{by:+5.2f}',
    }


def accel_is_gravity_reference(shared_data) -> bool:
    """True when the raw IMU can currently vouch for which way is down.

    The accelerometer only reads gravity while the craft is not being slung
    around: near-1g magnitude and modest body rates. When both hold, an AHRS
    built on it is trustworthy no matter how far the EKF has wandered.
    """
    imu = shared_data.get('highres_imu') or {}
    if not imu:
        return False
    ax = _f(imu.get('xacc'))
    ay = _f(imu.get('yacc'))
    az = _f(imu.get('zacc'))
    if ax is None or ay is None or az is None:
        return False
    amag = math.sqrt(ax * ax + ay * ay + az * az)
    if abs(amag - 9.80665) > 0.18 * 9.80665:
        return False
    gyro_mag = math.hypot(
        _f(imu.get('xgyro'), 0.0),
        math.hypot(_f(imu.get('ygyro'), 0.0), _f(imu.get('zgyro'), 0.0)),
    )
    return gyro_mag <= math.radians(60.0)


def read_pilot_attitude(
    shared_data,
    des_roll=0.0,
    des_pitch=0.0,
    yaw_rate=0.0,
):
    """Attitude feedback for human teleop.

    EKF roll/pitch drift mid-race (no continuous gravity aid). When the stick
    lean command is small, blend in controller AHRS (accel-aided) so "neutral"
    tracks gravity without needing a full hover pause. Hard lean stays on EKF.
    """
    import config as _cfg
    ekf_r, ekf_p, rs, ps = read_attitude(shared_data)
    # Note: `x or 1` would treat 0 as missing and keep the blend ON (033644).
    if int(getattr(_cfg, 'PILOT_LEVEL_AHRS', 0) or 0) == 0:
        return ekf_r, ekf_p, rs, ps
    co = shared_data.get('control_output') or {}
    ar = co.get('ahrs_roll')
    ap = co.get('ahrs_pitch')
    try:
        ar_f = float(ar)
        ap_f = float(ap)
    except (TypeError, ValueError):
        return ekf_r, ekf_p, rs, ps
    if not (math.isfinite(ar_f) and math.isfinite(ap_f)):
        return ekf_r, ekf_p, rs, ps
    # Reject tumbled / wild AHRS.
    if abs(ar_f) > math.radians(45.0) or abs(ap_f) > math.radians(45.0):
        return ekf_r, ekf_p, rs, ps
    disagree = max(abs(ar_f - float(ekf_r)), abs(ap_f - float(ekf_p)))
    trusted = accel_is_gravity_reference(shared_data)
    # Disagreement GROWS as the EKF drifts (141532: +4° → −23° over 50 s), so
    # a plain "they differ, stay on EKF" gate silences the fix exactly when it
    # is needed. Only bail when the accelerometer cannot vouch for AHRS.
    max_disagree = math.radians(
        float(getattr(_cfg, 'PILOT_LEVEL_AHRS_MAX_DISAGREE_DEG', 60.0) or 60.0)
    )
    if disagree > max_disagree:
        return ekf_r, ekf_p, rs, ps
    if disagree > math.radians(22.0) and not trusted:
        return ekf_r, ekf_p, rs, ps
    # While yawing hard, AHRS euler is still noisy — stay on EKF.
    yaw_gate = math.radians(
        float(getattr(_cfg, 'PILOT_LEVEL_AHRS_YAW_GATE_DEG', 35.0) or 35.0)
    )
    if abs(float(yaw_rate)) > yaw_gate:
        return ekf_r, ekf_p, rs, ps
    lean_cmd = math.hypot(float(des_roll), float(des_pitch))
    blend_rad = math.radians(
        float(getattr(_cfg, 'PILOT_LEVEL_AHRS_BLEND_DEG', 14.0) or 14.0)
    )
    # Confirmed drift: widen the window so the correction survives real stick
    # input instead of only firing on a near-neutral stick.
    drift_rad = math.radians(
        float(getattr(_cfg, 'PILOT_LEVEL_AHRS_DRIFT_DEG', 8.0) or 8.0)
    )
    if trusted and disagree > drift_rad:
        blend_rad *= float(
            getattr(_cfg, 'PILOT_LEVEL_AHRS_DRIFT_WIDEN', 2.5) or 1.0
        )
    w_ekf = (
        0.0 if blend_rad <= 1e-6
        else max(0.0, min(1.0, lean_cmd / blend_rad))
    )
    roll = (1.0 - w_ekf) * ar_f + w_ekf * float(ekf_r)
    pitch = (1.0 - w_ekf) * ap_f + w_ekf * float(ekf_p)
    return roll, pitch, rs, ps


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


def _ground_speed_kmh(shared_data):
    """Horizontal speed in km/h, or None.

    Wall-referenced like the rest of the estimator, so under slow-mo it
    reads low against a 1x run by the CE factor.
    """
    ekf = shared_data.get('ekf_state') or {}
    vel = ekf.get('velocity_ned') or []
    if len(vel) < 2:
        return None
    vn, ve = _f(vel[0]), _f(vel[1])
    if vn is None or ve is None:
        return None
    return math.hypot(vn, ve) * 3.6


def _ground_speed_kmh(shared_data):
    """Horizontal speed in km/h, or None.

    Wall-referenced like the rest of the estimator, so under slow-mo it
    reads low against a 1x run by the CE factor.
    """
    ekf = shared_data.get('ekf_state') or {}
    vel = ekf.get('velocity_ned') or []
    if len(vel) < 2:
        return None
    vn, ve = _f(vel[0]), _f(vel[1])
    if vn is None or ve is None:
        return None
    return math.hypot(vn, ve) * 3.6


def _vertical_rate_down(shared_data):
    """Measured vertical velocity, NED down-positive, or None.

    EKF velocity first (works on both sim builds); sim odometry only as a
    fallback, since VQ2 does not publish it.
    """
    ekf = shared_data.get('ekf_state') or {}
    vel = ekf.get('velocity_ned') or []
    if len(vel) > 2:
        vz = _f(vel[2])
        if vz is not None:
            return vz, 'ekf'
    vz = _f((shared_data.get('position_ned') or {}).get('vz'))
    if vz is not None:
        return vz, 'ekf_pos'
    vz = _f((shared_data.get('local_position_ned') or {}).get('vz'))
    if vz is not None:
        return vz, 'truth'
    return None, None


def _stick_rate_to_thrust_delta(
    stick_vert: float,
    *,
    climb_rate: float,
    sink_rate: float,
    climb_auth: float,
    sink_auth: float,
) -> float:
    """Map ±m/s stick command to a collective offset (no EKF loop)."""
    sv = float(stick_vert)
    if sv >= 0.0:
        scale = float(climb_auth) / max(1e-3, float(climb_rate))
        return max(0.0, min(float(climb_auth), sv * scale))
    scale = float(sink_auth) / max(1e-3, float(sink_rate))
    return max(-float(sink_auth), min(0.0, sv * scale))


def _stick_rate_to_thrust_delta(
    stick_vert: float,
    *,
    climb_rate: float,
    sink_rate: float,
    climb_auth: float,
    sink_auth: float,
) -> float:
    """Map ±m/s stick command to a collective offset (no EKF loop)."""
    sv = float(stick_vert)
    if sv >= 0.0:
        scale = float(climb_auth) / max(1e-3, float(climb_rate))
        return max(0.0, min(float(climb_auth), sv * scale))
    scale = float(sink_auth) / max(1e-3, float(sink_rate))
    return max(-float(sink_auth), min(0.0, sv * scale))


class VerticalRateHold:
    """Turn R/F into a climb-RATE command with active braking on release.

    An open-loop thrust offset is an *acceleration* command: releasing the key
    stops accelerating but leaves the vertical velocity, so the craft coasts
    and you have to counter-hold to stop it — and it also feels slow to start,
    because velocity has to build. Closing a PI loop on measured vertical
    velocity makes the key a rate command: hold = climb at a fixed rate,
    release = commanded rate zero, which the loop brakes toward.

    Sign convention matches controller.py's thrust PI: error is
    ``measured_down - target_down``, so descending faster than commanded is a
    positive error and asks for more thrust.
    """

    def __init__(self, *, kp=None, ki=None, authority=0.08):
        import config
        from control.pid import PIDConfig, PIDController

        self.authority = float(authority)
        self.pid = PIDController(PIDConfig(
            kp=float(config.KP_THRUST_VEL if kp is None else kp),
            ki=float(config.KI_THRUST_VEL if ki is None else ki),
            output_min=-self.authority,
            output_max=self.authority,
            integral_min=-config.THRUST_INTEGRAL_LIMIT,
            integral_max=config.THRUST_INTEGRAL_LIMIT,
        ))
        self.vz_down = None
        self.source = None

    def reset(self):
        self.pid.reset()

    def update(self, shared_data, climb_rate_cmd, dt):
        """Return (thrust_delta, measured_climb_rate_up, source).

        ``climb_rate_cmd`` is positive-up m/s (0 when no key is held).
        """
        vz_down, source = _vertical_rate_down(shared_data)
        self.vz_down, self.source = vz_down, source
        if vz_down is None:
            # No vertical velocity available — fall back to open loop so the
            # craft is still controllable, just without braking.
            return float(climb_rate_cmd) * 0.03, None, None
        target_down = -float(climb_rate_cmd)
        delta = self.pid.update(vz_down - target_down, dt)
        return float(delta), -vz_down, source


def _tilt_compensated_thrust(hover_thrust, des_roll, des_pitch, *, lean_boost=None):
    """Match kalman_planner: keep vertical lift while pitched/rolled.

    Phase 4.5 lean-hover defaults lean_boost=0 so HT / cos(tilt) is measured
    cleanly. Flight still uses config.LEAN_THRUST_BOOST via kalman_planner.
    """
    import config
    # Floor must allow full pilot lean (~28–35°) — old 0.88 capped at ~28°
    # and still sank on hard forward.
    cos_floor = float(
        getattr(config, 'MIN_TILT_COMPENSATION_COSINE', 0.70) or 0.70
    )
    cos_floor = max(0.55, min(0.95, cos_floor))
    tilt = max(
        cos_floor,
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
    hold_s = _countdown_hold_s(args)
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
    hold_s = _countdown_hold_s(args)
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

    hold_s = _countdown_hold_s(args)
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
                'vert_src': path.get('vert_src'),
                'pose_dz': path.get('pose_dz'),
            }
            recorder.write(row)
            if not args.quiet and int(elapsed * args.hz) % max(1, int(args.hz)) == 0:
                print(
                    f"{elapsed:5.1f} {path.get('phase')} "
                    f"nx={path.get('norm_x')} thr={tgt.get('thrust')} "
                    f"vert={path.get('vert_src')} dz={path.get('pose_dz')}",
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
def _seed_logging_preflight(shared_data, *, slow_mo_scale: float | None) -> None:
    """Confirm the fields HG-DAgger training needs will be written."""
    import config
    from pathlib import Path as _Path

    weights = _Path(getattr(config, 'YOLO_POSE_MODEL_PATH', 'models/gate_pose.pt'))
    print('', flush=True)
    print('=== SEED LOGGING (HG-DAgger) ===', flush=True)
    print(
        '  Full training telem is logs/telem_*.csv (NOT logs/tuning/).',
        flush=True,
    )
    print(
        '  Required columns: kp0..7_u/v/c, roll/pitch, gx/gy/gz_imu, '
        'cmd_thrust/roll/pitch/yaw_rate, control_authority, '
        'odo_*, active_gate',
        flush=True,
    )
    if weights.is_file():
        print(f'  YOLO pose weights: {weights}  OK', flush=True)
    else:
        print(
            f'  WARNING: missing {weights} — keypoints will be empty and '
            'train_policy will find no usable windows.',
            flush=True,
        )
    print(
        f'  detector={getattr(config, "GATE_DETECTOR_BACKEND", "?")}  '
        f'log_hz={shared_data.get("log_hz", "?")}',
        flush=True,
    )
    if slow_mo_scale is not None and slow_mo_scale < 1.0:
        print(
            f'  slow-mo x{slow_mo_scale:.2f}: set Cheat Engine / DxWnd to '
            f'the SAME factor, or sim-time and client diverge.',
            flush=True,
        )
    print('', flush=True)


def run_manual(args) -> int:
    """ANGLE self-level + hover trim; human commands lean / yaw / thrust."""
    import config
    from control.pid import PIDConfig, PIDController
    from setup import setup_components

    shared_data = {}
    components = setup_components(
        shared_data, int(time.time() * 1000),
        SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT,
        enable_planner=False,
    )
    controller = components['controller']
    logger = components.get('logger')
    if logger is not None:
        shared_data['_telem_path'] = getattr(logger, '_csv_path', None)
    recorder = Recorder(
        Path(args.csv) if args.csv else default_csv_path('manual')
    )

    slow_mo_on = bool(getattr(args, 'slow_mo', False)) or bool(
        getattr(config, 'PILOT_SLOW_MO', 0)
    )
    if getattr(args, 'slow_mo_scale', None) is not None:
        slow_mo_on = True
        slow_mo_scale = float(args.slow_mo_scale)
    else:
        slow_mo_scale = float(getattr(config, 'PILOT_SLOW_MO_SCALE', 0.77) or 0.77)
    if slow_mo_on:
        slow_mo_scale = max(0.05, min(1.0, slow_mo_scale))
        shared_data['log_hz'] = max(5.0, 50.0 * slow_mo_scale)
    else:
        slow_mo_scale = 1.0
        shared_data['log_hz'] = float(getattr(config, 'LOG_HZ', 50) or 50)

    _seed_logging_preflight(
        shared_data,
        slow_mo_scale=slow_mo_scale if slow_mo_on else None,
    )
    if logger is not None:
        print(f'  telem file: {logger._csv_path}', flush=True)
        print('', flush=True)

    lean_deg = getattr(args, 'lean_deg', None)
    if lean_deg is None:
        lean_deg = float(getattr(config, 'MANUAL_LEAN_DEG', 14.0) or 14.0)
    yaw_deg = getattr(args, 'yaw_rate_deg', None)
    if yaw_deg is None:
        yaw_deg = float(getattr(config, 'MANUAL_YAW_RATE_DEG', 40.0) or 40.0)
    climb_rate_cmd = getattr(args, 'climb_rate', None)
    if climb_rate_cmd is None:
        climb_rate_cmd = float(
            getattr(config, 'PILOT_CLIMB_RATE', 1.5) or 1.5
        )
    climb_auth = getattr(args, 'climb_auth', None)
    if climb_auth is None:
        climb_auth = float(getattr(config, 'PILOT_CLIMB_AUTH', 0.12) or 0.12)
    lean_rad = math.radians(float(lean_deg))
    yaw_rate_cmd = math.radians(float(yaw_deg))
    # Proven stick: R/F are thrust offsets unless the caller asked for rate-hold
    # via --climb-rate without --open-loop-thrust.
    want_rate_hold = (
        getattr(args, 'climb_rate', None) is not None
        and not bool(getattr(args, 'open_loop_thrust', False))
    )
    open_loop = not want_rate_hold
    climb_rate_cmd = float(climb_rate_cmd)
    # Key returns a raw thrust offset in open loop, else a rate setpoint.
    if open_loop:
        thrust_step = float(
            getattr(args, 'thrust_step', None)
            or getattr(config, 'MANUAL_THRUST_STEP', 0.028)
        )
    else:
        thrust_step = climb_rate_cmd
    vrate = None if open_loop else VerticalRateHold(
        kp=getattr(args, 'climb_kp', None),
        ki=getattr(args, 'climb_ki', None),
        authority=float(climb_auth),
    )
    cap_arg = getattr(args, 'capture', None)
    capture = (
        None if cap_arg is None
        else WaypointCapture(
            cap_arg or config.SPLINE_CAPTURE_PATH,
            continuous_hz=float(getattr(config, 'SPLINE_CAPTURE_HZ', 5.0)),
        )
    )
    marks_seen = 0
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
    if capture is not None:
        print('  M            mark waypoint (also auto-samples + GATE_PASSED)',
              flush=True)
        print(f'  capture -> {capture.path}', flush=True)
    print('  Esc / X      disarm and quit', flush=True)
    print('  O            toggle client slow-mo (match CE/DxWnd)', flush=True)
    if vrate is None:
        print('  R/F mode: OPEN LOOP thrust offset (coasts on release)', flush=True)
    else:
        print(
            f'  R/F mode: RATE hold +/-{climb_rate_cmd:.2f} m/s (release brakes to 0)',
            flush=True,
        )
    sink_step = float(getattr(config, 'MANUAL_SINK_STEP', 0.040))
    print(
        f'  lean={math.degrees(lean_rad):.0f}°  '
        f'yaw={math.degrees(yaw_rate_cmd):.0f}°/s  '
        f'HT={config.HOVER_THRUST:.3f}  '
        f'kp_att={config.KALMAN_KP_ATT}  '
        f'lean_boost={getattr(config, "LEAN_THRUST_BOOST", 0.0)}',
        flush=True,
    )
    if open_loop:
        print(
            f'  R=+{thrust_step:.3f}  F=-{sink_step:.3f}  (MANUAL_* stick)',
            flush=True,
        )
    if slow_mo_on:
        print(
            f'  slow-mo ON x{slow_mo_scale:.2f}  log_hz={shared_data.get("log_hz")}',
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

    hold_s = _countdown_hold_s(args)
    print(f'[SIM] early-start hold {hold_s:.1f}s before arm...', flush=True)
    time.sleep(hold_s)
    print('Arming...', flush=True)
    controller.arm()
    shared_data['flight_started'] = True
    # Seed-lap provenance for HG-DAgger BC: entire run is human.
    shared_data['control_authority'] = 'human'
    shared_data['intervention_id'] = ''

    hold_state: dict = {}
    last_t = None
    # Wall-rate loop, deliberately unscaled by slow-mo: the sticks are polled
    # once per iteration, so stretching the period to hold a constant sim-time
    # rate turns 0.2x into 250 ms of input latency. Under slow-mo this simply
    # sends more commands per simulated second, which is the good direction.
    period = 1.0 / max(args.hz, 1.0)
    started = time.monotonic()
    slowmos_seen = 0
    if not args.quiet:
        print(
            '\n    t   climb  vz_up  roll  pitch   yaw_r   thr   '
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
                sink_step=sink_step if open_loop else None,
            )
            if quit_req:
                print('\n[STOP] quit key', flush=True)
                break

            slowmos = int(hold_state.get('slowmo', 0))
            if slowmos > slowmos_seen:
                slowmos_seen = slowmos
                slow_mo_on = not slow_mo_on
                shared_data['log_hz'] = (
                    max(5.0, 50.0 * slow_mo_scale) if slow_mo_on else 50.0
                )
                print(
                    f'[SLOW-MO] {"ON" if slow_mo_on else "OFF"} '
                    f'x{slow_mo_scale:.2f}  log_hz={shared_data["log_hz"]}',
                    flush=True,
                )

            if capture is not None:
                marks = int(hold_state.get('marks', 0))
                while marks_seen < marks:
                    marks_seen += 1
                    print('  ' + capture.mark(shared_data, t=elapsed), flush=True)
                gmsg = capture.watch_gate_pass(shared_data, elapsed)
                if gmsg:
                    print('  ' + gmsg, flush=True)
                capture.maybe_sample(shared_data, elapsed)

            dt = period if last_t is None else max(1e-3, now - last_t)
            # Attitude PID sees wall dt (IMU is wall-referenced under CE).
            last_t = now
            roll, pitch, _, _ = read_attitude(shared_data)
            roll_rate = roll_pid.update(des_roll - roll, dt)
            pitch_rate = pitch_pid.update(des_pitch - pitch, dt)
            # Flat hover by default — W pitches forward without lofting.
            if bool(getattr(config, 'PILOT_TILT_COMPENSATE', 1)):
                lean_boost = float(
                    getattr(config, 'LEAN_THRUST_BOOST', 0.0) or 0.0
                )
                thrust = _tilt_compensated_thrust(
                    config.HOVER_THRUST, des_roll, des_pitch,
                    lean_boost=lean_boost,
                )
            else:
                thrust = float(config.HOVER_THRUST)
            stick_vert = float(thrust_delta)
            if vrate is None:
                climb_meas, climb_src = None, None
            else:
                thrust_delta, climb_meas, climb_src = vrate.update(
                    shared_data, thrust_delta, dt
                )
            bump = float(hold_state.get('thrust_bump') or 0.0)
            bump_t = float(hold_state.get('thrust_bump_t') or 0.0)
            if bump_t <= 0.0 or (now - bump_t) > 0.25:
                bump = 0.0
            if stick_vert < -1e-3:
                bump = 0.0
            # Wider clamp than race planner — manual R/F needs authority.
            thrust = float(max(0.06, min(0.45, thrust + thrust_delta + bump)))

            shared_data['control_authority'] = 'human'
            shared_data['intervention_id'] = ''
            shared_data['planner_target'] = {
                'kalman': True,
                'roll_rate': roll_rate,
                'pitch_rate': pitch_rate,
                'yaw_rate': yaw_rate,
                'thrust': thrust,
                'desired_roll': des_roll,
                'desired_pitch': des_pitch,
            }
            shared_data['planner_mode'] = 'manual_seed'
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
                'climb_cmd_mps': thrust_delta if vrate is None else None,
                'climb_meas_mps': climb_meas,
                'climb_src': climb_src,
                'gate1_range_m': _f(dual.get('gate1_range_m')),
                'n_solved': int(dual.get('n_solved') or 0),
            }
            recorder.write(row)
            if not args.quiet:
                print(
                    f"{elapsed:5.1f} {_fmt(climb, '6.2f')}"
                    f" {_fmt(climb_meas, '6.2f')}"
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

    if capture is not None:
        saved = capture.save()
        if saved is None:
            print(
                f'\n[CAPTURE] only {len(capture.waypoints)} waypoint(s) — '
                'need >=2, nothing written',
                flush=True,
            )
        else:
            print(
                f'\n[CAPTURE] {len(capture.waypoints)} waypoints -> {saved}',
                flush=True,
            )
            print(
                '[CAPTURE] replay: '
                f'pilot --replay {saved} --keep-until-gate N',
                flush=True,
            )
            print('[CAPTURE] keep EKF_USE_PNP identical on replay', flush=True)
    print(f'\nCSV: {recorder.path}')
    return 0


def run_coach(args) -> int:
    """HG-DAgger harness: policy flies, H intervenes, T returns authority."""
    import config
    from control.pid import PIDConfig, PIDController
    from policy_planner import PolicyPlanner
    from setup import setup_components

    shared_data = {}
    # Must be set before VisionRX starts, or capture stays detections-only
    # and missed gates never land in the label set.
    if getattr(args, 'save_frames', None) is not None:
        shared_data['vision_reference_capture_all'] = True
    which = str(getattr(args, 'planner', 'policy') or 'policy')
    os.environ['FLIGHT_MODE'] = which
    if getattr(args, 'weights', None):
        os.environ['POLICY_WEIGHTS'] = str(args.weights)
    components = setup_components(
        shared_data, int(time.time() * 1000),
        SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT,
    )
    controller = components['controller']
    planner = components.get('planner')
    if planner is None and which == 'policy':
        planner = PolicyPlanner(
            getattr(args, 'weights', None)
            or getattr(config, 'POLICY_WEIGHTS', 'models/policy.pt')
        )
        components['planner'] = planner
        shared_data['planner'] = planner
    if planner is None:
        print(f'[COACH] no planner for --planner {which}', flush=True)
        shutdown(components)
        return 2

    # ACRO stick, matching how every seed lap was flown: the sticks command
    # body rates directly, with no self-levelling and no lean cap. Taking over
    # in angle mode would hand the policy corrections generated by a different
    # plant than the demonstrations it learned from.
    angle_mode = bool(getattr(args, 'angle', False))
    if angle_mode:
        roll_rate_deg = float(getattr(config, 'MANUAL_LEAN_DEG', 14.0))
        pitch_rate_deg = roll_rate_deg
        yaw_deg = float(getattr(config, 'MANUAL_YAW_RATE_DEG', 40.0))
        thrust_step = float(getattr(config, 'MANUAL_THRUST_STEP', 0.028))
        sink_step = float(getattr(config, 'MANUAL_SINK_STEP', 0.040))
    else:
        roll_rate_deg = float(
            getattr(args, 'roll_rate_deg', None)
            or getattr(config, 'ACRO_ROLL_RATE_DEG', 180.0)
        )
        pitch_rate_deg = float(
            getattr(args, 'pitch_rate_deg', None)
            or getattr(config, 'ACRO_PITCH_RATE_DEG', 180.0)
        )
        yaw_deg = float(
            getattr(args, 'yaw_rate_deg', None)
            or getattr(config, 'ACRO_YAW_RATE_DEG', 160.0)
        )
        thrust_step = float(getattr(config, 'ACRO_CLIMB_AUTH', 0.55))
        sink_step = float(getattr(config, 'ACRO_SINK_AUTH', 0.55))
        # Acro yaw must not be clipped by the assist-era ceiling.
        config.YAW_RATE_MAX_RAD_S = math.radians(
            max(720.0, yaw_deg * 1.25)
        )
    lean_rad = math.radians(float(roll_rate_deg))
    pitch_rad = math.radians(float(pitch_rate_deg))
    yaw_rate_cmd = math.radians(float(yaw_deg))

    max_rate = config.KALMAN_MAX_RATE_RAD_S
    roll_pid = PIDController(PIDConfig(
        kp=config.KALMAN_KP_ATT, kd=config.KALMAN_KD_ATT,
        output_min=-max_rate, output_max=max_rate,
    ))
    pitch_pid = PIDController(PIDConfig(
        kp=config.KALMAN_KP_ATT, kd=config.KALMAN_KD_ATT,
        output_min=-max_rate, output_max=max_rate,
    ))

    mode = 'human' if bool(getattr(args, 'start_human', False)) else 'policy'
    intervention_id = 0
    shared_data['control_authority'] = mode if mode == 'human' else 'policy'
    shared_data['intervention_id'] = ''

    print('', flush=True)
    print('=== COACH (HG-DAgger) ===', flush=True)
    print('  Focus THIS console. Policy flies until you intervene.', flush=True)
    print('  H            HUMAN — take sticks NOW', flush=True)
    print('  T            POLICY — return control after recovery', flush=True)
    print('  Y            RESET — new attempt without restarting', flush=True)
    print('  K / Start    EXCLUDE toggle — drop these frames from training',
          flush=True)
    print('  Esc / X      disarm and quit', flush=True)
    if angle_mode:
        print(
            f'  human: ANGLE self-level  lean={roll_rate_deg:.0f}°  '
            f'yaw={yaw_deg:.0f}°/s',
            flush=True,
        )
    else:
        print(
            f'  human: ACRO rates  roll={roll_rate_deg:.0f}°/s  '
            f'pitch={pitch_rate_deg:.0f}°/s  yaw={yaw_deg:.0f}°/s  '
            f'collective ±{thrust_step:.2f}',
            flush=True,
        )
    print(
        f'  vision: keypoints only (PnP window off, EKF_USE_PNP='
        f'{int(bool(getattr(config, "EKF_USE_PNP", False)))})',
        flush=True,
    )
    print(f'  autopilot: {planner.name}', flush=True)
    if hasattr(planner, 'weights_path'):
        print(f'  weights: {planner.weights_path}', flush=True)
    cap = shared_data.get('gate_frame_capture') or {}
    if cap.get('enabled'):
        every = float(cap.get('interval_s') or 0.0)
        kind = (
            'all frames' if not cap.get('confirmed_detections_only')
            else 'detections only'
        )
        print(
            f'  frames: every {every:.1f}s ({kind}) -> {cap.get("directory")}',
            flush=True,
        )
    print('  TELEOP NOW when you see the policy drift.', flush=True)
    print('', flush=True)

    panel = None
    if bool(getattr(args, 'panel', False)):
        from obs_panel import ObservationPanel
        panel = ObservationPanel(
            with_context=bool(getattr(planner, '_with_context', False)),
            scale=float(getattr(args, 'panel_scale', 1.0) or 1.0),
        )
        print('[COACH] input panel on (q or Esc in the window closes it)',
              flush=True)

    def _pump_panel() -> None:
        nonlocal panel
        if panel is not None and not panel.show(shared_data):
            panel = None

    if bool(getattr(args, 'no_sim_reset', False)):
        print('[SIM] skip reset (--no-sim-reset)', flush=True)
        time.sleep(1.0)
    else:
        print('[SIM] reset (command 31000) before arm...', flush=True)
        controller.send_sim_reset()
        settle = max(0.5, float(getattr(config, 'SIM_RESET_SETTLE_S', 1.5)))
        t_settle = time.monotonic() + settle
        while time.monotonic() < t_settle:
            _pump_panel()
            time.sleep(0.02)

    hold_state: dict = {}
    attempt = {'n': 0}
    log = shared_data.get('log_event')

    # One telemetry row per policy decision. The policy's history buffer
    # advances once per control-loop iteration, and training builds its H-frame
    # windows from telemetry rows, so if the two rates differ the model is
    # trained on a different time span than it sees in flight. Logging at the
    # default 50 Hz against a 10 Hz loop made every intervention window cover
    # 0.66 s where the seed laps covered 3.2 s.
    shared_data['log_hz'] = float(max(1.0, args.hz))
    print(f'[COACH] telem log_hz={shared_data["log_hz"]:.0f} (= control loop)',
          flush=True)

    def _arm_attempt(t_reset: float) -> None:
        """Wait for the sim's GO, with a gate in view, then arm."""
        attempt['n'] += 1
        shared_data['attempt'] = attempt['n']
        # Arming with no gate in view hands the policy an all-sentinel
        # observation on its very first frame -- the one input it was never
        # trained on. Poll for a detection while the countdown runs.
        for _ in range(100):
            det = shared_data.get('gate_detection') or {}
            if det.get('center_px') is not None:
                break
            _pump_panel()
            time.sleep(0.02)
        # Race clock, not a wall-clock guess: under slow-mo the two disagree by
        # the slow-mo factor and every attempt starts early.
        if not _wait_for_race_go(
            shared_data, label='PAD', on_tick=_pump_panel,
        ):
            _wait_aligned_to_countdown(
                shared_data, t_reset, _countdown_hold_s(args),
                label='PAD', need_vision=True, vision_grace_s=2.0,
                on_tick=_pump_panel,
            )
        controller.arm()
        shared_data['flight_started'] = True
        if hasattr(planner, 'reset_episode'):
            planner.reset_episode()
        roll_pid.reset()
        pitch_pid.reset()
        for axis in ('roll', 'pitch', 'yaw', 'thrust'):
            hold_state[axis] = 0.0
            hold_state[f'{axis}_t'] = 0.0
        print(f'[COACH] attempt {attempt["n"]} armed — mode={mode.upper()}',
              flush=True)
        if log:
            log('COACH', f'attempt={attempt["n"]} mode={mode}')

    _arm_attempt(time.monotonic() - float(getattr(config, 'SIM_RESET_SETTLE_S', 1.5)))

    last_t = None
    period = 1.0 / max(args.hz, 1.0)
    started = time.monotonic()
    autos_seen = 0
    humans_seen = 0
    resets_seen = 0
    keeps_seen = 0
    excluded = False
    shared_data['exclude'] = 0

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
                thrust_step=float(thrust_step),
                now=now,
                sink_step=float(sink_step),
                pitch_rad=pitch_rad,
            )
            if quit_req:
                print('\n[STOP] quit key', flush=True)
                break

            keeps = int(hold_state.get('keeps', 0))
            if keeps > keeps_seen:
                keeps_seen = keeps
                excluded = not excluded
                shared_data['exclude'] = 1 if excluded else 0
                print(
                    '[COACH] EXCLUDE ON — these frames will NOT be trained on'
                    if excluded else
                    '[COACH] EXCLUDE OFF — recording again',
                    flush=True,
                )
                if log:
                    log('COACH', f'exclude={int(excluded)}')

            resets = int(hold_state.get('resets', 0))
            if resets > resets_seen:
                resets_seen = resets
                print('\n[COACH] RESET — new attempt', flush=True)
                controller.send_sim_reset()
                t_reset = time.monotonic()
                time.sleep(max(0.5, float(
                    getattr(config, 'SIM_RESET_SETTLE_S', 1.5)
                )))
                mode = (
                    'human' if bool(getattr(args, 'start_human', False))
                    else 'policy'
                )
                shared_data['control_authority'] = mode
                shared_data['intervention_id'] = ''
                _arm_attempt(t_reset)
                last_t = None
                started = time.monotonic()
                continue

            humans = int(hold_state.get('human', 0))
            if humans > humans_seen:
                humans_seen = humans
                if mode != 'human':
                    mode = 'human'
                    intervention_id += 1
                    shared_data['control_authority'] = 'human'
                    shared_data['intervention_id'] = str(intervention_id)
                    roll_pid.reset()
                    pitch_pid.reset()
                    print(
                        f'[COACH] TELEOP NOW  intervention={intervention_id}  '
                        '(T returns policy)',
                        flush=True,
                    )
                    if log:
                        log('COACH', f'human intervention_id={intervention_id}')
                else:
                    print('[COACH] H ignored — already HUMAN', flush=True)

            autos = int(hold_state.get('auto', 0))
            if autos > autos_seen:
                autos_seen = autos
                if mode != 'policy':
                    mode = 'policy'
                    shared_data['control_authority'] = 'policy'
                    # Keep intervention_id on the recovery tail so train_policy
                    # can still mark the post-handoff frames via --tail-s.
                    roll_pid.reset()
                    pitch_pid.reset()
                    if hasattr(planner, 'reset_episode'):
                        # Keep history — only clear on crash/reset.
                        pass
                    print(
                        f'[COACH] POLICY  (released intervention '
                        f'{intervention_id})',
                        flush=True,
                    )
                    if log:
                        log('COACH', f'policy resume intervention_id={intervention_id}')
                else:
                    print('[COACH] T ignored — already POLICY', flush=True)

            dt = period if last_t is None else max(1e-3, now - last_t)
            last_t = now

            if mode == 'policy':
                planner.compute_target(shared_data)
                shared_data['planner_mode'] = planner.name
            else:
                roll, pitch, _, _ = read_attitude(shared_data)
                lean_boost = float(
                    getattr(config, 'LEAN_THRUST_BOOST', 0.0) or 0.0
                )
                if angle_mode:
                    roll_rate = roll_pid.update(des_roll - roll, dt)
                    pitch_rate = pitch_pid.update(des_pitch - pitch, dt)
                    thrust = _tilt_compensated_thrust(
                        config.HOVER_THRUST, des_roll, des_pitch,
                        lean_boost=lean_boost,
                    )
                    thr_lo, thr_hi = 0.06, 0.45
                else:
                    # Acro: the sticks ARE body rates. Centre stick holds the
                    # current attitude rather than levelling it, and collective
                    # is compensated against *measured* tilt.
                    roll_rate = float(des_roll)
                    pitch_rate = float(des_pitch)
                    thrust = _tilt_compensated_thrust(
                        config.HOVER_THRUST, roll, pitch,
                        lean_boost=lean_boost,
                    )
                    thr_lo = float(getattr(config, 'ACRO_THRUST_MIN', 0.05))
                    thr_hi = float(getattr(config, 'ACRO_THRUST_MAX', 0.70))
                thrust = float(max(
                    thr_lo, min(thr_hi, thrust + float(thrust_delta))
                ))
                shared_data['planner_target'] = {
                    'kalman': True,
                    'acro': not angle_mode,
                    'unrestricted_rates': not angle_mode,
                    'roll_rate': roll_rate,
                    'pitch_rate': pitch_rate,
                    'yaw_rate': yaw_rate,
                    'thrust': thrust,
                    'desired_roll': roll if not angle_mode else des_roll,
                    'desired_pitch': pitch if not angle_mode else des_pitch,
                }
                shared_data['planner_mode'] = (
                    'coach_human_angle' if angle_mode else 'coach_human_acro'
                )

            controller.update()
            if panel is not None and not panel.show(shared_data):
                panel = None
            if not args.quiet and int(elapsed * 2) != int((elapsed - period) * 2):
                race = shared_data.get('race_status') or {}
                print(
                    f'[COACH] t={elapsed:5.1f}s  auth={shared_data.get("control_authority")}  '
                    f'interv={shared_data.get("intervention_id") or "-"}  '
                    f'gate={race.get("active_gate", "?")}',
                    flush=True,
                )
            time.sleep(period)
    except KeyboardInterrupt:
        print('\n[STOP] interrupted', flush=True)
    finally:
        try:
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
        if panel is not None:
            panel.close()
        shutdown(components)

    logger = components.get('logger')
    if logger is not None:
        print(f'\ntelem: {getattr(logger, "_csv_path", "?")}')
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


def _pilot_lock_info(
    shared_data,
    *,
    min_area: float = 1200.0,
    max_area: float | None = None,
    min_range: float = 1.5,
) -> dict:
    """YOLO/PnP lock status for pilot and observe-only acro recording.

    ``max_area``/``min_range`` reject a box that is too close to be the *next*
    gate. Asked for a lock at the instant a gate is cleared, the camera is
    still full of the gate being flown through: that box reads area ~10000 and
    dead-centre, so an unbounded check locks onto the gate already behind us.
    """
    from assist_planner import image_gate_norm

    nx, ny, src = image_gate_norm(shared_data)
    det = shared_data.get('gate_detection') or {}
    dual = shared_data.get('dual_gate_pnp') or {}
    area = _f(det.get('area_px')) if isinstance(det, dict) else None
    rng = _f(dual.get('gate1_range_m'))
    yolo_ok = (
        src == 'yolo'
        and nx is not None
        and ny is not None
        and area is not None
        and float(area) >= float(min_area)
        and (max_area is None or float(area) <= float(max_area))
    )
    pnp_ok = (
        src == 'dual_pnp'
        and nx is not None
        and ny is not None
        and rng is not None
        and float(min_range) <= float(rng) <= 30.0
    )
    locked = bool(yolo_ok or pnp_ok)
    return {
        'locked': locked,
        'nx': nx,
        'ny': ny,
        'source': src,
        'area_px': area,
        'range_m': rng,
    }


def _pilot_passed_gate(shared_data) -> int:
    """Best monotonic gate number published by race or vision state."""
    vals = []
    race = shared_data.get('race_status') or {}
    for value in (
        race.get('active_gate'),
        shared_data.get('last_gate_passed'),
    ):
        try:
            vals.append(int(value))
        except (TypeError, ValueError):
            pass
    return max(vals, default=0)


def _seed_assist_from_lock(planner, shared_data, now: float) -> bool:
    """Prime AssistImagePlanner to chase the currently locked gate."""
    import config
    import numpy as np
    from assist_planner import image_gate_norm

    nx, ny, src = image_gate_norm(shared_data)
    if nx is None or ny is None:
        return False
    det = shared_data.get('gate_detection') or {}
    dual = shared_data.get('dual_gate_pnp') or {}
    area = _f(det.get('area_px')) if isinstance(det, dict) else None
    rng = _f(dual.get('gate1_range_m'))
    if rng is None and area is not None and area > 50.0:
        rng = float((320.0 * 1.5) / math.sqrt(area))
    body = dual.get('gate1_body')

    # CRITICAL: do NOT set _arm_z to the current airborne z. That makes
    # climbed≈0 at handoff, so _seek_ny_thrust keeps returning seek_floor
    # (hover+boost) and the craft loft instead of chasing the locked gate.
    # NED origin is the arm point — prefer recorded pad z, else 0.
    pad_z = _f(shared_data.get('pilot_pad_z'))
    if pad_z is not None:
        planner._arm_z = float(pad_z)
    else:
        planner._arm_z = 0.0
    climbed_now = planner._climb_m(shared_data)
    if climbed_now < 0.2:
        # Fallback if pad_z missing / odom odd: use HUD climb estimate.
        est = _climb_estimate(shared_data)
        if est is not None and float(est) > climbed_now:
            z_now = None
            for key in ('local_position_ned', 'position_ned'):
                z = (shared_data.get(key) or {}).get('z')
                if z is not None and math.isfinite(float(z)):
                    z_now = float(z)
                    break
            if z_now is not None:
                planner._arm_z = float(z_now) + float(est)
                climbed_now = float(est)
    planner._climb_f = float(climbed_now)
    planner._peak_climbed = max(
        float(getattr(planner, '_peak_climbed', 0.0) or 0.0),
        float(climbed_now),
    )

    planner._have_filt = True
    planner._nx_f = float(nx)
    planner._ny_f = float(ny)
    planner._last_see_t = float(now)
    planner._seek_until = float(now) + float(
        getattr(config, 'ASSIST_SEEK_S', 14.0)
    )
    planner._coast_until = 0.0
    planner._seek_seen = True
    planner._gate_lock = True
    planner._lock_count = 99
    planner._lock_nx = float(nx)
    planner._lock_ok_t = float(now)
    planner._left_pad = True
    planner._airborne_t = float(now) - 1.0
    planner._lift_start_t = float(now) - 2.0
    planner._last_range_m = rng
    # Must be ndarray — list breaks (1-a)*_body_f + a*body_raw in assist.
    if body is not None:
        try:
            planner._body_f = np.asarray(
                [float(body[0]), float(body[1]), float(body[2])],
                dtype=float,
            ).copy()
        except (TypeError, ValueError, IndexError):
            planner._body_f = None
    else:
        planner._body_f = None
    # Pilot hands a live lock — do not inject course-2 right-yaw memory.
    planner._course_mem = False
    planner._course_mem_yaw_tgt = None
    planner._course_mem_spent = True
    planner._course_mem_done = True
    planner._yaw_pid.reset()
    planner._last_yaw_cmd = 0.0
    print(
        f'[PILOT] seeded assist from {src} '
        f'nx={float(nx):+.3f} ny={float(ny):+.3f} rng={rng} '
        f'climb={climbed_now:.2f}m arm_z={planner._arm_z}',
        flush=True,
    )
    return True


def _pilot_beep() -> None:
    """Audible lock cue on Windows; no-op elsewhere."""
    try:
        import winsound
        winsound.Beep(1200, 120)
    except Exception:
        print('\a', end='', flush=True)


def _countdown_hold_s(args) -> float:
    """Seconds from sim-reset to arm (match the ~3s race countdown)."""
    import config as _cfg
    v = getattr(args, 'early_start_hold_s', None)
    if v is None:
        v = getattr(_cfg, 'EARLY_START_HOLD_S', 3.0)
    return max(0.0, float(v))


def _wait_for_race_go(
    shared_data,
    *,
    timeout_s: float = 90.0,
    label: str = 'PAD',
    on_tick=None,
) -> bool:
    """Block until the sim's race clock says GO, not until a wall-clock guess.

    The countdown runs on *sim* time. ``EARLY_START_HOLD_S`` is a wall-clock
    approximation of it, which is correct at 1x and badly wrong under client
    slow-mo: a 3.3 sim-second countdown takes 16.5 s of wall time at 0.2x, so a
    3.55 s hold arms roughly thirteen seconds early, every run.

    The simulator publishes the answer. From MANUAL_INSTRUCTIONS: ``sim_boot_ms``
    advances in sim time and ``race_start_ms`` is -1 while pending, then the
    future ``sim_boot_ms`` of GO. So GO is exactly
    ``sim_boot_ms >= race_start_ms``.

    That test cannot be applied naively, because until the 31000 reset lands the
    sim still reports the *previous* race, whose start time is long past -- the
    test would pass instantly. Hence three phases: wait for the reset to land,
    wait for a countdown to be scheduled and still pending, then wait for GO.

    At 1x the 1.5 s reset settle often skips the ``start == -1`` blip entirely
    and we arrive already inside a fresh countdown (``boot < start``). That is
    also "reset has landed" — waiting only for ``start < 0`` hung the coach
    loop for 90 s and the panel never called ``imshow``.
    """
    deadline = time.monotonic() + float(timeout_s)

    def _race():
        return shared_data.get('race_status') or {}

    def _pair():
        r = _race()
        try:
            return int(r.get('sim_boot_ms')), int(r.get('race_start_ms'))
        except (TypeError, ValueError):
            return None, None

    def _tick() -> None:
        if on_tick is not None:
            on_tick()

    # Phase 1: reset has landed. Either the previous start cleared, or a
    # new countdown is already pending (the 1x case after settle).
    while time.monotonic() < deadline:
        boot, start = _pair()
        if start is not None and start < 0:
            break
        if (
            boot is not None and start is not None
            and start >= 0 and boot < start
        ):
            break
        _tick()
        time.sleep(0.02)

    # Phase 2: a countdown exists and has not fired yet.
    pending = False
    while time.monotonic() < deadline:
        boot, start = _pair()
        if boot is not None and start is not None and start >= 0:
            if boot < start:
                pending = True
                print(f'[{label}] countdown scheduled: '
                      f'{(start - boot) / 1000.0:.2f} sim-seconds to GO',
                      flush=True)
                break
            # Already past: no fresh countdown, fall through and fly.
            print(f'[{label}] no pending countdown — starting now', flush=True)
            return True
        _tick()
        time.sleep(0.02)

    if not pending:
        print(f'[{label}] race clock unavailable — falling back to wall hold',
              flush=True)
        return False

    # Phase 3: GO.
    while time.monotonic() < deadline:
        boot, start = _pair()
        if boot is not None and start is not None and boot >= start:
            print(f'[{label}] GO', flush=True)
            return True
        _tick()
        time.sleep(0.01)
    print(f'[{label}] timed out waiting for GO', flush=True)
    return False


def _wait_aligned_to_countdown(
    shared_data,
    t0: float,
    hold_s: float,
    *,
    label: str = 'PAD',
    need_vision: bool = True,
    vision_grace_s: float = 2.0,
    on_tick=None,
) -> bool:
    """Wait until ``t0 + hold_s`` (from reset), optionally polling vision.

    Critical: do **not** sleep settle/vision *then* sleep hold again — that
    stacks ~1s of dead sticks after the on-screen countdown hits GO.
    """
    deadline = float(t0) + float(hold_s)
    remaining = max(0.0, deadline - time.monotonic())
    print(
        f'[SIM] GO in {remaining:.2f}s '
        f'(aligned to countdown from reset, hold={float(hold_s):.2f}s)',
        flush=True,
    )
    ready = False
    while True:
        if on_tick is not None:
            on_tick()
        if need_vision:
            dual = shared_data.get('dual_gate_pnp') or {}
            det = shared_data.get('gate_detection') or {}
            if dual.get('gate1_body') is not None or (
                isinstance(det, dict) and det.get('center_px') is not None
            ):
                if not ready:
                    print(f'[{label}] ready', flush=True)
                    ready = True
        now = time.monotonic()
        if now >= deadline:
            break
        time.sleep(min(0.02, max(0.0, deadline - now)))
    if ready or not need_vision:
        return True
    grace_end = time.monotonic() + max(0.0, float(vision_grace_s))
    while time.monotonic() < grace_end:
        dual = shared_data.get('dual_gate_pnp') or {}
        det = shared_data.get('gate_detection') or {}
        if dual.get('gate1_body') is not None or (
            isinstance(det, dict) and det.get('center_px') is not None
        ):
            print(f'[{label}] ready', flush=True)
            return True
        time.sleep(0.05)
    return False


def run_pilot(args) -> int:
    """You fly; YOLO stays on. T hands the locked gate to assist; H returns sticks.

    Optional remember-path: ``--capture`` records the flight; ``--replay`` +
    ``--keep-until-gate N`` flies the captured spline through gate N then
    hands sticks back so T/H work for the rest of the course.

    ``fly`` / ``pilot --pure``: stick-only ANGLE mode; mid-run zero-attitude
    is logged on the practice attitude tape as an ``events`` entry.

    ``acro``: rate mode — sticks are body rates, no self-level / lean caps.
    """
    import config
    from control.pid import PIDConfig, PIDController
    from setup import setup_components
    from attitude_tape import AttitudeTapeRecorder
    from practice_store import PRACTICE_DIR, maybe_update_through_gate, save_run

    acro_mode = getattr(args, 'mode', None) == 'acro'
    acro_vision = acro_mode and not bool(getattr(args, 'no_vision', False))
    pure_fly = (
        bool(getattr(args, 'pure', False))
        or getattr(args, 'mode', None) == 'fly'
        or acro_mode
    )
    if pure_fly:
        config.EKF_USE_PNP = False
        config.EKF_GATE_HORIZON_GAIN = 0.0
        config.EKF_GATE_YAW_GAIN = 0.0
        config.PILOT_LEVEL_AHRS = 0
        config.PILOT_LEVEL_REALIGN_S = 0.0
        os.environ['EKF_USE_PNP'] = '0'
        os.environ['TAKEOFF_DURATION_S'] = '0'
        config.TAKEOFF_DURATION_S = 0.0
    if acro_mode:
        # Full-stick body rates — no soft-gain pad attenuation, no yaw clip.
        yaw_cap = float(
            getattr(args, 'yaw_rate_deg', None)
            or getattr(config, 'ACRO_YAW_RATE_DEG', 400.0)
            or 400.0
        )
        config.YAW_RATE_MAX_RAD_S = math.radians(max(720.0, yaw_cap * 1.25))
        config.PILOT_PAD_SOFT_GAIN = 1.0
        config.PILOT_PAD_SOFT_YAW = 1.0
        config.PILOT_PAD_SOFT_THRUST = 1.0
        if acro_vision:
            # Detection and raw capture stay on; skip OpenCV rendering/window
            # work so observation perturbs the manual control cadence less.
            config.VISION_DISPLAY = False

    shared_data = {}
    if acro_vision:
        # Keep a sparse raw-image record even when the live detector misses.
        # At the default 3/s this is small enough for a full CE-0.2 run and
        # allows improved models to reprocess the reference offline.
        shared_data['vision_reference_capture_all'] = True
    components = setup_components(
        shared_data, int(time.time() * 1000),
        SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT,
        # Acro vision is observation only: EKF_USE_PNP remains false, and
        # manual body-rate commands are sent verbatim. Skip constructing the
        # FLIGHT_MODE planner (assist pulls optional learner modules we do
        # not need for seed logging).
        enable_vision=(not pure_fly) or acro_vision,
        enable_planner=not pure_fly,
    )
    controller = components['controller']
    state_estimator = components.get('state_estimator')
    logger = components.get('logger')

    # Client slow-mo (seed laps at CE 0.2): scale telem rate so H=32 spans
    # ~0.64 s of sim time, matching 50 Hz at 1x.
    slow_mo_on = bool(getattr(args, 'slow_mo', False)) or bool(
        getattr(config, 'PILOT_SLOW_MO', 0)
    )
    if getattr(args, 'slow_mo_scale', None) is not None:
        slow_mo_on = True
        slow_mo_scale = float(args.slow_mo_scale)
    else:
        slow_mo_scale = float(getattr(config, 'PILOT_SLOW_MO_SCALE', 0.77) or 0.77)
    slow_mo_scale = max(0.05, min(1.0, slow_mo_scale))
    if slow_mo_on:
        shared_data['log_hz'] = max(5.0, 50.0 * slow_mo_scale)
    else:
        shared_data['log_hz'] = float(os.environ.get('LOG_HZ', '50') or 50)

    if acro_mode:
        log_event = shared_data.get('log_event')
        if log_event:
            log_event(
                'VISION',
                'acro_observe_only=1 ekf_pnp=0 control_feedback=0 display=0'
                if acro_vision else
                'acro_observe_only=0 (--no-vision)',
            )
        # HG-DAgger seed provenance — entire acro run is human.
        shared_data['control_authority'] = 'human'
        shared_data['intervention_id'] = ''
        if logger is not None:
            shared_data['_telem_path'] = getattr(logger, '_csv_path', None)
        _seed_logging_preflight(
            shared_data,
            slow_mo_scale=slow_mo_scale if slow_mo_on else None,
        )
        if logger is not None:
            print(f'  telem file: {logger._csv_path}', flush=True)
            if not acro_vision:
                print(
                    '  WARNING: --no-vision — no keypoints; unusable for '
                    'policy training.',
                    flush=True,
                )
            print('', flush=True)
    planner = None
    if not pure_fly:
        from assist_planner import AssistImagePlanner
        planner = AssistImagePlanner()
        shared_data['planner'] = planner
    else:
        shared_data['planner'] = None
    csv_label = 'acro' if acro_mode else ('fly' if pure_fly else 'pilot')
    recorder = Recorder(
        Path(args.csv) if args.csv else default_csv_path(csv_label)
    )
    practice_auto = (
        bool(int(getattr(config, 'PRACTICE_AUTO_SAVE', 1) or 0))
        and not bool(getattr(args, 'no_practice_save', False))
    )
    practice_rec: AttitudeTapeRecorder | None = None
    if practice_auto:
        PRACTICE_DIR.mkdir(parents=True, exist_ok=True)
        practice_rec = AttitudeTapeRecorder(
            name='practice_session',
            control='acro_rates' if acro_mode else 'angle',
            metadata=(
                {
                    'vision_observe_only': bool(acro_vision),
                    'sample_clock': 'perf_counter',
                    'reference_channels': [
                        'vision', 'race', 'attitude', 'gyro', 'accel', 'wire',
                    ],
                    'gate_frame_capture': dict(
                        shared_data.get('gate_frame_capture') or {}
                    ),
                }
                if acro_mode else None
            ),
        )

    if acro_mode:
        # Stick axes are body rates (°/s), not lean angles — no MAX_LEAN cap.
        lean_deg = getattr(args, 'roll_rate_deg', None)
        if lean_deg is None:
            lean_deg = float(
                getattr(config, 'ACRO_ROLL_RATE_DEG', 400.0) or 400.0
            )
        pitch_deg = getattr(args, 'pitch_rate_deg', None)
        if pitch_deg is None:
            pitch_deg = float(
                getattr(config, 'ACRO_PITCH_RATE_DEG', 400.0) or 400.0
            )
        yaw_deg = getattr(args, 'yaw_rate_deg', None)
        if yaw_deg is None:
            yaw_deg = float(
                getattr(config, 'ACRO_YAW_RATE_DEG', 400.0) or 400.0
            )
    else:
        lean_deg = getattr(args, 'lean_deg', None)
        if lean_deg is None:
            lean_deg = float(getattr(config, 'PILOT_LEAN_DEG', 38.0) or 38.0)
        pitch_deg = float(
            getattr(config, 'PILOT_PITCH_LEAN_DEG', lean_deg) or lean_deg
        )
        # Hard ceiling — keep commanded lean inside MAX_LEAN_RAD.
        max_lean_deg = math.degrees(
            float(getattr(config, 'MAX_LEAN_RAD', 1.0))
        )
        lean_deg = min(float(lean_deg), max_lean_deg)
        pitch_deg = min(float(pitch_deg), max_lean_deg)
        yaw_deg = getattr(args, 'yaw_rate_deg', None)
        if yaw_deg is None:
            yaw_deg = float(
                getattr(config, 'PILOT_YAW_RATE_DEG', 85.0) or 85.0
            )
    climb_rate_cmd = getattr(args, 'climb_rate', None)
    if climb_rate_cmd is None:
        climb_rate_cmd = float(
            getattr(config, 'PILOT_CLIMB_RATE', 1.8) or 1.8
        )
    climb_auth = getattr(args, 'climb_auth', None)
    if climb_auth is None:
        if acro_mode:
            climb_auth = float(
                getattr(config, 'ACRO_CLIMB_AUTH', 0.55) or 0.55
            )
        else:
            climb_auth = float(
                getattr(config, 'PILOT_CLIMB_AUTH', 0.15) or 0.15
            )
    lean_rad = math.radians(float(lean_deg))
    pitch_rad = math.radians(float(pitch_deg))
    yaw_rate_cmd = math.radians(float(yaw_deg))
    # Acro: triggers/R-F are absolute collective offsets (thrust levels),
    # not climb/sink rates around hover.
    open_loop = bool(getattr(args, 'open_loop_thrust', False)) or acro_mode
    climb_rate_cmd = float(climb_rate_cmd)
    sink_rate_cmd = float(
        getattr(config, 'PILOT_SINK_RATE', climb_rate_cmd) or climb_rate_cmd
    )
    if acro_mode:
        sink_auth = float(getattr(config, 'ACRO_SINK_AUTH', 0.55) or 0.55)
    else:
        sink_auth = float(
            getattr(config, 'PILOT_SINK_AUTH', climb_auth) or climb_auth
        )
    rate_stick = not open_loop
    if acro_mode:
        thrust_step = float(climb_auth)
        sink_rate_cmd = float(sink_auth)
    else:
        thrust_step = (
            float(getattr(args, 'thrust_step', 0.022))
            if open_loop else climb_rate_cmd
        )
    use_rate_hold = (
        rate_stick
        and bool(int(getattr(config, 'PILOT_RATE_HOLD', 0) or 0))
    )
    vrate = None if (open_loop or not use_rate_hold) else VerticalRateHold(
        kp=getattr(args, 'climb_kp', None),
        ki=getattr(args, 'climb_ki', None),
        authority=float(climb_auth),
    )
    min_area = float(getattr(args, 'lock_area', 1200.0) or 1200.0)
    # Hotter rate ceiling than assist — left↔right reverse speed.
    pilot_rate_deg = float(
        getattr(config, 'PILOT_MAX_RATE_DEG', 0.0) or 0.0
    )
    if pilot_rate_deg > 1.0:
        max_rate = math.radians(pilot_rate_deg)
    else:
        max_rate = float(config.KALMAN_MAX_RATE_RAD_S)
    roll_pid = PIDController(PIDConfig(
        kp=config.KALMAN_KP_ATT, kd=config.KALMAN_KD_ATT,
        output_min=-max_rate, output_max=max_rate,
    ))
    pitch_pid = PIDController(PIDConfig(
        kp=config.KALMAN_KP_ATT, kd=config.KALMAN_KD_ATT,
        output_min=-max_rate, output_max=max_rate,
    ))

    cap_arg = getattr(args, 'capture', None)
    replay_path = getattr(args, 'replay', None)
    attitude_path = getattr(args, 'replay_attitude', None)
    if attitude_path and replay_path:
        print('[FAIL] use only one of --replay / --replay-attitude', flush=True)
        return 1
    att_clock = None
    att_acro_rates = False
    att_start_delay_wall_s = 0.0
    if attitude_path:
        from attitude_tape import AttitudeTapeClock, load_attitude_tape
        try:
            att_clock = AttitudeTapeClock(load_attitude_tape(attitude_path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f'[FAIL] attitude tape: {exc}', flush=True)
            return 1
        att_acro_rates = str(
            att_clock.tape.get('control') or ''
        ).lower() == 'acro_rates'
        att_record_speed = float(att_clock.tape.get('sim_speed') or 1.0)
        att_start_offset_s = float(
            att_clock.tape.get('race_start_offset_s') or 0.0
        )
        att_start_delay_wall_s = max(0.0, att_start_offset_s) / max(
            1e-6, att_record_speed
        )
        print(
            f'[REPLAY] attitude tape {attitude_path}  '
            f'{att_clock.duration:.1f}s  n={len(att_clock.samples)}  '
            f'control={"acro_rates (exact/ZOH)" if att_acro_rates else "angle"}  '
            f'start={att_start_offset_s * 1000:.1f}ms after GO',
            flush=True,
        )
    assist_after = getattr(args, 'assist_after_gate', None)
    if assist_after is None:
        configured = int(getattr(config, 'PILOT_ASSIST_AFTER_GATE', 0) or 0)
        assist_after = configured if configured > 0 else None
    human_after = getattr(args, 'human_after_gate', None)
    if human_after is None:
        configured = int(getattr(config, 'PILOT_HUMAN_AFTER_GATE', 0) or 0)
        human_after = configured if configured > 0 else None
    if assist_after is not None:
        assist_after = int(assist_after)
        if human_after is None:
            human_after = assist_after + 1
        human_after = int(human_after)
        if not (replay_path or attitude_path):
            print('[FAIL] --assist-after-gate needs a replay source', flush=True)
            return 1
        if planner is None:
            print(
                '[FAIL] replay-to-assist needs pilot vision (not fly/acro/pure)',
                flush=True,
            )
            return 1
        if human_after <= assist_after:
            print(
                '[FAIL] --human-after-gate must be after --assist-after-gate',
                flush=True,
            )
            return 1
    keep_until = getattr(args, 'keep_until_gate', None)
    # Key press timeline: record which keys are held and for how long.
    from remember_timeline import (
        KeyReplayClock,
        KeyTimeline,
        apply_keys_to_hold_state,
        load_timeline,
    )
    remember_mode = bool(replay_path) or (cap_arg is not None)
    save_path = (
        (cap_arg if cap_arg not in (None, '') else None)
        or replay_path
        or getattr(config, 'REMEMBER_PATH', 'captured_controls.json')
    )
    timeline = None if not remember_mode else KeyTimeline(save_path)
    memorizing = timeline is not None and not bool(replay_path)
    mem_t0 = None
    commit_only_on_k = bool(replay_path)
    keeps_seen = 0

    replay_clock = None
    if replay_path:
        full_tl = load_timeline(replay_path)
        if full_tl is None:
            print(
                f'[FAIL] cannot load key timeline {replay_path!r} '
                '(need a fresh pilot --capture; type=key_timeline)',
                flush=True,
            )
            shutdown(components)
            return 1
        play_tl = full_tl
        post_yaw_rate = max(1.0, math.degrees(abs(yaw_rate_cmd)))
        frozen_g = int(getattr(full_tl, 'frozen_through_gate', 0) or 0)
        if frozen_g > 0:
            print(
                f'[REMEMBER] frozen_through_gate={frozen_g} — '
                'playing keys as-is (no rewrite)',
                flush=True,
            )
        else:
            yaw_msgs = full_tl.ensure_pilot_gate_yaws(
                yaw_rate_deg=post_yaw_rate
            )
            if yaw_msgs:
                for msg in yaw_msgs:
                    print(f'[REMEMBER] replay: inserted {msg}', flush=True)
                try:
                    full_tl.save()
                    print(
                        f'[REMEMBER] wrote gate yaws into {full_tl.path}',
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f'[REMEMBER] could not rewrite capture: {exc}',
                        flush=True,
                    )
        if keep_until is not None:
            try:
                play_tl, trim_warn = full_tl.trim_until_gate(int(keep_until))
            except ValueError as exc:
                print(f'[FAIL] keep-until-gate: {exc}', flush=True)
                shutdown(components)
                return 1
            if trim_warn:
                print(f'[WARN] {trim_warn}', flush=True)
            else:
                print(
                    f'[REMEMBER] trimmed through GATE {int(keep_until)}: '
                    f'{len(play_tl)} events, {play_tl.duration:.1f}s '
                    f'(keys held ~{play_tl.key_hold_s():.1f}s)',
                    flush=True,
                )
            # Only re-bake yaws when the capture is not frozen.
            if frozen_g <= 0:
                for msg in play_tl.ensure_pilot_gate_yaws(
                    yaw_rate_deg=post_yaw_rate
                ):
                    print(f'[REMEMBER] after trim: {msg}', flush=True)
        ekf_cap = full_tl.ekf_use_pnp
        if ekf_cap is not None and int(ekf_cap) != int(bool(config.EKF_USE_PNP)):
            print(
                f'[WARN] capture ekf_use_pnp={ekf_cap} but live '
                f'EKF_USE_PNP={int(bool(config.EKF_USE_PNP))} — match them.',
                flush=True,
            )
        try:
            replay_clock = KeyReplayClock(play_tl)
        except ValueError as exc:
            print(f'[FAIL] remember replay: {exc}', flush=True)
            shutdown(components)
            return 1
        # Fresh vertical PI each replay so odom noise from the last run
        # cannot bias F/R thrust on this one (keys stay the same).
        if vrate is not None:
            vrate.reset()
            print('[REMEMBER] climb PI reset for deterministic key replay', flush=True)
        if timeline is not None:
            n_seed = timeline.seed_prefix(play_tl)
            print(
                f'[REMEMBER] seeded {n_seed} key events through gate '
                f'{keep_until if keep_until is not None else "end"}; '
                'will record YOUR keys after handoff',
                flush=True,
            )

    print('', flush=True)
    if acro_mode:
        print(
            '=== ACRO (rate mode — no angle limits / no self-level) ===',
            flush=True,
        )
    elif pure_fly:
        print('=== FLY (pure stick — no vision / assist / corrections) ===',
              flush=True)
    else:
        print('=== PILOT (manual ↔ auto) ===', flush=True)
    print('  Focus THIS console for keys (pad works in background too).',
          flush=True)
    if acro_mode:
        print(
            '  Manual: WASD/arrows rates, Q/E yaw, R/F climb/sink, '
            'Space zero-rates',
            flush=True,
        )
    else:
        print(
            '  Manual: WASD/arrows lean, Q/E yaw, R/F climb/sink, Space level',
            flush=True,
        )
    if acro_mode:
        print(
            '  Pad: L=roll/pitch  Rx=yaw  RT/LT=thrust  RB=thrust+',
            flush=True,
        )
        print(
            '       A zero-rates  B quit  X/Z emergency-level  Y reset',
            flush=True,
        )
    else:
        print(
            '  Pad: L=roll/pitch  Rx=yaw  RT=climb  LT=sink  RB=thrust+',
            flush=True,
        )
        if pure_fly:
            print(
                '       A stick-level  B quit  X/Z zero-attitude  Y reset',
                flush=True,
            )
        else:
            print(
                '       A level  B quit  X human  Y reset  LB auto  Start keep',
                flush=True,
            )
    if acro_mode:
        print(
            f'  Rate caps: roll={math.degrees(lean_rad):.0f}°/s  '
            f'pitch={math.degrees(pitch_rad):.0f}°/s  '
            f'yaw={math.degrees(yaw_rate_cmd):.0f}°/s  '
            f'thrust=+{thrust_step:.2f}/-{sink_rate_cmd:.2f}  '
            f'HT={config.HOVER_THRUST:.3f}',
            flush=True,
        )
    else:
        print(
            f'  Soft caps: roll={math.degrees(lean_rad):.0f}°  '
            f'pitch={math.degrees(pitch_rad):.0f}°  '
            f'yaw={math.degrees(yaw_rate_cmd):.0f}°/s  '
            f'rate={math.degrees(max_rate):.0f}°/s  '
            f'climb={climb_rate_cmd:.2f} m/s  '
            f'sink_auth={sink_auth:.2f}  HT={config.HOVER_THRUST:.3f}',
            flush=True,
        )
    if acro_mode:
        print(
            '  Vertical: RT/R / LT/F = collective thrust levels '
            f'(+{thrust_step:.2f} / -{sink_rate_cmd:.2f} on hover)',
            flush=True,
        )
    elif use_rate_hold:
        print('  Vertical: EKF rate-hold (PILOT_RATE_HOLD=1)', flush=True)
    else:
        print(
            '  Vertical: direct collective (LT/F always cuts thrust)',
            flush=True,
        )
    if acro_mode:
        print(
            '  RATE MODE: sticks = body rates. Center stick stops rotation '
            'but does NOT self-level. No lean/angle caps.',
            flush=True,
        )
        print(
            (
                '  Vision: YOLO/PnP ON, OBSERVE-ONLY + recorded; no EKF or '
                'control influence (headless)'
                if acro_vision else
                '  Vision: OFF (--no-vision)'
            ),
            flush=True,
        )
        print(
            '  Off: assist, vision-to-EKF corrections, AHRS blend, EKF level '
            'realign, angle attitude loop',
            flush=True,
        )
        print(
            '  A / Space    zero rates only (craft keeps its attitude)',
            flush=True,
        )
        print(
            '  X / Z / H    emergency EKF level (optional; not needed for acro)',
            flush=True,
        )
    elif pure_fly:
        print(
            '  Off: YOLO/pose, assist (T), AHRS blend, EKF level realign, '
            'gate attitude aids',
            flush=True,
        )
        print(
            '  X / Z / H    ZERO ATTITUDE — declare current pose as level '
            '(clears EKF roll/pitch drift; keep yaw)',
            flush=True,
        )
        print(
            '  A / Space    stick-level only (desired lean=0; does not '
            're-zero the EKF)',
            flush=True,
        )
    else:
        print('  T            AUTO — assist flies the LOCKED gate', flush=True)
        print('  H            HUMAN — you take the sticks again', flush=True)
    print(
        '  Y / pad Y    RESET — sim pad; arms on your next stick/key',
        flush=True,
    )
    print(
        '  O / D-pad ↓  SLOW-MO — client time scale '
        f'(x{float(getattr(config, "PILOT_SLOW_MO_SCALE", 0.5) or 0.5):.2f}; '
        'match CE/DxWnd)',
        flush=True,
    )
    if att_clock is not None:
        print(
            f'  ATTITUDE REPLAY {att_clock.duration:.1f}s '
            f'(H aborts to sticks)',
            flush=True,
        )
    try:
        from gamepad_input import get_gamepad
        get_gamepad()  # announce once if connected
    except Exception:
        pass
    if replay_clock is not None:
        print(
            '  REPLAY/AUTO: sticks locked, but hold WASD to carry into HUMAN',
            flush=True,
        )
    if timeline is not None:
        print('  K            KEEP — save the key presses you like', flush=True)
        print(
            f'  remember -> {timeline.path}  (which keys + how long)',
            flush=True,
        )
    if replay_clock is not None:
        ku = keep_until if keep_until is not None else 'end'
        if assist_after is not None:
            ha = human_after if human_after is not None else int(assist_after) + 1
            print(
                f'  REPLAY keys through GATE {ku} → ASSIST (after GATE '
                f'{int(assist_after)}) → HUMAN (after GATE {ha}) '
                f'({len(replay_clock.timeline)} events, '
                f'{replay_clock._t_end:.1f}s)',
                flush=True,
            )
            print(
                '  ASSIST tip-through is closed-loop; after HUMAN your keys APPEND',
                flush=True,
            )
        else:
            print(
                f'  REPLAY your keys through gate {ku} then HUMAN '
                f'({len(replay_clock.timeline)} events, '
                f'{replay_clock._t_end:.1f}s)',
                flush=True,
            )
            print(
                '  After GATE passes: sticks → you; your keys APPEND to remember',
                flush=True,
            )
        print(
            '  Fly the next segment, then K to keep the extended timeline',
            flush=True,
        )
        print(
            f'  F sink: {float(getattr(config, "PILOT_SINK_RATE", 0.6)):.2f} m/s '
            f'until GATE 1, then '
            f'{float(getattr(config, "PILOT_G2_SINK_RATE", 1.0)):.2f} m/s '
            f'(PILOT_G2_SINK_RATE)',
            flush=True,
        )
    print('  Esc / X      disarm and quit', flush=True)
    print(
        f'  roll={math.degrees(lean_rad):.0f}°  '
        f'pitch={math.degrees(pitch_rad):.0f}°  '
        f'yaw={math.degrees(yaw_rate_cmd):.0f}°/s  '
        f'lock_area>={min_area:.0f}  HT={config.HOVER_THRUST:.3f}',
        flush=True,
    )
    print('', flush=True)

    # Seed LOCK banner before the control loop.
    with shared_data['lock']:
        shared_data['pilot_lock'] = {
            'locked': False,
            'mode': 'manual',
            'source': 'none',
            'nx': None,
            'ny': None,
            'area_px': None,
            'range_m': None,
            't': time.monotonic(),
        }

    if bool(getattr(args, 'no_sim_reset', False)):
        t_reset = time.monotonic()
        print('[SIM] no reset', flush=True)
    else:
        print('[SIM] reset (command 31000)...', flush=True)
        t_reset = time.monotonic()
        controller.send_sim_reset()
        time.sleep(0.25)

    # Pad NED z for assist climb math after T (must be pad, not airborne).
    pad_z = None
    for key in ('local_position_ned', 'position_ned'):
        z = (shared_data.get(key) or {}).get('z')
        if z is not None and math.isfinite(float(z)):
            pad_z = float(z)
            break
    if pad_z is None:
        pad_z = 0.0
    shared_data['pilot_pad_z'] = pad_z

    hold_state: dict = {}
    mode = 'attitude' if att_clock is not None else 'manual'
    autos_seen = 0
    humans_seen = 0
    zeros_seen = 0
    resets_seen = 0
    was_locked = False
    last_pad_reset = 0.0
    last_t = None
    att_last_idx = None
    handoff_wait_announced = False
    # Wall-rate loop; see run_manual. Slow-mo must not stretch this period or
    # stick polling becomes visibly laggy, and `elapsed` also drives the
    # attitude-tape playhead, which does its own speed inference.
    period = 1.0 / max(args.hz, 1.0)
    started = time.monotonic()
    slowmos_seen = 0
    # Deliberate stick/key gate — auto-arm after countdown early-start DQs.
    engage_frac = max(
        0.05,
        float(getattr(config, 'PILOT_ENGAGE_FRAC', 0.12) or 0.12),
    )
    engage_need = max(
        1, int(getattr(config, 'PILOT_ENGAGE_TICKS', 3) or 3)
    )
    engage_streak = {'n': 0, 'why': ''}
    # Attitude-tape replay is itself the input → arm after countdown.
    # Manual / fly: stay disarmed until YOU tip a stick or key.
    pilot_engaged = mode != 'manual'
    if pilot_engaged:
        hold_s = _countdown_hold_s(args)
        if pure_fly:
            _wait_aligned_to_countdown(
                shared_data, t_reset, hold_s,
                label='PAD', need_vision=False, vision_grace_s=0.0,
            )
        else:
            if not _wait_aligned_to_countdown(
                shared_data, t_reset, hold_s,
                label='PAD', need_vision=True,
            ):
                print('[FAIL] no gate in view', flush=True)
                shutdown(components)
                return 1
        if att_start_delay_wall_s > 0.0:
            time.sleep(att_start_delay_wall_s)
        controller.arm()
        shared_data['flight_started'] = True
        # Tape t=0 is arm/GO, not the start of the slow-motion countdown.
        started = time.monotonic()
        print('[SIM] armed — attitude replay', flush=True)
        if practice_rec is not None:
            practice_rec.start(time.perf_counter())
    else:
        shared_data['flight_started'] = False
        print(
            '[FLY] waiting for YOUR stick/key '
            f'(≥{engage_frac:.0%} for {engage_need} ticks) — '
            'no arm / no hover until then',
            flush=True,
        )
    if practice_auto and not args.quiet:
        print(
            f'[PRACTICE] auto-save ON -> {PRACTICE_DIR}/runs/{{partial,complete}}/ '
            '(zero-attitude events stored on the tape)',
            flush=True,
        )
    if not args.quiet:
        print(
            '\n    t   mode     lock  climb  km/h    thr   des_p   phase   src',
            flush=True,
        )

    def _zero_attitude(reason: str) -> None:
        if state_estimator is None:
            print(f'[FLY] zero-attitude failed — no EKF ({reason})', flush=True)
            return
        zero_fn = getattr(state_estimator, 'zero_tilt', None)
        if not callable(zero_fn):
            print(f'[FLY] zero-attitude unavailable ({reason})', flush=True)
            return
        try:
            _r, _p, yaw = zero_fn()
        except Exception as exc:
            print(f'[FLY] zero-attitude error: {exc}', flush=True)
            return
        roll_pid.reset()
        pitch_pid.reset()
        if vrate is not None:
            vrate.reset()
        t_mark = None
        # Live zeros go on the practice tape; tape-driven zeros must not
        # re-log (would duplicate events while replaying).
        if practice_rec is not None and att_clock is None:
            if not practice_rec.started:
                practice_rec.start(time.perf_counter())
            ev = practice_rec.mark_zero_attitude(
                time.perf_counter(), reason=reason,
            )
            t_mark = ev.get('t')
        print(
            f'[FLY] ZERO ATTITUDE — roll/pitch cleared '
            f'(yaw={math.degrees(float(yaw)):.1f}°'
            f'{f", tape t={float(t_mark):.2f}s" if t_mark is not None else ""})'
            f'  [{reason}]',
            flush=True,
        )
        log = shared_data.get('log_event')
        if log:
            log(
                'ZERO_ATT',
                f'yaw={math.degrees(float(yaw)):.2f} reason={reason}'
                + (f' t={float(t_mark):.3f}' if t_mark is not None else ''),
            )

    def _int_reference(value):
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None

    def _vec_reference(value, n: int):
        if value is None:
            return None
        try:
            vals = [_f(value[i]) for i in range(n)]
        except (TypeError, IndexError, KeyError):
            return None
        return vals if all(v is not None for v in vals) else None

    def _acro_reference_snapshot(
        lock: dict | None,
        sample_perf_counter_s: float,
    ) -> dict | None:
        """Observe-only state synchronized to the command tape sample."""
        if not acro_mode:
            return None
        lock = lock or {}
        det = shared_data.get('gate_detection') or {}
        dual = shared_data.get('dual_gate_pnp') or {}
        raw_candidate_packet = shared_data.get('gate_candidates') or {}
        race = shared_data.get('race_status') or {}
        att = shared_data.get('attitude') or {}
        imu = shared_data.get('highres_imu') or {}
        ctrl = shared_data.get('control_output') or {}
        nav = shared_data.get('navigation') or {}
        center_px = _vec_reference(det.get('center_px'), 2)
        corners_px = None
        raw_corners = det.get('corners_px')
        if raw_corners is not None:
            try:
                corners_px = [
                    [round(float(p[0]), 2), round(float(p[1]), 2)]
                    for p in raw_corners
                ]
            except (TypeError, ValueError, IndexError):
                corners_px = None
        vision_ref = {
            'locked': bool(lock.get('locked')),
            'source': str(lock.get('source') or 'none'),
            'method': str(det.get('method') or 'none'),
            'nx': _f(lock.get('nx')),
            'ny': _f(lock.get('ny')),
            'center_px': center_px,
            'corners_px': corners_px,
            'area_px': _f(lock.get('area_px')),
            'confidence': _f(det.get('confidence')),
            'range_m': _f(lock.get('range_m')),
            'frame_id': _int_reference(det.get('frame_id')),
            'timestamp_ns': _int_reference(
                det.get('ts') if det.get('ts') is not None else dual.get('ts')
            ),
            'sim_time_ns': _int_reference(nav.get('sim_time_ns')),
            'predicted': bool(det.get('predicted')),
            'pnp_held': bool(dual.get('held')),
            'pnp_reproj_px': _f(dual.get('gate1_reproj_px')),
            'gate1_body': _vec_reference(dual.get('gate1_body'), 3),
            'gate1_normal_body': _vec_reference(
                dual.get('gate1_normal_body'), 3
            ),
            # Keep every detector hypothesis.  Near/overlapping gates can
            # make the live selector switch to a future gate; replay can
            # recover the demonstrated identity only if the alternatives are
            # present in the reference tape.
            'candidates': [
                {
                    'center_px': _vec_reference(item.get('center_px'), 2),
                    'bbox_px': _vec_reference(item.get('bbox_px'), 4),
                    'area_px': _f(item.get('area_px')),
                    'confidence': _f(item.get('confidence')),
                    'hsv_confirmed': bool(item.get('hsv_confirmed')),
                }
                for item in (raw_candidate_packet.get('items') or [])
                if isinstance(item, dict)
            ],
            'candidates_frame_id': _int_reference(
                raw_candidate_packet.get('frame_id')
            ),
            'candidates_timestamp_ns': _int_reference(
                raw_candidate_packet.get('ts')
            ),
        }
        return {
            # Epoch timestamp links this command sample to both the camera
            # frame timestamp and the independent 50 Hz telemetry log.
            'sample_wall_time_ns': time.time_ns(),
            'sample_perf_counter_s': float(sample_perf_counter_s),
            'vision': vision_ref,
            'race': {
                'sim_boot_ms': _int_reference(race.get('sim_boot_ms')),
                'race_start_ms': _int_reference(race.get('race_start_ms')),
                'race_finish_ns': _int_reference(race.get('race_finish_ns')),
                'active_gate': _int_reference(race.get('active_gate')),
                'last_gate_time_ns': _int_reference(
                    race.get('last_gate_time')
                ),
                'received_perf_counter_s': _f(
                    race.get('received_perf_counter_s')
                ),
                'received_wall_time_ns': _int_reference(
                    race.get('received_wall_time_ns')
                ),
            },
            'attitude': {
                'roll': _f(att.get('roll')),
                'pitch': _f(att.get('pitch')),
                'yaw': _f(att.get('yaw')),
            },
            'gyro': {
                'x': _f(imu.get('xgyro')),
                'y': _f(imu.get('ygyro')),
                'z': _f(imu.get('zgyro')),
            },
            'accel': {
                'x': _f(imu.get('xacc')),
                'y': _f(imu.get('yacc')),
                'z': _f(imu.get('zacc')),
            },
            # Post-controller wire values capture rate signs/clamps exactly;
            # the outer sample retains the original pre-controller pad rates.
            'wire': {
                'roll_rate': _f(ctrl.get('roll_rate')),
                'pitch_rate': _f(ctrl.get('pitch_rate')),
                'yaw_rate': _f(ctrl.get('yaw_rate')),
                'thrust': _f(ctrl.get('thrust')),
            },
        }

    def _record_practice(
        _t_wall: float,
        *,
        des_roll: float,
        des_pitch: float,
        yaw_rate: float,
        thrust: float,
        lock: dict | None = None,
    ) -> None:
        if practice_rec is None or mode in ('auto', 'attitude'):
            return
        # QueryPerformanceCounter gives sub-millisecond placement on Windows;
        # time.monotonic() is quantized to 15.625 ms on this installation.
        t_wall = time.perf_counter()
        if not practice_rec.started:
            practice_rec.start(t_wall)
        pad = hold_state.get('pad') if isinstance(hold_state.get('pad'), dict) else {}
        practice_rec.sample(
            t_wall,
            des_roll=des_roll,
            des_pitch=des_pitch,
            yaw_rate=yaw_rate,
            thrust=thrust,
            pad_roll=pad.get('roll'),
            pad_pitch=pad.get('pitch'),
            pad_yaw=pad.get('yaw'),
            pad_thrust=pad.get('thrust'),
            reference=_acro_reference_snapshot(lock, t_wall),
        )
        gmsg = practice_rec.watch_gate_pass(shared_data, t_wall)
        if gmsg:
            print('  ' + gmsg, flush=True)
            g = practice_rec.max_gate()
            if g is not None:
                msg = maybe_update_through_gate(
                    practice_rec, int(g), source='live',
                )
                if msg:
                    print(f'[PRACTICE] {msg}', flush=True)

    def _archive_practice_run(reason: str) -> None:
        if not practice_auto or practice_rec is None or not practice_rec.samples:
            return
        race = shared_data.get('race_status') or {}
        finish_ns = race.get('race_finish_ns')
        try:
            finish_ns = int(finish_ns) if finish_ns else 0
        except (TypeError, ValueError):
            finish_ns = 0
        msg = save_run(
            practice_rec,
            reason=reason,
            source='live',
            race_finish_ns=finish_ns,
        )
        if msg:
            print(f'[PRACTICE] {msg}', flush=True)

    def _flight_input_active(
        des_roll, des_pitch, yaw_rate, stick_vert,
    ) -> bool:
        """True only on a deliberate stick/key — not pad filter leftovers."""
        lean_lim = max(float(lean_rad), float(pitch_rad), 1e-6)
        yaw_lim = max(float(yaw_rate_cmd), 1e-6)
        thr_lim = max(float(thrust_step), float(sink_rate_cmd), 1e-6)
        roll_f = abs(float(des_roll)) / lean_lim
        pitch_f = abs(float(des_pitch)) / lean_lim
        yaw_f = abs(float(yaw_rate)) / yaw_lim
        thr_f = abs(float(stick_vert)) / thr_lim
        best = max(roll_f, pitch_f, yaw_f, thr_f)
        if best < engage_frac:
            engage_streak['n'] = 0
            engage_streak['why'] = ''
            return False
        axis = (
            'pitch' if pitch_f >= best - 1e-9 else
            'roll' if roll_f >= best - 1e-9 else
            'yaw' if yaw_f >= best - 1e-9 else
            'thrust'
        )
        engage_streak['n'] = int(engage_streak['n']) + 1
        engage_streak['why'] = (
            f'{axis}={best:.0%} (≥{engage_frac:.0%} x{engage_need})'
        )
        return engage_streak['n'] >= engage_need

    def _engage_pilot(reason: str) -> None:
        """Arm and start the flight clock on first real pilot input."""
        nonlocal pilot_engaged, started
        if pilot_engaged:
            return
        controller.arm()
        shared_data['flight_started'] = True
        pilot_engaged = True
        started = time.monotonic()
        if practice_rec is not None:
            practice_rec.start(time.perf_counter())
        detail = engage_streak.get('why') or reason
        print(f'[FLY] armed — {detail}', flush=True)

    def _do_run_reset() -> None:
        """Pad Y / key Y: archive attempt, sim reset, wait for next stick."""
        nonlocal mode, was_locked, started, last_pad_reset, pad_z
        nonlocal pilot_engaged, att_last_idx, handoff_wait_announced
        now_r = time.monotonic()
        if (now_r - last_pad_reset) < 3.0:
            print('[SIM] reset ignored — wait a moment', flush=True)
            return
        last_pad_reset = now_r
        try:
            _archive_practice_run('reset')
        except Exception as exc:
            print(f'[PRACTICE] run archive skipped ({exc})', flush=True)
        print('[SIM] Y — reset run (cmd 31000)...', flush=True)
        log = shared_data.get('log_event')
        if log:
            log('SIM_RESET', 'pilot_pad_y')
        shared_data['flight_started'] = False
        shared_data['vision_reset_episode'] = True
        shared_data['course_bearing'] = None
        shared_data['gate_detection'] = None
        shared_data['dual_gate_pnp'] = None
        shared_data['kalman_path'] = None
        shared_data['planner_target'] = None
        shared_data['collision'] = None
        shared_data['last_gate_passed'] = None
        shared_data['post_pass_hunt'] = False
        try:
            controller.disarm()
        except Exception:
            pass
        t_reset = time.monotonic()
        controller.send_sim_reset()
        time.sleep(0.25)
        if planner is not None:
            resetter = getattr(planner, 'reset_episode', None)
            if callable(resetter):
                resetter()
        if state_estimator is not None:
            ekf_reset = getattr(state_estimator, 'reset_episode', None)
            if callable(ekf_reset):
                ekf_reset()
        shared_data['local_position_ned'] = None
        shared_data['vision_reset_episode'] = True

        pad_z = 0.0
        for key in ('local_position_ned', 'position_ned'):
            z = (shared_data.get(key) or {}).get('z')
            if z is not None and math.isfinite(float(z)):
                pad_z = float(z)
                break
        shared_data['pilot_pad_z'] = pad_z
        for axis in ('roll', 'pitch', 'yaw', 'thrust'):
            hold_state[axis] = 0.0
            hold_state[f'{axis}_t'] = 0.0
        hold_state['keys_held'] = {}
        hold_state.pop('thrust_bump', None)
        hold_state.pop('_assist_handoff_done', None)
        engage_streak['n'] = 0
        engage_streak['why'] = ''
        if vrate is not None:
            vrate.reset()
        roll_pid.reset()
        pitch_pid.reset()
        was_locked = False
        att_last_idx = None
        handoff_wait_announced = False
        started = time.monotonic()
        if att_clock is not None:
            att_clock.reset()
            mode = 'attitude'
            hold_s_r = _countdown_hold_s(args)
            _wait_aligned_to_countdown(
                shared_data, t_reset, hold_s_r,
                label='PAD',
                need_vision=not pure_fly,
                vision_grace_s=0.0 if pure_fly else 2.0,
            )
            if att_start_delay_wall_s > 0.0:
                time.sleep(att_start_delay_wall_s)
            controller.arm()
            shared_data['flight_started'] = True
            started = time.monotonic()
            pilot_engaged = True
            if practice_rec is not None:
                practice_rec.start(time.perf_counter())
            print('[SIM] re-armed — attitude replay', flush=True)
        else:
            mode = 'manual'
            pilot_engaged = False
            shared_data['flight_started'] = False
            if practice_rec is not None:
                practice_rec.clear()
            print(
                '[SIM] reset — touch sticks/keys to arm (no inputs until then)',
                flush=True,
            )

    timer_resolution_on = False
    if acro_mode and sys.platform == 'win32':
        try:
            import ctypes
            timer_resolution_on = ctypes.windll.winmm.timeBeginPeriod(1) == 0
            if timer_resolution_on:
                print(
                    f'[TIMER] 1 ms Windows scheduling; requested loop '
                    f'{float(args.hz):.0f} Hz',
                    flush=True,
                )
        except Exception as exc:
            print(f'[TIMER] 1 ms scheduling unavailable: {exc}', flush=True)

    try:
        while True:
            now = time.monotonic()
            # Freeze HUD/time until you engage (manual/fly).
            elapsed = (now - started) if pilot_engaged else 0.0
            if pilot_engaged and args.seconds > 0 and elapsed >= args.seconds:
                print('\n[STOP] time limit', flush=True)
                break

            (
                des_roll, des_pitch, yaw_rate, stick_vert, quit_req
            ) = _poll_manual_controls(
                hold_state,
                lean_rad=lean_rad,
                yaw_rate_cmd=yaw_rate_cmd,
                thrust_step=thrust_step,
                now=now,
                ignore_sticks=(att_clock is not None),
                sink_step=sink_rate_cmd,
                pitch_rad=pitch_rad,
            )
            if quit_req:
                print('\n[STOP] quit key', flush=True)
                break

            slowmos = int(hold_state.get('slowmo', 0))
            if slowmos > slowmos_seen:
                slowmos_seen = slowmos
                slow_mo_on = not slow_mo_on
                shared_data['log_hz'] = (
                    max(5.0, 50.0 * slow_mo_scale) if slow_mo_on else 50.0
                )
                print(
                    f'[SLOW-MO] {"ON" if slow_mo_on else "OFF"} '
                    f'x{slow_mo_scale:.2f}  log_hz={shared_data["log_hz"]}',
                    flush=True,
                )

            resets = int(hold_state.get('resets', 0))
            if resets > resets_seen:
                resets_seen = resets
                _do_run_reset()
                last_t = None
                continue

            # Sit dead on the pad until a deliberate stick/key.
            if mode == 'manual' and not pilot_engaged:
                if _flight_input_active(
                    des_roll, des_pitch, yaw_rate, stick_vert,
                ):
                    _engage_pilot('go')
                else:
                    time.sleep(period)
                    continue

            # Mid-run declare-level (fly: X/H; any mode: Z).
            zeros = int(hold_state.get('zero_att', 0))
            if zeros > zeros_seen:
                zeros_seen = zeros
                _zero_attitude('Z')
            humans = int(hold_state.get('human', 0))
            if humans > humans_seen:
                # In pure fly, X/H is zero-attitude (not assist handoff).
                if pure_fly:
                    humans_seen = humans
                    _zero_attitude('pad X / H')
                # else: handled below with existing HUMAN logic

            # Open-loop attitude tape (fly/pilot --replay-attitude PATH).
            if att_clock is not None:
                for ev in att_clock.due_events(elapsed):
                    if ev.get('type') == 'zero_attitude':
                        _zero_attitude(
                            f"tape@{float(ev.get('t', 0)):.2f}s"
                        )
                if att_acro_rates:
                    att_idx = att_clock.index_at(elapsed)
                    sample = (
                        None if att_idx is None
                        else att_clock.sample_index(att_idx)
                    )
                else:
                    att_idx = None
                    sample = att_clock.sample_at(elapsed)
                if sample is None:
                    print(f'\n[REPLAY] attitude tape ended @ {elapsed:.1f}s',
                          flush=True)
                    break
                des_roll = float(sample['des_roll'])
                des_pitch = float(sample['des_pitch'])
                yaw_rate = float(sample['yaw_rate'])
                # Thrust from tape replaces stick_vert path below.
                stick_vert = 0.0
                taped_thrust = float(sample['thrust'])
            else:
                taped_thrust = None

            lock = _pilot_lock_info(shared_data, min_area=min_area)
            locked = bool(lock['locked'])
            # Vision overlay reads this for green LOCK / amber weak colors.
            with shared_data['lock']:
                shared_data['pilot_lock'] = {
                    'locked': locked,
                    'mode': mode,
                    'source': lock.get('source'),
                    'nx': lock.get('nx'),
                    'ny': lock.get('ny'),
                    'area_px': lock.get('area_px'),
                    'range_m': lock.get('range_m'),
                    't': now,
                }
            if locked and not was_locked:
                print(
                    f'[PILOT] LOCKED  src={lock["source"]} '
                    f'nx={_fmt(lock["nx"], "+.3f")} '
                    f'ny={_fmt(lock["ny"], "+.3f")} '
                    f'area={_fmt(lock["area_px"], ".0f")} '
                    f'rng={_fmt(lock["range_m"], ".1f")}',
                    flush=True,
                )
                log = shared_data.get('log_event')
                if log:
                    log('PILOT', f'LOCKED src={lock["source"]}')
                _pilot_beep()
            elif was_locked and not locked:
                print('[PILOT] NO_LOCK', flush=True)
                log = shared_data.get('log_event')
                if log:
                    log('PILOT', 'NO_LOCK')
            was_locked = locked

            # Fly the deterministic prefix open-loop, then require a live
            # next-gate lock before switching to position-aware assist.
            passed_gate = _pilot_passed_gate(shared_data)
            if (
                mode == 'attitude'
                and assist_after is not None
                and passed_gate >= assist_after
            ):
                if not locked:
                    if not handoff_wait_announced:
                        print(
                            f'[PILOT] GATE {passed_gate} cleared — waiting for '
                            'next-gate LOCK before REPLAY→ASSIST',
                            flush=True,
                        )
                        handoff_wait_announced = True
                elif _seed_assist_from_lock(planner, shared_data, now):
                    att_clock = None
                    mode = 'auto'
                    handoff_wait_announced = False
                    if vrate is not None:
                        vrate.reset()
                    roll_pid.reset()
                    pitch_pid.reset()
                    for axis in ('roll', 'pitch', 'yaw', 'thrust'):
                        hold_state[axis] = 0.0
                        hold_state[f'{axis}_t'] = 0.0
                    print(
                        f'[PILOT] REPLAY→ASSIST after real GATE {passed_gate}; '
                        f'assist stays through GATE {human_after}',
                        flush=True,
                    )
                    log = shared_data.get('log_event')
                    if log:
                        log(
                            'PILOT',
                            f'replay_assist_handoff gate={passed_gate}',
                        )

            if (
                mode == 'auto'
                and human_after is not None
                and passed_gate >= human_after
            ):
                mode = 'manual'
                if vrate is not None:
                    vrate.reset()
                roll_pid.reset()
                pitch_pid.reset()
                print(
                    f'[PILOT] ASSIST→HUMAN after real GATE {passed_gate}',
                    flush=True,
                )
                log = shared_data.get('log_event')
                if log:
                    log('PILOT', f'assist_human_handoff gate={passed_gate}')

            autos = int(hold_state.get('auto', 0))
            if (
                not pure_fly
                and planner is not None
                and autos > autos_seen
            ):
                autos_seen = autos
                if mode == 'manual':
                    # Second+ T: freeze first-auto warm gains before flying.
                    if (
                        humans_seen > 0
                        and not bool(getattr(planner, '_learn_frozen', False))
                    ):
                        if planner.freeze_online_learn(reason='before_second_auto'):
                            log = shared_data.get('log_event')
                            if log:
                                log(
                                    'ASSIST',
                                    'learn_frozen before_second_auto '
                                    f'bank_k={planner._bank_bias_learner.gain:.3f}',
                                )
                    if locked and _seed_assist_from_lock(planner, shared_data, now):
                        mode = 'auto'
                        if vrate is not None:
                            vrate.reset()
                        roll_pid.reset()
                        pitch_pid.reset()
                        for axis in ('roll', 'pitch', 'yaw', 'thrust'):
                            hold_state[axis] = 0.0
                            hold_state[f'{axis}_t'] = 0.0
                        print(
                            '[PILOT] AUTO — assist chasing locked gate '
                            '(H = human)',
                            flush=True,
                        )
                        log = shared_data.get('log_event')
                        if log:
                            log('PILOT', 'auto_handoff')
                    else:
                        print(
                            '[PILOT] T ignored — no LOCK '
                            '(point at a gate until LOCKED)',
                            flush=True,
                        )
                else:
                    print('[PILOT] T ignored — already AUTO', flush=True)

            humans = int(hold_state.get('human', 0))
            if (
                not pure_fly
                and planner is not None
                and humans > humans_seen
            ):
                humans_seen = humans
                if mode == 'auto':
                    mode = 'manual'
                    # End of first auto → lock in the warm weights you liked.
                    if planner.freeze_online_learn(reason='first_auto'):
                        log = shared_data.get('log_event')
                        if log:
                            log(
                                'ASSIST',
                                'learn_frozen first_auto '
                                f'bank_k={planner._bank_bias_learner.gain:.3f} '
                                f'mild={planner._lat_yaw_learner.mild_mult:.2f} '
                                f'hard={planner._lat_yaw_learner.hard_mult:.2f}',
                            )
                    if vrate is not None:
                        vrate.reset()
                    roll_pid.reset()
                    pitch_pid.reset()
                    for axis in ('roll', 'pitch', 'yaw', 'thrust'):
                        hold_state[axis] = 0.0
                        hold_state[f'{axis}_t'] = 0.0
                    print(
                        '[PILOT] HUMAN — you have the sticks (T = auto on LOCK)',
                        flush=True,
                    )
                    log = shared_data.get('log_event')
                    if log:
                        log('PILOT', 'human_handoff')
                else:
                    print('[PILOT] H ignored — already HUMAN', flush=True)

            dt = period if last_t is None else max(1e-3, now - last_t)
            last_t = now
            path = shared_data.get('kalman_path') or {}
            tgt = shared_data.get('planner_target') or {}

            if mode == 'auto':
                try:
                    tgt = planner.compute_target(shared_data)
                    path = shared_data.get('kalman_path') or {}
                    controller.update()
                except Exception as exc:
                    # Don't kill the armed run on an assist glitch — give sticks back.
                    print(
                        f'\n[PILOT] AUTO failed ({type(exc).__name__}: {exc}) '
                        '— back to HUMAN',
                        flush=True,
                    )
                    mode = 'manual'
                    if vrate is not None:
                        vrate.reset()
                    roll_pid.reset()
                    pitch_pid.reset()
                    continue
            else:
                roll, pitch, _, _ = read_attitude(shared_data)
                tape_rates_active = bool(
                    mode == 'attitude' and att_acro_rates
                )
                direct_rates = bool(acro_mode or tape_rates_active)
                if direct_rates:
                    # Sticks ARE body rates. Center = 0 rate (no self-level).
                    roll_rate = float(des_roll)
                    pitch_rate = float(des_pitch)
                    lean_boost = float(
                        getattr(config, 'LEAN_THRUST_BOOST', 0.0) or 0.0
                    )
                    # Collective vs measured attitude (not a lean setpoint).
                    thrust = _tilt_compensated_thrust(
                        config.HOVER_THRUST, roll, pitch,
                        lean_boost=lean_boost,
                    )
                else:
                    roll_rate = roll_pid.update(des_roll - roll, dt)
                    pitch_rate = pitch_pid.update(des_pitch - pitch, dt)
                    lean_boost = float(
                        getattr(config, 'LEAN_THRUST_BOOST', 0.0) or 0.0
                    )
                    thrust = _tilt_compensated_thrust(
                        config.HOVER_THRUST, des_roll, des_pitch,
                        lean_boost=lean_boost,
                    )
                if direct_rates:
                    thr_lo = float(
                        getattr(config, 'ACRO_THRUST_MIN', 0.05) or 0.05
                    )
                    thr_hi = float(
                        getattr(config, 'ACRO_THRUST_MAX', 0.70) or 0.70
                    )
                else:
                    thr_lo, thr_hi = 0.06, 0.45
                if taped_thrust is not None:
                    thrust = float(max(thr_lo, min(thr_hi, taped_thrust)))
                else:
                    if vrate is None:
                        if open_loop:
                            # Stick already in collective units (thrust levels).
                            thrust_delta = float(stick_vert)
                        else:
                            thrust_delta = _stick_rate_to_thrust_delta(
                                float(stick_vert),
                                climb_rate=climb_rate_cmd,
                                sink_rate=sink_rate_cmd,
                                climb_auth=float(climb_auth),
                                sink_auth=float(sink_auth),
                            )
                    else:
                        thrust_delta, _, _ = vrate.update(
                            shared_data, stick_vert, dt
                        )
                    bump = float(hold_state.get('thrust_bump') or 0.0)
                    bump_t = float(hold_state.get('thrust_bump_t') or 0.0)
                    if bump_t <= 0.0 or (now - bump_t) > 0.25:
                        bump = 0.0
                    if float(stick_vert) < -1e-3:
                        bump = 0.0
                    # Wide clamp — tight 0.18–0.36 was eating top speed on tip.
                    # Acro uses ACRO_THRUST_* so full RT can punch harder.
                    thrust = float(
                        max(thr_lo, min(thr_hi, thrust + thrust_delta + bump))
                    )
                if acro_mode or pure_fly:
                    shared_data['control_authority'] = 'human'
                    shared_data['intervention_id'] = ''
                shared_data['planner_target'] = {
                    'kalman': True,
                    'acro': direct_rates,
                    'unrestricted_rates': direct_rates,
                    'roll_rate': roll_rate,
                    'pitch_rate': pitch_rate,
                    'yaw_rate': yaw_rate,
                    'thrust': thrust,
                    'desired_roll': (
                        float(roll) if direct_rates else float(des_roll)
                    ),
                    'desired_pitch': (
                        float(pitch) if direct_rates else float(des_pitch)
                    ),
                }
                shared_data['planner_mode'] = (
                    'attitude_replay' if att_clock is not None
                    else ('acro' if acro_mode else 'pilot_manual')
                )
                shared_data['kalman_path'] = {
                    'phase': (
                        'attitude_replay'
                        if att_clock is not None
                        else ('acro' if acro_mode else 'pilot_manual')
                    ),
                    'source': 'tape' if att_clock is not None else 'teleop',
                    'norm_x': lock.get('nx'),
                    'norm_y': lock.get('ny'),
                    'thrust': thrust,
                    'locked': locked,
                }
                path = shared_data['kalman_path']
                tgt = shared_data['planner_target']
                # Preserve exact acro tape cadence: one send per recorded
                # sample, with the prior command held between samples.
                if not tape_rates_active:
                    controller.update()
                elif att_idx != att_last_idx:
                    controller.update()
                    att_last_idx = att_idx
                _record_practice(
                    now,
                    des_roll=float(des_roll),
                    des_pitch=float(des_pitch),
                    yaw_rate=float(yaw_rate),
                    thrust=float(thrust),
                    lock=lock,
                )

            climb = _climb_estimate(shared_data)
            speed_kmh = _ground_speed_kmh(shared_data)
            row = {
                't': round(elapsed, 3),
                'mode': mode,
                'locked': int(locked),
                'lock_src': lock.get('source'),
                'speed_kmh': (
                    round(speed_kmh, 1) if speed_kmh is not None else None
                ),
                'nx': lock.get('nx'),
                'ny': lock.get('ny'),
                'area_px': lock.get('area_px'),
                'range_m': lock.get('range_m'),
                'phase': path.get('phase'),
                'thrust': tgt.get('thrust'),
                'des_pitch': tgt.get('desired_pitch'),
                'yaw_rate': tgt.get('yaw_rate'),
                'climb_m': climb,
            }
            recorder.write(row)
            if not args.quiet:
                lock_s = 'LOCK' if locked else '----'
                print(
                    f"{elapsed:5.1f} {mode:8s} {lock_s:4s} "
                    f"{_fmt(climb, '6.2f')} {_fmt(tgt.get('thrust'), '5.3f')} "
                    f"{_fmt(tgt.get('desired_pitch'), '6.3f')} "
                    f"{path.get('phase')} {lock.get('source')}",
                    flush=True,
                )
            time.sleep(period)
    except KeyboardInterrupt:
        print('\n[STOP] interrupted', flush=True)
    finally:
        try:
            _archive_practice_run('quit')
        except Exception as exc:
            print(f'[PRACTICE] run archive skipped ({exc})', flush=True)
        try:
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
        if timer_resolution_on:
            try:
                import ctypes
                ctypes.windll.winmm.timeEndPeriod(1)
            except Exception:
                pass

    print(f'\nCSV: {recorder.path}')
    # Always keep the best/fastest run under logs/best/.
    try:
        from best_run import latest_events_stem, maybe_save_best
        stem = latest_events_stem()
        if stem:
            maybe_save_best(
                events_stem=stem,
                pilot_csv=Path(recorder.path) if recorder.path else None,
                label=(
                    'acro' if getattr(args, 'mode', None) == 'acro'
                    else (
                        'fly'
                        if (
                            bool(getattr(args, 'pure', False))
                            or getattr(args, 'mode', None) == 'fly'
                        )
                        else 'pilot'
                    )
                ),
            )
    except Exception as exc:
        print(f'[BEST] skip ({exc})', flush=True)
    return 0


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
    pure_cli = (
        args.mode == 'fly'
        or bool(getattr(args, 'pure', False))
        or (args.mode == 'acro' and bool(getattr(args, 'no_vision', False)))
    )
    # Armed vision modes: warm YOLO before heartbeat. Acro is observe-only but
    # still pre-warms so detector loading cannot steal timing from the run.
    if (
        not pure_cli
        and args.mode in (
            'hover', 'step', 'lean-hover', 'crawl', 'drive', 'yaw-align',
            'authority', 'climb', 'acquire', 'manual', 'assist', 'coach',
            'policy', 'pilot', 'acro',
        )
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
    if args.mode == 'coach':
        return run_coach(args)
    if args.mode == 'policy':
        from tools.run_policy import run_from_args
        return run_from_args(args, prewarm=False)
    if args.mode in ('pilot', 'fly', 'acro'):
        return run_pilot(args)
    if args.mode == 'practice':
        from practice_store import format_list
        print(format_list(), flush=True)
        return 0
    return 2


if __name__ == '__main__':
    sys.exit(main())
