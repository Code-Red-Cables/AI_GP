"""Localization check + PID tuning harness for the dual-gate PnP + IMU path.

Three subcommands, smallest blast radius first:

  localize   Read-only. Never arms, never sends a flight command. Streams the
             PnP -> EKF chain: solve rate, gate ranges, range continuity,
             gate2 retention, and IMU-only drift across no-PnP stretches.
             Start here.

  hover      Arms and holds level at a fixed collective. This is the
             HOVER_THRUST trim test: if the drone climbs or sinks, the number
             is wrong and every downstream gain is fighting it. Keep a gate in
             view — the verdict then rests on the gate's image-space drift,
             which needs no integration.

  step       Arms, levels, then injects a desired-lean step and measures the
             inner attitude loop's rise / overshoot / settle. Tune
             KALMAN_KP_ATT and KALMAN_KD_ATT here.

Gains are passed as flags and exported to the environment *before* config is
imported, so the values under test are the ones the live planner would use:

  python tools/tune_flight.py localize --seconds 30
  python tools/tune_flight.py hover --hover-thrust 0.245 --seconds 12
  python tools/tune_flight.py step --axis pitch --amplitude-deg 8 \
      --kp-att 2.2 --kd-att 0.10

Every run appends a CSV under logs/tuning/ for offline comparison.

This simulator publishes neither ATTITUDE nor LOCAL_POSITION_NED, so there is
no ground truth and no external attitude reference. Roll/pitch come from the
EKF's IMU integration, body rates from raw HIGHRES_IMU gyro, and every
position number is PnP-corrected dead reckoning. That is the whole reason the
PnP chain exists, and it is why the vertical tests prefer an image-space
observable over any position estimate.

The vertical channel on this branch is open-loop image-based
(thrust = HOVER_THRUST - 0.030 * norm_y, clamped) — there is no altitude PID
to tune. Trim HOVER_THRUST with `hover`, then tune the attitude loop with
`step`, then the image yaw gain KALMAN_KP_YAW in a real run.
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

    p_loc = sub.add_parser('localize', help='read-only localization report')
    common(p_loc, 30.0)
    p_loc.add_argument('--park-ekf', action='store_true',
                       help='leave the EKF in ZUPT (PnP only). By default '
                            'localize sets flight_started so the EKF actually '
                            'integrates — otherwise it reports a frozen zero '
                            'state.')

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
    return parser


def export_gain_overrides(args) -> dict:
    """Push --flags into the environment so `import config` picks them up."""
    mapping = {
        'hover_thrust': 'HOVER_THRUST',
        'kp_att': 'KALMAN_KP_ATT',
        'kd_att': 'KALMAN_KD_ATT',
        'max_rate': 'KALMAN_MAX_RATE_RAD_S',
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
    return applied


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
    from setup import setup_components

    shared_data = {}
    components = setup_components(
        shared_data, int(time.time() * 1000),
        SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT,
    )
    recorder = Recorder(Path(args.csv) if args.csv
                        else default_csv_path('localize'))

    print('[SAFE] localize: never arms, never sends a flight command',
          flush=True)
    # The EKF only integrates while 'flight_started' is set — otherwise it is
    # held in ZUPT and would report a frozen zero state.
    if args.park_ekf:
        print('       EKF parked (--park-ekf): PnP only, EKF will read zeros',
              flush=True)
    else:
        shared_data['flight_started'] = True
        print('       EKF integrating. On a stationary pad the drift you see '
              'IS the IMU-only error;', flush=True)
        print('       carry or fly the drone to exercise PnP corrections.',
              flush=True)
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


def run_attitude_loop(args, desired_at, label):
    """Arm, close the inner attitude loop toward desired_at(t), log, disarm.

    `desired_at(elapsed)` returns (desired_roll_rad, desired_pitch_rad). This
    mirrors kalman_planner's inner loop exactly — same PIDController, same
    gains, same rate clamp — so numbers found here transfer to real flight.
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
    print(f'[ABORT] |climb| > {ABORT_ALTITUDE_M} m or |lean| > '
          f'{math.degrees(ABORT_LEAN_RAD):.0f} deg disarms immediately',
          flush=True)
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

            if max(abs(roll), abs(pitch)) > ABORT_LEAN_RAD:
                abort = f'lean {math.degrees(max(abs(roll), abs(pitch))):.0f} deg'
                break
            alt_for_abort = d_truth_z if d_truth_z is not None else d_ekf_z
            if alt_for_abort is not None and abs(alt_for_abort) > ABORT_ALTITUDE_M:
                label_src = 'truth' if d_truth_z is not None else 'EKF'
                abort = f'{label_src} altitude {alt_for_abort:+.1f} m'
                break

            des_roll, des_pitch = desired_at(elapsed)
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
            thrust = config.HOVER_THRUST

            # Same contract the kalman planner uses, so the controller takes
            # its direct rate path rather than the velocity fallback.
            shared_data['planner_target'] = {
                'kalman': True,
                'roll_rate': roll_rate,
                'pitch_rate': pitch_rate,
                'yaw_rate': 0.0,
                'thrust': thrust,
            }
            shared_data['planner_mode'] = 'kalman_dual_gate'
            controller.update()

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
            })
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
    level = lambda _elapsed: (0.0, 0.0)
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


def run_step(args) -> int:
    amplitude = math.radians(args.amplitude_deg)
    settle = args.settle_s

    def desired(elapsed):
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
    post = [r for r in rows if r['t'] >= settle]
    if len(post) < 5:
        print('not enough post-step samples — increase --seconds')
        print(f'\nCSV: {recorder.path}')
        return 1 if abort else 0

    t0 = post[0]['t']
    measured = [(r['t'] - t0, r[key]) for r in post]
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
    fb = rows[0].get('feedback', 'ekf')
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
def main() -> int:
    args = build_parser().parse_args()
    applied = export_gain_overrides(args)
    if applied:
        print('[OVERRIDE] ' + '  '.join(f'{k}={v}' for k, v in applied.items()),
              flush=True)
    if args.mode == 'localize':
        return run_localize(args)
    if args.mode == 'hover':
        return run_hover(args)
    if args.mode == 'step':
        return run_step(args)
    return 2


if __name__ == '__main__':
    sys.exit(main())
