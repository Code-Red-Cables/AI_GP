"""Hard observation gate — does the policy's input track the gate it is flying?

Phase 3b of the HG-DAgger plan. The earlier behavior-cloning attempt in this
repo failed because ``corr(gate_u, cmd_roll_rate)`` was +0.03: the detector was
often locked onto a different gate than the one being flown, so no amount of
DAgger could have fixed it. This script has to catch that *before* training.

The plan validated the observation against ODOMETRY. That build is gone, and
spec section 4.3 only guarantees ATTITUDE, HIGHRES_IMU and the camera, so the
gate is now ground-truth free. Three independent checks, none needing position:

  1. RIGIDITY. Under pure camera rotation a world-fixed point's image motion is
     depth-independent and predictable from HIGHRES_IMU alone, so it can be
     checked without any ground truth.

     In practice this check has turned out to be weak, and the reason is
     physical rather than a bug: racing flight is dominated by *translation*
     toward the gate, whose flow scales with speed over depth and so cannot be
     modelled without velocity and range -- exactly what is unavailable. On real
     laps it scores R^2 < 0.05 even with rotation making up 38% of the observed
     motion, and switching from the corner centroid to individual identified
     keypoints (8x the observations) did not change that. It is kept as a cheap
     diagnostic and never blocks a run; identity stability is measured far
     better by the centroid-jump statistic in check 3a.

  2. IDENTITY. At each ``active_gate`` increment the sim says a gate was just
     flown through. The tracked gate must therefore be large and near the image
     centre at that instant. If the detector were following some other gate,
     its apparent size would not peak as the drone passes through.

  3. CENTRING, with COUPLING as its fallback. A detector locked onto the wrong
     gate cannot keep its centroid at the image centre across a series of
     successful gate passes, so the distribution of |u - cx| is the primary
     test. Correlation against the pilot's stick is reported too, but it cannot
     be the gate on its own: a pilot who flies well drives the bearing error to
     zero and leaves correlation nothing to measure -- measured on real laps, a
     clean 17-gate run scored 0.20 where a sloppy 6-gate run scored 0.87. The
     run fails only when the bearing is large *and* uncorrelated, which is the
     original +0.03 disaster.

With ``--course-map`` and a log that still has finite ``odo_*``, the original
ODOMETRY bearing check runs too and is reported as check 0.

    python tools/eval_observation.py --telem logs/telem_....csv
    python tools/eval_observation.py --telem logs/telem_....csv --course-map course_map.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import camera_model as cm  # noqa: E402
from gate_bearing import (  # noqa: E402
    best_lagged_pearson,
    detector_bearing_rad,
    fit_rotation_flow,
    gate_by_active,
    keypoint_centroid_px,
    pearson,
    project_gate_centre_px,
    rotation_flow_basis,
    true_bearing_rad,
    true_gate_body,
)

KEYPOINT_COUNT = 8


def _num(value, default: float = float('nan')) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def load_course_map(path: Path | None) -> list[dict]:
    if path is None or not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding='utf-8'))
    # Accept course_map.json, probe_vq1.json, or a bare gate list.
    if isinstance(raw, dict) and raw.get('track_gates'):
        items = raw['track_gates']
    else:
        gates = raw.get('gates', raw) if isinstance(raw, dict) else raw
        if isinstance(gates, list):
            items = gates
        elif isinstance(gates, dict):
            items = [{'id': k, **v} if isinstance(v, dict) else v
                     for k, v in gates.items()]
        else:
            return []
    out: list[dict] = []
    for item in items:
        try:
            gid = int(item.get('id', item.get('gate_id')))
            pos = (
                item.get('pos')
                or item.get('position_ned')
                or (item.get('x'), item.get('y'), item.get('z'))
            )
            out.append({'id': gid, 'pos': tuple(float(v) for v in pos[:3])})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def row_centroid(row: dict) -> tuple[float, float] | None:
    """Detector centroid in pixels: keypoints when present, else gate centre."""
    if 'kp0_u' in row:
        kps = []
        confs = []
        for i in range(KEYPOINT_COUNT):
            kps.append((_num(row.get(f'kp{i}_u')), _num(row.get(f'kp{i}_v'))))
            confs.append(_num(row.get(f'kp{i}_c')))
        centroid = keypoint_centroid_px(kps, confs)
        if centroid is not None:
            return centroid
    u, v = _num(row.get('gate_u')), _num(row.get('gate_v'))
    if math.isfinite(u) and math.isfinite(v):
        return u, v
    return None


def attitude_for_row(row: dict) -> tuple[float, float]:
    """Roll/pitch exactly as race_obs will feed them to the policy."""
    from race_obs import attitude_from_row

    return attitude_from_row(row)


def row_keypoints(row: dict, *, min_confidence: float = 0.25) -> dict:
    """Visible keypoints as ``{index: (u, v)}``.

    Rigidity is measured per identified keypoint rather than on their centroid:
    the centroid of a *changing* visible set moves for reasons unrelated to
    camera motion, and tracking each point separately also yields up to eight
    observations per frame pair instead of one.
    """
    out: dict[int, tuple[float, float]] = {}
    for i in range(KEYPOINT_COUNT):
        u = _num(row.get(f'kp{i}_u'))
        v = _num(row.get(f'kp{i}_v'))
        c = _num(row.get(f'kp{i}_c'), 1.0)
        if not (math.isfinite(u) and math.isfinite(v)):
            continue
        if u == 0.0 and v == 0.0:
            continue
        if math.isfinite(c) and c < min_confidence:
            continue
        out[i] = (u, v)
    return out


def row_active_gate(row: dict) -> int | None:
    g = _num(row.get('active_gate'))
    return int(g) if math.isfinite(g) else None


def row_frame_id(row: dict) -> int | None:
    f = _num(row.get('gate_frame_id'))
    return int(f) if math.isfinite(f) else None


def row_vision_time(row: dict) -> float:
    """Seconds stamped on the camera frame, falling back to the log clock.

    The telemetry logger runs faster than the camera, so consecutive rows often
    carry the *same* frame. Timing the flow off the frame's own clock (and
    pairing only distinct frames) is what makes this check independent of the
    log rate and of client slow-mo.
    """
    for key in ('vision_sim_time_ns', 'gate_ts_ns'):
        ns = _num(row.get(key))
        if math.isfinite(ns) and ns > 0.0:
            return ns * 1e-9
    return _num(row.get('t'))


def check_rigidity(
    rows: list[dict],
    *,
    max_dt: float = 0.25,
    max_rot_rad: float = 0.15,
) -> dict:
    """Check 1 (diagnostic): centroid motion vs rotation-predicted motion.

    Fits the gyro coefficients rather than assuming the sim's axis signs, since
    assuming them produced a clean *negative* correlation on real data -- a
    convention mismatch, not a bad detector.

    ``max_rot_rad`` matters more than it looks. The flow model is a first-order
    approximation valid for small rotations, and at 10 Hz logging with the
    measured body rates (|gx| reaching 10 rad/s) a single step can span 30
    degrees, where the approximation is meaningless. Pairs rotating more than
    ``max_rot_rad`` are therefore excluded, and the count of what survives is
    reported so a weak result can be told apart from a failing one.
    """
    basis_rows: list = []
    measured: list[tuple[float, float]] = []
    # Kept for reference: correlation under the assumed (unfitted) convention.
    pred_u: list[float] = []
    meas_u: list[float] = []
    duplicate_frames = 0
    large_rotation = 0

    prev = None
    for row in rows:
        t = row_vision_time(row)
        kps = row_keypoints(row)
        gate = row_active_gate(row)
        frame = row_frame_id(row)
        if not kps or not math.isfinite(t):
            prev = None
            continue
        if prev is not None:
            t0, kps0, gate0, frame0 = prev
            dt = t - t0
            same_frame = (
                frame is not None and frame0 is not None and frame == frame0
            )
            if same_frame:
                # Re-logged image: zero measured motion against nonzero
                # predicted motion would bias the fit toward zero.
                duplicate_frames += 1
            # A gate change legitimately teleports the corners; a frame gap
            # breaks the small-rotation assumption. Neither is evidence.
            elif 0.0 < dt <= max_dt and gate == gate0:
                gx = _num(row.get('gx_imu'))
                gy = _num(row.get('gy_imu'))
                gz = _num(row.get('gz_imu'))
                rot = math.sqrt(gx * gx + gy * gy + gz * gz) * dt
                rates = np.array([gx, gy, gz])
                if not all(math.isfinite(g) for g in (gx, gy, gz)):
                    pass
                elif rot > max_rot_rad:
                    large_rotation += 1
                else:
                    # Only keypoints present in BOTH frames: the same physical
                    # corner, so its displacement is real motion.
                    for idx, (u1, v1) in kps.items():
                        if idx not in kps0:
                            continue
                        u0, v0 = kps0[idx]
                        basis = rotation_flow_basis(u0, v0, dt)
                        if basis is None:
                            continue
                        # Scale each column by its own measured rate, so a
                        # fitted coefficient of 1.0 means "this axis behaves
                        # exactly as assumed" and negative means inverted.
                        basis_rows.append(basis * rates)
                        measured.append((u1 - u0, v1 - v0))
                        pred_u.append(float(basis[0] @ rates))
                        meas_u.append(u1 - u0)
        if not (
            prev is not None and frame is not None and prev[3] == frame
        ):
            prev = (t, kps, gate, frame)

    fit = fit_rotation_flow(basis_rows, measured)
    fit['corr_u_assumed'] = pearson(pred_u, meas_u)
    fit['duplicate_frames'] = duplicate_frames
    fit['large_rotation'] = large_rotation
    fit['r2'] = max(
        (r for r in (fit['r2_u'], fit['r2_v']) if math.isfinite(r)),
        default=float('nan'),
    )
    return fit


def check_identity(
    rows: list[dict],
    *,
    window_s: float = 1.5,
    min_pass_area: float = 15000.0,
) -> dict:
    """Check 2: the gate being flown through is centred as it is cleared.

    Reports the largest-apparent-gate frame in the window before each pass, and
    that same frame's centre offset -- taking the best area and the best offset
    from different frames would let two unrelated moments look like one good one.

    ``min_pass_area`` is what makes this check mean what it says. A gate the
    drone is passing through fills the frame (50-80k px on measured laps); if
    the biggest gate in the window is only a few thousand px, the detector has
    already moved on to the *next* gate, which is legitimately off-centre while
    the drone is still turning toward it. Scoring that as a failure is exactly
    the false positive this threshold removes: one measured lap had a median
    offset of 149 px from samples whose median area was 3.6k px, while the same
    lap tracked its gate 2.4 px from frame centre across 1,164 frames.
    """
    passes: list[dict] = []
    prev_gate = None
    for idx, row in enumerate(rows):
        gate = row_active_gate(row)
        if gate is None:
            continue
        if prev_gate is not None and gate > prev_gate:
            t_pass = _num(row.get('t'))
            best = None
            if math.isfinite(t_pass):
                for back in range(idx, -1, -1):
                    t = _num(rows[back].get('t'))
                    if not math.isfinite(t) or t_pass - t > window_s:
                        break
                    centroid = row_centroid(rows[back])
                    if centroid is None:
                        continue
                    area = _num(rows[back].get('gate_area'), 0.0)
                    if best is None or area > best[0]:
                        best = (area, abs(centroid[0] - cm.CX), t_pass - t)
            near = best is not None and best[0] >= min_pass_area
            passes.append({
                'gate': prev_gate,
                'area_px': best[0] if best else float('nan'),
                'u_offset_px': best[1] if best else float('nan'),
                'age_s': best[2] if best else float('nan'),
                'near': near,
            })
        prev_gate = max(prev_gate, gate) if prev_gate is not None else gate

    seen = [p for p in passes if math.isfinite(p['u_offset_px'])]
    near = [p for p in seen if p['near']]
    return {
        'n_passes': len(passes),
        'n_with_vision': len(seen),
        'n_near': len(near),
        'median_u_offset_px': (
            statistics.median(p['u_offset_px'] for p in near) if near
            else float('nan')
        ),
        'median_area_px': (
            statistics.median(p['area_px'] for p in near) if near
            else float('nan')
        ),
        'passes': passes,
    }


def check_centring(rows: list[dict]) -> dict:
    """Check 3a: is the tracked gate where the drone is heading?

    This is the primary ground-truth-free identity test, and it exists because
    correlation alone is not one. A pilot who flies well drives the bearing
    error to zero, which destroys the variance any correlation needs -- on real
    laps the *clean* 17-gate run scored 0.20 while a sloppy 6-gate run scored
    0.87. Centring inverts that: a detector locked onto some other gate cannot
    keep its centroid within a few pixels of the image centre through a series
    of successful gate passes.

    Also reports frame-to-frame centroid jumps, which measure how often the
    tracked identity changes abruptly.
    """
    offsets: list[float] = []
    bearings: list[float] = []
    jumps: list[float] = []
    prev = None
    for row in rows:
        centroid = row_centroid(row)
        frame = row_frame_id(row)
        gate = row_active_gate(row)
        t = row_vision_time(row)
        if centroid is None:
            prev = None
            continue
        if prev is not None and frame is not None and prev[3] == frame:
            continue  # same image re-logged
        offsets.append(abs(centroid[0] - cm.CX))
        bearing = detector_bearing_rad(centroid[0], centroid[1])
        if bearing is not None:
            bearings.append(abs(math.degrees(bearing)))
        if prev is not None:
            t0, (u0, v0), gate0, _f0 = prev
            if gate == gate0 and 0.0 < t - t0 < 0.3:
                jumps.append(math.hypot(centroid[0] - u0, centroid[1] - v0))
        prev = (t, centroid, gate, frame)

    def _pct(vals: list[float], q: float) -> float:
        if not vals:
            return float('nan')
        s = sorted(vals)
        return s[max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))]

    return {
        'n': len(offsets),
        'median_offset_px': statistics.median(offsets) if offsets else float('nan'),
        'p90_offset_px': _pct(offsets, 0.9),
        'median_bearing_deg': (
            statistics.median(bearings) if bearings else float('nan')
        ),
        'frac_within_60px': (
            sum(1 for o in offsets if o <= 60.0) / len(offsets)
            if offsets else float('nan')
        ),
        'n_jumps': len(jumps),
        'median_jump_px': statistics.median(jumps) if jumps else float('nan'),
        'frac_jump_over_80px': (
            sum(1 for j in jumps if j > 80.0) / len(jumps)
            if jumps else float('nan')
        ),
    }


def check_coupling(rows: list[dict], *, max_lag_s: float = 1.5) -> dict:
    """Check 3: lagged correlation of bearing against the pilot's response.

    Tests several response channels because the right one depends on the mode.
    In acro the stick commands a body *rate*, but what actually turns the drone
    toward the gate is bank *angle*, so a sustained bearing error shows up as a
    sustained roll angle and only a brief rate pulse. Gate on the best channel.
    """
    bearings: list[float] = []
    channels: dict[str, list[float]] = {
        'cmd_roll_rate': [],
        'cmd_yaw_rate': [],
        'roll_angle': [],
    }
    times: list[float] = []
    for row in rows:
        centroid = row_centroid(row)
        if centroid is None:
            continue
        bearing = detector_bearing_rad(centroid[0], centroid[1])
        if bearing is None:
            continue
        roll_angle, _pitch = attitude_for_row(row)
        vals = {
            'cmd_roll_rate': _num(row.get('cmd_roll_rate')),
            'cmd_yaw_rate': _num(row.get('cmd_yaw_rate')),
            'roll_angle': roll_angle,
        }
        bearings.append(bearing)
        times.append(_num(row.get('t')))
        for key, value in vals.items():
            channels[key].append(value)

    dts = [
        b - a for a, b in zip(times, times[1:])
        if math.isfinite(a) and math.isfinite(b) and 0.0 < b - a < 1.0
    ]
    dt = statistics.median(dts) if dts else 0.02
    max_lag = max(1, int(round(max_lag_s / max(dt, 1e-3))))

    out: dict = {'n': len(bearings), 'dt_s': dt, 'max_lag': max_lag}
    best = 0.0
    best_channel = 'none'
    for key, series in channels.items():
        pairs = [
            (b, s) for b, s in zip(bearings, series)
            if math.isfinite(b) and math.isfinite(s)
        ]
        if len(pairs) < 4:
            out[key] = {'n': len(pairs), 'r': float('nan'), 'lag': 0}
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        r, lag = best_lagged_pearson(xs, ys, max_lag)
        out[key] = {'n': len(pairs), 'r': r, 'lag': lag, 'lag_s': lag * dt}
        if math.isfinite(r) and abs(r) > best:
            best = abs(r)
            best_channel = key
    out['corr_best'] = best
    out['best_channel'] = best_channel
    return out


def check_truth(rows: list[dict], course: list[dict]) -> dict:
    """Check 0 (optional): the original ODOMETRY bearing correlation."""
    true_b: list[float] = []
    det_b: list[float] = []
    u_err: list[float] = []
    for row in rows:
        odo = {
            'x': _num(row.get('odo_x')),
            'y': _num(row.get('odo_y')),
            'z': _num(row.get('odo_z')),
            'roll': _num(row.get('odo_roll'), 0.0),
            'pitch': _num(row.get('odo_pitch'), 0.0),
            'yaw': _num(row.get('odo_yaw'), 0.0),
        }
        if not all(math.isfinite(odo[k]) for k in ('x', 'y', 'z')):
            continue
        active = row_active_gate(row)
        if active is None:
            continue
        gate = gate_by_active(course, active)
        if gate is None:
            continue
        body = true_gate_body(odo, gate)
        tb = true_bearing_rad(body)
        centroid = row_centroid(row)
        if tb is None or centroid is None:
            continue
        db = detector_bearing_rad(centroid[0], centroid[1])
        if db is None:
            continue
        true_b.append(tb)
        det_b.append(db)
        proj = project_gate_centre_px(body)
        if proj is not None:
            u_err.append(centroid[0] - proj[0])
    return {
        'n': len(true_b),
        'corr_bearing': pearson(true_b, det_b),
        'median_u_err_px': (
            statistics.median(u_err) if u_err else float('nan')
        ),
    }


def attitude_source(rows: list[dict]) -> dict:
    """Which attitude channel the observation will actually use."""
    from race_obs import attitude_source_of_row

    counts = {'att_raw': 0, 'ahrs': 0, 'ekf': 0, 'none': 0}
    for row in rows:
        counts[attitude_source_of_row(row)] += 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--telem', type=Path, required=True)
    ap.add_argument(
        '--course-map', type=Path, default=None,
        help='optional gate map; enables the ODOMETRY check on logs that have it',
    )
    ap.add_argument(
        '--min-rigidity', type=float, default=0.30,
        help='minimum R^2 of the fitted rotation-flow model',
    )
    ap.add_argument(
        '--min-coupling', type=float, default=0.25,
        help='|lagged corr(bearing, response)| that counts as coupled',
    )
    ap.add_argument(
        '--max-bearing-deg', type=float, default=8.0,
        help='median bearing to the tracked gate that counts as centred',
    )
    ap.add_argument(
        '--max-pass-offset-px', type=float, default=140.0,
        help='max median |centroid_u - image centre| at a gate pass',
    )
    ap.add_argument(
        '--min-pass-area', type=float, default=15000.0,
        help='apparent gate area (px) that counts as the gate being passed '
             'rather than the next one already acquired',
    )
    args = ap.parse_args()

    if not args.telem.is_file():
        print(f'missing telem: {args.telem}')
        return 2
    with args.telem.open(newline='') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print('empty telem')
        return 2

    vis_rows = sum(1 for r in rows if row_centroid(r) is not None)
    print(f'telem={args.telem.name}  rows={len(rows)}  with_detection={vis_rows}')

    att = attitude_source(rows)
    trusted = att['att_raw'] + att['ahrs']
    print('\n=== attitude channel ===')
    print(f'  {att}')
    print(f"  trusted (gravity-referenced): {trusted}/{len(rows)} "
          f'({100.0 * trusted / max(len(rows), 1):.1f}%)')
    if att['att_raw'] == 0:
        print(
            '  note: sim ATTITUDE absent on this build, so the controller AHRS '
            'is the source. Expected; not a problem.'
        )
    if att['ekf'] or att['none']:
        print(
            f"  {att['ekf'] + att['none']} row(s) have no trusted attitude and "
            'are excluded from training by tools/train_policy.py.'
        )

    rigid = check_rigidity(rows)
    print('\n=== check 1: rigidity (fitted gyro flow vs measured centroid) ===')

    def _coef_s(coef):
        return '[' + ', '.join(f'{c:+.3f}' for c in coef) + ']' if coef else 'n/a'

    print(f"  keypoint pairs={rigid['n']}  skipped {rigid['duplicate_frames']} "
          f"re-logged frames, {rigid['large_rotation']} large-rotation steps")
    print(f"  horizontal R^2={rigid['r2_u']:+.3f}  coef={_coef_s(rigid['coef_u'])}")
    print(f"  vertical   R^2={rigid['r2_v']:+.3f}  coef={_coef_s(rigid['coef_v'])}")
    print(f"  rotation share of observed motion: {rigid['rotation_share']:.3f}")

    ident = check_identity(rows, min_pass_area=args.min_pass_area)
    print('\n=== check 2: identity at gate passes ===')
    print(f"  passes={ident['n_passes']}  with_vision={ident['n_with_vision']}  "
          f"near-gate samples={ident['n_near']}")
    print(f"  of those: median|u-cx|={ident['median_u_offset_px']:.1f} px  "
          f"median_area={ident['median_area_px']:.0f} px")
    for p in ident['passes']:
        tag = '' if p['near'] else '   (next gate — excluded)'
        print(f"    gate {p['gate']}: |u-cx|={p['u_offset_px']:.1f} px  "
              f"area={p['area_px']:.0f} px  age={p['age_s']:.2f} s{tag}")

    centre = check_centring(rows)
    print('\n=== check 3a: is the tracked gate where the drone is heading ===')
    print(f"  frames={centre['n']}  median|u-cx|={centre['median_offset_px']:.1f} px  "
          f"p90={centre['p90_offset_px']:.1f} px  "
          f"within 60px={100 * centre['frac_within_60px']:.1f}%")
    print(f"  median bearing={centre['median_bearing_deg']:.1f} deg")
    print(f"  centroid jumps: median={centre['median_jump_px']:.1f} px  "
          f">80px={100 * centre['frac_jump_over_80px']:.1f}% of "
          f"{centre['n_jumps']} steps")

    coup = check_coupling(rows)
    print('\n=== check 3b: bearing to pilot response coupling (diagnostic) ===')
    print(f"  n={coup['n']}  dt={coup['dt_s']:.3f}s  max_lag={coup['max_lag']}")
    for key in ('cmd_roll_rate', 'cmd_yaw_rate', 'roll_angle'):
        c = coup.get(key) or {}
        r = c.get('r', float('nan'))
        print(f"    {key:14s} n={c.get('n', 0):5d}  r={r:+.3f}  "
              f"lag={c.get('lag_s', float('nan')):.2f}s")
    print(f"  best channel: {coup['best_channel']} |r|={coup['corr_best']:.3f}")

    course = load_course_map(args.course_map)
    truth = None
    if course:
        truth = check_truth(rows, course)
        print('\n=== check 0: ODOMETRY bearing (optional) ===')
        if truth['n'] == 0:
            print('  skipped — no finite odo_* in this log')
        else:
            print(f"  n={truth['n']}  corr={truth['corr_bearing']:+.3f}  "
                  f"median u error={truth['median_u_err_px']:+.1f} px")

    failures: list[str] = []
    warnings: list[str] = []
    # Rigidity is a diagnostic, never a blocker. Translation flow it cannot
    # model scales with speed over depth, and at racing speed close to a gate
    # that term dominates, so a low score here does not by itself convict the
    # detector. Checks 2 and 3 decide.
    if rigid['n'] < 30:
        warnings.append(
            f"rigidity: only {rigid['n']} small-rotation frame pairs "
            f"({rigid['large_rotation']} steps rotated too far for the linear "
            'flow model) — not enough to judge'
        )
    elif not math.isfinite(rigid['r2']) or rigid['r2'] < args.min_rigidity:
        warnings.append(
            f"rigidity: R^2 {rigid['r2']:+.3f} < {args.min_rigidity:.2f}. "
            'Expected on every racing lap — translation toward the gate '
            'dominates the flow and cannot be modelled without range. '
            'Identity stability is judged by the jump statistic in check 3a.'
        )
    # A gate pass is a single instant, and the detector may legitimately have
    # switched to the *next* gate by then, so this check is noisy per-pass and
    # only carries weight with several samples.
    if ident['n_near'] == 0:
        warnings.append(
            'identity: no gate pass had a close-range detection of the gate '
            'being cleared — nothing to judge'
        )
    elif ident['n_near'] < 5:
        warnings.append(
            f"identity: only {ident['n_near']} gate passes had a close-range "
            'detection — too few to judge'
        )
    elif ident['median_u_offset_px'] > args.max_pass_offset_px:
        failures.append(
            f"identity: median |u-cx| {ident['median_u_offset_px']:.0f} px at gate "
            f'passes exceeds {args.max_pass_offset_px:.0f} px — the tracked gate '
            'is probably not the active one'
        )
    # Either the pilot nulls the bearing error (centred) or the error visibly
    # drives the stick (correlated). Either one proves the observation refers to
    # the gate actually being flown. Only failing *both* reproduces the original
    # disaster: a large bearing to some other gate that the pilot ignores.
    centred = (
        math.isfinite(centre['median_bearing_deg'])
        and centre['median_bearing_deg'] <= args.max_bearing_deg
    )
    correlated = coup['corr_best'] >= args.min_coupling
    if centre['n'] < 30:
        failures.append(f"centring: only {centre['n']} frames with a detection")
    elif not centred and not correlated:
        failures.append(
            f"identity: median bearing {centre['median_bearing_deg']:.1f} deg is "
            f"off-centre AND best coupling |r| {coup['corr_best']:.3f} < "
            f"{args.min_coupling:.2f} — the detector is tracking a gate the "
            'pilot is not flying to. This is the +0.03 failure that broke the '
            'last attempt.'
        )
    elif not correlated:
        warnings.append(
            f"coupling: best |r| {coup['corr_best']:.3f} is low, but median "
            f"bearing is only {centre['median_bearing_deg']:.1f} deg — the pilot "
            'nulled the error, which leaves correlation nothing to measure. '
            'Centring carries the verdict.'
        )
    if (
        math.isfinite(centre['frac_jump_over_80px'])
        and centre['frac_jump_over_80px'] > 0.25
    ):
        warnings.append(
            f"identity stability: {100 * centre['frac_jump_over_80px']:.0f}% of "
            'frames jump more than 80 px — the tracked gate changes often'
        )
    if truth is not None and truth['n'] >= 30:
        if not math.isfinite(truth['corr_bearing']) or truth['corr_bearing'] < 0.5:
            failures.append(
                f"odometry: corr {truth['corr_bearing']:+.3f} < 0.50"
            )

    print('\n=== hard gate ===')
    for w in warnings:
        print(f'  WARN — {w}')
    if failures:
        for f in failures:
            print(f'  FAIL — {f}')
        print('\n  Do not train on this data yet.')
        return 1
    print('  PASS — observation is coupled to the gate being flown.')
    if warnings:
        print('  (one check was inconclusive; see WARN above)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
