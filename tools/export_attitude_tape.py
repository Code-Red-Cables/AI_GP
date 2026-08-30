"""Export an open-loop attitude tape from a telem CSV finishing lap.

Angle tape (lean setpoints):

  python tools/export_attitude_tape.py \\
      --telem logs/telem_20260730_230757.csv \\
      --t0 98.32 --t1 142.5 \\
      --out logs/best/attitude_best.json

Acro rate tape from a 0.2x seed (times written in *sim* seconds):

  python tools/export_attitude_tape.py \\
      --telem logs/seed/telem_20260820_104720.csv \\
      --rates --from-motion --sim-scale 0.2 \\
      --out logs/seed/playback_104720.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _unsign(value, sign: float) -> float | None:
    v = _f(value)
    if v is None or sign == 0.0:
        return v
    return v / float(sign)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--telem', required=True, type=Path)
    ap.add_argument('--t0', type=float, default=None, help='wall time start (s)')
    ap.add_argument('--t1', type=float, default=None, help='wall time end (s)')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument(
        '--rates', action='store_true',
        help='acro body-rate tape from cmd_* (pre RATE_SIGN). '
             'Replay with fly/acro --replay-attitude.',
    )
    ap.add_argument(
        '--sim-scale', type=float, default=1.0,
        help='sim seconds per wall second (0.2 for CE 0.2x seeds). '
             'Tape timestamps are multiplied so 1x replay matches physics.',
    )
    ap.add_argument(
        '--from-motion', action='store_true',
        help='start at the first |rate| > 0.3 rad/s (or 1° lean)',
    )
    ap.add_argument(
        '--skip-idle', type=float, default=0.0,
        help='drop leading seconds where |des_pitch|<1° and |des_roll|<1°',
    )
    args = ap.parse_args()
    scale = float(args.sim_scale)
    if scale <= 0.0:
        print('[FAIL] --sim-scale must be > 0')
        return 1

    import config as _cfg
    sign_r = float(getattr(_cfg, 'RATE_SIGN_ROLL', 1.0))
    sign_p = float(getattr(_cfg, 'RATE_SIGN_PITCH', 1.0))
    sign_y = float(getattr(_cfg, 'RATE_SIGN_YAW', 1.0))

    raw = []
    with args.telem.open(encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            t = _f(row.get('t'))
            if t is None:
                continue
            if args.t0 is not None and t < args.t0:
                continue
            if args.t1 is not None and t > args.t1:
                continue
            if args.rates:
                dr = _unsign(row.get('cmd_roll_rate'), sign_r)
                dp = _unsign(row.get('cmd_pitch_rate'), sign_p)
                yr = _unsign(row.get('cmd_yaw_rate'), sign_y)
            else:
                dr = _f(row.get('desired_roll'))
                dp = _f(row.get('desired_pitch'))
                yr = _f(row.get('requested_yaw_rate'))
                if yr is None:
                    yr = _unsign(row.get('cmd_yaw_rate'), sign_y)
            thr = _f(row.get('cmd_thrust'))
            if None in (dr, dp, yr, thr):
                continue
            raw.append((t, dr, dp, yr, thr))

    if not raw:
        print('[FAIL] no samples in window')
        return 1

    if args.from_motion:
        cut_i = None
        for i, (_t, dr, dp, yr, _thr) in enumerate(raw):
            if args.rates:
                if max(abs(dr), abs(dp), abs(yr)) > 0.3:
                    cut_i = i
                    break
            elif (
                abs(dp) > math.radians(1.0)
                or abs(dr) > math.radians(1.0)
                or abs(yr) > math.radians(5.0)
            ):
                cut_i = i
                break
        if cut_i is not None:
            raw = raw[cut_i:]

    t0_wall = raw[0][0]
    samples = []
    for t, dr, dp, yr, thr in raw:
        samples.append({
            't': round((t - t0_wall) * scale, 4),
            'des_roll': round(dr, 5),
            'des_pitch': round(dp, 5),
            'yaw_rate': round(yr, 5),
            'thrust': round(thr, 5),
        })

    if args.skip_idle > 0:
        cut = None
        for s in samples:
            if (
                abs(s['des_pitch']) > math.radians(1.0)
                or abs(s['des_roll']) > math.radians(1.0)
                or abs(s['yaw_rate']) > math.radians(5.0)
            ):
                if s['t'] >= args.skip_idle:
                    cut = s['t']
                    break
        if cut is not None:
            samples = [
                {**s, 't': round(s['t'] - cut, 4)}
                for s in samples
                if s['t'] >= cut
            ]

    out = {
        'type': 'attitude_tape',
        'control': 'acro_rates' if args.rates else 'angle',
        'sim_speed': 1.0,
        'source': str(args.telem).replace('\\', '/'),
        't0_wall': t0_wall,
        't1_wall': raw[-1][0],
        'record_wall_scale': scale,
        'duration_s': samples[-1]['t'] if samples else 0.0,
        'n': len(samples),
        'samples': samples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out) + '\n', encoding='utf-8')
    print(
        f'[OK] {len(samples)} samples, {out["duration_s"]:.2f}s sim '
        f'(wall x{scale:.2f}, {out["control"]}) -> {args.out}',
        flush=True,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
