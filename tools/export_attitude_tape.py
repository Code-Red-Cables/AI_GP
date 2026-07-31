"""Export an open-loop attitude tape from a telem CSV finishing lap.

Example (best 40.424s lap from run 230757, after last Y-reset):

  python tools/export_attitude_tape.py \\
      --telem logs/telem_20260730_230757.csv \\
      --t0 98.32 --t1 142.5 \\
      --out logs/best/attitude_best.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--telem', required=True, type=Path)
    ap.add_argument('--t0', type=float, required=True, help='wall time start (s)')
    ap.add_argument('--t1', type=float, required=True, help='wall time end (s)')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument(
        '--skip-idle', type=float, default=0.0,
        help='drop leading seconds where |des_pitch|<1° and |des_roll|<1°',
    )
    args = ap.parse_args()

    samples = []
    with args.telem.open(encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            t = _f(row.get('t'))
            if t is None or t < args.t0 or t > args.t1:
                continue
            dr = _f(row.get('desired_roll'))
            dp = _f(row.get('desired_pitch'))
            yr = _f(row.get('requested_yaw_rate'))
            if yr is None:
                yr = _f(row.get('cmd_yaw_rate'))
            thr = _f(row.get('cmd_thrust'))
            if None in (dr, dp, yr, thr):
                continue
            samples.append({
                't': round(t - args.t0, 4),
                'des_roll': round(dr, 5),
                'des_pitch': round(dp, 5),
                'yaw_rate': round(yr, 5),
                'thrust': round(thr, 5),
            })

    if not samples:
        print('[FAIL] no samples in window')
        return 1

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
        'source': str(args.telem).replace('\\', '/'),
        't0_wall': args.t0,
        't1_wall': args.t1,
        'duration_s': samples[-1]['t'] if samples else 0.0,
        'n': len(samples),
        'samples': samples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out) + '\n', encoding='utf-8')
    print(
        f'[OK] {len(samples)} samples, {out["duration_s"]:.2f}s -> {args.out}',
        flush=True,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
