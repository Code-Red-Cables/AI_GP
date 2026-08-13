"""How far does the human tilt, and how fast does the pitch command alternate?

Two questions, both answered from the logs rather than guessed:

  1. The attitude envelope the training data covers. Outside it the policy has
     never seen an input and its output means nothing, so it is the natural
     place to put a guard.
  2. How often the commanded pitch rate changes sign. A human flying acro
     dithers the stick to hold an average rate; a policy that picks one bin and
     holds it integrates that rate into a tumble.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as _cfg  # noqa: E402
from race_obs import attitude_from_row  # noqa: E402


def _f(row, key):
    v = row.get(key)
    if v is None or v == '':
        return None
    try:
        f = float(v)
    except ValueError:
        return None
    return None if f != f else f


def pct(sorted_vals, q):
    if not sorted_vals:
        return float('nan')
    i = min(len(sorted_vals) - 1, max(0, int(q * (len(sorted_vals) - 1))))
    return sorted_vals[i]


def scan(paths, min_rows):
    rolls, pitches = [], []
    flips = 0
    steps = 0
    sat_runs = []          # lengths of consecutive same-sign saturated stretches
    used = 0
    sp = float(getattr(_cfg, 'RATE_SIGN_PITCH', 1.0))

    for p in paths:
        with open(p, newline='') as fh:
            rows = list(csv.DictReader(fh))
        if len(rows) < min_rows:
            continue
        live = [r for r in rows if _f(r, 'cmd_thrust') is not None]
        if len(live) < 50:
            continue
        used += 1
        prev_sign = 0
        run_len = 0
        for r in live:
            roll, pitch = attitude_from_row(r)
            if roll is not None:
                rolls.append(abs(math.degrees(roll)))
            if pitch is not None:
                pitches.append(abs(math.degrees(pitch)))
            cp = _f(r, 'cmd_pitch_rate')
            if cp is None:
                continue
            req = cp / sp
            sign = 0 if abs(req) < 0.5 else (1 if req > 0 else -1)
            steps += 1
            if sign != 0 and prev_sign != 0 and sign != prev_sign:
                flips += 1
            if abs(req) > 2.9:
                run_len += 1
            else:
                if run_len:
                    sat_runs.append(run_len)
                run_len = 0
            if sign != 0:
                prev_sign = sign
        if run_len:
            sat_runs.append(run_len)

    rolls.sort()
    pitches.sort()
    return {
        'runs': used,
        'n': len(pitches),
        'roll_p99': pct(rolls, 0.99),
        'roll_max': rolls[-1] if rolls else float('nan'),
        'pitch_p99': pct(pitches, 0.99),
        'pitch_max': pitches[-1] if pitches else float('nan'),
        'flip_rate': (flips / steps) if steps else float('nan'),
        'sat_run_mean': (sum(sat_runs) / len(sat_runs)) if sat_runs else 0.0,
        'sat_run_max': max(sat_runs) if sat_runs else 0,
    }


def report(label, s):
    print(f'{label}: {s["runs"]} run(s), {s["n"]} commanded frames')
    print(f'  |roll|   p99={s["roll_p99"]:6.1f} deg   max={s["roll_max"]:6.1f}')
    print(f'  |pitch|  p99={s["pitch_p99"]:6.1f} deg   max={s["pitch_max"]:6.1f}')
    print(f'  pitch cmd sign flips: {100.0 * s["flip_rate"]:.1f}% of frames')
    print(f'  saturated stretch: mean={s["sat_run_mean"]:.1f} frames  '
          f'max={s["sat_run_max"]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default='logs/telem_*.csv')
    ap.add_argument('--min-rows', type=int, default=1200)
    ap.add_argument('--policy', default=None,
                    help='a single policy-flown run to contrast')
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if args.policy:
        paths = [p for p in paths if os.path.abspath(p)
                 != os.path.abspath(args.policy)]
    report('human/all laps', scan(paths, args.min_rows))
    if args.policy:
        print()
        report('policy run', scan([args.policy], 0))


if __name__ == '__main__':
    main()
