"""What do the commands look like just after the race starts?

Compares the human laps against a policy run over the same window, in
*requested* units (RATE_SIGN undone), so the two are directly comparable.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as _cfg  # noqa: E402


def _f(row, key, default=None):
    v = row.get(key)
    if v is None or v == '':
        return default
    try:
        f = float(v)
    except ValueError:
        return default
    return None if f != f else f


def armed_rows(path, window_s):
    """Rows from the first commanded frame up to window_s later."""
    with open(path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    live = [r for r in rows if _f(r, 'cmd_thrust') is not None]
    if not live:
        return []
    t0 = float(live[0]['t'])
    return [r for r in live if float(r['t']) - t0 <= window_s]


def summarise(path, window_s):
    rows = armed_rows(path, window_s)
    if not rows:
        return None
    sp = float(getattr(_cfg, 'RATE_SIGN_PITCH', 1.0))
    sr = float(getattr(_cfg, 'RATE_SIGN_ROLL', 1.0))
    thr = [_f(r, 'cmd_thrust', 0.0) for r in rows]
    pit = [_f(r, 'cmd_pitch_rate', 0.0) / sp for r in rows]
    rol = [_f(r, 'cmd_roll_rate', 0.0) / sr for r in rows]
    n = len(rows)
    sat = sum(1 for p in pit if abs(p) > 2.9)
    return {
        'name': os.path.basename(path),
        'n': n,
        'thr_mean': sum(thr) / n,
        'thr_max': max(thr),
        'pitch_mean': sum(pit) / n,
        'pitch_max': max(pit),
        'pitch_min': min(pit),
        'pitch_sat_pct': 100.0 * sat / n,
        'roll_absmax': max(abs(x) for x in rol),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default='logs/telem_*.csv')
    ap.add_argument('--window', type=float, default=2.0,
                    help='seconds after first command to summarise')
    ap.add_argument('--min-rows', type=int, default=800,
                    help='skip short runs (aborted attempts)')
    ap.add_argument('--only', default=None, help='single file to summarise')
    args = ap.parse_args()

    paths = [args.only] if args.only else sorted(glob.glob(args.glob))
    hdr = (f'{"run":>28} {"n":>4} {"thr":>6} {"thrmx":>6} {"pitch":>7} '
           f'{"pmax":>6} {"pmin":>6} {"sat%":>5} {"|roll|":>6}')
    print(hdr)
    print('-' * len(hdr))
    for p in paths:
        with open(p, newline='') as fh:
            total = sum(1 for _ in fh) - 1
        if not args.only and total < args.min_rows:
            continue
        s = summarise(p, args.window)
        if s is None:
            continue
        print(f'{s["name"][-28:]:>28} {s["n"]:4d} {s["thr_mean"]:6.3f} '
              f'{s["thr_max"]:6.3f} {s["pitch_mean"]:+7.2f} '
              f'{s["pitch_max"]:+6.2f} {s["pitch_min"]:+6.2f} '
              f'{s["pitch_sat_pct"]:5.0f} {s["roll_absmax"]:6.2f}')


if __name__ == '__main__':
    main()
