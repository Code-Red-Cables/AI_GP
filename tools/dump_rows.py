"""Dump selected telemetry columns verbatim for a time window.

Deliberately prints the raw strings, so an empty field is visibly empty rather
than being coerced to a plausible-looking zero.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv_path', nargs='?', default=None)
    ap.add_argument('--cols', default='t,control_authority,planner,cmd_thrust,'
                                      'cmd_roll_rate,cmd_pitch_rate,ahrs_roll,'
                                      'ahrs_pitch,att_source,active_gate,'
                                      'sim_boot_ms,race_start_ms')
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--end', type=float, default=1e18)
    ap.add_argument('--stride', type=int, default=1)
    args = ap.parse_args()

    path = args.csv_path
    if path is None:
        files = sorted(glob.glob('logs/telem_*.csv'), key=os.path.getmtime)
        path = files[-1]
    cols = [c.strip() for c in args.cols.split(',')]

    with open(path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    t0 = float(rows[0]['t'])

    print(' '.join(f'{c[:12]:>12}' for c in cols))
    n = 0
    for r in rows:
        rel = float(r['t']) - t0
        if rel < args.start or rel > args.end:
            continue
        n += 1
        if n % args.stride:
            continue
        vals = []
        for c in cols:
            v = r.get(c)
            if c == 't':
                v = f'{rel:.2f}'
            vals.append('(empty)' if v in (None, '') else v)
        print(' '.join(f'{v[:12]:>12}' for v in vals))


if __name__ == '__main__':
    main()
