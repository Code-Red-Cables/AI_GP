"""Post-mortem for a single flight: what did the policy see, and what did it do?

Prints a time-sliced table of the attitude, the commands, and how much of the
gate was visible, so a crash can be attributed to vision, to the command, or to
the plant rather than guessed at.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _f(row, key, default=float('nan')):
    v = row.get(key)
    if v is None or v == '':
        return default
    try:
        return float(v)
    except ValueError:
        return default


def n_visible(row):
    n = 0
    for i in range(8):
        c = _f(row, f'kp{i}_c', 0.0)
        if c > 0.0:
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv_path', nargs='?', default=None)
    ap.add_argument('--every', type=float, default=1.0,
                    help='seconds between printed rows')
    args = ap.parse_args()

    path = args.csv_path
    if path is None:
        files = sorted(glob.glob('logs/telem_*.csv'), key=os.path.getmtime)
        path = files[-1]
    print(f'== {path}')

    with open(path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print('empty')
        return

    t0 = _f(rows[0], 't', 0.0)
    print(f'rows={len(rows)}  span={_f(rows[-1], "t") - t0:.1f}s')

    auth = {}
    for r in rows:
        a = (r.get('control_authority') or '?').strip()
        auth[a] = auth.get(a, 0) + 1
    print('authority:', auth)

    plan = {}
    for r in rows:
        p = (r.get('planner') or '?').strip()
        plan[p] = plan.get(p, 0) + 1
    print('planner:', plan)

    print()
    hdr = (f'{"t":>6} {"auth":>6} {"gate":>4} {"kp":>2} {"roll":>7} '
           f'{"pitch":>7} {"thr":>5} {"c_roll":>7} {"c_pitch":>7} '
           f'{"alt":>7} {"vd":>6}')
    print(hdr)
    print('-' * len(hdr))
    next_t = t0
    for r in rows:
        t = _f(r, 't')
        if t < next_t:
            continue
        next_t = t + args.every
        a = (r.get('control_authority') or '?').strip()[:6]
        print(f'{t - t0:6.1f} {a:>6} {_f(r, "active_gate", -1):4.0f} '
              f'{n_visible(r):2d} '
              f'{math.degrees(_f(r, "ahrs_roll", 0.0)):7.1f} '
              f'{math.degrees(_f(r, "ahrs_pitch", 0.0)):7.1f} '
              f'{_f(r, "cmd_thrust", 0.0):5.2f} '
              f'{_f(r, "cmd_roll_rate", 0.0):7.2f} '
              f'{_f(r, "cmd_pitch_rate", 0.0):7.2f} '
              f'{-_f(r, "pos_d", 0.0):7.1f} '
              f'{_f(r, "vel_d", 0.0):6.1f}')

    # Summary of what the policy asked for while it had authority.
    pol = [r for r in rows
           if (r.get('control_authority') or '').strip() in ('policy', 'auto')]
    if pol:
        thr = [_f(r, 'cmd_thrust', 0.0) for r in pol]
        rr = [_f(r, 'cmd_roll_rate', 0.0) for r in pol]
        pr = [_f(r, 'cmd_pitch_rate', 0.0) for r in pol]
        kp = [n_visible(r) for r in pol]
        print()
        print(f'while policy flew ({len(pol)} rows):')
        print(f'  thrust      mean={sum(thr)/len(thr):.3f} '
              f'min={min(thr):.3f} max={max(thr):.3f}')
        print(f'  roll_rate   mean={sum(rr)/len(rr):+.3f} '
              f'min={min(rr):+.2f} max={max(rr):+.2f}')
        print(f'  pitch_rate  mean={sum(pr)/len(pr):+.3f} '
              f'min={min(pr):+.2f} max={max(pr):+.2f}')
        blind = sum(1 for k in kp if k == 0)
        print(f'  frames with zero keypoints: {blind}/{len(kp)} '
              f'({100.0 * blind / len(kp):.0f}%)')
        print(f'  mean keypoints visible: {sum(kp)/len(kp):.1f}/8')


if __name__ == '__main__':
    main()
