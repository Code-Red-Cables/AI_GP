"""Test two specific claims about the detector against a flown run.

  1. "when I rotate, the detection leaves"  -> detection rate bucketed by |roll|
  2. "I see a far gate but not the near one" -> box area distribution, and how
     often the strongest candidate is small

Uses the raw candidate columns, so a miss by the detector is separated from a
rejection by the selection logic.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv_path', nargs='?', default=None)
    args = ap.parse_args()

    path = args.csv_path
    if path is None:
        path = sorted(glob.glob('logs/telem_*.csv'), key=os.path.getmtime)[-1]
    with open(path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    if 'gate_cand_n' not in (rows[0] if rows else {}):
        print('run predates candidate logging')
        return
    armed = [r for r in rows if r.get('cmd_thrust') not in (None, '')]
    print(f'== {os.path.basename(path)}  {len(armed)} commanded frames')

    # ---- claim 1: roll kills detection ------------------------------------
    buckets = [(0, 10), (10, 20), (20, 30), (30, 45), (45, 60), (60, 90),
               (90, 180), (180, 1e9)]
    print()
    print('detection rate vs |roll| (cand = YOLO found a box at all):')
    hdr = f'{"|roll| deg":>12} {"frames":>7} {"cand%":>7} {"got%":>7}'
    print(hdr)
    print('-' * len(hdr))
    for lo, hi in buckets:
        sel = []
        for r in armed:
            roll, _p = attitude_from_row(r)
            if roll is None or not math.isfinite(roll):
                continue
            d = abs(math.degrees(roll))
            if lo <= d < hi:
                sel.append(r)
        if not sel:
            continue
        cand = sum(1 for r in sel if (_f(r, 'gate_cand_n') or 0) > 0)
        got = sum(1 for r in sel
                  if (r.get('gate_method') or '').strip() not in ('', 'none'))
        label = f'{lo}-{hi}' if hi < 1e9 else f'>{lo}'
        print(f'{label:>12} {len(sel):7d} {100.0 * cand / len(sel):7.1f} '
              f'{100.0 * got / len(sel):7.1f}')

    # ---- claim 2: near gates missed ---------------------------------------
    # gate_area is the delivered detection's box area. Big area == near gate.
    areas = [(_f(r, 'gate_area') or 0.0) for r in armed
             if (r.get('gate_method') or '').strip() not in ('', 'none')]
    areas = [a for a in areas if a > 0]
    areas.sort()
    print()
    if areas:
        def q(p):
            return areas[min(len(areas) - 1, int(p * (len(areas) - 1)))]
        frame_px = 640.0 * 360.0
        print(f'delivered box area, {len(areas)} detections '
              f'(frame = {int(frame_px)} px):')
        for p in (0.05, 0.25, 0.50, 0.75, 0.95, 1.0):
            a = q(p)
            print(f'  p{int(p * 100):3d}  {a:9.0f} px  '
                  f'({100.0 * a / frame_px:5.1f}% of frame)')
        big = sum(1 for a in areas if a > 0.25 * frame_px)
        print(f'  detections filling >25% of frame: {big} '
              f'({100.0 * big / len(areas):.1f}%)')
        print('  If near-gate detections are rare while you were flying at '
              'gates, the detector is failing at large scale.')

    # ---- rejection vs roll -------------------------------------------------
    rej = [r for r in armed
           if (_f(r, 'gate_cand_n') or 0) > 0
           and (r.get('gate_method') or '').strip() in ('', 'none')]
    print()
    print(f'rejected-but-seen frames: {len(rej)}')
    if rej:
        rolls = []
        for r in rej:
            roll, _p = attitude_from_row(r)
            if roll is not None and math.isfinite(roll):
                rolls.append(abs(math.degrees(roll)))
        if rolls:
            rolls.sort()
            print(f'  |roll| on those frames: median='
                  f'{rolls[len(rolls) // 2]:.0f} deg  max={rolls[-1]:.0f} deg')


if __name__ == '__main__':
    main()
