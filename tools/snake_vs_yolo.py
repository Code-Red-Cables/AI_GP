"""Which detector saw the gate, YOLO or the colour snake?

Reads a run recorded with SNAKE_GATE_ENABLED=1 and cross-tabulates the two
detectors frame by frame. The interesting cell is "snake only": frames the pose
model lost and colour recovered. If that number is large the colour method is
worth fusing in; if it is near zero it is not, whatever its standalone rate.
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
    if not rows or 'snake_n' not in rows[0]:
        print(f'{os.path.basename(path)}: no snake_n column. Re-run with '
              'SNAKE_GATE_ENABLED=1 to record it.')
        return

    armed = [r for r in rows if r.get('cmd_thrust') not in (None, '')] or rows
    print(f'== {os.path.basename(path)}  {len(armed)} commanded frames')

    both = yolo_only = snake_only = neither = 0
    ms = []
    masks = []
    for r in armed:
        y = (r.get('gate_method') or '').strip() not in ('', 'none')
        s = (_f(r, 'snake_n') or 0) > 0
        both += bool(y and s)
        yolo_only += bool(y and not s)
        snake_only += bool(s and not y)
        neither += bool(not y and not s)
        t = _f(r, 'snake_ms')
        if t is not None:
            ms.append(t)
        mk = _f(r, 'snake_mask')
        if mk is not None:
            masks.append(mk)

    n = len(armed)
    print()
    print(f'  both detectors        {both:6d} ({100.0 * both / n:5.1f}%)')
    print(f'  YOLO only            {yolo_only:6d} '
          f'({100.0 * yolo_only / n:5.1f}%)')
    print(f'  snake only           {snake_only:6d} '
          f'({100.0 * snake_only / n:5.1f}%)  <-- what colour would add')
    print(f'  neither              {neither:6d} ({100.0 * neither / n:5.1f}%)')
    print()
    print(f'  YOLO  coverage {100.0 * (both + yolo_only) / n:5.1f}%')
    print(f'  snake coverage {100.0 * (both + snake_only) / n:5.1f}%')
    print(f'  union coverage {100.0 * (n - neither) / n:5.1f}%')
    if ms:
        ms.sort()
        print()
        print(f'  snake cost: median {ms[len(ms) // 2]:.1f} ms  '
              f'p95 {ms[int(0.95 * (len(ms) - 1))]:.1f} ms  max {ms[-1]:.1f} ms')
    if masks:
        masks.sort()
        print(f'  gate-coloured pixels: median '
              f'{100.0 * masks[len(masks) // 2]:.2f}% of frame')

    # The claim worth testing: does colour survive roll better than the CNN?
    print()
    print('coverage vs |roll|:')
    hdr = f'{"|roll| deg":>12} {"frames":>7} {"yolo%":>7} {"snake%":>7}'
    print(hdr)
    print('-' * len(hdr))
    for lo, hi in [(0, 15), (15, 30), (30, 45), (45, 60), (60, 90),
                   (90, 1e9)]:
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
        y = sum(1 for r in sel
                if (r.get('gate_method') or '').strip() not in ('', 'none'))
        s = sum(1 for r in sel if (_f(r, 'snake_n') or 0) > 0)
        label = f'{lo}-{hi}' if hi < 1e9 else f'>{lo}'
        print(f'{label:>12} {len(sel):7d} {100.0 * y / len(sel):7.1f} '
              f'{100.0 * s / len(sel):7.1f}')


if __name__ == '__main__':
    main()
