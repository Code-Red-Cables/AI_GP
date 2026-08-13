"""Split "no gate" frames into seen-but-rejected versus genuinely nothing.

Requires ``gate_cand_n`` / ``gate_cand_conf``, added to the logger for exactly
this question, so it only works on runs recorded after that change.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os


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
    ap.add_argument('--glob', default='logs/telem_*.csv')
    args = ap.parse_args()

    if args.csv_path:
        paths = [args.csv_path]
    else:
        paths = sorted(glob.glob(args.glob), key=os.path.getmtime)[-1:]

    for path in paths:
        with open(path, newline='') as fh:
            rows = list(csv.DictReader(fh))
        if not rows or 'gate_cand_n' not in rows[0]:
            print(f'{os.path.basename(path)}: no gate_cand_n column — '
                  'recorded before candidate logging was added')
            continue

        armed = [r for r in rows if r.get('cmd_thrust') not in (None, '')]
        if not armed:
            armed = rows

        both = seen_rejected = neither = delivered_no_cand = 0
        rejected_confs = []
        for r in armed:
            n = _f(r, 'gate_cand_n') or 0.0
            has_det = (r.get('gate_method') or '').strip() not in ('', 'none')
            if n > 0 and has_det:
                both += 1
            elif n > 0 and not has_det:
                seen_rejected += 1
                c = _f(r, 'gate_cand_conf')
                if c is not None:
                    rejected_confs.append(c)
            elif n == 0 and has_det:
                delivered_no_cand += 1
            else:
                neither += 1

        total = len(armed)
        print(f'== {os.path.basename(path)}  {total} commanded frames')
        print(f'  detector found AND policy got it : {both:6d} '
              f'({100.0 * both / total:5.1f}%)')
        print(f'  detector found, policy got NOTHING: {seen_rejected:6d} '
              f'({100.0 * seen_rejected / total:5.1f}%)  <-- selection loss')
        print(f'  nothing in view at all           : {neither:6d} '
              f'({100.0 * neither / total:5.1f}%)')
        if delivered_no_cand:
            print(f'  policy got one with no raw box   : {delivered_no_cand:6d} '
                  '(held/predicted)')
        if rejected_confs:
            rejected_confs.sort()
            mid = rejected_confs[len(rejected_confs) // 2]
            print(f'  rejected box confidence: median={mid:.2f} '
                  f'min={rejected_confs[0]:.2f} max={rejected_confs[-1]:.2f}')
            print('  A high median here means the detector was confident and '
                  'selection threw it away — lowering the confidence '
                  'threshold would not have helped.')

        if 'gate_reject' in rows[0]:
            reasons = {}
            raw_methods = {}
            for r in armed:
                if (r.get('gate_method') or '').strip() not in ('', 'none'):
                    continue
                why = (r.get('gate_reject') or '').strip() or 'unrecorded'
                reasons[why] = reasons.get(why, 0) + 1
                m = (r.get('gate_raw_method') or '').strip() or 'none'
                raw_methods[m] = raw_methods.get(m, 0) + 1
            print()
            print('  why the policy got nothing:')
            for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
                print(f'    {why:<14} {n:6d} ({100.0 * n / total:5.1f}%)')
            print('  detector method on those frames:')
            for m, n in sorted(raw_methods.items(), key=lambda kv: -kv[1])[:6]:
                print(f'    {m:<40} {n:6d}')


if __name__ == '__main__':
    main()
