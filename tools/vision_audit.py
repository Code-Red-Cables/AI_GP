"""How often did the detector see a gate, and how close were the misses?

Separates the two thresholds, because they are not equivalent:

  YOLO_CONFIDENCE_THRESHOLD (box, default 0.25)
      applied inside the detector at *record* time. A frame rejected here is
      absent from the log entirely, so lowering it cannot recover anything from
      runs already on disk -- it only changes future flights, which is exactly
      how a train/flight mismatch is created.

  YOLO_KEYPOINT_CONFIDENCE_THRESHOLD / RACE_MIN_KP_CONF (per corner, 0.25)
      applied when the observation is *built*. The logs keep every corner's raw
      confidence, so this one can be lowered retroactively and the existing runs
      re-derive for free.

The histogram of logged box confidence answers whether 0.25 was actually
rejecting anything close, and the per-corner histogram says how many corners a
lower corner threshold would hand back.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
    ap.add_argument('--glob', default='logs/telem_*.csv')
    ap.add_argument('--min-rows', type=int, default=1200)
    ap.add_argument('--only', default=None)
    args = ap.parse_args()

    paths = [args.only] if args.only else [
        p for p in sorted(glob.glob(args.glob))
        if sum(1 for _ in open(p)) - 1 >= args.min_rows
    ]

    total = 0
    armed = 0
    with_box = 0
    kp_hist = {}
    box_hist = {}
    kp_counts = {}
    predicted = 0
    method = {}

    edges = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.45, 0.50,
             0.60, 0.70, 0.80, 0.90, 1.01]

    def bucket(v):
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1]:
                return f'{edges[i]:.2f}-{edges[i + 1]:.2f}'
        return '>=1.0'

    for p in paths:
        with open(p, newline='') as fh:
            for r in csv.DictReader(fh):
                total += 1
                if r.get('cmd_thrust') in (None, ''):
                    continue
                armed += 1
                conf = _f(r, 'gate_conf')
                if conf is not None and conf > 0.0:
                    with_box += 1
                    box_hist[bucket(conf)] = box_hist.get(bucket(conf), 0) + 1
                m = (r.get('gate_method') or '').strip() or 'none'
                method[m] = method.get(m, 0) + 1
                if str(r.get('gate_predicted') or '').strip().lower() in (
                    '1', 'true', 'yes'
                ):
                    predicted += 1
                n_seen = 0
                for i in range(8):
                    c = _f(r, f'kp{i}_c')
                    if c is None:
                        continue
                    if c > 0.0:
                        kp_hist[bucket(c)] = kp_hist.get(bucket(c), 0) + 1
                    if c >= 0.25:
                        n_seen += 1
                kp_counts[n_seen] = kp_counts.get(n_seen, 0) + 1

    print(f'runs={len(paths)}  rows={total}  commanded={armed}')
    if not armed:
        return
    print(f'frames with a box detection: {with_box}/{armed} '
          f'({100.0 * with_box / armed:.1f}%)')
    print(f'frames flagged predicted (extrapolated, not measured): {predicted} '
          f'({100.0 * predicted / armed:.1f}%)')
    print(f'detector method mix: {method}')

    print()
    print('corners passing the 0.25 corner threshold, per frame:')
    for k in sorted(kp_counts):
        n = kp_counts[k]
        print(f'  {k} of 8 : {n:7d}  ({100.0 * n / armed:5.1f}%)')

    print()
    print('logged BOX confidence histogram (already filtered at 0.25):')
    for b in sorted(box_hist, key=lambda s: s):
        print(f'  {b:>11} {box_hist[b]:7d}')

    print()
    print('logged CORNER confidence histogram (raw, filtered at build time):')
    recoverable = 0
    for b in sorted(kp_hist, key=lambda s: s):
        n = kp_hist[b]
        print(f'  {b:>11} {n:7d}')
        try:
            lo = float(b.split('-')[0])
        except ValueError:
            lo = 1.0
        if 0.10 <= lo < 0.25:
            recoverable += n
    print()
    print(f'corners in 0.10-0.25 (recoverable by lowering the corner '
          f'threshold, no re-recording): {recoverable}')


if __name__ == '__main__':
    main()
