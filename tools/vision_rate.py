"""Is the detector missing gates, or is the log simply faster than the camera?

A row with no detection means one of two very different things:

  * the detector ran on a frame and found nothing        -> a vision problem
  * no new camera frame had arrived since the last row   -> a sampling artifact

They are told apart by ``gate_frame_id``: if the distinct frame count matches
the number of rows carrying a detection, every detection appears exactly once
and the blank rows are simply rows between camera frames.
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
    ap.add_argument('--glob', default='logs/telem_*.csv')
    ap.add_argument('--min-rows', type=int, default=1200)
    ap.add_argument('--max-runs', type=int, default=6)
    args = ap.parse_args()

    paths = [p for p in sorted(glob.glob(args.glob))
             if sum(1 for _ in open(p)) - 1 >= args.min_rows][:args.max_runs]

    hdr = (f'{"run":>26} {"rows":>6} {"span_s":>7} {"log_hz":>7} '
           f'{"det_rows":>8} {"frames":>7} {"vis_hz":>7} {"gap_p95":>8}')
    print(hdr)
    print('-' * len(hdr))
    for p in paths:
        with open(p, newline='') as fh:
            rows = list(csv.DictReader(fh))
        t0, t1 = float(rows[0]['t']), float(rows[-1]['t'])
        span = max(1e-6, t1 - t0)
        det_rows = 0
        ids = set()
        det_times = []
        for r in rows:
            fid = r.get('gate_frame_id')
            has = (r.get('gate_method') or '').strip() not in ('', 'none')
            if has:
                det_rows += 1
                det_times.append(float(r['t']))
                if fid not in (None, ''):
                    ids.add(fid)
        gaps = [b - a for a, b in zip(det_times, det_times[1:])]
        gaps.sort()
        gap95 = gaps[int(0.95 * (len(gaps) - 1))] if gaps else float('nan')
        print(f'{os.path.basename(p)[-26:]:>26} {len(rows):6d} {span:7.1f} '
              f'{len(rows) / span:7.1f} {det_rows:8d} {len(ids):7d} '
              f'{det_rows / span:7.1f} {gap95:8.2f}')

    print()
    print('det_rows == frames  -> each detection logged once; blanks are '
          'rows between camera frames (sampling, not blindness)')
    print('det_rows >> frames  -> detections are held across rows')
    print('gap_p95 is the 95th percentile seconds between detections')


if __name__ == '__main__':
    main()
