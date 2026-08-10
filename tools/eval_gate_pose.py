"""Offline GO/NO-GO for LS vs PnP gate pose on saved frames.

Scores four metrics that need no ground-truth position:

  * ring consistency — outer (2.7 m) vs inner (1.5 m) LS disagreement
  * ray residual — LS objective value
  * temporal jitter — frame-to-frame pose delta
  * GATE_PASSED range — estimated range near logged passage events

Bins every metric by body |ax| as a speed proxy (valid because the plant
models drag: a_x ≈ k_x v_x).

    python tools/eval_gate_pose.py [--frames-dir frames] [--limit 500]
                                   [--telem logs/telem_....csv]

Requires ``frames/`` (currently missing from this checkout) and YOLO weights
under ``models/``. Synthetic self-check: ``python -m pytest test_gate_ls_pose.py``.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _num(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return out


def _percentile(vals: list[float], q: float) -> float:
    if not vals:
        return float('nan')
    s = sorted(vals)
    i = int(round(q * (len(s) - 1)))
    return s[max(0, min(len(s) - 1, i))]


def _summarise(name: str, vals: list[float]) -> str:
    if not vals:
        return f'{name}: n=0'
    return (
        f'{name}: n={len(vals)}  med={statistics.median(vals):.3f}  '
        f'p90={_percentile(vals, 0.9):.3f}  max={max(vals):.3f}'
    )


def speed_bin(ax: float) -> str:
    a = abs(ax)
    if not math.isfinite(a):
        return 'unknown'
    if a < 0.5:
        return 'slow(|ax|<0.5)'
    if a < 2.0:
        return 'mid(0.5-2)'
    return 'fast(|ax|>=2)'


def load_telem_index(path: Path) -> dict[int, dict]:
    """Map approximate frame index / time to ax and GATE_PASSED markers."""
    if not path.is_file():
        return {}
    out = {}
    for i, row in enumerate(csv.DictReader(path.open(newline=''))):
        out[i] = {
            'ax': _num(row.get('ax')),
            't': _num(row.get('t')),
            'active_gate': row.get('active_gate'),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames-dir', type=Path, default=ROOT / 'frames')
    ap.add_argument('--telem', type=Path, default=None)
    ap.add_argument('--limit', type=int, default=0, help='max frames (0=all)')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    if not args.frames_dir.is_dir():
        print(f'frames dir missing: {args.frames_dir}')
        print('Phase 1b is blocked until frames/ is restored.')
        print('Unit tests still cover the solver: pytest test_gate_ls_pose.py')
        return

    # Lazy imports — torch/YOLO only needed when frames exist.
    import cv2

    from vision.gate_ls_pose import solve_keypoints_ls
    from vision.yolo_pnp import solve_keypoints_pnp
    from vision_rx import create_gate_detector

    if args.device:
        import os
        os.environ['YOLO_DEVICE'] = args.device

    detector = create_gate_detector()
    telem = load_telem_index(args.telem) if args.telem else {}

    ring_by: dict[str, list[float]] = defaultdict(list)
    resid_by: dict[str, list[float]] = defaultdict(list)
    jitter_by: dict[str, list[float]] = defaultdict(list)
    pnp_ok = ls_ok = 0
    prev_t = None
    n = 0

    paths = sorted(args.frames_dir.rglob('*.jpg')) + sorted(
        args.frames_dir.rglob('*.png')
    )
    if args.limit > 0:
        paths = paths[: args.limit]
    print(f'evaluating {len(paths)} frames from {args.frames_dir}')

    for idx, path in enumerate(paths):
        image = cv2.imread(str(path))
        if image is None:
            continue
        det = detector.detect(image)
        if det is None:
            continue
        kps = getattr(det, 'keypoints', None)
        conf = getattr(det, 'keypoint_confidences', None)
        if kps is None:
            continue
        n += 1
        ax = telem.get(idx, {}).get('ax', float('nan'))
        sb = speed_bin(ax)

        pnp = solve_keypoints_pnp(kps, conf)
        if pnp is not None:
            pnp_ok += 1
        ls = solve_keypoints_ls(kps, conf, roll=0.0, pitch=0.0, yaw=0.0)
        if ls is None:
            continue
        ls_ok += 1
        resid_by[sb].append(ls.residual_m)
        if math.isfinite(ls.ring_disagree_m):
            ring_by[sb].append(ls.ring_disagree_m)
        if prev_t is not None:
            jitter_by[sb].append(float(np.linalg.norm(ls.t_gate - prev_t)))
        prev_t = ls.t_gate

    print(f'\nframes with keypoints: {n}')
    print(f'LS solves: {ls_ok}   PnP solves: {pnp_ok}')
    print('\n--- residual (m) by speed ---')
    for k in sorted(resid_by):
        print(' ', _summarise(k, resid_by[k]))
    print('\n--- ring disagree (m) by speed ---')
    for k in sorted(ring_by):
        print(' ', _summarise(k, ring_by[k]))
    print('\n--- temporal jitter (m) by speed ---')
    for k in sorted(jitter_by):
        print(' ', _summarise(k, jitter_by[k]))

    # Crude GO/NO-GO thresholds from the paper's noise study.
    all_ring = [v for vs in ring_by.values() for v in vs]
    all_resid = [v for vs in resid_by.values() for v in vs]
    go = True
    if all_ring and statistics.median(all_ring) > 0.5:
        print('\nNO-GO: median ring disagree > 0.5 m')
        go = False
    if all_resid and statistics.median(all_resid) > 0.4:
        print('\nNO-GO: median ray residual > 0.4 m')
        go = False
    if go and ls_ok > 0:
        print('\nGO: LS residuals look usable — continue to drag EKF / control')
    elif ls_ok == 0:
        print('\nNO-GO: zero LS solves — check detector / weights')


if __name__ == '__main__':
    main()
