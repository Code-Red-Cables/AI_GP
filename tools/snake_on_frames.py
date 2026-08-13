"""Run snake gate detection over captured frames, offline.

These frames were saved on *confirmed* YOLO detections, so every one of them
contains a gate the pose model already found. That makes the hit rate here a
direct read on the colour method: anything it misses is a frame with a known,
YOLO-confirmed gate in it.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as _cfg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directory', nargs='?', default=None,
                    help='frames/run_* directory (default: newest)')
    ap.add_argument('--limit', type=int, default=150)
    ap.add_argument('--min-fitness', type=float, default=None)
    ap.add_argument('--min-length', type=int, default=None)
    ap.add_argument('--axis-aligned', action='store_true',
                    help="use the paper's axis-aligned squaring")
    ap.add_argument('--save-overlay', default=None,
                    help='write annotated frames here for eyeballing')
    args = ap.parse_args()

    import cv2

    from vision.snake_gate_detector import SnakeGateConfig, SnakeGateDetector

    d = args.directory
    if d is None:
        runs = sorted(glob.glob('frames/run_*'), key=os.path.getmtime)
        if not runs:
            print('no frames/run_* directories')
            return
        d = runs[-1]
    files = sorted(glob.glob(os.path.join(d, '*.jpg')))[: args.limit]
    if not files:
        print(f'no jpgs in {d}')
        return

    cfg = SnakeGateConfig(
        hsv_lower=_cfg.GATE_HSV_LOWER,
        hsv_upper=_cfg.GATE_HSV_UPPER,
        min_length_px=(args.min_length if args.min_length is not None
                       else _cfg.SNAKE_MIN_LENGTH_PX),
        min_color_fitness=(args.min_fitness if args.min_fitness is not None
                           else _cfg.SNAKE_MIN_COLOR_FITNESS),
        max_samples=_cfg.SNAKE_MAX_SAMPLES,
        use_rotated_rect=not args.axis_aligned,
    )
    det = SnakeGateDetector(cfg)
    print(f'== {d}  {len(files)} frames (each contains a YOLO-confirmed gate)')
    print(f'   HSV {cfg.hsv_lower}..{cfg.hsv_upper}  sigma_L={cfg.min_length_px}'
          f'  sigma_cf={cfg.min_color_fitness:.2f}  '
          f'rect={"axis" if args.axis_aligned else "rotated"}')

    if args.save_overlay:
        os.makedirs(args.save_overlay, exist_ok=True)

    hits = 0
    times = []
    masks = []
    fits = []
    for i, f in enumerate(files):
        img = cv2.imread(f)
        if img is None:
            continue
        res = det.detect(img)
        times.append(res.elapsed_ms)
        masks.append(res.mask_fraction)
        if res.found:
            hits += 1
            fits.append(res.best.color_fitness)
        if args.save_overlay and i < 40:
            out = img.copy()
            import numpy as np
            for g in res.gates:
                pts = np.asarray(g.corners_px, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(out, [pts], True, (250, 120, 250), 2)
            cv2.imwrite(
                os.path.join(args.save_overlay, os.path.basename(f)), out
            )

    n = len(times)
    if not n:
        print('no readable frames')
        return
    times.sort()
    masks.sort()
    print()
    print(f'  detected on {hits}/{n} ({100.0 * hits / n:.1f}%)')
    print(f'  cost: median {times[len(times) // 2]:.1f} ms  '
          f'p95 {times[int(0.95 * (n - 1))]:.1f} ms')
    print(f'  gate-coloured pixels: median '
          f'{100.0 * masks[len(masks) // 2]:.3f}% of frame  '
          f'max {100.0 * masks[-1]:.3f}%')
    if fits:
        fits.sort()
        print(f'  colour fitness on hits: median '
              f'{fits[len(fits) // 2]:.2f}  min {fits[0]:.2f}')
    if masks[len(masks) // 2] < 0.0005:
        print()
        print('  Almost no pixels match GATE_HSV_LOWER/UPPER. The colour window '
              'is calibrated for orange gates; retune it with '
              'tools/hsv_tuner.py before judging this method.')


if __name__ == '__main__':
    main()
