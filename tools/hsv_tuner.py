"""Interactive HSV tuner for the gate colour, with live snake detection.

Four panes, so the effect of a threshold change is visible at every stage:

    original            |  binary mask (exactly what the snake indexes)
    snake detections    |  colour-masked original

The snake pane is the point. A colour window can look clean as a mask and still
produce false gates, because the snake only needs a long enough run of colour in
two directions. Watching the detections themselves is the only way to tune
against false positives rather than against how tidy the mask looks.

    python tools/hsv_tuner.py                    # newest frames/run_* directory
    python tools/hsv_tuner.py frames/run_2026... # a specific run
    python tools/hsv_tuner.py path/to/frame.jpg  # a single image

Keys
    n / space   next frame          p   previous frame
    r           reset to config     a   toggle rotated / axis-aligned squaring
    s           print current values without quitting
    q / Esc     print and quit

Sliders start from ``config.GATE_HSV_LOWER`` / ``UPPER`` so a session begins
from what the aircraft is actually flying with.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as _cfg  # noqa: E402
from vision.snake_gate_detector import (  # noqa: E402
    SnakeGateConfig,
    SnakeGateDetector,
)

WIN = 'hsv tuner'
PANE_W, PANE_H = 640, 360
_SNAKE = (250, 120, 250)
_WHITE = (235, 235, 235)
_WARN = (90, 200, 250)
_BAD = (90, 90, 245)


def collect_frames(target: str | None) -> list[str]:
    if target and os.path.isfile(target):
        return [target]
    directory = target
    if directory is None:
        runs = sorted(glob.glob('frames/run_*'), key=os.path.getmtime)
        if not runs:
            return []
        directory = runs[-1]
    files = sorted(glob.glob(os.path.join(directory, '*.jpg')))
    files += sorted(glob.glob(os.path.join(directory, '*.png')))
    return files


def _label(img, lines, colour=_WHITE):
    y = 20
    for text, c in lines:
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    c or colour, 1, cv2.LINE_AA)
        y += 18


def compose(img, *, lo, hi, lo2, hi2, sigma_l, sigma_cf, rotated,
            caption='', max_samples=1500):
    """The four-pane grid. Shared by the live loop and ``--dump``."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lo, hi)
    if int(hi2[0]) > int(lo2[0]):
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo2, hi2))
    frac = float(mask.mean()) / 255.0

    det = SnakeGateDetector(SnakeGateConfig(
        hsv_lower=tuple(int(v) for v in lo),
        hsv_upper=tuple(int(v) for v in hi),
        min_length_px=int(sigma_l),
        min_color_fitness=float(sigma_cf),
        max_samples=int(max_samples),
        use_rotated_rect=bool(rotated),
    ))
    res = det.detect(img)

    binary = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    masked = cv2.bitwise_and(img, img, mask=mask)
    snake_view = img.copy()
    for gate in res.gates:
        pts = np.asarray(gate.corners_px, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(snake_view, [pts], True, _SNAKE, 2, cv2.LINE_AA)
        x, y = int(gate.corners_px[0][0]), int(gate.corners_px[0][1])
        cv2.putText(snake_view, f'{gate.color_fitness:.2f}',
                    (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    _SNAKE, 1, cv2.LINE_AA)

    n = len(res.gates)
    frac_colour = _BAD if frac > 0.15 else (_WARN if frac > 0.08 else None)
    gate_colour = _BAD if n > 2 else (_WARN if n == 2 else None)

    img_view = img.copy()
    _label(img_view, [
        (caption, None),
        (f'HSV {tuple(int(v) for v in lo)}..{tuple(int(v) for v in hi)}', None),
    ])
    _label(binary, [
        (f'mask {100.0 * frac:.2f}% of frame', frac_colour),
        ('above ~8% is usually too permissive', None),
    ])
    _label(snake_view, [
        (f'snake: {n} gate(s)   {res.elapsed_ms:.1f} ms', gate_colour),
        (f'sigma_L={int(sigma_l)}  sigma_cf={sigma_cf:.2f}  '
         f'{"rotated" if rotated else "axis"}', None),
        ('extra gates here = false positives', None),
    ])
    _label(masked, [('colour-masked original', None)])

    return np.vstack([
        np.hstack([img_view, binary]),
        np.hstack([snake_view, masked]),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('target', nargs='?', default=None,
                    help='image file, frames/run_* directory, or omit for newest')
    ap.add_argument('--scale', type=float, default=1.0,
                    help='window scale if the grid is too large for the screen')
    ap.add_argument('--min-length', type=int, default=None)
    ap.add_argument('--min-fitness', type=float, default=None)
    ap.add_argument('--dump', default=None,
                    help='render one frame with the config values to this PNG '
                         'and exit, for checking the layout without a display')
    args = ap.parse_args()

    files = collect_frames(args.target)
    if not files:
        print('no frames found. Pass an image or a frames/run_* directory.')
        return 1
    print(f'{len(files)} frame(s); n/p to step, q to quit')

    lo0 = tuple(int(v) for v in _cfg.GATE_HSV_LOWER)
    hi0 = tuple(int(v) for v in _cfg.GATE_HSV_UPPER)

    if args.dump:
        raw = cv2.imread(files[0])
        img0 = (raw if raw.shape[:2] == (PANE_H, PANE_W)
                else cv2.resize(raw, (PANE_W, PANE_H)))
        grid = compose(
            img0,
            lo=np.array(lo0, np.uint8), hi=np.array(hi0, np.uint8),
            lo2=np.array([0, 0, 0], np.uint8),
            hi2=np.array([0, 0, 0], np.uint8),
            sigma_l=args.min_length or _cfg.SNAKE_MIN_LENGTH_PX,
            sigma_cf=args.min_fitness or _cfg.SNAKE_MIN_COLOR_FITNESS,
            rotated=bool(_cfg.SNAKE_USE_ROTATED_RECT),
            caption=os.path.basename(files[0]),
            max_samples=_cfg.SNAKE_MAX_SAMPLES,
        )
        cv2.imwrite(args.dump, grid)
        print(f'wrote {args.dump}  ({grid.shape[1]}x{grid.shape[0]})')
        return 0

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    bars = [
        ('H lo', lo0[0], 179), ('H hi', hi0[0], 179),
        ('S lo', lo0[1], 255), ('S hi', hi0[1], 255),
        ('V lo', lo0[2], 255), ('V hi', hi0[2], 255),
        # Second hue band, for gates whose orange wraps past 179. Left empty by
        # default; config.GATE_HSV_* is a single band, so if this ends up doing
        # real work the config needs a second range rather than a wider first.
        ('H2 lo', 0, 179), ('H2 hi', 0, 179),
        ('sigma_L', args.min_length or _cfg.SNAKE_MIN_LENGTH_PX, 120),
        ('sigma_cf x100',
         int(100 * (args.min_fitness or _cfg.SNAKE_MIN_COLOR_FITNESS)), 100),
    ]
    for name, val, mx in bars:
        cv2.createTrackbar(name, WIN, val, mx, lambda _v: None)

    def reset():
        for name, val, _mx in bars:
            cv2.setTrackbarPos(name, WIN, val)

    idx = 0
    rotated = bool(_cfg.SNAKE_USE_ROTATED_RECT)
    cached_path = None
    img = None

    def emit(lo, hi, lo2, hi2, sl, cf):
        print()
        print('# paste into config.py')
        print(f'GATE_HSV_LOWER = {tuple(int(v) for v in lo)}')
        print(f'GATE_HSV_UPPER = {tuple(int(v) for v in hi)}')
        print(f'SNAKE_MIN_LENGTH_PX = {int(sl)}')
        print(f'SNAKE_MIN_COLOR_FITNESS = {cf:.2f}')
        if int(hi2[0]) > int(lo2[0]):
            print(f'# second hue band in use: {tuple(int(v) for v in lo2)}'
                  f' .. {tuple(int(v) for v in hi2)}')
            print('# config.GATE_HSV_* holds one band only — add a second range'
                  ' in vision_rx if this is needed.')

    while True:
        path = files[idx % len(files)]
        if path != cached_path:
            raw = cv2.imread(path)
            if raw is None:
                idx += 1
                continue
            img = (raw if raw.shape[:2] == (PANE_H, PANE_W)
                   else cv2.resize(raw, (PANE_W, PANE_H)))
            cached_path = path
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        def g(n):
            return cv2.getTrackbarPos(n, WIN)

        lo = np.array([g('H lo'), g('S lo'), g('V lo')], np.uint8)
        hi = np.array([g('H hi'), g('S hi'), g('V hi')], np.uint8)
        lo2 = np.array([g('H2 lo'), g('S lo'), g('V lo')], np.uint8)
        hi2 = np.array([g('H2 hi'), g('S hi'), g('V hi')], np.uint8)
        sigma_l = max(1, g('sigma_L'))
        sigma_cf = g('sigma_cf x100') / 100.0

        grid = compose(
            img, lo=lo, hi=hi, lo2=lo2, hi2=hi2,
            sigma_l=sigma_l, sigma_cf=sigma_cf, rotated=rotated,
            caption=(f'{os.path.basename(path)}  '
                     f'[{idx % len(files) + 1}/{len(files)}]'),
            max_samples=_cfg.SNAKE_MAX_SAMPLES,
        )
        if args.scale != 1.0:
            grid = cv2.resize(
                grid, None, fx=args.scale, fy=args.scale,
                interpolation=cv2.INTER_AREA,
            )
        cv2.imshow(WIN, grid)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27):
            emit(lo, hi, lo2, hi2, sigma_l, sigma_cf)
            break
        if key in (ord('n'), ord(' ')):
            idx += 1
        elif key == ord('p'):
            idx -= 1
        elif key == ord('r'):
            reset()
        elif key == ord('a'):
            rotated = not rotated
        elif key == ord('s'):
            emit(lo, hi, lo2, hi2, sigma_l, sigma_cf)

    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    sys.exit(main())
