"""A/B the HSV and YOLO-pose gate detectors against labelled ground truth.

Phase 3a of the HG-DAgger plan. VQ1 is described by the vendor as a "simple,
high-contrast, desaturated gate environment", so the classical HSV detector may
match or beat YOLO there while needing no weights, no GPU and no labelling --
and the Roboflow labels on hand are VQ2 imagery, so YOLO carries a real domain
risk on VQ1.

Two modes:

  labelled   point --data at a YOLO-pose dataset split. Scores corner error in
             pixels against the label file, which is the honest comparison.
  raw        point --frames at a directory of images with no labels. Reports
             detection rate and, for YOLO, mean keypoint confidence. Use this
             on fresh VQ1 captures before any relabelling exists.

    python tools/eval_detectors.py --data datasets/AIGP_8keypoints.v1i.yolov8 --split test
    python tools/eval_detectors.py --frames frames/vq1_run1
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Label layout: class cx cy w h then 8 * (kx ky visibility), all normalized.
KEYPOINT_COUNT = 8


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return float('nan')
    s = sorted(vals)
    return s[max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))]


def _summary(name: str, vals: list[float], unit: str = 'px') -> str:
    if not vals:
        return f'    {name:<22} n=0'
    return (f'    {name:<22} n={len(vals):<5} med={statistics.median(vals):6.2f}{unit} '
            f'p90={_pct(vals, 0.9):6.2f}{unit} max={max(vals):7.2f}{unit}')


def load_label(path: Path, width: int, height: int) -> np.ndarray | None:
    """Return (8,2) ground-truth keypoints in pixels, or None."""
    if not path.is_file():
        return None
    for line in path.read_text(encoding='utf-8').splitlines():
        fields = line.split()
        if len(fields) != 5 + KEYPOINT_COUNT * 3:
            continue
        vals = [float(v) for v in fields[1:]]
        kps = []
        for i in range(KEYPOINT_COUNT):
            kx, ky, vis = vals[4 + i * 3: 7 + i * 3]
            kps.append([kx * width, ky * height, vis])
        return np.asarray(kps, dtype=np.float64)
    return None


def corner_errors(pred: np.ndarray, truth: np.ndarray) -> list[float]:
    """Per-keypoint pixel error for visible truth points only."""
    out: list[float] = []
    for i in range(min(len(pred), len(truth))):
        if truth[i][2] <= 0:
            continue
        px, py = float(pred[i][0]), float(pred[i][1])
        if not (math.isfinite(px) and math.isfinite(py)):
            continue
        if px == 0.0 and py == 0.0:
            continue  # unseen keypoint convention
        out.append(float(math.hypot(px - truth[i][0], py - truth[i][1])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=None,
                    help='YOLO-pose dataset root containing data.yaml')
    ap.add_argument('--split', default='test', choices=('train', 'valid', 'test'))
    ap.add_argument('--frames', type=Path, default=None,
                    help='unlabelled image directory')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--weights', default=None,
                    help='override YOLO pose weights path')
    ap.add_argument('--skip-yolo', action='store_true')
    ap.add_argument('--skip-hsv', action='store_true')
    args = ap.parse_args()

    import cv2

    if args.data:
        img_dir = args.data / args.split / 'images'
        lbl_dir = args.data / args.split / 'labels'
        labelled = True
    elif args.frames:
        img_dir = args.frames
        lbl_dir = None
        labelled = False
    else:
        ap.error('pass --data or --frames')

    if not img_dir.is_dir():
        print(f'image dir missing: {img_dir}')
        return

    paths = sorted(
        [p for p in img_dir.iterdir()
         if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    )
    if args.limit > 0:
        paths = paths[: args.limit]
    print(f'{len(paths)} images from {img_dir}  (labelled={labelled})')

    # --- build detectors -------------------------------------------------
    hsv_det = None
    if not args.skip_hsv:
        from vision.gate_detector import GateVisionConfig, OrangeGateDetector
        import config as cfg
        hsv_det = OrangeGateDetector(GateVisionConfig(
            hsv_ranges=((cfg.GATE_HSV_LOWER, cfg.GATE_HSV_UPPER),),
            min_contour_area=cfg.GATE_MIN_CONTOUR_AREA,
        ))
        print('hsv detector ready')

    yolo_det = None
    if not args.skip_yolo:
        weights = Path(args.weights) if args.weights else ROOT / 'models' / 'gate_pose.pt'
        if not weights.is_file():
            print(f'yolo weights missing ({weights}) — skipping YOLO arm')
        else:
            from vision.yolo_pose_gate_detector import (
                PoseGateConfig, YoloPoseGateDetector,
            )
            yolo_det = YoloPoseGateDetector(PoseGateConfig(model_path=str(weights)))
            print(f'yolo detector ready ({weights.name})')

    # --- run -------------------------------------------------------------
    hsv_hits = yolo_hits = n = 0
    hsv_err: list[float] = []
    yolo_err: list[float] = []
    yolo_conf: list[float] = []
    hsv_center_err: list[float] = []
    yolo_center_err: list[float] = []

    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        n += 1
        h, w = image.shape[:2]
        truth = None
        if labelled and lbl_dir is not None:
            truth = load_label(lbl_dir / f'{path.stem}.txt', w, h)
        truth_center = None
        if truth is not None:
            vis = truth[truth[:, 2] > 0]
            if len(vis):
                truth_center = (float(vis[:, 0].mean()), float(vis[:, 1].mean()))

        if hsv_det is not None:
            det = hsv_det.detect(image)
            corners = getattr(det, 'corners', None)
            if corners is not None and float(getattr(det, 'confidence', 0.0)) > 0.0:
                hsv_hits += 1
                if truth_center is not None:
                    cx, cy = det.center_px
                    hsv_center_err.append(
                        math.hypot(cx - truth_center[0], cy - truth_center[1])
                    )
                # HSV yields a 4-corner quad, not the 8-keypoint model, so
                # compare it against the outer ring (label ids 0-3) only.
                if truth is not None:
                    quad = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
                    if len(quad) == 4:
                        outer = truth[:4]
                        pred4 = np.column_stack([quad, np.ones(4)])
                        hsv_err.extend(corner_errors(pred4, outer))

        if yolo_det is not None:
            det = yolo_det.detect(image)
            kps = getattr(det, 'keypoints', None)
            if kps is not None:
                yolo_hits += 1
                kps = np.asarray(kps, dtype=np.float64).reshape(-1, 2)
                confs = getattr(det, 'keypoint_confidences', None)
                if confs is not None:
                    yolo_conf.extend(
                        float(c) for c in np.asarray(confs).reshape(-1)
                        if math.isfinite(float(c))
                    )
                if truth_center is not None:
                    cx, cy = det.center_px
                    yolo_center_err.append(
                        math.hypot(cx - truth_center[0], cy - truth_center[1])
                    )
                if truth is not None and len(kps) >= KEYPOINT_COUNT:
                    pred = np.column_stack([kps[:KEYPOINT_COUNT], np.ones(KEYPOINT_COUNT)])
                    yolo_err.extend(corner_errors(pred, truth))

    # --- report ----------------------------------------------------------
    print(f'\nframes scored: {n}')
    print('\n=== detection rate ===')
    if hsv_det is not None:
        print(f'    hsv   {hsv_hits:>5}/{n}  ({100.0 * hsv_hits / max(n, 1):5.1f}%)')
    if yolo_det is not None:
        print(f'    yolo  {yolo_hits:>5}/{n}  ({100.0 * yolo_hits / max(n, 1):5.1f}%)')

    if labelled:
        print('\n=== corner error vs labels ===')
        print(_summary('hsv (outer ring)', hsv_err))
        print(_summary('yolo (8 keypoints)', yolo_err))
        print('\n=== centre error vs labels ===')
        print(_summary('hsv centre', hsv_center_err))
        print(_summary('yolo centre', yolo_center_err))
    if yolo_conf:
        print('\n=== yolo keypoint confidence ===')
        print(f'    med={statistics.median(yolo_conf):.3f}  '
              f'p10={_pct(yolo_conf, 0.1):.3f}')

    print('\n=== verdict ===')
    if labelled and hsv_err and yolo_err:
        h_med, y_med = statistics.median(hsv_err), statistics.median(yolo_err)
        winner = 'hsv' if h_med <= y_med else 'yolo'
        print(f'    lower median corner error: {winner} '
              f'(hsv {h_med:.2f}px vs yolo {y_med:.2f}px)')
        print('    NOTE: these labels are VQ2 imagery. Re-run with --frames on')
        print('    fresh VQ1 captures before trusting this for VQ1.')
    else:
        print('    no labelled comparison available; use detection rate above')


if __name__ == '__main__':
    main()
