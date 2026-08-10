"""Identify drag coefficients k_x, k_y from telemetry + LS pose velocity.

Regresses body accel against vision-derived body velocity over segments where
a gate is continuously visible:

    a_x ≈ k_x * v_x
    a_y ≈ k_y * v_y

    python tools/identify_drag.py --telem logs/telem_....csv

Writes suggested ``DRAG_KX`` / ``DRAG_KY`` values. Without telem (current
checkout), prints the config defaults and exits.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _num(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return out


def fit_k(accel: np.ndarray, vel: np.ndarray) -> float:
    """Least-squares k in a = k v, through origin."""
    mask = np.isfinite(accel) & np.isfinite(vel) & (np.abs(vel) > 0.2)
    if int(mask.sum()) < 20:
        return float('nan')
    v = vel[mask]
    a = accel[mask]
    return float(np.dot(v, a) / np.dot(v, v))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--telem', type=Path, default=None)
    args = ap.parse_args()

    if args.telem is None or not args.telem.is_file():
        print('No telem CSV available.')
        print('Defaults in config.py: DRAG_KX=-0.50  DRAG_KY=-0.50')
        print('Re-run after a flight that logs ax/ay and a continuous gate lock.')
        return

    rows = list(csv.DictReader(args.telem.open(newline='')))
    # Prefer explicit body velocity columns if present; otherwise difference
    # LS range along the approach as a rough forward-speed proxy.
    ax = np.array([_num(r.get('ax')) for r in rows])
    ay = np.array([_num(r.get('ay')) for r in rows])
    if 'vel_body_x' in rows[0]:
        vx = np.array([_num(r.get('vel_body_x')) for r in rows])
        vy = np.array([_num(r.get('vel_body_y')) for r in rows])
    elif 'gate_range' in rows[0] and 't' in rows[0]:
        t = np.array([_num(r.get('t')) for r in rows])
        rng = np.array([_num(r.get('gate_range')) for r in rows])
        vx = np.full_like(rng, np.nan)
        vy = np.zeros_like(rng)
        for i in range(1, len(rows)):
            dt = t[i] - t[i - 1]
            if dt > 1e-3 and math.isfinite(rng[i]) and math.isfinite(rng[i - 1]):
                # Closing on the gate → positive forward speed.
                vx[i] = -(rng[i] - rng[i - 1]) / dt
    else:
        print('telem lacks vel_body_* and gate_range — cannot fit')
        return

    kx = fit_k(ax, vx)
    ky = fit_k(ay, vy)
    print(f'samples={len(rows)}')
    print(f'DRAG_KX={kx:.4f}' if math.isfinite(kx) else 'DRAG_KX=unfit')
    print(f'DRAG_KY={ky:.4f}' if math.isfinite(ky) else 'DRAG_KY=unfit')
    print('Set via env or config.py, then re-fly.')


if __name__ == '__main__':
    main()
