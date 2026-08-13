"""Replay a telemetry file through a trained policy and compare to the human.

Answers the one question a flight cannot: is the model broken, or is it fine
offline and only failing in closed loop? If predictions track the human on data
the policy was trained near, the model works and the failure is distribution
shift. If they saturate here too, the model itself is wrong.

    python tools/probe_policy.py --weights models/policy_r2.pt \
        --telem logs/telem_....csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from race_obs import LABEL_NAMES  # noqa: E402
from tools.train_policy import build_windows, load_run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights', type=Path, nargs='+', required=True)
    ap.add_argument('--telem', type=Path, required=True)
    ap.add_argument('--history', type=int, default=32)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    import torch

    from policy_net import load_policy

    loaded = load_run(
        args.telem, lead_s=0.0, tail_s=0.0, sort_by_u=False,
        drop_collision_s=0.0, drop_policy_frames=False,
    )
    if loaded is None:
        print('empty telem')
        return 2
    built = build_windows([loaded], args.history)
    if built is None:
        print('no usable windows')
        return 2
    X, Y, _W, _G = built
    if args.limit:
        X, Y = X[: args.limit], Y[: args.limit]
    print(f'{args.telem.name}: {len(X)} windows\n')

    for w in args.weights:
        model, _blob = load_policy(w)
        model.eval()
        with torch.no_grad():
            P = model(torch.from_numpy(X)).numpy()
        print(f'=== {w.name}')
        for i, ch in enumerate(LABEL_NAMES):
            p, y = P[:, i], Y[:, i]
            corr = (
                float(np.corrcoef(p, y)[0, 1])
                if p.std() > 1e-9 and y.std() > 1e-9 else float('nan')
            )
            print(f'  {ch:11s} pred {p.mean():+.3f}±{p.std():.3f}   '
                  f'human {y.mean():+.3f}±{y.std():.3f}   corr={corr:+.3f}')
        sat_p = 100.0 * float(np.mean(np.abs(P[:, 1]) > 2.0))
        sat_y = 100.0 * float(np.mean(np.abs(Y[:, 1]) > 2.0))
        print(f'  |roll| > 2 rad/s: policy {sat_p:.1f}%  human {sat_y:.1f}%')
        print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
