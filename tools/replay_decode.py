"""Replay a checkpoint over a logged run and compare bin-decode strategies.

Open loop: the observations come from the log, so this does not show what the
aircraft would have done. It shows what the *commands* would have been for
identical inputs, which is exactly what is needed to tell a decode problem
apart from a training problem.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from race_obs import (  # noqa: E402
    LABEL_NAMES,
    decode_bin_probs,
    feature_dim,
    observation_from_row,
    stack_history,
)


def load_model(path):
    import torch

    from policy_net import load_policy

    model, blob = load_policy(path)
    model.eval()
    return model, blob, blob['arch'], torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('weights')
    ap.add_argument('csv_path', nargs='?', default=None)
    ap.add_argument('--windows', default='0,1,2,3',
                    help='decode windows to compare (0 = old argmax)')
    args = ap.parse_args()

    path = args.csv_path
    if path is None:
        files = sorted(glob.glob('logs/telem_*.csv'), key=os.path.getmtime)
        path = files[-1]

    model, blob, arch, torch = load_model(args.weights)
    history = int(arch.get('history', 32))
    bins = int(arch.get('bins', 0))
    if not bins:
        print('checkpoint has no categorical head; nothing to compare')
        return
    ctx = bool(blob.get('context', False))
    n_feat = feature_dim(with_context=ctx)

    with open(path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    print(f'== {os.path.basename(path)}  {len(rows)} rows')
    print(f'   weights={os.path.basename(args.weights)} history={history} '
          f'bins={bins} context={ctx}')

    obs_rows = [observation_from_row(r, with_context=ctx) for r in rows]

    # Forward every frame once; decode the same probabilities several ways.
    probs_seq = []
    buf = []
    for o in obs_rows:
        buf.append(o)
        if len(buf) > history:
            buf.pop(0)
        win = stack_history(buf, history)
        x = torch.tensor([win], dtype=torch.float32)
        with torch.no_grad():
            out = model(x).detach().cpu().numpy()[0]
        logits = out[0] if out.ndim == 3 else out       # first chunk step
        p = np.exp(logits - logits.max(axis=-1, keepdims=True))
        p /= np.maximum(1e-12, p.sum(axis=-1, keepdims=True))
        probs_seq.append(p)

    conf = np.array([p.max(axis=-1) for p in probs_seq])   # (T, LABEL_DIM)
    print()
    print('mean top-bin probability per channel '
          '(1.0 = fully certain, 0.05 = uniform over 21):')
    for c, name in enumerate(LABEL_NAMES):
        print(f'  {name:11} {conf[:, c].mean():.3f}')

    hdr = (f'{"window":>7} {"channel":>11} {"mean":>8} {"min":>8} {"max":>8} '
           f'{"zero%":>6} {"sat%":>6}')
    print()
    print(hdr)
    print('-' * len(hdr))
    for w in [int(x) for x in args.windows.split(',')]:
        dec = np.array([decode_bin_probs(p, bins, window=w) for p in probs_seq])
        for c, name in enumerate(LABEL_NAMES):
            col = dec[:, c]
            zero = 100.0 * float(np.mean(np.abs(col) < 1e-6))
            sat = 100.0 * float(np.mean(np.abs(col) > 2.9)) if name != 'thrust' \
                else 100.0 * float(np.mean(col > 0.65))
            print(f'{w:7d} {name:>11} {col.mean():8.3f} {col.min():8.3f} '
                  f'{col.max():8.3f} {zero:6.1f} {sat:6.1f}')
        print()


if __name__ == '__main__':
    main()
