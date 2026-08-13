"""Does the policy commit where the human commits?

For each frame of a human lap, run the checkpoint on the same observation and
compare its command against what the human actually did. The headline number is
the *gain on decisive frames*: among frames where the human pushed hard, the
ratio of the policy's magnitude to the human's. A gain near 1.0 means the policy
commits as hard as the pilot; a gain near 0 means it coasts through the moments
that matter, which no amount of extra training on the same loss will fix.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as _cfg  # noqa: E402
from race_obs import (  # noqa: E402
    LABEL_NAMES,
    attitude_is_trusted,
    decode_bin_probs,
    labels_from_row,
    observation_from_row,
    stack_history,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('weights')
    ap.add_argument('--glob', default='logs/telem_2026081[01]*.csv')
    ap.add_argument('--min-rows', type=int, default=1200)
    ap.add_argument('--max-runs', type=int, default=6)
    ap.add_argument('--decisive', type=float, default=1.0,
                    help='|human command| above this counts as decisive')
    ap.add_argument('--window', type=int, default=None)
    args = ap.parse_args()

    import torch

    from policy_net import load_policy

    model, blob = load_policy(args.weights)
    model.eval()
    arch = blob['arch']
    history = int(arch.get('history', 32))
    bins = int(arch.get('bins', 0))
    ctx = bool(blob.get('context', False))

    paths = [p for p in sorted(glob.glob(args.glob))
             if sum(1 for _ in open(p)) - 1 >= args.min_rows]
    paths = paths[:args.max_runs]
    print(f'weights={os.path.basename(args.weights)} history={history} '
          f'bins={bins} context={ctx}')
    print(f'runs={len(paths)}')

    preds, acts = [], []
    for p in paths:
        with open(p, newline='') as fh:
            rows = list(csv.DictReader(fh))
        buf = []
        for r in rows:
            buf.append(observation_from_row(r, with_context=ctx))
            if len(buf) > history:
                buf.pop(0)
            if r.get('cmd_thrust') in (None, '') or not attitude_is_trusted(r):
                continue
            win = stack_history(buf, history)
            x = torch.tensor([win], dtype=torch.float32)
            with torch.no_grad():
                out = model(x).detach().cpu().numpy()[0]
            if bins:
                logits = out[0] if out.ndim == 3 else out
                pr = np.exp(logits - logits.max(axis=-1, keepdims=True))
                pr /= np.maximum(1e-12, pr.sum(axis=-1, keepdims=True))
                y = decode_bin_probs(pr, bins, window=args.window)
            else:
                y = list(out[0] if out.ndim == 2 else out)
            preds.append(y)
            acts.append(labels_from_row(r))

    P = np.asarray(preds, dtype=np.float64)
    A = np.asarray(acts, dtype=np.float64)
    print(f'frames={len(P)}')
    print()
    hdr = (f'{"channel":>11} {"corr":>6} {"pred_rms":>9} {"human_rms":>9} '
           f'{"decisive":>9} {"gain":>6}')
    print(hdr)
    print('-' * len(hdr))
    for c, name in enumerate(LABEL_NAMES):
        p, a = P[:, c], A[:, c]
        ok = np.isfinite(p) & np.isfinite(a)
        p, a = p[ok], a[ok]
        if len(p) < 10:
            continue
        corr = (float(np.corrcoef(p, a)[0, 1])
                if p.std() > 1e-9 and a.std() > 1e-9 else float('nan'))
        thresh = args.decisive
        if name == 'thrust':
            thresh = float(getattr(_cfg, 'HOVER_THRUST', 0.375)) + 0.15
            dec = a > thresh
        else:
            dec = np.abs(a) > thresh
        n_dec = int(dec.sum())
        if n_dec:
            gain = float(np.mean(np.abs(p[dec])) / max(1e-9,
                                                       np.mean(np.abs(a[dec]))))
        else:
            gain = float('nan')
        print(f'{name:>11} {corr:6.3f} {np.sqrt((p**2).mean()):9.3f} '
              f'{np.sqrt((a**2).mean()):9.3f} {n_dec:9d} {gain:6.2f}')

    print()
    print('gain 1.0 = matches the pilot on decisive frames; '
          'gain 0.1 = coasts through them')


if __name__ == '__main__':
    main()
