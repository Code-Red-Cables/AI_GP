"""Score autonomous HG-DAgger runs: gates cleared and failure points.

Phase 7 of the plan. Fully autonomous evaluation only — point this at telem
from ``FLIGHT_MODE=policy`` / ``main.py`` runs that never took human input.
Reports success rate over a batch and which gate most often ends a run.

    python tools/score_policy.py --glob 'logs/telem_policy_*.csv'
    python tools/score_policy.py --telem logs/telem_A.csv logs/telem_B.csv
"""
from __future__ import annotations

import argparse
import csv
import glob as globlib
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _num(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return out if math.isfinite(out) else float('nan')


def score_telem(path: Path, *, finish_gate: int) -> dict:
    with path.open(newline='') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {
            'path': path, 'gates_cleared': 0, 'finished': False,
            'fail_gate': None, 'human_frames': 0, 'duration_s': 0.0,
        }

    gates = []
    human = 0
    finish_ns = False
    for row in rows:
        ag = _num(row.get('active_gate'))
        if math.isfinite(ag):
            gates.append(int(ag))
        if str(row.get('control_authority', 'policy')) == 'human':
            human += 1
        fns = _num(row.get('race_finish_ns'))
        if math.isfinite(fns) and fns > 0:
            finish_ns = True

    max_active = max(gates) if gates else 0
    # active_gate advances after a clear, so gates_cleared ≈ max_active.
    cleared = max(0, max_active)
    finished = finish_ns or cleared >= finish_gate
    fail_gate = None if finished else cleared
    t0 = _num(rows[0].get('t'))
    t1 = _num(rows[-1].get('t'))
    duration = (t1 - t0) if math.isfinite(t0) and math.isfinite(t1) else float('nan')
    return {
        'path': path,
        'gates_cleared': cleared,
        'finished': finished,
        'fail_gate': fail_gate,
        'human_frames': human,
        'duration_s': duration,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--telem', type=Path, nargs='*', default=[])
    ap.add_argument('--glob', default=None)
    ap.add_argument(
        '--finish-gate', type=int, default=17,
        help='active_gate index that counts as a finished lap',
    )
    ap.add_argument(
        '--allow-human', action='store_true',
        help='do not fail the batch when human frames appear (debug only)',
    )
    args = ap.parse_args()

    paths = list(args.telem)
    if args.glob:
        paths += [Path(p) for p in globlib.glob(args.glob)]
    paths = [p for p in dict.fromkeys(paths) if p.is_file()]
    if not paths:
        print('no telemetry files — pass --telem or --glob')
        return 2

    results = [score_telem(p, finish_gate=args.finish_gate) for p in paths]
    n = len(results)
    finishes = sum(1 for r in results if r['finished'])
    cleared = [r['gates_cleared'] for r in results]
    fail_counts = Counter(
        r['fail_gate'] for r in results if r['fail_gate'] is not None
    )
    human_runs = sum(1 for r in results if r['human_frames'] > 0)

    print(f'scored {n} run(s)  finish_gate>={args.finish_gate}')
    print(f'  success rate     {finishes}/{n}  ({100.0 * finishes / n:.1f}%)')
    print(f'  gates cleared    med={statistics.median(cleared):.1f}  '
          f'mean={statistics.mean(cleared):.2f}  max={max(cleared)}')
    if fail_counts:
        print('  failure gates (first uncleared index):')
        for gate, count in sorted(fail_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f'    gate {gate}: {count} run(s)')
    else:
        print('  failure gates    (none — every run finished)')

    print('\nper-run:')
    for r in results:
        flag = 'FINISH' if r['finished'] else f"FAIL@{r['fail_gate']}"
        human = f"  HUMAN_FRAMES={r['human_frames']}" if r['human_frames'] else ''
        print(
            f"  {r['path'].name}: {flag}  cleared={r['gates_cleared']}  "
            f"t={r['duration_s']:.1f}s{human}"
        )

    print('\n=== compliance ===')
    if human_runs and not args.allow_human:
        print(
            f'  FAIL — {human_runs} run(s) contain human frames. '
            'Spec §7: timed evaluation must be zero human input. '
            'Use coach only for training rounds.'
        )
        return 1
    if human_runs:
        print(f'  WARN — {human_runs} run(s) contain human frames (--allow-human)')
    else:
        print('  PASS — no human frames in the batch')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
