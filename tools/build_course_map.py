"""Build a per-gate course map for the race planner's feed-forward arcs.

Derives turn radius and heading change between consecutive gates from a
reference human/pilot CSV (``pilot_*.csv`` / ``telem_*.csv``) when available.
Without telemetry, writes a stub map the planner can still load — edit the
``turn_deg`` / ``turn_radius_m`` fields by hand from a flown lap.

    python tools/build_course_map.py [--telem logs/best/pilot_best.csv]
                                     [--out course_map.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _num(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return out


def from_telem(path: Path) -> dict[int, dict]:
    """Estimate per-gate turn from yaw change between consecutive GATE passes.

    Uses ``active_gate`` column when present; otherwise returns empty.
    """
    rows = list(csv.DictReader(path.open(newline='')))
    if not rows or 'active_gate' not in rows[0]:
        return {}

    by_gate: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        try:
            gid = int(float(row['active_gate']))
        except (TypeError, ValueError):
            continue
        by_gate[gid].append(row)

    out: dict[int, dict] = {}
    gates = sorted(by_gate)
    for gid in gates:
        chunk = by_gate[gid]
        yaws = [_num(r.get('yaw')) for r in chunk]
        yaws = [y for y in yaws if math.isfinite(y)]
        if len(yaws) < 2:
            continue
        # Net heading change while this gate was active.
        turn = math.atan2(math.sin(yaws[-1] - yaws[0]), math.cos(yaws[-1] - yaws[0]))
        # Rough radius from average forward speed / yaw rate.
        ax = [_num(r.get('ax')) for r in chunk]
        ax = [a for a in ax if math.isfinite(a)]
        # Without fitted k_x, leave radius at the planner default.
        out[gid] = {
            'id': gid,
            'turn_deg': round(math.degrees(turn), 1),
            'turn_radius_m': 1.5,
            'n_samples': len(chunk),
        }
    return out


def stub_map(n_gates: int = 17) -> dict[int, dict]:
    """Default map: mild right-hand turns, 1.5 m radius (paper default)."""
    return {
        i: {
            'id': i,
            'turn_deg': 90.0,
            'turn_radius_m': 1.5,
            'note': 'stub — replace after a scored lap',
        }
        for i in range(1, n_gates + 1)
    }


def from_probe(path: Path) -> dict[int, dict]:
    """Load gate NED positions from ``tools/probe_vq1.py`` JSON output."""
    raw = json.loads(path.read_text(encoding='utf-8'))
    out: dict[int, dict] = {}
    for g in raw.get('track_gates') or []:
        try:
            gid = int(g.get('gate_id', g.get('id')))
            pos = g.get('position_ned') or g.get('pos')
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        out[gid] = {
            'id': gid,
            'pos': [x, y, z],
            'width': g.get('width'),
            'height': g.get('height'),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--telem', type=Path, default=None)
    ap.add_argument(
        '--probe', type=Path, default=None,
        help='artifacts/vq1_probe.json from tools/probe_vq1.py (preferred: '
             'live TRACK_DATA positions)',
    )
    ap.add_argument('--out', type=Path, default=ROOT / 'course_map.json')
    ap.add_argument('--gates', type=int, default=17)
    args = ap.parse_args()

    gates: dict[int, dict] = {}
    source = 'stub'
    if args.probe and args.probe.is_file():
        gates = from_probe(args.probe)
        source = str(args.probe)
    if args.telem and args.telem.is_file():
        telem_gates = from_telem(args.telem)
        if telem_gates:
            # Merge turn estimates onto any positions already loaded.
            for gid, meta in telem_gates.items():
                gates.setdefault(gid, {'id': gid}).update(meta)
            source = f'{source}+{args.telem}' if gates else str(args.telem)
    if not gates:
        gates = stub_map(args.gates)
        source = 'stub'

    payload = {
        'source': source,
        'gates': [gates[k] for k in sorted(gates)],
    }
    args.out.write_text(json.dumps(payload, indent=2))
    n_pos = sum(1 for g in gates.values() if g.get('pos'))
    print(
        f'wrote {args.out} ({len(gates)} gates, {n_pos} with pos, '
        f'source={source})'
    )


if __name__ == '__main__':
    main()
