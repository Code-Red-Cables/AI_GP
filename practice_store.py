"""Practice checkpoints from continuous controller (pad) attitude tapes.

The sim cannot teleport mid-course, so "start from gate N" means:
replay the best saved attitude commands through gate N-1, then hand sticks.

Stores the actual commanded lean / yaw-rate / thrust (what the pad produced
after soft-gain mapping) — not digital WASD keys.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from attitude_tape import (
    AttitudeTapeRecorder,
    load_attitude_tape,
    save_attitude_tape,
    trim_tape_until_gate,
)

ROOT = Path(__file__).resolve().parent


def _practice_dir() -> Path:
    try:
        import config as _cfg
        raw = getattr(_cfg, 'PRACTICE_DIR', None)
    except Exception:
        raw = None
    if raw:
        p = Path(str(raw))
        return p if p.is_absolute() else ROOT / p
    env = os.environ.get('PRACTICE_DIR')
    if env:
        p = Path(env)
        return p if p.is_absolute() else ROOT / p
    return ROOT / 'practice'


PRACTICE_DIR = _practice_dir()
INDEX_PATH = PRACTICE_DIR / 'index.json'


def _ensure_dir() -> Path:
    PRACTICE_DIR.mkdir(parents=True, exist_ok=True)
    return PRACTICE_DIR


def through_path(gate: int) -> Path:
    return _ensure_dir() / f'through_gate_{int(gate)}.json'


def load_index() -> dict[str, Any]:
    if not INDEX_PATH.exists():
        return {'gates': {}}
    try:
        raw = json.loads(INDEX_PATH.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {'gates': {}}
    if not isinstance(raw, dict):
        return {'gates': {}}
    raw.setdefault('gates', {})
    return raw


def save_index(index: dict[str, Any]) -> None:
    _ensure_dir()
    tmp = str(INDEX_PATH) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2)
    os.replace(tmp, INDEX_PATH)


def _gate_t(tape: dict[str, Any], gate: int) -> Optional[float]:
    for gp in tape.get('gate_passes') or []:
        if int(gp.get('gate', -1)) == int(gate):
            return float(gp['t'])
    return None


def time_to_gate(tape: dict[str, Any], gate: int) -> Optional[float]:
    return _gate_t(tape, gate)


def split_time(
    tape: dict[str, Any], from_gate: int, to_gate: int
) -> Optional[float]:
    t1 = time_to_gate(tape, int(to_gate))
    if t1 is None:
        return None
    if int(from_gate) <= 0:
        return t1
    t0 = time_to_gate(tape, int(from_gate))
    if t0 is None:
        return None
    return max(0.0, t1 - t0)


def maybe_update_through_gate(
    recorder_or_tape: AttitudeTapeRecorder | dict[str, Any],
    gate: int,
    *,
    source: str = 'live',
    force: bool = False,
    collision: bool = False,
) -> Optional[str]:
    """If this through-gate attitude tape is a new best (or ``force``), save it."""
    g = int(gate)
    if g < 1:
        return None
    if collision and not force:
        return f'skip G{g}: collision'
    if isinstance(recorder_or_tape, AttitudeTapeRecorder):
        tape = recorder_or_tape.to_tape()
    else:
        tape = recorder_or_tape
    t_gate = time_to_gate(tape, g)
    if t_gate is None:
        return f'skip G{g}: no gate tag'
    for prev in range(1, g):
        if time_to_gate(tape, prev) is None:
            return f'skip G{g}: missing earlier GATE {prev} tag'
    if not (tape.get('samples') or []):
        return f'skip G{g}: no attitude samples'

    index = load_index()
    gates = index.setdefault('gates', {})
    key = str(g)
    prev_best = gates.get(key) or {}
    prev_t = prev_best.get('time_s')
    if (
        not force
        and prev_t is not None
        and float(t_gate) >= float(prev_t) - 1e-4
    ):
        return (
            f'skip G{g}: {t_gate:.3f}s not faster than best '
            f'{float(prev_t):.3f}s'
        )

    try:
        out = trim_tape_until_gate(tape, g, after_s=0.35)
    except ValueError as exc:
        return f'skip G{g}: {exc}'

    stamp = time.strftime('%Y%m%d_%H%M%S')
    out['name'] = f'through_gate_{g}'
    out['frozen_through_gate'] = g
    out['practice'] = {
        'through_gate': g,
        'time_s': round(float(t_gate), 4),
        'source': source,
        'saved_at': stamp,
        'kind': 'attitude_tape',
    }
    path = through_path(g)
    save_attitude_tape(path, out)

    archive = (
        _ensure_dir() / 'history' / f'through_g{g}_{stamp}_{t_gate:.3f}s.json'
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(path, archive)
    except OSError:
        pass

    gates[key] = {
        'through_gate': g,
        'time_s': round(float(t_gate), 4),
        'path': str(path),
        'source': source,
        'saved_at': stamp,
        'n_samples': int(out.get('n') or 0),
        'kind': 'attitude_tape',
    }
    if g >= 2:
        sp = split_time(tape, g - 1, g)
        if sp is not None:
            gates[key]['split_from_prev_s'] = round(float(sp), 4)
    save_index(index)
    verb = 'NEW BEST' if prev_t is not None else 'SAVED'
    prev_txt = (
        f' (was {float(prev_t):.3f}s)' if prev_t is not None else ''
    )
    return (
        f'{verb} practice through GATE {g}: {t_gate:.3f}s{prev_txt} '
        f'-> {path} ({out.get("n", 0)} attitude samples)'
    )


def sync_all_gates(
    recorder_or_tape: AttitudeTapeRecorder | dict[str, Any],
    *,
    source: str = 'keep',
    force: bool = False,
    collision: bool = False,
) -> list[str]:
    if isinstance(recorder_or_tape, AttitudeTapeRecorder):
        gmax = recorder_or_tape.max_gate()
        tape = recorder_or_tape
    else:
        passes = recorder_or_tape.get('gate_passes') or []
        gmax = max((int(gp['gate']) for gp in passes), default=None)
        tape = recorder_or_tape
    msgs = []
    if gmax is None:
        return msgs
    for g in range(1, int(gmax) + 1):
        msg = maybe_update_through_gate(
            tape, g, source=source, force=force, collision=collision,
        )
        if msg:
            msgs.append(msg)
    return msgs


def best_through(gate: int) -> Optional[Path]:
    g = int(gate)
    path = through_path(g)
    if path.exists():
        return path
    index = load_index()
    meta = (index.get('gates') or {}).get(str(g)) or {}
    alt = meta.get('path')
    if alt and Path(alt).exists():
        return Path(alt)
    return None


def longest_tape_covering(gate: int) -> Optional[Path]:
    """Prefer the farthest saved through-gate tape that still contains ``gate``.

    So ``--practice-from-gate 4`` and ``--practice-from-gate 5`` share the
    exact same early samples (from through_gate_5.json trimmed to 4 vs 5),
    instead of diverging because through_gate_4.json was frozen from an
    earlier save.
    """
    g = int(gate)
    index = load_index()
    gates = index.get('gates') or {}
    best_g = None
    best_path = None
    for key, meta in gates.items():
        try:
            gk = int(key)
        except (TypeError, ValueError):
            continue
        if gk < g:
            continue
        path = Path(str(meta.get('path') or through_path(gk)))
        if not path.exists():
            path = through_path(gk)
        if not path.exists():
            continue
        if best_g is None or gk > best_g:
            best_g = gk
            best_path = path
    if best_path is not None:
        return best_path
    return best_through(g)


def load_through_gate(gate: int) -> Optional[dict[str, Any]]:
    """Load attitude tape through ``gate``, sourced from the longest cover."""
    g = int(gate)
    path = longest_tape_covering(g)
    if path is None:
        return None
    try:
        tape = load_attitude_tape(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    # Already exactly this gate (and no farther samples) — use as-is.
    src_g = int(
        (tape.get('practice') or {}).get('through_gate')
        or tape.get('trimmed_through_gate')
        or tape.get('frozen_through_gate')
        or 0
    )
    if src_g == g or path == through_path(g):
        # Still trim if a longer tape was chosen.
        if src_g > g or (
            time_to_gate(tape, g) is not None
            and any(
                int(gp.get('gate', -1)) > g
                for gp in (tape.get('gate_passes') or [])
            )
        ):
            try:
                tape = trim_tape_until_gate(tape, g, after_s=0.35)
            except ValueError:
                return None
        tape['_source_path'] = str(path)
        return tape
    try:
        tape = trim_tape_until_gate(tape, g, after_s=0.35)
    except ValueError:
        return None
    tape['_source_path'] = str(path)
    return tape


def format_list() -> str:
    index = load_index()
    gates = index.get('gates') or {}
    if not gates:
        return (
            'No practice checkpoints yet.\n'
            'Fly pilot with the pad — faster through-gate attitude tapes '
            'auto-save under practice/.\n'
            'Then:  python tools/tune_flight.py pilot --practice-from-gate N'
        )
    lines = [
        'Practice checkpoints (best attitude tape to clear each gate):',
        '',
        f'{"Gate":>6}  {"time":>8}  {"split":>8}  samples  path',
    ]
    for key in sorted(gates, key=lambda k: int(k)):
        meta = gates[key]
        t = meta.get('time_s')
        sp = meta.get('split_from_prev_s')
        t_s = f'{float(t):.3f}s' if t is not None else '—'
        sp_s = f'{float(sp):.3f}s' if sp is not None else '—'
        n = meta.get('n_samples', '—')
        lines.append(
            f'  g{int(key):<4}  {t_s:>8}  {sp_s:>8}  {n!s:>7}  '
            f'{meta.get("path", "")}'
        )
    lines.append('')
    lines.append(
        'Replay attitude through gate N, then you fly N+1+:'
    )
    lines.append(
        '  python tools/tune_flight.py pilot --practice-from-gate N'
    )
    lines.append(
        '  Uses the longest saved tape (trimmed to N) so from-4 and '
        'from-5 share the same early stick inputs.'
    )
    return '\n'.join(lines)
