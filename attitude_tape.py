"""Open-loop attitude tape capture / replay (desired lean + yaw rate + thrust)."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Optional


def load_attitude_tape(path) -> dict[str, Any]:
    p = Path(path)
    data = json.loads(p.read_text(encoding='utf-8'))
    if data.get('type') != 'attitude_tape':
        raise ValueError(f'{p}: expected type=attitude_tape')
    samples = data.get('samples') or []
    if not samples:
        raise ValueError(f'{p}: empty samples')
    return data


def save_attitude_tape(path, tape: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(tape)
    payload['type'] = 'attitude_tape'
    samples = payload.get('samples') or []
    if samples:
        payload['duration_s'] = float(samples[-1]['t'])
        payload['n'] = len(samples)
    tmp = str(p) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f)
    os.replace(tmp, p)
    return p


def trim_tape_until_gate(
    tape: dict[str, Any],
    gate: int,
    *,
    after_s: float = 0.35,
) -> dict[str, Any]:
    """Keep samples through gate ``gate`` (+ short pad)."""
    g = int(gate)
    passes = list(tape.get('gate_passes') or [])
    gate_t = None
    for gp in passes:
        if int(gp.get('gate', -1)) == g:
            gate_t = float(gp['t'])
            break
    if gate_t is None:
        raise ValueError(f'no gate {g} marker on attitude tape')
    cut = float(gate_t) + max(0.0, float(after_s))
    samples = [
        dict(s) for s in (tape.get('samples') or [])
        if float(s['t']) <= cut + 1e-9
    ]
    if not samples:
        raise ValueError(f'gate {g} tape empty after trim')
    # Hold last attitude for a beat so handoff is not a hard zero.
    last = dict(samples[-1])
    hold_t = round(cut + 0.05, 4)
    if float(last['t']) < hold_t - 1e-4:
        last['t'] = hold_t
        samples.append(last)
    out = {
        'type': 'attitude_tape',
        'name': f"{tape.get('name', 'tape')}_until_g{g}",
        'gate_passes': [
            dict(gp) for gp in passes if float(gp['t']) <= cut + 1e-9
        ],
        'samples': samples,
        'duration_s': float(samples[-1]['t']),
        'n': len(samples),
        'trimmed_through_gate': g,
    }
    if tape.get('practice'):
        out['practice'] = dict(tape['practice'])
    return out


class AttitudeTapeRecorder:
    """Capture continuous controller commands (+ optional raw pad axes)."""

    def __init__(self, *, name: str = 'practice_session'):
        self.name = name
        self.samples: list[dict[str, float]] = []
        self.gate_passes: list[dict[str, Any]] = []
        self._t0: Optional[float] = None
        self._last_gate = -1
        self._last_sample_t = -1.0

    def clear(self) -> None:
        self.samples.clear()
        self.gate_passes.clear()
        self._t0 = None
        self._last_gate = -1
        self._last_sample_t = -1.0

    def start(self, t_wall: Optional[float] = None) -> None:
        self.clear()
        self._t0 = float(t_wall if t_wall is not None else time.monotonic())

    def seed_from_tape(
        self, tape: dict[str, Any], t_wall_now: float
    ) -> int:
        """Load a prefix tape and continue recording from its end time."""
        samples = [dict(s) for s in (tape.get('samples') or [])]
        passes = [dict(gp) for gp in (tape.get('gate_passes') or [])]
        self.samples = samples
        self.gate_passes = passes
        self._last_gate = self.max_gate() if passes else -1
        if self._last_gate is None:
            self._last_gate = -1
        dur = float(samples[-1]['t']) if samples else 0.0
        self._last_sample_t = dur if samples else -1.0
        self._t0 = float(t_wall_now) - dur
        return len(samples)

    @property
    def started(self) -> bool:
        return self._t0 is not None

    def elapsed(self, t_wall: float) -> float:
        if self._t0 is None:
            self._t0 = float(t_wall)
        return max(0.0, float(t_wall) - float(self._t0))

    def sample(
        self,
        t_wall: float,
        *,
        des_roll: float,
        des_pitch: float,
        yaw_rate: float,
        thrust: float,
        pad_roll: float | None = None,
        pad_pitch: float | None = None,
        pad_yaw: float | None = None,
        pad_thrust: float | None = None,
        min_dt: float = 1.0 / 60.0,
        t: float | None = None,
    ) -> None:
        if t is None:
            t = round(self.elapsed(t_wall), 4)
        else:
            t = round(float(t), 4)
            if self._t0 is None:
                self._t0 = float(t_wall) - t
        if self.samples and (t - self._last_sample_t) < float(min_dt) - 1e-9:
            # Overwrite last sample in the same control tick bucket.
            row = self.samples[-1]
        else:
            row = {'t': t}
            self.samples.append(row)
            self._last_sample_t = t
        row['t'] = t
        row['des_roll'] = round(float(des_roll), 5)
        row['des_pitch'] = round(float(des_pitch), 5)
        row['yaw_rate'] = round(float(yaw_rate), 5)
        row['thrust'] = round(float(thrust), 5)
        if pad_roll is not None:
            row['pad_roll'] = round(float(pad_roll), 4)
        if pad_pitch is not None:
            row['pad_pitch'] = round(float(pad_pitch), 4)
        if pad_yaw is not None:
            row['pad_yaw'] = round(float(pad_yaw), 4)
        if pad_thrust is not None:
            row['pad_thrust'] = round(float(pad_thrust), 4)

    def watch_gate_pass(
        self,
        shared_data,
        t_wall: float,
        *,
        t: float | None = None,
    ) -> Optional[str]:
        race = shared_data.get('race_status') or {}
        ag = race.get('active_gate')
        latched = shared_data.get('last_gate_passed')
        try:
            candidates = []
            if ag is not None:
                candidates.append(int(ag))
            if latched is not None:
                candidates.append(int(latched))
        except (TypeError, ValueError):
            return None
        if not candidates:
            return None
        ag_i = max(candidates)
        if ag_i <= self._last_gate:
            return None
        self._last_gate = ag_i
        if ag_i < 1:
            return None
        if t is None:
            t = round(self.elapsed(t_wall), 4)
        else:
            t = round(float(t), 4)
        self.gate_passes.append({'t': t, 'gate': ag_i})
        return f'GATE {ag_i} @ t={t:.2f}s (attitude tape)'

    def max_gate(self) -> Optional[int]:
        if not self.gate_passes:
            return None
        return max(int(gp['gate']) for gp in self.gate_passes)

    def time_to_gate(self, gate: int) -> Optional[float]:
        g = int(gate)
        for gp in self.gate_passes:
            if int(gp['gate']) == g:
                return float(gp['t'])
        return None

    def to_tape(self) -> dict[str, Any]:
        samples = list(self.samples)
        return {
            'type': 'attitude_tape',
            'name': self.name,
            'gate_passes': [dict(gp) for gp in self.gate_passes],
            'samples': samples,
            'duration_s': float(samples[-1]['t']) if samples else 0.0,
            'n': len(samples),
        }

    def clone_through_gate(self, gate: int, *, after_s: float = 0.35) -> dict[str, Any]:
        return trim_tape_until_gate(self.to_tape(), gate, after_s=after_s)


class AttitudeTapeClock:
    """Linear-interpolate attitude samples by elapsed seconds since arm."""

    def __init__(self, tape: dict[str, Any]):
        self.tape = tape
        self.samples = list(tape['samples'])
        self.duration = float(tape.get('duration_s') or self.samples[-1]['t'])
        self.gate_passes = list(tape.get('gate_passes') or [])
        self._i = 0

    def reset(self) -> None:
        self._i = 0

    def gate_t(self, gate: int) -> Optional[float]:
        g = int(gate)
        for gp in self.gate_passes:
            if int(gp.get('gate', -1)) == g:
                return float(gp['t'])
        return None

    def sample_at(self, t: float) -> Optional[dict[str, float]]:
        t = float(t)
        if t < 0.0:
            t = 0.0
        if t > self.duration + 0.25:
            return None  # done
        while (
            self._i + 1 < len(self.samples)
            and float(self.samples[self._i + 1]['t']) <= t
        ):
            self._i += 1
        a = self.samples[self._i]
        if self._i + 1 >= len(self.samples):
            return {
                'des_roll': float(a['des_roll']),
                'des_pitch': float(a['des_pitch']),
                'yaw_rate': float(a['yaw_rate']),
                'thrust': float(a['thrust']),
            }
        b = self.samples[self._i + 1]
        ta, tb = float(a['t']), float(b['t'])
        if tb <= ta:
            u = 0.0
        else:
            u = max(0.0, min(1.0, (t - ta) / (tb - ta)))

        def lerp(ka):
            return (1.0 - u) * float(a[ka]) + u * float(b[ka])

        return {
            'des_roll': lerp('des_roll'),
            'des_pitch': lerp('des_pitch'),
            'yaw_rate': lerp('yaw_rate'),
            'thrust': lerp('thrust'),
        }
