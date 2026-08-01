"""Remember / replay as a keyboard timeline.

Records which flight keys were held and for how long, then replays those
exact presses through the same hold-to-fly mapping as manual mode.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

# Keys that affect flight (not T/H/K/M/quit).
FLIGHT_KEYS = frozenset({'w', 's', 'a', 'd', 'q', 'e', 'r', 'f', 'up', 'down', 'left', 'right'})


def _f(value, default=None):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


class KeyTimeline:
    """Ordered key down/up events (+ optional gate markers)."""

    def __init__(self, path, *, name: str = 'remembered'):
        self.path = Path(path)
        self.name = name
        self.events: list[dict] = []
        self.ekf_use_pnp = None
        self.frozen_through_gate = 0
        self._last_active_gate = None
        self.saved_once = False
        self._t_offset = 0.0
        self._held: set[str] = set()

    def __len__(self):
        return len(self.events)

    def clear(self) -> None:
        self.events.clear()
        self._last_active_gate = None
        self._t_offset = 0.0
        self._held.clear()

    @property
    def duration(self) -> float:
        if not self.events:
            return 0.0
        return max(float(e['t']) for e in self.events) - min(
            float(e['t']) for e in self.events
        )

    def seed_prefix(self, other: 'KeyTimeline') -> int:
        n0 = len(self.events)
        for e in other.events:
            self.events.append(dict(e))
            ag = e.get('active_gate')
            if ag is not None:
                prev = self._last_active_gate
                self._last_active_gate = (
                    int(ag) if prev is None else max(prev, int(ag))
                )
        if self.events:
            self._t_offset = max(float(e['t']) for e in self.events)
        if other.ekf_use_pnp is not None:
            self.ekf_use_pnp = other.ekf_use_pnp
        # Resume held-key state from prefix end.
        self._held = other.keys_held_at(self._t_offset + 1e-6)
        return len(self.events) - n0

    def clip_at(self, t: float) -> None:
        """Drop events after ``t`` so human keys append from this timeline time."""
        t = float(t)
        self._sort_events()
        self.events = [
            dict(e) for e in self.events if float(e['t']) <= t + 1e-9
        ]
        self._held = self.keys_held_at(t + 1e-6)
        self._t_offset = t
        if self.events:
            gates = [
                int(e['active_gate'])
                for e in self.events
                if e.get('active_gate') is not None
            ]
            self._last_active_gate = max(gates) if gates else self._last_active_gate

    def gate_pass_t(self, gate: int) -> float | None:
        g = int(gate)
        for e in self.events:
            if (
                str(e.get('event') or '') == 'gate_pass'
                and int(e.get('active_gate', -1)) == g
            ):
                return float(e['t'])
        return None

    def post_gate_yaw_done_t(self, gate: int) -> float | None:
        """Latest timeline time for synthetic post-gate yaw on ``gate``."""
        g = int(gate)
        done = None
        for e in self.events:
            try:
                ag = int(e.get('active_gate', -1))
            except (TypeError, ValueError):
                continue
            if ag != g:
                continue
            if str(e.get('event') or '') == 'post_gate_yaw':
                done = float(e['t']) if done is None else max(done, float(e['t']))
            if e.get('key') and e.get('synthetic') and not e.get('down'):
                done = float(e['t']) if done is None else max(done, float(e['t']))
        return done

    def note_keys(self, t_rel: float, held: set[str]) -> list[dict]:
        """Diff ``held`` vs last set; append down/up events. Returns new events."""
        t = round(float(self._t_offset + t_rel), 4)
        flight = {k for k in held if k in FLIGHT_KEYS}
        new_ev = []
        for k in sorted(flight - self._held):
            ev = {'t': t, 'key': k, 'down': True}
            self.events.append(ev)
            new_ev.append(ev)
        for k in sorted(self._held - flight):
            ev = {'t': t, 'key': k, 'down': False}
            self.events.append(ev)
            new_ev.append(ev)
        self._held = set(flight)
        return new_ev

    def watch_gate_pass(self, shared_data, t_rel: float):
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
        if self._last_active_gate is None:
            self._last_active_gate = -1
        prev = int(self._last_active_gate)
        if ag_i <= prev:
            return None
        self._last_active_gate = ag_i
        if ag_i < 1:
            return None
        t = round(float(self._t_offset + t_rel), 4)
        ev = {
            't': t,
            'event': 'gate_pass',
            'active_gate': ag_i,
            'name': f'gate{ag_i}',
        }
        self.events.append(ev)
        return f'GATE {ag_i} @ t={t:.2f}s (key timeline)'

    def max_gate(self) -> int | None:
        gates = [
            int(e['active_gate'])
            for e in self.events
            if e.get('active_gate') is not None
        ]
        return max(gates) if gates else None

    def keys_held_at(self, t: float) -> set[str]:
        held: set[str] = set()
        for e in self.events:
            if float(e['t']) > t:
                break
            key = e.get('key')
            if not key:
                continue
            if e.get('down'):
                held.add(str(key))
            else:
                held.discard(str(key))
        return held

    def _sort_events(self) -> None:
        # downs before ups at the same timestamp so chords apply cleanly.
        self.events.sort(
            key=lambda e: (
                float(e['t']),
                0 if (e.get('key') and e.get('down')) else 1,
                0 if e.get('event') else 1,
            )
        )

    def ensure_post_gate_yaw_key(
        self,
        gate: int = 1,
        *,
        key: str = 'e',
        yaw_deg: float = 20.0,
        yaw_rate_deg: float = 35.0,
        lead_s: float | None = None,
    ) -> bool:
        """Insert a synthetic key hold around a gate_pass (as if you pressed it).

        Post-gate-1 yaw-right: ``E`` for ``yaw_deg / yaw_rate_deg`` seconds,
        starting ``lead_s`` *before* the GATE 1 tag (default 0.45s early).
        Replaces any prior synthetic post-gate yaw for that gate.
        """
        import config as _cfg

        g = int(gate)
        key = str(key).lower()
        if lead_s is None:
            lead_s = float(
                getattr(_cfg, 'PILOT_POST_G1_YAW_LEAD_S', 0.45) or 0.45
            )
        # lead_s > 0: start before gate_pass; lead_s < 0: start |lead| after.
        lead_s = float(lead_s)

        gate_t = None
        for e in self.events:
            if (
                str(e.get('event') or '') == 'gate_pass'
                and int(e.get('active_gate', -1)) == g
            ):
                gate_t = float(e['t'])
                break
        if gate_t is None:
            return False

        # Drop previous synthetic injection for this gate only.
        self.events = [
            e for e in self.events
            if not (
                e.get('synthetic')
                and int(e.get('active_gate', -1)) == g
            )
        ]

        dur = float(yaw_deg) / max(1.0, float(yaw_rate_deg))
        t0 = round(max(0.0, gate_t - lead_s), 4)
        t1 = round(t0 + dur, 4)
        # If a real (non-synthetic) hold already covers the window, mark only.
        mid = t0 + 0.5 * dur
        if key in self.keys_held_at(t0) and key in self.keys_held_at(mid):
            self.events.append({
                't': t1,
                'event': 'post_gate_yaw',
                'active_gate': g,
                'key': key,
                'yaw_deg': float(yaw_deg),
                'lead_s': lead_s,
                'synthetic': False,
            })
            self._sort_events()
            return False
        if key in self.keys_held_at(max(0.0, t0 - 1e-4)):
            self.events.append({
                't': round(max(0.0, t0 - 0.001), 4),
                'key': key,
                'down': False,
                'synthetic': True,
                'active_gate': g,
            })
        self.events.append({
            't': t0, 'key': key, 'down': True,
            'synthetic': True, 'active_gate': g,
        })
        self.events.append({
            't': t1, 'key': key, 'down': False,
            'synthetic': True, 'active_gate': g,
        })
        self.events.append({
            't': t1,
            'event': 'post_gate_yaw',
            'active_gate': g,
            'key': key,
            'yaw_deg': float(yaw_deg),
            'lead_s': lead_s,
            'synthetic': True,
        })
        self._sort_events()
        self._held = self.keys_held_at(t1 + 1e-6)
        return True

    def ensure_pilot_gate_yaws(self, yaw_rate_deg: float | None = None) -> list[str]:
        """Apply configured post-gate yaw keys (g1→g2 right, g2→g3 left).

        If ``frozen_through_gate`` is set on the timeline, do not rewrite any
        yaw for gates at or before that freeze — a G2 miss after editing those
        means we broke the locked path.
        """
        import config as _cfg

        rate = float(
            yaw_rate_deg
            if yaw_rate_deg is not None
            else getattr(_cfg, 'PILOT_POST_G1_YAW_RATE_DEG', 35.0) or 35.0
        )
        frozen = int(getattr(self, 'frozen_through_gate', 0) or 0)
        msgs = []
        specs = (
            (
                1,
                str(getattr(_cfg, 'PILOT_POST_G1_YAW_KEY', 'e') or 'e'),
                float(getattr(_cfg, 'PILOT_POST_G1_YAW_DEG', 20.0) or 20.0),
                float(getattr(_cfg, 'PILOT_POST_G1_YAW_LEAD_S', 0.45) or 0.45),
                'right',
            ),
            (
                2,
                str(getattr(_cfg, 'PILOT_POST_G2_YAW_KEY', 'q') or 'q'),
                float(getattr(_cfg, 'PILOT_POST_G2_YAW_DEG', 25.0) or 25.0),
                float(getattr(_cfg, 'PILOT_POST_G2_YAW_LEAD_S', -0.15)),
                'left',
            ),
        )
        for gate, key, yaw_deg, lead_s, label in specs:
            # Freeze through gate N: never rewrite approach keys at/before N.
            # If post-gate yaw for that gate is already baked (lock/rebuild),
            # leave it alone so config cannot undo a nudge or break G2.
            if frozen and gate < frozen:
                continue
            if frozen and gate == frozen and float(lead_s) >= 0:
                continue
            if frozen and gate <= frozen:
                already = any(
                    str(e.get('event') or '') == 'post_gate_yaw'
                    and int(e.get('active_gate', -1)) == gate
                    for e in self.events
                )
                if already:
                    continue
            if self.ensure_post_gate_yaw_key(
                gate, key=key, yaw_deg=yaw_deg, yaw_rate_deg=rate, lead_s=lead_s,
            ):
                when = (
                    f'{abs(lead_s):.2f}s after' if lead_s < 0
                    else f'{lead_s:.2f}s before' if lead_s > 0
                    else 'at'
                )
                msgs.append(
                    f'KEY {key.upper()} ~{yaw_deg:.0f}° {label} {when} GATE {gate}'
                )
        return msgs

    def trim_until_gate(self, gate: int, *, require: bool = False):
        g = int(gate)
        gate_t = None
        for e in self.events:
            if (
                str(e.get('event') or '') == 'gate_pass'
                and int(e.get('active_gate', -1)) == g
            ):
                gate_t = float(e['t'])
                break
        if gate_t is None:
            msg = (
                f'no gate {g} tag on key timeline ({len(self.events)} events) — '
                f'playing FULL timeline ({self.duration:.1f}s). '
                f'Re-capture, HOLD keys through the gate, wait for '
                f'"GATE {g} @ t=..." before K.'
            )
            if require:
                raise ValueError(msg)
            out = KeyTimeline(self.path, name=f'{self.name}_full')
            out.ekf_use_pnp = self.ekf_use_pnp
            out.events = [dict(e) for e in self.events]
            return out, msg
        try:
            import config as _cfg
            after_s = float(getattr(_cfg, 'PILOT_TRIM_AFTER_GATE_S', 4.0))
        except Exception:
            after_s = 4.0
        cut_t = gate_t + max(0.0, after_s)
        # Keep any already-baked post-gate yaw for this gate.
        yaw_done = self.post_gate_yaw_done_t(g)
        if yaw_done is not None:
            cut_t = max(cut_t, float(yaw_done))
        out = KeyTimeline(self.path, name=f'{self.name}_until_g{g}')
        out.ekf_use_pnp = self.ekf_use_pnp
        out.frozen_through_gate = int(getattr(self, 'frozen_through_gate', 0) or 0)
        out.events = [dict(e) for e in self.events if float(e['t']) <= cut_t + 1e-9]
        # Ensure any keys still down at cut are released so replay ends clean.
        held = out.keys_held_at(cut_t + 1e-6)
        for k in sorted(held):
            out.events.append({'t': round(cut_t + 0.001, 4), 'key': k, 'down': False})
        if len(out.events) < 1:
            raise ValueError(f'gate {g} timeline empty')
        return out, None

    def key_hold_s(self) -> float:
        """Total time any flight key was held (union of intervals)."""
        if not self.events:
            return 0.0
        # Sweep line on down/up.
        t_end = max(float(e['t']) for e in self.events) + 0.05
        # Sample at 50 Hz for a simple measure.
        total = 0.0
        dt = 0.02
        t = min(float(e['t']) for e in self.events)
        while t <= t_end:
            if self.keys_held_at(t):
                total += dt
            t += dt
        return total

    def trim_idle_edges(self, *, pad_s: float = 0.15) -> int:
        """Drop leading/trailing idle; re-base times to 0."""
        if not self.events:
            return 0
        first_t = None
        last_t = None
        held: set[str] = set()
        for e in self.events:
            key = e.get('key')
            if key:
                if e.get('down'):
                    held.add(str(key))
                else:
                    held.discard(str(key))
                if held and first_t is None:
                    first_t = float(e['t'])
                if held:
                    last_t = float(e['t'])
            elif e.get('event') == 'gate_pass':
                if first_t is None:
                    first_t = float(e['t'])
                last_t = float(e['t'])
        if first_t is None:
            return 0
        t0 = max(0.0, first_t - pad_s)
        t1 = (last_t if last_t is not None else first_t) + pad_s
        kept = [dict(e) for e in self.events if t0 - 1e-9 <= float(e['t']) <= t1 + 1e-9]
        removed = len(self.events) - len(kept)
        if not kept:
            return 0
        base = float(kept[0]['t'])
        for e in kept:
            e['t'] = round(float(e['t']) - base, 4)
        self.events = kept
        return removed

    def save(self):
        if len(self.events) < 1:
            return None
        import config

        # Release any keys still held at end.
        if self._held:
            t = round(self._t_offset + (self.duration if self.events else 0.0), 4)
            if self.events:
                t = max(t, max(float(e['t']) for e in self.events) + 0.001)
            for k in sorted(self._held):
                self.events.append({'t': t, 'key': k, 'down': False})
            self._held.clear()

        for msg in self.ensure_pilot_gate_yaws():
            print(f'[REMEMBER] appended {msg}', flush=True)

        removed = self.trim_idle_edges()
        hold_s = self.key_hold_s()
        if removed:
            print(
                f'[REMEMBER] trimmed idle; {len(self.events)} key events, '
                f'{self.duration:.1f}s span, keys held ~{hold_s:.1f}s',
                flush=True,
            )
        if hold_s < 0.4:
            print(
                f'[REMEMBER] WARNING: only ~{hold_s:.2f}s of keys held — '
                'HOLD W (and other keys) longer or replay will barely move.',
                flush=True,
            )

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'type': 'key_timeline',
            'name': self.name,
            'ekf_use_pnp': int(
                self.ekf_use_pnp
                if self.ekf_use_pnp is not None
                else bool(getattr(config, 'EKF_USE_PNP', True))
            ),
            'events': self.events,
        }
        if int(getattr(self, 'frozen_through_gate', 0) or 0) > 0:
            payload['frozen_through_gate'] = int(self.frozen_through_gate)
        tmp = str(self.path) + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, self.path)
        self.saved_once = True
        return self.path


def load_timeline(path) -> KeyTimeline | None:
    """Load a key timeline (or reject legacy control_timeline files)."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return None
    if raw.get('type') == 'control_timeline' or (
        'samples' in raw and 'events' not in raw
    ):
        print(
            f'[REMEMBER] {path} is an old stick-sample capture — '
            're-record with pilot --capture (key press timeline).',
            flush=True,
        )
        return None
    if raw.get('type') == 'key_timeline' or 'events' in raw:
        tl = KeyTimeline(path, name=raw.get('name', 'remembered'))
        tl.ekf_use_pnp = raw.get('ekf_use_pnp')
        tl.frozen_through_gate = int(raw.get('frozen_through_gate') or 0)
        for e in raw.get('events', []):
            if 't' not in e:
                continue
            tl.events.append(e)
        if not tl.events:
            return None
        tl.trim_idle_edges()
        return tl
    return None


def apply_keys_to_hold_state(
    held: set[str],
    hold_state: dict,
    *,
    lean_rad: float,
    yaw_rate_cmd: float,
    thrust_step: float,
    now: float,
    sink_step: float | None = None,
    pitch_rad: float | None = None,
) -> None:
    """Drive the same hold_state axes manual mode uses, from a key set."""
    import config as _cfg
    fwd = float(getattr(_cfg, 'FORWARD_PITCH_SIGN', 1.0))
    yaw_sign = float(getattr(_cfg, 'RATE_SIGN_YAW', 1.0))
    pitch_lim = float(lean_rad) if pitch_rad is None else float(pitch_rad)
    if sink_step is None:
        # Match climb by default — callers pass PILOT_G2_SINK_RATE after G1.
        sink_step = float(
            getattr(_cfg, 'PILOT_SINK_RATE', abs(thrust_step))
        )

    for axis in ('roll', 'pitch', 'yaw', 'thrust'):
        hold_state[axis] = 0.0
        hold_state[f'{axis}_t'] = 0.0

    def _press(axis: str, value: float) -> None:
        hold_state[axis] = value
        hold_state[f'{axis}_t'] = now

    for key in held:
        if key in ('a', 'left'):
            _press('roll', -lean_rad)
        elif key in ('d', 'right'):
            _press('roll', lean_rad)
        elif key in ('w', 'up'):
            _press('pitch', fwd * pitch_lim)
        elif key in ('s', 'down'):
            _press('pitch', -fwd * pitch_lim)
        elif key == 'q':
            _press('yaw', -yaw_sign * yaw_rate_cmd)
        elif key == 'e':
            _press('yaw', yaw_sign * yaw_rate_cmd)
        elif key == 'r':
            _press('thrust', thrust_step)
        elif key == 'f':
            _press('thrust', -abs(sink_step))


class KeyReplayClock:
    """Wall-clock playback cursor over a :class:`KeyTimeline`."""

    def __init__(self, timeline: KeyTimeline):
        if timeline is None or len(timeline) < 1:
            raise ValueError('key timeline is empty')
        self.timeline = timeline
        self._t0 = None
        self._finished = False
        t0 = min(float(e['t']) for e in timeline.events)
        self._t_shift = t0
        self._t_end = max(float(e['t']) for e in timeline.events) - t0
        # Small pad after last release so the final tip is flown.
        self._t_end = max(self._t_end, 0.05) + 0.15
        hold_s = timeline.key_hold_s()
        print(
            f'[REMEMBER] key replay {len(timeline)} events, '
            f'{self._t_end:.2f}s, keys held ~{hold_s:.1f}s, '
            f'EKF_USE_PNP={timeline.ekf_use_pnp}',
            flush=True,
        )
        if hold_s < 0.4:
            print(
                '[REMEMBER] WARNING: almost no keys held in capture — '
                're-run pilot --capture and HOLD W through the gate.',
                flush=True,
            )

    def reset_episode(self) -> None:
        self._t0 = None
        self._last = None
        self._accum = 0.0
        self._finished = False

    def tick(self, time_scale: float = 1.0) -> tuple[float, set[str], bool]:
        """Return (elapsed, keys_held, finished).

        ``time_scale`` < 1 advances the playhead slower than wall clock
        (pair with a matching sim speedhack).
        """
        now = time.monotonic()
        if self._last is None:
            self._last = now
            self._accum = 0.0
            self._t0 = now
        dt = now - self._last
        self._last = now
        scale = max(1e-3, float(time_scale))
        self._accum += dt * scale
        elapsed = self._accum
        if elapsed >= self._t_end:
            self._finished = True
            return elapsed, set(), True
        held = self.timeline.keys_held_at(elapsed + self._t_shift)
        return elapsed, held, False


# Back-compat aliases used by older call sites / tests.
ControlTimeline = KeyTimeline
TimelineReplayPlanner = KeyReplayClock  # type: ignore
