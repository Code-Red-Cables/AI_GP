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


def infer_reference_sim_speed(tape: dict[str, Any]) -> float | None:
    """Fit sim-clock speed from timestamped race packets in tape samples."""
    by_receive: dict[float, float] = {}
    for sample in tape.get('samples') or []:
        race = (sample.get('reference') or {}).get('race') or {}
        try:
            receive_s = float(race['received_perf_counter_s'])
            boot_s = float(race['sim_boot_ms']) / 1000.0
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(receive_s) and math.isfinite(boot_s):
            by_receive[receive_s] = boot_s
    points = sorted(by_receive.items())
    if len(points) < 3 or points[-1][0] - points[0][0] < 1.0:
        return None
    x0 = sum(x for x, _ in points) / len(points)
    y0 = sum(y for _, y in points) / len(points)
    denom = sum((x - x0) ** 2 for x, _ in points)
    if denom <= 0:
        return None
    speed = sum((x - x0) * (y - y0) for x, y in points) / denom
    if not math.isfinite(speed) or not 0.02 <= speed <= 4.0:
        return None
    for nice in (0.1, 0.2, 0.25, 0.5, 1.0, 2.0):
        if abs(speed - nice) <= 0.02 * nice:
            return nice
    return speed


def infer_recorded_start_phase(
    tape: dict[str, Any], sim_speed: float
) -> float | None:
    """Extrapolate race phase at tape t=0 from a timestamped race packet."""
    for sample in list(tape.get('samples') or [])[:32]:
        ref = sample.get('reference') or {}
        race = ref.get('race') or {}
        try:
            boot_ms = int(race['sim_boot_ms'])
            start_ms = int(race['race_start_ms'])
            race_rx = float(race['received_perf_counter_s'])
            sample_pc = float(ref['sample_perf_counter_s'])
        except (KeyError, TypeError, ValueError):
            continue
        age_wall_s = sample_pc - race_rx
        if start_ms < 0 or not -0.01 <= age_wall_s <= 2.0:
            continue
        phase_s = (boot_ms - start_ms) / 1000.0
        phase_s += age_wall_s * float(sim_speed)
        if -0.05 <= phase_s <= 0.5:
            return phase_s
    return None


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
    events = [
        dict(ev)
        for ev in (tape.get('events') or [])
        if float(ev.get('t', -1)) <= cut + 1e-9
    ]
    out = {
        'type': 'attitude_tape',
        'name': f"{tape.get('name', 'tape')}_until_g{g}",
        'control': tape.get('control', 'angle'),
        'gate_passes': [
            dict(gp) for gp in passes if float(gp['t']) <= cut + 1e-9
        ],
        'events': events,
        'samples': samples,
        'duration_s': float(samples[-1]['t']),
        'n': len(samples),
        'trimmed_through_gate': g,
    }
    # Trimming is also used to promote a completed run into a replay
    # reference. Preserve playback-critical metadata; dropping ``control`` or
    # ``sim_speed`` silently changes acro rates into angle replay or runs a
    # slow-motion tape at the wrong rate.
    for key in (
        'sim_speed', 'race_start_offset_s', 'race_start_offset_source',
        'go_delay_s', 'run', 'practice',
        'rate_ref', 'reference', 'vision_observe_only', 'sample_clock',
        'reference_channels', 'gate_frame_capture',
    ):
        if tape.get(key) is not None:
            value = tape[key]
            out[key] = dict(value) if isinstance(value, dict) else value
    return out


class AttitudeTapeRecorder:
    """Capture continuous controller commands (+ optional raw pad axes)."""

    def __init__(
        self,
        *,
        name: str = 'practice_session',
        control: str = 'angle',
        metadata: dict[str, Any] | None = None,
    ):
        self.name = name
        # 'angle' = lean setpoints; 'acro_rates' = body rates (rad/s).
        self.control = str(control or 'angle')
        self.metadata = dict(metadata or {})
        self.samples: list[dict[str, float]] = []
        self.gate_passes: list[dict[str, Any]] = []
        # Discrete markers (e.g. mid-run zero-attitude) keyed by tape time.
        self.events: list[dict[str, Any]] = []
        self._t0: Optional[float] = None
        self._last_gate = -1
        self._last_sample_t = -1.0

    def clear(self) -> None:
        self.samples.clear()
        self.gate_passes.clear()
        self.events.clear()
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
        events = [dict(ev) for ev in (tape.get('events') or [])]
        self.samples = samples
        self.gate_passes = passes
        self.events = events
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
        reference: dict[str, Any] | None = None,
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
        if reference is not None:
            # Synchronized observe-only channels (vision/race/IMU) used by a
            # future trajectory tracker. Replay ignores this field today.
            row['reference'] = dict(reference)

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

    def mark_zero_attitude(
        self,
        t_wall: float,
        *,
        reason: str = 'zero',
        t: float | None = None,
    ) -> dict[str, Any]:
        """Record a mid-run declare-level (EKF roll/pitch clear)."""
        if t is None:
            t = round(self.elapsed(t_wall), 4)
        else:
            t = round(float(t), 4)
            if self._t0 is None:
                self._t0 = float(t_wall) - t
        ev = {
            't': t,
            'type': 'zero_attitude',
            'reason': str(reason),
        }
        self.events.append(ev)
        return ev

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
        tape = {
            'type': 'attitude_tape',
            'name': self.name,
            'control': self.control,
            'gate_passes': [dict(gp) for gp in self.gate_passes],
            'events': [dict(ev) for ev in self.events],
            'samples': samples,
            'duration_s': float(samples[-1]['t']) if samples else 0.0,
            'n': len(samples),
        }
        tape.update(self.metadata)
        if tape.get('sim_speed') is None:
            speed = infer_reference_sim_speed(tape)
            if speed is not None:
                tape['sim_speed'] = speed
        if tape.get('race_start_offset_s') is None and tape.get('sim_speed'):
            phase = infer_recorded_start_phase(tape, float(tape['sim_speed']))
            if phase is not None:
                tape['race_start_offset_s'] = round(phase, 6)
                tape['race_start_offset_source'] = 'timestamped_race_clock'
        return tape

    def clone_through_gate(self, gate: int, *, after_s: float = 0.35) -> dict[str, Any]:
        return trim_tape_until_gate(self.to_tape(), gate, after_s=after_s)


class AttitudeTapeClock:
    """Linear-interpolate attitude samples by elapsed seconds since arm."""

    # Optional recorded reference channels (tools/build_tracking_tape.py).
    # ``ref_g*`` are achieved body rates (gyro); ``ref_roll/pitch/yaw`` are
    # attitude angles, and ``ref_yaw`` interpolates on the circle.
    REF_KEYS = (
        'ref_roll', 'ref_pitch', 'ref_yaw',
        'ref_gr', 'ref_gp', 'ref_gy',
    )

    def __init__(self, tape: dict[str, Any]):
        self.tape = tape
        raw_samples = list(tape['samples'])
        # Recording starts when the pilot engages, but the first sample is
        # written only after that control-loop iteration has reached the send
        # point.  Treat the first *sent command* as replay t=0.  Previously
        # index_at(0) returned sample 0 while sample 1 retained its absolute
        # recorder timestamp, stretching the first ZOH interval by ``t0``.
        # With observe-only vision that was 151 ms + 63 ms instead of 63 ms,
        # enough to turn the opening pitch flick into a full flip.
        self.time_origin = float(raw_samples[0]['t'])
        self.samples = []
        for raw in raw_samples:
            sample = dict(raw)
            sample['t'] = max(0.0, float(raw['t']) - self.time_origin)
            self.samples.append(sample)
        self.duration = float(self.samples[-1]['t'])
        self.gate_passes = []
        for raw in (tape.get('gate_passes') or []):
            gate_pass = dict(raw)
            gate_pass['t'] = max(
                0.0, float(raw.get('t', 0.0)) - self.time_origin
            )
            self.gate_passes.append(gate_pass)
        first = self.samples[0]
        self.ref_keys = tuple(k for k in self.REF_KEYS if k in first)
        self.has_attitude_ref = len(self.ref_keys) > 0
        self.events = sorted(
            (
                {
                    **dict(ev),
                    't': max(
                        0.0,
                        float(ev.get('t', 0.0)) - self.time_origin,
                    ),
                }
                for ev in (tape.get('events') or [])
            ),
            key=lambda ev: float(ev.get('t', 0.0)),
        )
        self._i = 0
        self._event_i = 0

    def reset(self) -> None:
        self._i = 0
        self._event_i = 0

    def gate_t(self, gate: int) -> Optional[float]:
        g = int(gate)
        for gp in self.gate_passes:
            if int(gp.get('gate', -1)) == g:
                return float(gp['t'])
        return None

    def due_events(self, t: float) -> list[dict[str, Any]]:
        """Pop discrete events with ``event.t <= t`` that have not fired yet."""
        t = float(t)
        out: list[dict[str, Any]] = []
        while self._event_i < len(self.events):
            ev = self.events[self._event_i]
            if float(ev.get('t', 0.0)) > t + 1e-9:
                break
            out.append(ev)
            self._event_i += 1
        return out

    def hz(self) -> float:
        """Mean sample rate of the tape, in samples per second of tape time."""
        if len(self.samples) < 2 or self.duration <= 0.0:
            return 0.0
        return float(len(self.samples)) / float(self.duration)

    def index_at(self, t: float) -> Optional[int]:
        """Index of the last sample at or before ``t`` (zero-order hold).

        Acro sticks are steppy — adjacent samples can differ by a full 3.14
        rad/s flick — so interpolating between them synthesises rates that
        were never commanded (0.147 rad/s RMS against a 0.26 rad/s signal).
        Replaying by index reproduces the recorded value exactly.
        """
        t = float(t)
        if t < 0.0:
            t = 0.0
        if t > self.duration + 0.25:
            return None
        while (
            self._i + 1 < len(self.samples)
            and float(self.samples[self._i + 1]['t']) <= t
        ):
            self._i += 1
        return self._i

    def sample_index(self, i: int) -> dict[str, float]:
        """The recorded sample at ``i``, verbatim — no interpolation."""
        a = self.samples[int(i)]
        out = {
            'des_roll': float(a['des_roll']),
            'des_pitch': float(a['des_pitch']),
            'yaw_rate': float(a['yaw_rate']),
            'thrust': float(a['thrust']),
        }
        for key in self.ref_keys:
            out[key] = float(a[key])
        return out

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
            out = {
                'des_roll': float(a['des_roll']),
                'des_pitch': float(a['des_pitch']),
                'yaw_rate': float(a['yaw_rate']),
                'thrust': float(a['thrust']),
            }
            for key in self.ref_keys:
                out[key] = float(a[key])
            return out
        b = self.samples[self._i + 1]
        ta, tb = float(a['t']), float(b['t'])
        if tb <= ta:
            u = 0.0
        else:
            u = max(0.0, min(1.0, (t - ta) / (tb - ta)))

        def lerp(ka):
            return (1.0 - u) * float(a[ka]) + u * float(b[ka])

        out = {
            'des_roll': lerp('des_roll'),
            'des_pitch': lerp('des_pitch'),
            'yaw_rate': lerp('yaw_rate'),
            'thrust': lerp('thrust'),
        }
        for key in self.ref_keys:
            if key == 'ref_yaw':
                # Shortest-arc blend so the ±π seam does not spin the setpoint.
                ya, yb = float(a[key]), float(b[key])
                d = (yb - ya + math.pi) % (2.0 * math.pi) - math.pi
                out[key] = ya + u * d
            else:
                out[key] = lerp(key)
        return out
