"""Simulator-time clock for CE / DxWnd slow-mo.

``sim_boot_ms`` from race_status advances in *sim* seconds, but only arrives
at ~4 Hz. Driving control deadlines straight off that packet freezes the
playhead between updates. This module free-runs on the wall clock at the
measured CE speed and gently trims that rate whenever a fresh sim_boot
sample arrives (see MANUAL_INSTRUCTIONS.md "The playhead is hybrid").

Used by AssistImagePlanner when ``ASSIST_SIM_CLOCK`` is on. Imported at
module load even for acro observe-only, so the file must exist.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional


def sim_boot_s(shared_data: Mapping[str, Any]) -> Optional[float]:
    """Current simulator boot time in seconds, or None if unavailable."""
    race = shared_data.get('race_status') or {}
    raw = race.get('sim_boot_ms')
    if raw is None or raw == '' or raw == 'nan':
        return None
    try:
        ms = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(ms):
        return None
    return ms * 1e-3


class SimClock:
    """Wall→sim seconds with CE-speed estimate trimmed by sim_boot_ms."""

    def __init__(
        self,
        *,
        trim_gain: float = 0.25,
        trim_frac: float = 0.15,
        speed_min: float = 0.02,
        speed_max: float = 4.0,
        initial_speed: float = 1.0,
    ):
        self._trim_gain = float(trim_gain)
        self._trim_frac = float(trim_frac)
        self._speed_min = float(speed_min)
        self._speed_max = float(speed_max)
        self._speed = max(self._speed_min, min(self._speed_max, float(initial_speed)))
        self._last_wall: Optional[float] = None
        self._last_sim: Optional[float] = None
        self._last_out: Optional[float] = None
        self._span0: Optional[tuple[float, float]] = None
        self._span1: Optional[tuple[float, float]] = None

    @property
    def speed(self) -> float:
        """Current estimate of d(sim)/d(wall)."""
        return self._speed

    def reset(self) -> None:
        self._speed = 1.0
        self._last_wall = None
        self._last_sim = None
        self._last_out = None
        self._span0 = None
        self._span1 = None

    def lock_speed_from_span(self) -> float:
        """Snap ``speed`` to the wall→sim slope seen since the first tick.

        The per-packet trim is too slow to learn 0.2x before launch. A 3 s
        countdown at 0.2x is ~15 s of wall with several ``sim_boot_ms``
        steps — enough for a one-shot slope.
        """
        if self._span0 is None or self._span1 is None:
            return self._speed
        dt_wall = self._span1[0] - self._span0[0]
        dt_sim = self._span1[1] - self._span0[1]
        if dt_wall < 0.8 or dt_sim <= 1e-4:
            return self._speed
        measured = dt_sim / dt_wall
        if not math.isfinite(measured):
            return self._speed
        for nice in (0.1, 0.2, 0.25, 0.5, 1.0, 2.0):
            if abs(measured - nice) <= max(0.03, 0.08 * nice):
                self._speed = nice
                return self._speed
        self._speed = max(self._speed_min, min(self._speed_max, measured))
        return self._speed

    def tick(self, wall: float, sim_boot: Optional[float]) -> float:
        """Advance and return simulator time (seconds).

        ``wall`` is ``time.monotonic()``. ``sim_boot`` is
        ``sim_boot_s(shared_data)`` (may be None between packets).
        """
        wall = float(wall)
        sim = (
            float(sim_boot)
            if sim_boot is not None and math.isfinite(float(sim_boot))
            else None
        )

        if self._last_wall is None or self._last_out is None:
            self._last_wall = wall
            self._last_sim = sim
            self._last_out = 0.0 if sim is None else sim
            if sim is not None:
                self._span0 = (wall, sim)
                self._span1 = (wall, sim)
            return self._last_out

        dt_wall = max(0.0, wall - self._last_wall)
        out = self._last_out + dt_wall * self._speed

        if (
            sim is not None
            and self._last_sim is not None
            and dt_wall > 1e-3
        ):
            dsim = sim - self._last_sim
            if dsim > 1e-4:
                measured = dsim / dt_wall
                if math.isfinite(measured):
                    err = measured - self._speed
                    # Cap the per-update trim (±trim_frac of current speed).
                    cap = abs(self._speed) * self._trim_frac
                    if cap > 0.0:
                        err = max(-cap, min(cap, err))
                    self._speed = max(
                        self._speed_min,
                        min(self._speed_max, self._speed + self._trim_gain * err),
                    )

        if sim is not None:
            self._last_sim = sim
            if self._span0 is None:
                self._span0 = (wall, sim)
            self._span1 = (wall, sim)

        if out < self._last_out:
            out = self._last_out
        self._last_wall = wall
        self._last_out = out
        return out
