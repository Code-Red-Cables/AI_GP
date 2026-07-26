"""Episode termination / truncation logic (privileged; training + eval only).

`terminated` = the episode ended for a reason intrinsic to the task (finish, fatal
collision). `truncated` = an external cutoff (time budget, stuck watchdog). This split
matches Gymnasium semantics so bootstrapping is correct at truncation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..config import TerminationConfig


@dataclass
class TerminationState:
    terminated: bool = False
    truncated: bool = False
    reason: str = ""


class TerminationChecker:
    """Accumulates per-step `dt` rather than differencing an absolute sim clock.

    MEASURED 2026-07-24: episodes ran to the 3000-step hard cap with neither timeout
    (60 s) nor the stuck watchdog (12 s) firing. Cause: elapsed time was computed as
    `sim_time - sim_time0`, and the sim clock can rewind mid-episode (an external
    reset — e.g. a second trainer attached to the same sim — restarts it) or freeze,
    making the difference negative/stuck forever. The env already computes a robust
    per-step dt (sim-time delta with a camera-period fallback); summing it here keeps
    both watchdogs alive no matter what the absolute clock does.
    """

    def __init__(self, cfg: TerminationConfig):
        self.cfg = cfg
        self._elapsed = 0.0
        self._since_gate = 0.0
        self._last_gate: Optional[int] = None

    def reset(self, active_gate: Optional[int]) -> None:
        self._elapsed = 0.0
        self._since_gate = 0.0
        self._last_gate = active_gate

    def check(
        self,
        dt: float,
        active_gate: Optional[int],
        finished: bool,
        collision_threat: int,
    ) -> TerminationState:
        cfg = self.cfg
        # one step can't claim more than 1 s (guards against clock jumps in dt itself)
        dt = max(0.0, min(float(dt), 1.0))
        self._elapsed += dt
        self._since_gate += dt

        # progress watchdog
        if active_gate is not None and active_gate != self._last_gate:
            self._last_gate = active_gate
            self._since_gate = 0.0

        if cfg.finish_terminates and finished:
            return TerminationState(terminated=True, reason="finish")
        if cfg.collision_terminates and collision_threat >= cfg.collision_threat_min and collision_threat > 0:
            return TerminationState(terminated=True, reason="collision")
        if self._elapsed >= cfg.episode_timeout_s:
            return TerminationState(truncated=True, reason="timeout")
        if self._since_gate >= cfg.stuck_no_gate_s:
            return TerminationState(truncated=True, reason="stuck")
        return TerminationState()
