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
    def __init__(self, cfg: TerminationConfig):
        self.cfg = cfg
        self._sim_time0: Optional[float] = None
        self._last_gate: Optional[int] = None
        self._last_gate_sim_time: float = 0.0

    def reset(self, sim_time: float, active_gate: Optional[int]) -> None:
        self._sim_time0 = sim_time
        self._last_gate = active_gate
        self._last_gate_sim_time = sim_time

    def check(
        self,
        sim_time: float,
        active_gate: Optional[int],
        finished: bool,
        collision_threat: int,
    ) -> TerminationState:
        cfg = self.cfg
        if self._sim_time0 is None:
            self._sim_time0 = sim_time
        elapsed = sim_time - self._sim_time0

        # progress watchdog
        if active_gate is not None and active_gate != self._last_gate:
            self._last_gate = active_gate
            self._last_gate_sim_time = sim_time

        if cfg.finish_terminates and finished:
            return TerminationState(terminated=True, reason="finish")
        if cfg.collision_terminates and collision_threat >= cfg.collision_threat_min and collision_threat > 0:
            return TerminationState(terminated=True, reason="collision")
        if elapsed >= cfg.episode_timeout_s:
            return TerminationState(truncated=True, reason="timeout")
        if (sim_time - self._last_gate_sim_time) >= cfg.stuck_no_gate_s:
            return TerminationState(truncated=True, reason="stuck")
        return TerminationState()
