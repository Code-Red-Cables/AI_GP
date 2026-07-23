"""Componentized reward. Privileged state is used ONLY here (and in termination/eval),
never in the observation. Every term is computed and logged separately (raw + weighted)
so reward exploits are visible and weights are tunable from config.

Design for VQ2 (docs/simulator_audit.md §7): gate crossing comes from the sim's own
`active_gate_idx` increment — that is the ground-truth "crossed the aperture" signal and
needs no gate pose. Dense progress uses privileged position when available, else a
deployment-legal vision proxy (detected-gate area growth).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ..config import RewardConfig


@dataclass
class StepContext:
    """Everything the reward needs for one transition. Privileged fields are explicit."""
    # timing
    dt_sim: float = 0.0                    # sim-time elapsed this step (s)
    # privileged race state
    active_gate: Optional[int] = None      # PRIV: race_status.active_gate
    num_gates: Optional[int] = None        # PRIV: total gates (finish detection)
    finished: bool = False                 # PRIV: race_finish_ns > 0
    collision_threat: int = 0              # PRIV: COLLISION.threat_level this step (0 = none)
    # privileged geometric progress (optional; VQ2 usually None)
    dist_to_gate: Optional[float] = None   # PRIV: distance to next gate plane (m)
    # deployment-legal vision proxy
    gate_area_px: Optional[float] = None   # LEGAL: detected active-gate area (px^2)
    gate_visible: bool = False             # LEGAL: a gate is detected this frame
    # action (for control cost)
    action: Optional[list] = None          # normalized [-1,1]^4
    prev_action: Optional[list] = None


@dataclass
class RewardComponents:
    progress: float = 0.0
    gate_pass: float = 0.0
    finish: float = 0.0
    time_penalty: float = 0.0
    collision: float = 0.0
    control: float = 0.0
    offcourse: float = 0.0
    backtrack: float = 0.0
    alive: float = 0.0
    total: float = 0.0
    raw: dict = field(default_factory=dict)   # unweighted terms, for logging/exploit checks

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in
             ("progress", "gate_pass", "finish", "time_penalty", "collision",
              "control", "offcourse", "backtrack", "alive", "total")}
        d["raw"] = dict(self.raw)
        return d


class RewardComputer:
    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg
        self._prev_gate: Optional[int] = None
        self._prev_dist: Optional[float] = None
        self._prev_sqrt_area: Optional[float] = None
        self._finished = False

    def reset(self, ctx: Optional[StepContext] = None) -> None:
        self._prev_gate = ctx.active_gate if ctx else None
        self._prev_dist = ctx.dist_to_gate if ctx else None
        self._prev_sqrt_area = (
            math.sqrt(ctx.gate_area_px) if (ctx and ctx.gate_area_px) else None
        )
        self._finished = False

    # ------------------------------------------------------------------
    def _progress_term(self, ctx: StepContext) -> tuple[float, float]:
        """Return (raw_progress, source_is_privileged as 1.0/0.0).

        Privileged: -Δdistance to gate plane (closing distance is positive reward).
        Legal proxy: Δ(sqrt(detected gate area)) — bigger gate ⇒ closer. Both clipped.
        """
        cfg = self.cfg
        if cfg.use_privileged_progress and ctx.dist_to_gate is not None:
            if self._prev_dist is None:
                self._prev_dist = ctx.dist_to_gate
            raw = self._prev_dist - ctx.dist_to_gate       # closing = positive
            self._prev_dist = ctx.dist_to_gate
            return float(max(-cfg.progress_clip, min(cfg.progress_clip, raw))), 1.0

        # vision proxy
        if ctx.gate_visible and ctx.gate_area_px and ctx.gate_area_px > 0:
            s = math.sqrt(ctx.gate_area_px)
            if self._prev_sqrt_area is None:
                self._prev_sqrt_area = s
            raw = (s - self._prev_sqrt_area) * 0.01        # scale px-> ~O(1)
            self._prev_sqrt_area = s
            return float(max(-cfg.progress_clip, min(cfg.progress_clip, raw))), 0.0
        return 0.0, 0.0

    def compute(self, ctx: StepContext) -> RewardComponents:
        cfg = self.cfg
        rc = RewardComponents()
        raw: dict = {}

        # --- progress (dense) ---
        raw_prog, priv = self._progress_term(ctx)
        raw["progress"] = raw_prog
        raw["progress_privileged"] = priv
        rc.progress = cfg.w_progress * raw_prog

        # --- gate pass / backtrack (sparse, PRIV) ---
        gate_delta = 0
        if ctx.active_gate is not None and self._prev_gate is not None:
            gate_delta = int(ctx.active_gate) - int(self._prev_gate)
        if ctx.active_gate is not None:
            self._prev_gate = int(ctx.active_gate)
        raw["gate_delta"] = gate_delta
        if gate_delta > 0:
            rc.gate_pass = cfg.w_gate * gate_delta
            # a new gate resets the vision-proxy baseline (now chasing the next gate)
            self._prev_sqrt_area = None
            self._prev_dist = None
        elif gate_delta < 0:
            rc.backtrack = cfg.w_gate * gate_delta   # negative

        # --- finish (PRIV) ---
        if ctx.finished and not self._finished:
            self._finished = True
            rc.finish = cfg.w_finish
        raw["finished"] = float(ctx.finished)

        # --- time penalty (sim time, not wall clock) ---
        rc.time_penalty = -cfg.w_time * max(0.0, ctx.dt_sim)
        raw["dt_sim"] = ctx.dt_sim

        # --- collision (PRIV) ---
        if ctx.collision_threat and ctx.collision_threat > 0:
            rc.collision = -cfg.w_collision
        raw["collision_threat"] = ctx.collision_threat

        # --- control cost: smoothness (Δa) + excessive rate magnitude ---
        control_cost = 0.0
        if ctx.action is not None:
            a = ctx.action
            if ctx.prev_action is not None:
                control_cost += sum((a[i] - ctx.prev_action[i]) ** 2 for i in range(len(a)))
            control_cost += sum(a[i] ** 2 for i in range(1, len(a)))  # penalize big rates
        rc.control = -cfg.w_control * control_cost
        raw["control_cost"] = control_cost

        # --- off-course: no gate visible (LEGAL proxy) ---
        if not ctx.gate_visible:
            rc.offcourse = -cfg.w_offcourse
        raw["gate_visible"] = float(ctx.gate_visible)

        # --- survival bonus (optional; makes staying alive strictly better than crashing) ---
        rc.alive = cfg.w_alive

        rc.total = (rc.progress + rc.gate_pass + rc.finish + rc.time_penalty
                    + rc.collision + rc.control + rc.offcourse + rc.backtrack + rc.alive)
        rc.raw = raw
        return rc
