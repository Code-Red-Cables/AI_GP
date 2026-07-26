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
    collision_is_gate: bool = False        # PRIV: collision id 1001 (gate frame) vs 1002 (env)
    # privileged geometric progress (optional; VQ2 usually None)
    dist_to_gate: Optional[float] = None   # PRIV: distance to next gate plane (m)
    # deployment-legal vision proxy
    gate_area_px: Optional[float] = None   # LEGAL: detected active-gate area (px^2)
    gate_visible: bool = False             # LEGAL: a gate is detected this frame
    gate_center: Optional[tuple] = None    # LEGAL: detection center, normalized (u,v) in [0,1]
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
    centering: float = 0.0
    total: float = 0.0
    raw: dict = field(default_factory=dict)   # unweighted terms, for logging/exploit checks

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in
             ("progress", "gate_pass", "finish", "time_penalty", "collision",
              "control", "offcourse", "backtrack", "alive", "centering", "total")}
        d["raw"] = dict(self.raw)
        return d


class RewardComputer:
    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg
        self._prev_gate: Optional[int] = None
        self._prev_dist: Optional[float] = None
        self._prev_sqrt_area: Optional[float] = None
        self._prev_center: Optional[tuple] = None
        self._seg_high: Optional[float] = None   # max sqrt(area) achieved this gate segment
        self._prev_phi: Optional[float] = None   # centering potential, previous step
        self._last_seen_sqrt: float = 0.0        # sqrt(area) at the last visible frame
        self._blind_steps: int = 0               # consecutive frames without detection
        self._finished = False

    def reset(self, ctx: Optional[StepContext] = None) -> None:
        self._prev_gate = ctx.active_gate if ctx else None
        self._prev_dist = ctx.dist_to_gate if ctx else None
        self._prev_sqrt_area = (
            math.sqrt(ctx.gate_area_px) if (ctx and ctx.gate_area_px) else None
        )
        self._prev_center = ctx.gate_center if ctx else None
        self._seg_high = self._prev_sqrt_area
        self._prev_phi = None
        self._last_seen_sqrt = 0.0
        self._blind_steps = 0
        self._finished = False

    # ------------------------------------------------------------------
    def _centering_term(self, ctx: StepContext) -> float:
        """Potential-based shaping: raw reward = φ(s_t) - φ(s_{t-1}), where φ is how
        horizontally centered the detected gate is (0 best, -0.5 at edge/not visible).

        Pure potential shaping — any closed loop of states sums to zero, so unlike a
        per-step visibility bonus it cannot be farmed by hovering, and unlike a
        per-step penalty it cannot make crashing early attractive (w_offcourse scar).

        CLOSE-RANGE HOLD (2026-07-25): threading a gate necessarily takes it out of
        the camera frame in the final meters, so charging "not visible" as the worst
        state taxed the pass itself — measured run 1785018804: textbook approaches
        veered up and over the gate right at the aperture (area ~2000) once the
        critic absorbed the charge, and live passes stopped. If the gate was LARGE
        when last seen, sight loss is expected crossing behavior: hold φ neutral for
        a few frames (long enough for the pass to register and reset the baseline).
        Losing a DISTANT gate still charges immediately.
        """
        cfg = self.cfg
        if ctx.gate_visible and ctx.gate_center is not None:
            phi = -min(0.5, abs(float(ctx.gate_center[0]) - 0.5))
            self._blind_steps = 0
            self._last_seen_sqrt = math.sqrt(ctx.gate_area_px) if ctx.gate_area_px else 0.0
        else:
            self._blind_steps += 1
            crossing = (cfg.center_hold_sqrt_px > 0
                        and self._last_seen_sqrt >= cfg.center_hold_sqrt_px
                        and self._blind_steps <= cfg.center_hold_steps
                        and self._prev_phi is not None)
            phi = self._prev_phi if crossing else -0.5
        prev, self._prev_phi = self._prev_phi, phi
        if prev is None:
            return 0.0
        return phi - prev

    # ------------------------------------------------------------------
    def _progress_term(self, ctx: StepContext) -> tuple[float, float]:
        """Return (raw_progress, source_is_privileged as 1.0/0.0).

        Privileged: -Δdistance to gate plane (closing distance is positive reward).
        Legal proxy: new HIGHS of sqrt(detected gate area) within the current gate
        segment — bigger gate ⇒ closer, and each px of apparent size pays at most
        once per gate so detector flicker/dropout can't be farmed. Both clipped.
        """
        cfg = self.cfg
        if cfg.use_privileged_progress and ctx.dist_to_gate is not None:
            if self._prev_dist is None:
                self._prev_dist = ctx.dist_to_gate
            raw = self._prev_dist - ctx.dist_to_gate       # closing = positive
            self._prev_dist = ctx.dist_to_gate
            return float(max(-cfg.progress_clip, min(cfg.progress_clip, raw))), 1.0

        # vision proxy — HIGH-WATER-MARK credit (2026-07-24). Paying every positive
        # delta was ratchet-exploitable: area growth paid +, but a shrink below the
        # detection threshold became a DROPOUT (baseline reset) instead of an equal
        # negative, so wobbling in front of a gate farmed unbounded reward (measured
        # +13/episode with 0 gates passed). Now a gate segment pays only for NEW
        # maxima of sqrt(area): total payout is bounded by (closest approach - first
        # sighting) per gate, and no oscillation/dropout/switch pattern re-earns it.
        if ctx.gate_visible and ctx.gate_area_px and ctx.gate_area_px > 0:
            s = math.sqrt(ctx.gate_area_px)
            prev_s, prev_c = self._prev_sqrt_area, self._prev_center
            self._prev_sqrt_area, self._prev_center = s, ctx.gate_center

            # Temporal consistency: a big jump in size or position between consecutive
            # frames means the detector switched targets (gate -> sign -> other gate),
            # so the delta would compare areas of different objects.
            consistent = prev_s is not None and prev_s > 0
            jump = cfg.progress_area_jump
            if consistent and jump > 0 and not (1.0 / jump <= s / prev_s <= jump):
                consistent = False
            if (consistent and cfg.progress_center_jump > 0 and prev_c is not None
                    and ctx.gate_center is not None):
                du = ctx.gate_center[0] - prev_c[0]
                dv = ctx.gate_center[1] - prev_c[1]
                if math.hypot(du, dv) > cfg.progress_center_jump * math.sqrt(2.0):
                    consistent = False

            if not consistent or self._seg_high is None:
                # (re)acquisition or target switch: mark this size as already achieved
                # so the jump itself never pays, then resume normal crediting.
                self._seg_high = s if self._seg_high is None else max(self._seg_high, s)
                return 0.0, 0.0

            if s <= self._seg_high:
                return 0.0, 0.0
            raw = (s - self._seg_high) * 0.01              # scale px-> ~O(1)
            self._seg_high = s
            return float(max(-cfg.progress_clip, min(cfg.progress_clip, raw))), 0.0

        # detection dropout: clear the step baseline (so re-acquisition doesn't yield
        # a stale delta) but KEEP the segment high-water mark — that persistence is
        # what makes the dropout ratchet unprofitable.
        self._prev_sqrt_area = None
        self._prev_center = None
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
            self._prev_center = None
            self._seg_high = None
            self._prev_dist = None
            self._prev_phi = None    # don't charge the view-switch to the next gate
            self._last_seen_sqrt = 0.0
            self._blind_steps = 0
        elif gate_delta < 0:
            rc.backtrack = cfg.w_gate * gate_delta   # negative

        # --- centering shaping (dense, LEGAL, potential-based) ---
        # AFTER the gate block: on a pass the baseline was just cleared, so the view
        # switching to the next gate is re-armed rather than charged as de-centering.
        raw_center = self._centering_term(ctx)
        raw["centering"] = raw_center
        rc.centering = cfg.w_center * raw_center

        # --- finish (PRIV) ---
        if ctx.finished and not self._finished:
            self._finished = True
            rc.finish = cfg.w_finish
        raw["finished"] = float(ctx.finished)

        # --- time penalty (sim time, not wall clock) ---
        rc.time_penalty = -cfg.w_time * max(0.0, ctx.dt_sim)
        raw["dt_sim"] = ctx.dt_sim

        # --- collision (PRIV) ---
        # Gate-frame strikes (sim COLLISION id 1001) are near-misses of a threading
        # attempt; punishing them like ground/obstacle crashes (id 1002) re-teaches
        # gate fear at close range. Softer penalty keeps attempts worth trying.
        if ctx.collision_threat and ctx.collision_threat > 0:
            rc.collision = -(cfg.w_collision_gate if ctx.collision_is_gate
                             else cfg.w_collision)
        raw["collision_threat"] = ctx.collision_threat
        raw["collision_is_gate"] = float(ctx.collision_is_gate)

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
                    + rc.collision + rc.control + rc.offcourse + rc.backtrack
                    + rc.alive + rc.centering)
        rc.raw = raw
        return rc
