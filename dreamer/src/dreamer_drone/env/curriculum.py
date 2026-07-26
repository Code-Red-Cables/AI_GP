"""Success-rate-driven curriculum.

Advancement depends on measured success over a rolling window (prompt Phase 3), not a
fixed step count. Each stage names a reset distribution the env's `reset_manager`
realizes (some stages need privileged spawn control the sim may or may not expose — the
env degrades gracefully to a plain reset when a distribution is unavailable).
"""
from __future__ import annotations

from collections import deque

from ..config import CurriculumConfig


# Per-stage success criteria (2026-07-24). Success used to be "finished the whole race"
# for EVERY stage, so a policy that couldn't already win never promoted out of stage 0
# ("stage=hover" forever in the training logs). Each stage now has an attainable goal:
#   ("survive", _)  episode ended without a collision
#   ("gates", n)    passed at least n gates this episode
#   ("finish", _)   crossed the finish line
_STAGE_GOALS: dict[str, tuple[str, int]] = {
    "hover": ("survive", 0),
    "single_gate": ("gates", 1),
    "random_near_gate": ("gates", 1),
    "two_gate": ("gates", 2),
    "short_segment": ("gates", 3),
    "full_course": ("finish", 0),
    "full_course_fast": ("finish", 0),
    "recovery": ("gates", 1),
}


class Curriculum:
    def __init__(self, cfg: CurriculumConfig):
        self.cfg = cfg
        self.stage = cfg.start_stage if cfg.enabled else len(cfg.stages) - 1
        self._results: deque[bool] = deque(maxlen=cfg.promote_window)

    @property
    def stage_name(self) -> str:
        return self.cfg.stages[self.stage]

    @property
    def at_final_stage(self) -> bool:
        return self.stage >= len(self.cfg.stages) - 1

    def episode_success(self, term_reason: str, gates_passed: int) -> bool:
        """Stage-relative success for one finished episode."""
        if term_reason == "finish":
            return True
        kind, n = _STAGE_GOALS.get(self.stage_name, ("finish", 0))
        if kind == "survive":
            return term_reason != "collision"
        if kind == "gates":
            return gates_passed >= n
        return False

    def record_episode(self, success: bool) -> bool:
        """Record an episode outcome; return True if the stage advanced."""
        if not self.cfg.enabled:
            return False
        self._results.append(bool(success))
        if len(self._results) < self._results.maxlen:
            return False
        rate = sum(self._results) / len(self._results)
        if rate >= self.cfg.promote_success_rate and not self.at_final_stage:
            self.stage += 1
            self._results.clear()
            return True
        return False

    def state(self) -> dict:
        rate = (sum(self._results) / len(self._results)) if self._results else 0.0
        return {
            "stage": self.stage,
            "stage_name": self.stage_name,
            "success_rate": rate,
            "window": len(self._results),
        }
