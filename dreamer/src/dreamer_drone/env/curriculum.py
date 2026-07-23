"""Success-rate-driven curriculum.

Advancement depends on measured success over a rolling window (prompt Phase 3), not a
fixed step count. Each stage names a reset distribution the env's `reset_manager`
realizes (some stages need privileged spawn control the sim may or may not expose — the
env degrades gracefully to a plain reset when a distribution is unavailable).
"""
from __future__ import annotations

from collections import deque

from ..config import CurriculumConfig


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
