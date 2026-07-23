"""Stateful action sender: normalized [-1,1]^4 -> scaled SET_ATTITUDE_TARGET, with
optional slew-rate limiting, low-pass filtering, action-hold (when the control loop runs
faster than fresh observations), a watchdog that decays to a neutral hover on missing
data, and an emergency-neutral. Shared by training and deployment for zero skew.
"""
from __future__ import annotations

import numpy as np

from ..config import ActionConfig
from ..env.spaces import ACTION_DIM, neutral_action, scale_action
from .mavlink_io import MavlinkIO


class ActionSender:
    def __init__(self, io: MavlinkIO, cfg: ActionConfig):
        self.io = io
        self.cfg = cfg
        self._last_norm = neutral_action()
        self._filtered = neutral_action()

    def reset(self) -> None:
        self._last_norm = neutral_action()
        self._filtered = neutral_action()

    def _condition(self, norm_action: np.ndarray) -> np.ndarray:
        a = np.clip(np.asarray(norm_action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        if a.shape[0] != ACTION_DIM:
            raise ValueError(f"expected {ACTION_DIM} action dims, got {a.shape[0]}")
        # slew-rate limit
        if self.cfg.slew_rate_limit > 0:
            delta = np.clip(a - self._last_norm,
                            -self.cfg.slew_rate_limit, self.cfg.slew_rate_limit)
            a = self._last_norm + delta
        # low-pass filter
        if self.cfg.lpf_alpha > 0:
            self._filtered = (1 - self.cfg.lpf_alpha) * self._filtered + self.cfg.lpf_alpha * a
            a = self._filtered.copy()
        self._last_norm = a.copy()
        return a

    def send(self, norm_action: np.ndarray) -> np.ndarray:
        """Condition, scale, and transmit an action. Returns the conditioned normalized
        action actually applied (for the replay buffer / prev-action feature)."""
        a = self._condition(norm_action)
        phys = scale_action(a, self.cfg)
        self.io.send_attitude_target(phys.thrust, phys.roll_rate,
                                     phys.pitch_rate, phys.yaw_rate)
        return a

    def send_neutral(self) -> None:
        """Emergency / watchdog: hover thrust, zero rates."""
        phys = scale_action(neutral_action(), self.cfg)
        self.io.send_attitude_target(phys.thrust, phys.roll_rate,
                                     phys.pitch_rate, phys.yaw_rate)
        self._last_norm = neutral_action()
