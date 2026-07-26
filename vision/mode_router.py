"""OpenCV/AI source selection with frame-count hysteresis."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional


class VisionMode(str, enum.Enum):
    OPENCV = "opencv"
    AI = "ai"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class ModeRouterConfig:
    mode: VisionMode = VisionMode.OPENCV
    confidence_threshold: float = 0.28
    low_confidence_frames: int = 8
    recovery_frames: int = 5
    cooldown_frames: int = 20


class VisionModeRouter:
    """Choose exactly one command source; never blend conflicting commands."""

    def __init__(self, config: Optional[ModeRouterConfig] = None):
        self.config = config or ModeRouterConfig()
        self.active_source = (
            "ai" if self.config.mode == VisionMode.AI else "opencv"
        )
        self._low_count = 0
        self._good_count = 0
        self._cooldown = 0

    def update(self, confidence: float, ai_available: bool) -> str:
        mode = self.config.mode
        if mode == VisionMode.OPENCV:
            self.active_source = "opencv"
            return self.active_source
        if mode == VisionMode.AI:
            self.active_source = "ai" if ai_available else "safe"
            return self.active_source

        if self._cooldown > 0:
            self._cooldown -= 1
        good = confidence >= self.config.confidence_threshold
        self._good_count = self._good_count + 1 if good else 0
        self._low_count = self._low_count + 1 if not good else 0

        if (
            self.active_source == "opencv"
            and ai_available
            and self._cooldown == 0
            and self._low_count >= self.config.low_confidence_frames
        ):
            self.active_source = "ai"
            self._cooldown = self.config.cooldown_frames
            self._good_count = 0
        elif (
            self.active_source == "ai"
            and self._cooldown == 0
            and self._good_count >= self.config.recovery_frames
        ):
            self.active_source = "opencv"
            self._cooldown = self.config.cooldown_frames
            self._low_count = 0
        elif self.active_source == "ai" and not ai_available:
            self.active_source = "opencv"
        return self.active_source
