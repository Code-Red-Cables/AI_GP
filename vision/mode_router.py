"""Exclusive selection between OpenCV navigation and the existing AI policy."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional


class GateNavigationMode(str, enum.Enum):
    OPENCV = "opencv"
    EXISTING_AI = "existing_ai"


# Compatibility export retained for callers from the prior implementation.
VisionMode = GateNavigationMode


@dataclass(frozen=True)
class ModeRouterConfig:
    mode: GateNavigationMode = GateNavigationMode.OPENCV


class VisionModeRouter:
    """Choose one command owner; OpenCV and AI outputs are never blended."""

    def __init__(self, config: Optional[ModeRouterConfig] = None):
        self.config = config or ModeRouterConfig()
        self.active_source = (
            "ai"
            if self.config.mode == GateNavigationMode.EXISTING_AI
            else "opencv"
        )

    def update(self, confidence: float = 0.0, ai_available: bool = False) -> str:
        del confidence
        if self.config.mode == GateNavigationMode.OPENCV:
            self.active_source = "opencv"
        else:
            self.active_source = "ai" if ai_available else "safe"
        return self.active_source
