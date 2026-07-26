"""Rate / jitter / latency meters used by the probe and by the env watchdogs.

Do not assume any nominal rate — everything here is measured from event timestamps.
"""
from __future__ import annotations

import statistics
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class RateStats:
    count: int
    mean_hz: float
    mean_dt_ms: float
    std_dt_ms: float
    min_dt_ms: float
    max_dt_ms: float
    p95_dt_ms: float


class RateMeter:
    """Tracks inter-event intervals from wall-clock (or supplied) timestamps."""

    def __init__(self, window: int = 512):
        self._dts: deque[float] = deque(maxlen=window)
        self._last: Optional[float] = None
        self.count = 0

    def tick(self, t: Optional[float] = None) -> None:
        t = time.time() if t is None else t
        if self._last is not None:
            dt = t - self._last
            if dt > 0:
                self._dts.append(dt)
        self._last = t
        self.count += 1

    def stats(self) -> Optional[RateStats]:
        if len(self._dts) < 2:
            return None
        dts = list(self._dts)
        dts_ms = [d * 1000.0 for d in dts]
        mean_dt = statistics.fmean(dts)
        srt = sorted(dts_ms)
        p95 = srt[min(len(srt) - 1, int(0.95 * len(srt)))]
        return RateStats(
            count=self.count,
            mean_hz=(1.0 / mean_dt) if mean_dt > 0 else 0.0,
            mean_dt_ms=mean_dt * 1000.0,
            std_dt_ms=(statistics.pstdev(dts_ms) if len(dts_ms) > 1 else 0.0),
            min_dt_ms=min(dts_ms),
            max_dt_ms=max(dts_ms),
            p95_dt_ms=p95,
        )
