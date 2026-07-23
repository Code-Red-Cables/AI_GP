"""Privileged-state reader — TRAINING / REWARD / EVAL ONLY.

This module reads the PRIV signals (race status, collision, gate poses, position if
present) from `MavlinkIO`. **It must never be imported by `deploy/controller.py`.** The
leakage test asserts that. Keeping all privileged reads behind this one class makes
accidental leakage into the observation structurally hard.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

from .mavlink_io import MavlinkIO


@dataclass
class PrivilegedSnapshot:
    sim_time_s: float
    active_gate: Optional[int]
    num_gates: Optional[int]
    finished: bool
    collision_threat: int          # threat of a *new* collision this step (0 = none)
    position: Optional[tuple]      # (x,y,z) if available (VQ2: usually None)
    dist_to_gate: Optional[float]  # privileged geometric progress if computable


class PrivilegedState:
    def __init__(self, io: MavlinkIO):
        self.io = io
        self._last_collision_seq = 0
        self._t0_wall = time.time()

    def reset(self) -> None:
        self._last_collision_seq = self.io.collision_seq
        self._t0_wall = time.time()

    def _new_collision_threat(self) -> int:
        seq = self.io.collision_seq
        if seq > self._last_collision_seq:
            self._last_collision_seq = seq
            col = self.io.get("collision") or {}
            return int(col.get("threat", 1) or 1)
        return 0

    def sim_time_s(self) -> float:
        """Prefer sim clock (race_status.sim_boot / IMU time_usec); fall back to wall."""
        imu = self.io.get("highres_imu")
        if imu and imu.get("ts_us"):
            return float(imu["ts_us"]) / 1e6
        return time.time() - self._t0_wall

    def snapshot(self) -> PrivilegedSnapshot:
        rs = self.io.get("race_status") or {}
        gates = self.io.get("track_gates")
        pos = None
        for key in ("odometry", "local_position_ned"):
            p = self.io.get(key)
            if p is not None:
                pos = (p.get("x"), p.get("y"), p.get("z"))
                break

        active_gate = rs.get("active_gate")
        num_gates = len(gates) if gates else None
        finished = bool(rs.get("race_finish_ns", 0) and rs["race_finish_ns"] > 0)

        # privileged geometric progress: only if we have BOTH position and a real gate pose
        dist_to_gate: Optional[float] = None
        if pos is not None and gates and active_gate is not None:
            idx = int(active_gate)
            if 0 <= idx < len(gates):
                gx, gy, gz = gates[idx]["pos"]
                if any(v != 0.0 for v in (gx, gy, gz)):  # VQ2 nulls these to 0
                    dist_to_gate = math.dist(pos, (gx, gy, gz))

        return PrivilegedSnapshot(
            sim_time_s=self.sim_time_s(),
            active_gate=active_gate,
            num_gates=num_gates,
            finished=finished,
            collision_threat=self._new_collision_threat(),
            position=pos,
            dist_to_gate=dist_to_gate,
        )
