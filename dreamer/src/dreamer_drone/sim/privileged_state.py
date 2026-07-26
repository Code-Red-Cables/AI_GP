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
    collision_is_gate: bool        # this step's collision hit a gate frame (id 1001),
                                   # not terrain/obstacles (id 1002)
    position: Optional[tuple]      # (x,y,z) if available (VQ2: usually None)
    dist_to_gate: Optional[float]  # privileged geometric progress if computable


class PrivilegedState:
    def __init__(self, io: MavlinkIO):
        self.io = io
        self._last_collision_seq = 0
        self._t0_wall = time.time()
        self._gate_hwm: Optional[int] = None

    def reset(self) -> None:
        self._last_collision_seq = self.io.collision_seq
        self._t0_wall = time.time()
        self._gate_hwm = None

    def _new_collision(self) -> tuple[int, bool]:
        """Returns (threat_level, is_gate) for a NEW collision this step, else (0, False)."""
        seq = self.io.collision_seq
        if seq > self._last_collision_seq:
            self._last_collision_seq = seq
            col = self.io.get("collision") or {}
            threat = int(col.get("threat", 1) or 1)
            return threat, col.get("id") == 1001
        return 0, False

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

        # Monotonic high-water mark: the sim's race_status stream flickers between
        # the advanced gate index and a stale 0 (~0.75s period, measured 2026-07-25).
        # Pre-pass the signal is a solid 0 — upward transitions are real, downward
        # ones are telemetry noise. Without this the gate reward re-fires every few
        # steps forever after the first pass (infinite reward farm).
        active_gate = rs.get("active_gate")
        if active_gate is not None:
            if self._gate_hwm is None or int(active_gate) > self._gate_hwm:
                self._gate_hwm = int(active_gate)
            active_gate = self._gate_hwm
        else:
            active_gate = self._gate_hwm
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

        threat, is_gate = self._new_collision()
        return PrivilegedSnapshot(
            sim_time_s=self.sim_time_s(),
            active_gate=active_gate,
            num_gates=num_gates,
            finished=finished,
            collision_threat=threat,
            collision_is_gate=is_gate,
            position=pos,
            dist_to_gate=dist_to_gate,
        )
