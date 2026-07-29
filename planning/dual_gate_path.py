"""Path through Gate 1; yaw keeps Gate 1 framed, then slews to Gate 2.

Uses EKF NED gate centres. Approach waypoint sits in front of Gate 1 along
its through-axis (G1→G2 when available).

Yaw policy (fixes the “yawed away from the gate” failure):
- approach / through: look at **Gate 1** so the camera stays on the target
- exit: look at **Gate 2** (or the EKF dead-reckon memory of it)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DualGatePath:
    approach_ned: np.ndarray
    gate1_ned: np.ndarray
    exit_ned: np.ndarray
    gate2_ned: Optional[np.ndarray]
    through_unit_ned: np.ndarray
    look_yaw_rad: float
    target_ned: np.ndarray
    phase: str  # approach | through | exit


class DualGatePathPlanner:
    def __init__(
        self,
        *,
        approach_distance_m: float = 3.5,
        exit_distance_m: float = 1.5,
        switch_radius_m: float = 0.9,
    ):
        self.approach_distance_m = approach_distance_m
        self.exit_distance_m = exit_distance_m
        self.switch_radius_m = switch_radius_m

    def plan(
        self,
        position_ned: np.ndarray,
        gate1_ned: np.ndarray,
        gate2_ned: Optional[np.ndarray],
        through_hint_ned: Optional[np.ndarray] = None,
    ) -> DualGatePath:
        p = np.asarray(position_ned, dtype=np.float64).reshape(3)
        g1 = np.asarray(gate1_ned, dtype=np.float64).reshape(3)
        g2 = (
            None
            if gate2_ned is None
            else np.asarray(gate2_ned, dtype=np.float64).reshape(3)
        )

        if through_hint_ned is not None:
            through = np.asarray(through_hint_ned, dtype=np.float64).reshape(3)
        elif g2 is not None:
            through = g2 - g1
        else:
            through = g1 - p
        norm = float(np.linalg.norm(through))
        if norm < 1e-3:
            through = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            through = through / norm

        approach = g1 - through * self.approach_distance_m
        exit_pt = g1 + through * self.exit_distance_m

        dist_gate = float(np.linalg.norm(p - g1))
        past = float(np.dot(p - g1, through)) > 0.15

        if past and dist_gate < self.switch_radius_m * 2.5:
            phase = 'exit'
            target = exit_pt if g2 is None else g2
        elif dist_gate < self.switch_radius_m:
            phase = 'through'
            target = g1
        else:
            phase = 'approach'
            target = approach

        # Keep Gate 1 in FOV until we are past it; then look at Gate 2.
        if phase == 'exit' and g2 is not None:
            look = g2 - p
        else:
            look = g1 - p
        look_yaw = math.atan2(float(look[1]), float(look[0]))

        return DualGatePath(
            approach_ned=approach,
            gate1_ned=g1,
            exit_ned=exit_pt,
            gate2_ned=g2,
            through_unit_ned=through,
            look_yaw_rad=look_yaw,
            target_ned=np.asarray(target, dtype=np.float64).reshape(3),
            phase=phase,
        )


def body_look_yaw(
    yaw_rad: float,
    gate1_body: Optional[np.ndarray],
    gate2_body: Optional[np.ndarray] = None,
    *,
    phase: str = 'approach',
    center_deadband_rad: float = 0.12,
    gate2_blend: float = 0.25,
) -> Optional[float]:
    """Absolute look yaw from body-frame PnP (preferred over NED look-at).

    Primary: null Gate 1 bearing (keep the approach gate centred).
    Soft Gate 2 bias only when Gate 1 is already near centre, or on exit.
    """
    if gate1_body is None and gate2_body is None:
        return None
    g1 = None if gate1_body is None else np.asarray(gate1_body, dtype=np.float64)
    g2 = None if gate2_body is None else np.asarray(gate2_body, dtype=np.float64)

    if phase == 'exit' and g2 is not None:
        bearing = math.atan2(float(g2[1]), float(g2[0]))
    elif g1 is not None:
        bearing = math.atan2(float(g1[1]), float(g1[0]))
        if (
            g2 is not None
            and abs(bearing) < center_deadband_rad
            and gate2_blend > 0.0
        ):
            b2 = math.atan2(float(g2[1]), float(g2[0]))
            bearing = (1.0 - gate2_blend) * bearing + gate2_blend * b2
    else:
        bearing = math.atan2(float(g2[1]), float(g2[0]))

    # Reject nonsense behind-camera bearings — do not spin 180°.
    if abs(bearing) > math.radians(80.0):
        bearing = math.copysign(math.radians(80.0), bearing)
    return yaw_rad + bearing
