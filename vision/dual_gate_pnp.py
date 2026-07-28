"""Locate the two closest YOLO pose gates and PnP-solve both.

Gate centres are returned in the body frame with the drone as the origin
(camera→body via camera_model). Sorted near→far by PnP range.

Filters out tiny / low-confidence junk (station signs, far speckles) that
were stealing the nearest-gate slot and spinning yaw in run 043043.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from vision.yolo_pnp import GatePnP, solve_corners_pnp

# Reject detections that are too small in the image or absurdly close/far.
MIN_BBOX_AREA_PX = 900.0       # ~30x30 — junk at 5x6 px was winning "nearest"
MIN_CONFIDENCE = 0.45
MAX_RANGE_M = 28.0
MIN_RANGE_M = 0.8


@dataclass
class DualGateObservation:
    gate1: GatePnP
    gate2: Optional[GatePnP]
    gate1_body: np.ndarray
    gate2_body: Optional[np.ndarray]
    gate1_through_body: Optional[np.ndarray]
    timestamp: float


def _bbox_area(bbox) -> float:
    if bbox is None or len(bbox) < 4:
        return 0.0
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _through_body(gate: GatePnP) -> Optional[np.ndarray]:
    """Unit through-gate axis in body (levelled horizontal flight direction)."""
    import camera_model as cm

    if not gate.solved:
        return None
    z_cam = gate.R_cg[:, 2]
    z_body = cm.cam_to_body(z_cam)
    if float(z_body[0]) < 0.0:
        z_body = -z_body
    z_body = np.array([float(z_body[0]), float(z_body[1]), 0.0])
    n = float(np.linalg.norm(z_body))
    if n < 1e-6:
        return None
    return z_body / n


def _solve_candidate(
    candidate: object,
    *,
    min_confidence: float,
) -> Optional[GatePnP]:
    box = getattr(candidate, 'box', None)
    keypoints = getattr(candidate, 'keypoints', None)
    if box is None or keypoints is None:
        return None
    conf = float(getattr(box, 'confidence', 0.0))
    if conf < min_confidence:
        return None
    bbox = getattr(box, 'bbox', None)
    if _bbox_area(bbox) < MIN_BBOX_AREA_PX:
        return None
    gate = solve_corners_pnp(
        keypoints,
        confidence=conf,
        bbox=bbox,
    )
    if gate is None or not gate.solved:
        return None
    if not (MIN_RANGE_M < gate.range_m < MAX_RANGE_M):
        return None
    return gate


def observe_two_closest_gates(
    candidates: Sequence[object],
    *,
    timestamp: float,
    min_confidence: float = MIN_CONFIDENCE,
    preferred: object | None = None,
) -> Optional[DualGateObservation]:
    """Solve PnP for visible gates; Gate1 prefers the detector's selected target.

    Using nearest-by-range alone stole the lock onto a different instance than
    YOLO steered to, so image yaw fought the camera (run 043523).
    """
    solved: List[GatePnP] = []
    preferred_gate: Optional[GatePnP] = None
    if preferred is not None:
        preferred_gate = _solve_candidate(
            preferred, min_confidence=min_confidence
        )

    for candidate in candidates:
        if preferred is not None and candidate is preferred:
            continue
        gate = _solve_candidate(candidate, min_confidence=min_confidence)
        if gate is not None:
            solved.append(gate)

    solved.sort(key=lambda g: (g.range_m, -_bbox_area(g.bbox)))
    if preferred_gate is not None:
        g1 = preferred_gate
        # Second = nearest other gate.
        g2 = solved[0] if solved else None
    else:
        if not solved:
            return None
        g1 = solved[0]
        g2 = solved[1] if len(solved) >= 2 else None

    return DualGateObservation(
        gate1=g1,
        gate2=g2,
        gate1_body=g1.center_body(),
        gate2_body=None if g2 is None else g2.center_body(),
        gate1_through_body=_through_body(g1),
        timestamp=timestamp,
    )
