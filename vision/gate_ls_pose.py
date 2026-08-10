"""Bearing-vector least-squares gate pose (Li & de Croon RAS 2020).

Unlike PnP, which solves for 6 unknowns (rotation + translation) from a planar
target, this solver takes attitude from the AHRS and solves only for the
camera position relative to the gate. The closed-form 3x3 system is:

    A = sum_i (I - v_i v_i^T)
    b = sum_i (I - v_i v_i^T) @ p_i
    t = A^{-1} b

where p_i are known 3D gate-corner positions and v_i are unit bearing rays
expressed in the same frame.

Gate frame matches ``vision.yolo_pnp``: X right, Y down, Z through the
opening (flight direction). The gate is assumed upright; its yaw in NED is
taken from the horizontal bearing of the keypoint centroid so a single gate
in view does not need a course map.

Design rule for control: steer on bearing, gate on range. Bearing depends on
the centroid pixel and attitude; range depends on apparent corner spread and
is the quantity motion blur corrupts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

import camera_model as cm
from vision.yolo_pnp import (
    GATE_INNER_M,
    KEYPOINT_COUNT,
    KEYPOINT_OBJECT_POINTS,
    MIN_CORNER_SPREAD_PX,
    MIN_KEYPOINTS_FOR_POSE,
    OUTER_RING_IDX,
)

INNER_RING_IDX = (4, 5, 6, 7)
GATE_OUTER_M = 2.7


@dataclass
class GateLSPose:
    """Camera position in the gate frame, plus diagnostics."""

    t_gate: np.ndarray          # camera origin in gate frame (m)
    range_m: float              # ||t||, distance to gate centre
    lateral_m: float            # t_x — right of centreline is +
    vertical_m: float           # t_y — below centre is + (Y down)
    through_m: float            # t_z — behind the gate plane is +
    bearing_rad: float          # atan2(lateral, through) in the horizontal plane
    residual_m: float           # mean perpendicular distance of rays to t
    keypoint_ids: tuple[int, ...]
    outer_t: Optional[np.ndarray] = None
    inner_t: Optional[np.ndarray] = None
    ring_disagree_m: float = float('nan')

    @property
    def body_forward_range(self) -> float:
        """Positive distance in front of the drone along the approach."""
        # Camera sits on the approach side of the gate plane, so through_m
        # (gate Z of the camera) is negative when looking at the opening.
        return float(max(0.0, -self.through_m))


def _usable_mask(
    pts: np.ndarray,
    conf: np.ndarray,
    min_conf: float,
) -> np.ndarray:
    return (
        np.isfinite(pts).all(axis=1)
        & np.isfinite(conf)
        & (conf >= min_conf)
        & ~np.all(pts == 0.0, axis=1)
    )


def _rays_ned(
    pts: np.ndarray,
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    """Unit bearing rays in NED for each image point."""
    R_wb = cm.rot_world_body(roll, pitch, yaw)
    R_wc = R_wb @ cm.R_BC  # cam -> NED
    rays = []
    for u, v in pts:
        ray_cam = cm.pixel_to_ray(float(u), float(v))
        ray = R_wc @ ray_cam
        n = float(np.linalg.norm(ray))
        rays.append(ray / max(n, 1e-12))
    return np.asarray(rays, dtype=np.float64)


def _gate_yaw_from_rays(rays_ned: np.ndarray) -> float:
    """Initial gate-yaw guess: heading of the mean horizontal bearing."""
    mean = rays_ned.mean(axis=0)
    horiz = np.array([mean[0], mean[1], 0.0], dtype=np.float64)
    n = float(np.linalg.norm(horiz))
    if n < 1e-9:
        return 0.0
    return float(math.atan2(horiz[1], horiz[0]))


def _refine_gate_yaw(
    corners_gate: np.ndarray,
    rays_ned: np.ndarray,
    yaw_hint: float,
    *,
    span_rad: float = math.radians(90.0),
    step_rad: float = math.radians(2.0),
) -> float:
    """Pick the gate yaw that minimises the LS ray residual.

    The centroid-ray heading is only correct on the centreline. A lateral
    offset angles the mean ray away from +Z_gate; searching residual over a
    ±span window around the hint recovers the upright-gate yaw.
    """
    def _score(yaw: float) -> float:
        R_gn = _rot_ned_from_gate(yaw).T
        rays_gate = (R_gn @ rays_ned.T).T
        norms = np.linalg.norm(rays_gate, axis=1, keepdims=True)
        rays_gate = rays_gate / np.clip(norms, 1e-12, None)
        _t, res = _ls_position(corners_gate, rays_gate)
        return float(res)

    best_yaw = float(yaw_hint)
    best_res = float('inf')
    n_steps = max(1, int(round(2.0 * span_rad / step_rad)))
    for i in range(n_steps + 1):
        yaw = yaw_hint - span_rad + i * (2.0 * span_rad / n_steps)
        res = _score(yaw)
        if res < best_res:
            best_res = res
            best_yaw = float(yaw)

    # Fine pass around the coarse winner.
    fine_span = step_rad
    fine_step = step_rad / 10.0
    n_fine = max(1, int(round(2.0 * fine_span / fine_step)))
    for i in range(n_fine + 1):
        yaw = best_yaw - fine_span + i * (2.0 * fine_span / n_fine)
        res = _score(yaw)
        if res < best_res:
            best_res = res
            best_yaw = float(yaw)
    return best_yaw


def _rot_ned_from_gate(gate_yaw: float) -> np.ndarray:
    """NED <- gate: gate X right, Y down, Z through at heading ``gate_yaw``."""
    # Gate Z in NED: horizontal through direction.
    cz = math.cos(gate_yaw)
    sz = math.sin(gate_yaw)
    z = np.array([cz, sz, 0.0])
    # Gate Y = world down.
    y = np.array([0.0, 0.0, 1.0])
    # Gate X = Y × Z (right when facing through).
    x = np.cross(y, z)
    x = x / max(float(np.linalg.norm(x)), 1e-12)
    # Re-orthogonalise Z in case of numerical drift.
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


def _ls_position(
    corners_gate: np.ndarray,
    rays_gate: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Closed-form camera position in gate frame + mean ray residual."""
    a = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    projectors = []
    for p, v in zip(corners_gate, rays_gate):
        vv = np.outer(v, v)
        proj = eye - vv
        projectors.append(proj)
        a += proj
        b += proj @ p
    try:
        t = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        t = np.linalg.lstsq(a, b, rcond=None)[0]
    # Mean perpendicular distance of each corner-ray to t.
    residuals = []
    for p, v, proj in zip(corners_gate, rays_gate, projectors):
        residuals.append(float(np.linalg.norm(proj @ (t - p))))
    return t.astype(np.float64), float(np.mean(residuals)) if residuals else float('nan')


def _pack(t: np.ndarray, residual: float, ids: tuple[int, ...]) -> GateLSPose:
    lateral = float(t[0])
    vertical = float(t[1])
    through = float(t[2])
    range_m = float(np.linalg.norm(t))
    bearing = float(math.atan2(lateral, max(-through, 1e-6)))
    return GateLSPose(
        t_gate=t,
        range_m=range_m,
        lateral_m=lateral,
        vertical_m=vertical,
        through_m=through,
        bearing_rad=bearing,
        residual_m=residual,
        keypoint_ids=ids,
    )


def solve_keypoints_ls(
    keypoints_px: Sequence[Sequence[float]],
    keypoint_confidences: Optional[Sequence[float]] = None,
    *,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    min_keypoint_confidence: float = 0.0,
    gate_yaw: Optional[float] = None,
) -> Optional[GateLSPose]:
    """LS gate pose from the eight-keypoint model plus AHRS attitude.

    Returns ``None`` when fewer than ``MIN_KEYPOINTS_FOR_POSE`` corners are
    usable or the image spread is degenerate.
    """
    pts = np.asarray(keypoints_px, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] != KEYPOINT_COUNT:
        return None
    if keypoint_confidences is None:
        conf = np.ones(KEYPOINT_COUNT, dtype=np.float64)
    else:
        conf = np.asarray(keypoint_confidences, dtype=np.float64).reshape(-1)
        if conf.shape[0] != KEYPOINT_COUNT:
            return None

    usable = _usable_mask(pts, conf, min_keypoint_confidence)
    ids = tuple(int(i) for i in np.nonzero(usable)[0])
    if len(ids) < MIN_KEYPOINTS_FOR_POSE:
        return None

    img = pts[list(ids)]
    spread = max(float(np.ptp(img[:, 0])), float(np.ptp(img[:, 1])))
    if spread < MIN_CORNER_SPREAD_PX:
        return None

    rays_ned = _rays_ned(img, roll, pitch, yaw)
    if gate_yaw is None:
        hint = _gate_yaw_from_rays(rays_ned)
        corners_all = KEYPOINT_OBJECT_POINTS[list(ids)]
        gate_yaw = _refine_gate_yaw(corners_all, rays_ned, hint)
    R_ng = _rot_ned_from_gate(gate_yaw)          # NED <- gate
    R_gn = R_ng.T                                 # gate <- NED
    rays_gate = (R_gn @ rays_ned.T).T
    # Renormalise after rotation (should already be unit).
    rays_gate = rays_gate / np.clip(
        np.linalg.norm(rays_gate, axis=1, keepdims=True), 1e-12, None
    )

    corners = KEYPOINT_OBJECT_POINTS[list(ids)]
    t, residual = _ls_position(corners, rays_gate)
    pose = _pack(t, residual, ids)

    # Per-ring solves for the consistency check (Phase 1b).
    outer_ids = tuple(i for i in ids if i in OUTER_RING_IDX)
    inner_ids = tuple(i for i in ids if i in INNER_RING_IDX)
    if len(outer_ids) >= MIN_KEYPOINTS_FOR_POSE:
        o_idx = [ids.index(i) for i in outer_ids]
        pose.outer_t, _ = _ls_position(corners[o_idx], rays_gate[o_idx])
    if len(inner_ids) >= MIN_KEYPOINTS_FOR_POSE:
        i_idx = [ids.index(i) for i in inner_ids]
        pose.inner_t, _ = _ls_position(corners[i_idx], rays_gate[i_idx])
    if pose.outer_t is not None and pose.inner_t is not None:
        pose.ring_disagree_m = float(np.linalg.norm(pose.outer_t - pose.inner_t))
    return pose


def solve_ring_ls(
    corners_px: Sequence[Sequence[float]],
    object_points: np.ndarray,
    *,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    gate_yaw: Optional[float] = None,
) -> Optional[GateLSPose]:
    """LS pose from an explicit (N,2) corner set and matching 3D points."""
    pts = np.asarray(corners_px, dtype=np.float64).reshape(-1, 2)
    obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] != obj.shape[0] or pts.shape[0] < MIN_KEYPOINTS_FOR_POSE:
        return None
    if not np.isfinite(pts).all():
        return None
    spread = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])))
    if spread < MIN_CORNER_SPREAD_PX:
        return None

    rays_ned = _rays_ned(pts, roll, pitch, yaw)
    if gate_yaw is None:
        hint = _gate_yaw_from_rays(rays_ned)
        gate_yaw = _refine_gate_yaw(obj, rays_ned, hint)
    R_gn = _rot_ned_from_gate(gate_yaw).T
    rays_gate = (R_gn @ rays_ned.T).T
    rays_gate = rays_gate / np.clip(
        np.linalg.norm(rays_gate, axis=1, keepdims=True), 1e-12, None
    )
    t, residual = _ls_position(obj, rays_gate)
    return _pack(t, residual, tuple(range(pts.shape[0])))
