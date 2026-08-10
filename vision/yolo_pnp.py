"""YOLO gate corners -> PnP pose (the perception core of the VIO pipeline).

The live pose model labels eight coplanar corners — the 2.7 m outer square and
the 1.5 m flyable opening, each clockwise from top-left — and
``solve_keypoints_pnp`` turns whatever subset is visible into a gate pose.
Four good points are enough, so a gate half out of frame still solves. The
legacy four-outer-corner path (``solve_corners_pnp``) remains for tools that
still hand in a TL/TR/BL/BR quad.

With the pinhole intrinsics from spec 3.8 (fx=fy=320, cx=320, cy=180, no
distortion) a single gate view is a textbook planar PnP problem: IPPE (or
SQPNP as a fallback) returns the gate's pose in the camera-optical frame.

From that one solve we get, at up to 30 Hz:
  * range + bearing to the gate (far better than the HSV size-method estimate)
  * the drone's ABSOLUTE roll & pitch (gates hang upright, so the gate's
    vertical axis is the world's gravity axis)
  * the drone's yaw + position relative to the gate plane (which the state
    estimator anchors into the world frame)

Gate frame convention used here:
  X = right along the gate plane (approach view), Y = down (gravity),
  Z = X x Y = the gate normal pointing AWAY from the approaching camera —
    i.e. the direction of flight THROUGH the gate.
The origin is the gate centre. solvePnP maps X_cam = R_cg @ X_gate + t_cg.

Course constraint: every race gate is right-side up (never inverted). IPPE's
planar ambiguity is resolved by keeping only solutions whose gate +Y aligns
with camera-down (R[1,1] > 0) and whose +Z points into the scene (R[2,2] > 0).

On Q2_CV the live path does NOT run a second YOLO inference: VisionRX reuses
the keypoints already produced by YoloPoseGateDetector and calls
``solve_keypoints_pnp`` via ``dual_gate_pnp``. ``YoloGatePnP`` remains for
offline tools and standalone use.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np

import camera_model as cm
import config

GATE_OUTER_M = 2.7          # outer square side (spec 3.7) — the labeled corners
_HALF = GATE_OUTER_M / 2.0

# Corner order for solvePnP: TL, TR, BL, BR (gate X right, Y down).
# NOTE: vision.gate_detector.order_corners returns TL, TR, BR, BL (cyclic).
# Feeding that order here rejects the solve or flips the pose — always run
# corners through ``corners_tl_tr_bl_br`` before OBJECT_POINTS.
OBJECT_POINTS = np.array([
    [-_HALF, -_HALF, 0.0],   # TL
    [+_HALF, -_HALF, 0.0],   # TR
    [-_HALF, +_HALF, 0.0],   # BL
    [+_HALF, +_HALF, 0.0],   # BR
], dtype=np.float64)

# --- eight-keypoint model -------------------------------------------------
# The pose model now labels both rings: the 2.7 m outer square and the 1.5 m
# flyable opening. Index order follows the training set exactly — each ring
# clockwise from top-left — so these are indexed by keypoint id, NOT reordered
# geometrically the way the four-corner path has to be.
GATE_INNER_M = 1.5
_HALF_IN = GATE_INNER_M / 2.0
KEYPOINT_OBJECT_POINTS = np.array([
    [-_HALF,    -_HALF,    0.0],   # 0 outer TL
    [+_HALF,    -_HALF,    0.0],   # 1 outer TR
    [+_HALF,    +_HALF,    0.0],   # 2 outer BR
    [-_HALF,    +_HALF,    0.0],   # 3 outer BL
    [-_HALF_IN, -_HALF_IN, 0.0],   # 4 inner TL
    [+_HALF_IN, -_HALF_IN, 0.0],   # 5 inner TR
    [+_HALF_IN, +_HALF_IN, 0.0],   # 6 inner BR
    [-_HALF_IN, +_HALF_IN, 0.0],   # 7 inner BL
], dtype=np.float64)
KEYPOINT_COUNT = len(KEYPOINT_OBJECT_POINTS)
OUTER_RING_IDX = (0, 1, 2, 3)
# Planar PnP needs four points; gates leave frame constantly, so a partial
# ring is the normal case rather than the exception.
MIN_KEYPOINTS_FOR_POSE = 4


def corners_tl_tr_bl_br(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Canonicalise four image corners to TL, TR, BL, BR for OBJECT_POINTS.

    YOLO keypoint indices are *supposed* to be TL,TR,BL,BR, but near edges /
    partial views they swap. Geometric cyclic order is TL,TR,BR,BL — we convert
    that to the BL/BR layout solvePnP expects.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    cyclic = pts[np.argsort(angles, kind='stable')]
    # cyclic[0] after roll = top-left-most (min x+y).
    start = int(np.argmin(cyclic.sum(axis=1)))
    tl_tr_br_bl = np.roll(cyclic, -start, axis=0)
    # TL, TR, BR, BL → TL, TR, BL, BR
    return np.stack(
        [tl_tr_br_bl[0], tl_tr_br_bl[1], tl_tr_br_bl[3], tl_tr_br_bl[2]],
        axis=0,
    ).astype(np.float64)

# Solved with generic SOLVEPNP_IPPE (planar target). IPPE_SQUARE was tried first
# but it requires the exact ArUco object layout (Y-up, TL,TR,BR,BL) — fed our
# Y-down points it returned garbage (0.2 m range on a 10 m gate, 8000 px reproj).

# Confidence floor 0.35 -> 0.50 (2026-07-27, runs 4/5): station signs/markers
# detect as "gates" at conf 0.35-0.45 and, being NEARER than the real gate,
# were picked as the primary target — the VIO anchored gate 0 on a SIGN and the
# whole world frame was born wrong (z sign flipped run-to-run). Real gates
# score 0.8-0.9 even far out.
# 0.50 -> 0.45: partial/edge-clipped views of REAL gates during the approach
# score below 0.5 and were dropped, starving the VIO of fixes exactly when the
# gate nears the frame edge. Junk detections sit at <=0.42 measured, so 0.45
# still excludes them while keeping degraded-but-real views.
DEFAULT_CONF = 0.45          # YOLO box confidence floor
MIN_CORNER_SPREAD_PX = 12.0  # degenerate/far quads solve badly; skip below this
MAX_RANGE_M = 45.0           # beyond this the corner pixels are sub-pixel noise
# Discard solves that cannot re-draw their corners. Validation on the dataset:
# good solves sit at 0.3-1.0 px; the flipped planar-ambiguity solutions show up
# at 3-4.3 px with wild attitudes (roll +39deg) — 2.0 px separates them cleanly.
REPROJ_ERR_MAX_PX = 2.0


@dataclass
class GatePnP:
    corners_px: np.ndarray        # (4,2) TL,TR,BL,BR
    confidence: float
    bbox: tuple                   # (x1, y1, x2, y2) px
    # PnP solution (None if the solve failed / was rejected)
    R_cg: Optional[np.ndarray] = None   # gate -> camera-optical rotation
    t_cg: Optional[np.ndarray] = None   # gate origin in camera-optical frame (m)
    reproj_err_px: float = float("inf")
    # Keypoints that actually fed the solve. ``corners_px`` stays the four
    # outer corners whatever the model produced, so every existing consumer
    # keeps the shape it expects.
    keypoints_px: Optional[np.ndarray] = None    # (K,2) by keypoint id
    keypoint_ids: tuple = ()                     # ids used, in solve order

    @property
    def solved(self) -> bool:
        return self.R_cg is not None

    @property
    def points_used(self) -> int:
        return len(self.keypoint_ids) if self.keypoint_ids else 4

    @property
    def range_m(self) -> float:
        return float(np.linalg.norm(self.t_cg)) if self.solved else float("inf")

    def center_body(self) -> np.ndarray:
        """Gate centre as a body-frame vector (x fwd, y right, z down)."""
        return cm.cam_to_body(self.t_cg)

    def cam_pose_in_gate(self) -> tuple[np.ndarray, np.ndarray]:
        """(R_gc, p_g): camera rotation and position expressed in the gate frame."""
        R_gc = self.R_cg.T
        p_g = -R_gc @ self.t_cg
        return R_gc, p_g

    def rvec_tvec(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """OpenCV rvec/tvec for drawFrameAxes (camera ← gate)."""
        if not self.solved:
            return None
        rvec, _ = cv2.Rodrigues(np.asarray(self.R_cg, dtype=np.float64))
        tvec = np.asarray(self.t_cg, dtype=np.float64).reshape(3, 1)
        return rvec, tvec


def draw_gate_frame_axes(
    image: np.ndarray,
    gate: GatePnP,
    *,
    axis_length_m: float = 1.35,
    thickness: int = 2,
    label: Optional[str] = None,
) -> None:
    """Project gate XYZ onto the image (X right, Y down, Z through).

    Length is in gate-plane metres. Grow with range so a pad view (~20 m)
    still shows a readable triad (old 1.2 m cap was ~20 px at that distance).
    Cap at the outer half-diagonal so axes stay inside the gate square.
    """
    pose = gate.rvec_tvec()
    if pose is None:
        return
    rvec, tvec = pose
    # ~half outer side by default; stretch toward full outer at long range.
    range_scale = float(np.clip(gate.range_m / 8.0, 1.0, 2.2))
    length = float(
        np.clip(axis_length_m * range_scale, 0.8, GATE_OUTER_M)
    )
    cv2.drawFrameAxes(image, cm.K, None, rvec, tvec, length, thickness)
    if label:
        origin, _ = cv2.projectPoints(
            np.zeros((1, 3), dtype=np.float64),
            rvec,
            tvec,
            cm.K,
            None,
        )
        u, v = origin.reshape(2)
        if np.isfinite(u) and np.isfinite(v):
            cv2.putText(
                image,
                label,
                (int(u) + 8, int(v) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )


def solve_keypoints_pnp(
    keypoints_px: Sequence[Sequence[float]],
    keypoint_confidences: Optional[Sequence[float]] = None,
    confidence: float = 1.0,
    bbox: Optional[tuple] = None,
    min_keypoint_confidence: float = 0.0,
) -> Optional[GatePnP]:
    """PnP-solve the eight-keypoint gate model from whatever is visible.

    Points are indexed by keypoint id (0-3 outer ring, 4-7 inner ring, each
    clockwise from top-left) and are NOT reordered — the model is trained to
    hold that identity, which is what lets a partial ring still solve. Any
    four good points are enough, so a gate half out of frame keeps producing
    a pose instead of dropping out.

    A (4,2) input is treated as the four outer corners and routed to
    ``solve_corners_pnp`` so the long-standing four-corner path is untouched.
    """
    pts = np.asarray(keypoints_px, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 4:
        return solve_corners_pnp(pts, confidence=confidence, bbox=bbox)
    if pts.shape[0] != KEYPOINT_COUNT:
        return None

    if keypoint_confidences is None:
        conf = np.ones(KEYPOINT_COUNT, dtype=np.float64)
    else:
        conf = np.asarray(
            keypoint_confidences, dtype=np.float64
        ).reshape(-1)
        if conf.shape[0] != KEYPOINT_COUNT:
            return None
    # An unseen corner comes back as (0,0) with no confidence; taking it at
    # face value would drag the pose to a corner of the image.
    usable = (
        np.isfinite(pts).all(axis=1)
        & np.isfinite(conf)
        & (conf >= min_keypoint_confidence)
        & ~np.all(pts == 0.0, axis=1)
    )
    ids = tuple(int(i) for i in np.nonzero(usable)[0])
    if len(ids) < MIN_KEYPOINTS_FOR_POSE:
        return None

    img_pts = pts[list(ids)]
    spread = max(np.ptp(img_pts[:, 0]), np.ptp(img_pts[:, 1]))
    if spread < MIN_CORNER_SPREAD_PX:
        return None
    solution = _solve_planar(KEYPOINT_OBJECT_POINTS[list(ids)], img_pts)
    if solution is None:
        return None
    R_cg, t_cg, err = solution

    # Hand downstream the four outer corners it has always been given, taken
    # from the fitted pose so the shape holds even when a corner was missing.
    rvec, _ = cv2.Rodrigues(R_cg)
    outer, _ = cv2.projectPoints(
        KEYPOINT_OBJECT_POINTS[list(OUTER_RING_IDX)],
        rvec,
        t_cg.reshape(3, 1),
        cm.K,
        None,
    )
    outer = outer.reshape(-1, 2)
    # OBJECT_POINTS order is TL, TR, BL, BR; the ring is TL, TR, BR, BL.
    corners = np.stack([outer[0], outer[1], outer[3], outer[2]], axis=0)
    if bbox is None:
        x1, y1 = img_pts.min(axis=0)
        x2, y2 = img_pts.max(axis=0)
        bbox = (float(x1), float(y1), float(x2), float(y2))
    return GatePnP(
        corners_px=corners,
        confidence=float(confidence),
        bbox=tuple(bbox),
        R_cg=R_cg,
        t_cg=t_cg,
        reproj_err_px=err,
        keypoints_px=pts,
        keypoint_ids=ids,
    )


def solve_corners_pnp(
    corners_px: Sequence[Sequence[float]],
    confidence: float = 1.0,
    bbox: Optional[tuple] = None,
) -> Optional[GatePnP]:
    """PnP-solve a (4,2) TL/TR/BL/BR corner set without a second YOLO pass.

    Returns a solved ``GatePnP`` or ``None`` when the quad is degenerate,
    out of range, or fails the reprojection-error gate. Used by VisionRX to
    feed the VIO from the live pose detector's existing keypoints.
    """
    corners = np.asarray(corners_px, dtype=np.float64).reshape(-1, 2)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        return None
    if bbox is None:
        x1, y1 = corners.min(axis=0)
        x2, y2 = corners.max(axis=0)
        bbox = (float(x1), float(y1), float(x2), float(y2))
    gate = GatePnP(
        corners_px=corners,
        confidence=float(confidence),
        bbox=tuple(bbox),
    )
    _solve_gate(gate)
    return gate if gate.solved else None


def _is_upright_through_solution(R: np.ndarray) -> bool:
    """True if gate +Y ≈ camera-down and +Z ≈ into the scene.

    All race gates hang right-side up — never accept an inverted solution.
    """
    # Gate +Y expressed in camera: R[:, 1]. Camera +Y is down.
    gate_y_along_cam_down = float(R[1, 1])
    # Gate +Z (through) along camera forward.
    gate_z_into_scene = float(R[2, 2])
    return gate_y_along_cam_down > 0.25 and gate_z_into_scene > 0.0


def _solve_gate(g: GatePnP) -> None:
    # Always re-order — do not trust YOLO keypoint index order alone.
    c = corners_tl_tr_bl_br(g.corners_px)
    g.corners_px = c
    # Image sanity: top row must sit above bottom row (upright gate in pixels).
    if float(np.mean(c[0:2, 1])) >= float(np.mean(c[2:4, 1])):
        return
    spread = max(np.ptp(c[:, 0]), np.ptp(c[:, 1]))
    if spread < MIN_CORNER_SPREAD_PX:
        return
    solution = _solve_planar(OBJECT_POINTS, c)
    if solution is None:
        return
    g.R_cg, g.t_cg, g.reproj_err_px = solution


def _solve_planar(obj_pts: np.ndarray, img_pts: np.ndarray):
    """Planar PnP over matched object/image points, ambiguity resolved.

    IPPE returns two mirrored solutions for a planar target. Keep only the one
    that stands the gate upright and puts it in front of the camera, then the
    lower reprojection error of whatever survives.
    """
    obj = np.ascontiguousarray(obj_pts.reshape(-1, 1, 3), dtype=np.float64)
    img = np.ascontiguousarray(img_pts.reshape(-1, 1, 2), dtype=np.float64)

    def _attempt(flag):
        try:
            ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                obj, img, cm.K, None, flags=flag
            )
        except cv2.error:
            return []
        if not ok:
            return []
        # IPPE fits a homography when given more than four points and hands
        # back NaN on a near fronto-parallel gate, which is exactly the view
        # on a straight approach.
        return [
            (rv, tv)
            for rv, tv in zip(rvecs, tvecs)
            if np.isfinite(rv).all() and np.isfinite(tv).all()
        ]

    pairs = _attempt(cv2.SOLVEPNP_IPPE)
    if not pairs:
        # SQPNP is stable on the degenerate views but returns a single
        # solution, so it is a fallback rather than the default: IPPE's pair
        # is what lets the upright test throw out the mirrored pose.
        pairs = _attempt(cv2.SOLVEPNP_SQPNP)
    if not pairs:
        return None
    rvecs = [rv for rv, _ in pairs]
    tvecs = [tv for _, tv in pairs]

    best = None
    best_err = REPROJ_ERR_MAX_PX
    for rv, tv in zip(rvecs, tvecs):
        tt = np.asarray(tv, dtype=np.float64).ravel()
        if not (0.5 < float(tt[2]) < MAX_RANGE_M):
            continue
        RR, _ = cv2.Rodrigues(rv)
        if not _is_upright_through_solution(RR):
            continue
        proj, _ = cv2.projectPoints(obj, rv, tv, cm.K, None)
        err = float(
            np.linalg.norm(
                proj.reshape(-1, 2) - img.reshape(-1, 2), axis=1
            ).mean()
        )
        if err < best_err:
            best_err = err
            best = (RR, tt, err)
    return best


class YoloGatePnP:
    """Wraps the trained YOLO-pose model + per-gate PnP solves."""

    def __init__(
        self,
        weights: Optional[str] = None,
        conf: float = DEFAULT_CONF,
        device: Optional[str] = None,
    ):
        from ultralytics import YOLO   # lazy: heavy import
        if weights is None:
            weights = config.YOLO_POSE_MODEL_PATH
        self.model = YOLO(weights)
        self.conf = conf
        self.device = device
        # one warmup inference so the first live frame isn't a 2 s stall
        self.model.predict(np.zeros((360, 640, 3), dtype=np.uint8),
                           verbose=False, conf=conf, device=device)

    # ------------------------------------------------------------------
    def detect(self, img_bgr: np.ndarray) -> List[GatePnP]:
        """All gates in the frame (sorted best-first), PnP-solved where possible."""
        res = self.model.predict(img_bgr, verbose=False, conf=self.conf,
                                 device=self.device)[0]
        gates: List[GatePnP] = []
        if res.boxes is None or len(res.boxes) == 0:
            return gates
        kps = res.keypoints.xy.cpu().numpy()          # (N, 4|8, 2)
        kconfs = (
            res.keypoints.conf.cpu().numpy()
            if res.keypoints.conf is not None
            else None
        )
        confs = res.boxes.conf.cpu().numpy()
        boxes = res.boxes.xyxy.cpu().numpy()
        for idx, (corners, conf, box) in enumerate(zip(kps, confs, boxes)):
            point_conf = None if kconfs is None else kconfs[idx]
            g = solve_keypoints_pnp(
                corners.astype(np.float64),
                keypoint_confidences=point_conf,
                confidence=float(conf),
                bbox=tuple(box.tolist()),
            )
            if g is None:
                # Keep an unsolved placeholder so callers still see the box.
                g = GatePnP(
                    corners_px=np.asarray(corners[:4], dtype=np.float64),
                    confidence=float(conf),
                    bbox=tuple(box.tolist()),
                    keypoints_px=np.asarray(corners, dtype=np.float64),
                )
            gates.append(g)
        # Primary target = the gate we are actually flying at: rank by
        # confidence x apparent size, NOT by range — ranking by range made a
        # nearer spurious detection (a station sign) the primary and poisoned
        # the VIO anchor (see DEFAULT_CONF note).
        def score(g: GatePnP) -> float:
            x1, y1, x2, y2 = g.bbox
            return g.confidence * math.sqrt(max(1.0, (x2 - x1) * (y2 - y1)))
        gates.sort(key=lambda g: (not g.solved, -score(g)))
        return gates
