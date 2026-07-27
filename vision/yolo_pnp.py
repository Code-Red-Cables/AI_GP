"""YOLO gate corners -> PnP pose (the perception core of the VIO pipeline).

A YOLOv8n-pose model (trained on 256 sim frames, box mAP50 0.925 / corner mAP50
0.959) finds every gate in frame with its 4 OUTER corners in TL, TR, BL, BR
order. The outer gate square is 2.7 m x 2.7 m (spec 3.7); with the pinhole
intrinsics from spec 3.8 (fx=fy=320, cx=320, cy=180, no distortion) a single
gate view is a textbook planar-square PnP problem: cv2.solvePnP(IPPE_SQUARE)
returns the gate's pose in the camera-optical frame.

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
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

import camera_model as cm

GATE_OUTER_M = 2.7          # outer square side (spec 3.7) — the labeled corners
_HALF = GATE_OUTER_M / 2.0

# Corner order matches the dataset labels: TL, TR, BL, BR (X right, Y down).
OBJECT_POINTS = np.array([
    [-_HALF, -_HALF, 0.0],   # TL
    [+_HALF, -_HALF, 0.0],   # TR
    [-_HALF, +_HALF, 0.0],   # BL
    [+_HALF, +_HALF, 0.0],   # BR
], dtype=np.float64)

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

    @property
    def solved(self) -> bool:
        return self.R_cg is not None

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


class YoloGatePnP:
    """Wraps the trained YOLOv8n-pose model + per-gate PnP solves."""

    def __init__(self, weights: str = "yolo26n.pt", conf: float = DEFAULT_CONF,
                 device: Optional[str] = None):
        from ultralytics import YOLO   # lazy: heavy import
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
        kps = res.keypoints.xy.cpu().numpy()          # (N, 4, 2)
        confs = res.boxes.conf.cpu().numpy()
        boxes = res.boxes.xyxy.cpu().numpy()
        for corners, conf, box in zip(kps, confs, boxes):
            g = GatePnP(corners_px=corners.astype(np.float64),
                        confidence=float(conf), bbox=tuple(box.tolist()))
            self._solve(g)
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

    # ------------------------------------------------------------------
    def _solve(self, g: GatePnP) -> None:
        c = g.corners_px
        spread = max(np.ptp(c[:, 0]), np.ptp(c[:, 1]))
        if spread < MIN_CORNER_SPREAD_PX:
            return
        img_pts = c.reshape(-1, 1, 2)
        try:
            ok, rvec, tvec = cv2.solvePnP(
                OBJECT_POINTS.reshape(-1, 1, 3), img_pts, cm.K, None,
                flags=cv2.SOLVEPNP_IPPE)
        except cv2.error:
            return
        if not ok:
            return
        t = tvec.ravel()
        if not (0.5 < t[2] < MAX_RANGE_M):      # gate must be in front, in range
            return
        proj, _ = cv2.projectPoints(OBJECT_POINTS, rvec, tvec, cm.K, None)
        err = float(np.linalg.norm(proj.reshape(-1, 2) - img_pts.reshape(-1, 2),
                                   axis=1).mean())
        if err > REPROJ_ERR_MAX_PX:
            return
        R, _ = cv2.Rodrigues(rvec)
        g.R_cg, g.t_cg, g.reproj_err_px = R, t, err
