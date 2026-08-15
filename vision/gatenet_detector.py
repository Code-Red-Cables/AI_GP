"""GateNet: four inner-aperture corners, from the gatenet_handoff bundle.

The friend's model predicts the 1.5 m flyable opening as TL, TR, BR, BL. That
is the same order as this project's inner ring (keypoints 4-7, clockwise from
top-left), so the mapping is an index shift, not a geometric guess.

Two ways to run it:

  * Observe-only (``GATENET_ENABLED=1``). YOLO still feeds the policy;
    the panel shows GateNet only so the two models are not mixed on one frame.
  * ``GATE_DETECTOR_BACKEND=gatenet``. GateNet *is* the detector. Outer-ring
    keypoints stay unseen; inner 4-7 carry the four corners that cleared the
    score threshold.

Gate on the weakest per-corner peak score, never on the collapsed confidence
head — that is the author's own instruction in gatenet_handoff/README.md.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from vision.gate_detector import DetectorDebug, GateDetection
from vision.yolo_gate_detector import YoloGateBox
from vision.yolo_pose_gate_detector import PoseGateCandidate, PoseGateDebug

HANDOFF_DIR = Path(__file__).resolve().parent.parent / 'gatenet_handoff'
DEFAULT_ONNX = HANDOFF_DIR / 'gatenet.onnx'
NET_SIZE = (320, 180)
CORNER_NAMES = ('TL', 'TR', 'BR', 'BL')
# race_obs inner ring: 4=TL, 5=TR, 6=BR, 7=BL. Same clockwise-from-TL order.
INNER_KP_OFFSET = 4


def _load_decode():
    root = str(HANDOFF_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)
    from aigp.perception.cnn_decode import decode_heatmaps, preprocess
    return decode_heatmaps, preprocess


def inner_keypoints_from_corners(
    corners_px: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
) -> tuple[list[list[float]], list[float]]:
    """Eight-keypoint row: outer unseen, inner 4-7 from GateNet.

    Corners below ``threshold`` are left at (0, 0) with confidence 0 so the
    observation builder treats them as not-seen rather than as a corner at the
    origin.
    """
    kps = [[0.0, 0.0] for _ in range(8)]
    conf = [0.0] * 8
    pts = np.asarray(corners_px, dtype=float).reshape(4, 2)
    sc = np.asarray(scores, dtype=float).reshape(4)
    for i in range(4):
        if sc[i] >= threshold:
            kps[INNER_KP_OFFSET + i] = [float(pts[i, 0]), float(pts[i, 1])]
            conf[INNER_KP_OFFSET + i] = float(sc[i])
    return kps, conf


@dataclass
class GateNetResult:
    corners_px: np.ndarray          # (4, 2) TL TR BR BL, source-image pixels
    scores: np.ndarray              # (4,) per-corner peak scores
    n_seen: int
    elapsed_ms: float
    found: bool
    threshold: float = 0.45

    @property
    def _seen_mask(self) -> np.ndarray:
        return self.scores >= self.threshold

    @property
    def center_px(self) -> tuple[float, float]:
        mask = self._seen_mask
        pts = self.corners_px[mask] if mask.any() else self.corners_px
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())

    @property
    def bbox_px(self) -> tuple[float, float, float, float]:
        mask = self._seen_mask
        pts = self.corners_px[mask] if mask.any() else self.corners_px
        return (
            float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()),
        )


class GateNetDetector:
    """ONNX GateNet, same detect() shape as YoloPoseGateDetector."""

    def __init__(
        self,
        *,
        model_path: Optional[str] = None,
        score_threshold: float = 0.45,
        min_corners: int = 2,
        providers: Optional[list[str]] = None,
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_ONNX
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f'GateNet weights missing at {self.model_path}'
            )
        self.score_threshold = float(score_threshold)
        self.min_corners = int(min_corners)
        self._decode, self._preprocess = _load_decode()
        import onnxruntime as ort

        wanted = providers or [
            'CUDAExecutionProvider', 'CPUExecutionProvider',
        ]
        available = set(ort.get_available_providers())
        use = [p for p in wanted if p in available] or ['CPUExecutionProvider']
        self._sess = ort.InferenceSession(str(self.model_path), providers=use)
        self._out_names = [o.name for o in self._sess.get_outputs()]
        self._provider = self._sess.get_providers()[0]
        self.last_pose_debug: Optional[PoseGateDebug] = None
        self.last_result: Optional[GateNetResult] = None
        self.last_debug: Optional[DetectorDebug] = None

    def infer(self, frame_bgr: np.ndarray) -> GateNetResult:
        started = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        blob = self._preprocess(frame_bgr, NET_SIZE)
        hm, off, _conf = self._sess.run(
            self._out_names, {'image': blob},
        )
        corners_n, scores = self._decode(hm, off, NET_SIZE)
        corners_px = corners_n[0] * np.array([w, h], dtype=np.float32)
        sc = scores[0]
        n_seen = int((sc >= self.score_threshold).sum())
        res = GateNetResult(
            corners_px=corners_px.astype(np.float32),
            scores=sc.astype(np.float32),
            n_seen=n_seen,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            found=n_seen >= self.min_corners,
            threshold=self.score_threshold,
        )
        self.last_result = res
        empty = np.zeros((1, 1), dtype=np.uint8)
        self.last_debug = DetectorDebug(
            empty, empty,
            timings_ms={'gatenet': res.elapsed_ms, 'total': res.elapsed_ms},
        )
        return res

    def detect(self, frame, hint=None, timestamp=None) -> GateDetection:
        del hint
        h, w = frame.shape[:2]
        res = self.infer(frame)
        kps, kconf = inner_keypoints_from_corners(
            res.corners_px, res.scores, threshold=self.score_threshold,
        )
        xs = [p[0] for p, c in zip(kps, kconf) if c > 0.0]
        ys = [p[1] for p, c in zip(kps, kconf) if c > 0.0]
        if xs:
            bbox = (
                int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)),
            )
            cx, cy = float(np.mean(xs)), float(np.mean(ys))
            area = float(max(1.0, (max(xs) - min(xs)) * (max(ys) - min(ys))))
        else:
            bbox = (0, 0, 0, 0)
            cx = cy = 0.0
            area = 0.0

        seen_pts = np.asarray(
            [p for p, c in zip(kps, kconf) if c > 0.0], dtype=np.float32,
        )
        corners = seen_pts if len(seen_pts) else None

        box = YoloGateBox(
            bbox=(float(bbox[0]), float(bbox[1]),
                  float(bbox[2]), float(bbox[3])),
            confidence=float(res.scores.max()) if len(res.scores) else 0.0,
        )
        candidate = PoseGateCandidate(
            box=box,
            keypoints=np.asarray(kps, dtype=np.float32),
            keypoint_confidences=np.asarray(kconf, dtype=np.float32),
        )
        self.last_pose_debug = PoseGateDebug(
            candidates=[candidate],
            selected=candidate if res.found else None,
            center=(cx, cy) if res.found else None,
            center_source='gatenet_inner',
            keypoint_reason='gatenet' if res.found else 'below_threshold',
        )

        conf = float(res.scores.min()) if res.n_seen == 4 else (
            float(res.scores[res.scores >= self.score_threshold].min())
            if res.n_seen else 0.0
        )
        return GateDetection(
            found=bool(res.found),
            center_x=cx,
            center_y=cy,
            apparent_area=area,
            confidence=conf,
            corners=corners,
            method='gatenet_inner',
            corners_reliable=res.n_seen == 4,
            frame_width=w,
            frame_height=h,
            bbox=bbox,
            timestamp=timestamp,
        )

    def set_prefer_horizontal(self, nx) -> None:
        del nx

    def reset_episode(self) -> None:
        self.last_pose_debug = None
        self.last_result = None
        self.last_debug = None
