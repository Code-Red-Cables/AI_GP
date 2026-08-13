"""Snake gate detection, after Li, Ozo, De Wagter and de Croon.

Implements Algorithms 1 and 2 of ``reference/paper0.pdf`` section 3.1: sample
random pixels, and where one lands on the gate colour, "snake" along that colour
up and down, then left and right from each end. Candidates that survive a
minimum-length test are squared off, their corners refined, and finally scored by
colour fitness -- the fraction of the candidate polygon's perimeter that is
actually gate-coloured.

Why this exists here: it is a *comparison* detector, run observe-only alongside
the YOLO pose model so its behaviour on this course can be measured before
anything is committed to. Nothing in the control path consumes it.

Faithfulness notes, and the places this deliberately differs:

  * The paper searches the raw distorted image on purpose, to skip undistortion.
    Same here -- the frame arrives as it comes off the wire.
  * The vertical/horizontal snake takes diagonal steps (Algorithm 2), which is
    what lets it follow an oblique bar. Kept exactly, including the diagonals.
  * The paper reports a true positive rate of 0.46 overall, rising to about 70%
    at close range. Do not expect this to beat a trained CNN; it is here because
    it is nearly free and fails in different situations, which is what makes it
    useful as a second opinion.
  * The paper detects a four-corner single-ring gate. This course's gates have an
    outer square and an inner opening, so ``KEYPOINT_COUNT`` eight-point output
    is not attempted -- four outer corners are returned and the caller decides
    what to do with them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

try:                                             # pragma: no cover
    import cv2
except ImportError:                              # pragma: no cover
    cv2 = None


@dataclass(frozen=True)
class SnakeGateConfig:
    """Thresholds from the paper, with its recommended values as defaults."""

    # Gate colour, HSV. Same calibration the rest of the pipeline uses.
    hsv_lower: tuple[int, int, int] = (0, 75, 140)
    hsv_upper: tuple[int, int, int] = (23, 255, 255)
    # sigma_L, the minimum length threshold. The paper's ROC sweep picks 25 as
    # the knee: below 15 the false positives climb, above 35 the true positive
    # rate falls off sharply.
    min_length_px: int = 25
    # sigma_cf, the colour fitness threshold.
    min_color_fitness: float = 0.35
    # maxSample in Algorithm 1.
    max_samples: int = 3000
    # Half-width of the small squares used for corner refinement.
    refine_half_px: int = 8
    # Cap on returned candidates, best colour fitness first.
    max_detections: int = 4
    # The paper's 2017 race entry kept only the single best-fitness gate.
    best_only: bool = False
    # Overlap above which two detections are treated as the same gate, keeping
    # the higher colour fitness. The paper notes that duplicate seeds on one gate
    # inflate the detection count and says they are harmless for navigation --
    # true, but they make the count useless for judging false positives, and a
    # display that draws four boxes on one gate looks like the detector is
    # broken when it is not.
    merge_iou: float = 0.3
    # DEVIATION FROM THE PAPER, and the reason it is here.
    #
    # The paper squares the snake points off with an axis-aligned "minimum
    # length" square and then scores colour fitness along that square's
    # perimeter. Measured on synthetic gates, that construction collapses the
    # moment the gate is not upright, because the square's edges stop lying on
    # the bars: fitness runs 1.00 at 0 deg, 0.34 at 5 deg, 0.19 at 10 deg and
    # 0.09 at 20 deg, so against the paper's sigma_cf of 0.35 the detector is
    # blind past roughly 5 degrees of roll. That is unusable on a course flown
    # with 45-90 deg of bank.
    #
    # With this set, the points are squared off with a minimum-area *rotated*
    # rectangle instead and fitness is scored along that. Set it False to get
    # the published behaviour for comparison.
    use_rotated_rect: bool = True


@dataclass
class SnakeGate:
    corners_px: Sequence[Sequence[float]]        # 4 corners, TL TR BR BL order
    center_px: tuple[float, float]
    color_fitness: float
    bbox_px: tuple[float, float, float, float]
    area_px: float


@dataclass
class SnakeResult:
    gates: list[SnakeGate] = field(default_factory=list)
    samples_used: int = 0
    mask_fraction: float = 0.0
    elapsed_ms: float = 0.0

    @property
    def found(self) -> bool:
        return bool(self.gates)

    @property
    def best(self) -> Optional[SnakeGate]:
        return self.gates[0] if self.gates else None


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return (inter / union) if union > 0.0 else 0.0


def _merge_overlaps(gates: list, iou_threshold: float) -> list:
    """Greedy suppression, assuming ``gates`` is already best-first."""
    if iou_threshold <= 0.0:
        return gates
    kept: list = []
    for g in gates:
        if any(_iou(g.bbox_px, k.bbox_px) > iou_threshold for k in kept):
            continue
        kept.append(g)
    return kept


class SnakeGateDetector:
    """Colour-snake gate detector. Stateless between frames, by design."""

    def __init__(self, config: Optional[SnakeGateConfig] = None):
        self.config = config or SnakeGateConfig()
        self._rng = np.random.default_rng(12345)

    # ------------------------------------------------------------------
    def color_mask(self, frame: np.ndarray) -> np.ndarray:
        """Boolean gate-colour mask. One HSV conversion per frame.

        The paper tests ``isTargetColor`` per pixel as it snakes. Doing that
        through OpenCV per pixel in Python would be hopeless, so the membership
        test is precomputed into a mask and the snake indexes it. The traversal
        is unchanged; only where the colour test's result comes from differs.
        """
        if cv2 is None:
            raise RuntimeError('snake gate detection requires cv2')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lo = np.asarray(self.config.hsv_lower, dtype=np.uint8)
        hi = np.asarray(self.config.hsv_upper, dtype=np.uint8)
        return cv2.inRange(hsv, lo, hi).astype(bool)

    # ---- Algorithm 2 --------------------------------------------------
    @staticmethod
    def _snake(mask: np.ndarray, x: int, y: int, vertical: bool) -> tuple:
        """Follow the colour in both directions along one axis.

        Returns the two extreme points. Diagonal steps are what let this track
        an oblique bar, and they are in the paper's pseudocode.
        """
        h, w = mask.shape

        def walk(sign: int) -> tuple[int, int]:
            cx, cy = x, y
            while True:
                if vertical:
                    ny = cy + sign
                    if not (0 <= ny < h):
                        return cx, cy
                    if mask[ny, cx]:
                        cy = ny
                    elif cx - 1 >= 0 and mask[ny, cx - 1]:
                        cx, cy = cx - 1, ny
                    elif cx + 1 < w and mask[ny, cx + 1]:
                        cx, cy = cx + 1, ny
                    else:
                        return cx, cy
                else:
                    nx = cx + sign
                    if not (0 <= nx < w):
                        return cx, cy
                    if mask[cy, nx]:
                        cx = nx
                    elif cy - 1 >= 0 and mask[cy - 1, nx]:
                        cx, cy = nx, cy - 1
                    elif cy + 1 < h and mask[cy + 1, nx]:
                        cx, cy = nx, cy + 1
                    else:
                        return cx, cy

        return walk(-1), walk(+1)

    # ------------------------------------------------------------------
    def _refine_corner(self, mask: np.ndarray, x: int, y: int) -> tuple:
        """Centroid of gate-coloured pixels in a small square about (x, y).

        The paper's histogram refinement, which exists because an overexposed
        bar stops the snake short of the true corner.
        """
        r = self.config.refine_half_px
        h, w = mask.shape
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        patch = mask[y0:y1, x0:x1]
        if not patch.any():
            return float(x), float(y)
        ys, xs = np.nonzero(patch)
        return float(x0 + xs.mean()), float(y0 + ys.mean())

    def _color_fitness(self, mask: np.ndarray, corners) -> float:
        """Fraction of the polygon's perimeter that is gate-coloured (eq. 1).

        Vectorised. Walking the perimeter a pixel at a time in Python cost about
        60 ms per frame on 640x360, which is slower than the camera and would
        have starved the pose model this is supposed to be compared against.
        """
        h, w = mask.shape
        xs_all = []
        ys_all = []
        n = len(corners)
        for i in range(n):
            x0, y0 = corners[i]
            x1, y1 = corners[(i + 1) % n]
            steps = int(max(abs(x1 - x0), abs(y1 - y0)))
            if steps <= 0:
                continue
            t = np.linspace(0.0, 1.0, steps + 1)
            xs_all.append(np.rint(x0 + t * (x1 - x0)))
            ys_all.append(np.rint(y0 + t * (y1 - y0)))
        if not xs_all:
            return 0.0
        xs = np.concatenate(xs_all).astype(np.int32)
        ys = np.concatenate(ys_all).astype(np.int32)
        keep = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if not keep.any():
            return 0.0
        xs, ys = xs[keep], ys[keep]
        return float(mask[ys, xs].mean())

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> SnakeResult:
        import time

        started = time.perf_counter()
        cfg = self.config
        mask = self.color_mask(frame)
        h, w = mask.shape
        result = SnakeResult(
            mask_fraction=float(mask.mean()) if mask.size else 0.0
        )
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            result.elapsed_ms = (time.perf_counter() - started) * 1000.0
            return result

        # The paper samples the whole image uniformly. Sampling only coloured
        # pixels is the same algorithm with the wasted draws skipped, and it is
        # what makes this affordable in Python.
        n_draw = min(cfg.max_samples, len(xs))
        idx = self._rng.choice(len(xs), size=n_draw, replace=False)
        found: list[SnakeGate] = []
        for k in idx:
            result.samples_used += 1
            x0, y0 = int(xs[k]), int(ys[k])

            p1, p2 = self._snake(mask, x0, y0, vertical=True)
            if abs(p2[1] - p1[1]) < cfg.min_length_px:
                continue
            p3a, p3b = self._snake(mask, p1[0], p1[1], vertical=False)
            p4a, p4b = self._snake(mask, p2[0], p2[1], vertical=False)
            span_top = abs(p3b[0] - p3a[0])
            span_bot = abs(p4b[0] - p4a[0])
            if span_top < cfg.min_length_px and span_bot < cfg.min_length_px:
                continue

            pts = np.array([p1, p2, p3a, p3b, p4a, p4b], dtype=float)
            xmin, ymin = pts[:, 0].min(), pts[:, 1].min()
            xmax, ymax = pts[:, 0].max(), pts[:, 1].max()
            if (xmax - xmin) < cfg.min_length_px:
                continue

            # Deduplicate before refining. Many seeds land on one gate, and
            # corner refinement plus the colour-fitness walk are the expensive
            # part; rejecting a duplicate here rather than after costs an order
            # of magnitude less per frame.
            box = (float(xmin), float(ymin), float(xmax), float(ymax))
            if any(_iou(box, g.bbox_px) > cfg.merge_iou for g in found):
                continue

            if cfg.use_rotated_rect:
                rect = cv2.minAreaRect(pts.astype(np.float32))
                square = [tuple(p) for p in cv2.boxPoints(rect)]
            else:
                square = [
                    (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax),
                ]
            refined = [
                self._refine_corner(mask, int(round(cx)), int(round(cy)))
                for cx, cy in square
            ]
            fitness = self._color_fitness(mask, refined)
            if fitness < cfg.min_color_fitness:
                continue

            arr = np.asarray(refined, dtype=float)
            found.append(SnakeGate(
                corners_px=[[float(a), float(b)] for a, b in refined],
                center_px=(float(arr[:, 0].mean()), float(arr[:, 1].mean())),
                color_fitness=float(fitness),
                bbox_px=(float(xmin), float(ymin), float(xmax), float(ymax)),
                area_px=float((xmax - xmin) * (ymax - ymin)),
            ))
            # Algorithm 1 returns on its first accepted gate. Keeping a few more
            # is useful for comparing against YOLO's candidate list, but running
            # every sample to exhaustion is what made this slower than the
            # camera. Duplicates are already suppressed above, so this quota
            # counts distinct gates.
            if len(found) >= cfg.max_detections:
                break

        found.sort(key=lambda g: -g.color_fitness)
        found = _merge_overlaps(found, cfg.merge_iou)
        if cfg.best_only:
            found = found[:1]
        result.gates = found[: cfg.max_detections]
        result.elapsed_ms = (time.perf_counter() - started) * 1000.0
        return result
