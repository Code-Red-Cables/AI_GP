"""Gate detection: HSV threshold -> contour -> square (see PLAN.md 8.2).

Round One is a high-contrast, *desaturated* environment, so the flyable gate is
the salient saturated/bright object. The detector masks that color, finds the
gate's square frame (a square annulus), and reports the inner opening's center,
corners, bounding box, area and a heuristic confidence.

`detect_gate()` is pure (frame in -> detection out, no disk writes). The optional
`draw_detection()` helper and the `__main__` self-test handle any visualization.
"""

import sys
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# HSV thresholds.
#
# Configured for glowing red/orange gates using a two-piece mask to handle
# the OpenCV hue wrap-around at 180/0.
# --------------------------------------------------------------------------- #
LOWER_HSV = (0, 0, 80)
UPPER_HSV = (15, 255, 255)
LOWER_HSV2 = (170, 0, 80) 
UPPER_HSV2 = (180, 255, 255)

# Default tuning knobs. Any of these (incl. the HSV thresholds above) may be
# overridden per-call via the `cfg` dict so callers/tests can pass their own.
DEFAULT_CFG = {
    "lower_hsv": LOWER_HSV,
    "upper_hsv": UPPER_HSV,
    "lower_hsv2": LOWER_HSV2,
    "upper_hsv2": UPPER_HSV2,
    "kernel_size": 5,        # morphology kernel (odd, small)
    "min_area": 150.0,       # area floor in px for a candidate contour
    "min_extent": 0.15,      # contour_area / bbox_area floor
    "approx_eps_frac": 0.06, # approxPolyDP epsilon as fraction of perimeter
}


@dataclass
class GateDetection:
    center_px: tuple[float, float]        # (u, v)
    corners_px: list[tuple[float, float]] | None   # TL, TR, BR, BL or None
    bbox_px: tuple[float, float, float, float]     # x, y, w, h
    area_px: float
    confidence: float                     # 0..1 heuristic


def _make_cfg(cfg: Optional[dict]) -> dict:
    """Merge a caller-supplied override dict over the module defaults."""
    merged = dict(DEFAULT_CFG)
    if cfg:
        merged.update(cfg)
    return merged


def _build_mask(hsv: np.ndarray, cfg: dict) -> np.ndarray:
    """inRange (+ optional wrapped-hue range) then open/close morphology."""
    lower = np.array(cfg["lower_hsv"], dtype=np.uint8)
    upper = np.array(cfg["upper_hsv"], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    if cfg["lower_hsv2"] is not None and cfg["upper_hsv2"] is not None:
        lower2 = np.array(cfg["lower_hsv2"], dtype=np.uint8)
        upper2 = np.array(cfg["upper_hsv2"], dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower2, upper2))

    k = max(1, int(cfg["kernel_size"]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _squareness(w: float, h: float) -> float:
    """1.0 for a perfect square bbox, decaying toward 0 as the aspect departs."""
    if w <= 0 or h <= 0:
        return 0.0
    aspect = min(w, h) / max(w, h)  # 0..1, 1 == square
    return aspect


def _centroid(contour: np.ndarray) -> Optional[tuple]:
    m = cv2.moments(contour)
    if m["m00"] <= 0:
        return None
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def _order_corners(pts: np.ndarray) -> list:
    """Order 4 points as TL, TR, BR, BL using the sum/diff trick."""
    pts = pts.reshape(-1, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return [tuple(tl), tuple(tr), tuple(br), tuple(bl)]


def detect_gates(bgr: np.ndarray, cfg: Optional[dict] = None) -> list[GateDetection]:
    """All plausible gates in the frame, best first. [] when none.
    Replaces single-detection detect_gate(); a thin detect_gate() wrapper
    returning the first element is kept for the old tests."""
    if bgr is None or bgr.size == 0:
        return []

    cfg = _make_cfg(cfg)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = _build_mask(hsv, cfg)

    # RETR_CCOMP gives a 2-level hierarchy: outer boundaries + their holes, so we
    # can recover the gate's inner opening contour.
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours or hierarchy is None:
        return []
    hierarchy = hierarchy[0]  # shape (N, 4): [next, prev, first_child, parent]

    min_area = float(cfg["min_area"])
    min_extent = float(cfg["min_extent"])

    candidates = []
    for i, cnt in enumerate(contours):
        # Only consider outer contours (no parent) as the gate frame body.
        if hierarchy[i][3] != -1:
            continue

        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        bbox_area = float(w * h)
        if bbox_area <= 0:
            continue

        square = _squareness(w, h)
        extent = area / bbox_area
        if extent < min_extent:
            continue

        candidates.append((area, extent, square, i, x, y, w, h, cnt))

    detections = []
    for (area, extent, square_outer, idx, x, y, w, h, outer) in candidates:
        bbox_px = (float(x), float(y), float(w), float(h))

        # Prefer the inner-hole contour (the flyable opening) for center/area/corners.
        child = hierarchy[idx][2]
        inner = None
        inner_area = -1.0
        j = child
        while j != -1:
            a = cv2.contourArea(contours[j])
            if a > inner_area:
                inner_area = a
                inner = contours[j]
            j = hierarchy[j][0]  # next sibling

        if inner is not None and inner_area >= max(1.0, 0.05 * min_area):
            center = _centroid(inner)
            area_px = float(inner_area)
            opening_contour = inner
        else:
            center = _centroid(outer)
            area_px = float(cv2.contourArea(outer))
            opening_contour = outer

        if center is None:
            # Degenerate moments; fall back to bbox center.
            center = (x + w / 2.0, y + h / 2.0)

        # Upgrade path: approximate the inner opening to an ordered 4-point quad.
        corners_px = None
        if opening_contour is not None:
            # Try to find a 4-corner approximation by sweeping epsilon
            best_approx = None
            peri = cv2.arcLength(opening_contour, True)
            
            for eps_frac in [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]:
                eps = eps_frac * peri
                approx = cv2.approxPolyDP(opening_contour, eps, True)
                if len(approx) == 4:
                    best_approx = approx
                    break
            
            if best_approx is None:
                eps = float(cfg.get("approx_eps_frac", 0.06)) * peri
                best_approx = cv2.approxPolyDP(opening_contour, eps, True)
                
            if len(best_approx) == 4:
                # Sub-pixel corner refinement
                corners_f32 = best_approx.astype(np.float32)
                # Find edges by converting the mask to a grayscale image (it's already a mask, but we need single channel)
                # Actually, we can just use the mask we already computed.
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                cv2.cornerSubPix(mask, corners_f32, (5, 5), (-1, -1), criteria)
                corners_px = _order_corners(corners_f32)

        # Confidence model: product of factors in [0,1]
        # - squareness of bbox
        ow, oh = w, h
        if opening_contour is not None:
            _, _, ow, oh = cv2.boundingRect(opening_contour)
        # Soften squareness penalty: a gate viewed at an angle is thin.
        square = max(0.5, _squareness(ow, oh))
        
        # - mask fill ratio (extent of the outer contour)
        # Extent is typically 0.5-0.8 for a rotated gate frame. We scale it so 0.5 -> 1.0.
        fill_ratio = min(1.0, extent / 0.4)
        
        # - corner count (1.0 with 4 corners, 0.6 without)
        corner_score = 1.0 if corners_px is not None else 0.6
        
        # - area fraction sanity
        frame_px = float(bgr.shape[0] * bgr.shape[1])
        rel_area = area_px / frame_px if frame_px > 0 else 0.0
        
        # Penalize blobs smaller than ~400px (rel_area=0.0017) so noise drops below 0.4 conf
        area_sanity = min(1.0, rel_area / 0.0017)
        
        confidence = float(np.clip(square * fill_ratio * corner_score * area_sanity, 0.0, 1.0))

        detections.append(GateDetection(
            center_px=(float(center[0]), float(center[1])),
            corners_px=corners_px,
            bbox_px=bbox_px,
            area_px=area_px,
            confidence=confidence,
        ))

    # Sort detections: best (nearest/most confident) first
    detections.sort(key=lambda d: d.area_px * d.confidence, reverse=True)
    return detections

def detect_gate(bgr: np.ndarray, cfg: Optional[dict] = None) -> Optional[GateDetection]:
    dets = detect_gates(bgr, cfg)
    return dets[0] if dets else None


def draw_detection(bgr: np.ndarray, det: Optional[GateDetection]) -> np.ndarray:
    """Return an annotated COPY of `bgr` (bbox, center, corners, confidence)."""
    out = bgr.copy()
    if det is None:
        cv2.putText(out, "no gate", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return out

    x, y, w, h = (int(v) for v in det.bbox_px)
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)

    u, v = int(round(det.center_px[0])), int(round(det.center_px[1]))
    cv2.circle(out, (u, v), 5, (0, 0, 255), -1)

    if det.corners_px is not None:
        labels = ("TL", "TR", "BR", "BL")
        for (cx, cy), label in zip(det.corners_px, labels):
            cx, cy = int(round(cx)), int(round(cy))
            cv2.circle(out, (cx, cy), 4, (255, 0, 255), -1)
            cv2.putText(out, label, (cx + 4, cy - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 0, 255), 1, cv2.LINE_AA)

    cv2.putText(out, f"conf {det.confidence:.2f}", (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def _synthesize_test_image() -> tuple:
    """Build a 640x360 desaturated-gray frame with a bright colored square frame.

    Returns (image_bgr, cfg) where cfg's HSV thresholds match the painted color
    so the self-test exercises the full pipeline without real sim frames.
    """
    h, w = 360, 640
    img = np.full((h, w, 3), 110, dtype=np.uint8)  # near-neutral gray background

    # Slight desaturated texture so the background is not perfectly flat.
    noise = np.random.randint(-8, 8, (h, w, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    cx, cy = w // 2, h // 2  # (320, 180)
    outer = 200
    inner = 120
    color = (40, 200, 60)  # bright green (BGR), saturated -> stands out

    # Draw the outer filled square, then punch the inner hole back to background.
    o0 = (cx - outer // 2, cy - outer // 2)
    o1 = (cx + outer // 2, cy + outer // 2)
    cv2.rectangle(img, o0, o1, color, thickness=-1)
    i0 = (cx - inner // 2, cy - inner // 2)
    i1 = (cx + inner // 2, cy + inner // 2)
    cv2.rectangle(img, i0, i1, (110, 110, 110), thickness=-1)

    # HSV thresholds tuned to the synthetic green: bright + saturated.
    cfg = {
        "lower_hsv": (40, 80, 80),
        "upper_hsv": (80, 255, 255),
        "lower_hsv2": None,
        "upper_hsv2": None,
    }
    return img, cfg


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        frame = cv2.imread(path)
        if frame is None:
            print(f"Failed to read image: {path}")
            sys.exit(1)
        cfg = None  # use module thresholds for real frames
        print(f"Loaded {path} ({frame.shape[1]}x{frame.shape[0]})")
    else:
        frame, cfg = _synthesize_test_image()
        print("No image given; synthesized a 640x360 test frame with a gate.")

    det = detect_gate(frame, cfg)
    print("Detection:", det)

    overlay = draw_detection(frame, det)
    out_path = "_detect_debug.png"
    cv2.imwrite(out_path, overlay)
    print(f"Wrote overlay to {out_path}")

    if det is None:
        sys.exit(2)