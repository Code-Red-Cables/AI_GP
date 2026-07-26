"""Gate geometry estimation (see PLAN.md 8.3).

Turns a 2D `GateDetection` (pixels) into a 3D estimate of where the gate is
relative to the drone, using the pinhole `camera_model`:

- **size method** (always available): assume the gate opening is a fronto-parallel
  1.5 m square; recover optical-axis depth from its apparent pixel size, then
  deproject the opening center to a camera-frame point.
- **PnP method** (when 4 corners are detected): `cv2.solvePnP` on the known
  1.5 m square gives a full metric pose (position + the gate's surface normal),
  independent of the fronto-parallel assumption. This is the "reverse solver".

Outputs a plain dict matching `shared_data['vision']` (PLAN 8.6). All vectors are
in metres. `gate_body` is the gate center relative to the drone in BODY axes
(x-fwd, y-right, z-down); `gate_ned` is the absolute world position and is only
filled when both attitude and drone NED position are supplied.
"""

import math

import numpy as np

try:  # cv2 only needed for the PnP upgrade; degrade gracefully without it.
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

import camera_model as cm

MAX_PNP_REPROJECTION_ERROR_PX = 6.0
MIN_PNP_RANGE_M = 0.20
MAX_PNP_RANGE_M = 50.0

# Object points of the flyable inner square in the gate's own frame (metres),
# ordered TL, TR, BR, BL to match both gate_detector._order_corners AND the
# canonical order cv2.SOLVEPNP_IPPE_SQUARE expects: a y-UP planar square
# [(-s,+s,0),(+s,+s,0),(+s,-s,0),(-s,-s,0)] with +z as the surface normal.
def build_gate_object_points(width_m=cm.GATE_INNER_M, height_m=cm.GATE_INNER_M):
    """Return planar gate corners TL, TR, BR, BL in metres (y is up)."""
    half_width = float(width_m) / 2.0
    half_height = float(height_m) / 2.0
    return np.array(
        [
            [-half_width, half_height, 0.0],
            [half_width, half_height, 0.0],
            [half_width, -half_height, 0.0],
            [-half_width, -half_height, 0.0],
        ],
        dtype=np.float32,
    )


_GATE_OBJ_PTS = build_gate_object_points()


def _opening_side_px(det):
    """Apparent side length (px) of the gate opening, from corners if available."""
    if det.corners_px is not None and len(det.corners_px) == 4:
        pts = [np.asarray(c, float) for c in det.corners_px]
        edges = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
        return float(np.mean(edges))
    # Fall back to the side of a square with the measured opening area.
    return float(math.sqrt(max(det.area_px, 1.0)))


def _bearing(gate_body):
    """(azimuth, elevation) in radians of a body-frame point (x-fwd, y-right, z-down)."""
    x, y, z = gate_body
    az = math.atan2(y, x)                       # +right
    el = math.atan2(-z, math.hypot(x, y))       # +up (z is down)
    return float(az), float(el)


def _solve_pnp(det):
    """Return validated PnP pose and reprojection error, or None on failure."""
    if (
        cv2 is None
        or not getattr(det, "corners_reliable", True)
        or det.corners_px is None
        or len(det.corners_px) != 4
    ):
        return None
    img_pts = np.array(det.corners_px, dtype=np.float32)
    ok, rvec, tvec = cv2.solvePnP(
        _GATE_OBJ_PTS, img_pts, cm.K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )

    def reprojection(rv, tv):
        if (
            not np.all(np.isfinite(rv))
            or not np.all(np.isfinite(tv))
        ):
            return float("inf")
        projected, _ = cv2.projectPoints(_GATE_OBJ_PTS, rv, tv, cm.K, None)
        residual = projected.reshape(-1, 2) - img_pts
        return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))

    reprojection_error = reprojection(rvec, tvec) if ok else float("inf")
    # IPPE can return NaNs or a poor degenerate solution for a nearly perfect
    # fronto-parallel square. ITERATIVE is deterministic and stable there.
    if not ok or reprojection_error > MAX_PNP_REPROJECTION_ERROR_PX:
        ok, rvec, tvec = cv2.solvePnP(
            _GATE_OBJ_PTS, img_pts, cm.K, None, flags=cv2.SOLVEPNP_ITERATIVE
        )
        reprojection_error = reprojection(rvec, tvec) if ok else float("inf")
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    gate_cam = tvec.reshape(3)
    range_m = float(np.linalg.norm(gate_cam))
    if (
        gate_cam[2] <= 0.0
        or not (MIN_PNP_RANGE_M <= range_m <= MAX_PNP_RANGE_M)
    ):
        return None
    if not np.isfinite(reprojection_error) or reprojection_error > MAX_PNP_REPROJECTION_ERROR_PX:
        return None
    normal_cam = R @ np.array([0.0, 0.0, 1.0])  # gate +z (surface normal) in cam frame
    return gate_cam, normal_cam, reprojection_error, rvec, tvec


def estimate_gate(det, attitude=None, position_ned=None, use_pnp=True, ts=None):
    """Estimate the detected gate's 3D pose relative to the drone.

    Parameters
    ----------
    det : GateDetection           detector output (or None -> returns a "not detected" dict)
    attitude : dict | None        {'roll','pitch','yaw'} radians; enables NED axes
    position_ned : tuple | None   drone (x,y,z) in world NED; enables absolute gate_ned
    use_pnp : bool                try solvePnP when corners are available
    ts : int | None               timestamp (ns) to stamp the estimate

    Returns a dict (PLAN 8.6 schema). `gate_ned` is None unless attitude (and,
    for an absolute position, position_ned) are provided.
    """
    if det is None:
        return {"ts": ts, "detected": False, "confidence": 0.0}

    method = "size"
    normal_body = None
    reprojection_error = None

    pnp = _solve_pnp(det) if use_pnp else None
    if pnp is not None:
        gate_cam, normal_cam, reprojection_error, pnp_rvec, pnp_tvec = pnp
        gate_body = cm.cam_to_body(gate_cam)
        normal_body = cm.cam_to_body(normal_cam)
        method = "pnp"
    else:
        side_px = _opening_side_px(det)
        Z = cm.range_from_size(side_px, real_size=cm.GATE_INNER_M, f=cm.FX)
        u, v = det.center_px
        gate_cam = cm.deproject(u, v, Z)
        gate_body = cm.cam_to_body(gate_cam)

    range_m = float(np.linalg.norm(gate_body))
    az, el = _bearing(gate_body)

    gate_ned = None
    if attitude is not None:
        r, p, y = attitude["roll"], attitude["pitch"], attitude["yaw"]
        offset_ned = cm.body_to_ned(gate_body, r, p, y)
        if normal_body is not None:
            normal_body = tuple(float(c) for c in cm.body_to_ned(normal_body, r, p, y))
        if position_ned is not None:
            gate_ned = tuple(float(a + b) for a, b in zip(position_ned, offset_ned))
        else:
            gate_ned = tuple(float(c) for c in offset_ned)  # relative, world axes

    confidence = float(det.confidence)
    if reprojection_error is not None:
        # PnP contributes up to 20% of the final confidence. A pose at the
        # configured reprojection limit contributes zero.
        pnp_quality = max(
            0.0, 1.0 - reprojection_error / MAX_PNP_REPROJECTION_ERROR_PX
        )
        confidence = float(np.clip(0.8 * confidence + 0.2 * pnp_quality, 0.0, 1.0))
        det.pnp_reprojection_error = reprojection_error
        det.pnp_rvec = pnp_rvec
        det.pnp_tvec = pnp_tvec

    return {
        "ts": ts,
        "detected": True,
        "confidence": confidence,
        "center_px": det.center_px,
        "corners_px": det.corners_px,
        "area_px": det.area_px,
        "range_m": range_m,
        "bearing": (az, el),
        "gate_body": tuple(float(c) for c in gate_body),
        "gate_ned": gate_ned,
        "normal_body": normal_body,
        "method": method,
        "detection_method": getattr(det, "method", "unknown"),
        "pnp_reprojection_error": reprojection_error,
    }
