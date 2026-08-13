"""True bearing to the active race gate from ODOMETRY + TRACK_DATA.

Phase 3b of the HG-DAgger plan uses this as the hard observation gate: the
detector's corner centroid must correlate with the bearing that ODOMETRY and
the gate map imply. If that correlation is near zero, the observation is
broken and DAgger cannot fix it.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np

import camera_model as cm


def _num(value: Any, default: float = float('nan')) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def gate_by_active(
    track_gates: Sequence[Mapping],
    active_gate: int,
) -> Optional[dict]:
    """Return the TRACK_DATA gate whose id matches ``active_gate``, else None.

    Sim race_status.active_gate is the index of the *next* gate to clear. On
    VQ1 TRACK_DATA that index matches gate ``id`` in the track payload.
    """
    for gate in track_gates or ():
        try:
            gid = int(gate.get('id', gate.get('gate_id')))
        except (TypeError, ValueError):
            continue
        if gid == int(active_gate):
            g = dict(gate)
            if 'pos' not in g and g.get('position_ned') is not None:
                g['pos'] = g['position_ned']
            g['id'] = gid
            return g
    return None


def true_gate_body(
    odo: Mapping,
    gate: Mapping,
) -> Optional[np.ndarray]:
    """Active-gate centre in the drone body frame (x fwd, y right, z down)."""
    pos = np.array([
        _num(odo.get('x')),
        _num(odo.get('y')),
        _num(odo.get('z')),
    ], dtype=float)
    gpos = gate.get('pos') or ()
    if len(gpos) < 3:
        return None
    gate_ned = np.array(
        [_num(gpos[0]), _num(gpos[1]), _num(gpos[2])], dtype=float
    )
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(gate_ned)):
        return None
    roll = _num(odo.get('roll'), 0.0)
    pitch = _num(odo.get('pitch'), 0.0)
    yaw = _num(odo.get('yaw'), 0.0)
    return cm.ned_to_body(gate_ned - pos, roll, pitch, yaw)


def true_bearing_rad(body: np.ndarray) -> Optional[float]:
    """Horizontal bearing in the body frame: atan2(right, forward)."""
    if body is None or len(body) < 2:
        return None
    x, y = float(body[0]), float(body[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    if abs(x) + abs(y) < 1e-6:
        return None
    return math.atan2(y, x)


def project_gate_centre_px(body: np.ndarray) -> Optional[tuple[float, float]]:
    """Project a body-frame gate centre to image pixels, or None if behind cam."""
    if body is None or len(body) < 3:
        return None
    cam = cm.body_to_cam(np.asarray(body, dtype=float))
    if float(cam[2]) <= 0.05:
        return None
    u, v = cm.project(cam)
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    return float(u), float(v)


def detector_bearing_rad(
    centre_u: float,
    centre_v: float,
    *,
    frame_w: float = cm.WIDTH,
    frame_h: float = cm.HEIGHT,
) -> Optional[float]:
    """Image-space horizontal bearing proxy from a corner centroid.

    Uses the pinhole ray's body-frame atan2(y, x), matching ``true_bearing_rad``
    so the two can be correlated without an arbitrary scale factor.
    """
    if not (math.isfinite(centre_u) and math.isfinite(centre_v)):
        return None
    ray_cam = cm.pixel_to_ray(centre_u, centre_v)
    ray_body = cm.cam_to_body(ray_cam)
    return true_bearing_rad(ray_body)


def keypoint_centroid_px(
    keypoints: Sequence[Sequence[float]],
    confidences: Optional[Sequence[float]] = None,
    *,
    min_confidence: float = 0.25,
) -> Optional[tuple[float, float]]:
    """Mean of visible keypoints; falls back to None when none are usable."""
    confs = list(confidences) if confidences is not None else []
    xs: list[float] = []
    ys: list[float] = []
    for i, pair in enumerate(keypoints or ()):
        try:
            u, v = float(pair[0]), float(pair[1])
        except (TypeError, IndexError, ValueError):
            continue
        if not (math.isfinite(u) and math.isfinite(v)):
            continue
        if u == 0.0 and v == 0.0:
            continue
        conf = _num(confs[i], 1.0) if i < len(confs) else 1.0
        if math.isfinite(conf) and conf < min_confidence:
            continue
        xs.append(u)
        ys.append(v)
    if not xs:
        return None
    return sum(xs) / len(xs), sum(ys) / len(ys)


def rotation_flow_basis(
    u: float,
    v: float,
    dt: float,
) -> Optional[np.ndarray]:
    """Per-axis pixel flow for a unit body rate, as a (2, 3) matrix.

    Column j is the (du, dv) a world-fixed point at ``(u, v)`` would show for
    1 rad/s about body axis j over ``dt``. Multiplying by the measured body
    rates reproduces ``rotation_flow_px``; keeping the columns separate lets a
    caller *fit* the gyro sign/scale convention instead of assuming it, which
    matters because this simulator already needs RATE_SIGN_PITCH = -1 on the
    command side.
    """
    if not all(math.isfinite(x) for x in (u, v, dt)) or dt <= 0.0:
        return None
    basis = np.zeros((2, 3), dtype=float)
    for axis in range(3):
        rates = [0.0, 0.0, 0.0]
        rates[axis] = 1.0
        flow = rotation_flow_px(u, v, rates[0], rates[1], rates[2], dt)
        if flow is None:
            return None
        basis[0, axis] = flow[0]
        basis[1, axis] = flow[1]
    return basis


def _fit_axis(A: np.ndarray, y: np.ndarray) -> tuple[Optional[list[float]], float]:
    if len(y) < 8:
        return None, float('nan')
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.dot(y - y.mean(), y - y.mean()))
    r2 = float('nan') if ss_tot <= 1e-9 else 1.0 - ss_res / ss_tot
    return [float(c) for c in coef], r2


def fit_rotation_flow(
    basis_rows: Sequence[np.ndarray],
    measured: Sequence[Sequence[float]],
) -> dict:
    """Least-squares fit of measured centroid flow onto per-axis rotation flow.

    The horizontal and vertical image axes are fitted separately, with their own
    three coefficients. Sharing coefficients across both would be physically
    tighter, but it makes the whole fit collapse if any single axis sign or unit
    convention is off -- and this simulator already needs RATE_SIGN_PITCH = -1
    on the command side, so its gyro conventions are not something to assume.

    R^2 near 1 means the tracked point moves as a world-fixed point must, i.e.
    the detector is following one object. R^2 near 0 means it is not -- but see
    ``rotation_share``: when the drone is barely rotating, translation flow
    dominates and a low R^2 proves nothing either way.
    """
    if not basis_rows or len(basis_rows) != len(measured):
        return {
            'n': 0, 'coef_u': None, 'coef_v': None,
            'r2_u': float('nan'), 'r2_v': float('nan'),
            'rotation_share': float('nan'),
        }
    au: list[list[float]] = []
    av: list[list[float]] = []
    yu: list[float] = []
    yv: list[float] = []
    for basis, obs in zip(basis_rows, measured):
        if basis is None:
            continue
        du, dv = float(obs[0]), float(obs[1])
        if not (math.isfinite(du) and math.isfinite(dv)):
            continue
        au.append(list(basis[0]))
        yu.append(du)
        av.append(list(basis[1]))
        yv.append(dv)
    n = len(yu)
    if n < 8:
        return {
            'n': n, 'coef_u': None, 'coef_v': None,
            'r2_u': float('nan'), 'r2_v': float('nan'),
            'rotation_share': float('nan'),
        }
    Au, Av = np.asarray(au, float), np.asarray(av, float)
    Yu, Yv = np.asarray(yu, float), np.asarray(yv, float)
    coef_u, r2_u = _fit_axis(Au, Yu)
    coef_v, r2_v = _fit_axis(Av, Yv)

    # How much of the observed motion could rotation account for at all? If the
    # fitted rotation term is tiny next to the measured motion, the flight was
    # nearly rotation-free and this check has no power.
    share = float('nan')
    if coef_u is not None:
        pred = np.abs(Au @ np.asarray(coef_u, float))
        meas = np.abs(Yu)
        denom = float(meas.sum())
        if denom > 1e-9:
            share = float(pred.sum() / denom)
    return {
        'n': n,
        'coef_u': coef_u,
        'coef_v': coef_v,
        'r2_u': r2_u,
        'r2_v': r2_v,
        'rotation_share': share,
    }


def rotation_flow_px(
    u: float,
    v: float,
    gx: float,
    gy: float,
    gz: float,
    dt: float,
) -> Optional[tuple[float, float]]:
    """Pixel displacement of a *world-fixed* point from body rotation alone.

    This is the ground-truth-free replacement for the ODOMETRY bearing check.
    A static scene point's image motion under pure camera rotation is
    independent of its depth, so it can be predicted from HIGHRES_IMU alone:

        u_dot = fx * [ wz*b - wy*(1 + a^2) + wx*a*b ]
        v_dot = fy * [ wx*(1 + b^2) - wz*a - wy*a*b ]

    with a = (u - cx)/fx, b = (v - cy)/fy and w the angular rate in the
    camera-optical frame. Correlating this prediction against the detector's
    measured centroid motion tests the property that actually matters: that the
    detector is tracking one fixed object rather than hopping between gates.
    Translation adds depth-dependent flow, so expect a strong but imperfect
    correlation -- an identity-swapping detector scores near zero.
    """
    if not all(math.isfinite(x) for x in (u, v, gx, gy, gz, dt)):
        return None
    if dt <= 0.0:
        return None
    w_body = np.array([float(gx), float(gy), float(gz)], dtype=float)
    wx, wy, wz = cm.R_CB @ w_body
    a = (float(u) - cm.CX) / cm.FX
    b = (float(v) - cm.CY) / cm.FY
    u_dot = cm.FX * (wz * b - wy * (1.0 + a * a) + wx * a * b)
    v_dot = cm.FY * (wx * (1.0 + b * b) - wz * a - wy * a * b)
    return float(u_dot * dt), float(v_dot * dt)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson r; NaN when undefined."""
    if len(xs) < 3 or len(xs) != len(ys):
        return float('nan')
    a = np.asarray(xs, dtype=float)
    b = np.asarray(ys, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return float('nan')
    return float(np.dot(a, b) / denom)


def best_lagged_pearson(
    xs: Sequence[float],
    ys: Sequence[float],
    max_lag: int = 12,
) -> tuple[float, int]:
    """Strongest |r| between ``xs`` and ``ys`` shifted by up to ``max_lag``.

    A human pilot does not react instantaneously, so the zero-lag correlation
    between a bearing error and the stick that corrects it understates the
    coupling. Positive lag means ``ys`` follows ``xs``.
    """
    best_r = float('nan')
    best_lag = 0
    n = min(len(xs), len(ys))
    if n < 4:
        return best_r, best_lag
    for lag in range(0, int(max_lag) + 1):
        a = xs[: n - lag] if lag else xs[:n]
        b = ys[lag:n] if lag else ys[:n]
        r = pearson(a, b)
        if math.isfinite(r) and (
            not math.isfinite(best_r) or abs(r) > abs(best_r)
        ):
            best_r, best_lag = r, lag
    return best_r, best_lag
