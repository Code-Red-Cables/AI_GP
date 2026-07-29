"""Pure path geometry for spline waypoint following.

Ported unchanged in substance from ``spline_planner.py`` on the
``origin/spline-path`` branch; the only edits are dropping the ``config``
imports so these stay side-effect free and unit-testable without a
simulator or a blackboard.

  build_spline_path  waypoints -> dense centripetal Catmull-Rom samples
  path_curvature     per-sample curvature (1/m) for corner slowing
  speed_profile      curvature- and brake-aware speed at each sample

Frames: every position is NED metres. Nothing here reads shared state.
"""
from __future__ import annotations

import math

import numpy as np

SAMPLES_PER_SEG = 40        # Catmull-Rom samples per waypoint segment
CURV_STENCIL_M = 3.5        # arc-length each side used to estimate curvature
FWD_WINDOW = 40             # samples ahead of last progress when projecting
BACK_WINDOW = 8             # samples behind last progress when projecting


def _catmull_rom_segment(p0, p1, p2, p3, n, alpha=0.5):
    """Sample the centripetal Catmull-Rom curve from ``p1`` to ``p2`` (``n`` points, the
    end ``p2`` EXCLUDED so segments concatenate without duplicate joints).

    Centripetal parameterisation (alpha=0.5) is used over the uniform variant because it
    cannot produce the cusps / self-intersections uniform Catmull-Rom makes on unevenly
    spaced control points -- important for arbitrary gate layouts.
    """
    def _knot(ti, pi, pj):
        d = float(np.linalg.norm(pj - pi))
        return ti + (d ** alpha if d > 1e-9 else 1e-9)

    t0 = 0.0
    t1 = _knot(t0, p0, p1)
    t2 = _knot(t1, p1, p2)
    t3 = _knot(t2, p2, p3)
    out = []
    for i in range(n):
        t = t1 + (t2 - t1) * (i / n)
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        c = (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2
        out.append(c)
    return out


def build_spline_path(positions, samples_per_seg=SAMPLES_PER_SEG):
    """Build a dense polyline through ``positions`` (a list of NED ``np.array`` points)
    plus its cumulative arc length and the arc length of each input waypoint.

    Returns ``(points[Nx3], cum_s[N], wp_s[len(positions)])`` where ``points`` passes
    THROUGH every input position (so ``cum_s[wp_s_index]`` is each waypoint's arc length).
    Coincident consecutive points must be removed by the caller; with 0 or 1 points the
    path degenerates to that single point.
    """
    pts_in = [np.asarray(p, float) for p in positions]
    if len(pts_in) <= 1:
        single = np.asarray(pts_in, float) if pts_in else np.zeros((1, 3))
        return single, np.zeros(len(single)), np.zeros(len(pts_in))

    # Phantom endpoints (reflect the first/last leg) so the end segments have neighbours.
    ext = ([pts_in[0] + (pts_in[0] - pts_in[1])] + pts_in +
           [pts_in[-1] + (pts_in[-1] - pts_in[-2])])

    samples = []
    wp_sample_idx = [0]                       # sample index of each input waypoint
    for i in range(1, len(ext) - 2):          # one iteration per real segment
        samples.extend(_catmull_rom_segment(ext[i - 1], ext[i], ext[i + 1], ext[i + 2],
                                            samples_per_seg))
        wp_sample_idx.append(len(samples))    # where the NEXT waypoint will land
    samples.append(ext[-2])                   # append the final waypoint (segments drop it)

    pts = np.asarray(samples, float)
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum_s = np.concatenate([[0.0], np.cumsum(seg_len)])
    wp_idx = np.clip(wp_sample_idx, 0, len(cum_s) - 1)
    return pts, cum_s, cum_s[wp_idx]


def path_curvature(pts, cum_s, stencil_m=CURV_STENCIL_M, horizontal_only=True):
    """Menger curvature ``1/R`` (1/m) at each polyline sample.

    Uses neighbours roughly ``stencil_m`` of arc length to each side (not the immediate
    samples) so the dense sampling doesn't make the estimate noisy. Straight runs give ~0;
    a tight bend gives a large value. Endpoints (no full stencil) stay 0.

    ``horizontal_only`` (default True) measures the bend in the N/E plane only, ignoring the
    vertical (down) component. The speed profile uses this curvature against ``A_LAT_MAX``,
    which is the LATERAL/cornering accel budget set by the ROLL authority (g*tan(roll)). A
    pure climb/dive is NOT a lateral corner -- it is driven by THRUST (a separate, stronger
    authority, already bounded by MAX_VSPEED and the vd cap), so penalising vertical curvature
    as if it were a horizontal turn needlessly throttles forward speed. (Concrete case: the
    captured course's first gates are dead straight horizontally [R~44 m] but bob ~1 m in
    altitude; in full 3D that reads as an R~9 m "corner" and capped the start at ~10 m/s when
    the airframe could carry ~22 m/s straight through. Set False to limit vertical curves too.)
    The stencil still walks the TRUE 3-D arc length, so braking distances stay physical.
    """
    n = len(pts)
    curv = np.zeros(n)
    cp = pts.copy()
    if horizontal_only:
        cp[:, 2] = 0.0                 # flatten to N/E; vertical curves don't cost roll
    for i in range(n):
        a, b = i, i
        while a > 0 and cum_s[i] - cum_s[a] < stencil_m:
            a -= 1
        while b < n - 1 and cum_s[b] - cum_s[i] < stencil_m:
            b += 1
        if a == i or b == i:
            continue
        A, B, C = cp[a], cp[i], cp[b]
        ab = np.linalg.norm(B - A)
        bc = np.linalg.norm(C - B)
        ca = np.linalg.norm(A - C)
        denom = ab * bc * ca
        if denom < 1e-9:
            continue
        area = 0.5 * np.linalg.norm(np.cross(B - A, C - A))   # triangle ABC area
        curv[i] = 4.0 * area / denom                          # 1/circumradius
    return curv


def speed_profile(curv, cum_s, cruise, a_lat, a_lon, end_speed=0.0):
    """Max safe speed (m/s) at each sample.

    Three limits combine: (1) ``cruise`` cap; (2) cornering -- ``sqrt(a_lat/curvature)`` so
    the lateral accel needed to hold the bend stays within ``a_lat``; (3) a BACKWARD braking
    pass -- ``sqrt(v_next^2 + 2*a_lon*ds)`` -- so the drone can decelerate in time for every
    upcoming corner and for the FINISH. ``end_speed`` is the speed allowed AT the last
    waypoint: 0.0 brakes to a full stop (settle), a positive value lets the drone CROSS the
    finish gate at speed (a race doesn't need to stop -- the timer ends at the gate), and
    ``None`` leaves the end unconstrained (looping paths never stop). The backward pass is
    what makes it slow *before* a turn rather than arriving too hot. A no-op where the path
    is gentle enough that ``cruise`` is feasible everywhere (so slow runs are unchanged).
    """
    n = len(curv)
    v = np.minimum(cruise, np.sqrt(a_lat / np.maximum(curv, 1e-6)))
    if n == 0:
        return v
    if end_speed is not None:
        v[-1] = min(v[-1], end_speed)
    for i in range(n - 2, -1, -1):
        ds = cum_s[i + 1] - cum_s[i]
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * a_lon * ds))
    return v


