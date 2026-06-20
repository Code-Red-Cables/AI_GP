"""Planning: follow a smooth SPLINE through the mission waypoints at constant cruise speed.

Where ``planner.Planner`` STOPS at every waypoint (commanded speed = KP_POS * distance, so
it decelerates into each point, dwells, then re-accelerates), this planner fits a
centripetal Catmull-Rom spline through the SAME waypoint positions and flies it
CONTINUOUSLY using a pure-pursuit "carrot":

1. Once at init, sample a Catmull-Rom spline through the waypoint positions into a dense
   polyline and tabulate its cumulative arc length (the polyline passes through every
   waypoint, so the path is faithful -- it just rounds the corners).
2. Once at init, also tabulate a curvature-/brake-aware SPEED PROFILE along the path:
   full ``CRUISE_SPEED`` on the straights, slowed for corners (so the lateral accel needed
   to hold the bend stays within ``A_LAT_MAX``) and braked ahead of them and the final
   waypoint (at ``A_LON_MAX``). This is what keeps the drone from overshooting a turn at
   high speed; it is a no-op when the path is gentle enough to hold cruise throughout.
3. Each tick, project the drone onto the path (closest sample, searched forward from the
   last progress so we never snap backwards), place a CARROT a SPEED-SCALED distance ahead
   (``clamp(LOOKAHEAD_TIME * speed, LOOKAHEAD_M, LOOKAHEAD_MAX)`` -- short when slow so the
   dense gates/climb-out track tightly, long when fast so the straights stay smooth), and
   command the profile's speed for the current point toward the carrot. A looping mission
   never stops (the carrot and progress wrap).

Because the carrot keeps moving ahead, the velocity command never falls to zero at an
intermediate waypoint: the drone flies through the course in one continuous motion instead
of the stop-start of the waypoint planner.

Same contract as ``Planner``: reads telemetry from ``shared_data`` and writes
``shared_data['target'] = {'mode':'velocity', 'vel_ned':(vn,ve,vd), 'yaw':rad, ...}`` --
the controller maps it to attitude+thrust UNCHANGED. Yaw is interpolated along the path
from the waypoint headings (a ``yaw=None`` waypoint -> HOLD current heading, exactly as in
the waypoint planner). Safety watchdogs mirror ``Planner``: stale telemetry -> hover; an
altitude-envelope runaway guard; and a cross-track-envelope guard (too far OFF the path ->
hover so the velocity loop brakes back toward it).
"""

import math
import threading
import time

import numpy as np

from mission import DEFAULT_ARRIVE_RADIUS_M
from config import (MAX_SPEED, MAX_VSPEED, MAX_WP_DIST_M, CRUISE_SPEED, LOOKAHEAD_M,
                    LOOKAHEAD_TIME, LOOKAHEAD_MAX, A_LAT_MAX, A_LON_MAX, FINISH_SPEED,
                    KP_VERT_PATH)

# --------------------------------------------------------------------------------------
# Guidance tuning.
#   CRUISE_SPEED / LOOKAHEAD_M / A_LAT_MAX / A_LON_MAX live in config.py (the run knobs).
#   The speed at each point is the curvature-/brake-aware PROFILE (see speed_profile): it is
#   capped at CRUISE_SPEED, slows for corners (sqrt(A_LAT_MAX/curvature)) and brakes ahead of
#   them and the final waypoint (A_LON_MAX) so the drone never overshoots a turn at speed.
#   SAMPLES_PER_SEG is the spline resolution; CURV_STENCIL_M smooths the curvature estimate;
#   FWD/BACK_WINDOW bound the per-tick projection search (in samples) so the carrot tracks
#   forward without snapping back on a near self-approach.
# --------------------------------------------------------------------------------------
SAMPLES_PER_SEG = 40        # Catmull-Rom samples per waypoint segment (denser = smoother)
CURV_STENCIL_M = 3.5        # arc-length each side used to estimate curvature (damps noise)
FWD_WINDOW = 40             # samples ahead of last progress to search when projecting
BACK_WINDOW = 8             # samples behind last progress to search (small backward fix)

# --------------------------------------------------------------------------------------
# Safety guards (identical in spirit to planner.Planner -- pure client-side fail-safes).
# --------------------------------------------------------------------------------------
TELEM_TIMEOUT_NS = 500_000_000     # 500 ms without a pose -> hover (we'd be flying blind)
MAX_ALT_M = 15.0                   # height (m above the arm point) beyond which we recover


def _quat_yaw(q):
    """Yaw (rad, NED) from a quaternion ``(w, x, y, z)`` -- fallback heading source when no
    ATTITUDE message is available (odometry only). Mirrors planner._quat_yaw."""
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


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


class SplinePlanner:
    """Follows a Catmull-Rom spline through ``shared_data['mission']`` at cruise speed."""

    def __init__(self, data):
        self.data = data
        if 'lock' not in self.data:
            self.data['lock'] = threading.RLock()
        self.mission = self.data.get('mission')
        self._loop = bool(self.mission.loop) if self.mission else False
        self._i = 0                 # index of the closest path sample (monotone progress)
        self._complete = False      # True once a non-looping path's end is reached

        # Build the spline once. Drop consecutive coincident waypoints (a zero-length leg
        # breaks the knot spacing) while keeping each surviving waypoint's yaw aligned.
        self._wp_pos, self._wp_yaw = [], []
        if self.mission and self.mission.waypoints:
            for w in self.mission.waypoints:
                p = np.asarray(w.pos, float)
                if self._wp_pos and np.linalg.norm(p - self._wp_pos[-1]) <= 1e-6:
                    continue
                self._wp_pos.append(p)
                self._wp_yaw.append(w.yaw)
        if self._wp_pos:
            self._pts, self._cum_s, self._wp_s = build_spline_path(self._wp_pos)
            self._curv = path_curvature(self._pts, self._cum_s)
            # Curvature-/brake-aware speed profile (per sample): cruise on the straights,
            # slow for corners, and at the finish either CROSS at FINISH_SPEED or (looping)
            # never stop.
            end_speed = None if self._loop else FINISH_SPEED
            self._vmax = speed_profile(self._curv, self._cum_s, CRUISE_SPEED,
                                       A_LAT_MAX, A_LON_MAX, end_speed=end_speed)
        else:
            self._pts, self._cum_s, self._wp_s = np.zeros((1, 3)), np.zeros(1), np.zeros(0)
            self._curv = np.zeros(1)
            self._vmax = np.zeros(1)
        self._s_end = float(self._cum_s[-1])

    # ---- shared helpers (mirror planner.Planner) -------------------------------------
    @staticmethod
    def _wrap(a):
        """Wrap an angle to (-pi, pi]."""
        return math.atan2(math.sin(a), math.cos(a))

    def _snapshot(self):
        with self.data['lock']:
            return {
                'attitude': self.data.get('attitude'),
                'odometry': self.data.get('odometry'),
                'position_ned': self.data.get('position_ned'),
            }

    @staticmethod
    def _pose(s):
        """Return ``(position_ned tuple|None, yaw rad|None, telem_ts ns|None)``."""
        pos, odo, att = s['position_ned'], s['odometry'], s['attitude']
        position = ts = None
        if pos is not None:
            position = (pos['x'], pos['y'], pos['z'])
            ts = pos.get('ts')
        elif odo is not None:
            position = tuple(odo['pos'])
            ts = odo.get('ts')
        yaw = None
        if att is not None:
            yaw = att['yaw']
            ts = att.get('ts', ts)
        elif odo is not None:
            yaw = _quat_yaw(odo['q'])
        return position, yaw, ts

    def _publish(self, target):
        with self.data['lock']:
            self.data['target'] = target
        return target

    def _hover(self, yaw, source, now, **extra):
        return {'mode': 'velocity', 'vel_ned': (0.0, 0.0, 0.0),
                'yaw': float(yaw), 'source': source, 'ts': now, **extra}

    # ---- path queries -----------------------------------------------------------------
    def _project(self, pos):
        """Closest path sample to ``pos`` searched in a window FORWARD of the last
        progress. Returns ``(s_proj, idx, cross_track_dist)`` and advances ``self._i``."""
        lo = max(0, self._i - BACK_WINDOW)
        hi = min(len(self._pts), self._i + FWD_WINDOW + 1)
        d = np.linalg.norm(self._pts[lo:hi] - pos, axis=1)
        k = lo + int(np.argmin(d))
        self._i = k
        return float(self._cum_s[k]), k, float(d[k - lo])

    def _point_at_s(self, s):
        """Linear-interpolated path point at arc length ``s`` (clamped to the path)."""
        s = min(max(s, 0.0), self._s_end)
        j = int(np.searchsorted(self._cum_s, s))
        if j <= 0:
            return self._pts[0]
        if j >= len(self._pts):
            return self._pts[-1]
        s0, s1 = self._cum_s[j - 1], self._cum_s[j]
        f = 0.0 if s1 <= s0 else (s - s0) / (s1 - s0)
        return self._pts[j - 1] + f * (self._pts[j] - self._pts[j - 1])

    def _yaw_at_s(self, s):
        """Commanded yaw (rad) interpolated between the bracketing waypoint headings, or
        ``None`` to HOLD the current heading when either bracket waypoint is yaw=None."""
        wp_s, yaws = self._wp_s, self._wp_yaw
        if len(wp_s) == 0:
            return None
        if len(wp_s) == 1 or s <= wp_s[0]:
            return yaws[0]
        if s >= wp_s[-1]:
            return yaws[-1]
        j = int(np.searchsorted(wp_s, s))      # wp_s[j-1] <= s < wp_s[j]
        ya, yb = yaws[j - 1], yaws[j]
        if ya is None or yb is None:
            return None                        # hold heading across a yaw=None segment
        s0, s1 = wp_s[j - 1], wp_s[j]
        f = 0.0 if s1 <= s0 else (s - s0) / (s1 - s0)
        return ya + self._wrap(yb - ya) * f

    # ---- main tick --------------------------------------------------------------------
    def compute_target(self):
        """Compute and publish the velocity setpoint for this tick. Returns the target."""
        now = time.time_ns()
        snap = self._snapshot()
        position, yaw, telem_ts = self._pose(snap)
        yaw_hold = yaw if yaw is not None else 0.0

        # Watchdog: no recent pose at all -> hover (we'd otherwise be flying blind).
        if position is None or telem_ts is None or (now - telem_ts) > TELEM_TIMEOUT_NS:
            return self._publish(self._hover(yaw_hold, 'watchdog_hover', now))

        pos = np.asarray(position, float)

        # Altitude-envelope fail-safe: NED z is negative-up; past the ceiling the vertical
        # loop has run away -> abandon the path and descend until back below it.
        altitude = -float(pos[2])
        if altitude > MAX_ALT_M:
            return self._publish({'mode': 'velocity', 'vel_ned': (0.0, 0.0, MAX_VSPEED),
                                  'yaw': yaw_hold, 'range_m': altitude,
                                  'source': 'alt_guard', 'ts': now})

        if not self._wp_pos:
            return self._publish(self._hover(yaw_hold, 'no_mission', now))

        s_proj, idx, xte = self._project(pos)

        # Cross-track-envelope fail-safe: if we are this far OFF the path, the horizontal
        # loop has run away -> hover so the velocity loop leans back toward the course.
        if xte > MAX_WP_DIST_M:
            return self._publish(self._hover(yaw_hold, 'dist_guard', now, range_m=xte))

        remaining = self._s_end - s_proj          # arc length left to the final waypoint

        # Completion (non-looping): once we are within an arrival radius of the path end,
        # settle on the final waypoint's heading and hold. A looping path never completes.
        if not self._loop and remaining <= DEFAULT_ARRIVE_RADIUS_M:
            self._complete = True
        if self._complete:
            yaw_end = self._wp_yaw[-1]
            return self._publish(self._hover(
                yaw_end if yaw_end is not None else yaw_hold, 'spline_done', now,
                range_m=remaining, wp_index=len(self._wp_pos) - 1))
        if self._loop and remaining <= DEFAULT_ARRIVE_RADIUS_M:
            self._i = idx = 0                      # wrap progress to the start of the path
            s_proj = 0.0
            remaining = self._s_end

        # Speed-scaled lookahead: short when slow (dense gates / the climb-out -> tight
        # tracking incl. altitude, so we don't cut UNDER the first gates), long when fast
        # (smooth straights, no pure-pursuit wobble). Driven by MEASURED speed so it stays
        # short while the drone is still accelerating off the start line.
        pned = snap.get('position_ned')
        if pned is not None and pned.get('vx') is not None:
            speed_meas = math.hypot(pned.get('vx', 0.0), pned.get('vy', 0.0))
        else:
            speed_meas = float(self._vmax[idx])          # fallback: planned speed
        lookahead = min(LOOKAHEAD_MAX, max(LOOKAHEAD_M, LOOKAHEAD_TIME * speed_meas))

        # Carrot: a point `lookahead` further along the path (wrapping when looping).
        s_carrot = s_proj + lookahead
        if self._loop and s_carrot > self._s_end:
            s_carrot -= self._s_end
        carrot = self._point_at_s(s_carrot)
        to_carrot = carrot - pos
        dist_c = float(np.linalg.norm(to_carrot))

        # Speed toward the carrot comes from the curvature-/brake-aware profile (cruise on
        # straights, slowed for the corner ahead and braked into the final waypoint), so the
        # drone never arrives at a turn too hot to make it. Capped at MAX_SPEED.
        speed = min(float(self._vmax[idx]), MAX_SPEED)

        # DECOUPLED horizontal / vertical command. The airframe steers forward with PITCH and
        # changes height with THRUST -- independent axes -- so needing to climb must NEVER rob
        # forward speed. The old `vel = to_carrot * speed/dist` normalised the 3-D vector to
        # `speed`, so when the drone was BELOW the path (e.g. at the start: alt 0, first gate at
        # +0.87 m) the carrot pointed UP and a chunk of the speed budget went vertical -> off the
        # line it sank, then crawled forward while clawing back altitude instead of accelerating.
        # Fix: command the FULL profile speed along the carrot's HORIZONTAL bearing (the curvature
        # caps are horizontal too -- see path_curvature), and drive height SEPARATELY.
        horiz = to_carrot[:2]
        hdist = float(np.linalg.norm(horiz))
        if hdist > 1e-6:
            vn, ve = float(horiz[0] / hdist * speed), float(horiz[1] / hdist * speed)
        else:
            vn = ve = 0.0

        # Vertical = path-slope FEED-FORWARD toward the carrot + an altitude(position) P pull.
        # The controller's vertical loop tracks vertical VELOCITY only (no altitude feedback), so
        # without the position term the drone settles BELOW the path on climbs/descents and clips
        # gate bottoms. The term does NOT shrink with horizontal speed: below path (pos_z>path_z)
        # -> vd more negative -> climb; above -> descend. KP_VERT_PATH=0 -> slope feed-forward only.
        path_z = float(self._point_at_s(s_proj)[2])
        slope_ff = float(to_carrot[2] / dist_c * speed) if dist_c > 1e-6 else 0.0
        vd = slope_ff + KP_VERT_PATH * (path_z - float(pos[2]))
        vd = max(-MAX_VSPEED, min(MAX_VSPEED, vd))      # cap vertical separately

        yaw_at = self._yaw_at_s(s_proj)
        yaw_cmd = yaw_at if yaw_at is not None else yaw_hold

        wp_index = min(int(np.searchsorted(self._wp_s, s_proj)), len(self._wp_pos) - 1)
        return self._publish({'mode': 'velocity',
                              'vel_ned': (vn, ve, vd),
                              'yaw': float(yaw_cmd),
                              'range_m': remaining,
                              'source': f'spline:wp{wp_index}',
                              'wp_index': wp_index,
                              'xte_m': xte,
                              's_m': s_proj,
                              'lookahead_m': lookahead,
                              'ts': now})
