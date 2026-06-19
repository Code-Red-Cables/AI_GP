"""Planning: follow a smooth SPLINE through the mission waypoints at constant cruise speed.

Where ``planner.Planner`` STOPS at every waypoint (commanded speed = KP_POS * distance, so
it decelerates into each point, dwells, then re-accelerates), this planner fits a
centripetal Catmull-Rom spline through the SAME waypoint positions and flies it
CONTINUOUSLY using a pure-pursuit "carrot":

1. Once at init, sample a Catmull-Rom spline through the waypoint positions into a dense
   polyline and tabulate its cumulative arc length (the polyline passes through every
   waypoint, so the path is faithful -- it just rounds the corners).
2. Each tick, project the drone onto the path (closest sample, searched forward from the
   last progress so we never snap backwards), place a CARROT ``LOOKAHEAD_M`` further along
   the path, and command a constant ``CRUISE_SPEED`` toward the carrot.
3. Speed only tapers in the FINAL approach (``KP_POS_END * remaining-arc-length``) so the
   drone settles on the last waypoint -- unless the mission loops, in which case the carrot
   wraps and it never slows.

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
from config import MAX_SPEED, MAX_VSPEED, MAX_WP_DIST_M, CRUISE_SPEED, LOOKAHEAD_M

# --------------------------------------------------------------------------------------
# Guidance tuning.
#   CRUISE_SPEED / LOOKAHEAD_M live in config.py (the run knobs). KP_POS_END sets how hard
#   the FINAL approach decelerates (commanded speed = KP_POS_END * remaining arc length,
#   capped at CRUISE_SPEED) so the drone settles on the last waypoint instead of overshoot.
#   SAMPLES_PER_SEG is the spline resolution; FWD/BACK_WINDOW bound the per-tick projection
#   search (in samples) so the carrot tracks forward without snapping back on a near self-
#   approach.
# --------------------------------------------------------------------------------------
KP_POS_END = 0.8            # final-approach speed = KP_POS_END * remaining arc length (m/s)
SAMPLES_PER_SEG = 40        # Catmull-Rom samples per waypoint segment (denser = smoother)
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
        else:
            self._pts, self._cum_s, self._wp_s = np.zeros((1, 3)), np.zeros(1), np.zeros(0)
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
        position, yaw, telem_ts = self._pose(self._snapshot())
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
            self._i = 0                            # wrap progress to the start of the path
            s_proj = 0.0
            remaining = self._s_end

        # Carrot: a point LOOKAHEAD_M further along the path (wrapping when looping).
        s_carrot = s_proj + LOOKAHEAD_M
        if self._loop and s_carrot > self._s_end:
            s_carrot -= self._s_end
        carrot = self._point_at_s(s_carrot)
        to_carrot = carrot - pos
        dist_c = float(np.linalg.norm(to_carrot))

        # Constant cruise toward the carrot, tapering only on the final approach so the
        # drone decelerates into the last waypoint (skipped while looping).
        speed = CRUISE_SPEED
        if not self._loop:
            speed = min(CRUISE_SPEED, KP_POS_END * remaining)
        speed = min(speed, MAX_SPEED)

        if dist_c > 1e-6:
            vel = to_carrot * (speed / dist_c)
        else:
            vel = np.zeros(3)
        vn, ve, vd = float(vel[0]), float(vel[1]), float(vel[2])
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
                              'ts': now})
