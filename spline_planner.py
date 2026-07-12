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


from guidance.path import Path, carrot_velocity

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

        import config
        # Build the spline once. Drop consecutive coincident waypoints (a zero-length leg
        # breaks the knot spacing) while keeping each surviving waypoint's yaw aligned.
        self._wp_pos, self._wp_yaw = [], []
        if self.mission and self.mission.waypoints:
            for w in self.mission.waypoints:
                self._wp_pos.append(w.pos)
                self._wp_yaw.append(w.yaw)
                
        self.path = Path(self._wp_pos, self._wp_yaw, loop=self._loop, cfg=config)
        self._s_end = self.path.length()

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
            
        import config

        pned = snap.get('position_ned')
        if pned is not None and pned.get('vx') is not None:
            vel_ned = np.array([pned.get('vx', 0.0), pned.get('vy', 0.0), pned.get('vz', 0.0)])
        else:
            vel_ned = None
            
        vel_cmd, yaw_cmd, s_proj, lookahead = carrot_velocity(self.path, pos, vel_ned, config, getattr(self, '_last_t', 0.0))
        self._last_t = s_proj

        # Cross-track-envelope fail-safe
        path_p, _ = self.path.sample(s_proj)
        xte = float(np.linalg.norm(path_p[:2] - pos[:2]))
        if xte > MAX_WP_DIST_M:
            return self._publish(self._hover(yaw_hold, 'dist_guard', now, range_m=xte))

        remaining = self._s_end - s_proj

        if not self._loop and remaining <= DEFAULT_ARRIVE_RADIUS_M:
            self._complete = True
        if self._complete:
            yaw_end = self._wp_yaw[-1]
            return self._publish(self._hover(
                yaw_end if yaw_end is not None else yaw_hold, 'spline_done', now,
                range_m=remaining, wp_index=len(self._wp_pos) - 1))
        if self._loop and remaining <= DEFAULT_ARRIVE_RADIUS_M:
            self._last_t = 0.0
            
        # If yaw_cmd is None, hold
        if yaw_cmd is None:
            yaw_cmd = yaw_hold

        wp_index = min(int(np.searchsorted(self.path._wp_s, s_proj)), len(self._wp_pos) - 1) if hasattr(self.path, '_wp_s') and len(self.path._wp_s) > 0 else 0
        return self._publish({'mode': 'velocity',
                              'vel_ned': tuple(float(c) for c in vel_cmd),
                              'yaw': float(yaw_cmd),
                              'range_m': remaining,
                              'source': f'spline:wp{wp_index}',
                              'wp_index': wp_index,
                              'xte_m': xte,
                              's_m': s_proj,
                              'lookahead_m': lookahead,
                              'ts': now})
