"""Planning: follow a fixed, preplanned waypoint mission (vision-free).

This branch flies a DETERMINISTIC path instead of reacting to camera detections. The
planner reads only the drone's measured NED position + heading from the shared blackboard
and drives it through an ordered list of Waypoints (see ``mission.py``):

1. For the current waypoint, command a velocity TOWARD its position (P control on the
   position error, capped) and command its target YAW.
2. Once the drone is within the waypoint's position radius AND heading tolerance, and has
   dwelt there briefly (so it settles / completes an in-place turn), advance to the next
   waypoint.
3. At the final waypoint, hold a hover (or, if the mission loops, restart it).

There is NO vision, gate map or track geometry here -- purely the preplanned mission plus
telemetry. Output is written to ``shared_data['target']`` as
``{'mode':'velocity', 'vel_ned':(vn,ve,vd), 'yaw':rad, ...}`` exactly as ``controller.py``
expects; the controller maps it to attitude+thrust (roll/pitch/yaw + collective thrust).

Safety watchdogs are retained: stale telemetry -> hover; an altitude-envelope guard
catches a vertical-loop runaway and descends.
"""

import math
import os
import threading
import time

import numpy as np

from mission import DEFAULT_ARRIVE_RADIUS_M, DEFAULT_YAW_TOL_RAD, DEFAULT_DWELL_S

# --------------------------------------------------------------------------------------
# Guidance tuning. The commanded speed is proportional to the distance remaining (so the
# drone decelerates into each waypoint), capped at MAX_SPEED, with the vertical component
# capped separately. Raised from the 1.5 / 1.0 m/s first-flight values to fly the same path
# faster; override per run with the MAX_SPEED_MPS / MAX_VSPEED_MPS env vars (no code edit),
# e.g. `MAX_SPEED_MPS=4 python main.py`. If the drone can't actually reach the commanded
# speed it has saturated its lean cap -- raise controller.MAX_LEAN_RAD / KP_LEAN.
# --------------------------------------------------------------------------------------
MAX_SPEED = float(os.environ.get('MAX_SPEED_MPS', '3.0'))    # m/s cap on commanded velocity magnitude
MAX_VSPEED = float(os.environ.get('MAX_VSPEED_MPS', '1.5'))  # m/s cap on the vertical (climb/descend) component
KP_POS = 0.8               # commanded speed = KP_POS * distance-to-waypoint (then capped)

# --------------------------------------------------------------------------------------
# Safety guards (kept from the reactive planner -- pure client-side fail-safes).
# --------------------------------------------------------------------------------------
TELEM_TIMEOUT_NS = 500_000_000     # 500 ms without a pose -> hover (we'd be flying blind)
MAX_ALT_M = 15.0                   # height (m above the arm point) beyond which we recover
# Distance to the ACTIVE waypoint beyond which the horizontal loop has clearly run away
# (log run_1781506925 flew 180 m off a 5 m square) -> stop commanding translation and
# hover so the velocity loop brakes back. MUST exceed the mission's LONGEST leg or the
# drone hovers at one waypoint instead of flying to the next (the captured course stalled
# at wp1 with the old 25 m default, since wp1->wp2 is 29 m). Default 60 m clears the
# captured course (longest leg ~39 m) with margin while still catching a true 100 m+
# runaway; override per run with the MAX_WP_DIST_M env var.
MAX_WP_DIST_M = float(os.environ.get('MAX_WP_DIST_M', '60.0'))

# Legacy aliases kept so the run logger's gains snapshot stays populated.
PASS_THROUGH_DIST = DEFAULT_ARRIVE_RADIUS_M
CONF_MIN = 0.0


def _quat_yaw(q):
    """Yaw (rad, NED) from a quaternion ``(w, x, y, z)`` -- the planner's only fallback
    heading source when no ATTITUDE message is available (odometry only)."""
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class Planner:
    """Follows the preplanned waypoint mission on ``shared_data['mission']``."""

    def __init__(self, data):
        self.data = data
        if 'lock' not in self.data:
            self.data['lock'] = threading.RLock()
        self.mission = self.data.get('mission')
        self._idx = 0                # index of the waypoint we are currently flying to
        self._reached_since = None   # time_ns the current waypoint was first satisfied
        self._complete = False       # True once a non-looping mission's last wp is reached

    @staticmethod
    def _wrap(a):
        """Wrap an angle to (-pi, pi]."""
        return math.atan2(math.sin(a), math.cos(a))

    def _snapshot(self):
        """Atomically read the telemetry the planner needs."""
        with self.data['lock']:
            return {
                'attitude': self.data.get('attitude'),
                'odometry': self.data.get('odometry'),
                'position_ned': self.data.get('position_ned'),
            }

    @staticmethod
    def _pose(s):
        """Return ``(position_ned tuple|None, yaw rad|None, telem_ts ns|None)``.

        Prefer LOCAL_POSITION_NED for the absolute position; fall back to ODOMETRY.
        Prefer the ATTITUDE Euler yaw; fall back to the odometry quaternion.
        """
        pos = s['position_ned']
        odo = s['odometry']
        att = s['attitude']

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

    def _current_wp(self):
        if not self.mission or not self.mission.waypoints:
            return None
        i = min(self._idx, len(self.mission.waypoints) - 1)
        return self.mission.waypoints[i]

    def _advance(self):
        """Step to the next waypoint, wrapping or completing at the end."""
        n = len(self.mission.waypoints)
        if self._idx < n - 1:
            self._idx += 1
        elif self.mission.loop:
            self._idx = 0
        else:
            self._complete = True   # stay on the last waypoint and hold
        self._reached_since = None

    def _hover(self, yaw, source, now):
        return {'mode': 'velocity', 'vel_ned': (0.0, 0.0, 0.0),
                'yaw': float(yaw), 'source': source, 'ts': now}

    def _publish(self, target):
        with self.data['lock']:
            self.data['target'] = target
        return target

    def compute_target(self):
        """Compute and publish the velocity setpoint for this tick. Returns the target."""
        now = time.time_ns()
        s = self._snapshot()
        position, yaw, telem_ts = self._pose(s)
        yaw_hold = yaw if yaw is not None else 0.0

        # Watchdog: no recent pose at all -> hover (we'd otherwise be flying blind).
        if position is None or telem_ts is None or (now - telem_ts) > TELEM_TIMEOUT_NS:
            return self._publish(self._hover(yaw_hold, 'watchdog_hover', now))

        # Altitude-envelope fail-safe: NED z is negative-up; past the ceiling the vertical
        # loop has run away -> abandon the waypoint and descend until back below it.
        altitude = -float(position[2])
        if altitude > MAX_ALT_M:
            return self._publish({'mode': 'velocity', 'vel_ned': (0.0, 0.0, MAX_VSPEED),
                                  'yaw': yaw_hold, 'range_m': altitude,
                                  'source': 'alt_guard', 'ts': now})

        wp = self._current_wp()
        if wp is None:
            return self._publish(self._hover(yaw_hold, 'no_mission', now))

        # Arrival test against the (possibly per-waypoint) tolerances.
        p = np.asarray(position, float)
        offset = np.asarray(wp.pos, float) - p
        dist = float(np.linalg.norm(offset))

        arrive_r = wp.arrive_radius if wp.arrive_radius is not None else DEFAULT_ARRIVE_RADIUS_M
        yaw_tol = wp.yaw_tol if wp.yaw_tol is not None else DEFAULT_YAW_TOL_RAD
        dwell = wp.dwell_s if wp.dwell_s is not None else DEFAULT_DWELL_S

        # Horizontal-envelope fail-safe: if we are this far from the waypoint we are
        # supposed to be flying TO, the horizontal loop has run away -> stop commanding
        # translation and hover. The velocity loop then leans backward to brake us back
        # toward the course (vs. the alt guard, which only catches vertical runaway).
        if dist > MAX_WP_DIST_M:
            return self._publish({'mode': 'velocity', 'vel_ned': (0.0, 0.0, 0.0),
                                  'yaw': yaw_hold, 'range_m': dist,
                                  'source': 'dist_guard', 'wp_index': self._idx, 'ts': now})

        # A waypoint with yaw=None holds the current heading (no commanded turn) and does
        # not gate arrival on heading -- used for a straight-up takeoff / pure translation.
        if wp.yaw is None:
            yaw_ok = True
        else:
            yaw_ok = abs(self._wrap(wp.yaw - yaw_hold)) <= yaw_tol

        reached = dist <= arrive_r and yaw_ok
        if reached:
            if self._reached_since is None:
                self._reached_since = now
            if (now - self._reached_since) >= int(dwell * 1e9):
                self._advance()
                # Recompute against the waypoint we just stepped to so there is no stutter.
                wp = self._current_wp()
                offset = np.asarray(wp.pos, float) - p
                dist = float(np.linalg.norm(offset))
        else:
            self._reached_since = None

        # Heading to command: the (possibly new) waypoint's yaw, or HOLD the current
        # heading when its yaw is None.
        yaw_cmd = wp.yaw if wp.yaw is not None else yaw_hold

        # P-control velocity toward the (current) waypoint: speed scales with the distance
        # remaining and is capped, so the drone decelerates into the point and settles.
        if dist > 1e-6:
            speed = min(MAX_SPEED, KP_POS * dist)
            vel = offset * (speed / dist)
        else:
            vel = np.zeros(3)
        vn, ve, vd = float(vel[0]), float(vel[1]), float(vel[2])
        vd = max(-MAX_VSPEED, min(MAX_VSPEED, vd))   # cap vertical separately

        return self._publish({'mode': 'velocity',
                              'vel_ned': (vn, ve, vd),
                              'yaw': float(yaw_cmd),
                              'range_m': dist,
                              'source': wp.name or f'wp{self._idx}',
                              'wp_index': self._idx,
                              'ts': now})
