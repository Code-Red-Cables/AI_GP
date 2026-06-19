"""Offline test: the spline planner flies the waypoints CONTINUOUSLY (no sim).

We run the REAL ``SplinePlanner`` against the same simple kinematic drone used by
``test_waypoint_mission.py`` (integrate the commanded NED velocity into position; slew yaw
toward the command at the controller's yaw-rate cap) and assert:

* the built spline passes THROUGH every waypoint (it is an interpolating curve);
* flown end-to-end, the drone passes within tolerance of every waypoint in order and
  settles on the last one;
* the velocity command NEVER decelerates to a stop at an intermediate waypoint -- the
  whole point of the spline path -- it holds (near) cruise speed across the cruise region
  and only tapers on the final approach;
* the commanded velocity respects the speed caps.

Run with any Python that has numpy:

    python test_spline_mission.py
"""

import math
import threading
import time

import numpy as np

from mission import Mission, Waypoint, DEFAULT_ARRIVE_RADIUS_M
from spline_planner import SplinePlanner, build_spline_path, KP_POS_END
from config import MAX_SPEED, MAX_VSPEED, CRUISE_SPEED, LOOKAHEAD_M

YAW_SLEW_RAD_S = math.radians(70.0)   # matches controller.YAW_RATE_MAX


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _level_mission():
    """A level (constant-altitude) path with two gentle bends, all headings = North.

    Constant altitude keeps the vertical command ~0 so the cruise-speed assertion reads
    the horizontal speed cleanly; the bends exercise the spline's corner rounding.
    """
    d = -3.0
    pts = [(0.0, 0.0), (15.0, 0.0), (30.0, 4.0), (45.0, 4.0), (60.0, 0.0)]
    return Mission([Waypoint(n, e, d, 0.0, name=f"wp{i}") for i, (n, e) in enumerate(pts)],
                   name="spline_test")


def _fly(mission, *, start=(0.0, 0.0, -3.0), start_yaw_deg=0.0, dt=0.05, max_iters=40000):
    """Fly the mission with the real SplinePlanner + a kinematic drone.

    Returns ``(planner, min_dist_to_each_wp[list], speed_samples[list of (remaining, speed)])``.
    """
    data = {"lock": threading.RLock(), "mission": mission}
    pl = SplinePlanner(data)

    pos = np.array(start, float)
    yaw = math.radians(start_yaw_deg)
    min_d = [float("inf")] * len(mission.waypoints)
    speeds = []                                   # (remaining_arc_len, horiz_speed)

    for _ in range(max_iters):
        now = time.time_ns()
        with data["lock"]:
            data["attitude"] = {"roll": 0.0, "pitch": 0.0, "yaw": yaw, "ts": now}
            data["position_ned"] = {"x": pos[0], "y": pos[1], "z": pos[2],
                                    "vx": 0.0, "vy": 0.0, "vz": 0.0, "ts": now}
        target = pl.compute_target()

        for i, wp in enumerate(mission.waypoints):
            min_d[i] = min(min_d[i], float(np.linalg.norm(np.asarray(wp.pos) - pos)))

        vn, ve, vd = target["vel_ned"]
        speeds.append((target.get("range_m"), math.hypot(vn, ve)))

        if pl._complete:
            break

        pos += np.array([vn, ve, vd]) * dt
        yaw_err = _wrap(target["yaw"] - yaw)
        yaw += max(-YAW_SLEW_RAD_S * dt, min(YAW_SLEW_RAD_S * dt, yaw_err))

    return pl, min_d, speeds


def test_spline_interpolates_waypoints():
    """The built spline passes THROUGH every control point, arc length is monotone."""
    m = _level_mission()
    positions = [np.asarray(w.pos, float) for w in m.waypoints]
    pts, cum_s, wp_s = build_spline_path(positions)

    assert np.all(np.diff(cum_s) >= -1e-9), "arc length must be non-decreasing"
    assert len(wp_s) == len(positions)
    for i, p in enumerate(positions):
        j = int(np.searchsorted(cum_s, wp_s[i]))
        j = min(j, len(pts) - 1)
        err = float(np.linalg.norm(pts[j] - p))
        assert err < 1e-6, f"spline misses waypoint {i} by {err:.4f} m"
    print(f"PASS spline interpolates  {len(positions)} control points, "
          f"path length {cum_s[-1]:.1f} m")


def test_spline_flight_visits_waypoints():
    """Flown end-to-end, the drone passes near every waypoint and settles on the last."""
    m = _level_mission()
    pl, min_d, _ = _fly(m)

    assert pl._complete, "spline mission never completed"
    # Pure pursuit rounds corners, so intermediate waypoints are passed CLOSE but not dead
    # on; the lookahead bounds the miss. The final waypoint is settled on precisely.
    inter_tol = LOOKAHEAD_M + 0.5
    for i in range(len(m.waypoints) - 1):
        assert min_d[i] <= inter_tol, (
            f"wp{i}: closest approach {min_d[i]:.2f} m > {inter_tol:.2f} m")
    assert min_d[-1] <= DEFAULT_ARRIVE_RADIUS_M + 0.3, (
        f"final wp: settled {min_d[-1]:.2f} m away")
    print(f"PASS spline flight        passed all {len(m.waypoints)} waypoints "
          f"(worst intermediate miss {max(min_d[:-1]):.2f} m), settled on the last")


def test_continuous_cruise_speed():
    """The command holds ~cruise across the path and never stalls at an intermediate wp.

    The waypoint planner would dip to ~0 m/s at each of the interior waypoints; the spline
    planner must keep moving. We check the cruise region (excluding the final-approach
    taper zone, where slowing down IS intended) stays at/above most of cruise speed.
    """
    m = _level_mission()
    pl, _, speeds = _fly(m)

    taper_zone = CRUISE_SPEED / KP_POS_END        # remaining arc length where taper starts
    cruise = [spd for rem, spd in speeds if rem is not None and rem > taper_zone + 0.5]
    assert cruise, "no cruise-region samples captured"
    slowest = min(cruise)
    # Level path -> vertical command ~0 -> horizontal speed should equal cruise; allow a
    # small margin for the carrot direction during the bends.
    assert slowest >= 0.9 * CRUISE_SPEED, (
        f"cruise dipped to {slowest:.2f} m/s (< 0.9 * {CRUISE_SPEED}) -- it is stalling at "
        f"waypoints, not flying through them")
    print(f"PASS continuous cruise    min cruise-region speed {slowest:.2f} m/s "
          f">= {0.9 * CRUISE_SPEED:.2f} (never stalls at a waypoint)")


def test_speed_caps():
    """Commanded velocity never exceeds the configured speed caps."""
    m = _level_mission()
    pl, _, _ = _fly(m)
    # Re-fly capturing full 3D velocity to check both caps.
    data = {"lock": threading.RLock(), "mission": _level_mission()}
    p = SplinePlanner(data)
    pos = np.array([0.0, 0.0, -3.0])
    worst_h = worst_v = worst_3d = 0.0
    for _ in range(8000):
        now = time.time_ns()
        with data["lock"]:
            data["attitude"] = {"roll": 0, "pitch": 0, "yaw": 0.0, "ts": now}
            data["position_ned"] = {"x": pos[0], "y": pos[1], "z": pos[2],
                                    "vx": 0, "vy": 0, "vz": 0, "ts": now}
        t = p.compute_target()
        vn, ve, vd = t["vel_ned"]
        worst_h = max(worst_h, math.hypot(vn, ve))
        worst_v = max(worst_v, abs(vd))
        worst_3d = max(worst_3d, math.sqrt(vn * vn + ve * ve + vd * vd))
        assert math.sqrt(vn * vn + ve * ve + vd * vd) <= MAX_SPEED + 1e-6
        assert abs(vd) <= MAX_VSPEED + 1e-6
        if p._complete:
            break
        pos += np.array([vn, ve, vd]) * 0.05
    print(f"PASS speed caps           max |v|={worst_3d:.2f}<= {MAX_SPEED} m/s, "
          f"max vert={worst_v:.2f} <= {MAX_VSPEED} m/s")


if __name__ == "__main__":
    test_spline_interpolates_waypoints()
    test_spline_flight_visits_waypoints()
    test_continuous_cruise_speed()
    test_speed_caps()
    print("ALL SPLINE MISSION TESTS PASSED")
