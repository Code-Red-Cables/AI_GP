"""Offline test: the spline planner flies the waypoints CONTINUOUSLY (no sim).

We run the REAL ``SplinePlanner`` against the same simple kinematic drone used by
``test_waypoint_mission.py`` (integrate the commanded NED velocity into position; slew yaw
toward the command at the controller's yaw-rate cap) and assert:

* the built spline passes THROUGH every waypoint (it is an interpolating curve);
* the curvature-/brake-aware speed profile slows for corners and the final waypoint, brakes
  feasibly (within A_LON), and is a NO-OP where cruise is feasible (so slow runs are
  unchanged) -- this is what stops the drone overshooting turns at high cruise speed;
* flown end-to-end the drone passes within tolerance of every waypoint and settles on the
  last; on a straight path it holds cruise (never stalls at an intermediate waypoint);
* the commanded velocity respects the speed caps.

The kinematic drone tracks the commanded velocity perfectly, so the speed tests are
deterministic. We pin ``spline_planner.CRUISE_SPEED`` to a known value so the assertions
don't depend on whatever cruise speed is currently set in config.py.

Run with any Python that has numpy:

    python test_spline_mission.py
"""

import math
import threading
import time

import numpy as np

from mission import Mission, Waypoint, DEFAULT_ARRIVE_RADIUS_M
import spline_planner as sp
from spline_planner import SplinePlanner
from guidance.path import build_spline_path, path_curvature, speed_profile
from config import (MAX_SPEED, MAX_VSPEED, LOOKAHEAD_M, LOOKAHEAD_TIME, LOOKAHEAD_MAX,
                    A_LAT_MAX, A_LON_MAX)

YAW_SLEW_RAD_S = math.radians(70.0)   # matches controller.YAW_RATE_MAX
TEST_CRUISE = 6.0                     # pin cruise for deterministic, config-independent tests


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _level_mission():
    """A level (constant-altitude) path with two gentle bends, all headings = North.
    Constant altitude keeps the vertical command ~0; the bends exercise corner rounding."""
    d = -3.0
    pts = [(0.0, 0.0), (15.0, 0.0), (30.0, 4.0), (45.0, 4.0), (60.0, 0.0)]
    return Mission([Waypoint(n, e, d, 0.0, name=f"wp{i}") for i, (n, e) in enumerate(pts)],
                   name="spline_test")


def _straight_mission():
    """A dead-straight level path -- curvature ~0, so the speed profile is cruise-limited
    only (used to prove the drone never stalls at an intermediate waypoint)."""
    d = -3.0
    pts = [(0.0, 0.0), (20.0, 0.0), (40.0, 0.0), (60.0, 0.0), (80.0, 0.0)]
    return Mission([Waypoint(n, e, d, 0.0, name=f"wp{i}") for i, (n, e) in enumerate(pts)],
                   name="straight_test")


def _fly(mission, *, cruise=TEST_CRUISE, start=(0.0, 0.0, -3.0), start_yaw_deg=0.0,
         dt=0.02, max_iters=60000):
    """Fly the mission with the real SplinePlanner + a kinematic drone (cruise pinned).

    Returns ``(planner, min_dist_to_each_wp[list], speed_samples[list of (remaining, speed)])``.
    """
    import config
    config.CRUISE_SPEED = cruise                       # pin before the planner builds its profile
    config.FINISH_SPEED = 0.0                          # tests assert the settle-on-last behaviour
    data = {"lock": threading.RLock(), "mission": mission}
    pl = SplinePlanner(data)

    pos = np.array(start, float)
    yaw = math.radians(start_yaw_deg)
    min_d = [float("inf")] * len(mission.waypoints)
    speeds = []                                    # (remaining_arc_len, horiz_speed)

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
        j = min(int(np.searchsorted(cum_s, wp_s[i])), len(pts) - 1)
        err = float(np.linalg.norm(pts[j] - p))
        assert err < 1e-6, f"spline misses waypoint {i} by {err:.4f} m"
    print(f"PASS spline interpolates  {len(positions)} control points, "
          f"path length {cum_s[-1]:.1f} m")


def test_speed_profile_slows_for_corners():
    """The profile dips for a corner, reaches cruise on the straight, brakes feasibly, stops."""
    # An L-shape: two 40 m straights meeting at a 90deg corner.
    pts = [np.array([0., 0., 0.]), np.array([40., 0., 0.]), np.array([40., 40., 0.])]
    pp, cum_s, _ = build_spline_path(pts)
    curv = path_curvature(pp, cum_s)
    cruise, a_lat, a_lon = 10.0, 6.0, 4.0
    v = speed_profile(curv, cum_s, cruise, a_lat, a_lon, end_speed=0.0)

    # Curvature peaks near the corner (arc-length midpoint), ~0 on the straights.
    corner_i = int(np.argmax(curv))
    assert curv[corner_i] > 0.05, "corner should have real curvature"
    assert abs(v[corner_i]) < cruise, "must slow below cruise AT the corner"
    assert v.max() >= 0.99 * cruise, "must reach full cruise on the long straight"
    assert v[-1] < 1e-6, "end_speed=0 must brake to a stop at the final waypoint"
    # The required lateral accel at the corner is within budget (that is the speed cap).
    assert v[corner_i] ** 2 * curv[corner_i] <= a_lat + 1e-6, "corner speed must respect A_LAT"
    # Braking is feasible everywhere: each step's slowdown is within A_LON.
    for i in range(len(v) - 1):
        ds = cum_s[i + 1] - cum_s[i]
        assert v[i] <= math.sqrt(v[i + 1] ** 2 + 2 * a_lon * ds) + 1e-6, "infeasible braking"
    # A positive end_speed lets the drone CROSS the finish at speed instead of stopping.
    vf = speed_profile(curv, cum_s, cruise, a_lat, a_lon, end_speed=5.0)
    assert vf[-1] >= 5.0 - 1e-6, "end_speed>0 must cross the finish at that speed"
    assert vf[0] >= v[0] - 1e-6, "crossing the finish is never slower than stopping"
    print(f"PASS profile corners      corner v={v[corner_i]:.1f} < cruise={cruise:.0f}, "
          f"stops at end (0.0) / crosses at {vf[-1]:.0f} (end_speed=5)")


def test_speed_profile_noop_when_feasible():
    """On a gentle path at low cruise, the profile is just the cruise cap (slow runs intact)."""
    pts = [np.array([0., 0., 0.]), np.array([30., 2., 0.]), np.array([60., 0., 0.])]
    pp, cum_s, _ = build_spline_path(pts)
    curv = path_curvature(pp, cum_s)
    cruise = 2.0
    v = speed_profile(curv, cum_s, cruise, A_LAT_MAX, A_LON_MAX, end_speed=0.0)
    brake_dist = cruise ** 2 / (2 * A_LON_MAX)        # final-approach ramp length
    interior = v[cum_s < cum_s[-1] - brake_dist - 1.0]
    assert interior.size and np.all(interior >= 0.999 * cruise), \
        "a gentle path at low cruise must hold cruise everywhere but the final approach"
    print(f"PASS profile no-op        gentle path holds cruise {cruise:.1f} m/s "
          f"(no corner limiting)")


def test_spline_flight_visits_waypoints():
    """Flown end-to-end, the drone passes near every waypoint and settles on the last."""
    m = _level_mission()
    pl, min_d, _ = _fly(m)

    assert pl._complete, "spline mission never completed"
    # Speed-scaled lookahead at the test cruise sets how much pure pursuit rounds corners.
    eff_lookahead = min(LOOKAHEAD_MAX, max(LOOKAHEAD_M, LOOKAHEAD_TIME * TEST_CRUISE))
    inter_tol = eff_lookahead + 0.5    # lookahead bounds the corner-cut miss
    for i in range(len(m.waypoints) - 1):
        assert min_d[i] <= inter_tol, (
            f"wp{i}: closest approach {min_d[i]:.2f} m > {inter_tol:.2f} m")
    assert min_d[-1] <= DEFAULT_ARRIVE_RADIUS_M + 0.3, f"final wp: settled {min_d[-1]:.2f} m away"
    print(f"PASS spline flight        passed all {len(m.waypoints)} waypoints "
          f"(worst intermediate miss {max(min_d[:-1]):.2f} m), settled on the last")


def test_continuous_cruise_no_stall():
    """On a straight path the command holds cruise and never stalls at an intermediate wp.

    The stop-at-each waypoint planner dips to ~0 m/s at every interior waypoint; the spline
    planner must keep moving. Curvature is ~0 here, so only the final-approach braking slows
    it -- the interior must stay at (near) cruise.
    """
    m = _straight_mission()
    pl, _, speeds = _fly(m, cruise=TEST_CRUISE)

    brake_dist = TEST_CRUISE ** 2 / (2 * A_LON_MAX)
    cruise = [spd for rem, spd in speeds if rem is not None and rem > brake_dist + 1.0]
    assert cruise, "no cruise-region samples captured"
    slowest = min(cruise)
    assert slowest >= 0.95 * TEST_CRUISE, (
        f"cruise dipped to {slowest:.2f} m/s (< 0.95 * {TEST_CRUISE}) -- stalling at "
        f"waypoints instead of flying through them")
    print(f"PASS continuous cruise    min straight-cruise speed {slowest:.2f} m/s "
          f">= {0.95 * TEST_CRUISE:.2f} (never stalls at a waypoint)")


def test_speed_caps():
    """Commanded velocity never exceeds the configured speed caps."""
    m = _level_mission()
    sp.CRUISE_SPEED = TEST_CRUISE
    data = {"lock": threading.RLock(), "mission": m}
    p = SplinePlanner(data)
    pos = np.array([0.0, 0.0, -3.0])
    worst_h = worst_v = worst_3d = 0.0
    for _ in range(20000):
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
        pos += np.array([vn, ve, vd]) * 0.02
    print(f"PASS speed caps           max |v|={worst_3d:.2f} <= {MAX_SPEED} m/s, "
          f"max vert={worst_v:.2f} <= {MAX_VSPEED} m/s")


if __name__ == "__main__":
    test_spline_interpolates_waypoints()
    test_speed_profile_slows_for_corners()
    test_speed_profile_noop_when_feasible()
    test_spline_flight_visits_waypoints()
    test_continuous_cruise_no_stall()
    test_speed_caps()
    print("ALL SPLINE MISSION TESTS PASSED")
