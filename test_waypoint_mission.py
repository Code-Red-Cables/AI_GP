"""Offline test: the preplanned waypoint planner flies the square mission.

No simulator. We run the REAL ``Planner`` against a simple kinematic model of the drone
(integrate the commanded NED velocity into position; slew yaw toward the commanded
heading at the controller's yaw-rate cap) and assert the drone visits every waypoint --
the corners of a counter-clockwise square -- in order, at the correct heading, then stops.
This validates point-to-point accuracy and the per-waypoint orientation control with no
vision in the loop.

Run with any Python that has numpy:

    python test_waypoint_mission.py
"""

import math
import threading
import time

import numpy as np

from mission import (Mission, Waypoint, square_mission, load_mission, save_mission,
                     DEFAULT_ARRIVE_RADIUS_M, DEFAULT_YAW_TOL_RAD)
from planner import Planner, MAX_SPEED, MAX_VSPEED

YAW_SLEW_RAD_S = math.radians(70.0)   # matches controller.YAW_RATE_MAX


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _fly(mission, *, start_yaw_deg=35.0, dt=0.05, max_iters=40000):
    """Fly the mission with the real planner + a kinematic drone. Returns the list of
    completed waypoints as ``(idx, name, pos_when_reached, yaw_when_reached)``."""
    # dwell=0 so the test does not depend on wall-clock; arrival alone advances.
    for wp in mission.waypoints:
        wp.dwell_s = 0.0

    data = {"lock": threading.RLock(), "mission": mission}
    pl = Planner(data)

    pos = np.array([0.0, 0.0, 0.0])      # armed on the ground at the origin
    yaw = math.radians(start_yaw_deg)    # start mis-aligned to prove it turns to heading
    completed = []
    prev_idx, prev_complete = pl._idx, pl._complete

    for _ in range(max_iters):
        now = time.time_ns()
        with data["lock"]:
            data["attitude"] = {"roll": 0.0, "pitch": 0.0, "yaw": yaw, "ts": now}
            data["position_ned"] = {"x": pos[0], "y": pos[1], "z": pos[2],
                                    "vx": 0.0, "vy": 0.0, "vz": 0.0, "ts": now}

        target = pl.compute_target()

        # Record a waypoint completion (index advanced, or the final wp just completed).
        if pl._idx != prev_idx:
            wp = mission.waypoints[prev_idx]
            completed.append((prev_idx, wp.name, pos.copy(), yaw))
            prev_idx = pl._idx
        if pl._complete and not prev_complete:
            wp = mission.waypoints[-1]
            completed.append((len(mission.waypoints) - 1, wp.name, pos.copy(), yaw))
            prev_complete = True
            break

        # Kinematic integration: perfect velocity tracking + first-order yaw slew.
        vn, ve, vd = target["vel_ned"]
        pos += np.array([vn, ve, vd]) * dt
        yaw_err = _wrap(target["yaw"] - yaw)
        yaw += max(-YAW_SLEW_RAD_S * dt, min(YAW_SLEW_RAD_S * dt, yaw_err))

    return pl, completed


def test_square_structure():
    """square_mission builds the CCW corners, every one facing forward (yaw=None)."""
    m = square_mission(side_m=5.0, altitude_m=2.0)
    names = [w.name for w in m.waypoints]
    assert names == ["takeoff", "cornerB", "cornerC", "cornerD", "cornerA"], names
    # The (n, e) corners the waypoints sit on. EVERY waypoint holds the boot heading
    # (yaw=None) -- the drone strafes the square facing forward, never turning.
    expect = {
        "takeoff": (0.0, 0.0),
        "cornerB": (5.0, 0.0),
        "cornerC": (5.0, -5.0),
        "cornerD": (0.0, -5.0),
        "cornerA": (0.0, 0.0),
    }
    for w in m.waypoints:
        en, ee = expect[w.name]
        assert abs(w.pos[0] - en) < 1e-9 and abs(w.pos[1] - ee) < 1e-9, w
        assert abs(w.pos[2] - (-2.0)) < 1e-9, w
        assert w.yaw is None, f"{w.name} should hold heading (yaw=None), got {w.yaw}"
    print("PASS square structure  5 waypoints, CCW corners, all facing forward")


def test_square_flight():
    """Flown end-to-end, the planner visits all 5 waypoints in order, on-position."""
    m = square_mission(side_m=5.0, altitude_m=2.0)
    pl, completed = _fly(m)

    assert pl._complete, "mission never completed"
    idxs = [c[0] for c in completed]
    assert idxs == list(range(len(m.waypoints))), f"out-of-order / missing: {idxs}"

    pos_tol = DEFAULT_ARRIVE_RADIUS_M + 0.2     # arrival radius + integration slop
    yaw_tol = DEFAULT_YAW_TOL_RAD + math.radians(2.0)
    for idx, name, pos, yaw in completed:
        wp = m.waypoints[idx]
        derr = float(np.linalg.norm(np.asarray(wp.pos) - pos))
        assert derr <= pos_tol, f"{name}: reached {pos} but wp {wp.pos} (err {derr:.2f}m)"
        if wp.yaw is not None:   # yaw=None waypoints (takeoff) hold heading, not gated on it
            yerr = abs(_wrap(wp.yaw - yaw))
            assert yerr <= yaw_tol, f"{name}: heading off by {math.degrees(yerr):.1f}deg"
    print(f"PASS square flight     visited {len(completed)} waypoints in order, "
          f"each within {pos_tol:.2f}m / {math.degrees(yaw_tol):.0f}deg")


def test_command_caps():
    """Commanded velocity never exceeds the configured speed caps."""
    m = square_mission(side_m=8.0, altitude_m=3.0)
    for wp in m.waypoints:
        wp.dwell_s = 0.0
    data = {"lock": threading.RLock(), "mission": m}
    pl = Planner(data)
    pos = np.array([0.0, 0.0, 0.0])
    worst_h = worst_v = 0.0
    for _ in range(2000):
        now = time.time_ns()
        with data["lock"]:
            data["attitude"] = {"roll": 0, "pitch": 0, "yaw": 0.0, "ts": now}
            data["position_ned"] = {"x": pos[0], "y": pos[1], "z": pos[2],
                                    "vx": 0, "vy": 0, "vz": 0, "ts": now}
        t = pl.compute_target()
        vn, ve, vd = t["vel_ned"]
        worst_h = max(worst_h, math.hypot(vn, ve))
        worst_v = max(worst_v, abs(vd))
        pos += np.array([vn, ve, vd]) * 0.05
        assert math.sqrt(vn * vn + ve * ve + vd * vd) <= MAX_SPEED + 1e-6
        assert abs(vd) <= MAX_VSPEED + 1e-6
    print(f"PASS command caps      max horiz={worst_h:.2f}<= {MAX_SPEED} m/s, "
          f"max vert={worst_v:.2f} <= {MAX_VSPEED} m/s")


def test_clockwise_mirror():
    """counter_clockwise=False mirrors the square's shape across the East axis."""
    m = square_mission(side_m=5.0, counter_clockwise=False)
    # The second corner (after the +North leg) should now sit on the +East side instead
    # of -East, and -- like every waypoint -- still hold the forward heading (yaw=None).
    cornerC = next(w for w in m.waypoints if w.name == "cornerC")
    assert cornerC.pos[1] > 0, f"clockwise corner should be +East, got {cornerC.pos}"
    assert all(w.yaw is None for w in m.waypoints), "clockwise should still face forward"
    print("PASS clockwise mirror  East-side square, still facing forward")


def test_mission_json_roundtrip(tmp_path="_mission_roundtrip.json"):
    """A mission survives save -> load unchanged."""
    import os
    m = square_mission(side_m=4.0, altitude_m=2.5, loop=True)
    try:
        save_mission(m, tmp_path)
        m2 = load_mission(tmp_path)
        assert m2 is not None and len(m2) == len(m) and m2.loop == m.loop
        for a, b in zip(m.waypoints, m2.waypoints):
            assert np.allclose(a.pos, b.pos)
            if a.yaw is None:
                assert b.yaw is None, f"yaw=None should survive round-trip, got {b.yaw}"
            else:
                assert b.yaw is not None and abs(_wrap(a.yaw - b.yaw)) < 1e-6
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    print("PASS mission json      save/load round-trips")


if __name__ == "__main__":
    test_square_structure()
    test_square_flight()
    test_command_caps()
    test_clockwise_mirror()
    test_mission_json_roundtrip()
    print("ALL WAYPOINT MISSION TESTS PASSED")
