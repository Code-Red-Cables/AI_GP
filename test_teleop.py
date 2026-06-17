"""Offline tests for manual keyboard teleop (no sim, no real keyboard).

Exercises TeleopPlanner's intent->velocity mapping (body sticks rotated along the physical
nose direction), the vertical/yaw axes, the telemetry watchdog, and KeyboardTeleop's
B-capture writing a mission JSON that load_mission round-trips. Run: `python test_teleop.py`.
"""

import math
import os
import tempfile
import threading
import time

import teleop
from mission import load_mission

SPEED = teleop.TELEOP_SPEED
VSPEED = teleop.TELEOP_VSPEED
TOL = 1e-6


def _data():
    return {'lock': threading.RLock()}


def _set_pose(data, n=0.0, e=0.0, d=-2.0, yaw=0.0):
    now = time.time_ns()
    with data['lock']:
        data['position_ned'] = {'x': n, 'y': e, 'z': d, 'vx': 0.0, 'vy': 0.0,
                                'vz': 0.0, 'ts': now}
        data['attitude'] = {'roll': 0.0, 'pitch': 0.0, 'yaw': yaw, 'ts': now}


def _set_intent(data, fwd=0.0, right=0.0, up=0.0, turn=0.0):
    with data['lock']:
        data['teleop'] = {'fwd': fwd, 'right': right, 'up': up, 'turn': turn,
                          'ts': time.time_ns()}


def _approx(a, b, tol=1e-4):
    return abs(a - b) <= tol


def test_horizontal_mapping():
    """W/A/S/D map along the PHYSICAL nose (cos y, -sin y) for forward, (sin y, cos y) right."""
    cases = [
        # (yaw_deg, fwd, right, expected_vn, expected_ve)
        (0,    1, 0,  +SPEED, 0.0),     # facing North -> W goes North
        (0,    0, 1,  0.0,    +SPEED),  # right of North is East
        (0,   -1, 0,  -SPEED, 0.0),     # S goes South
        (90,   1, 0,  0.0,    -SPEED),  # left-handed yaw: nose at +90 points West physically
        (90,   0, 1,  +SPEED, 0.0),     # right of that is North
        (-90,  1, 0,  0.0,    +SPEED),  # nose at -90 points East physically
    ]
    for yaw_deg, fwd, right, evn, eve in cases:
        data = _data()
        p = teleop.TeleopPlanner(data)
        _set_pose(data, yaw=math.radians(yaw_deg))
        _set_intent(data, fwd=fwd, right=right)
        t = p.compute_target()
        vn, ve, vd = t['vel_ned']
        assert _approx(vn, evn) and _approx(ve, eve), \
            f"yaw={yaw_deg} fwd={fwd} right={right}: got ({vn:.3f},{ve:.3f}) want ({evn:.3f},{eve:.3f})"
        assert _approx(vd, 0.0), f"no vertical stick should mean vd=0, got {vd}"
    print("PASSED test_horizontal_mapping")


def test_diagonal_is_capped():
    """Holding two horizontal keys must not exceed the single-axis speed cap."""
    data = _data()
    p = teleop.TeleopPlanner(data)
    _set_pose(data, yaw=0.0)
    _set_intent(data, fwd=1, right=1)
    vn, ve, vd = p.compute_target()['vel_ned']
    assert _approx(math.hypot(vn, ve), SPEED), f"diagonal speed {math.hypot(vn, ve)} != {SPEED}"
    print("PASSED test_diagonal_is_capped")


def test_vertical_axis():
    """Space climbs (vd negative, up in NED); Ctrl descends (vd positive)."""
    data = _data()
    p = teleop.TeleopPlanner(data)
    _set_pose(data, yaw=0.0)
    _set_intent(data, up=1)
    assert _approx(p.compute_target()['vel_ned'][2], -VSPEED), "Space should climb (vd<0)"
    _set_intent(data, up=-1)
    assert _approx(p.compute_target()['vel_ned'][2], +VSPEED), "Ctrl should descend (vd>0)"
    print("PASSED test_vertical_axis")


def test_yaw_integration():
    """Q/E integrate the heading setpoint; first tick only seeds it (no jump)."""
    data = _data()
    p = teleop.TeleopPlanner(data)
    _set_pose(data, yaw=0.0)
    _set_intent(data, turn=1)
    p.compute_target()                       # tick 1: seeds desired_yaw, dt=0
    assert _approx(p._desired_yaw, 0.0), f"first tick should not turn, got {p._desired_yaw}"
    p._last_t = time.time() - 0.1            # pretend 0.1 s elapsed since last tick
    y = p.compute_target()['yaw']
    expected = teleop.TELEOP_YAW_RATE * 0.1
    assert y > 0 and _approx(y, expected, tol=expected * 0.5), \
        f"turn=+1 over 0.1s should add ~{expected:.3f} rad, got {y:.3f}"
    # Reverse: turn left decreases the heading.
    _set_intent(data, turn=-1)
    p._last_t = time.time() - 0.1
    assert p.compute_target()['yaw'] < y, "turn=-1 should decrease the heading"
    print("PASSED test_yaw_integration")


def test_watchdog_hover():
    """Stale telemetry -> hover (zero velocity), and the heading setpoint is reset."""
    data = _data()
    p = teleop.TeleopPlanner(data)
    stale = time.time_ns() - 10 * teleop.TELEM_TIMEOUT_NS
    with data['lock']:
        data['position_ned'] = {'x': 0, 'y': 0, 'z': -2, 'vx': 0, 'vy': 0, 'vz': 0, 'ts': stale}
        data['attitude'] = {'roll': 0, 'pitch': 0, 'yaw': 0.5, 'ts': stale}
    _set_intent(data, fwd=1, up=1)
    t = p.compute_target()
    assert t['source'] == 'watchdog_hover', f"expected watchdog_hover, got {t['source']}"
    assert t['vel_ned'] == (0.0, 0.0, 0.0), f"watchdog should hover, got {t['vel_ned']}"
    print("PASSED test_watchdog_hover")


def test_capture_waypoints():
    """B-capture appends the live pose and writes a JSON that load_mission round-trips."""
    data = _data()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'caps.json')
        kt = teleop.KeyboardTeleop(data, capture_path=path)   # construct only; no thread
        _set_pose(data, n=1.0, e=2.0, d=-3.0, yaw=math.radians(45))
        kt._capture_waypoint()
        _set_pose(data, n=-4.0, e=5.0, d=-6.0, yaw=math.radians(-90))
        kt._capture_waypoint()

        m = load_mission(path)
        assert m is not None and len(m.waypoints) == 2, "expected 2 captured waypoints"
        w0, w1 = m.waypoints
        assert w0.pos == (1.0, 2.0, -3.0) and _approx(w0.yaw_deg, 45.0)
        assert w1.pos == (-4.0, 5.0, -6.0) and _approx(w1.yaw_deg, -90.0)
        assert w0.name == 'wp0' and w1.name == 'wp1', "captured waypoints should be named wpN"
    print("PASSED test_capture_waypoints")


if __name__ == '__main__':
    test_horizontal_mapping()
    test_diagonal_is_capped()
    test_vertical_axis()
    test_yaw_integration()
    test_watchdog_hover()
    test_capture_waypoints()
    print("\nALL TELEOP TESTS PASSED")
