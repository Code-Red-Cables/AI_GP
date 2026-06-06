"""Acceptance tests for camera_model.py. Run as a script:

    & "C:\\Users\\rocky\\docs\\AI_GP\\PyAIPilotExample\\myenv\\Scripts\\python.exe" test_camera_model.py

Prints "ALL TESTS PASSED" and exits 0 on success; exits non-zero on failure.
"""

import sys

import numpy as np

import camera_model as cm

EXACT = 1e-6
PX = 0.5


def test_project_deproject_roundtrip():
    Z = 5.0
    for (u, v) in [(320.0, 180.0), (100.0, 50.0), (500.0, 300.0), (0.0, 0.0), (639.0, 359.0)]:
        p = cm.deproject(u, v, Z)
        u2, v2 = cm.project(p)
        assert abs(u2 - u) < EXACT and abs(v2 - v) < EXACT, (u, v, u2, v2)


def test_straight_ahead_level():
    # body forward & level -> camera looks up, horizon below optical axis (v > CY).
    p_cam = cm.body_to_cam(np.array([10.0, 0.0, 0.0]))
    u, v = cm.project(p_cam)
    assert abs(u - cm.CX) < PX, u
    assert v > cm.CY, v
    assert abs(v - 296.0) < 1.0, v  # expected ~296


def test_straight_ahead_20up():
    # direction 20deg above horizontal -> projects to optical center.
    t = np.radians(cm.CAMERA_TILT_UP_DEG)
    direction = np.array([np.cos(t), 0.0, -np.sin(t)]) * 10.0
    p_cam = cm.body_to_cam(direction)
    u, v = cm.project(p_cam)
    assert abs(u - cm.CX) < PX, u
    assert abs(v - cm.CY) < PX, v


def test_ahead_and_right():
    p_cam = cm.body_to_cam(np.array([10.0, 2.0, 0.0]))
    u, v = cm.project(p_cam)
    assert u > cm.CX, u


def test_ahead_and_below():
    level = cm.project(cm.body_to_cam(np.array([10.0, 0.0, 0.0])))[1]
    below = cm.project(cm.body_to_cam(np.array([10.0, 0.0, 2.0])))[1]
    assert below > level, (level, below)


def test_range_from_size():
    assert abs(cm.range_from_size(96.0) - 5.0) < EXACT, cm.range_from_size(96.0)


def test_body_to_ned():
    p = np.array([1.0, 2.0, 3.0])
    assert np.allclose(cm.body_to_ned(p, 0.0, 0.0, 0.0), p, atol=EXACT)
    yaw90 = cm.body_to_ned(np.array([1.0, 0.0, 0.0]), 0.0, 0.0, np.pi / 2)
    assert np.allclose(yaw90, [0.0, 1.0, 0.0], atol=EXACT), yaw90
    r, p2, y = 0.3, -0.2, 1.1
    back = cm.ned_to_body(cm.body_to_ned(p, r, p2, y), r, p2, y)
    assert np.allclose(back, p, atol=EXACT), back


def test_cam_body_roundtrip():
    p = np.array([1.0, 2.0, 3.0])
    assert np.allclose(cm.cam_to_body(cm.body_to_cam(p)), p, atol=EXACT)


def main():
    tests = [
        test_project_deproject_roundtrip,
        test_straight_ahead_level,
        test_straight_ahead_20up,
        test_ahead_and_right,
        test_ahead_and_below,
        test_range_from_size,
        test_body_to_ned,
        test_cam_body_roundtrip,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print("PASS", t.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL", t.__name__, "->", e)
    if failed:
        print(f"{failed} TEST(S) FAILED")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
