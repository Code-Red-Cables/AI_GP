"""Offline smoke test: detector -> estimator -> planner -> controller send path.

Runs the full perception+planning pipeline on a synthetic gate frame with no
simulator, exercising every module's interface together. Run with the bundled
interpreter (numpy + cv2 + pymavlink):

    & "<repo>/PyAIPilotExample/myenv/Scripts/python.exe" test_pipeline_smoke.py
"""
import math
import threading
import time

import numpy as np

import camera_model as cm
from vision.gate_detector import detect_gate, _synthesize_test_image
from gate_estimator import estimate_gate
from planner import Planner
import controller as ctrl


def test_detect_and_estimate():
    img, cfg = _synthesize_test_image()
    det = detect_gate(img, cfg)
    assert det is not None, "detector found no gate in synthetic frame"
    # level attitude -> gate should be roughly straight ahead (body +x), centered.
    est = estimate_gate(det, attitude={"roll": 0, "pitch": 0, "yaw": 0},
                        position_ned=(0.0, 0.0, -2.0), ts=time.time_ns())
    assert est["detected"]
    gx, gy, gz = est["gate_body"]
    # Gate at image center, but the camera is tilted UP 20deg, so in the BODY frame
    # the gate is forward (x>0), horizontally centered (y~0), and UP (z<0) by ~20deg.
    assert gx > 0, f"gate should be ahead (x>0), got {est['gate_body']}"
    assert abs(gy) < 0.5, f"gate should be horizontally centered, got {est['gate_body']}"
    assert gz < 0, f"gate at image center should be UP (z<0) under the 20deg tilt, got {gz}"
    az, el = est["bearing"]
    assert abs(az) < math.radians(3), f"azimuth should be ~0, got {math.degrees(az):.1f}deg"
    assert abs(el - math.radians(cm.CAMERA_TILT_UP_DEG)) < math.radians(4), \
        f"elevation should be ~+20deg (camera tilt), got {math.degrees(el):.1f}deg"
    assert est["range_m"] > 0
    print(f"PASS detect+estimate  method={est['method']} range={est['range_m']:.2f}m "
          f"gate_body=({gx:.2f},{gy:.2f},{gz:.2f}) el={math.degrees(el):.1f}deg")
    return est


def test_planner_vision(est):
    data = {"lock": threading.RLock()}
    data["vision"] = est
    data["attitude"] = {"roll": 0, "pitch": 0, "yaw": 0, "ts": time.time_ns()}
    data["odometry"] = {"pos": (0.0, 0.0, -2.0), "q": (1, 0, 0, 0),
                        "vel": (0, 0, 0), "rates": (0, 0, 0), "ts": time.time_ns()}
    planner = Planner(data)
    target = planner.compute_target()
    assert target["mode"] == "velocity"
    assert target["source"] in ("vision", "vision_level"), target["source"]
    vn, ve, vd = target["vel_ned"]
    assert vn > 0, f"should fly forward (North+), got {target['vel_ned']}"
    speed = math.sqrt(vn * vn + ve * ve + vd * vd)
    assert speed <= __import__("planner").MAX_SPEED + 1e-6
    print(f"PASS planner(vision)  source={target['source']} "
          f"vel_ned=({vn:+.2f},{ve:+.2f},{vd:+.2f}) yaw={target['yaw']:+.2f}")


def test_planner_known_geometry():
    data = {"lock": threading.RLock()}
    data["attitude"] = {"roll": 0, "pitch": 0, "yaw": 0, "ts": time.time_ns()}
    data["position_ned"] = {"x": 0.0, "y": 0.0, "z": -2.0, "vx": 0, "vy": 0, "vz": 0, "ts": time.time_ns()}
    data["gates"] = [{"gate_id": 0, "pos_ned": (10.0, 0.0, -2.0),
                      "orient_ned": (1, 0, 0, 0), "width": 1.5, "height": 1.5}]
    data["race"] = {"active_gate_index": 0, "ts": time.time_ns()}
    planner = Planner(data)
    target = planner.compute_target()
    assert target["source"] == "known", target["source"]
    vn, ve, vd = target["vel_ned"]
    assert vn > 0 and abs(ve) < 1e-6, f"should head straight North to gate, got {target['vel_ned']}"
    print(f"PASS planner(known)   source={target['source']} vel_ned=({vn:+.2f},{ve:+.2f},{vd:+.2f})")


def test_planner_watchdog():
    data = {"lock": threading.RLock()}  # no telemetry at all
    target = Planner(data).compute_target()
    assert target["vel_ned"] == (0.0, 0.0, 0.0) and "hover" in target["source"]
    print(f"PASS planner(watchdog) source={target['source']} -> hover")


def test_controller_send_path():
    """Verify the velocity send builds a valid MAVLink call (capture args, no socket)."""
    captured = {}

    class FakeMav:
        def set_position_target_local_ned_send(self, *args):
            captured["args"] = args

    class FakeConn:
        target_system = 1
        target_component = 1
        mav = FakeMav()

    ctrl.send_velocity_ned(FakeConn(), 0, 1.0, 0.0, -0.2, 0.5)
    assert "args" in captured and len(captured["args"]) == 16, captured
    print("PASS controller send  set_position_target_local_ned_send built OK")


if __name__ == "__main__":
    est = test_detect_and_estimate()
    test_planner_vision(est)
    test_planner_known_geometry()
    test_planner_watchdog()
    test_controller_send_path()
    print("ALL PIPELINE SMOKE TESTS PASSED")
