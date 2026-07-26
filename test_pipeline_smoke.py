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
    """Verify the attitude send builds a valid MAVLink call (capture args, no socket)."""
    captured = {}

    class FakeMav:
        def set_attitude_target_send(self, *args):
            captured["args"] = args

    class FakeConn:
        target_system = 1
        target_component = 1
        mav = FakeMav()

    ctrl.send_rate_target(FakeConn(), 0, 0.1, -0.2, 0.3, 0.5)
    # time_boot, sys, comp, mask, quat(list), rollrate, pitchrate, yawrate, thrust
    assert "args" in captured and len(captured["args"]) == 9, captured
    assert len(captured["args"][4]) == 4, "quaternion must have 4 elements"
    assert captured["args"][5] == 0.1, "roll rate must reach the rollrate field"
    assert captured["args"][6] == -0.2, "pitch rate must reach the pitchrate field"
    assert captured["args"][7] == 0.3, "yaw rate must reach the yawrate field"
    assert captured["args"][8] == 0.5, "thrust must reach the thrust field"
    print("PASS controller send  set_attitude_target (rate) built OK")


def test_velocity_to_attitude_signs():
    """ANGLE-mode mapping sign checks: forward -> nose-down, right -> roll-right, etc."""
    # Heading North (yaw=0). Desired velocity due North (forward) -> nose-down pitch.
    roll, pitch, yaw, thrust = ctrl.velocity_to_attitude(
        (2.0, 0.0, 0.0), 0.0, 0.0, (0.0, 0.0, 0.0))
    # This simulator's measured pitch/rate convention is inverted (controller.py
    # RATE_SIGN_PITCH), so its empirically verified forward command is positive.
    assert pitch > 0, f"forward flight should use positive sim pitch, got {pitch}"
    assert abs(roll) < 1e-6, f"pure-forward should not roll, got {roll}"
    # Desired velocity due East while heading North -> roll right (positive).
    roll, pitch, yaw, thrust = ctrl.velocity_to_attitude(
        (0.0, 2.0, 0.0), 0.0, 0.0, (0.0, 0.0, 0.0))
    assert roll > 0, f"rightward flight should roll right (pos), got {roll}"
    # Commanded climb (vd<0) while sinking (vz>0) -> more than hover thrust.
    *_, thrust_climb = ctrl.velocity_to_attitude(
        (0.0, 0.0, -1.0), 0.0, 0.0, (0.0, 0.0, 0.5))
    *_, thrust_descend = ctrl.velocity_to_attitude(
        (0.0, 0.0, 1.0), 0.0, 0.0, (0.0, 0.0, 0.0))
    assert thrust_climb > ctrl.HOVER_THRUST, f"climb cmd should exceed hover, got {thrust_climb}"
    assert thrust_descend < ctrl.HOVER_THRUST, f"descend cmd should drop thrust, got {thrust_descend}"
    # Lean is capped.
    roll, pitch, *_ = ctrl.velocity_to_attitude(
        (100.0, 0.0, 0.0), 0.0, 0.0, (0.0, 0.0, 0.0))
    assert abs(pitch) <= ctrl.MAX_LEAN_RAD + 1e-9, "pitch must be capped at MAX_LEAN_RAD"
    print("PASS velocity->attitude  signs & caps OK")


if __name__ == "__main__":
    est = test_detect_and_estimate()
    test_planner_vision(est)
    test_planner_known_geometry()
    test_planner_watchdog()
    test_controller_send_path()
    test_velocity_to_attitude_signs()
    print("ALL PIPELINE SMOKE TESTS PASSED")
