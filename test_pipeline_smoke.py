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


def test_planner_goto_waypoint():
    """The preplanned planner steers toward its current waypoint's NED position."""
    from mission import Mission, Waypoint
    data = {"lock": threading.RLock()}
    # One waypoint 10 m North + 4 m up from the drone, facing North.
    data["mission"] = Mission([Waypoint(10.0, 0.0, -6.0, 0.0, name="wpN")])
    data["attitude"] = {"roll": 0, "pitch": 0, "yaw": 0, "ts": time.time_ns()}
    data["position_ned"] = {"x": 0.0, "y": 0.0, "z": -2.0, "vx": 0, "vy": 0, "vz": 0, "ts": time.time_ns()}
    target = Planner(data).compute_target()
    assert target["mode"] == "velocity"
    assert target["source"] == "wpN", target["source"]
    vn, ve, vd = target["vel_ned"]
    assert vn > 0 and abs(ve) < 1e-6, f"should head straight North, got {target['vel_ned']}"
    assert vd < 0, f"waypoint is higher (z<-2) so should climb (vd<0), got {vd}"
    speed = math.sqrt(vn * vn + ve * ve + vd * vd)
    assert speed <= __import__("planner").MAX_SPEED + 1e-6
    print(f"PASS planner(goto)    source={target['source']} "
          f"vel_ned=({vn:+.2f},{ve:+.2f},{vd:+.2f}) yaw={target['yaw']:+.2f}")


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
    """Body-frame mapping with the sim's LEFT-HANDED yaw (rotate error by -yaw).

    velocity_to_attitude(vel_ned, yaw_cmd, yaw_now, vel_now); vel_now is the measured NED
    velocity (vn, ve, vz). Forward error -> pitch (LEAN_SIGN_FWD), lateral -> roll
    (LEAN_SIGN_LAT); checks follow the sign constants so this survives a live sign flip.
    """
    sign = lambda x: math.copysign(1.0, x)
    # Heading North (yaw=0): pure-North desired -> pure forward -> pitch only.
    roll, pitch, yaw, thrust = ctrl.velocity_to_attitude((2.0, 0.0, 0.0), 0.0, 0.0, (0.0, 0.0, 0.0))
    assert abs(roll) < 1e-6, f"pure-forward should not roll, got {roll}"
    assert sign(pitch) == sign(ctrl.LEAN_SIGN_FWD), f"forward error must pitch with LEAN_SIGN_FWD, got {pitch}"
    # Left-handed yaw: facing WEST is yaw=-90; physical forward there is EAST, so a desired
    # velocity due EAST (+e) is body-FORWARD -> pitch only (no roll). This is the case the
    # old +yaw rotation got backwards.
    roll, pitch, yaw, thrust = ctrl.velocity_to_attitude((0.0, 2.0, 0.0), 0.0, math.radians(-90), (0.0, 0.0, 0.0))
    assert abs(roll) < 1e-6, f"East-while-yaw=-90 is pure forward, should not roll, got {roll}"
    assert sign(pitch) == sign(ctrl.LEAN_SIGN_FWD), f"forward (East@yaw=-90) must pitch with LEAN_SIGN_FWD, got {pitch}"
    # Commanded climb (vd<0) while sinking (vz>0) -> more than hover thrust.
    *_, thrust_climb = ctrl.velocity_to_attitude((0.0, 0.0, -1.0), 0.0, 0.0, (0.0, 0.0, 0.5))
    *_, thrust_descend = ctrl.velocity_to_attitude((0.0, 0.0, 1.0), 0.0, 0.0, (0.0, 0.0, 0.0))
    assert thrust_climb > ctrl.HOVER_THRUST, f"climb cmd should exceed hover, got {thrust_climb}"
    assert thrust_descend < ctrl.HOVER_THRUST, f"descend cmd should drop thrust, got {thrust_descend}"
    # Lean is capped.
    roll, pitch, *_ = ctrl.velocity_to_attitude((100.0, 0.0, 0.0), 0.0, 0.0, (0.0, 0.0, 0.0))
    assert abs(pitch) <= ctrl.MAX_LEAN_RAD + 1e-9, "pitch must be capped at MAX_LEAN_RAD"
    print("PASS velocity->attitude  body-frame (-yaw) signs & caps OK")


if __name__ == "__main__":
    test_detect_and_estimate()
    test_planner_goto_waypoint()
    test_planner_watchdog()
    test_controller_send_path()
    test_velocity_to_attitude_signs()
    print("ALL PIPELINE SMOKE TESTS PASSED")
