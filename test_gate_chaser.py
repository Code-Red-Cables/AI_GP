"""Offline test for GateChaser (no sim, no cv2). Feeds synthetic vision/attitude through
shared_data and asserts the commanded velocity points sensibly at the gate.

Run:  python test_gate_chaser.py
"""

import math
import threading
import time

from gate_chaser import GateChaser
import config as cfg


def _data(gate_body, yaw=0.0, detected=True, age_ns=0):
    now = time.time_ns()
    return {
        'lock': threading.RLock(),
        'attitude': {'roll': 0.0, 'pitch': 0.0, 'yaw': yaw, 'ts': now},
        'position_ned': {'x': None, 'y': None, 'z': -5.0, 'vx': 0.0, 'vy': 0.0, 'vz': 0.0, 'ts': now},
        'vision': ({'detected': detected, 'gate_body': gate_body, 'range_m': math.sqrt(sum(c*c for c in gate_body)),
                    'ts': now - age_ns} if gate_body is not None else None),
    }


def test_climbs_to_high_gate():
    """Gate ahead and ABOVE -> command forward + CLIMB (vd negative), no big strafe."""
    d = _data([20.0, 0.3, -8.0])           # forward 20, slightly right, 8 m up
    gc = GateChaser(d)
    t = gc.compute_target()
    vn, ve, vd = t['vel_ned']
    assert vn > 0, f"should move forward toward gate (vn={vn})"
    assert vd < -0.5, f"gate is above -> must climb (vd={vd})"
    assert t['source'] == 'gate_track'
    print(f"PASS high gate climb   vn={vn:.2f} ve={ve:.2f} vd={vd:.2f} (forward + climb)")


def test_strafes_to_center():
    """Gate off to the RIGHT (level, near) -> strafe right (yaw=0 -> ve>0)."""
    d = _data([8.0, 4.0, 0.0])             # forward 8, 4 m right, same height
    gc = GateChaser(d)
    vn, ve, vd = gc.compute_target()['vel_ned']
    assert ve > 0, f"gate to the right -> strafe right (ve={ve})"
    assert abs(vd) < 0.3, f"same height -> little vertical (vd={vd})"
    print(f"PASS strafe to center  vn={vn:.2f} ve={ve:.2f} vd={vd:.2f} (strafe right)")


def test_yaw_rotates_command():
    """With yaw=90deg, a body-forward command must rotate into NED (East), proving the
    body->NED mapping uses the live heading."""
    d0 = _data([10.0, 0.0, 0.0], yaw=0.0)
    vn0, ve0, _ = GateChaser(d0).compute_target()['vel_ned']
    d9 = _data([10.0, 0.0, 0.0], yaw=math.radians(90))
    vn9, ve9, _ = GateChaser(d9).compute_target()['vel_ned']
    assert vn0 > 0.05 and abs(ve0) < 0.05, f"yaw0: forward=+N ({vn0:.2f},{ve0:.2f})"
    # physical nose at yaw=90 (left-handed sim): forward = (cos90, -sin90) = (0,-1) -> -E
    assert abs(vn9) < 0.05 and ve9 < -0.05, f"yaw90: forward rotated off North ({vn9:.2f},{ve9:.2f})"
    print(f"PASS yaw rotates cmd   yaw0->({vn0:.2f},{ve0:.2f})  yaw90->({vn9:.2f},{ve9:.2f})")


def test_commit_then_coast_on_loss():
    """Close gate -> commit (drive straight, no strafe). Then lose it -> coast forward briefly."""
    d = _data([2.0, 1.5, 0.0])             # within GATE_CLOSE_RANGE
    gc = GateChaser(d)
    t = gc.compute_target()
    vn, ve, vd = t['vel_ned']
    assert t['source'] == 'gate_commit', t['source']
    assert ve == 0.0 or abs(ve) < 1e-9, f"commit must stop strafing (ve={ve})"
    assert vn > 0, "commit drives forward"
    # now the gate drops out of frame (passing through) -> should COAST forward, not hover
    with d['lock']:
        d['vision'] = None
    t2 = gc.compute_target()
    assert t2['source'] == 'gate_pass', t2['source']
    assert t2['vel_ned'][0] > 0, "coast should keep moving forward through the gate"
    print(f"PASS commit + coast    commit ve={ve:.2f}; after loss src={t2['source']} vn={t2['vel_ned'][0]:.2f}")


def test_lost_gate_hovers():
    """No detection (and not just-committed) -> hover (zero velocity), hold heading."""
    d = _data(None, detected=False)
    gc = GateChaser(d)
    t = gc.compute_target()
    assert t['vel_ned'] == (0.0, 0.0, 0.0), t['vel_ned']
    assert t['source'] == 'gate_lost'
    print("PASS lost -> hover     zero velocity when no gate in view")


def test_stale_vision_is_lost():
    """A detection older than the timeout counts as lost."""
    d = _data([10.0, 0.0, 0.0], age_ns=int((cfg.GATE_VISION_TIMEOUT_S + 0.5) * 1e9))
    gc = GateChaser(d)
    t = gc.compute_target()
    assert t['source'] == 'gate_lost', f"stale vision should be lost ({t['source']})"
    print("PASS stale -> lost     old detection ignored")


if __name__ == "__main__":
    test_climbs_to_high_gate()
    test_strafes_to_center()
    test_yaw_rotates_command()
    test_commit_then_coast_on_loss()
    test_lost_gate_hovers()
    test_stale_vision_is_lost()
    print("ALL GATE CHASER TESTS PASSED")
