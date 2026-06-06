"""Offline tests for learn-then-replay course preplanning (preplanning branch).

Covers the CourseMap persistence/averaging and the planner's preplan behaviour:
  * preplan flies toward a mapped gate when vision has no fresh lock (fixes gate-2 hover),
  * a default planner (no map / preplan off) is UNCHANGED — never emits the 'preplan'
    source,
  * the planner LEARNS a gate's position when it is passed (vision-confirmed),
  * the self gate-pass detector advances the internal index when the sim sends no
    active_gate_index.

Run with the project venv (numpy):  ./venv/bin/python test_preplanning.py
"""
import math
import os
import tempfile
import threading
import time

import numpy as np

from course_map import CourseMap
from planner import Planner


def _fresh(data):
    """Stamp telemetry timestamps to 'now' so the watchdog stays disarmed."""
    now = time.time_ns()
    if 'attitude' in data:
        data['attitude']['ts'] = now
    if 'position_ned' in data:
        data['position_ned']['ts'] = now
    if 'vision' in data and data['vision'] is not None:
        data['vision']['ts'] = now
    return now


def _base_data(preplan=False, learn=False, course_map=None, race_idx=None):
    data = {
        'lock': threading.RLock(),
        'preplan': preplan,
        'learn': learn,
    }
    if course_map is not None:
        data['course_map'] = course_map
    data['attitude'] = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'ts': time.time_ns()}
    if race_idx is not None:
        data['race'] = {'active_gate_index': race_idx, 'ts': time.time_ns()}
    return data


def _set_pos(data, n, e, d):
    data['position_ned'] = {'x': n, 'y': e, 'z': d, 'vx': 0, 'vy': 0, 'vz': 0,
                            'ts': time.time_ns()}


# --------------------------------------------------------------------------- CourseMap
def test_coursemap_record_and_average():
    cm = CourseMap(path="/tmp/_unused.json")
    cm.record(0, (10.0, 0.0, -2.0))
    cm.record(0, (12.0, 2.0, -4.0))          # average -> (11, 1, -3)
    got = cm.get(0)
    assert got is not None and len(got) == 3
    assert abs(got[0] - 11.0) < 1e-6 and abs(got[1] - 1.0) < 1e-6 and abs(got[2] + 3.0) < 1e-6, got
    assert cm.get(5) is None
    assert cm.indices() == [0]
    assert cm.has_any()
    print(f"PASS coursemap average        gate0={got}")


def test_coursemap_save_load_atomic():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "course_map.json")
        cm = CourseMap(path)
        cm.record(0, (1.0, 2.0, -3.0))
        cm.record(2, (4.0, 5.0, -6.0))
        cm.save()
        assert os.path.exists(path), "map file not written"
        assert not os.path.exists(path + ".tmp"), "temp file left behind (not atomic)"
        # Reload into a fresh instance.
        cm2 = CourseMap(path).load()
        assert cm2.indices() == [0, 2]
        assert abs(cm2.get(2)[0] - 4.0) < 1e-6
        # Corrupt file -> empty map, no crash.
        with open(path, "w") as f:
            f.write("{ this is not json")
        cm3 = CourseMap(path).load()
        assert not cm3.has_any()
        print(f"PASS coursemap save/load      indices={cm2.indices()} (atomic, corrupt-safe)")


# --------------------------------------------------------------------------- planner replay
def test_planner_preplan_flies_to_mapped_gate():
    """With a mapped gate and NO vision, preplan steers toward the known position."""
    cm = CourseMap(path="/tmp/_unused.json")
    cm.record(0, (10.0, 0.0, -2.0))
    data = _base_data(preplan=True, course_map=cm, race_idx=0)
    _set_pos(data, 0.0, 0.0, -2.0)
    _fresh(data)
    target = Planner(data).compute_target()
    assert target['source'] == 'preplan', target['source']
    vn, ve, vd = target['vel_ned']
    assert vn > 0 and abs(ve) < 1e-6, f"should head straight North to mapped gate, got {target['vel_ned']}"
    print(f"PASS planner preplan replay   source={target['source']} vel_ned=({vn:+.2f},{ve:+.2f},{vd:+.2f})")


def test_planner_without_preplan_unchanged():
    """No course_map and preplan off: never emits 'preplan'; hovers with no target."""
    data = _base_data(preplan=False, course_map=None, race_idx=0)
    _set_pos(data, 0.0, 0.0, -2.0)
    _fresh(data)
    target = Planner(data).compute_target()
    assert target['source'] != 'preplan', target['source']
    assert target['vel_ned'] == (0.0, 0.0, 0.0) and 'hover' in target['source'], target
    print(f"PASS planner default unchanged source={target['source']} -> hover (no preplan)")


def test_planner_learns_on_gate_pass():
    """A vision-confirmed gate is recorded into the map when active_gate_index advances."""
    cm = CourseMap(path="/tmp/_unused.json")
    data = _base_data(preplan=True, learn=True, course_map=cm, race_idx=0)
    _set_pos(data, 0.0, 0.0, -2.0)
    # A clean, confident, in-range detection dead ahead (body +x). Level attitude, so the
    # body->NED transform is identity-ish and the gate world pos ~= position + (8,0,0).
    data['vision'] = {'detected': True, 'confidence': 0.9, 'gate_body': (8.0, 0.0, 0.0),
                      'ts': time.time_ns()}
    _fresh(data)
    planner = Planner(data)
    t0 = planner.compute_target()
    assert t0['source'] in ('vision', 'gate_memory'), t0['source']
    assert not cm.has_any(), "nothing should be recorded before the gate is passed"
    # Now the sim advances to gate 1 -> the just-passed gate 0 is learned.
    data['race']['active_gate_index'] = 1
    _fresh(data)
    planner.compute_target()
    rec = cm.get(0)
    assert rec is not None, "gate 0 was not recorded on pass"
    assert abs(rec[0] - 8.0) < 1.0 and abs(rec[1]) < 1.0, f"recorded gate0 looks wrong: {rec}"
    print(f"PASS planner learns on pass   recorded gate0={tuple(round(v,2) for v in rec)}")


def test_self_pass_detector_advances_without_race():
    """With NO active_gate_index, the planner detects the pass and bumps its own index."""
    cm = CourseMap(path="/tmp/_unused.json")
    cm.record(0, (10.0, 0.0, -2.0))
    data = _base_data(preplan=True, course_map=cm, race_idx=None)   # no 'race'
    planner = Planner(data)

    # Far from gate 0 -> not armed yet.
    _set_pos(data, 0.0, 0.0, -2.0); _fresh(data)
    planner.compute_target()
    assert planner._auto_idx == 0 and not planner._armed_pass

    # Inside the pass radius -> arm.
    _set_pos(data, 9.2, 0.0, -2.0); _fresh(data)
    planner.compute_target()
    assert planner._armed_pass, "should arm when within PASS_THROUGH_DIST"
    assert planner._auto_idx == 0

    # Clearly past the gate -> advance the internal index.
    _set_pos(data, 15.0, 0.0, -2.0); _fresh(data)
    planner.compute_target()
    assert planner._auto_idx == 1, f"self-pass should advance index, got {planner._auto_idx}"
    print(f"PASS self-pass detector       auto_idx -> {planner._auto_idx} (no race index)")


if __name__ == "__main__":
    test_coursemap_record_and_average()
    test_coursemap_save_load_atomic()
    test_planner_preplan_flies_to_mapped_gate()
    test_planner_without_preplan_unchanged()
    test_planner_learns_on_gate_pass()
    test_self_pass_detector_advances_without_race()
    print("ALL PREPLANNING TESTS PASSED")
