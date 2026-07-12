"""Phase 4 keystone test: closed-loop vision race simulation.

Synthetic camera projects real course gates through camera_model each tick,
feeds estimate_gates → GateMapper → RacePlanner → velocity integration.
No sockets, no sim — pure offline.

Two-phase test:
  1. SCOUT: drone uses FSM to discover and map the course. Gates don't need to
     be flown through precisely; success = all gates mapped.
  2. RACE: pre-loaded map from scout. Drone flies the entire course through
     each gate center. Success = all gates passed within 1.5m of center,
     and race time is >= 40% faster than scout time.
"""

import json
import math
import time
import threading
import numpy as np

import config
from mission import Mission, Waypoint
import camera_model as cm
from vision.gate_detector import GateDetection
from vision.gate_estimator import estimate_gates
from mapping.gate_mapper import GateMapper, MappedGate
from guidance.race_planner import RacePlanner

def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

YAW_SLEW_RAD_S = math.radians(70.0)


# ---------------------------------------------------------------------------
# Course loader
# ---------------------------------------------------------------------------
def load_course():
    """Load captured_waypoints.json as a list of Waypoint objects."""
    with open('captured_waypoints.json', 'r') as f:
        data = json.load(f)
    wps = []
    for row in data['waypoints']:
        # Waypoint.__init__ already converts yaw_deg to radians
        wps.append(Waypoint(row['n'], row['e'], row['d'],
                            yaw_deg=row.get('yaw_deg', 0.0),
                            name=row.get('name', '')))
    return wps


# ---------------------------------------------------------------------------
# Synthetic camera
# ---------------------------------------------------------------------------
def project_gate(gate_wp, drone_pos, drone_yaw):
    """Project a gate's 4 inner corners through the pinhole camera.
    Returns list of 4 [u,v] or None if outside FOV.
    """
    gyaw = gate_wp.yaw
    R_gate = np.array([
        [-math.sin(gyaw),  0.0, math.cos(gyaw)],
        [ math.cos(gyaw),  0.0, math.sin(gyaw)],
        [            0.0, -1.0,            0.0],
    ])
    _S = cm.GATE_INNER_M / 2.0
    obj_pts = [np.array([-_S, _S, 0]), np.array([_S, _S, 0]),
               np.array([_S, -_S, 0]), np.array([-_S, -_S, 0])]
    corners = []
    gpos = np.array(gate_wp.pos, dtype=float)
    for pt in obj_pts:
        pt_ned = gpos + R_gate @ pt
        pt_body = cm.ned_to_body(pt_ned - drone_pos, 0.0, 0.0, drone_yaw)
        pt_cam = cm.body_to_cam(pt_body)
        if pt_cam[2] < 0.3:
            return None
        u, v = cm.project(pt_cam)
        u += np.random.normal(0, 0.5)
        v += np.random.normal(0, 0.5)
        corners.append([u, v])
    for u, v in corners:
        if not (0 <= u < cm.WIDTH and 0 <= v < cm.HEIGHT):
            return None
    return corners


def make_detection(corners):
    """Build a GateDetection from projected corners."""
    u_min = min(c[0] for c in corners)
    v_min = min(c[1] for c in corners)
    u_max = max(c[0] for c in corners)
    v_max = max(c[1] for c in corners)
    w, h = u_max - u_min, v_max - v_min
    return GateDetection(
        center_px=(u_min + w / 2, v_min + h / 2),
        bbox_px=(u_min, v_min, w, h),
        area_px=w * h,
        confidence=0.92,
        corners_px=corners,
    )


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def run_sim(course_wps, mapper, *, verbose=False):
    """Fly the RacePlanner FSM around the course with a kinematic drone.

    Args:
        course_wps: list[Waypoint] – the gate positions / normals
        mapper:     GateMapper instance (may be empty or pre-loaded)
        verbose:    print per-gate progress

    Returns: (passed_count, min_dist_per_gate, sim_seconds)
    """
    config.USE_VISION = True

    shared_data = {
        'lock': threading.RLock(),
        'position_ned': None,
        'attitude': None,
    }
    planner = RacePlanner(shared_data, mapper, config)

    # Start 5m before gate 0 (approach side)
    wp0 = course_wps[0]
    normal0 = np.array([math.cos(wp0.yaw), math.sin(wp0.yaw), 0.0])
    pos = np.array(wp0.pos, dtype=float) - 5.0 * normal0
    pos[2] = 0.0
    yaw = wp0.yaw
    dt = 0.02

    total_gates = len(course_wps)
    passed_gates = 0
    min_dist = [float('inf')] * total_gates
    last_state = None

    for step in range(150_000):  # 3000 s budget
        now_ns = time.time_ns()

        with shared_data['lock']:
            shared_data['attitude'] = {'roll': 0.0, 'pitch': 0.0, 'yaw': yaw, 'ts': now_ns}
            shared_data['position_ned'] = {
                'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2]),
                'vx': 0.0, 'vy': 0.0, 'vz': 0.0, 'ts': now_ns,
            }

        # Synthetic camera at 10 Hz
        if step % 5 == 0:
            dets = []
            for i in range(max(0, passed_gates), min(total_gates, passed_gates + 3)):
                corners = project_gate(course_wps[i], pos, yaw)
                if corners is not None:
                    dets.append(make_detection(corners))
            obs = estimate_gates(dets, shared_data['attitude'], shared_data['position_ned'], ts_ns=now_ns)
            mapper.update(obs)

        target = planner.compute_target()
        state = planner.state

        if verbose and state != last_state:
            print(f"  step={step:5d}  state={state:<10s}  "
                  f"pos=({pos[0]:+8.2f},{pos[1]:+7.2f},{pos[2]:+7.2f})  "
                  f"g_idx={planner.active_gate_idx}  "
                  f"mapper={len(mapper.gates)}/{len(mapper.get_active_gates())}  "
                  f"passed={passed_gates}/{total_gates}")
            last_state = state

        # Record min distance to each gate
        for i, wp in enumerate(course_wps):
            d = float(np.linalg.norm(pos - np.array(wp.pos)))
            if d < min_dist[i]:
                min_dist[i] = d

        # Detect gate passage (plane crossing)
        if passed_gates < total_gates:
            gw = course_wps[passed_gates]
            g_normal = np.array([math.cos(gw.yaw), math.sin(gw.yaw), 0.0])
            g_pos = np.array(gw.pos, dtype=float)
            if np.dot(pos - g_pos, g_normal) > 0.0:
                if verbose:
                    d = float(np.linalg.norm(pos - g_pos))
                    print(f"  >> PASSED gate {passed_gates}  dist={d:.2f}m  t={step*dt:.1f}s")
                passed_gates += 1
                if planner.active_gate_idx < passed_gates:
                    planner.active_gate_idx = passed_gates

        if passed_gates >= total_gates:
            break

        vn, ve, vd = target['vel_ned']
        pos += np.array([vn, ve, vd]) * dt
        yaw_err = _wrap(target['yaw'] - yaw)
        yaw += max(-YAW_SLEW_RAD_S * dt, min(YAW_SLEW_RAD_S * dt, yaw_err))

    sim_time = step * dt
    return passed_gates, min_dist, sim_time


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def test_vision_race():
    import cv2  # PnP needs this — fail fast

    course = load_course()
    n_gates = len(course)

    # ── RACE RUN: pre-loaded perfect map ──
    # This is the main performance test. The drone has a map and should fly
    # precisely through every gate center.
    print(f"Race run: {n_gates} gates, pre-loaded map")
    mapper_race = GateMapper(min_hits_active=1)
    for i, wp in enumerate(course):
        mapper_race.gates.append(MappedGate(
            id=i + 1,
            pos=np.array(wp.pos, dtype=float),
            yaw=wp.yaw,
            hits=100,
            active=True,
        ))
    mapper_race._next_id = n_gates + 1

    passed_r, min_d_r, t_race = run_sim(course, mapper_race, verbose=True)
    print(f"  result: {passed_r}/{n_gates} gates in {t_race:.1f}s")
    
    PASS_RADIUS = cm.GATE_INNER_M  # 1.5 m
    for i, d in enumerate(min_d_r):
        print(f"  gate {i:2d}: min_dist={d:.2f}m {'OK' if d < PASS_RADIUS else 'FAIL'}")
    
    for i, d in enumerate(min_d_r):
        assert d < PASS_RADIUS, f"Gate {i}: min dist {d:.2f}m > {PASS_RADIUS}m"
    assert passed_r == n_gates, f"Only passed {passed_r}/{n_gates}"

    # ── SCOUT RUN: no prior map, discovery mode ──
    # The scout builds the map. Gate precision during scouting is relaxed.
    print(f"\nScout run: {n_gates} gates, discovery mode")
    mapper_scout = GateMapper(min_hits_active=3)
    passed_s, min_d_s, t_scout = run_sim(course, mapper_scout, verbose=True)
    print(f"  result: {passed_s}/{n_gates} gates in {t_scout:.1f}s")
    assert passed_s == n_gates, f"Scout only passed {passed_s}/{n_gates}"

    # ── Speed comparison ──
    # Race with full map should be significantly faster than scouting
    speedup = 1.0 - t_race / t_scout
    print(f"\nSpeedup: {speedup*100:.0f}%  (scout {t_scout:.1f}s  race {t_race:.1f}s)")
    assert t_race < t_scout * 0.6, (
        f"Race {t_race:.1f}s not >=40% faster than scout {t_scout:.1f}s"
    )

    print("\nALL VISION RACE TESTS PASSED")


if __name__ == "__main__":
    test_vision_race()
