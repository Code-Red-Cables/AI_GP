import threading
import time
import numpy as np

from mapping.gate_mapper import GateMapper, MappedGate
from guidance.race_planner import RacePlanner

class MockCfg:
    MAX_ALT_M = 15.0
    MAX_VSPEED = 3.0
    TAKEOFF_ALT_M = 3.0
    SCAN_SPEED = 1.0
    SCAN_YAW_MAX_DEG = 45.0
    SCAN_ALT_MIN_M = 1.0
    APPROACH_STANDOFF_M = 3.0
    EXIT_OVERSHOOT_M = 2.5
    COMMIT_DIST_M = 4.0
    MAX_PATH_DIST_M = 60.0
    CRUISE_SPEED = 6.0
    LOOKAHEAD_M = 2.0
    LOOKAHEAD_TIME = 0.5
    LOOKAHEAD_MAX = 10.0
    A_LAT_MAX = 15.0
    A_LON_MAX = 10.0
    KP_VERT_PATH = 2.0
    FINISH_SPEED = 0.0

def test_fsm():
    shared_data = {
        'lock': threading.RLock(),
        'position_ned': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'ts': time.time_ns()},
        'attitude': {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'ts': time.time_ns()}
    }
    mapper = GateMapper()
    planner = RacePlanner(shared_data, mapper, MockCfg())

    # 1. Start in TAKEOFF
    assert planner.state == 'TAKEOFF'
    target = planner.compute_target()
    assert target['state'] == 'TAKEOFF'
    assert target['vel_ned'][2] == -2.0

    # 2. Ascend to TAKEOFF_ALT_M
    shared_data['position_ned']['z'] = -3.0
    target = planner.compute_target()
    
    # 3. No gates mapped -> SCAN
    # Wait, the first compute target might transition but still return SCAN
    assert planner.state == 'SCAN'
    
    # 4. Add a gate -> APPROACH
    gate1 = MappedGate(id=1, pos=np.array([10.0, 0.0, -3.0]), yaw=0.0, hits=5, active=True)
    mapper.gates.append(gate1)
    
    target = planner.compute_target()
    assert planner.state == 'APPROACH'
    assert target['state'] == 'APPROACH'
    
    # 5. Move close to gate -> PASS
    shared_data['position_ned']['x'] = 7.0 # Dist = 3.0, COMMIT_DIST is 4.0
    target = planner.compute_target()
    assert planner.state == 'PASS'
    assert target['state'] == 'PASS'
    
    # 6. Cross plane -> ADVANCE
    shared_data['position_ned']['x'] = 11.0 # Past the gate
    target = planner.compute_target()
    # It passes through ADVANCE and becomes SCAN immediately because there is no next gate mapped
    assert planner.state == 'SCAN'
    assert target['source'] == 'fsm:ADVANCE'
    
    # 7. Next target -> SCAN (since next gate not mapped)
    target = planner.compute_target()
    assert planner.state == 'SCAN'
    
    # 8. Guard Test (Alt Guard)
    shared_data['position_ned']['z'] = -20.0
    target = planner.compute_target()
    assert target['state'] == 'HOLD'
    assert target['source'] == 'alt_guard'
    
    # 9. Guard Test (Watchdog)
    shared_data['position_ned']['ts'] = time.time_ns() - 1_000_000_000 # Stale
    target = planner.compute_target()
    assert target['state'] == 'HOLD'
    assert target['source'] == 'watchdog_hover'
    
    print("ALL RACE FSM TESTS PASSED")

if __name__ == "__main__":
    test_fsm()
