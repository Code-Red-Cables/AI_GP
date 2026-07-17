import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import random
import json
from vision.gate_estimator import GateObservation
from mapping.gate_mapper import GateMapper

class MockCfg:
    GATE_ASSOC_CHI2 = 9.0
    MIN_OBS_CONFIRM = 3
    CONFIRM_WINDOW_S = 2.0
    CANDIDATE_PRUNE_S = 3.0
    PNP_SIGMA0_M = 0.3
    PNP_SIGMA_K = 0.05
    SIZE_METHOD_VAR_MULT = 4.0
    FAR_VERT_RANGE_M = 12.0
    VIS_CONF_MIN = 0.10
    VIS_MAX_RANGE_M = 40.0
    POS_FIX_MAX_RANGE_M = 20.0

def run_test():
    cfg = MockCfg()
    mapper = GateMapper(cfg=cfg)

    # 3 static gates
    true_gates = [
        {'pos': np.array([10.0, 5.0, -2.0]), 'yaw': 0.5},
        {'pos': np.array([20.0, 10.0, -2.0]), 'yaw': 1.0},
        {'pos': np.array([30.0, 15.0, -2.0]), 'yaw': 1.5},
    ]
    
    drone_pos = np.array([0.0, 0.0, -2.0])
    
    # 5% garbage bursts
    for i in range(100):
        obs_list = []
        ts_ns = i * 100000000  # 10 Hz
        
        # True gates
        for g in true_gates:
            if random.random() < 0.8:  # 80% detection rate
                noisy_pos = g['pos'] + np.random.normal(0, 0.2, size=3)
                
                true_normal = np.array([np.cos(g['yaw']), np.sin(g['yaw']), 0.0])
                if random.random() < 0.5:
                    obs_normal = -true_normal
                else:
                    obs_normal = true_normal
                
                obs_list.append(GateObservation(
                    ts_ns=ts_ns,
                    frame_id=i,
                    gate_body=np.array([1.0, 0.0, 0.0]),
                    normal_body=np.zeros(3),
                    gate_ned=noisy_pos,
                    normal_ned=obs_normal,
                    range_m=float(np.linalg.norm(noisy_pos - drone_pos)),
                    method="pnp",
                    confidence=0.9,
                    center_px=(320.0, 240.0)
                ))
        
        # 5% garbage bursts
        if random.random() < 0.05:
            for _ in range(5):
                fp_pos = np.random.uniform(-50, 50, size=3)
                fp_yaw = random.uniform(-3.14, 3.14)
                fp_normal = np.array([np.cos(fp_yaw), np.sin(fp_yaw), 0.0])
                obs_list.append(GateObservation(
                    ts_ns=ts_ns,
                    frame_id=i,
                    gate_body=np.array([1.0, 0.0, 0.0]),
                    normal_body=np.zeros(3),
                    gate_ned=fp_pos,
                    normal_ned=fp_normal,
                    range_m=10.0,
                    method="pnp",
                    confidence=0.5,
                    center_px=(320.0, 240.0)
                ))
                
        mapper.update(obs_list, drone_pos_ned=drone_pos, now_ns=ts_ns)
    
    # Prune one last time to clear old unconfirmed candidates
    mapper.update([], drone_pos_ned=drone_pos, now_ns=150*100000000)
    
    course = mapper.course()
    assert len(course) == 3, f"Expected 3 confirmed gates, got {len(course)}"
    assert len(mapper.gates) == 3, f"Expected exactly 3 mapped gates (no phantoms), got {len(mapper.gates)}"
    
    for i, cg in enumerate(course):
        tg = true_gates[i]
        pos_err = np.linalg.norm(cg.pos - tg['pos'])
        assert pos_err < 0.5, f"Gate {i} RMSE > 0.5m: {pos_err}"
        print(f"Gate {i} pos_err: {pos_err:.3f}")
        
        dot_product = np.dot(cg.normal, drone_pos - tg['pos'])
        assert dot_product > 0, f"Normal not facing drone! normal={cg.normal}"

    # JSON round-trip
    j = mapper.to_json()
    mapper2 = GateMapper.from_json(j, cfg=cfg)
    assert len(mapper2.course()) == 3
    assert np.allclose(mapper2.course()[0].pos, course[0].pos)
    
    print("ALL MAPPER TESTS PASSED")

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    run_test()
