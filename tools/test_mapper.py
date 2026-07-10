import numpy as np
import random
from vision.gate_estimator import GateObservation
from mapping.gate_mapper import GateMapper

def run_test():
    mapper = GateMapper(match_radius=2.0, alpha=0.2, min_hits_active=5)

    true_gate_pos = np.array([10.0, 5.0, -2.0])
    true_gate_yaw = 0.5
    
    # 20 frames of simulation
    for i in range(20):
        obs_list = []
        
        # 1. Provide noisy observation of the true gate
        noisy_pos = true_gate_pos + np.random.normal(0, 0.5, size=3)
        noisy_yaw = true_gate_yaw + np.random.normal(0, 0.1)
        obs_normal = np.array([np.cos(noisy_yaw), np.sin(noisy_yaw), 0.0])
        obs_list.append(GateObservation(
            ts_ns=i*100000000,
            frame_id=i,
            gate_body=np.zeros(3),
            normal_body=np.zeros(3),
            gate_ned=noisy_pos,
            normal_ned=obs_normal,
            range_m=10.0,
            method="pnp",
            confidence=0.9,
            center_px=(320.0, 240.0)
        ))
        
        # 2. Add random false positive noise occasionally (1 hit)
        if random.random() < 0.5:
            fp_pos = np.random.uniform(-50, 50, size=3)
            fp_yaw = random.uniform(-3.14, 3.14)
            fp_normal = np.array([np.cos(fp_yaw), np.sin(fp_yaw), 0.0])
            obs_list.append(GateObservation(
                ts_ns=i*100000000,
                frame_id=i,
                gate_body=np.zeros(3),
                normal_body=np.zeros(3),
                gate_ned=fp_pos,
                normal_ned=fp_normal,
                range_m=10.0,
                method="pnp",
                confidence=0.5,
                center_px=(320.0, 240.0)
            ))
            
        mapper.update(obs_list)
        
    active = mapper.get_active_gates()
    assert len(active) == 1, f"Expected 1 active gate, got {len(active)}"
    
    ag = active[0]
    pos_err = np.linalg.norm(ag.pos - true_gate_pos)
    yaw_err = abs(ag.yaw - true_gate_yaw)
    
    print(f"Active gate 1: pos={ag.pos}, err={pos_err:.3f}")
    print(f"Active gate 1 yaw: {ag.yaw:.3f}, err={yaw_err:.3f}")
    print(f"Total mapped gates (including inactive noise): {len(mapper.gates)}")
    
    assert pos_err < 1.0, f"Position error {pos_err} too large"
    assert yaw_err < 0.5, f"Yaw error {yaw_err} too large"
    
    print("ALL MAPPER TESTS PASSED")

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    run_test()
