import math
import numpy as np
from dataclasses import dataclass
from typing import List
from vision.gate_estimator import GateObservation

@dataclass
class MappedGate:
    id: int
    pos: np.ndarray      # shape (3,) NED position
    yaw: float           # radians
    hits: int = 1
    active: bool = False

def _circular_interp(y1: float, y2: float, alpha: float) -> float:
    # shortest path circular interpolation
    dy = (y2 - y1 + math.pi) % (2 * math.pi) - math.pi
    return y1 + alpha * dy

class GateMapper:
    def __init__(self, match_radius: float = 2.0, alpha: float = 0.2, min_hits_active: int = 5):
        self.gates: List[MappedGate] = []
        self.match_radius = match_radius
        self.alpha = alpha
        self.min_hits_active = min_hits_active
        self._next_id = 1

    def update(self, observations: List[GateObservation]):
        """
        Integrates a list of new observations into the persistent map.
        """
        for obs in observations:
            obs_pos = obs.gate_ned
            obs_yaw = float(np.arctan2(obs.normal_ned[1], obs.normal_ned[0]))
            
            best_dist = float('inf')
            best_gate = None
            
            for g in self.gates:
                d = float(np.linalg.norm(g.pos - obs_pos))
                if d < self.match_radius and d < best_dist:
                    best_dist = d
                    best_gate = g
                    
            if best_gate is not None:
                best_gate.pos = (1 - self.alpha) * best_gate.pos + self.alpha * obs_pos
                best_gate.yaw = _circular_interp(best_gate.yaw, obs_yaw, self.alpha)
                best_gate.hits += 1
                if best_gate.hits >= self.min_hits_active:
                    best_gate.active = True
            else:
                new_gate = MappedGate(
                    id=self._next_id,
                    pos=obs_pos.copy(),
                    yaw=obs_yaw,
                    hits=1,
                    active=(self.min_hits_active <= 1)
                )
                self.gates.append(new_gate)
                self._next_id += 1

    def get_active_gates(self) -> List[MappedGate]:
        return [g for g in self.gates if g.active]
