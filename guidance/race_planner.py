import time
import math
import numpy as np

from mapping.gate_mapper import GateMapper
from guidance.path import Path, carrot_velocity

class RacePlanner:
    def __init__(self, shared_data, mapper: GateMapper, cfg):
        self.data = shared_data
        self.mapper = mapper
        self.cfg = cfg
        self.state = 'TAKEOFF'
        self.active_gate_idx = 0      # "which gate are we targeting next"
        
        # FSM state vars
        self.boot_yaw = None
        self.scan_start_ts = None
        self.scan_base_yaw = None
        
        self.path = None
        self.frozen_vel = None
        self.frozen_yaw = None
        
    def _hover(self, yaw_rad, source, ts):
        return {'mode': 'velocity', 'vel_ned': (0.0, 0.0, 0.0), 'yaw': yaw_rad, 'source': source, 'state': 'HOLD', 'ts': ts}
        
    def _find_target_gate(self):
        """Find the mapper gate closest to the current target index.
        
        Returns (MappedGate, index_in_mapper_gates) or (None, None).
        The planner tracks gates by active_gate_idx, but the mapper may
        have created them in any order. We find the active gate whose
        position is best aligned with what we expect as the 'next' gate.
        
        Simple strategy: return the Nth active gate by insertion order,
        clamped to bounds. The mapper already maintains insertion order
        matching observation order (i.e. closest gates first).
        """
        active = self.mapper.get_active_gates()
        if self.active_gate_idx < len(active):
            return active[self.active_gate_idx], self.active_gate_idx
        return None, None
        
    def compute_target(self) -> dict:
        now = time.time_ns()
        with self.data['lock']:
            pos_dict = self.data.get('position_ned')
            att = self.data.get('attitude')
            
        if not pos_dict or not att:
            return self._hover(0.0, 'watchdog_hover', now)
            
        pos = np.array([pos_dict['x'], pos_dict['y'], pos_dict['z']])
        yaw = att['yaw']
        
        # Telemetry watchdog
        ts_diff = now - pos_dict['ts']
        if ts_diff > 500_000_000:
            return self._hover(yaw, 'watchdog_hover', now)
            
        # Alt guard
        max_alt = getattr(self.cfg, 'MAX_ALT_M', 30.0)
        if -pos[2] > max_alt:
            return {'mode': 'velocity', 'vel_ned': (0.0, 0.0, getattr(self.cfg, 'MAX_VSPEED', 3.0)), 'yaw': yaw, 'source': 'alt_guard', 'state': 'HOLD', 'ts': now}
            
        # FSM logic
        if self.state == 'TAKEOFF':
            if self.boot_yaw is None:
                self.boot_yaw = yaw
            takeoff_alt = getattr(self.cfg, 'TAKEOFF_ALT_M', 3.0)
            if -pos[2] >= takeoff_alt - 0.5:
                # Transition
                gate, _ = self._find_target_gate()
                if gate is not None:
                    self.state = 'APPROACH'
                else:
                    self.state = 'SCAN'
                    self.scan_start_ts = now
                    self.scan_base_yaw = yaw
                return self.compute_target()
            else:
                return {'mode': 'velocity', 'vel_ned': (0.0, 0.0, -2.0), 'yaw': self.boot_yaw, 'source': 'fsm:TAKEOFF', 'state': 'TAKEOFF', 'ts': now}
                
        elif self.state == 'SCAN':
            gate, _ = self._find_target_gate()
            if gate is not None:
                self.state = 'APPROACH'
                return self.compute_target()
            else:
                dt_s = (now - self.scan_start_ts) / 1e9
                scan_yaw_max = math.radians(getattr(self.cfg, 'SCAN_YAW_MAX_DEG', 45.0))
                
                # ── Seek direction from previously seen gates ──
                all_gates = self.mapper.gates
                idx = self.active_gate_idx
                if idx >= 2 and len(all_gates) >= idx:
                    g_prev = all_gates[idx - 2]
                    g_last = all_gates[idx - 1]
                    seek_dir = g_last.pos - g_prev.pos
                    seek_alt = g_last.pos[2] + (g_last.pos[2] - g_prev.pos[2])
                elif idx >= 1 and len(all_gates) >= idx:
                    g_last = all_gates[idx - 1]
                    seek_dir = np.array([math.cos(g_last.yaw), math.sin(g_last.yaw), 0.0])
                    seek_alt = g_last.pos[2]
                else:
                    seek_dir = np.array([math.cos(self.scan_base_yaw), math.sin(self.scan_base_yaw), 0.0])
                    seek_alt = pos[2]
                    
                scan_speed = getattr(self.cfg, 'SCAN_SPEED', 1.0)
                h_dir = seek_dir[:2].copy()
                h_norm = float(np.linalg.norm(h_dir))
                if h_norm > 1e-6:
                    h_dir /= h_norm
                else:
                    h_dir = np.array([math.cos(self.scan_base_yaw), math.sin(self.scan_base_yaw)])
                    
                base_yaw = float(math.atan2(h_dir[1], h_dir[0]))
                yaw_cmd = base_yaw + scan_yaw_max * math.sin(dt_s * 0.5)
                
                vn = float(scan_speed * h_dir[0])
                ve = float(scan_speed * h_dir[1])
                
                kp_vert = getattr(self.cfg, 'KP_VERT_PATH', 1.2)
                max_vspeed = getattr(self.cfg, 'MAX_VSPEED', 3.0)
                vd = kp_vert * (seek_alt - pos[2])
                vd = max(-max_vspeed, min(max_vspeed, vd))
                
                return {'mode': 'velocity', 'vel_ned': (vn, ve, vd), 'yaw': yaw_cmd, 'source': 'fsm:SCAN', 'state': 'SCAN', 'ts': now}
                
        elif self.state == 'APPROACH':
            gate, _ = self._find_target_gate()
            if gate is None:
                self.state = 'SCAN'
                self.scan_start_ts = now
                self.scan_base_yaw = yaw
                return self.compute_target()
            
            dist_to_gate = float(np.linalg.norm(pos - gate.pos))
            normal = np.array([math.cos(gate.yaw), math.sin(gate.yaw), 0.0])
            
            commit_dist = getattr(self.cfg, 'COMMIT_DIST_M', 4.0)
            if dist_to_gate < commit_dist:
                # Before entering PASS, ensure we have a velocity toward and through the gate
                cruise = getattr(self.cfg, 'CRUISE_SPEED', 3.0)
                # Point toward the exit (gate center + overshoot along normal)
                overshoot_m = getattr(self.cfg, 'EXIT_OVERSHOOT_M', 2.5)
                exit_pt = gate.pos + overshoot_m * normal
                to_exit = exit_pt - pos
                dist_exit = float(np.linalg.norm(to_exit))
                if dist_exit > 1e-6:
                    self.frozen_vel = to_exit / dist_exit * cruise
                else:
                    self.frozen_vel = normal * cruise
                self.frozen_yaw = gate.yaw
                self.state = 'PASS'
                return self.compute_target()
            
            # Build approach path: current pos → standoff → gate center → exit
            standoff_m = getattr(self.cfg, 'APPROACH_STANDOFF_M', 3.0)
            overshoot_m = getattr(self.cfg, 'EXIT_OVERSHOOT_M', 2.5)
            
            standoff = gate.pos - standoff_m * normal
            exit_pt = gate.pos + overshoot_m * normal
            
            pts = [pos.copy(), standoff, gate.pos.copy(), exit_pt]
            yaws = [yaw, gate.yaw, gate.yaw, gate.yaw]
            
            self.path = Path(pts, yaws, loop=False, cfg=self.cfg)
            vel_ned = np.array([pos_dict.get('vx', 0.0), pos_dict.get('vy', 0.0), pos_dict.get('vz', 0.0)])
            
            vel_cmd, yaw_cmd, s_proj, _ = carrot_velocity(self.path, pos, vel_ned, self.cfg, getattr(self, '_last_t', 0.0))
            self._last_t = s_proj
            
            self.frozen_vel = vel_cmd
            self.frozen_yaw = yaw_cmd
            
            # Cross-track distance guard
            xte = float(np.linalg.norm(self.path.sample(s_proj)[0][:2] - pos[:2]))
            if xte > getattr(self.cfg, 'MAX_PATH_DIST_M', 60.0):
                return self._hover(yaw, 'dist_guard', now)
                
            return {'mode': 'velocity', 'vel_ned': tuple(float(c) for c in vel_cmd), 'yaw': float(yaw_cmd), 'source': 'fsm:APPROACH', 'state': 'APPROACH', 'ts': now}
                    
        elif self.state == 'PASS':
            gate, _ = self._find_target_gate()
            if gate is not None:
                normal = np.array([math.cos(gate.yaw), math.sin(gate.yaw), 0.0])
                vec = pos - gate.pos
                plane_dist = float(np.dot(vec, normal))
                
                if plane_dist > 0.0:
                    self.state = 'ADVANCE'
                    return self.compute_target()
                    
                # If no frozen_vel, steer toward the gate exit
                if self.frozen_vel is None or float(np.linalg.norm(self.frozen_vel)) < 0.1:
                    cruise = getattr(self.cfg, 'CRUISE_SPEED', 3.0)
                    overshoot_m = getattr(self.cfg, 'EXIT_OVERSHOOT_M', 2.5)
                    exit_pt = gate.pos + overshoot_m * normal
                    to_exit = exit_pt - pos
                    dist_exit = float(np.linalg.norm(to_exit))
                    if dist_exit > 1e-6:
                        self.frozen_vel = to_exit / dist_exit * cruise
                    else:
                        self.frozen_vel = normal * cruise
                    self.frozen_yaw = gate.yaw
                    
            if self.frozen_vel is None:
                self.frozen_vel = np.array([0.0, 0.0, 0.0])
            if self.frozen_yaw is None:
                self.frozen_yaw = yaw
            return {'mode': 'velocity', 'vel_ned': tuple(float(c) for c in self.frozen_vel), 'yaw': float(self.frozen_yaw), 'source': 'fsm:PASS', 'state': 'PASS', 'ts': now}
                
        elif self.state == 'ADVANCE':
            self.active_gate_idx += 1
            self.frozen_vel = None
            self.frozen_yaw = None
            self._last_t = 0.0
            gate, _ = self._find_target_gate()
            if gate is not None:
                self.state = 'APPROACH'
            else:
                self.state = 'SCAN'
                self.scan_start_ts = now
                self.scan_base_yaw = yaw
            return self._hover(yaw, 'fsm:ADVANCE', now)  # transition tick
            
        elif self.state == 'FINISH':
            return self._hover(yaw, 'fsm:FINISH', now)
            
        # Default fallback
        return self._hover(yaw, 'fsm:UNKNOWN', now)
