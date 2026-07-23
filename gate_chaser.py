import math
import time

import config

CX = config.CAMERA_CX
CY = config.CAMERA_CY
F  = config.CAMERA_FOCAL_PX
MIN_CONFIDENCE = 0.30


class GateChaserPlanner:
    """Visual servo: fly toward the nearest detected gate."""
    name = 'gate_chase'

    def __init__(self):
        self._lost_since = None

    def compute_target(self, shared_data):
        shared_data['planner_mode'] = self.name
        log = shared_data.get('log_event', lambda e, d='': None)
        gate = shared_data.get('gate_detection')

        if gate is None or gate.get('confidence', 0.0) < MIN_CONFIDENCE:
            conf_str = f"{gate['confidence']:.2f}" if gate else 'none'
            if self._lost_since is None:
                self._lost_since = time.time()
                log('GATE_LOST', f'conf={conf_str}')
            return {'vn': 0.0, 've': 0.0, 'vd': 0.0, 'yaw_rate': 0.0}

        self._lost_since = None

        u, v    = gate['center_px']
        area_px = gate['area_px']
        conf    = gate['confidence']

        # Range estimate: apparent side = F * L / r  →  r = F * L / sqrt(area)
        side_px = math.sqrt(max(area_px, 1.0))
        rng = F * config.GATE_INNER_M / side_px

        # Body-frame gate offset (camera-tilt corrected)
        gy = rng * (u - CX) / F
        gz = rng * ((v - CY) / F - math.tan(config.CAMERA_TILT_RAD))

        # Body-frame velocity commands
        r_v = gy * config.GATE_KP_LAT
        r_v = max(-config.GATE_MAX_STRAFE, min(config.GATE_MAX_STRAFE, r_v))

        vd = gz * config.GATE_KP_VERT
        vd = max(-config.GATE_MAX_VERT, min(config.GATE_MAX_VERT, vd))

        if rng < config.GATE_CLOSE_RANGE:
            f_v = config.GATE_APPROACH_SPD * 1.5
            log('GATE_COMMIT', f'rng={rng:.1f}m')
        else:
            f_v = min(config.GATE_APPROACH_SPD, rng * 0.15)

        # Body → NED
        yaw = (shared_data.get('attitude') or {}).get('yaw', 0.0)
        vn  =  math.cos(yaw) * f_v - math.sin(yaw) * r_v
        ve  =  math.sin(yaw) * f_v + math.cos(yaw) * r_v

        print(
            f'[CHASE] rng={rng:5.1f}m  u={u:.0f}  v={v:.0f}  '
            f'gy={gy:+.2f}  gz={gz:+.2f}  conf={conf:.2f}  '
            f'r_v={r_v:+.2f}  vd={vd:+.2f}  ned=({vn:+.2f},{ve:+.2f},{vd:+.2f})',
            flush=True,
        )

        return {'vn': vn, 've': ve, 'vd': vd, 'yaw_rate': 0.0}
