"""HoverPlanner -- the VQ2 Phase-0 milestone: take off to a set altitude and hold, using
ONLY our own IMU+baro state estimate (no LOCAL_POSITION_NED, no ATTITUDE from the sim).

It commands zero horizontal velocity and a vertical velocity that drives the estimated
altitude (``shared_data['position_ned']['z']``, produced by state_estimator.py from baro)
toward ``HOVER_ALT_M``, holding the boot heading. If the drone takes off and holds a stable
hover on estimated state alone, the AHRS + baro pipeline works and we can move on to the
reactive gate-chaser (Phase 1). Same ``compute_target()`` -> velocity contract as the other
planners, so the controller is unchanged.
"""

import threading
import time

from config import MAX_VSPEED

HOVER_ALT_M = 1.5      # target height above the arm point (m)
KP_CLIMB = 0.6         # vertical velocity (m/s) commanded per metre of altitude error
TELEM_TIMEOUT_NS = 500_000_000


class HoverPlanner:

    def __init__(self, data, target_alt_m=HOVER_ALT_M, kp_climb=KP_CLIMB):
        self.data = data
        if 'lock' not in self.data:
            self.data['lock'] = threading.RLock()
        self.target_z = -abs(target_alt_m)        # NED down: up is negative
        self.kp = kp_climb

    def compute_target(self):
        now = time.time_ns()
        with self.data['lock']:
            pos = self.data.get('position_ned')
            att = self.data.get('attitude')
        yaw = att.get('yaw', 0.0) if att else 0.0

        # Watchdog: no fresh estimate -> hold level, hover thrust (vd=0).
        if not pos or pos.get('z') is None or pos.get('ts') is None or \
                (now - pos['ts']) > TELEM_TIMEOUT_NS:
            return self._publish({'mode': 'velocity', 'vel_ned': (0.0, 0.0, 0.0),
                                  'yaw': float(yaw), 'source': 'vq2_hover_nostate', 'ts': now})

        z = pos['z']
        err = self.target_z - z                   # <0 => below target => climb (vd negative)
        vd = max(-MAX_VSPEED, min(MAX_VSPEED, self.kp * err))
        return self._publish({'mode': 'velocity', 'vel_ned': (0.0, 0.0, float(vd)),
                              'yaw': float(yaw), 'range_m': abs(err),
                              'source': 'vq2_hover', 'ts': now})

    def _publish(self, target):
        with self.data['lock']:
            self.data['target'] = target
        return target
