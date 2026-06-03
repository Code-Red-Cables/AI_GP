"""Planning: pick the active gate and emit a velocity setpoint (see PLAN.md 8.4 / Task C).

The planner reads the shared blackboard each control tick and decides where to go:

1. If a **fresh, confident vision detection** exists, steer toward the gate it sees
   (body-relative vector rotated into NED with the current attitude — needs no
   absolute position fix).
2. Otherwise fall back to the **known track geometry**: the gate indexed by the
   race-status ``active_gate_index`` (the sim increments this as gates are passed,
   so it doubles as our gate-passed signal — no manual bookkeeping).
3. If telemetry is stale (we'd be flying blind) or nothing is targetable, command a
   **hover** (zero velocity) rather than guessing — a safety watchdog.

Output is written to ``shared_data['target']`` as a NED velocity command + desired
yaw; `controller.py` maps it to attitude+thrust (roll/pitch/yaw/thrust) via SET_ATTITUDE_TARGET.
"""

import math
import time

import numpy as np

import camera_model as cm

# --------------------------------------------------------------------------------------
# Guidance tuning (start conservative; tune against the deterministic course).
# --------------------------------------------------------------------------------------
MAX_SPEED = 2.0            # m/s cap on commanded velocity magnitude
MAX_VSPEED = 1.0           # m/s cap on the *vertical* component (climb or descend).
                           # The FPV camera is tilted up 20deg, so a gate seen near
                           # image-centre is estimated well above the drone, and a single
                           # size-based range is noisy; without a separate vertical limit
                           # the guidance commands an aggressive climb that the vehicle's
                           # vertical loop overshoots. Keep this < MAX_SPEED.
KP_POS = 3              # proportional gain: speed = KP_POS * distance, capped
PASS_THROUGH_DIST = 2.5    # m: within this range, punch through at full speed
ARRIVE_DIST = 0.1          # m: closer than this we consider ourselves at the waypoint

CONF_MIN = 0.40            # min vision confidence to trust a detection
VISION_TIMEOUT_NS = 300_000_000    # 300 ms: older vision is "stale"
TELEM_TIMEOUT_NS = 500_000_000     # 500 ms without pose -> hover (watchdog)

# Altitude-envelope safety guard. NED z is negative-up with the origin at the arm
# point on the ground; race gates sit only a few metres up. If we ever climb past
# MAX_ALT_M the vertical loop has run away (observed: a 586 m climb in
# logs/run_1780515186.jsonl), so we abandon the gate target and descend at MAX_VSPEED
# until back below the ceiling. A pure client-side fail-safe.
MAX_ALT_M = 15.0           # m above the arm point we ever allow before recovering


def _clamp_speed(vec, max_speed):
    """Scale a velocity vector so its magnitude never exceeds max_speed."""
    n = float(np.linalg.norm(vec))
    if n > max_speed and n > 1e-9:
        return vec * (max_speed / n)
    return vec


class Planner:
    """Decides the current target setpoint from the shared blackboard."""

    def __init__(self, data):
        self.data = data
        if 'lock' not in self.data:
            import threading
            self.data['lock'] = threading.RLock()

    def _snapshot(self):
        """Atomically read the bits of shared_data we need."""
        with self.data['lock']:
            return {
                'vision': self.data.get('vision'),
                'gates': self.data.get('gates'),
                'race': self.data.get('race'),
                'attitude': self.data.get('attitude'),
                'odometry': self.data.get('odometry'),
                'position_ned': self.data.get('position_ned'),
            }

    @staticmethod
    def _pose(s):
        """Return (position_ned tuple|None, (roll,pitch,yaw)|None) from telemetry."""
        odo, pos, att = s['odometry'], s['position_ned'], s['attitude']
        position = None
        if odo is not None:
            position = odo['pos']
        elif pos is not None:
            position = (pos['x'], pos['y'], pos['z'])

        rpy = None
        if att is not None:
            rpy = (att['roll'], att['pitch'], att['yaw'])
        elif odo is not None:
            from vision_rx import _quat_to_rpy
            d = _quat_to_rpy(odo['q'])
            rpy = (d['roll'], d['pitch'], d['yaw'])
        return position, rpy

    def _target_offset_ned(self, s, now):
        """Vector from the drone to the target gate, in world NED axes.

        Returns (offset_ned, source) or (None, reason) when not targetable.
        """
        position, rpy = self._pose(s)

        # 1) Fresh, confident vision detection -> steer to what we see.
        vis = s['vision']
        if (vis and vis.get('detected') and vis.get('confidence', 0) >= CONF_MIN
                and vis.get('ts') and (now - vis['ts']) <= VISION_TIMEOUT_NS):
            gate_body = np.asarray(vis['gate_body'], float)
            if rpy is not None:
                return cm.body_to_ned(gate_body, *rpy), 'vision'
            # No attitude: treat body vector as NED (best effort, level assumption).
            return gate_body, 'vision_level'

        # 2) Known track geometry indexed by active_gate_index.
        gates, race = s['gates'], s['race']
        if gates and position is not None:
            idx = int(race['active_gate_index']) if race and race.get('active_gate_index', -1) >= 0 else 0
            idx = max(0, min(idx, len(gates) - 1))
            gate_pos = np.asarray(gates[idx]['pos_ned'], float)
            return gate_pos - np.asarray(position, float), 'known'

        return None, 'no_target'

    def _publish(self, target):
        """Store the target on the blackboard and return it."""
        with self.data['lock']:
            self.data['target'] = target
        return target

    def compute_target(self):
        """Compute and publish the current velocity setpoint. Returns the target dict."""
        now = time.time_ns()
        s = self._snapshot()
        position, rpy = self._pose(s)
        yaw_hold = rpy[2] if rpy else 0.0

        # Watchdog: no recent pose at all -> hover (we'd otherwise be flying blind).
        att_ts = s['attitude']['ts'] if s['attitude'] else (s['odometry']['ts'] if s['odometry'] else None)
        telem_stale = att_ts is None or (now - att_ts) > TELEM_TIMEOUT_NS
        if telem_stale:
            return self._publish({'mode': 'velocity', 'vel_ned': (0.0, 0.0, 0.0),
                                  'yaw': yaw_hold, 'source': 'watchdog_hover', 'ts': now})

        # Altitude-envelope safety guard (overrides the gate target). NED z is
        # negative-up, so height above the arm point is -z. Past the ceiling the
        # vertical loop has run away -> abandon the gate and descend back down.
        if position is not None:
            altitude = -float(position[2])
            if altitude > MAX_ALT_M:
                return self._publish({'mode': 'velocity', 'vel_ned': (0.0, 0.0, MAX_VSPEED),
                                      'yaw': yaw_hold, 'range_m': altitude,
                                      'source': 'alt_guard', 'ts': now})

        offset, source = self._target_offset_ned(s, now)
        if offset is None:
            return self._publish({'mode': 'velocity', 'vel_ned': (0.0, 0.0, 0.0),
                                  'yaw': yaw_hold, 'source': 'hover', 'ts': now})

        dist = float(np.linalg.norm(offset))
        if dist <= ARRIVE_DIST:
            speed = MAX_SPEED  # keep moving through; sim advances the gate index
        elif dist <= PASS_THROUGH_DIST:
            speed = MAX_SPEED  # commit to the pass — don't decelerate inside the gate
        else:
            speed = min(MAX_SPEED, KP_POS * dist)

        direction = offset / dist if dist > 1e-9 else np.zeros(3)
        vel_ned = _clamp_speed(direction * speed, MAX_SPEED)

        # Cap the vertical component separately. The up-tilted camera biases the gate
        # elevation upward and the range estimate is noisy, so a large commanded
        # climb/descent is rarely justified and is what triggers vertical overshoot.
        vd = max(-MAX_VSPEED, min(MAX_VSPEED, float(vel_ned[2])))

        # Point the nose (and thus the camera) at the gate: NED yaw, 0=North,+East.
        yaw = math.atan2(offset[1], offset[0])

        return self._publish({'mode': 'velocity',
                              'vel_ned': (float(vel_ned[0]), float(vel_ned[1]), vd),
                              'yaw': float(yaw),
                              'range_m': dist,
                              'source': source,
                              'ts': now})
