"""GateChaser -- VQ2 reactive visual servo: fly toward the detected gate.

VQ2 gives no absolute position (no LOCAL_POSITION_NED, no baro, no mag) -- only the IMU
(attitude, via state_estimator) and the camera. So we DON'T navigate a map; we chase the gate
the camera sees, frame by frame. Every frame ``shared_data['vision']['gate_body']`` is the gate
centre in BODY axes (x forward, y right, z down) -- a DRIFT-FREE relative fix. We turn that into
a velocity command:

  * forward  : approach the gate (throttled down while we're still off-centre / off-height, so
               the drone CENTRES and CLIMBS to the gate first, then drives through);
  * strafe   : null the lateral offset (y) -> gate horizontally centred;
  * vertical : null the vertical offset (z) -> match the gate's height (this is also our only
               altitude reference under VQ2 -- the gate tells us how high to be).

All loops close through VISION, so the controller's missing horizontal velocity feedback
doesn't matter: a velocity command becomes a lean, the drone moves, and the next frame's
gate_body reflects the result. Body velocities are mapped to NED with the SAME convention as
teleop (the inverse of controller.velocity_to_attitude), so a "forward" command becomes a
forward lean. Heading is HELD (no yaw command in v1 -- the gate starts ahead and the 90deg FOV
is wide; yaw-to-acquire-the-next-gate is a later step).

Pass-through: within ``GATE_CLOSE_RANGE`` we COMMIT -- drive straight forward at the gate's
height and stop making late lateral corrections (which would only jerk us off-line). When the
gate then leaves the frame (too close to see), we COAST forward for ``GATE_COAST_S`` to clear
it, then hover and wait to reacquire the next gate.

Same ``compute_target()`` -> ``{'mode':'velocity','vel_ned','yaw',...}`` contract as the other
planners, so controller.py is unchanged.
"""

import math
import threading
import time

from config import (MAX_SPEED, MAX_VSPEED, GATE_APPROACH_SPEED, GATE_KP_FWD, GATE_KP_LAT,
                    GATE_KP_VERT, GATE_MAX_STRAFE, GATE_ALIGN_FALLOFF, GATE_MIN_ALIGN,
                    GATE_CLOSE_RANGE, GATE_COAST_S, GATE_VISION_TIMEOUT_S)

TELEM_TIMEOUT_NS = 500_000_000
MAX_ALT_M = 30.0                 # sanity guard: if our (drifty) altitude blows past this, hover


def _clamp(x, lim):
    return max(-lim, min(lim, x))


class GateChaser:

    def __init__(self, data):
        self.data = data
        if 'lock' not in self.data:
            self.data['lock'] = threading.RLock()
        self._committed = False      # within close range -> punching through
        self._commit_ts = 0          # when we last saw a close gate (for coast-through)
        self._last_vd = 0.0          # vertical command held through a brief gate loss

    def _snapshot(self):
        with self.data['lock']:
            return {
                'attitude': self.data.get('attitude'),
                'vision': self.data.get('vision'),
                'position_ned': self.data.get('position_ned'),
            }

    def _publish(self, target):
        with self.data['lock']:
            self.data['target'] = target
        return target

    def _ned(self, f_v, r_v, vd, yaw_now, **extra):
        """Body (forward, right, vd) -> NED velocity along the physical nose (cos y, -sin y),
        matching teleop / the inverse of the controller's body loop. Returns a target dict."""
        c, s = math.cos(yaw_now), math.sin(yaw_now)
        vn = c * f_v + s * r_v
        ve = -s * f_v + c * r_v
        vd = _clamp(vd, MAX_VSPEED)
        return {'mode': 'velocity', 'vel_ned': (float(vn), float(ve), float(vd)),
                'yaw': float(yaw_now), 'ts': time.time_ns(), **extra}

    def compute_target(self):
        now = time.time_ns()
        snap = self._snapshot()
        att = snap['attitude']
        yaw_now = att.get('yaw', 0.0) if att else 0.0

        # Altitude sanity guard (our vertical estimate is drifty without baro): if it claims
        # we've blown well past any gate height, descend gently rather than trust a bad climb.
        pos = snap['position_ned']
        if pos and pos.get('z') is not None and -pos['z'] > MAX_ALT_M:
            return self._publish(self._ned(0.0, 0.0, MAX_VSPEED, yaw_now,
                                           source='alt_guard', range_m=-pos['z']))

        vis = snap['vision']
        fresh = (vis and vis.get('detected') and vis.get('ts') is not None
                 and (now - vis['ts']) < GATE_VISION_TIMEOUT_S * 1e9
                 and vis.get('gate_body') is not None)

        if not fresh:
            # Lost the gate. If we just committed to a close gate, COAST forward to clear it
            # (it left the frame because we're passing through); otherwise hover and wait.
            if self._committed and (now - self._commit_ts) < GATE_COAST_S * 1e9:
                print(f'[CHASE] gate_pass  yaw={yaw_now:+.3f}', flush=True)
                return self._publish(self._ned(GATE_APPROACH_SPEED, 0.0, self._last_vd, yaw_now,
                                               source='gate_pass'))
            self._committed = False
            print(f'[CHASE] gate_LOST  yaw={yaw_now:+.3f}', flush=True)
            return self._publish(self._ned(0.0, 0.0, 0.0, yaw_now, source='gate_lost'))

        gx, gy, gz = vis['gate_body']                       # forward, right, down (body)
        rng = vis.get('range_m') or math.sqrt(gx * gx + gy * gy + gz * gz)
        vd = _clamp(GATE_KP_VERT * gz, MAX_VSPEED)          # gz<0 (gate above) -> climb
        self._last_vd = vd

        if rng < GATE_CLOSE_RANGE:
            # Commit: punch straight through at the gate's height, stop late lateral steering.
            self._committed = True
            self._commit_ts = now
            print(f'[CHASE] COMMIT  rng={rng:.1f}m  gb=({gx:+.1f},{gy:+.1f},{gz:+.1f})'
                  f'  vd={vd:+.2f}  yaw={yaw_now:+.3f}', flush=True)
            return self._publish(self._ned(GATE_APPROACH_SPEED, 0.0, vd, yaw_now,
                                           source='gate_commit', range_m=rng))

        self._committed = False
        # Forward approach, THROTTLED by how far off-centre/height we still are, so the drone
        # centres + climbs to the gate before driving through it (perp offset = lateral+vertical).
        perp = math.hypot(gy, gz)
        align = max(GATE_MIN_ALIGN, 1.0 - perp / GATE_ALIGN_FALLOFF)
        f_v = min(GATE_APPROACH_SPEED, GATE_KP_FWD * gx) * align
        f_v = max(0.0, min(f_v, MAX_SPEED))
        r_v = _clamp(GATE_KP_LAT * gy, GATE_MAX_STRAFE)     # gy>0 (gate right) -> strafe right
        tgt = self._ned(f_v, r_v, vd, yaw_now, source='gate_track', range_m=rng)
        vel = tgt['vel_ned']
        print(f'[CHASE] track  rng={rng:5.1f}m  gb=({gx:+5.1f},{gy:+5.1f},{gz:+5.1f})'
              f'  r_v={r_v:+.2f}  ned=({vel[0]:+.2f},{vel[1]:+.2f},{vel[2]:+.2f})'
              f'  yaw={yaw_now:+.3f}', flush=True)
        return self._publish(tgt)
