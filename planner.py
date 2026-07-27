"""Planning: pick the active gate and emit a velocity setpoint (see PLAN.md 8.4 / Task C).

The planner reads the shared blackboard each control tick and decides where to go:

1. If a **fresh, confident vision detection** exists, steer toward the gate it sees
   (body-relative vector rotated into NED with the current attitude — needs no
   absolute position fix).
2. Otherwise fall back to the **preplanned course map**: the learned world position of
   the gate indexed by ``active_gate_index`` (the course is deterministic — spec 3.5 —
   so a map recorded on a prior run is replayable). This is what carries us toward a gate
   that is not yet in the narrow FOV instead of hovering. Inert unless preplanning is on.
3. Otherwise fall back to the **sim-broadcast track geometry** — absent on this sim
   (spec 4.3 defines no track-data message), kept only for forward-compat.
4. If telemetry is stale (we'd be flying blind) or nothing is targetable, command a
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
YAW_FREEZE_DIST = 4.0      # m: within this range, FREEZE the yaw lock and fly straight
                           # through. At point-blank the gate fills the frame and its
                           # detected bearing whipsaws (log 2026-06-06: gb_az swung
                           # -10->+43->-47->+37 deg in 0.5s while passing gate 1), which
                           # would otherwise re-aim the lock and veer us off the exit.
ARRIVE_DIST = 0.1          # m: closer than this we consider ourselves at the waypoint

# After passing a gate, the next gate is usually further down-course and NOT yet in the
# narrow FOV. If we drop to a hover the instant the gate index advances, two bad things
# happen (log 2026-06-06): (1) the hard brake pitches the airframe ~19deg ("look up then
# down") which swings the camera, and (2) we park just past the gate and the next one never
# enters frame, so vision never re-acquires and we sit forever. Instead we COAST straight
# ahead on the heading we passed through on until the next gate is detected (or the window
# expires) — steady camera, and the course ahead scrolls into view.
POST_GATE_COAST_NS = 6_000_000_000   # coast/search this long (ns) after a gate advance
POST_GATE_COAST_SPEED = 1.2          # m/s forward during the coast (< MAX_SPEED, gentle)
# The NEXT gate sits BELOW the line we flew through gate 1 on, and the up-tilted (~20deg)
# FPV camera can't see anything that low from here (log 2026-06-06: 0 detections in 4s of
# level forward coast). So while coasting we also DESCEND gently to bring the lower gate up
# into the camera's view, down to a safety floor.
POST_GATE_DESCEND_SPEED = 0.6        # m/s descent during the post-gate search (<= MAX_VSPEED)
POST_GATE_MIN_ALT_M = 0.4            # don't descend below this height (NED -z) while searching

CONF_MIN = 0.25            # min vision confidence to trust a detection. Loosened 0.40 ->
                           # 0.25 so a partially-visible / edge-clipped gate (low squareness,
                           # so low confidence) is accepted sooner — e.g. the lower next-gate
                           # as it first edges into the up-tilted frame during the descent.
VISION_TIMEOUT_NS = 300_000_000    # 300 ms: older vision is "stale"
TELEM_TIMEOUT_NS = 500_000_000     # 500 ms without pose -> hover (watchdog)

# Vision plausibility gate. The gate is a fixed object, so its estimated range cannot
# teleport between frames. A burst of garbage detections (range jumping 13m -> 38 -> 64m
# and the bearing swinging ~80deg in 0.4s) once hijacked guidance: the planner chased the
# phantom, commanded a 104 deg/s yaw that swept the REAL gate out of the camera FOV, and
# tracking died for the remaining 11s of the run (log 2026-06-05). Reject any detection
# beyond MAX_GATE_RANGE_M, or that jumps more than RANGE_JUMP_MAX_M from the last accepted
# range while that reference is still recent, so a single bad frame can't whip us around.
MAX_GATE_RANGE_M = 35.0            # gates on this course sit within ~25m; beyond = garbage
RANGE_JUMP_MAX_M = 12.0            # max believable range change vs. the last good detection
RANGE_MEMORY_NS = 700_000_000      # how long the last accepted range stays a reference

# Gate memory. The camera FOV is narrow, so the gate leaves view ~1s into the approach
# and detection stops; with no track geometry from the sim, the planner had nothing to
# track and froze in place ~2-3m off the start every run (logs 2026-06-05/06). Instead,
# remember the gate's ABSOLUTE NED position and keep dead-reckoning toward it (via
# odometry) while vision is unavailable, so the drone flies in, closes the distance, and
# re-acquires the gate rather than parking.
GATE_MEMORY_TIMEOUT_NS = 6_000_000_000   # forget a remembered gate after 6s unseen

# Filtered WORLD waypoint. The per-frame gate BEARING is unreliable while the drone
# maneuvers: log 2026-06-06 shows yaw_tgt swinging -179deg -> -82deg with the drone
# nearly stationary and the gate dead-ahead at a steady ~21m — i.e. the single-frame
# bearing is corrupt, and caching the LAST frame's world position (old gate-memory)
# locked onto a point ~16m off to the side, sending the drone curving away. A gate is a
# FIXED object, so its world position cannot teleport: seed the waypoint from the first
# clean detection, REJECT frames whose implied world position jumps more than
# GATE_WORLD_JUMP_M from the current belief, and EMA-refine with the rest. The drone then
# always flies to a stable world point instead of chasing per-frame bearing noise.
#
# HEIGHT is handled separately from the HORIZONTAL bearing. The up-tilted (20deg) camera
# sees a gate near image-centre at long range, which back-projects as ~8m UP in body
# frame -> the estimated gate HEIGHT is 10-14m at >18m range, but converges to the true
# ~2m once inside ~15m (log+reconstruction 2026-06-06). A waypoint seeded with the bad
# far height froze (every good near-frame was >6m away in 3D, so the jump gate rejected
# it) and the drone climbed to 5.8m and parked above the gate. Fix: gate the JUMP test on
# the HORIZONTAL distance only (bearing IS reliable far out), and only fold a frame's
# vertical estimate into the waypoint once the gate is within GATE_VERT_TRUST_RANGE_M;
# beyond that, keep the waypoint level with the drone so we approach flat, then settle
# onto the true gate height as we close in.
GATE_WORLD_JUMP_M = 6.0            # reject a detection whose HORIZONTAL pos jumps this far
GATE_WORLD_EMA = 0.30             # blend weight for accepted detections (low = stable)
GATE_VERT_TRUST_RANGE_M = 15.0    # only believe the gate's ELEVATION within this range
# Vertical aim bias: aim this far BELOW the estimated gate centre (metres, NED +down).
# The estimated centre reads a bit high (up-tilted camera + the gate ring's bright top edge
# biasing the centroid up), so a small downward bias recentres us in the opening. But too
# much drops us so low on exit that the NEXT gate sits above the frame — keep it modest.
GATE_AIM_DOWN_M = 0.3
# Re-seed guard: if the belief is wrong (e.g. a bad seed) the good detections all fail the
# jump gate forever. So when consecutive frames AGREE with each other but DISAGREE with the
# belief, treat the belief as stale and jump to the new cluster. A lone corrupt frame never
# forms a cluster, so this re-acquires a moved/mis-seeded gate without chasing noise.
GATE_RESEED_COUNT = 3             # consecutive consistent off-belief frames to re-seed
GATE_RESEED_SPREAD_M = 4.0        # cluster must be at least this tight to count
GATE_RESEED_WINDOW_NS = 1_500_000_000  # discard candidate frames older than this

# Steer yaw off the LIVE detection bearing, fly FORWARD along the nose. The per-frame
# world reconstruction is unreliable while the airframe pitches/rolls: log 2026-06-06
# shows a FIXED gate's reconstructed world bearing swinging -178deg -> -85deg in 1.6s
# (independent of roll-sign), while the drone barely moved. Chasing that reconstructed
# lateral offset made the planner yaw away, banked the quad to strafe, slid the real gate
# out of the narrow FOV, and spun into a runaway. Instead: yaw to centre the measured gate
# azimuth (atan2(body_y, body_x), which reads ~1deg when we fly straight) and command the
# horizontal velocity purely along the current heading. The controller then needs only
# PITCH (forward) and ~no ROLL, so the camera stays steady and the gate stays centred.
ROLL_TRUST_MAX = math.radians(8.0)   # skip folding vision into the waypoint above this roll

# Yaw heading-LATCH with a deadband. The per-frame vision azimuth is only trustworthy
# while we fly straight: log 2026-06-06 shows az pinned at ~1deg for 1.4s of stable
# straight flight, then the instant the yaw loop engaged, az and yaw ran away TOGETHER
# (az 1->45deg, yaw -180->-127) — positive feedback. Geometry check: at the runaway the
# gate was ~35deg to the LEFT but vision reported +33deg RIGHT, because yawing swept the
# nose across the tunnel of OTHER gates and the bearing inverted. So we do NOT chase every
# per-frame degree. We LATCH a heading and only re-aim when the gate is CONSISTENTLY and
# significantly off-centre (|az| > YAW_DEADBAND). Inside the band we hold heading, which
# keeps us in the stable straight-flight regime — and the first gate is dead ahead, so
# holding heading flies us straight through it.
# 9deg -> 3deg (2026-07-27, VIO/PnP pipeline): the wide deadband protected against
# the HSV detector's garbage per-frame bearings, but it let the gate wander to the
# frame edge before any correction — run 6 lost sight 1.7 s into the approach and
# never recovered. PnP bearings are geometric and already filtered by the
# world-consistency check (_vision_accepted), so track tightly and keep the camera
# ON the gate: continuous sight = continuous VIO fixes = everything else works.
YAW_DEADBAND_RAD = math.radians(3.0)   # within this azimuth, hold heading (no re-aim)

# Altitude-envelope safety guard. NED z is negative-up with the origin at the arm
# point on the ground; race gates sit only a few metres up. If we ever climb past
# MAX_ALT_M the vertical loop has run away (observed: a 586 m climb in
# logs/run_1780515186.jsonl), so we abandon the gate target and descend at MAX_VSPEED
# until back below the ceiling. A pure client-side fail-safe.
MAX_ALT_M = 15.0           # m above the arm point we ever allow before recovering

# Adapter from the image-space navigator to this planner's existing command
# contract (NED velocity + absolute yaw).
NAVIGATION_TIMEOUT_NS = 350_000_000
YAW_RATE_TARGET_HORIZON_S = 0.50


class Planner:
    """Decides the current target setpoint from the shared blackboard."""

    def __init__(self, data):
        self.data = data
        if 'lock' not in self.data:
            import threading
            self.data['lock'] = threading.RLock()
        # Vision plausibility-gate memory (see MAX_GATE_RANGE_M / RANGE_JUMP_MAX_M).
        self._last_range = None
        self._last_range_ts = 0
        # Gate memory (see GATE_MEMORY_TIMEOUT_NS): last seen gate absolute NED position.
        self._gate_world = None
        self._gate_world_ts = 0
        self._last_gate_idx = 0
        # Re-seed candidate cluster (see GATE_RESEED_*): off-belief measurements + ts.
        self._reseed_buf = []
        # Latched yaw heading (see YAW_DEADBAND_RAD): only re-aimed when the gate is
        # consistently off-centre; held otherwise to stay in stable straight flight.
        self._yaw_lock = None
        # Post-gate coast (see POST_GATE_COAST_*): keep flying forward to find the next gate.
        self._coast_until_ns = 0
        # True on ticks where the fresh detection AGREED with the world belief (set in
        # _target_offset_ned). The yaw lock only re-aims on those ticks: run 4 showed a
        # single frame whose "best gate" was a DIFFERENT gate re-aiming the heading 90deg
        # right and driving the drone off the course line.
        self._vision_accepted = False
        # Reactive (pnp mode) guidance state: latched approach heading + last-seen time.
        self._rx_heading = None
        self._rx_last_ns = 0
        self._rx_committed = False

        # ---- Preplanning (learn-then-replay the DETERMINISTIC course; spec 3.5) --------
        # The sim broadcasts no track geometry (spec 4.3), so we build our own gate map:
        # record each gate's world position when we pass it, persist it, and on later runs
        # fly to the KNOWN next-gate position when vision can't yet see it (fixes the
        # gate-2 hover). ALL of this is inert unless a course_map is injected AND the
        # preplan/learn flags are set, so a default run is identical to the reactive planner.
        self._course_map = self.data.get('course_map')
        self._preplan = bool(self.data.get('preplan', False))
        self._learn = bool(self.data.get('learn', False))
        # The index the planner currently believes is the active gate (race-provided when
        # available, else self-detected). Read by _target_offset_ned for the map lookup.
        self._cur_idx = 0
        # True once the CURRENT gate belief (_gate_world) has been confirmed by a vision
        # detection. Only vision-confirmed beliefs are recorded into the map, so a
        # preplan-seeded waypoint never feeds back on itself and drifts. Reset on advance.
        self._gate_world_vis = False
        self._recorded = set()   # gate indices already written this run (record once)
        # Self gate-pass detector — used ONLY when the sim provides no active_gate_index.
        # active_gate_index (race) is authoritative when present; this is the fallback so
        # learning/replay still advance gate-to-gate without it.
        self._auto_idx = 0
        self._armed_pass = False

    @staticmethod
    def _wrap(a):
        """Wrap an angle to (-pi, pi]."""
        return math.atan2(math.sin(a), math.cos(a))

    def _fresh_gate_azimuth(self, s, now):
        """Body-frame azimuth (rad, +=gate to the right) of a fresh, confident detection.

        This is the DIRECT measurement of where the gate sits relative to the nose; it is
        far more reliable for steering than the attitude-sensitive world reconstruction,
        and it is what we drive to zero to keep the gate centred. Returns None when there
        is no usable detection (caller then falls back to the world-waypoint bearing).
        """
        vis = s['vision']
        if not (vis and vis.get('detected') and vis.get('confidence', 0) >= CONF_MIN
                and vis.get('ts') and (now - vis['ts']) <= VISION_TIMEOUT_NS
                and vis.get('gate_body')):
            return None
        gb = np.asarray(vis['gate_body'], float)
        rng = float(np.linalg.norm(gb))
        if not (0.0 < rng <= MAX_GATE_RANGE_M):
            return None
        return math.atan2(float(gb[1]), float(gb[0]))

    def _vision_plausible(self, rng, now):
        """Reject teleporting/garbage gate detections that would hijack guidance."""
        if not (0.0 < rng <= MAX_GATE_RANGE_M):
            return False
        if (self._last_range is not None
                and (now - self._last_range_ts) <= RANGE_MEMORY_NS
                and abs(rng - self._last_range) > RANGE_JUMP_MAX_M):
            return False
        return True

    def _snapshot(self):
        """Atomically read the bits of shared_data we need."""
        with self.data['lock']:
            snap = {
                'vision': self.data.get('vision'),
                'navigation': self.data.get('navigation'),
                'control_source': self.data.get('control_source'),
                'gates': self.data.get('gates'),
                'race': self.data.get('race'),
                'attitude': self.data.get('attitude'),
                'odometry': self.data.get('odometry'),
                'position_ned': self.data.get('position_ned'),
            }
            # Adapt the sim's race broadcast to the 'active_gate_index' contract this
            # planner was written against (nothing ever published a 'race' key).
            # PREFER the VIO's SETTLED index: the raw active_gate telemetry flickers
            # (run 4: a flicker at 7 m out dropped the waypoint and started the
            # post-gate coast BEFORE the gate) — the estimator debounces it 0.5 s.
            if snap['race'] is None:
                settled = self.data.get('gate_idx_settled')
                rs = self.data.get('race_status')
                if settled is not None:
                    snap['race'] = {'active_gate_index': int(settled)}
                elif rs is not None:
                    # settled index not published until the first advance; 0 until then
                    snap['race'] = {'active_gate_index': 0}
        return snap

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

        updated = False

        # 1) Fresh, confident, PLAUSIBLE vision detection -> fold it into the filtered
        # WORLD waypoint (outlier-reject + EMA). We do NOT steer off a single frame's
        # bearing: that bearing swings wildly while maneuvering (log 2026-06-06).
        vis = s['vision']
        if (vis and vis.get('detected') and vis.get('confidence', 0) >= CONF_MIN
                and vis.get('ts') and (now - vis['ts']) <= VISION_TIMEOUT_NS
                and rpy is not None and position is not None
                and abs(rpy[0]) <= ROLL_TRUST_MAX):
            gate_body = np.asarray(vis['gate_body'], float)
            rng = float(np.linalg.norm(gate_body))
            if self._vision_plausible(rng, now):
                self._last_range, self._last_range_ts = rng, now
                meas_world = np.asarray(position, float) + cm.body_to_ned(gate_body, *rpy)
                # Only believe the gate's HEIGHT once we are close enough to judge it; far
                # out the up-tilted camera over-estimates the elevation badly.
                trust_z = rng <= GATE_VERT_TRUST_RANGE_M
                stale = (self._gate_world is None
                         or (now - self._gate_world_ts) > GATE_MEMORY_TIMEOUT_NS)
                if stale:
                    # First lock: trust the bearing; only adopt the measured height if
                    # close, else stay level with the drone (approach flat, settle later).
                    seed = meas_world.copy()
                    if not trust_z:
                        seed[2] = float(position[2])
                    self._gate_world = seed
                    self._gate_world_ts = now
                    self._reseed_buf = []
                    updated = True
                elif np.linalg.norm(meas_world[:2] - self._gate_world[:2]) <= GATE_WORLD_JUMP_M:
                    # HORIZONTAL position agrees with the belief (bearing is reliable even
                    # at range): refine N/E slowly. Fold in the height ONLY when close.
                    new = self._gate_world.copy()
                    new[0] = (1.0 - GATE_WORLD_EMA) * new[0] + GATE_WORLD_EMA * meas_world[0]
                    new[1] = (1.0 - GATE_WORLD_EMA) * new[1] + GATE_WORLD_EMA * meas_world[1]
                    if trust_z:
                        new[2] = (1.0 - GATE_WORLD_EMA) * new[2] + GATE_WORLD_EMA * meas_world[2]
                    self._gate_world = new
                    self._gate_world_ts = now
                    self._reseed_buf = []
                    updated = True
                else:
                    # Horizontal position disagrees with the belief. Could be a lone corrupt
                    # frame (drop it) OR the belief is wrong (e.g. a bad far seed) -> if a
                    # tight cluster of consecutive off-belief frames forms, re-seed to it.
                    self._reseed_buf = [(m, t) for (m, t) in self._reseed_buf
                                        if (now - t) <= GATE_RESEED_WINDOW_NS]
                    self._reseed_buf.append((meas_world, now))
                    pts = np.asarray([m[:2] for (m, _) in self._reseed_buf])
                    if (len(self._reseed_buf) >= GATE_RESEED_COUNT
                            and float(np.linalg.norm(pts - pts.mean(axis=0), axis=1).max())
                                <= GATE_RESEED_SPREAD_M):
                        seed = meas_world.copy()
                        if not trust_z:
                            seed[2] = float(position[2])
                        self._gate_world = seed
                        self._gate_world_ts = now
                        self._reseed_buf = []
                        updated = True

        # Record whether vision CONFIRMED the current world belief this tick (used to
        # decide what is safe to learn into the course map — preplan-only seeds are not,
        # and the yaw lock only re-aims on a CONFIRMED detection; see _vision_accepted).
        self._vision_accepted = bool(updated)
        if updated:
            self._gate_world_vis = True

        # 2) Fly to the filtered world waypoint (whether or not vision updated it this
        # tick — dead-reckoning via odometry keeps us closing the gap when it's out of
        # view). This is a fixed point, so steering to it is stable.
        if (self._gate_world is not None and position is not None
                and (now - self._gate_world_ts) <= GATE_MEMORY_TIMEOUT_NS):
            return (self._gate_world - np.asarray(position, float),
                    'vision' if updated else 'gate_memory')

        # 3) Preplanned course map (learned from prior DETERMINISTIC runs). When vision
        # has no fresh lock, fly to the KNOWN world position of the current gate so it
        # scrolls into the narrow up-tilted FOV — instead of coasting blind and parking
        # just past the previous gate (the gate-2 hover failure). Seeding _gate_world here
        # lets the existing yaw-lock / vertical-trust / pass-through machinery apply, and
        # lets a later vision lock EMA-refine from a good prior instead of acquiring cold.
        if self._preplan and self._course_map is not None and position is not None:
            wp = self._course_map.get(self._cur_idx)
            if wp is not None:
                wp = np.asarray(wp, float)
                if self._gate_world is None:
                    self._gate_world = wp.copy()
                    self._gate_world_ts = now
                    self._gate_world_vis = False   # a seed, NOT a vision confirmation
                return wp - np.asarray(position, float), 'preplan'

        # 4) Known track geometry from the sim broadcast (absent on this sim — kept for
        # forward-compat in case a future sim issue starts sending it).
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

    # ------------------------------------------------------------------
    # Gate-relative reactive guidance for the PnP pipeline (see compute_target).
    # ------------------------------------------------------------------
    def _pnp_reactive(self, s, now, yaw_hold, position):
        vis = s['vision']
        fresh = (vis and vis.get('detected')
                 and vis.get('confidence', 0) >= CONF_MIN
                 and vis.get('ts') and (now - vis['ts']) <= VISION_TIMEOUT_NS
                 and vis.get('gate_body'))
        if fresh:
            gb = np.asarray(vis['gate_body'], float)
            rng = float(np.linalg.norm(gb))
            az = math.atan2(float(gb[1]), float(gb[0]))
            if 0.0 < rng <= MAX_GATE_RANGE_M:
                self._rx_last_ns = now
                # Inside the commit distance the aperture fills the frame and the
                # detection whipsaws: LATCH the heading and punch straight through.
                if rng > YAW_FREEZE_DIST:
                    self._rx_heading = self._wrap(yaw_hold + az)
                self._rx_committed = rng <= YAW_FREEZE_DIST
                speed = MAX_SPEED if rng <= PASS_THROUGH_DIST \
                    else min(MAX_SPEED, max(0.6, KP_POS * 0.25 * rng))
                hdg = self._rx_heading if self._rx_heading is not None \
                    else self._wrap(yaw_hold + az)
                vn = speed * math.cos(hdg)
                ve = speed * math.sin(hdg)
                # Vertical straight from the body-frame elevation (camera tilt is
                # already folded into gate_body); aim a touch below the centre.
                vd = speed * (float(gb[2]) + GATE_AIM_DOWN_M) / max(rng, 1e-6)
                vd = max(-MAX_VSPEED, min(MAX_VSPEED, vd))
                return self._publish({'mode': 'velocity',
                                      'vel_ned': (float(vn), float(ve), float(vd)),
                                      'yaw': float(hdg), 'range_m': rng,
                                      'source': 'pnp_track', 'ts': now})

        # ---- blind ----
        last = getattr(self, '_rx_last_ns', 0)
        hdg = self._rx_heading if getattr(self, '_rx_heading', None) is not None \
            else yaw_hold
        blind_s = (now - last) / 1e9 if last else 1e9
        if blind_s < 3.0:
            # Just lost sight (usually: passing through / gate at frame edge).
            # Coast straight ahead OPEN-LOOP: gentle fixed forward lean, hover
            # thrust — do NOT close the loop on a blind velocity belief.
            return self._publish({'mode': 'velocity',
                                  'vel_ned': (1.0 * math.cos(hdg), 1.0 * math.sin(hdg), 0.1),
                                  'yaw': float(hdg), 'open_loop': True,
                                  'source': 'pnp_coast', 'ts': now})
        # Lost for good: slow level search sweep (open-loop hover + yaw steps).
        self._rx_heading = self._wrap(hdg + math.radians(0.4))  # ~24 deg/s at 60 Hz
        return self._publish({'mode': 'velocity', 'vel_ned': (0.0, 0.0, 0.0),
                              'yaw': float(self._rx_heading), 'open_loop': True,
                              'source': 'pnp_search', 'ts': now})

    def compute_target(self):
        """Compute and publish the current velocity setpoint. Returns the target dict."""
        now = time.time_ns()
        s = self._snapshot()
        position, rpy = self._pose(s)
        yaw_hold = rpy[2] if rpy else 0.0

        # When the sim advances the active gate, the remembered position belongs to the
        # gate we just passed -> drop it so we don't dead-reckon backward into it. The
        # next gate's memory is seeded as soon as vision picks it up.
        race = s.get('race')
        if race and race.get('active_gate_index', -1) >= 0:
            gidx = int(race['active_gate_index'])   # authoritative when the sim sends it
        else:
            gidx = self._auto_idx                   # self-detected fallback (see below)
        self._cur_idx = gidx
        if gidx != self._last_gate_idx:
            # LEARN: record the gate we just passed into the course map BEFORE dropping
            # the belief. Only vision-confirmed beliefs are stored, so preplan seeds never
            # feed back on themselves. Persist incrementally so a crash still keeps the map.
            if (self._learn and self._course_map is not None and self._gate_world is not None
                    and self._gate_world_vis and self._last_gate_idx not in self._recorded):
                self._course_map.record(self._last_gate_idx,
                                        tuple(float(v) for v in self._gate_world))
                self._course_map.save()
                self._recorded.add(self._last_gate_idx)
            self._gate_world = None
            self._gate_world_vis = False
            self._reseed_buf = []
            # Keep flying the heading we passed through on (don't snap to whatever garbage
            # bearing the just-passed gate gives at point-blank). The lock re-aims itself
            # once the NEXT gate is acquired off-centre at a sane range.
            self._yaw_lock = yaw_hold
            # Coast forward to bring the next gate into frame instead of braking to a hover.
            self._coast_until_ns = now + POST_GATE_COAST_NS
            self._last_gate_idx = gidx

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

        # New deterministic image-space controller. It does not require world
        # pose; yaw is only used here to express body forward/right velocities
        # in the repository's established NED target format.
        nav = s.get('navigation')
        control_source = s.get('control_source')
        if control_source == 'safe':
            return self._publish({'mode': 'velocity', 'vel_ned': (0.0, 0.0, 0.0),
                                  'yaw': yaw_hold, 'source': 'ai_safe_hover', 'ts': now})
        if control_source == 'ai':
            # Controller consumes the AI rate/thrust action directly.
            return self._publish({'mode': 'velocity', 'vel_ned': (0.0, 0.0, 0.0),
                                  'yaw': yaw_hold, 'source': 'ai_active', 'ts': now})
        if (control_source == 'opencv' and nav and nav.get('ts')
                and now - nav['ts'] <= NAVIGATION_TIMEOUT_NS):
            forward = float(nav.get('forward_mps', 0.0))
            right = float(nav.get('right_mps', 0.0))
            down = float(nav.get('down_mps', 0.0))
            c, sy = math.cos(yaw_hold), math.sin(yaw_hold)
            vn = c * forward - sy * right
            ve = sy * forward + c * right
            yaw = self._wrap(
                yaw_hold
                + float(nav.get('yaw_rate_rps', 0.0)) * YAW_RATE_TARGET_HORIZON_S
            )
            vis_range = s['vision'].get('range_m') if s.get('vision') else None
            return self._publish({
                'mode': 'velocity',
                'vel_ned': (float(vn), float(ve), float(down)),
                'yaw': float(yaw),
                'range_m': vis_range,
                'source': f"opencv_{nav.get('state', 'UNKNOWN').lower()}",
                'navigation_state': nav.get('state'),
                'ts': now,
            })
        if control_source == 'opencv':
            # No fresh image-space command yet. Do not fall through to map or
            # simulator geometry in OpenCV-only mode.
            return self._publish({'mode': 'velocity', 'vel_ned': (0.0, 0.0, 0.0),
                                  'yaw': yaw_hold, 'source': 'opencv_stale_hover',
                                  'ts': now})
        if control_source == 'pnp':
            # GATE-RELATIVE reactive guidance (2026-07-27). The world-waypoint
            # machinery below assumes a trustworthy absolute pose; runs 4-7 showed
            # that once the VIO position diverges (blind stretch -> anchor churn)
            # that machinery chases ghosts at max thrust and never recovers. PnP
            # gives the gate's BODY-frame vector directly, so while a gate is in
            # sight we steer on it reactively (yaw-servo the azimuth, speed from
            # range) — immune to absolute-pose drift. Blind phases fly OPEN-LOOP
            # (level coast / slow search), never off the drifted velocity belief.
            return self._pnp_reactive(s, now, yaw_hold, position)

        offset, source = self._target_offset_ned(s, now)
        if offset is None:
            # No gate target. If we just passed a gate, COAST straight ahead (steady camera)
            # to scroll the next gate into the FOV rather than braking to a hover and parking.
            if now < self._coast_until_ns:
                vn = POST_GATE_COAST_SPEED * math.cos(yaw_hold)
                ve = POST_GATE_COAST_SPEED * math.sin(yaw_hold)
                # Descend to bring the lower next-gate into the up-tilted camera, with a floor.
                vd = POST_GATE_DESCEND_SPEED
                if position is not None and (-float(position[2])) <= POST_GATE_MIN_ALT_M:
                    vd = 0.0
                return self._publish({'mode': 'velocity', 'vel_ned': (float(vn), float(ve), float(vd)),
                                      'yaw': yaw_hold, 'source': 'post_gate_coast', 'ts': now})
            # Blind and out of ideas: rotate slowly in place to sweep the camera
            # across the course until a gate re-enters the frame (run 6 parked in
            # a permanent zero-detection hover after losing sight). Slow enough
            # (~25 deg/s) that YOLO gets several clean frames per gate crossing.
            search_yaw = self._wrap(yaw_hold + math.radians(25.0) * 0.5)
            return self._publish({'mode': 'velocity', 'vel_ned': (0.0, 0.0, 0.0),
                                  'yaw': search_yaw, 'source': 'search_spin', 'ts': now})

        # Aim a touch BELOW the estimated gate centre so we don't clip the top edge
        # (NED +z is down). See GATE_AIM_DOWN_M.
        offset = offset.copy()
        offset[2] += GATE_AIM_DOWN_M

        dist = float(np.linalg.norm(offset))
        horiz = float(np.linalg.norm(offset[:2]))

        # Self gate-pass detection: advance our own gate counter when we close to within
        # the pass radius and then move clearly past it. Used ONLY when the sim provides
        # no active_gate_index (otherwise race is authoritative and _auto_idx is unused),
        # so learn/replay can still step gate-to-gate. The advance takes effect next tick
        # via the same reset/record path as a race-driven advance.
        if not (race and race.get('active_gate_index', -1) >= 0):
            if dist < PASS_THROUGH_DIST:
                self._armed_pass = True
            elif self._armed_pass and dist > PASS_THROUGH_DIST + 1.5:
                self._auto_idx += 1
                self._armed_pass = False
        if dist <= PASS_THROUGH_DIST:
            speed = MAX_SPEED  # commit to the pass — don't decelerate inside the gate
        else:
            speed = min(MAX_SPEED, KP_POS * dist)

        # YAW: heading LATCH with a deadband (see YAW_DEADBAND_RAD). The per-frame vision
        # azimuth is only trustworthy in straight flight; chasing it degree-by-degree spins
        # us out (az and yaw ran away together in log 2026-06-06). So we hold a latched
        # heading and only re-aim when a FRESH detection puts the gate consistently beyond
        # the deadband. Inside the band (or with no fresh detection) we hold the lock, which
        # keeps us flying straight — and the first gate is dead ahead, so that flies us
        # through it. The lock self-corrects near a gate: a small heading error grows past
        # the deadband as range shrinks, triggering a gentle re-aim.
        az = self._fresh_gate_azimuth(s, now)
        if self._yaw_lock is None:
            # First fix: aim at wherever the gate is.
            self._yaw_lock = self._wrap(yaw_hold + az) if az is not None \
                else math.atan2(float(offset[1]), float(offset[0]))
        elif (dist > YAW_FREEZE_DIST and az is not None and abs(az) > YAW_DEADBAND_RAD
                and self._vision_accepted):
            # Gate is genuinely off-centre AND we're not yet committed to the pass AND
            # the detection agreed with the world belief this tick (a frame locked onto
            # a DIFFERENT gate must not steer us) -> deliberate re-aim onto it. Inside
            # YAW_FREEZE_DIST we hold the lock and punch straight through.
            self._yaw_lock = self._wrap(yaw_hold + az)
        yaw = self._yaw_lock

        # HORIZONTAL velocity points along the CURRENT heading (fly where we look), so the
        # controller only needs PITCH (forward) and ~no ROLL. As yaw centres the gate, the
        # nose — and thus this velocity — aligns with it; the drone curves onto the gate
        # without ever strafing/banking.
        h = speed * (horiz / dist) if dist > 1e-9 else 0.0
        vn = h * math.cos(yaw_hold)
        ve = h * math.sin(yaw_hold)

        # VERTICAL: close the height gap toward the (close-range-trusted) gate height,
        # capped separately — a noisy big climb/descent is what overshot before.
        vd = 0.0
        if dist > 1e-9:
            vd = max(-MAX_VSPEED, min(MAX_VSPEED, speed * float(offset[2]) / dist))

        return self._publish({'mode': 'velocity',
                              'vel_ned': (float(vn), float(ve), float(vd)),
                              'yaw': float(yaw),
                              'range_m': dist,
                              'source': source,
                              'ts': now})
