"""StateEstimator -- rebuild attitude + altitude from raw IMU for VQ2.

VQ2 blocks ATTITUDE / LOCAL_POSITION_NED / ODOMETRY (spec 9.3). This background thread is
our replacement: it consumes ``shared_data['imu']`` (HIGHRES_IMU) and PUBLISHES the state the
rest of the stack expects, using the SAME schema the old MAVLink messages did, so the
controller and logger are UNCHANGED:

  * ``shared_data['attitude']`` <- AHRS estimate (roll/pitch/yaw + body rates), see ahrs.py.
  * ``shared_data['position_ned']`` <- vertical state only: ``z`` (NED down, from baro) and
    ``vz`` (from an inertial-baro complementary observer). HORIZONTAL state is NOT observable
    from IMU+baro alone (double-integrating accel drifts unbounded), so ``x``/``y`` are None
    and ``vx``/``vy`` are published as 0.0. (0.0 -- not None -- so the controller's velocity
    loop still reads ``vz``; its horizontal branch just runs without feedback, which is fine
    while we HOVER / fly slow. Real horizontal velocity is a Phase-1 problem: the reactive
    gate-chaser will servo on the camera, not on a horizontal velocity loop.)

Vertical observer (NED, z/vz positive DOWN): predict with the gravity-removed vertical
acceleration from the accelerometer (rotated to world via the AHRS attitude), correct toward
barometric altitude. Baro is drift-free, accel is low-latency -> together: smooth AND no drift.

Lifecycle follows the repo convention (create_* / get_thread_for_join). Pure consumer of
shared_data -- it never touches MAVLink or another thread's internals.
"""

import math
import threading
import time

from ahrs import AHRS, ACC_SIGN, GYRO_SIGN, GRAVITY

# Vertical complementary-observer gains (1/s). L_POS pulls altitude toward baro; L_VEL pulls
# vertical velocity. Higher = trust baro more (less lag, more noise); lower = trust accel
# integration more (smoother, slower to correct drift). Tuned conservatively for slow flight.
L_POS = 1.0
L_VEL = 0.6
POLL_HZ = 400          # poll faster than the IMU so we never sit on a fresh sample
MAX_DT = 0.1           # ignore implausibly large gaps (s) -- reset dt rather than lurch


class StateEstimator:

    def __init__(self, data, l_pos=L_POS, l_vel=L_VEL, hz=POLL_HZ):
        self.data = data
        if 'lock' not in self.data:
            self.data['lock'] = threading.RLock()
        self.ahrs = AHRS()
        # Seed the AHRS yaw from the mission heading. Without a magnetometer (VQ2 sends NaN),
        # the AHRS has no absolute heading reference and defaults to 0 (North). The drone
        # actually boots facing along the course (toward the first waypoint), so compute the
        # heading from the arm point (origin) to wp0 and initialize yaw there. Without this,
        # the 180° yaw error corrupts the body→NED rotation and the drone flies backward.
        mission = data.get('mission')
        if mission and mission.waypoints:
            wp0 = mission.waypoints[0]
            n, e, _ = wp0.pos
            self.ahrs.yaw = math.atan2(e, n)  # heading from origin toward wp0
            self.ahrs._init_yaw = True         # prevent mag-snap overwrite
        self.l_pos, self.l_vel = l_pos, l_vel
        self.hz = hz
        self._last_imu_t = None      # last processed IMU time_usec (dedupe + dt source)
        self._baro_ref = None        # pressure_alt at start -> NED origin is the arm point
        self._z = 0.0                # NED down position (m), +down
        self._vz = 0.0               # NED down velocity (m/s), +down
        # Dead-reckoned horizontal state (NED North/East). Drifts without an absolute
        # reference, but gives the spline planner a usable position estimate so it can
        # navigate the mission instead of falling into watchdog_hover.
        self._x = 0.0                # NED North position (m)
        self._y = 0.0                # NED East position (m)
        self._vx = 0.0               # NED North velocity (m/s)
        self._vy = 0.0               # NED East velocity (m/s)
        self.thread = None
        self.is_running = False

    @classmethod
    def create_state_estimator(cls, data, **kwargs):
        se = cls(data, **kwargs)
        se.thread = threading.Thread(target=se._loop, daemon=False)
        se.is_running = True
        se.thread.start()
        print("State estimator running (IMU->attitude, baro->altitude, dead-reckoned horizontal)", flush=True)
        return se

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    def _loop(self):
        dt = 1.0 / self.hz
        while self.is_running:
            try:
                self.step()
            except Exception:
                pass                 # keep the estimator alive regardless of a bad sample
            time.sleep(dt)

    # -- vertical inertial-baro observer ------------------------------------------------
    def _vertical(self, accel, roll, pitch, pressure_alt, dt):
        """Advance (z, vz) NED-down using accel (predict) + baro (correct)."""
        # World-down component of TRUE acceleration: a_down = (R @ f_body)[2] + g, where
        # f_body is the specific force (rest-level convention (0,0,-g) via ACC_SIGN) and the
        # third row of the ZYX body->earth matrix is [-sin p, cos p sin r, cos p cos r].
        fx, fy, fz = (s * c for s, c in zip(ACC_SIGN, accel))
        sp, cp = math.sin(pitch), math.cos(pitch)
        sr, cr = math.sin(roll), math.cos(roll)
        a_down = (-sp * fx + cp * sr * fy + cp * cr * fz) + GRAVITY

        if pressure_alt is not None:
            if self._baro_ref is None:               # first baro -> arm point = NED origin
                self._baro_ref = pressure_alt
            z_baro = self._baro_ref - pressure_alt   # +down: climbing (alt up) -> z more negative
            e = z_baro - self._z
            self._z += (self._vz + self.l_pos * e) * dt
            self._vz += (a_down + self.l_vel * e) * dt
        else:                                        # no baro: dead-reckon vertical (drifts)
            self._z += self._vz * dt
            self._vz += a_down * dt

    # -- horizontal dead-reckoning (no absolute reference, drifts over time) ------------
    def _horizontal(self, accel, roll, pitch, yaw, dt):
        """Advance (x, y, vx, vy) NED-North/East by integrating gravity-removed accel."""
        fx, fy, fz = (s * c for s, c in zip(ACC_SIGN, accel))
        sp, cp = math.sin(pitch), math.cos(pitch)
        sr, cr = math.sin(roll), math.cos(roll)
        sy, cy = math.sin(yaw), math.cos(yaw)
        # Full ZYX body->earth rotation, North and East rows:
        #   a_north = (cy*cp)*fx + (cy*sp*sr - sy*cr)*fy + (cy*sp*cr + sy*sr)*fz
        #   a_east  = (sy*cp)*fx + (sy*sp*sr + cy*cr)*fy + (sy*sp*cr - cy*sr)*fz
        # Gravity only affects the Down component, so no +g term here.
        a_north = (cy*cp)*fx + (cy*sp*sr - sy*cr)*fy + (cy*sp*cr + sy*sr)*fz
        a_east  = (sy*cp)*fx + (sy*sp*sr + cy*cr)*fy + (sy*sp*cr - cy*sr)*fz
        self._vx += a_north * dt
        self._vy += a_east * dt
        self._x += self._vx * dt
        self._y += self._vy * dt

    # -- one estimator tick -------------------------------------------------------------
    def step(self):
        """Process the latest IMU sample (if new) and publish attitude + vertical state."""
        with self.data['lock']:
            imu = self.data.get('imu')
        if not imu:
            return

        t_usec = imu.get('time_usec')
        # dt from the sim's IMU clock when available (most accurate); else wall clock.
        if t_usec is not None and self._last_imu_t is not None:
            if t_usec == self._last_imu_t:
                return                               # no new sample since last tick
            dt = (t_usec - self._last_imu_t) * 1e-6
        elif self._last_imu_t is None:
            self._last_imu_t = t_usec if t_usec is not None else -1
            return                                   # need a previous sample to form dt
        else:
            dt = 1.0 / self.hz
        self._last_imu_t = t_usec
        if dt <= 0.0 or dt > MAX_DT:
            return                                   # bad/huge gap -> skip (don't lurch)

        gyro = imu.get('gyro', (0.0, 0.0, 0.0))
        accel = imu.get('acc', (0.0, 0.0, 0.0))
        mag = imu.get('mag')
        palt = imu.get('pressure_alt')
        # Sanitize: the sim sends NaN for unpopulated HIGHRES_IMU fields (the magnetometer in
        # particular). Non-finite gyro/accel would poison the whole filter -> skip the sample;
        # non-finite mag/baro -> treat as ABSENT (yaw dead-reckons on gyro; vertical on accel).
        if not all(math.isfinite(v) for v in (tuple(gyro) + tuple(accel))):
            return
        if mag is not None and not all(math.isfinite(v) for v in mag):
            mag = None
        if palt is not None and not math.isfinite(palt):
            palt = None

        roll, pitch, yaw = self.ahrs.update(gyro, accel, dt, mag=mag)
        if not (math.isfinite(roll) and math.isfinite(pitch) and math.isfinite(yaw)):
            return                                   # never publish a non-finite attitude
        self._vertical(accel, roll, pitch, palt, dt)
        self._horizontal(accel, roll, pitch, yaw, dt)

        now = time.time_ns()
        with self.data['lock']:
            self.data['attitude'] = {
                'roll': roll, 'pitch': pitch, 'yaw': yaw,
                'rollspeed': GYRO_SIGN[0] * gyro[0], 'pitchspeed': GYRO_SIGN[1] * gyro[1],
                'yawspeed': GYRO_SIGN[2] * gyro[2],
                'ts': now, 'estimated': True,
            }
            self.data['position_ned'] = {
                'x': self._x, 'y': self._y, 'z': self._z,
                'vx': self._vx, 'vy': self._vy, 'vz': self._vz,
                'ts': now, 'estimated': True,
            }

