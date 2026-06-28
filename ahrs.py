"""AHRS -- attitude (roll/pitch/yaw) from raw IMU, for VQ2.

VQ2 blocks the ``ATTITUDE`` message (spec 9.3), so we must reconstruct orientation from
``HIGHRES_IMU`` ourselves. This is a classic complementary filter:

* the GYRO (angular rate) is integrated for a smooth, low-latency attitude -- but it DRIFTS;
* the ACCELEROMETER senses the gravity direction, giving an absolute (drift-free) roll/pitch
  reference -- but it is noisy and corrupted by linear acceleration;
* the MAGNETOMETER gives an absolute yaw/heading reference -- but it is noisy.

The filter trusts the gyro over a short time constant ``TAU`` and gently pulls roll/pitch
toward the accel estimate and yaw toward the mag estimate, so it is both smooth AND drift-free.
At the VQ2 SAFE-baseline speeds (gentle, near-level) linear accel is small, so the accel ~=
gravity assumption holds well; this is why we fly slow first.

FRAME / SIGN CONVENTIONS (MAVLink NED body: X forward, Y right, Z down):
  * gyro = (p, q, r) body rates about X/Y/Z (rad/s).
  * accel = specific force (m/s^2). At rest LEVEL a quad reads ~ (0, 0, -9.81): the sensor
    measures the reaction to gravity, so the true "down" (gravity) direction in body is
    ``-accel/|accel|``. From that:  roll = atan2(-ay, -az),  pitch = atan2(ax, hypot(ay, az)).
    (Verified by construction in test_ahrs.py; CONFIRM the accel sign live -- if roll/pitch
    come out flipped or 180-off, the sim uses the opposite accel sign: flip ACC_SIGN.)
  * mag yaw is tilt-compensated; 0 = North, +East (NED). Heading sign is the most convention-
    dependent piece -- if the estimated yaw turns the wrong way vs. the gyro, flip MAG_YAW_SIGN.

Euler-based (not quaternion) for clarity and testability; valid while the drone stays away
from +/-90 deg pitch (true at VQ2 baseline). Upgrade to a quaternion/Mahony core if we later
fly high-angle. Pure numpy -- no MAVLink, no threads -- so it is unit-testable offline (we
have NO ground truth under VQ2, so the synthetic tests in test_ahrs.py are how we validate).
"""

import math

# Per-axis sensor signs (X, Y, Z), CALIBRATED 2026-06-27 against VQ2 training ground truth.
# The sim's IMU has ad-hoc per-axis sign flips (NOT a consistent frame rotation):
#   ACCEL:  X flipped, Y ok, Z ok.
#     * X: at rest nose-up +17.8deg, NED expects ax=+3.0 but the sim reports ax=-3.0 (az=-9.34 ok).
#     * Y: verified from gentle-roll samples (accel~gravity) -- roll_acc with +1 matches truth.
#   GYRO:   X (roll rate) flipped, Y ok, Z ok.
#     * raw gyroX was the EXACT NEGATIVE of the true roll rate (e.g. -0.48 vs +0.45 rad/s); our
#       AHRS integrates the gyro, so without this flip roll ran away into a tumble. gyroY matched
#       the true pitch rate, so only X. (This was the real roll-divergence cause; an earlier
#       accel-Y flip just masked it.) Z (yaw rate) unverified -- no magnetometer; harmless for
#       hover since the yaw loop holds the current estimate. Flip GYRO_SIGN[2] if yaw misbehaves.
ACC_SIGN = (-1.0, 1.0, 1.0)
GYRO_SIGN = (-1.0, 1.0, 1.0)
MAG_YAW_SIGN = 1.0    # flip to -1.0 if mag-derived heading turns the wrong way (mag absent in VQ2)
TAU = 0.5             # complementary time constant (s): higher = trust gyro longer / smoother
MIN_ACC_FRAC = 0.5    # ignore accel correction when |accel| is this far from 1g (it's not gravity)
GRAVITY = 9.80665


def _wrap(a):
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def roll_pitch_from_accel(accel):
    """Absolute roll/pitch (rad) from the gravity direction the accelerometer senses.

    ``accel`` is the raw body specific force (ax, ay, az); at rest level ~ (0, 0, -g).
    Returns ``(roll, pitch)`` or ``None`` if the sample is degenerate / not ~1g (so the
    caller skips the accel correction during a hard manoeuvre, when accel != gravity)."""
    ax, ay, az = (s * c for s, c in zip(ACC_SIGN, accel))
    if not (math.isfinite(ax) and math.isfinite(ay) and math.isfinite(az)):
        return None                       # sim sends NaN for unpopulated IMU fields
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm < 1e-6:
        return None
    roll = math.atan2(-ay, -az)
    pitch = math.atan2(ax, math.hypot(ay, az))
    return roll, pitch, norm


def yaw_from_mag(mag, roll, pitch):
    """Tilt-compensated heading (rad, 0=North, +East) from the magnetometer, or ``None`` if
    the magnetometer is absent/zero (some sims don't populate it -> caller holds gyro yaw)."""
    mx, my, mz = mag
    if not (math.isfinite(mx) and math.isfinite(my) and math.isfinite(mz)):
        return None                       # NaN/absent magnetometer -> caller holds gyro yaw
    if abs(mx) + abs(my) + abs(mz) < 1e-9:
        return None
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    # Rotate the body-frame field into the horizontal plane, then take the heading.
    xh = mx * cp + my * sr * sp + mz * cr * sp
    yh = my * cr - mz * sr
    return MAG_YAW_SIGN * math.atan2(-yh, xh)


class AHRS:
    """Complementary-filter attitude estimator. Feed it IMU samples; read roll/pitch/yaw."""

    def __init__(self, tau=TAU):
        self.tau = tau
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self._init_acc = False     # seed roll/pitch from the first good accel sample
        self._init_yaw = False     # seed yaw from the first good mag sample

    def update(self, gyro, accel, dt, mag=None):
        """Advance the estimate by ``dt`` seconds. Returns ``(roll, pitch, yaw)`` (rad).

        gyro = (p, q, r) body rates (rad/s); accel = body specific force (m/s^2);
        mag = optional body magnetic field (any units; only the direction matters).
        """
        if dt <= 0.0 or dt > 0.5:
            return self.roll, self.pitch, self.yaw    # bad/huge dt -> hold (skip on first tick)
        p, q, r = (s * g for s, g in zip(GYRO_SIGN, gyro))   # per-axis sim gyro sign correction

        # --- gyro PREDICT: integrate body rates as Euler-angle rates (full transform so a
        #     coordinated turn doesn't bleed yaw into roll/pitch).
        sr, cr = math.sin(self.roll), math.cos(self.roll)
        tp = math.tan(self.pitch)
        cp = math.cos(self.pitch)
        cp = cp if abs(cp) > 1e-4 else 1e-4          # guard near +/-90 deg pitch
        roll_dot = p + sr * tp * q + cr * tp * r
        pitch_dot = cr * q - sr * r
        yaw_dot = (sr / cp) * q + (cr / cp) * r
        roll_g = self.roll + roll_dot * dt
        pitch_g = self.pitch + pitch_dot * dt
        yaw_g = self.yaw + yaw_dot * dt

        # --- accel CORRECT (roll/pitch): blend toward the gravity-derived angles, but only
        #     when |accel| is close to 1g (otherwise we're accelerating and it isn't gravity).
        alpha = self.tau / (self.tau + dt)           # ~1 => mostly gyro; (1-alpha) pulls to accel
        rp = roll_pitch_from_accel(accel)
        if rp is not None:
            roll_a, pitch_a, norm = rp
            near_g = abs(norm - GRAVITY) <= MIN_ACC_FRAC * GRAVITY
            if not self._init_acc:                   # snap to the first sample (no slow converge)
                self.roll, self.pitch = roll_a, pitch_a
                self._init_acc = True
            elif near_g:
                self.roll = _wrap(alpha * roll_g + (1.0 - alpha) * roll_a)
                self.pitch = _wrap(alpha * pitch_g + (1.0 - alpha) * pitch_a)
            else:
                self.roll, self.pitch = _wrap(roll_g), _wrap(pitch_g)
        else:
            self.roll, self.pitch = _wrap(roll_g), _wrap(pitch_g)

        # --- mag CORRECT (yaw): blend toward tilt-compensated heading if mag is present.
        yaw_m = yaw_from_mag(mag, self.roll, self.pitch) if mag is not None else None
        if yaw_m is not None:
            if not self._init_yaw:
                self.yaw = yaw_m
                self._init_yaw = True
            else:
                # blend on the shortest angular path so the +/-pi wrap doesn't spin it
                self.yaw = _wrap(yaw_g + (1.0 - alpha) * _wrap(yaw_m - yaw_g))
        else:
            self.yaw = _wrap(yaw_g)                   # no mag -> dead-reckon yaw from gyro

        return self.roll, self.pitch, self.yaw

    def euler(self):
        return self.roll, self.pitch, self.yaw
