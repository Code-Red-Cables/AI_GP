"""Offline synthetic tests for the AHRS attitude estimator (no sim, no ground truth needed).

Under VQ2 we have no ATTITUDE message to check against, so we validate the filter against
SYNTHETIC IMU data we generate from a KNOWN orientation: build the gravity/mag vectors a
perfect IMU would report at a given (roll, pitch, yaw), feed them in, and assert the filter
recovers that orientation. This pins the frame/sign conventions before we ever touch the sim.

Run:  python test_ahrs.py   (needs numpy)
"""

import math
import numpy as np

from ahrs import AHRS, GRAVITY, ACC_SIGN, roll_pitch_from_accel, yaw_from_mag


def _R_body_to_earth(roll, pitch, yaw):
    """Aerospace ZYX rotation (body->earth), NED."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def _imu_at_rest(roll, pitch, yaw, mag_earth=(1.0, 0.0, 0.0)):
    """The (accel, mag) a perfect IMU reports while held STILL at this orientation.

    Specific force at rest = -gravity (NED gravity = +Z, down), expressed in BODY frame.
    Magnetic field (earth, default = North) likewise rotated into the body frame.
    """
    R = _R_body_to_earth(roll, pitch, yaw)
    g_earth = np.array([0.0, 0.0, GRAVITY])           # gravity points DOWN (+Z) in NED
    accel_body = R.T @ (-g_earth)                      # specific force at rest, body frame (NED)
    # Emit it in the SIM's accelerometer convention (per-axis ACC_SIGN), so the filter -- which
    # multiplies by ACC_SIGN -- recovers the true orientation. Keeps tests valid for any ACC_SIGN.
    accel_sim = tuple(s * c for s, c in zip(ACC_SIGN, accel_body))
    mag_body = R.T @ np.array(mag_earth, float)
    return accel_sim, tuple(mag_body)


def _deg(x):
    return math.degrees(x)


def test_accel_formula_signs():
    """The raw accel->roll/pitch formula matches the documented NED convention."""
    for roll_d, pitch_d in [(0, 0), (30, 0), (-25, 0), (0, 20), (0, -15), (15, 10)]:
        roll, pitch = math.radians(roll_d), math.radians(pitch_d)
        accel, _ = _imu_at_rest(roll, pitch, 0.0)
        r, p, norm = roll_pitch_from_accel(accel)
        assert abs(_deg(r) - roll_d) < 0.5, f"roll {roll_d}: got {_deg(r):.1f}"
        assert abs(_deg(p) - pitch_d) < 0.5, f"pitch {pitch_d}: got {_deg(p):.1f}"
        assert abs(norm - GRAVITY) < 1e-3
    print("PASS accel signs       roll/pitch recovered from gravity at 6 orientations")


def test_mag_yaw_signs():
    """Tilt-compensated mag heading recovers yaw (level and tilted)."""
    for yaw_d in [0, 45, 90, -90, 179, -135]:
        yaw = math.radians(yaw_d)
        _, mag = _imu_at_rest(0.0, 0.0, yaw)
        ym = yaw_from_mag(mag, 0.0, 0.0)
        assert abs(math.degrees(math.atan2(math.sin(ym - yaw), math.cos(ym - yaw)))) < 0.5, \
            f"yaw {yaw_d}: got {_deg(ym):.1f}"
    # tilted: heading must still come out right with roll/pitch applied
    yaw = math.radians(60)
    accel, mag = _imu_at_rest(math.radians(20), math.radians(-15), yaw)
    ym = yaw_from_mag(mag, math.radians(20), math.radians(-15))
    assert abs(math.degrees(math.atan2(math.sin(ym - yaw), math.cos(ym - yaw)))) < 0.5
    print("PASS mag yaw signs     heading recovered level + tilted")


def test_static_orientation_converges():
    """Held still, the filter snaps to the true roll/pitch/yaw."""
    for roll_d, pitch_d, yaw_d in [(0, 0, 0), (30, 0, 90), (-20, 15, -45), (10, -25, 170)]:
        roll, pitch, yaw = map(math.radians, (roll_d, pitch_d, yaw_d))
        accel, mag = _imu_at_rest(roll, pitch, yaw)
        f = AHRS()
        for _ in range(50):
            f.update((0, 0, 0), accel, 0.01, mag=mag)
        er, ep, ey = f.euler()
        assert abs(_deg(er) - roll_d) < 1.0, f"roll {roll_d}: {_deg(er):.1f}"
        assert abs(_deg(ep) - pitch_d) < 1.0, f"pitch {pitch_d}: {_deg(ep):.1f}"
        dy = math.degrees(math.atan2(math.sin(ey - yaw), math.cos(ey - yaw)))
        assert abs(dy) < 1.0, f"yaw {yaw_d}: {_deg(ey):.1f}"
    print("PASS static converge   filter locks onto 4 known static orientations")


def test_gyro_yaw_dead_reckons():
    """With no magnetometer, yaw integrates the gyro (relative heading)."""
    f = AHRS()
    accel, _ = _imu_at_rest(0.0, 0.0, 0.0)            # stays level (yaw doesn't tilt gravity)
    rate = 0.5                                         # rad/s yaw
    for _ in range(200):                               # 2.0 s
        f.update((0.0, 0.0, rate), accel, 0.01, mag=None)
    er, ep, ey = f.euler()
    assert abs(ey - 1.0) < 0.05, f"yaw should be ~1.0 rad, got {ey:.3f}"
    assert abs(_deg(er)) < 1.0 and abs(_deg(ep)) < 1.0, "roll/pitch must stay level"
    print(f"PASS gyro dead-reckon  yaw integrated to {ey:.3f} rad (expected 1.0)")


def test_accel_rejects_gyro_drift():
    """A constant gyro bias must NOT make roll run away -- accel pulls it back (bounded)."""
    f = AHRS()
    accel, mag = _imu_at_rest(0.0, 0.0, 0.0)
    bias = 0.1                                         # rad/s phantom roll rate
    for _ in range(500):                               # 5 s -- uncorrected this would be 0.5 rad
        f.update((bias, 0.0, 0.0), accel, 0.01, mag=mag)
    er, _, _ = f.euler()
    assert abs(er) < math.radians(5.0), f"roll drifted to {_deg(er):.1f} deg (accel not correcting)"
    print(f"PASS drift rejection   roll bounded to {_deg(er):.2f} deg despite gyro bias")


if __name__ == "__main__":
    test_accel_formula_signs()
    test_mag_yaw_signs()
    test_static_orientation_converges()
    test_gyro_yaw_dead_reckons()
    test_accel_rejects_gyro_drift()
    print("ALL AHRS TESTS PASSED")
