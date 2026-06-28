"""Offline test for StateEstimator (no sim). Feeds synthetic HIGHRES_IMU through shared_data
and checks it publishes attitude (from AHRS) and vertical state (from the baro observer) in
the schema the controller expects. No ground truth exists under VQ2, so -- as with the AHRS --
we generate the IMU from a KNOWN orientation / altitude profile and assert recovery.

Run:  python test_state_estimator.py   (needs numpy)
"""

import math
import threading

import numpy as np

from state_estimator import StateEstimator
from ahrs import GRAVITY, ACC_SIGN


def _R(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def _imu_rest(roll, pitch, yaw):
    """accel + mag a still IMU reports at this orientation (NED)."""
    R = _R(roll, pitch, yaw)
    accel_ned = R.T @ np.array([0.0, 0.0, -GRAVITY])        # specific force at rest (NED)
    accel = tuple(s * c for s, c in zip(ACC_SIGN, accel_ned))   # emit in sim's accel convention
    mag = tuple(R.T @ np.array([1.0, 0.0, 0.0]))            # earth field = North
    return accel, mag


def _feed(se, data, *, accel, mag, gyro, pressure_alt, t_usec):
    with data['lock']:
        data['imu'] = {'acc': accel, 'gyro': gyro, 'mag': mag,
                       'pressure_alt': pressure_alt, 'time_usec': t_usec,
                       'ts': t_usec * 1000}
    se.step()


def test_publishes_attitude_schema():
    """Static tilt -> publishes attitude with the right keys and roll/pitch/yaw values."""
    data = {'lock': threading.RLock()}
    se = StateEstimator(data)
    roll, pitch, yaw = math.radians(15), math.radians(-10), math.radians(40)
    accel, mag = _imu_rest(roll, pitch, yaw)
    t = 1_000_000
    for _ in range(60):
        _feed(se, data, accel=accel, mag=mag, gyro=(0, 0, 0),
              pressure_alt=100.0, t_usec=t)
        t += 5000                                          # 5 ms -> 200 Hz
    att = data['attitude']
    for k in ('roll', 'pitch', 'yaw', 'rollspeed', 'pitchspeed', 'yawspeed', 'ts'):
        assert k in att, f"attitude missing key {k}"
    assert abs(math.degrees(att['roll']) - 15) < 1.0, att['roll']
    assert abs(math.degrees(att['pitch']) + 10) < 1.0, att['pitch']
    dy = math.degrees(math.atan2(math.sin(att['yaw'] - yaw), math.cos(att['yaw'] - yaw)))
    assert abs(dy) < 1.0, att['yaw']
    assert att.get('estimated') is True
    print("PASS attitude publish  roll/pitch/yaw recovered, schema intact")


def test_position_schema_horizontal_unknown():
    """position_ned has z/vz but marks horizontal unknown (x/y None, vx/vy 0.0)."""
    data = {'lock': threading.RLock()}
    se = StateEstimator(data)
    accel, mag = _imu_rest(0, 0, 0)
    t = 1_000_000
    for _ in range(10):
        _feed(se, data, accel=accel, mag=mag, gyro=(0, 0, 0), pressure_alt=100.0, t_usec=t)
        t += 5000
    pos = data['position_ned']
    assert pos['x'] is None and pos['y'] is None, "horizontal position must be unknown"
    assert pos['vx'] == 0.0 and pos['vy'] == 0.0, "horizontal vel published as 0.0 (reads vz)"
    assert pos['z'] is not None and pos['vz'] is not None
    print("PASS position schema   z/vz present, horizontal marked unknown")


def test_baro_altitude_and_climb_rate():
    """A steady baro climb -> estimated z tracks it and vz converges to the climb rate."""
    data = {'lock': threading.RLock()}
    se = StateEstimator(data)
    accel, mag = _imu_rest(0, 0, 0)                         # level, constant-velocity climb
    climb = 0.5                                             # m/s upward (altitude increasing)
    palt0 = 200.0
    dt = 0.005
    t = 1_000_000
    n = 800                                                 # 4 s
    for i in range(n):
        palt = palt0 + climb * (i * dt)                    # altitude rising
        _feed(se, data, accel=accel, mag=mag, gyro=(0, 0, 0), pressure_alt=palt, t_usec=t)
        t += int(dt * 1e6)
    pos = data['position_ned']
    z_baro = (palt0) - (palt0 + climb * ((n - 1) * dt))    # ref - last_alt  (NED down, +down)
    assert abs(pos['z'] - z_baro) < 0.15, f"z {pos['z']:.3f} vs baro {z_baro:.3f}"
    assert abs(pos['vz'] - (-climb)) < 0.1, f"vz {pos['vz']:.3f} should be ~{-climb}"
    print(f"PASS baro vertical     z={pos['z']:.2f} m (baro {z_baro:.2f}), "
          f"vz={pos['vz']:.2f} m/s (climb {-climb})")


def test_no_new_sample_is_noop():
    """Re-feeding the SAME time_usec must not advance the estimate (dedupe)."""
    data = {'lock': threading.RLock()}
    se = StateEstimator(data)
    accel, mag = _imu_rest(0, 0, 0)
    _feed(se, data, accel=accel, mag=mag, gyro=(0, 0, 0), pressure_alt=100.0, t_usec=5000)
    _feed(se, data, accel=accel, mag=mag, gyro=(0, 0, 0), pressure_alt=100.0, t_usec=10000)
    z1 = data['position_ned']['z']
    for _ in range(5):                                      # same t_usec -> ignored
        _feed(se, data, accel=accel, mag=mag, gyro=(0, 5.0, 0), pressure_alt=100.0, t_usec=10000)
    assert data['position_ned']['z'] == z1, "stale sample changed the estimate"
    print("PASS sample dedupe      repeated time_usec is a no-op")


def test_nan_imu_fields_never_poison_estimate():
    """The sim sends NaN for unpopulated HIGHRES_IMU fields (esp. the magnetometer). The
    estimate must stay FINITE -- a NaN reaching the controller tumbles the sim (the bug we hit:
    yaw=nan from t=0). Yaw should dead-reckon on gyro when mag is NaN."""
    nan = float('nan')
    data = {'lock': threading.RLock()}
    se = StateEstimator(data)
    accel, _ = _imu_rest(0, 0, 0)
    t = 1_000_000
    for i in range(100):
        # NaN magnetometer the whole time; a small yaw rate so we can see gyro dead-reckoning.
        _feed(se, data, accel=accel, mag=(nan, nan, nan), gyro=(0.0, 0.0, 0.2),
              pressure_alt=150.0, t_usec=t)
        t += 5000
    att = data['attitude']
    pos = data['position_ned']
    for k in ('roll', 'pitch', 'yaw'):
        assert math.isfinite(att[k]), f"attitude {k} is non-finite ({att[k]})"
    assert math.isfinite(pos['z']) and math.isfinite(pos['vz']), "vertical state non-finite"
    assert att['yaw'] > 0.05, "yaw should dead-reckon from gyro when mag is NaN"
    # A burst of NaN accel must be skipped, not published.
    z_before = pos['z']
    _feed(se, data, accel=(nan, nan, nan), mag=(nan, nan, nan), gyro=(nan, 0, 0),
          pressure_alt=150.0, t_usec=t)
    assert data['position_ned']['z'] == z_before, "NaN accel sample should be skipped"
    print("PASS NaN robustness    NaN mag/accel never produce a non-finite estimate")


if __name__ == "__main__":
    test_publishes_attitude_schema()
    test_position_schema_horizontal_unknown()
    test_baro_altitude_and_climb_rate()
    test_no_new_sample_is_noop()
    test_nan_imu_fields_never_poison_estimate()
    print("ALL STATE ESTIMATOR TESTS PASSED")
