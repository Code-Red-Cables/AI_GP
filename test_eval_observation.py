"""The observation hard gate must pass a good detector and fail a broken one."""
from __future__ import annotations

import math
import unittest

import camera_model as cm
from gate_bearing import (
    project_gate_centre_px,
    rotation_flow_px,
    true_gate_body,
)
from tools.eval_observation import (
    attitude_source,
    check_centring,
    check_coupling,
    check_identity,
    check_rigidity,
    check_truth,
)


_FRAME = [0]


def _row(t, u, v, *, gate=0, gx=0.0, gy=0.0, gz=0.0, roll_cmd=0.0,
         area=5000.0, frame=None, roll_angle=0.0):
    if frame is None:
        _FRAME[0] += 1
        frame = _FRAME[0]
    row = {
        't': f'{t:.3f}',
        'gate_frame_id': str(frame),
        'vision_sim_time_ns': str(int(t * 1e9)),
        'active_gate': str(gate),
        'gx_imu': str(gx), 'gy_imu': str(gy), 'gz_imu': str(gz),
        'cmd_roll_rate': str(roll_cmd),
        'cmd_yaw_rate': '0.0',
        'gate_area': str(area),
        'att_raw_roll': str(roll_angle), 'att_raw_pitch': '0.0',
    }
    for k in range(8):
        row[f'kp{k}_u'] = f'{u:.3f}'
        row[f'kp{k}_v'] = f'{v:.3f}'
        row[f'kp{k}_c'] = '1.0'
    return row


class RigidityTests(unittest.TestCase):
    def _rigid_track(self, n=90, dt=0.03):
        """A centroid moving exactly as the gyro says a fixed point must."""
        u, v = 300.0, 170.0
        t = 0.0
        rows = [_row(t, u, v)]
        for i in range(n):
            gz = 0.6 * math.sin(i * 0.3)
            gy = 0.2 * math.cos(i * 0.2)
            flow = rotation_flow_px(u, v, 0.0, gy, gz, dt)
            u += flow[0]
            v += flow[1]
            t += dt
            rows.append(_row(t, u, v, gy=gy, gz=gz))
        return rows

    def test_rigid_tracking_scores_high(self):
        stats = check_rigidity(self._rigid_track())
        self.assertGreaterEqual(stats['n'], 30)
        self.assertGreater(stats['r2_u'], 0.95)
        # Coefficients recover the assumed convention: unit scale, no flips.
        self.assertAlmostEqual(stats['coef_u'][2], 1.0, delta=0.15)

    def test_identity_hopping_detector_scores_low(self):
        """A centroid that jumps between gates must not look rigid."""
        jumps = [137, 402, 210, 355, 180, 460, 250, 300]
        rows = []
        t = 0.0
        for i in range(90):
            gz = 0.6 * math.sin(i * 0.3)
            rows.append(_row(t, float(jumps[i % len(jumps)]), 180.0, gz=gz))
            t += 0.03
        stats = check_rigidity(rows)
        self.assertLess(stats['r2_u'], 0.3)

    def test_survives_inverted_gyro_axis(self):
        """A sign-flipped gyro axis must be fitted, not reported as failure."""
        rows = self._rigid_track()
        for row in rows:
            row['gz_imu'] = str(-float(row['gz_imu']))
        stats = check_rigidity(rows)
        self.assertGreater(stats['r2_u'], 0.95)
        self.assertLess(stats['coef_u'][2], 0.0)

    def test_relogged_frames_are_skipped(self):
        """Same camera frame twice would pair zero motion with nonzero flow."""
        rows = [
            _row(0.00, 300.0, 180.0, frame=7, gz=0.5),
            _row(0.03, 300.0, 180.0, frame=7, gz=0.5),
        ]
        stats = check_rigidity(rows)
        self.assertEqual(stats['n'], 0)
        self.assertEqual(stats['duplicate_frames'], 1)

    def test_large_rotation_steps_are_excluded(self):
        """The linear flow model is invalid across a 30 degree step."""
        rows = [
            _row(0.00, 300.0, 180.0, gz=10.0),
            _row(0.10, 250.0, 180.0, gz=10.0),
        ]
        stats = check_rigidity(rows)
        self.assertEqual(stats['n'], 0)
        self.assertEqual(stats['large_rotation'], 1)

    def test_gate_change_pairs_are_skipped(self):
        """A legitimate gate swap teleports the centroid; ignore that pair."""
        rows = [
            _row(0.00, 300.0, 180.0, gate=0),
            _row(0.03, 100.0, 180.0, gate=1),
        ]
        self.assertEqual(check_rigidity(rows)['n'], 0)

    def test_frame_gap_pairs_are_skipped(self):
        rows = [
            _row(0.0, 300.0, 180.0),
            _row(9.0, 305.0, 180.0),
        ]
        self.assertEqual(check_rigidity(rows)['n'], 0)


class IdentityTests(unittest.TestCase):
    def test_centred_gate_at_pass_is_accepted(self):
        rows = [
            _row(0.0, cm.CX + 5.0, 180.0, gate=0, area=40000.0),
            _row(0.1, cm.CX + 2.0, 180.0, gate=0, area=60000.0),
            _row(0.2, cm.CX + 1.0, 180.0, gate=1, area=60000.0),
        ]
        stats = check_identity(rows)
        self.assertEqual(stats['n_passes'], 1)
        self.assertEqual(stats['n_near'], 1)
        self.assertLess(stats['median_u_offset_px'], 20.0)

    def test_offset_near_gate_at_pass_is_flagged(self):
        rows = [
            _row(0.0, 10.0, 180.0, gate=0, area=50000.0),
            _row(0.1, 12.0, 180.0, gate=0, area=52000.0),
            _row(0.2, 14.0, 180.0, gate=1, area=52000.0),
        ]
        stats = check_identity(rows)
        self.assertEqual(stats['n_near'], 1)
        self.assertGreater(stats['median_u_offset_px'], 140.0)

    def test_distant_next_gate_is_excluded_not_failed(self):
        """A small off-centre gate at the pass is the *next* gate, not a fault."""
        rows = [
            _row(0.0, 10.0, 180.0, gate=0, area=3000.0),
            _row(0.1, 12.0, 180.0, gate=0, area=3600.0),
            _row(0.2, 14.0, 180.0, gate=1, area=3600.0),
        ]
        stats = check_identity(rows)
        self.assertEqual(stats['n_passes'], 1)
        self.assertEqual(stats['n_with_vision'], 1)
        self.assertEqual(stats['n_near'], 0)
        self.assertTrue(math.isnan(stats['median_u_offset_px']))

    def test_no_pass_reports_zero(self):
        rows = [_row(i * 0.1, 300.0, 180.0, gate=0) for i in range(5)]
        self.assertEqual(check_identity(rows)['n_passes'], 0)


class CouplingTests(unittest.TestCase):
    def test_pilot_correcting_bearing_correlates(self):
        rows = []
        for i in range(60):
            u = cm.CX + 120.0 * math.sin(i * 0.25)
            # Pilot rolls proportionally to the bearing error, 2 frames later.
            cmd = 0.9 * math.sin(max(0, i - 2) * 0.25)
            rows.append(_row(i * 0.05, u, 180.0, roll_cmd=cmd))
        stats = check_coupling(rows)
        self.assertGreater(abs(stats['cmd_roll_rate']['r']), 0.8)
        self.assertGreater(stats['corr_best'], 0.8)

    def test_bank_angle_channel_is_checked(self):
        """In acro the stick is a rate; bank angle is what turns the drone."""
        rows = []
        for i in range(60):
            u = cm.CX + 120.0 * math.sin(i * 0.25)
            rows.append(_row(
                i * 0.05, u, 180.0,
                roll_cmd=0.0,
                roll_angle=0.5 * math.sin(i * 0.25),
            ))
        stats = check_coupling(rows)
        self.assertGreater(abs(stats['roll_angle']['r']), 0.9)
        self.assertEqual(stats['best_channel'], 'roll_angle')

    def test_unrelated_command_does_not_correlate(self):
        rows = []
        for i in range(80):
            u = cm.CX + 120.0 * math.sin(i * 0.25)
            cmd = 0.9 * math.sin(i * 1.97 + 1.1)
            rows.append(_row(i * 0.05, u, 180.0, roll_cmd=cmd))
        stats = check_coupling(rows)
        self.assertLess(abs(stats['cmd_roll_rate']['r']), 0.6)


class CentringTests(unittest.TestCase):
    def test_centred_tracking_passes_even_with_zero_correlation(self):
        """A pilot who nulls the error leaves correlation nothing to measure."""
        rows = [
            _row(i * 0.03, cm.CX + (1.0 if i % 2 else -1.0), 180.0)
            for i in range(80)
        ]
        centre = check_centring(rows)
        self.assertLess(centre['median_bearing_deg'], 1.0)
        self.assertGreater(centre['frac_within_60px'], 0.95)
        # The same data has no usable correlation signal at all.
        self.assertLess(abs(check_coupling(rows)['corr_best']), 0.25)

    def test_off_centre_tracking_is_measured(self):
        """A detector locked on another gate sits far from centre."""
        rows = [_row(i * 0.03, 40.0, 180.0) for i in range(80)]
        centre = check_centring(rows)
        self.assertGreater(centre['median_bearing_deg'], 8.0)
        self.assertLess(centre['frac_within_60px'], 0.05)

    def test_identity_hopping_shows_large_jumps(self):
        jumps = [80.0, 500.0] * 40
        rows = [_row(i * 0.03, jumps[i], 180.0) for i in range(80)]
        centre = check_centring(rows)
        self.assertGreater(centre['frac_jump_over_80px'], 0.9)

    def test_relogged_frames_do_not_inflate_counts(self):
        rows = [
            _row(0.00, 100.0, 180.0, frame=5),
            _row(0.03, 100.0, 180.0, frame=5),
            _row(0.06, 100.0, 180.0, frame=6),
        ]
        self.assertEqual(check_centring(rows)['n'], 2)


class TruthTests(unittest.TestCase):
    def test_perfect_detector_passes_odometry_check(self):
        course = [{'id': 0, 'pos': (12.0, 0.0, -1.0)}]
        rows = []
        for y in (-2.0, -1.0, 0.0, 1.0, 2.0) * 10:
            odo = {
                'x': 0.0, 'y': y, 'z': -1.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            }
            body = true_gate_body(odo, course[0])
            u, v = project_gate_centre_px(body)
            row = _row(0.0, u, v)
            row.update({
                'odo_x': '0.0', 'odo_y': str(y), 'odo_z': '-1.0',
                'odo_roll': '0.0', 'odo_pitch': '0.0', 'odo_yaw': '0.0',
            })
            rows.append(row)
        stats = check_truth(rows, course)
        self.assertGreaterEqual(stats['n'], 30)
        self.assertGreater(stats['corr_bearing'], 0.95)

    def test_missing_odometry_yields_no_samples(self):
        rows = [_row(0.0, 300.0, 180.0)]
        self.assertEqual(check_truth(rows, [{'id': 0, 'pos': (1.0, 0, 0)}])['n'], 0)


class AttitudeSourceTests(unittest.TestCase):
    ROWS = [
        {'att_raw_roll': '0.1', 'att_raw_pitch': '0.0',
         'ahrs_roll': '0.2', 'ahrs_pitch': '0.0', 'roll': '0.3', 'pitch': '0.0'},
        {'att_raw_roll': 'nan', 'att_raw_pitch': 'nan',
         'ahrs_roll': '0.2', 'ahrs_pitch': '0.0', 'roll': '0.3', 'pitch': '0.0'},
        {'att_raw_roll': 'nan', 'att_raw_pitch': 'nan',
         'ahrs_roll': 'nan', 'ahrs_pitch': 'nan', 'roll': '0.3', 'pitch': '0.0'},
        {'att_raw_roll': 'nan', 'att_raw_pitch': 'nan',
         'ahrs_roll': 'nan', 'ahrs_pitch': 'nan', 'roll': 'nan', 'pitch': 'nan'},
    ]

    def test_prefers_raw_then_ahrs_then_ekf(self):
        counts = attitude_source(self.ROWS)
        self.assertEqual(counts['att_raw'], 1)
        self.assertEqual(counts['ahrs'], 1)
        self.assertEqual(counts['ekf'], 1)
        self.assertEqual(counts['none'], 1)

    def test_only_gravity_referenced_rows_are_trusted(self):
        from race_obs import attitude_is_trusted

        self.assertEqual(
            [attitude_is_trusted(r) for r in self.ROWS],
            [True, True, False, False],
        )

    def test_untrusted_rows_are_excluded_from_training(self):
        """An EKF-attitude row must not become a training window."""
        import csv
        import tempfile
        from pathlib import Path

        from tools.train_policy import load_run

        fields = [
            't', 'control_authority', 'cmd_thrust', 'cmd_roll_rate',
            'cmd_pitch_rate', 'cmd_yaw_rate', 'ahrs_roll', 'ahrs_pitch',
            'roll', 'pitch', 'gx_imu', 'gy_imu', 'gz_imu',
        ] + [f'kp{i}_{a}' for i in range(8) for a in ('u', 'v', 'c')]
        tmp = tempfile.NamedTemporaryFile(
            'w', suffix='.csv', delete=False, newline=''
        )
        writer = csv.DictWriter(tmp, fieldnames=fields)
        writer.writeheader()
        for i in range(4):
            row = {k: 'nan' for k in fields}
            row.update({
                't': f'{i * 0.1:.1f}', 'control_authority': 'human',
                'cmd_thrust': '0.3', 'cmd_roll_rate': '0.0',
                'cmd_pitch_rate': '0.0', 'cmd_yaw_rate': '0.0',
                'gx_imu': '0', 'gy_imu': '0', 'gz_imu': '0',
                'roll': '0.1', 'pitch': '0.1',
            })
            # Only the first two rows carry a trusted AHRS attitude.
            if i < 2:
                row['ahrs_roll'] = '0.05'
                row['ahrs_pitch'] = '0.05'
            for k in range(8):
                row[f'kp{k}_u'] = '100'
                row[f'kp{k}_v'] = '100'
                row[f'kp{k}_c'] = '1'
            writer.writerow(row)
        tmp.close()

        _obs, _lab, _w, valid, _marked, _gates, _att, _dt = load_run(
            Path(tmp.name), lead_s=0.0, tail_s=0.0, sort_by_u=False
        )
        self.assertEqual(list(valid), [True, True, False, False])


if __name__ == '__main__':
    unittest.main()
