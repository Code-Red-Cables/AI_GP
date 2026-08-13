"""Unit tests for HG-DAgger aggregation / intervention windowing."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.train_policy import collision_times, gate_balance_weights, load_run


def _write(rows):
    tmp = tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False, newline='')
    path = Path(tmp.name)
    fields = [
        't', 'control_authority', 'active_gate', 'attempt', 'exclude',
        'cmd_thrust', 'cmd_roll_rate', 'cmd_pitch_rate', 'cmd_yaw_rate',
        'ahrs_roll', 'ahrs_pitch',
        'roll', 'pitch', 'gx_imu', 'gy_imu', 'gz_imu',
    ] + [f'kp{i}_{a}' for i in range(8) for a in ('u', 'v', 'c')]
    writer = csv.DictWriter(tmp, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        full = {k: 'nan' for k in fields}
        # A visible gate and a trusted attitude by default, so a row is usable
        # unless a test deliberately breaks it.
        full.update({'ahrs_roll': '0.0', 'ahrs_pitch': '0.0'})
        for i in range(8):
            full[f'kp{i}_u'] = '100'
            full[f'kp{i}_v'] = '100'
            full[f'kp{i}_c'] = '1'
        full.update(row)
        writer.writerow(full)
    tmp.close()
    return path


class TrainPolicyTests(unittest.TestCase):
    def test_backdates_and_tails_intervention(self):
        # 0.1 s steps. Human only at t=1.0; lead=0.5 and tail=0.5 should
        # mark [0.5, 1.5].
        rows = []
        for i in range(21):
            t = i * 0.1
            auth = 'human' if abs(t - 1.0) < 1e-9 else 'policy'
            rows.append({
                't': f'{t:.1f}',
                'control_authority': auth,
                'cmd_thrust': '0.3',
                'cmd_roll_rate': '0.0',
                'cmd_pitch_rate': '0.0',
                'cmd_yaw_rate': '0.0',
                'roll': '0', 'pitch': '0',
                'gx_imu': '0', 'gy_imu': '0', 'gz_imu': '0',
            })
        path = _write(rows)
        loaded = load_run(path, lead_s=0.5, tail_s=0.5, sort_by_u=False)
        self.assertIsNotNone(loaded)
        _obs, _lab, _w, _valid, marked, _gates, _att, _dt = loaded
        # Indices 5..15 inclusive (t=0.5 .. 1.5).
        self.assertTrue(marked[5:16].all())
        self.assertFalse(marked[4])
        self.assertFalse(marked[16])


class SeedRunTests(unittest.TestCase):
    @staticmethod
    def _rows(n, policy_idx=()):
        return [
            {'t': f'{i * 0.1:.1f}',
             'control_authority': 'policy' if i in policy_idx else 'human',
             'cmd_thrust': '0.3', 'cmd_roll_rate': '0.0',
             'cmd_pitch_rate': '0.0', 'cmd_yaw_rate': '0.0',
             'gx_imu': '0', 'gy_imu': '0', 'gz_imu': '0'}
            for i in range(n)
        ]

    def test_all_human_run_has_no_interventions(self):
        """A demonstration lap is not one giant intervention."""
        _o, _l, w, _v, marked, _g, _a, _dt = load_run(
            _write(self._rows(10)), lead_s=0.5, tail_s=0.5, sort_by_u=False
        )
        self.assertFalse(marked.any())
        self.assertTrue((w == 1.0).all())

    def test_stray_prearm_policy_rows_do_not_make_it_a_dagger_round(self):
        """Pre-arm rows carry the default authority; 2 of 200 is not a round."""
        rows = self._rows(200, policy_idx=(0, 1))
        _o, _l, _w, _v, marked, _g, _a, _dt = load_run(
            _write(rows), lead_s=0.5, tail_s=0.5, sort_by_u=False
        )
        self.assertFalse(marked.any())

    def test_mixed_run_still_marks_interventions(self):
        rows = []
        for i in range(10):
            rows.append({
                't': f'{i * 0.1:.1f}',
                'control_authority': 'human' if i == 5 else 'policy',
                'cmd_thrust': '0.3', 'cmd_roll_rate': '0.0',
                'cmd_pitch_rate': '0.0', 'cmd_yaw_rate': '0.0',
                'gx_imu': '0', 'gy_imu': '0', 'gz_imu': '0',
            })
        _o, _l, _w, _v, marked, _g, _a, _dt = load_run(
            _write(rows), lead_s=0.2, tail_s=0.2, sort_by_u=False
        )
        self.assertTrue(marked[5])
        self.assertTrue(marked[3])   # back-dated lead
        self.assertTrue(marked[7])   # recovery tail
        self.assertFalse(marked[0])


class AttemptSegmentTests(unittest.TestCase):
    def test_window_does_not_span_a_reset(self):
        """A sim reset teleports the drone; a window may not cross it."""
        from tools.train_policy import build_windows

        rows = []
        for i in range(12):
            rows.append({
                't': f'{i * 0.1:.1f}', 'control_authority': 'human',
                'attempt': '1' if i < 6 else '2',
                'cmd_thrust': '0.3', 'cmd_roll_rate': '0.0',
                'cmd_pitch_rate': '0.0', 'cmd_yaw_rate': '0.0',
                'gx_imu': '0', 'gy_imu': '0', 'gz_imu': '0',
                # Distinct corner positions per attempt.
                **{f'kp{k}_u': ('100' if i < 6 else '500') for k in range(8)},
            })
        loaded = load_run(_write(rows), lead_s=0.0, tail_s=0.0, sort_by_u=False)
        X, _Y, _W, _G = build_windows([loaded], history=4)
        # The first row of attempt 2 must be padded from itself, not from
        # attempt 1's frames, so every frame in its window is the new position.
        first_of_attempt2 = X[6]
        self.assertTrue((first_of_attempt2[:, 0] == first_of_attempt2[0, 0]).all())
        self.assertGreater(first_of_attempt2[0, 0], 0.5)

    def test_single_attempt_behaves_as_before(self):
        from tools.train_policy import build_windows

        rows = [
            {'t': f'{i * 0.1:.1f}', 'control_authority': 'human',
             'attempt': '0',
             'cmd_thrust': '0.3', 'cmd_roll_rate': '0.0',
             'cmd_pitch_rate': '0.0', 'cmd_yaw_rate': '0.0',
             'gx_imu': '0', 'gy_imu': '0', 'gz_imu': '0'}
            for i in range(10)
        ]
        loaded = load_run(_write(rows), lead_s=0.0, tail_s=0.0, sort_by_u=False)
        X, _Y, _W, _G = build_windows([loaded], history=4)
        self.assertEqual(len(X), 10)


class PolicyFrameTests(unittest.TestCase):
    @staticmethod
    def _round(n=100, human_idx=range(40, 60)):
        hs = set(human_idx)
        return [
            {'t': f'{i * 0.1:.1f}',
             'control_authority': 'human' if i in hs else 'policy',
             'cmd_thrust': '0.3', 'cmd_roll_rate': '0.0',
             'cmd_pitch_rate': '0.0', 'cmd_yaw_rate': '0.0',
             'gx_imu': '0', 'gy_imu': '0', 'gz_imu': '0'}
            for i in range(n)
        ]

    def test_policy_frames_are_dropped_from_a_dagger_round(self):
        """Their labels are the policy's own output — self-imitation."""
        _o, _l, _w, valid, marked, _g, _a, _dt = load_run(
            _write(self._round()), lead_s=0.0, tail_s=0.0, sort_by_u=False
        )
        self.assertTrue(valid[40:60].all())
        self.assertFalse(valid[:40].any())
        self.assertFalse(valid[60:].any())
        self.assertEqual(int(valid.sum()), int(marked.sum()))

    def test_lead_and_tail_survive_the_drop(self):
        _o, _l, _w, valid, _m, _g, _a, _dt = load_run(
            _write(self._round()), lead_s=0.5, tail_s=0.5, sort_by_u=False
        )
        # 0.5 s at 0.1 s steps = 5 frames either side of 40..59.
        self.assertTrue(valid[35:65].all())
        self.assertFalse(valid[34])
        self.assertFalse(valid[65])

    def test_seed_run_keeps_every_frame(self):
        rows = [
            {'t': f'{i * 0.1:.1f}', 'control_authority': 'human',
             'cmd_thrust': '0.3', 'cmd_roll_rate': '0.0',
             'cmd_pitch_rate': '0.0', 'cmd_yaw_rate': '0.0',
             'gx_imu': '0', 'gy_imu': '0', 'gz_imu': '0'}
            for i in range(20)
        ]
        _o, _l, _w, valid, _m, _g, _a, _dt = load_run(
            _write(rows), lead_s=0.0, tail_s=0.0, sort_by_u=False
        )
        self.assertTrue(valid.all())

    def test_keep_policy_frames_opt_out(self):
        _o, _l, _w, valid, _m, _g, _a, _dt = load_run(
            _write(self._round()), lead_s=0.0, tail_s=0.0, sort_by_u=False,
            drop_policy_frames=False,
        )
        self.assertTrue(valid.all())


class WindowStrideTests(unittest.TestCase):
    @staticmethod
    def _rows(n, dt):
        return [
            {'t': f'{i * dt:.4f}', 'control_authority': 'human',
             'cmd_thrust': '0.3', 'cmd_roll_rate': '0.0',
             'cmd_pitch_rate': '0.0', 'cmd_yaw_rate': '0.0',
             'gx_imu': '0', 'gy_imu': '0', 'gz_imu': '0',
             **{f'kp{k}_u': f'{100 + i}' for k in range(8)}}
            for i in range(n)
        ]

    def test_fast_log_is_strided_to_match_deployment(self):
        """A 50 Hz log must cover the same span as a 10 Hz one."""
        from tools.train_policy import build_windows

        fast = load_run(_write(self._rows(200, 0.02)), lead_s=0.0, tail_s=0.0,
                        sort_by_u=False)
        X, _Y, _W, _G = build_windows([fast], history=4, target_dt=0.1)
        # Stride 5: the window ending at row 100 samples rows 85, 90, 95, 100.
        w = X[100]
        self.assertAlmostEqual(float(w[0, 0] * 640.0), 185.0, places=0)
        self.assertAlmostEqual(float(w[-1, 0] * 640.0), 200.0, places=0)

    def test_matching_rate_is_unstrided(self):
        from tools.train_policy import build_windows

        slow = load_run(_write(self._rows(50, 0.1)), lead_s=0.0, tail_s=0.0,
                        sort_by_u=False)
        X, _Y, _W, _G = build_windows([slow], history=4, target_dt=0.1)
        w = X[10]
        self.assertAlmostEqual(float(w[0, 0] * 640.0), 107.0, places=0)
        self.assertAlmostEqual(float(w[-1, 0] * 640.0), 110.0, places=0)

    def test_disabled_when_target_dt_zero(self):
        from tools.train_policy import build_windows

        fast = load_run(_write(self._rows(60, 0.02)), lead_s=0.0, tail_s=0.0,
                        sort_by_u=False)
        X, _Y, _W, _G = build_windows([fast], history=4, target_dt=0.0)
        w = X[20]
        self.assertAlmostEqual(float(w[-1, 0] * 640.0) - float(w[0, 0] * 640.0),
                               3.0, places=0)


class ExcludeTests(unittest.TestCase):
    def test_excluded_rows_are_not_trained_on(self):
        rows = []
        for i in range(10):
            rows.append({
                't': f'{i * 0.1:.1f}', 'control_authority': 'human',
                'exclude': '1' if 3 <= i <= 5 else '0',
                'cmd_thrust': '0.3', 'cmd_roll_rate': '0.0',
                'cmd_pitch_rate': '0.0', 'cmd_yaw_rate': '0.0',
                'gx_imu': '0', 'gy_imu': '0', 'gz_imu': '0',
            })
        _o, _l, _w, valid, _m, _g, _a, _dt = load_run(
            _write(rows), lead_s=0.0, tail_s=0.0, sort_by_u=False
        )
        self.assertFalse(valid[3:6].any())
        self.assertTrue(valid[0])
        self.assertTrue(valid[9])

    def test_missing_exclude_column_keeps_rows(self):
        rows = [
            {'t': f'{i * 0.1:.1f}', 'control_authority': 'human',
             'cmd_thrust': '0.3', 'cmd_roll_rate': '0.0',
             'cmd_pitch_rate': '0.0', 'cmd_yaw_rate': '0.0',
             'gx_imu': '0', 'gy_imu': '0', 'gz_imu': '0'}
            for i in range(5)
        ]
        _o, _l, _w, valid, _m, _g, _a, _dt = load_run(
            _write(rows), lead_s=0.0, tail_s=0.0, sort_by_u=False
        )
        self.assertTrue(valid.all())


class GateBalanceTests(unittest.TestCase):
    def test_dominant_gate_is_downweighted(self):
        gates = np.array([0] * 100 + [1] * 10 + [2] * 10, dtype=np.int32)
        w = gate_balance_weights(gates)
        self.assertLess(w[gates == 0][0], 1.0)
        self.assertGreaterEqual(w[gates == 1][0], 1.0)
        # Total influence of the dominant gate must fall.
        self.assertLess(w[gates == 0].sum(), 100.0)

    def test_clip_bounds_the_factor(self):
        gates = np.array([0] * 10000 + [1] * 2, dtype=np.int32)
        w = gate_balance_weights(gates, clip=4.0)
        self.assertGreaterEqual(w.min(), 0.25 - 1e-6)
        self.assertLessEqual(w.max(), 4.0 + 1e-6)

    def test_single_gate_is_left_alone(self):
        gates = np.zeros(50, dtype=np.int32)
        self.assertTrue((gate_balance_weights(gates) == 1.0).all())

    def test_unknown_gate_rows_are_ignored(self):
        gates = np.array([-1] * 10 + [0] * 10 + [1] * 10, dtype=np.int32)
        w = gate_balance_weights(gates)
        self.assertTrue((w[gates == -1] == 1.0).all())


class CollisionFilterTests(unittest.TestCase):
    def _run_with_events(self, event_text):
        path = _write([
            {
                't': f'{i * 0.1:.1f}', 'control_authority': 'human',
                'cmd_thrust': '0.3', 'cmd_roll_rate': '0.0',
                'cmd_pitch_rate': '0.0', 'cmd_yaw_rate': '0.0',
                'ahrs_roll': '0.0', 'ahrs_pitch': '0.0',
                'roll': '0', 'pitch': '0',
                'gx_imu': '0', 'gy_imu': '0', 'gz_imu': '0',
            }
            for i in range(20)
        ])
        events = path.with_name(
            path.name.replace('telem', 'events')
        ).with_suffix('.txt')
        events.write_text(event_text)
        return path

    def test_parses_impulse_and_time(self):
        path = self._run_with_events(
            '    1.000  COLLISION    Gate  threat=2  impulse=6.30kg\n'
            '    1.500  COLLISION    Environment  threat=1  impulse=0.20kg\n'
        )
        self.assertEqual(collision_times(path), [1.0, 1.5])
        self.assertEqual(collision_times(path, min_impulse=1.0), [1.0])

    def test_rows_near_contact_are_invalidated(self):
        path = self._run_with_events(
            '    1.000  COLLISION    Gate  threat=2  impulse=6.30kg\n'
        )
        _o, _l, _w, valid, _m, _g, _a, _dt = load_run(
            path, lead_s=0.0, tail_s=0.0, sort_by_u=False,
            drop_collision_s=0.25,
        )
        # t = 0.8 .. 1.2 sits within 0.25 s of the 1.0 s contact.
        self.assertFalse(valid[8:13].any())
        self.assertTrue(valid[0])
        self.assertTrue(valid[15])

    def test_filter_off_by_default(self):
        path = self._run_with_events(
            '    1.000  COLLISION    Gate  threat=2  impulse=6.30kg\n'
        )
        _o, _l, _w, valid, _m, _g, _a, _dt = load_run(
            path, lead_s=0.0, tail_s=0.0, sort_by_u=False
        )
        self.assertTrue(valid.all())


if __name__ == '__main__':
    unittest.main()
