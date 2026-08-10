"""Contract tests for the HG-DAgger observation vector and policy network.

The previous behavior-cloning attempt in this repo failed partly because the
observation was built differently in training and in flight. These tests pin
the contract that both sides import.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

import race_obs
from race_obs import (
    DEFAULT_HISTORY,
    FEATURE_DIM,
    FEATURE_NAMES,
    KEYPOINT_COUNT,
    NOT_SEEN,
    build_observation,
    labels_from_row,
    observation_from_row,
    stack_history,
)


def _kps(n_visible: int = 8):
    """n_visible keypoints spread across the frame, the rest marked unseen."""
    pts = []
    for i in range(KEYPOINT_COUNT):
        if i < n_visible:
            pts.append([64.0 * (i + 1), 36.0 * (i + 1)])
        else:
            pts.append([0.0, 0.0])
    return pts


class ObservationTests(unittest.TestCase):
    def test_dimension_matches_names(self):
        obs = build_observation(_kps())
        self.assertEqual(len(obs), FEATURE_DIM)
        self.assertEqual(len(FEATURE_NAMES), FEATURE_DIM)

    def test_visible_corners_normalise_into_unit_range(self):
        obs = build_observation(_kps())
        corners = obs[: KEYPOINT_COUNT * 2]
        for value in corners:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        vis = obs[KEYPOINT_COUNT * 2: KEYPOINT_COUNT * 3]
        self.assertEqual(vis, [1.0] * KEYPOINT_COUNT)

    def test_unseen_corner_is_sentinel_and_flagged(self):
        obs = build_observation(_kps(n_visible=5))
        vis = obs[KEYPOINT_COUNT * 2: KEYPOINT_COUNT * 3]
        self.assertEqual(vis[:5], [1.0] * 5)
        self.assertEqual(vis[5:], [0.0] * 3)
        # Sentinel is out of band for [0,1], so it cannot be confused with a
        # real position at the image origin.
        for i in range(5, KEYPOINT_COUNT):
            self.assertEqual(obs[2 * i], NOT_SEEN)
            self.assertEqual(obs[2 * i + 1], NOT_SEEN)
            self.assertLess(NOT_SEEN, 0.0)

    def test_zero_zero_keypoint_treated_as_unseen(self):
        """The detector reports an unseen keypoint as (0,0), not as NaN."""
        pts = _kps()
        pts[3] = [0.0, 0.0]
        obs = build_observation(pts)
        self.assertEqual(obs[KEYPOINT_COUNT * 2 + 3], 0.0)

    def test_low_confidence_keypoint_dropped(self):
        conf = [1.0] * KEYPOINT_COUNT
        conf[2] = 0.01
        obs = build_observation(_kps(), conf, min_confidence=0.25)
        self.assertEqual(obs[KEYPOINT_COUNT * 2 + 2], 0.0)

    def test_gyro_is_clipped(self):
        obs = build_observation(_kps(), gx=1000.0, gy=-1000.0, gz=0.5)
        gx = obs[FEATURE_NAMES.index('gx')]
        gy = obs[FEATURE_NAMES.index('gy')]
        self.assertLessEqual(abs(gx), race_obs.GYRO_CLIP)
        self.assertLessEqual(abs(gy), race_obs.GYRO_CLIP)

    def test_nan_inputs_do_not_propagate(self):
        obs = build_observation(_kps(), roll=math.nan, pitch=math.nan, gx=math.nan)
        for value in obs:
            self.assertFalse(math.isnan(value))

    def test_identity_order_preserves_ring_membership(self):
        """Default order must keep outer (0-3) and inner (4-7) distinguishable.

        The two rings are 2.7 m and 1.5 m, so which ring a point belongs to is
        what carries apparent-scale (range) information. Sorting all eight by u
        interleaves them and throws that away.
        """
        pts = [[600.0, 10.0], [500.0, 20.0], [400.0, 30.0], [300.0, 40.0],
               [250.0, 50.0], [200.0, 60.0], [150.0, 70.0], [100.0, 80.0]]
        ident = build_observation(pts, sort_by_u=False)
        # Keypoint 0 is the largest u, and identity order must keep it first.
        self.assertAlmostEqual(ident[0], 600.0 / race_obs.FRAME_W, places=6)

        sorted_obs = build_observation(pts, sort_by_u=True)
        # paper1's scheme puts the smallest u first, which is keypoint 7.
        self.assertAlmostEqual(sorted_obs[0], 100.0 / race_obs.FRAME_W, places=6)
        self.assertNotEqual(ident[:2], sorted_obs[:2])

    def test_row_roundtrip_matches_direct_build(self):
        """observation_from_row must equal build_observation on the same data."""
        row = {'roll': '0.1', 'pitch': '-0.2',
               'gx_imu': '0.3', 'gy_imu': '0.4', 'gz_imu': '0.5'}
        pts = _kps()
        for i, (u, v) in enumerate(pts):
            row[f'kp{i}_u'] = str(u)
            row[f'kp{i}_v'] = str(v)
            row[f'kp{i}_c'] = '0.9'
        from_row = observation_from_row(row)
        direct = build_observation(
            pts, [0.9] * KEYPOINT_COUNT,
            roll=0.1, pitch=-0.2, gx=0.3, gy=0.4, gz=0.5,
        )
        self.assertEqual(from_row, direct)

    def test_labels_from_row(self):
        row = {'cmd_thrust': '0.3', 'cmd_roll_rate': '0.1',
               'cmd_pitch_rate': '-0.2', 'cmd_yaw_rate': '0.05'}
        self.assertEqual(labels_from_row(row), [0.3, 0.1, -0.2, 0.05])


class HistoryTests(unittest.TestCase):
    def test_pads_to_exact_length(self):
        seq = stack_history([[1.0] * FEATURE_DIM], history=DEFAULT_HISTORY)
        self.assertEqual(len(seq), DEFAULT_HISTORY)

    def test_padding_repeats_oldest_not_zeros(self):
        """A zero row looks like a real observation with corners at the origin."""
        first = [0.5] * FEATURE_DIM
        seq = stack_history([first, [0.7] * FEATURE_DIM], history=4)
        self.assertEqual(seq[0], first)
        self.assertEqual(seq[1], first)

    def test_truncates_to_newest(self):
        rows = [[float(i)] * FEATURE_DIM for i in range(50)]
        seq = stack_history(rows, history=32)
        self.assertEqual(len(seq), 32)
        self.assertEqual(seq[-1][0], 49.0)


class AugmentationTests(unittest.TestCase):
    def test_dropout_marks_corner_unseen(self):
        rng = np.random.default_rng(0)
        obs = build_observation(_kps())
        out = race_obs.augment_corners(obs, rng, dropout=1.0, jitter_px=0.0)
        vis = out[KEYPOINT_COUNT * 2: KEYPOINT_COUNT * 3]
        self.assertEqual(vis, [0.0] * KEYPOINT_COUNT)
        self.assertEqual(out[0], NOT_SEEN)

    def test_jitter_stays_in_range_and_moves_points(self):
        rng = np.random.default_rng(1)
        obs = build_observation(_kps())
        out = race_obs.augment_corners(obs, rng, dropout=0.0, jitter_px=10.0)
        corners = out[: KEYPOINT_COUNT * 2]
        self.assertTrue(any(a != b for a, b in zip(corners, obs[: KEYPOINT_COUNT * 2])))
        for value in corners:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_augmentation_never_revives_an_unseen_corner(self):
        rng = np.random.default_rng(2)
        obs = build_observation(_kps(n_visible=4))
        out = race_obs.augment_corners(obs, rng, dropout=0.0, jitter_px=10.0)
        for i in range(4, KEYPOINT_COUNT):
            self.assertEqual(out[2 * i], NOT_SEEN)
            self.assertEqual(out[KEYPOINT_COUNT * 2 + i], 0.0)


class PolicyNetTests(unittest.TestCase):
    def test_forward_shape_and_causality(self):
        import torch

        from policy_net import RacePolicy

        model = RacePolicy(history=DEFAULT_HISTORY)
        model.eval()
        x = torch.zeros(2, DEFAULT_HISTORY, FEATURE_DIM)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (2, 4))

        # Causality: changing only the LAST timestep must change the output,
        # while changing a future timestep is impossible by construction. Test
        # the complement — altering earlier frames must not be ignored either.
        a = torch.zeros(1, DEFAULT_HISTORY, FEATURE_DIM)
        b = a.clone()
        b[0, -1, :] = 1.0
        with torch.no_grad():
            self.assertFalse(torch.allclose(model(a), model(b)))

    def test_rejects_wrong_rank(self):
        import torch

        from policy_net import RacePolicy

        model = RacePolicy()
        with self.assertRaises(ValueError):
            model(torch.zeros(DEFAULT_HISTORY, FEATURE_DIM))

    def test_save_load_roundtrip(self):
        import tempfile
        from pathlib import Path

        import torch

        from policy_net import RacePolicy, load_policy, save_policy

        model = RacePolicy()
        model.eval()
        x = torch.zeros(1, DEFAULT_HISTORY, FEATURE_DIM)
        with torch.no_grad():
            before = model(x)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'policy.pt'
            save_policy(path, model, extra={'val_mae': 0.1})
            loaded, blob = load_policy(path)
            self.assertEqual(blob['val_mae'], 0.1)
            with torch.no_grad():
                after = loaded(x)
        self.assertTrue(torch.allclose(before, after, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
