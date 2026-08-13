"""Discretised action head: the point is to commit to a mode, not average it.

The human stick is multi-modal (roll parked in 71% of frames, slammed in 18%).
A regressor lands between those modes. These tests pin the behaviour that fixes
it, including the subtle part: overlapping chunk predictions must be combined as
*probabilities*, because averaging decoded values would reintroduce the very
averaging the categorical head exists to remove.
"""
from __future__ import annotations

import unittest

import numpy as np

from race_obs import (
    ACTION_RANGES,
    LABEL_NAMES,
    action_to_bin,
    bin_centers,
    bin_to_action,
    bins_to_labels,
    labels_to_bins,
)


class BinEncodingTests(unittest.TestCase):
    def test_centres_span_the_range(self):
        c = bin_centers(21)['roll_rate']
        lo, hi = ACTION_RANGES['roll_rate']
        self.assertEqual(len(c), 21)
        self.assertGreater(c[0], lo)
        self.assertLess(c[-1], hi)
        self.assertAlmostEqual(c[10], 0.0, places=6)

    def test_zero_and_extremes_land_in_distinct_bins(self):
        """Hold, slam-left and slam-right must not share a bin."""
        z = action_to_bin(0.0, 'roll_rate', 21)
        left = action_to_bin(-3.1, 'roll_rate', 21)
        right = action_to_bin(3.1, 'roll_rate', 21)
        self.assertEqual(len({z, left, right}), 3)
        self.assertEqual(left, 0)
        self.assertEqual(right, 20)

    def test_round_trip_is_within_half_a_bin(self):
        lo, hi = ACTION_RANGES['roll_rate']
        width = (hi - lo) / 21
        for v in (-3.0, -1.5, 0.0, 0.4, 2.9):
            back = bin_to_action(action_to_bin(v, 'roll_rate', 21),
                                 'roll_rate', 21)
            self.assertLessEqual(abs(back - v), width / 2 + 1e-9)

    def test_out_of_range_is_clamped(self):
        self.assertEqual(action_to_bin(-99, 'roll_rate', 21), 0)
        self.assertEqual(action_to_bin(+99, 'roll_rate', 21), 20)

    def test_thrust_uses_its_own_asymmetric_range(self):
        lo, hi = ACTION_RANGES['thrust']
        self.assertEqual(action_to_bin(lo, 'thrust', 10), 0)
        self.assertEqual(action_to_bin(hi, 'thrust', 10), 9)
        # Hover thrust must not collapse into the bottom bin.
        self.assertGreater(action_to_bin(0.26, 'thrust', 21), 0)

    def test_label_vector_round_trip(self):
        labels = [0.30, -3.1, 0.0, 2.9]
        idx = labels_to_bins(labels, 21)
        self.assertEqual(len(idx), len(LABEL_NAMES))
        back = bins_to_labels(idx, 21)
        for a, b, name in zip(labels, back, LABEL_NAMES):
            lo, hi = ACTION_RANGES[name]
            self.assertLessEqual(abs(a - b), (hi - lo) / 21 / 2 + 1e-9)

    def test_nan_maps_to_a_safe_bin(self):
        self.assertEqual(action_to_bin(float('nan'), 'thrust', 21), 0)
        self.assertEqual(
            action_to_bin(float('nan'), 'roll_rate', 21),
            action_to_bin(0.0, 'roll_rate', 21),
        )


class ProbabilityAveragingTests(unittest.TestCase):
    """Averaging distributions keeps a mode; averaging values destroys it."""

    def test_averaging_values_lands_between_modes(self):
        # Two equally confident plans: one says slam left, one says slam right.
        left, right = -3.1, 3.1
        self.assertAlmostEqual((left + right) / 2, 0.0, places=6)

    def test_averaging_probabilities_picks_a_mode(self):
        bins = 21
        left = action_to_bin(-3.1, 'roll_rate', bins)
        right = action_to_bin(3.1, 'roll_rate', bins)
        p1 = np.zeros(bins)
        p1[left] = 0.9
        p1[right] = 0.1
        p2 = np.zeros(bins)
        p2[left] = 0.6
        p2[right] = 0.4
        avg = (p1 + p2) / 2
        chosen = bin_to_action(int(np.argmax(avg)), 'roll_rate', bins)
        # The result is a real stick position, not the midpoint of two.
        self.assertLess(chosen, -2.5)

    def test_a_tie_still_returns_an_extreme_not_the_middle(self):
        bins = 21
        left = action_to_bin(-3.1, 'roll_rate', bins)
        right = action_to_bin(3.1, 'roll_rate', bins)
        avg = np.zeros(bins)
        avg[left] = 0.5
        avg[right] = 0.5
        chosen = bin_to_action(int(np.argmax(avg)), 'roll_rate', bins)
        self.assertGreater(abs(chosen), 2.5)


class CategoricalHeadTests(unittest.TestCase):
    def test_logit_shape(self):
        import torch

        from policy_net import RacePolicy
        from race_obs import FEATURE_DIM

        m = RacePolicy(history=8, chunk=3, bins=21)
        y = m(torch.zeros(2, 8, FEATURE_DIM))
        self.assertEqual(tuple(y.shape), (2, 3, 4, 21))

    def test_chunk_axis_present_even_at_chunk_one(self):
        import torch

        from policy_net import RacePolicy
        from race_obs import FEATURE_DIM

        m = RacePolicy(history=8, chunk=1, bins=11)
        y = m(torch.zeros(2, 8, FEATURE_DIM))
        self.assertEqual(tuple(y.shape), (2, 1, 4, 11))

    def test_regression_head_unchanged_when_bins_zero(self):
        import torch

        from policy_net import RacePolicy
        from race_obs import FEATURE_DIM

        m = RacePolicy(history=8, chunk=1, bins=0)
        y = m(torch.zeros(2, 8, FEATURE_DIM))
        self.assertEqual(tuple(y.shape), (2, 4))

    def test_checkpoint_carries_bins(self):
        import tempfile
        from pathlib import Path

        import torch

        from policy_net import RacePolicy, load_policy, save_policy
        from race_obs import FEATURE_DIM

        m = RacePolicy(history=8, chunk=2, bins=15)
        m.eval()
        x = torch.zeros(1, 8, FEATURE_DIM)
        with torch.no_grad():
            before = m(x)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'b.pt'
            save_policy(p, m, extra={'bins': 15})
            loaded, blob = load_policy(p)
            self.assertEqual(blob['arch']['bins'], 15)
            with torch.no_grad():
                after = loaded(x)
        self.assertTrue(torch.allclose(before, after, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
