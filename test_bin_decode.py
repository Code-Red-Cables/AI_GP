"""Bin decoding must pick a mode, not average across modes, and not snap.

The failure this pins: a hard argmax emitted exactly one bin centre per channel
forever -- zero roll for a whole flight, saturated pitch held into a tumble.
"""
from __future__ import annotations

import unittest

from race_obs import (
    ACTION_RANGES,
    LABEL_NAMES,
    bin_centers,
    decode_bin_probs,
)

N = 21
ROLL = LABEL_NAMES.index('roll_rate')


def flat_probs(n_bins=N):
    return [[1.0 / n_bins] * n_bins for _ in LABEL_NAMES]


def spike(idx, n_bins=N, mass=1.0):
    row = [(1.0 - mass) / (n_bins - 1)] * n_bins
    row[idx] = mass
    return row


class DecodeTests(unittest.TestCase):
    def test_window_zero_is_argmax(self):
        probs = flat_probs()
        probs[ROLL] = spike(3, mass=0.9)
        got = decode_bin_probs(probs, N, window=0)
        self.assertAlmostEqual(got[ROLL], bin_centers(N)['roll_rate'][3])

    def test_confident_spike_decodes_near_its_centre(self):
        probs = flat_probs()
        probs[ROLL] = spike(15, mass=0.99)
        got = decode_bin_probs(probs, N, window=2)
        self.assertAlmostEqual(got[ROLL], bin_centers(N)['roll_rate'][15],
                               delta=0.05)

    def test_bimodal_does_not_collapse_to_the_middle(self):
        """Mass at both extremes must decode to one extreme, never to zero."""
        n = N
        row = [0.0] * n
        row[0] = 0.5
        row[n - 1] = 0.5
        probs = flat_probs()
        probs[ROLL] = row
        got = decode_bin_probs(probs, n, window=2)[ROLL]
        lo, hi = ACTION_RANGES['roll_rate']
        self.assertGreater(abs(got), 0.8 * hi,
                           f'bimodal decode collapsed to {got}')

    def test_broad_peak_lands_between_bin_centres(self):
        """The whole point: output is not restricted to bin centres."""
        centres = bin_centers(N)['roll_rate']
        row = [0.0] * N
        row[10] = 0.5
        row[11] = 0.5
        probs = flat_probs()
        probs[ROLL] = row
        got = decode_bin_probs(probs, N, window=2)[ROLL]
        self.assertAlmostEqual(got, 0.5 * (centres[10] + centres[11]), places=6)
        self.assertNotAlmostEqual(got, centres[10])
        self.assertNotAlmostEqual(got, centres[11])

    def test_neighbour_mass_pulls_a_saturated_bin_inward(self):
        """A peak at the outermost bin should not decode as full saturation
        when the neighbours carry real mass."""
        centres = bin_centers(N)['roll_rate']
        row = [0.0] * N
        row[N - 1] = 0.4
        row[N - 2] = 0.3
        row[N - 3] = 0.3
        probs = flat_probs()
        probs[ROLL] = row
        got = decode_bin_probs(probs, N, window=2)[ROLL]
        self.assertLess(got, centres[N - 1])
        self.assertGreater(got, centres[N - 3])

    def test_unnormalised_input_is_accepted(self):
        probs = flat_probs()
        probs[ROLL] = [x * 7.0 for x in spike(4, mass=0.9)]
        got = decode_bin_probs(probs, N, window=1)[ROLL]
        self.assertAlmostEqual(got, bin_centers(N)['roll_rate'][4], delta=0.2)

    def test_all_zero_row_falls_back_to_argmax_centre(self):
        probs = flat_probs()
        probs[ROLL] = [0.0] * N
        got = decode_bin_probs(probs, N, window=2)[ROLL]
        self.assertAlmostEqual(got, bin_centers(N)['roll_rate'][0])

    def test_returns_one_value_per_label(self):
        got = decode_bin_probs(flat_probs(), N, window=2)
        self.assertEqual(len(got), len(LABEL_NAMES))

    def test_every_channel_uses_its_own_range(self):
        """thrust is not symmetric about zero; its centres must differ."""
        probs = flat_probs()
        for c, name in enumerate(LABEL_NAMES):
            probs[c] = spike(N - 1, mass=0.99)
        got = decode_bin_probs(probs, N, window=0)
        thr = LABEL_NAMES.index('thrust')
        self.assertAlmostEqual(got[thr], bin_centers(N)['thrust'][N - 1])
        self.assertNotAlmostEqual(got[thr], got[ROLL])


if __name__ == '__main__':
    unittest.main()
