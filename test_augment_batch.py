"""The vectorised augmentation must match the scalar reference implementation."""
from __future__ import annotations

import unittest

import numpy as np

from race_obs import (
    FEATURE_DIM,
    KEYPOINT_COUNT,
    NOT_SEEN,
    augment_corners,
    build_observation,
)
from tools.train_policy import augment_batch


def _window(n_frames=4, n_visible=8):
    kps = [(100.0 + 30 * i, 120.0 + 12 * i) for i in range(n_visible)]
    kps += [(0.0, 0.0)] * (KEYPOINT_COUNT - n_visible)
    obs = build_observation(kps, [0.9] * n_visible + [0.0] * (8 - n_visible))
    return np.array([obs] * n_frames, dtype=np.float32)


class AugmentBatchTests(unittest.TestCase):
    def test_no_dropout_no_jitter_is_identity(self):
        import torch

        x = torch.from_numpy(_window()[None, ...])
        out = augment_batch(x, np.random.default_rng(0),
                            jitter_px=0.0, dropout=0.0).numpy()
        np.testing.assert_allclose(out, x.numpy(), atol=1e-6)

    def test_dropout_one_marks_every_corner_unseen(self):
        import torch

        x = torch.from_numpy(_window()[None, ...])
        out = augment_batch(x, np.random.default_rng(0),
                            jitter_px=0.0, dropout=1.0).numpy()[0]
        for i in range(KEYPOINT_COUNT):
            self.assertEqual(out[0, 2 * i], NOT_SEEN)
            self.assertEqual(out[0, KEYPOINT_COUNT * 2 + i], 0.0)

    def test_unseen_corners_are_never_revived(self):
        import torch

        x = torch.from_numpy(_window(n_visible=3)[None, ...])
        out = augment_batch(x, np.random.default_rng(1),
                            jitter_px=10.0, dropout=0.5).numpy()[0]
        for i in range(3, KEYPOINT_COUNT):
            self.assertEqual(out[0, 2 * i], NOT_SEEN)
            self.assertEqual(out[0, KEYPOINT_COUNT * 2 + i], 0.0)

    def test_jitter_stays_in_range(self):
        import torch

        x = torch.from_numpy(_window()[None, ...])
        out = augment_batch(x, np.random.default_rng(2),
                            jitter_px=50.0, dropout=0.0).numpy()
        corners = out[..., : 2 * KEYPOINT_COUNT]
        self.assertGreaterEqual(corners.min(), NOT_SEEN)
        self.assertLessEqual(corners.max(), 1.0)

    def test_matches_scalar_reference_statistically(self):
        """Same distribution as the per-row reference: mean shift near zero."""
        import torch

        base = _window(n_frames=1)          # (1, FEATURE_DIM)
        row = base[0]
        x = torch.from_numpy(np.tile(base, (400, 1, 1)))
        vec = augment_batch(x, np.random.default_rng(7),
                            jitter_px=10.0, dropout=0.0).numpy()

        rng_s = np.random.default_rng(7)
        scal = np.array([
            augment_corners(list(row), rng_s, jitter_px=10.0, dropout=0.0)
            for _ in range(400)
        ], dtype=np.float64)
        u0_vec = vec[:, 0, 0]
        u0_scal = scal[:, 0]
        self.assertAlmostEqual(float(u0_vec.mean()), float(u0_scal.mean()),
                               delta=0.01)
        self.assertAlmostEqual(float(u0_vec.std()), float(u0_scal.std()),
                               delta=0.01)

    def test_shape_and_dtype_preserved(self):
        import torch

        x = torch.from_numpy(_window(n_frames=6)[None, ...])
        out = augment_batch(x, np.random.default_rng(0),
                            jitter_px=5.0, dropout=0.1)
        self.assertEqual(tuple(out.shape), tuple(x.shape))
        self.assertEqual(out.dtype, x.dtype)

    def test_does_not_mutate_the_input(self):
        import torch

        x = torch.from_numpy(_window()[None, ...])
        before = x.numpy().copy()
        augment_batch(x, np.random.default_rng(3), jitter_px=20.0, dropout=0.5)
        np.testing.assert_allclose(x.numpy(), before)


if __name__ == '__main__':
    unittest.main()
