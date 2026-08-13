"""Course context channels, action chunking, and checkpoint compatibility."""
from __future__ import annotations

import math
import unittest

import race_obs
from race_obs import (
    CONTEXT_CHANNELS,
    FEATURE_DIM,
    FEATURE_DIM_CTX,
    N_GATES,
    build_observation,
    context_features,
    feature_dim,
    observation_from_row,
    stack_history,
)


def _kps(n=8):
    return [(100.0 + 10 * i, 150.0 + 5 * i) for i in range(n)]


class ContextFeatureTests(unittest.TestCase):
    def test_dims_line_up(self):
        self.assertEqual(len(CONTEXT_CHANNELS), N_GATES + 1)
        self.assertEqual(FEATURE_DIM_CTX, FEATURE_DIM + N_GATES + 1)
        self.assertEqual(feature_dim(False), FEATURE_DIM)
        self.assertEqual(feature_dim(True), FEATURE_DIM_CTX)

    def test_one_hot_marks_the_active_gate(self):
        f = context_features(7)
        self.assertEqual(f[7], 1.0)
        self.assertEqual(sum(f[:N_GATES]), 1.0)
        self.assertAlmostEqual(f[N_GATES], 7 / (N_GATES - 1))

    def test_out_of_range_and_missing_are_safe(self):
        self.assertEqual(context_features(999)[N_GATES - 1], 1.0)
        self.assertEqual(context_features(-5)[0], 1.0)
        self.assertEqual(sum(context_features(None)), 0.0)
        self.assertEqual(sum(context_features('nan')), 0.0)

    def test_observation_width_switches_with_context(self):
        self.assertEqual(len(build_observation(_kps())), FEATURE_DIM)
        self.assertEqual(
            len(build_observation(_kps(), gate_index=3)), FEATURE_DIM_CTX
        )

    def test_context_is_appended_after_the_base_features(self):
        base = build_observation(_kps())
        ctx = build_observation(_kps(), gate_index=3)
        self.assertEqual(base, ctx[:FEATURE_DIM])
        self.assertEqual(ctx[FEATURE_DIM + 3], 1.0)

    def test_row_path_reads_active_gate(self):
        row = {'active_gate': '5', 'roll': '0.0', 'pitch': '0.0',
               'gx_imu': '0', 'gy_imu': '0', 'gz_imu': '0',
               'ahrs_roll': '0.0', 'ahrs_pitch': '0.0'}
        for i in range(8):
            row[f'kp{i}_u'] = '100'
            row[f'kp{i}_v'] = '100'
            row[f'kp{i}_c'] = '1'
        self.assertEqual(len(observation_from_row(row)), FEATURE_DIM)
        wide = observation_from_row(row, with_context=True)
        self.assertEqual(len(wide), FEATURE_DIM_CTX)
        self.assertEqual(wide[FEATURE_DIM + 5], 1.0)

    def test_history_padding_preserves_context_width(self):
        rows = [build_observation(_kps(), gate_index=2)]
        out = stack_history(rows, 32)
        self.assertEqual(len(out), 32)
        self.assertTrue(all(len(r) == FEATURE_DIM_CTX for r in out))


class ChunkTests(unittest.TestCase):
    def test_single_step_shape_unchanged(self):
        import torch

        from policy_net import RacePolicy

        m = RacePolicy(history=8, chunk=1)
        y = m(torch.zeros(3, 8, FEATURE_DIM))
        self.assertEqual(tuple(y.shape), (3, 4))

    def test_chunked_shape(self):
        import torch

        from policy_net import RacePolicy

        m = RacePolicy(history=8, chunk=5)
        y = m(torch.zeros(3, 8, FEATURE_DIM))
        self.assertEqual(tuple(y.shape), (3, 5, 4))

    def test_context_width_model(self):
        import torch

        from policy_net import RacePolicy

        m = RacePolicy(n_in=FEATURE_DIM_CTX, history=8, chunk=2)
        y = m(torch.zeros(2, 8, FEATURE_DIM_CTX))
        self.assertEqual(tuple(y.shape), (2, 2, 4))

    def test_checkpoint_roundtrip_keeps_chunk_and_context(self):
        import tempfile
        from pathlib import Path

        import torch

        from policy_net import RacePolicy, load_policy, save_policy

        m = RacePolicy(n_in=FEATURE_DIM_CTX, history=8, chunk=4)
        m.eval()
        x = torch.zeros(1, 8, FEATURE_DIM_CTX)
        with torch.no_grad():
            before = m(x)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'p.pt'
            save_policy(p, m, extra={'context': True, 'chunk': 4})
            loaded, blob = load_policy(p)
            self.assertTrue(blob['context'])
            self.assertEqual(blob['arch']['chunk'], 4)
            self.assertEqual(blob['arch']['n_in'], FEATURE_DIM_CTX)
            with torch.no_grad():
                after = loaded(x)
        self.assertTrue(torch.allclose(before, after, atol=1e-6))

    def test_old_checkpoint_without_chunk_still_loads(self):
        """Checkpoints written before chunking existed must keep working."""
        import tempfile
        from pathlib import Path

        import torch

        from policy_net import RacePolicy, load_policy

        m = RacePolicy(history=8)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'old.pt'
            torch.save({
                'arch': {'n_in': FEATURE_DIM, 'n_out': 4, 'history': 8},
                'state_dict': m.state_dict(),
            }, p)
            loaded, _blob = load_policy(p)
        self.assertEqual(loaded.chunk, 1)


class EnsembleTests(unittest.TestCase):
    def test_temporal_ensembling_averages_overlapping_plans(self):
        """Successive chunk predictions for 'now' should be averaged."""
        import numpy as np

        chunk = 3
        plans: list = []
        # Three plans, each predicting roll for now/next/next-next.
        # Plan i predicts value (i + 1) for every horizon step.
        for i in range(3):
            plans.append(np.full((chunk, 4), float(i + 1)))

        acc, n = None, 0
        for age, plan in enumerate(reversed(plans)):
            if age >= len(plan):
                continue
            acc = plan[age] if acc is None else acc + plan[age]
            n += 1
        avg = acc / n
        # Newest plan's step 0, middle plan's step 1, oldest plan's step 2.
        self.assertAlmostEqual(float(avg[0]), (3 + 2 + 1) / 3)

    def test_plans_older_than_the_horizon_are_dropped(self):
        import numpy as np

        chunk = 2
        plans = [np.full((chunk, 4), float(i + 1)) for i in range(5)]
        acc, n = None, 0
        for age, plan in enumerate(reversed(plans)):
            if age >= len(plan):
                continue
            acc = plan[age] if acc is None else acc + plan[age]
            n += 1
        self.assertEqual(n, chunk)


if __name__ == '__main__':
    unittest.main()
