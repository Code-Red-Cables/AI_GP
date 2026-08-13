"""Flight-path integration: the planner must accept what training produced.

The single most damaging class of bug in this pipeline is training and flight
disagreeing about the observation -- it is invisible in training metrics and only
shows up as a policy that flies badly. These tests build a checkpoint the way
``tools/train_policy.py`` does and then drive ``PolicyPlanner`` with a synthetic
``shared_data``, asserting the widths line up and a command comes out.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from race_obs import FEATURE_DIM, FEATURE_DIM_CTX


def _shared(gate=7):
    return {
        'gate_detection': {
            'keypoints_px': [(100.0 + 10 * i, 150.0 + 5 * i) for i in range(8)],
            'keypoint_confidences': [0.9] * 8,
        },
        'highres_imu': {'xgyro': 0.1, 'ygyro': -0.2, 'zgyro': 0.05},
        'control_output': {'ahrs_roll': 0.05, 'ahrs_pitch': -0.02},
        'race_status': {'active_gate': gate},
    }


def _checkpoint(tmp: Path, *, context: bool, chunk: int, bins: int = 0) -> Path:
    from policy_net import RacePolicy, save_policy

    n_in = FEATURE_DIM_CTX if context else FEATURE_DIM
    model = RacePolicy(n_in=n_in, history=8, chunk=chunk, bins=bins)
    path = tmp / f'ctx{int(context)}_chunk{chunk}_bins{bins}.pt'
    save_policy(
        path, model,
        extra={'context': context, 'chunk': chunk, 'bins': bins},
    )
    return path


class PlannerIntegrationTests(unittest.TestCase):
    def _run(self, *, context: bool, chunk: int, bins: int = 0, steps: int = 8):
        from policy_planner import PolicyPlanner

        with tempfile.TemporaryDirectory() as tmp:
            path = _checkpoint(
                Path(tmp), context=context, chunk=chunk, bins=bins
            )
            planner = PolicyPlanner(str(path))
            shared = _shared()
            target = None
            for _ in range(steps):
                target = planner.compute_target(shared)
            return target, shared

    def test_plain_checkpoint_flies(self):
        target, shared = self._run(context=False, chunk=1)
        for key in ('thrust', 'roll_rate', 'pitch_rate', 'yaw_rate'):
            self.assertIn(key, target)
        self.assertEqual(shared['policy_obs']['n_vis'], 8.0)

    def test_context_checkpoint_reads_active_gate(self):
        target, _shared = self._run(context=True, chunk=1)
        self.assertIn('thrust', target)

    def test_chunked_checkpoint_produces_one_command_per_call(self):
        target, _shared = self._run(context=False, chunk=5)
        self.assertIsInstance(target['roll_rate'], float)

    def test_context_and_chunk_together(self):
        target, _shared = self._run(context=True, chunk=5)
        self.assertIsInstance(target['roll_rate'], float)

    def test_categorical_head_decodes_inside_the_action_range(self):
        """A discretised policy decodes to a continuous value in range.

        It is deliberately NOT restricted to bin centres: snapping to the
        winning centre made roll sit at exactly 0.0000 for an entire flight
        and pinned pitch to full deflection. See ``decode_bin_probs``.
        """
        from race_obs import ACTION_RANGES

        target, _shared = self._run(context=False, chunk=1, bins=21)
        lo, hi = ACTION_RANGES['roll_rate']
        self.assertGreaterEqual(target['roll_rate'], lo)
        self.assertLessEqual(target['roll_rate'], hi)
        self.assertIsInstance(target['roll_rate'], float)

    def test_decode_window_zero_restores_exact_bin_centres(self):
        """The old snapping behaviour stays reachable for comparison."""
        import config
        from race_obs import bin_centers

        old = getattr(config, 'BIN_DECODE_WINDOW', None)
        config.BIN_DECODE_WINDOW = 0
        try:
            target, _shared = self._run(context=False, chunk=1, bins=21)
        finally:
            if old is None:
                delattr(config, 'BIN_DECODE_WINDOW')
            else:
                config.BIN_DECODE_WINDOW = old
        centres = bin_centers(21)['roll_rate']
        self.assertTrue(
            any(abs(target['roll_rate'] - c) < 1e-6 for c in centres),
            f"roll {target['roll_rate']} is not a bin centre",
        )

    def test_categorical_with_chunk_and_context(self):
        target, _shared = self._run(context=True, chunk=5, bins=21)
        self.assertIsInstance(target['roll_rate'], float)

    def test_commands_stay_unrestricted_like_the_demonstrations(self):
        target, _shared = self._run(context=False, chunk=1)
        self.assertTrue(target['acro'])
        self.assertTrue(target['unrestricted_rates'])

    def test_context_mismatch_is_caught_loudly(self):
        """A context checkpoint fed a context-free observation must not run."""
        from policy_planner import PolicyPlanner

        with tempfile.TemporaryDirectory() as tmp:
            path = _checkpoint(Path(tmp), context=True, chunk=1)
            planner = PolicyPlanner(str(path))
            # Force the planner to build the narrow observation.
            planner._with_context = False
            with self.assertRaises(RuntimeError):
                planner.compute_target(_shared())

    def test_thrust_is_clamped_to_the_plant_range(self):
        import config

        target, _shared = self._run(context=False, chunk=1)
        self.assertGreaterEqual(target['thrust'], float(config.MIN_THRUST))
        self.assertLessEqual(target['thrust'], float(config.MAX_THRUST))

    def test_reset_episode_clears_history_and_plans(self):
        from policy_planner import PolicyPlanner

        with tempfile.TemporaryDirectory() as tmp:
            path = _checkpoint(Path(tmp), context=False, chunk=3)
            planner = PolicyPlanner(str(path))
            shared = _shared()
            for _ in range(4):
                planner.compute_target(shared)
            self.assertGreater(len(planner._buf), 0)
            planner.reset_episode()
            self.assertEqual(len(planner._buf), 0)
            self.assertEqual(len(planner._plans), 0)


if __name__ == '__main__':
    unittest.main()
