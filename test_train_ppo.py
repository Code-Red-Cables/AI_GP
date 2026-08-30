"""Unit tests for PPO reward / GAE (no sim)."""
from __future__ import annotations

import time
import types
import unittest

import numpy as np

from race_obs import KEYPOINT_COUNT, approach_potential, build_observation, visible_span
from tools.train_ppo import (
    GATE_REWARD,
    NEXT_GATE_BONUS,
    TIME_COST,
    _floor_hit,
    compute_gae,
    step_reward,
)


def _gate_box(cx: float, cy: float, w: float, h: float):
    hw, hh = w / 2.0, h / 2.0
    outer = [
        [cx - hw, cy - hh],
        [cx + hw, cy - hh],
        [cx + hw, cy + hh],
        [cx - hw, cy + hh],
    ]
    s = 0.55
    inner = [
        [cx - hw * s, cy - hh * s],
        [cx + hw * s, cy - hh * s],
        [cx + hw * s, cy + hh * s],
        [cx - hw * s, cy + hh * s],
    ]
    return outer + inner


def _obs(cx=320.0, cy=180.0, w=80.0, h=80.0, n_visible=8):
    pts = _gate_box(cx, cy, w, h)
    if n_visible < KEYPOINT_COUNT:
        for i in range(n_visible, KEYPOINT_COUNT):
            pts[i] = [0.0, 0.0]
    return build_observation(pts)


class RewardTests(unittest.TestCase):
    def test_sitting_still_is_only_time_cost(self):
        obs = _obs()
        r = step_reward(
            prev_gate=0, gate=0, finished=False, dt=0.05,
            prev_obs=obs, obs=obs,
        )
        self.assertAlmostEqual(r, -TIME_COST * 0.05, places=5)

    def test_staring_does_not_beat_a_pass(self):
        pad = _obs(w=80.0, h=80.0)
        sit = 0.0
        for _ in range(80):
            sit += step_reward(
                prev_gate=0, gate=0, finished=False, dt=0.05,
                prev_obs=pad, obs=pad,
            )
        passed = step_reward(
            prev_gate=0, gate=1, finished=False, dt=0.05,
            prev_obs=pad, obs=pad, new_target=True,
        )
        self.assertLess(sit, passed)
        self.assertGreater(passed, GATE_REWARD - 0.1)

    def test_approaching_centered_gate_is_positive(self):
        far = _obs(w=80.0, h=80.0)
        near = _obs(w=220.0, h=220.0)
        r = step_reward(
            prev_gate=0, gate=0, finished=False, dt=0.05,
            prev_obs=far, obs=near,
        )
        self.assertGreater(r, 0.3)
        self.assertGreater(approach_potential(near), approach_potential(far))

    def test_shrink_without_dropout_is_not_a_tax(self):
        near = _obs(w=220.0, h=220.0)
        far = _obs(w=80.0, h=80.0)
        away = step_reward(
            prev_gate=0, gate=0, finished=False, dt=0.05,
            prev_obs=near, obs=far,
        )
        sit = step_reward(
            prev_gate=0, gate=0, finished=False, dt=0.05,
            prev_obs=near, obs=near,
        )
        self.assertAlmostEqual(away, sit, places=4)

    def test_off_axis_approach_is_weaker_than_centered(self):
        far = _obs(cx=320.0, cy=180.0, w=80.0, h=80.0)
        center_near = _obs(cx=320.0, cy=180.0, w=200.0, h=200.0)
        side_near = _obs(cx=520.0, cy=180.0, w=200.0, h=200.0)
        r_center = step_reward(
            prev_gate=0, gate=0, finished=False, dt=0.05,
            prev_obs=far, obs=center_near,
        )
        r_side = step_reward(
            prev_gate=0, gate=0, finished=False, dt=0.05,
            prev_obs=far, obs=side_near,
        )
        self.assertGreater(r_center, r_side)

    def test_lost_lock_without_pass_is_penalized(self):
        seen = _obs()
        blank = _obs(n_visible=0)
        lost = step_reward(
            prev_gate=0, gate=0, finished=False, dt=0.05,
            prev_obs=seen, obs=blank,
        )
        sit = step_reward(
            prev_gate=0, gate=0, finished=False, dt=0.05,
            prev_obs=seen, obs=seen,
        )
        self.assertLess(lost, sit - 0.1)

    def test_lost_lock_on_a_pass_is_not_an_extra_cost(self):
        seen = _obs()
        blank = _obs(n_visible=0)
        r = step_reward(
            prev_gate=0, gate=1, finished=False, dt=0.05,
            prev_obs=seen, obs=blank, new_target=True,
        )
        self.assertAlmostEqual(r, GATE_REWARD - TIME_COST * 0.05, places=5)

    def test_close_lost_lock_is_a_punch_not_a_flyaway(self):
        close = _obs(w=280.0, h=280.0)
        blank = _obs(n_visible=0)
        r = step_reward(
            prev_gate=0, gate=0, finished=False, dt=0.05,
            prev_obs=close, obs=blank,
        )
        self.assertAlmostEqual(r, -TIME_COST * 0.05, places=5)

    def test_second_gate_pays_more_than_the_first(self):
        obs = _obs()
        g1 = step_reward(
            prev_gate=0, gate=1, finished=False, dt=0.05,
            prev_obs=obs, obs=obs, new_target=True,
        )
        g2 = step_reward(
            prev_gate=1, gate=2, finished=False, dt=0.05,
            prev_obs=obs, obs=obs, new_target=True,
        )
        self.assertGreater(g2, g1 + NEXT_GATE_BONUS - 0.05)

    def test_new_target_does_not_score_a_span_jump(self):
        small = _obs(w=80.0, h=80.0)
        huge = _obs(w=300.0, h=300.0)
        r = step_reward(
            prev_gate=0, gate=1, finished=False, dt=0.05,
            prev_obs=small, obs=huge, new_target=True,
        )
        self.assertAlmostEqual(r, GATE_REWARD - TIME_COST * 0.05, places=5)

    def test_crash_is_not_penalized(self):
        obs = _obs()
        kwargs = dict(
            prev_gate=0, gate=0, finished=False, dt=0.05,
            prev_obs=obs, obs=obs,
        )
        self.assertEqual(
            step_reward(crashed=True, **kwargs),
            step_reward(crashed=False, **kwargs),
        )

    def test_finish_bonus(self):
        obs = _obs()
        r = step_reward(
            prev_gate=16, gate=17, finished=True, dt=0.05,
            prev_obs=obs, obs=obs, new_target=True,
        )
        self.assertGreater(r, 12.0 + GATE_REWARD - 0.1)

    def test_outer_span_grows_when_the_box_grows(self):
        far = _obs(w=80.0, h=80.0)
        near = _obs(w=200.0, h=200.0)
        self.assertGreater(visible_span(near), visible_span(far))


class FloorHitTests(unittest.TestCase):
    def test_env_hit_after_grace_ends_the_episode(self):
        monitor = types.SimpleNamespace(grace_until=0.0)
        shared = {
            'collision': {
                'id': 1002,
                'impulse': 0.28,
                'ts': time.time_ns(),
            }
        }
        self.assertTrue(_floor_hit(shared, monitor, time.monotonic()))

    def test_grace_period_ignores_the_arming_thump(self):
        monitor = types.SimpleNamespace(grace_until=time.monotonic() + 2.0)
        shared = {
            'collision': {
                'id': 1002,
                'impulse': 0.28,
                'ts': time.time_ns(),
            }
        }
        self.assertFalse(_floor_hit(shared, monitor, time.monotonic()))


class GaeTests(unittest.TestCase):
    def test_length_and_finite(self):
        rewards = [1.0, 0.0, 0.0]
        values = [0.5, 0.4, 0.3]
        dones = [False, False, True]
        adv, ret = compute_gae(
            rewards, values, dones, gamma=0.99, lam=0.95, last_value=0.0,
        )
        self.assertEqual(len(adv), 3)
        self.assertTrue(np.isfinite(adv).all())
        self.assertTrue(np.isfinite(ret).all())


if __name__ == '__main__':
    unittest.main()
