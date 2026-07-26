"""Offline test of the decoupled collector<->learner machinery (no simulator).

Uses a FakeEnv matching the DroneRacingEnv reset/step contract to exercise: the collector
thread, thread-safe replay writes concurrent with learner reads, random->policy mode
switch, weight sync, stochastic inference, and a few real train_steps.
"""
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from dreamer_drone.config import Config
from dreamer_drone.dreamer.agent import DreamerAgent
from dreamer_drone.dreamer.replay import SequenceReplay
from dreamer_drone.env.spaces import ACTION_DIM, VECTOR_DIM
from dreamer_drone.train.collector import Collector


class FakeEnv:
    """Minimal stand-in: random legal obs, terminates after ~25 steps."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.pm = SimpleNamespace(relaunch=lambda: None)
        self._t = 0
        self._gate = 0

    def _obs(self):
        h, w = self.cfg.obs.image_h, self.cfg.obs.image_w
        return {
            "image": np.random.randint(0, 256, (h, w, 3), dtype=np.uint8),
            "vector": np.random.randn(VECTOR_DIM).astype(np.float32),
            "valid": np.ones(2, dtype=np.float32),
        }

    def reset(self, seed=None):
        self._t = 0
        self._gate = 0
        return self._obs(), {"stage": "hover"}

    def step(self, action):
        assert np.asarray(action).shape[-1] == ACTION_DIM
        self._t += 1
        if self._t % 10 == 0:
            self._gate += 1
        term = self._t >= 25
        info = {"active_gate": self._gate, "term_reason": "finish" if term else "",
                "stage": "hover"}
        return self._obs(), float(np.random.randn()), term, False, info

    def close(self):
        pass


def _tiny_cfg():
    cfg = Config()
    cfg.train.device = "cpu"
    cfg.train.collector_device = "cpu"
    cfg.train.max_episode_steps = 30
    cfg.obs.image_h = cfg.obs.image_w = 64
    m = cfg.model
    m.deter_dim, m.stoch_dim, m.stoch_classes, m.hidden = 32, 8, 8, 32
    m.cnn_depth, m.reward_bins, m.critic_bins = 8, 41, 41
    cfg.train.seq_len = 16
    cfg.train.batch_size = 4
    cfg.train.imag_horizon = 4
    return cfg


def test_collector_fills_replay_and_learner_trains():
    cfg = _tiny_cfg()
    replay = SequenceReplay(cfg.train.replay_capacity)
    agent = DreamerAgent(cfg, device="cpu")
    env = FakeEnv(cfg)
    collector = Collector(env, replay, cfg, device="cpu")

    collector.start()
    try:
        # random prefill phase
        t0 = time.time()
        while len(replay) < 60 and time.time() - t0 < 15:
            time.sleep(0.2)
        assert len(replay) >= 60, f"collector did not fill replay (got {len(replay)})"
        assert collector.env_steps > 0 and collector.episodes > 0

        # switch to policy mode with synced weights, keep collecting
        collector.sync(agent)
        collector.mode = "policy"
        steps_before = collector.env_steps
        time.sleep(1.0)
        assert collector.env_steps > steps_before, "policy-mode collection stalled"

        # learner trains on data collected concurrently
        assert replay.can_sample(cfg.train.seq_len)
        for _ in range(3):
            batch = replay.sample(cfg.train.batch_size, cfg.train.seq_len, agent.device)
            m = agent.train_step(batch)
            assert all(np.isfinite(v) for v in m.values())
        collector.sync(agent)  # push updated weights back to the collector copy
    finally:
        collector.stop()

    assert collector.thread is not None and not collector.thread.is_alive()
    print(f"env_steps={collector.env_steps} episodes={collector.episodes} "
          f"replay={len(replay)} last={collector.last_info}")


if __name__ == "__main__":
    test_collector_fills_replay_and_learner_trains()
    print("PASS test_collector_fills_replay_and_learner_trains")
    print("\nDecoupled trainer machinery verified (no sim).")
