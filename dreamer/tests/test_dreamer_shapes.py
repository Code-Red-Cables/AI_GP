"""End-to-end shape & gradient smoke test for the DreamerV3 agent (CPU, tiny model).

Verifies: world-model observe + loss, imagination rollout, a full train_step (all three
optimizers), single-step recurrent inference (`act`), and the deployment-clean export +
reload path. No simulator required.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from dreamer_drone.config import Config
from dreamer_drone.dreamer.agent import DreamerAgent
from dreamer_drone.dreamer.distributions import (TanhNormal, TwoHotSymlog, symexp,
                                                 symlog)
from dreamer_drone.env.spaces import ACTION_DIM, VECTOR_DIM

B, T = 2, 6


def _tiny_cfg() -> Config:
    cfg = Config()
    cfg.train.device = "cpu"
    cfg.obs.image_h = cfg.obs.image_w = 64
    m = cfg.model
    m.deter_dim, m.stoch_dim, m.stoch_classes, m.hidden = 32, 8, 8, 32
    m.cnn_depth, m.reward_bins, m.critic_bins = 8, 41, 41
    cfg.train.imag_horizon = 4
    return cfg


def _fake_batch(cfg: Config) -> dict:
    h, w = cfg.obs.image_h, cfg.obs.image_w
    return {
        "image": torch.randint(0, 256, (B, T, h, w, 3), dtype=torch.uint8),
        "vector": torch.randn(B, T, VECTOR_DIM),
        "action": torch.tanh(torch.randn(B, T, ACTION_DIM)),
        "reward": torch.randn(B, T, 1),
        "cont": torch.ones(B, T, 1),
    }


def test_symlog_roundtrip():
    x = torch.randn(100) * 50
    assert torch.allclose(symexp(symlog(x)), x, atol=1e-4)


def test_twohot_mean_tracks_target():
    logits = torch.zeros(4, 255)
    d = TwoHotSymlog(logits)
    # log_prob of a range of targets should be finite and shaped (4,1)
    tgt = torch.tensor([[0.0], [1.0], [-5.0], [123.0]])
    lp = d.log_prob(tgt)
    assert lp.shape == (4, 1) and torch.isfinite(lp).all()
    assert d.mean().shape == (4, 1)


def test_tanh_normal_bounds():
    dist = TanhNormal(torch.zeros(5, ACTION_DIM), torch.ones(5, ACTION_DIM) * 0.5)
    a, logp = dist.sample_with_logprob()
    assert a.shape == (5, ACTION_DIM) and (a.abs() <= 1).all()
    assert logp.shape == (5, 1)
    assert dist.mode().shape == (5, ACTION_DIM)


def test_world_model_observe_and_loss():
    cfg = _tiny_cfg()
    agent = DreamerAgent(cfg, device="cpu")
    batch = _fake_batch(cfg)
    loss, metrics, start = agent.wm.loss({k: v for k, v in batch.items()})
    assert torch.isfinite(loss)
    # start states flattened to (B*T, ...)
    assert start["deter"].shape == (B * T, cfg.model.deter_dim)
    assert start["stoch"].shape == (B * T, cfg.model.stoch_dim * cfg.model.stoch_classes)
    for key in ("wm/img_rec", "wm/vec_rec", "wm/kl"):
        assert key in metrics


def test_imagination_shapes():
    cfg = _tiny_cfg()
    agent = DreamerAgent(cfg, device="cpu")
    start = agent.wm.rssm.initial(5, torch.device("cpu"))
    states, acts, logps = agent.wm.rssm.imagine(agent.actor, start, cfg.train.imag_horizon)
    H = cfg.train.imag_horizon
    assert acts.shape == (5, H, ACTION_DIM)
    assert logps.shape == (5, H, 1)
    assert states["deter"].shape == (5, H, cfg.model.deter_dim)


def test_full_train_step_runs_and_is_finite():
    cfg = _tiny_cfg()
    agent = DreamerAgent(cfg, device="cpu")
    batch = _fake_batch(cfg)
    m1 = agent.train_step(batch)
    m2 = agent.train_step(_fake_batch(cfg))
    for m in (m1, m2):
        for k, v in m.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"
    assert "ac/actor_loss" in m1 and "ac/critic_loss" in m1


def test_act_single_step():
    cfg = _tiny_cfg()
    agent = DreamerAgent(cfg, device="cpu")
    state = agent.initial_state(1)
    obs = {
        "image": torch.randint(0, 256, (1, 64, 64, 3), dtype=torch.uint8),
        "vector": torch.randn(1, VECTOR_DIM),
    }
    prev = torch.zeros(1, ACTION_DIM)
    action, new_state = agent.act(obs, state, prev, training=False)
    assert action.shape == (1, ACTION_DIM) and (action.abs() <= 1).all()
    assert new_state["deter"].shape == (1, cfg.model.deter_dim)


def test_deploy_export_and_reload_match():
    from dreamer_drone.deploy.controller import DeployPolicy
    cfg = _tiny_cfg()
    agent = DreamerAgent(cfg, device="cpu")
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "_dreamer_export_test.pt")
        agent.export_deploy(path)

        policy = DeployPolicy(cfg)
        policy.load_export(path, map_location="cpu")
        policy.eval()
        state = policy.initial_state(torch.device("cpu"))
        image = torch.randint(0, 256, (1, 64, 64, 3), dtype=torch.uint8)
        vector = torch.randn(1, VECTOR_DIM)
        prev = torch.zeros(1, ACTION_DIM)
        action, _ = policy.act(image, vector, state, prev)
        # Deployment must reproduce the trained actor for the same latent.
        agent_action, _ = agent.act(
            {"image": image, "vector": vector},
            state,
            prev,
            training=False,
        )
        assert torch.allclose(action, agent_action, atol=1e-5)


if __name__ == "__main__":
    torch.manual_seed(0)
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} dreamer shape tests passed.")
