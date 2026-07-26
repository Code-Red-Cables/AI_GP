"""DreamerV3 agent: world model + actor + critic, trained in imagination.

Batch convention (see docs/system_architecture.md):
    image  (B,T,H,W,C) uint8   vector (B,T,Dv)   action (B,T,A)   reward (B,T,1)
    cont   (B,T,1) = 1-done     [optional] aux (B,T,Da) privileged auxiliary targets
`action[:,t]` is the action executed at step t; the RSSM is fed the previous action
(shifted, zero at t=0) alongside obs t. Reward/continue heads predict the step-t values
from the posterior feature at t.

Auxiliary privileged heads read the *latent feature* (built only from legal obs+action)
to shape the representation. They never feed privileged values INTO the latent, so the
actor — which reads the same latent — remains deployment-clean.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import Config
from ..env.spaces import ACTION_DIM, VECTOR_DIM
from .distributions import (TanhNormal, TwoHotSymlog, categorical_kl, symexp,
                            symlog)
from .networks import (BernoulliHead, ImageDecoder, ImageEncoder, ScalarHead,
                       VectorDecoder, VectorEncoder, mlp)
from .rssm import RSSM


# --------------------------------------------------------------------------- #
class Actor(nn.Module):
    def __init__(self, feat_dim: int, action_dim: int, hidden: int, layers: int = 2,
                 min_std: float = 0.02, max_std: float = 1.0):
        # min_std lowered 0.1 -> 0.02 (2026-07-25): the demo actions live in a ±0.02-0.04
        # band, so with a 0.1 noise floor the BC log-prob saturated at its ceiling (~-5.4
        # observed vs ~-6.4 theoretical max) while sampled flight still random-walked off
        # the demo corridor within ~25 steps and spun out. Precision hover control needs
        # a tighter floor than DreamerV3's Atari-bred default.
        super().__init__()
        self.net = mlp(feat_dim, hidden, 2 * action_dim, layers)
        self.action_dim = action_dim
        self.min_std, self.max_std = min_std, max_std

    def _dist(self, feat: Tensor) -> TanhNormal:
        out = self.net(feat)
        mean, std_raw = out.chunk(2, dim=-1)
        std = self.min_std + (self.max_std - self.min_std) * torch.sigmoid(std_raw)
        return TanhNormal(mean, std)

    def sample(self, feat: Tensor) -> tuple[Tensor, Tensor]:
        return self._dist(feat).sample_with_logprob()

    def mode(self, feat: Tensor) -> Tensor:
        return self._dist(feat).mode()

    def entropy(self, feat: Tensor) -> Tensor:
        return self._dist(feat).entropy()

    def log_prob(self, feat: Tensor, action: Tensor) -> Tensor:
        return self._dist(feat).log_prob(action)


class Critic(nn.Module):
    def __init__(self, feat_dim: int, hidden: int, bins: int, layers: int = 2):
        super().__init__()
        self.head = ScalarHead(feat_dim, hidden, bins, layers)

    def dist(self, feat: Tensor) -> TwoHotSymlog:
        return TwoHotSymlog(self.head(feat))

    def value(self, feat: Tensor) -> Tensor:
        return self.dist(feat).mean()


class WorldModel(nn.Module):
    def __init__(self, cfg: Config, aux_dim: int = 0):
        super().__init__()
        m = cfg.model
        img_hwc = (cfg.obs.image_h, cfg.obs.image_w, 1 if cfg.obs.grayscale else 3)
        self.img_enc = ImageEncoder(img_hwc, m.cnn_depth, m.hidden)
        self.vec_enc = VectorEncoder(VECTOR_DIM, m.hidden, m.hidden // 2, m.vector_layers)
        embed_dim = self.img_enc.out_dim + self.vec_enc.out_dim
        self.rssm = RSSM(ACTION_DIM, embed_dim, m.deter_dim, m.stoch_dim,
                         m.stoch_classes, m.hidden)
        feat = self.rssm.feature_dim
        self.img_dec = ImageDecoder(feat, img_hwc, m.cnn_depth, self.img_enc.enc_hw)
        self.vec_dec = VectorDecoder(feat, m.hidden, VECTOR_DIM, m.vector_layers)
        self.reward_head = ScalarHead(feat, m.hidden, m.reward_bins, layers=1)
        self.cont_head = BernoulliHead(feat, m.hidden, layers=1)
        self.aux_head = (mlp(feat, m.hidden, aux_dim, layers=1) if aux_dim > 0 else None)
        self.cfg = cfg
        self.feat_dim = feat

    def encode(self, image: Tensor, vector: Tensor) -> Tensor:
        return torch.cat([self.img_enc(image), self.vec_enc(vector)], dim=-1)

    def loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict, dict]:
        m = self.cfg.model
        image, vector = batch["image"], batch["vector"]
        action, reward, cont = batch["action"], batch["reward"], batch["cont"]
        B, T = image.shape[0], image.shape[1]
        device = image.device

        embeds = self.encode(image, vector)
        prev_actions = torch.cat([torch.zeros_like(action[:, :1]), action[:, :-1]], dim=1)
        posts, priors = self.rssm.observe(embeds, prev_actions,
                                          self.rssm.initial(B, device))
        feats = self.rssm.feature(posts)                       # (B,T,F)

        # reconstruction
        img_target = image.float() / 255.0 - 0.5
        img_rec = F.mse_loss(self.img_dec(feats), img_target)
        vec_rec = F.mse_loss(self.vec_dec(feats), symlog(vector))
        # reward / continue
        rew_lp = TwoHotSymlog(self.reward_head(feats)).log_prob(reward).mean()
        cont_lp = -F.binary_cross_entropy_with_logits(self.cont_head(feats), cont)
        # KL (balanced, free-bits)
        kl_dyn = categorical_kl(posts["logits"].detach(), priors["logits"])
        kl_rep = categorical_kl(posts["logits"], priors["logits"].detach())
        free = m.free_bits
        kl = (m.kl_balance * kl_dyn.clamp(min=free).mean()
              + (1 - m.kl_balance) * kl_rep.clamp(min=free).mean())

        loss = img_rec + vec_rec - rew_lp - cont_lp + kl
        metrics = {
            "wm/img_rec": img_rec.item(), "wm/vec_rec": vec_rec.item(),
            "wm/reward_lp": rew_lp.item(), "wm/cont_lp": cont_lp.item(),
            "wm/kl": kl.item(),
        }
        if self.aux_head is not None and "aux" in batch:
            aux_loss = F.mse_loss(self.aux_head(feats), symlog(batch["aux"]))
            loss = loss + aux_loss
            metrics["wm/aux"] = aux_loss.item()
        # flatten posterior states as imagination start points (detached)
        start = {k: v.reshape(B * T, *v.shape[2:]).detach() for k, v in posts.items()}
        return loss, metrics, start


def lambda_return(reward: Tensor, value: Tensor, discount: Tensor, lam: float) -> Tensor:
    """Generalized λ-return over the horizon axis (dim=1).
    reward,value,discount: (N, H, 1). Returns (N, H, 1)."""
    H = reward.shape[1]
    returns = [None] * H
    last = value[:, -1]
    for t in reversed(range(H)):
        bootstrap = value[:, t + 1] if t + 1 < H else last
        returns[t] = reward[:, t] + discount[:, t] * ((1 - lam) * bootstrap +
                                                      lam * (returns[t + 1] if t + 1 < H else last))
        last = returns[t]
    return torch.stack(returns, dim=1)


class DreamerAgent:
    def __init__(self, cfg: Config, aux_dim: int = 0, device: Optional[str] = None):
        self.cfg = cfg
        self.device = torch.device(device or (
            "cuda" if torch.cuda.is_available() and cfg.train.device == "cuda" else "cpu"))
        self.wm = WorldModel(cfg, aux_dim=aux_dim).to(self.device)
        self.actor = Actor(self.wm.feat_dim, ACTION_DIM, cfg.model.hidden).to(self.device)
        self.critic = Critic(self.wm.feat_dim, cfg.model.hidden, cfg.model.critic_bins).to(self.device)
        self.target_critic = Critic(self.wm.feat_dim, cfg.model.hidden, cfg.model.critic_bins).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        t = cfg.train
        self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=t.lr_model)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=t.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=t.lr_critic)
        self._ret_std = 1.0  # EMA of return scale for actor normalization

    # ---- training ----------------------------------------------------------
    def train_step(self, batch: dict[str, Tensor], demo_batch: Optional[dict] = None,
                   bc_weight: float = 0.0) -> dict:
        batch = {k: v.to(self.device) for k, v in batch.items()}
        # 1) world model
        wm_loss, metrics, start = self.wm.loss(batch)
        self.opt_wm.zero_grad(set_to_none=True)
        wm_loss.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), self.cfg.train.grad_clip)
        self.opt_wm.step()

        # 2) imagine + actor-critic (with optional behavior-cloning on demos)
        ac_metrics = self._actor_critic(start, demo_batch, bc_weight)
        metrics.update(ac_metrics)
        self._update_target()
        metrics["wm/loss"] = wm_loss.item()
        return metrics

    def _bc_loss(self, demo_batch: dict[str, Tensor]) -> Tensor:
        """Behavior-cloning: make the actor imitate the (gate-passing) demo actions.
        World-model feats are detached, so BC trains only the actor, not the WM."""
        demo = {k: v.to(self.device) for k, v in demo_batch.items()}
        B = demo["image"].shape[0]
        with torch.no_grad():
            embeds = self.wm.encode(demo["image"], demo["vector"])
            prev_a = torch.cat([torch.zeros_like(demo["action"][:, :1]),
                                demo["action"][:, :-1]], dim=1)
            posts, _ = self.wm.rssm.observe(embeds, prev_a, self.wm.rssm.initial(B, self.device))
            feats = self.wm.rssm.feature(posts)
        return -self.actor.log_prob(feats, demo["action"]).mean()

    def _actor_critic(self, start: dict[str, Tensor], demo_batch: Optional[dict] = None,
                      bc_weight: float = 0.0) -> dict:
        t = self.cfg.train
        states, actions, logps = self.wm.rssm.imagine(self.actor, start, t.imag_horizon)
        feats = self.wm.rssm.feature(states)                     # (N,H,F)
        reward = TwoHotSymlog(self.wm.reward_head(feats)).mean()
        cont = torch.sigmoid(self.wm.cont_head(feats))
        discount = t.gamma * cont
        with torch.no_grad():
            value_t = self.target_critic.value(feats)
        returns = lambda_return(reward, value_t, discount, t.lambda_)

        # actor: maximize normalized returns + entropy (pathwise through dynamics)
        ret = returns.detach()
        scale = ret.std().clamp(min=1.0).item()
        self._ret_std = 0.99 * self._ret_std + 0.01 * scale
        adv = (returns - value_t) / self._ret_std
        entropy = self.actor.entropy(feats.detach())
        actor_loss = -(adv).mean() - t.actor_entropy * entropy.mean()
        bc_val = 0.0
        if demo_batch is not None and bc_weight > 0:
            bc = self._bc_loss(demo_batch)
            actor_loss = actor_loss + bc_weight * bc
            bc_val = bc.item()
        self.opt_actor.zero_grad(set_to_none=True)
        self.opt_wm.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), t.grad_clip)
        self.opt_actor.step()

        # critic: two-hot regression to λ-returns
        v_dist = self.critic.dist(feats.detach())
        critic_loss = -v_dist.log_prob(returns.detach()).mean()
        self.opt_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), t.grad_clip)
        self.opt_critic.step()
        self.opt_wm.zero_grad(set_to_none=True)  # discard imagination grads on WM

        return {
            "ac/actor_loss": actor_loss.item(),
            "ac/critic_loss": critic_loss.item(),
            "ac/return_mean": returns.mean().item(),
            "ac/entropy": entropy.mean().item(),
            "ac/bc_loss": bc_val,
        }

    def _update_target(self, tau: float = 0.02) -> None:
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
                tp.mul_(1 - tau).add_(tau * p)

    # ---- inference (used by eval + deploy) ---------------------------------
    @torch.no_grad()
    def initial_state(self, batch: int = 1) -> dict[str, Tensor]:
        return self.wm.rssm.initial(batch, self.device)

    @torch.no_grad()
    def act(self, obs: dict[str, Tensor], state: dict[str, Tensor], prev_action: Tensor,
            training: bool = False) -> tuple[Tensor, dict[str, Tensor]]:
        """One-step recurrent inference. `obs` tensors are (B,...) for the current frame."""
        embed = self.wm.encode(obs["image"], obs["vector"])
        post, _ = self.wm.rssm.obs_step(state, prev_action, embed, sample=training)
        feat = self.wm.rssm.feature(post)
        action = self.actor.sample(feat)[0] if training else self.actor.mode(feat)
        return action, post

    # ---- checkpoint --------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({
            "wm": self.wm.state_dict(), "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(), "config": self.cfg.to_dict(),
        }, path)

    def load(self, path: str, actor_only: bool = False) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.wm.load_state_dict(ckpt["wm"])
        self.actor.load_state_dict(ckpt["actor"])
        if not actor_only and "critic" in ckpt:
            self.critic.load_state_dict(ckpt["critic"])
            self.target_critic.load_state_dict(ckpt["critic"])

    def export_deploy(self, path: str) -> None:
        """Save ONLY the deployment-clean forward path (encoders + RSSM + actor).
        No decoder, critic, reward head, or optimizer — see deploy/controller.py."""
        torch.save({
            "img_enc": self.wm.img_enc.state_dict(),
            "vec_enc": self.wm.vec_enc.state_dict(),
            "rssm": self.wm.rssm.state_dict(),
            "actor": self.actor.state_dict(),
            "config": self.cfg.to_dict(),
        }, path)
