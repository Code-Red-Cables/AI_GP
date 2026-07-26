"""Recurrent State-Space Model (RSSM) with a deterministic GRU path and discrete
(categorical) stochastic latents, per DreamerV3.

State is a dict:
    deter:  (..., deter_dim)                    GRU hidden state h
    stoch:  (..., stoch_dim * stoch_classes)    flattened one-hot sample z
    logits: (..., stoch_dim, stoch_classes)     categorical parameters (for KL)

`feature(state)` = concat(deter, stoch) is the input to every head and the actor/critic.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .distributions import OneHotCategoricalST
from .networks import mlp


class RSSM(nn.Module):
    def __init__(self, action_dim: int, embed_dim: int, deter_dim: int,
                 stoch_dim: int, stoch_classes: int, hidden: int, unimix: float = 0.01):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes
        self.stoch_flat = stoch_dim * stoch_classes
        self.unimix = unimix

        self._img_in = nn.Sequential(
            nn.Linear(self.stoch_flat + action_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self._cell = nn.GRUCell(hidden, deter_dim)
        self._prior = mlp(deter_dim, hidden, self.stoch_flat, layers=1)
        self._post = mlp(deter_dim + embed_dim, hidden, self.stoch_flat, layers=1)

    @property
    def feature_dim(self) -> int:
        return self.deter_dim + self.stoch_flat

    def initial(self, batch: int, device) -> dict[str, Tensor]:
        return {
            "deter": torch.zeros(batch, self.deter_dim, device=device),
            "stoch": torch.zeros(batch, self.stoch_flat, device=device),
            "logits": torch.zeros(batch, self.stoch_dim, self.stoch_classes, device=device),
        }

    def feature(self, state: dict[str, Tensor]) -> Tensor:
        return torch.cat([state["deter"], state["stoch"]], dim=-1)

    def _to_state(self, deter: Tensor, logits_flat: Tensor, sample: bool) -> dict[str, Tensor]:
        logits = logits_flat.reshape(*logits_flat.shape[:-1], self.stoch_dim, self.stoch_classes)
        dist = OneHotCategoricalST(logits, unimix=self.unimix)
        stoch = dist.sample() if sample else dist.mode()
        stoch = stoch.reshape(*stoch.shape[:-2], self.stoch_flat)
        return {"deter": deter, "stoch": stoch, "logits": logits}

    def img_step(self, prev: dict[str, Tensor], prev_action: Tensor,
                 sample: bool = True) -> dict[str, Tensor]:
        x = torch.cat([prev["stoch"], prev_action], dim=-1)
        x = self._img_in(x)
        deter = self._cell(x, prev["deter"])
        return self._to_state(deter, self._prior(deter), sample)

    def obs_step(self, prev: dict[str, Tensor], prev_action: Tensor, embed: Tensor,
                 sample: bool = True) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        prior = self.img_step(prev, prev_action, sample)
        x = torch.cat([prior["deter"], embed], dim=-1)
        post = self._to_state(prior["deter"], self._post(x), sample)
        return post, prior

    def observe(self, embeds: Tensor, actions: Tensor,
                init: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Roll posteriors over time.
        embeds:  (B, T, embed)   actions: (B, T, A) — actions[t] led to embeds[t].
        Returns stacked (posts, priors), each dict of (B, T, ...)."""
        B, T = embeds.shape[0], embeds.shape[1]
        state = init
        posts, priors = [], []
        for t in range(T):
            state, prior = self.obs_step(state, actions[:, t], embeds[:, t])
            posts.append(state)
            priors.append(prior)
        return self._stack(posts), self._stack(priors)

    def imagine(self, actor, init: dict[str, Tensor], horizon: int):
        """Roll priors forward using `actor` on the latent feature.
        Returns (states over horizon, actions, action_logprobs)."""
        state = init
        states, acts, logps = [], [], []
        for _ in range(horizon):
            feat = self.feature(state)
            action, logp = actor.sample(feat)
            state = self.img_step(state, action)
            states.append(state)
            acts.append(action)
            logps.append(logp)
        return self._stack(states), torch.stack(acts, 1), torch.stack(logps, 1)

    @staticmethod
    def _stack(seq: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        return {k: torch.stack([s[k] for s in seq], dim=1) for k in seq[0]}
