"""DreamerV3 distribution utilities: symlog/symexp transforms, two-hot symlog
regression head for reward & value, straight-through categorical latents, and a
tanh-squashed Normal for continuous actions.

References the DreamerV3 design (Hafner et al. 2023): symlog prediction targets,
two-hot encoded scalar heads, discrete stochastic latents with unimix, and free-bits KL.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal


def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


class TwoHotSymlog:
    """Two-hot encoded, symlog-transformed scalar distribution.

    `logits` shape (..., num_bins) predict a scalar in symlog space over a fixed support.
    `.mean()` decodes to real space via symexp; `.log_prob(target)` is the cross-entropy
    against the two-hot encoding of symlog(target).
    """

    def __init__(self, logits: Tensor, low: float = -20.0, high: float = 20.0):
        self.logits = logits
        self.num_bins = logits.shape[-1]
        self.support = torch.linspace(low, high, self.num_bins,
                                      device=logits.device, dtype=logits.dtype)

    def mean(self) -> Tensor:
        probs = F.softmax(self.logits, dim=-1)
        return symexp((probs * self.support).sum(dim=-1, keepdim=True))

    def log_prob(self, target: Tensor) -> Tensor:
        # target: (..., 1) real-valued
        y = symlog(target).squeeze(-1)                       # (...)
        y = y.clamp(self.support[0], self.support[-1])
        # locate the bracketing bins
        idx = torch.searchsorted(self.support, y.contiguous())
        below = (idx - 1).clamp(0, self.num_bins - 1)
        above = idx.clamp(0, self.num_bins - 1)
        s_below = self.support[below]
        s_above = self.support[above]
        span = (s_above - s_below)
        w_above = torch.where(span > 0, (y - s_below) / span, torch.zeros_like(y))
        w_below = 1.0 - w_above
        two_hot = torch.zeros_like(self.logits)
        two_hot.scatter_(-1, below.unsqueeze(-1), w_below.unsqueeze(-1))
        two_hot.scatter_add_(-1, above.unsqueeze(-1), w_above.unsqueeze(-1))
        log_probs = F.log_softmax(self.logits, dim=-1)
        return (two_hot * log_probs).sum(dim=-1, keepdim=True)


class OneHotCategoricalST:
    """Categorical with straight-through gradients and unimix, for discrete latents.

    `logits` shape (..., num_classes). Sampling returns one-hot with the softmax gradient
    passed through. `unimix` blends a uniform to keep probabilities from collapsing.
    """

    def __init__(self, logits: Tensor, unimix: float = 0.01):
        probs = F.softmax(logits, dim=-1)
        if unimix > 0:
            uniform = torch.ones_like(probs) / probs.shape[-1]
            probs = (1 - unimix) * probs + unimix * uniform
            logits = torch.log(probs + 1e-8)
        self.logits = logits
        self.probs = probs

    def sample(self) -> Tensor:
        shape = self.probs.shape
        flat = self.probs.reshape(-1, shape[-1])
        idx = torch.multinomial(flat, 1).reshape(shape[:-1])
        onehot = F.one_hot(idx, shape[-1]).to(self.probs.dtype)
        # straight-through estimator
        return onehot + (self.probs - self.probs.detach())

    def mode(self) -> Tensor:
        idx = self.probs.argmax(dim=-1)
        onehot = F.one_hot(idx, self.probs.shape[-1]).to(self.probs.dtype)
        return onehot + (self.probs - self.probs.detach())

    def entropy(self) -> Tensor:
        return -(self.probs * torch.log(self.probs + 1e-8)).sum(-1)


def categorical_kl(post_logits: Tensor, prior_logits: Tensor) -> Tensor:
    """KL(posterior || prior) for independent categoricals, summed over latent vars."""
    post = F.softmax(post_logits, dim=-1)
    log_post = F.log_softmax(post_logits, dim=-1)
    log_prior = F.log_softmax(prior_logits, dim=-1)
    return (post * (log_post - log_prior)).sum(-1).sum(-1)   # sum over classes then vars


class TanhNormal:
    """Tanh-squashed diagonal Normal → continuous actions in (-1, 1)."""

    def __init__(self, mean: Tensor, std: Tensor):
        self.base = Normal(mean, std)

    def sample_with_logprob(self) -> tuple[Tensor, Tensor]:
        pre = self.base.rsample()
        action = torch.tanh(pre)
        logp = self.base.log_prob(pre) - torch.log(1 - action.pow(2) + 1e-6)
        return action, logp.sum(-1, keepdim=True)

    def log_prob(self, action: Tensor) -> Tensor:
        """Log-prob of a given action in (-1,1) — used for behavior cloning on demos."""
        a = action.clamp(-0.999, 0.999)
        pre = 0.5 * (torch.log1p(a) - torch.log1p(-a))   # atanh
        logp = self.base.log_prob(pre) - torch.log(1 - a.pow(2) + 1e-6)
        return logp.sum(-1, keepdim=True)

    def mode(self) -> Tensor:
        return torch.tanh(self.base.mean)

    def entropy(self) -> Tensor:
        # base-Normal entropy is a smooth, well-behaved bonus target
        return self.base.entropy().sum(-1, keepdim=True)
