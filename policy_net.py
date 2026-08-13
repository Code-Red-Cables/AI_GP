"""The vision student network: TCN over an observation history -> command.

Architecture follows paper1 (Xing et al., CoRL 2024) appendix A.3: a 3-layer
Temporal Convolutional Network encoding the observation history into a
temporal embedding of length 128, followed by a two-layer MLP that outputs the
four-dimensional command, collective thrust plus body rates.

Why a temporal model at all, rather than the per-frame MLP this repo tried
before: gate corners are frequently absent. Paper1 Fig. 10 shows stretches with
no usable corner projection whatsoever, and its Table 5 quantifies the cost of
having no memory -- H=4 gives 0% success, H=8 gives 84%, H=32 gives 100%. A
single-frame policy has to guess during every dropout, which is what the
earlier attempt here did with H=1.

The TCN is causal: output at time t depends only on inputs at or before t, so
the same weights run unchanged online with a rolling buffer.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from race_obs import DEFAULT_HISTORY, FEATURE_DIM, LABEL_DIM


class CausalConv1d(nn.Conv1d):
    """Conv1d that pads only on the left, so no future timestep leaks in."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        self._left_pad = (kernel_size - 1) * dilation
        super().__init__(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self._left_pad:
            x = nn.functional.pad(x, (self._left_pad, 0))
        return super().forward(x)


class TemporalBlock(nn.Module):
    """Dilated causal conv, ReLU, dropout, with a residual connection."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int, dropout: float):
        super().__init__()
        self.conv = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        # 1x1 projection only when the channel count changes.
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop(self.relu(self.conv(x)))
        res = x if self.down is None else self.down(x)
        return out + res


class RacePolicy(nn.Module):
    """Observation history -> ``chunk`` future [thrust, roll, pitch, yaw] steps.

    ``chunk > 1`` is action chunking: the network predicts a short burst of
    future commands instead of only the next one. Predicting a sequence forces
    the model to commit to a coherent short-term plan rather than re-deciding
    every frame, and it lets the planner average overlapping predictions, which
    is the standard remedy for the jitter-then-diverge failure of single-step
    behaviour cloning.
    """

    def __init__(
        self,
        n_in: int = FEATURE_DIM,
        n_out: int = LABEL_DIM,
        history: int = DEFAULT_HISTORY,
        channels: int = 128,
        kernel_size: int = 3,
        layers: int = 3,
        hidden: int = 128,
        dropout: float = 0.1,
        chunk: int = 1,
        bins: int = 0,
    ):
        super().__init__()
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.history = int(history)
        self.chunk = max(1, int(chunk))
        # bins > 0 switches the head from regression to per-channel
        # classification over discretised actions, so the output can commit to
        # a mode instead of averaging across them. See race_obs.ACTION_RANGES.
        self.bins = max(0, int(bins))

        blocks: list[nn.Module] = []
        in_ch = self.n_in
        for i in range(layers):
            # Exponential dilation so a 3-layer, kernel-3 stack sees 15 frames;
            # the final-timestep readout still benefits from the full window.
            blocks.append(
                TemporalBlock(in_ch, channels, kernel_size, 2 ** i, dropout)
            )
            in_ch = channels
        self.tcn = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(
                hidden,
                self.n_out * self.chunk * (self.bins if self.bins else 1),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is (batch, history, features).

        Regression head: (batch, n_out), or (batch, chunk, n_out).
        Categorical head: (batch, chunk, n_out, bins) logits — chunk axis kept
        even at chunk=1 so callers have one shape to handle.
        """
        if x.dim() != 3:
            raise ValueError(f'expected (batch, history, features), got {tuple(x.shape)}')
        # Conv1d wants (batch, channels, time).
        h = self.tcn(x.transpose(1, 2))
        # Read out the newest timestep only — this is a control policy, not a
        # sequence-to-sequence model.
        out = self.head(h[:, :, -1])
        if self.bins:
            return out.view(-1, self.chunk, self.n_out, self.bins)
        if self.chunk == 1:
            return out
        return out.view(-1, self.chunk, self.n_out)

    def arch(self) -> dict:
        return {
            'n_in': self.n_in,
            'n_out': self.n_out,
            'history': self.history,
            'chunk': self.chunk,
            'bins': self.bins,
        }


def save_policy(path, model: RacePolicy, *, extra: dict | None = None) -> None:
    blob = {
        'arch': model.arch(),
        'state_dict': model.state_dict(),
    }
    if extra:
        blob.update(extra)
    torch.save(blob, path)


def load_policy(path, map_location='cpu') -> tuple[RacePolicy, dict]:
    blob = torch.load(path, map_location=map_location, weights_only=False)
    arch = blob['arch']
    model = RacePolicy(
        n_in=arch['n_in'], n_out=arch['n_out'], history=arch['history'],
        chunk=int(arch.get('chunk', 1)),
        bins=int(arch.get('bins', 0)),
    )
    model.load_state_dict(blob['state_dict'])
    model.eval()
    return model, blob
