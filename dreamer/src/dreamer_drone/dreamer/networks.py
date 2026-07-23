"""Encoders, decoders, and scalar heads for the DreamerV3 world model.

Image encoder/decoder adapt to arbitrary obs resolutions (the spatial size after the
conv stack is measured at construction, and the decoder interpolates its output to the
exact target H×W), so 64×64 / 96×64 / 128×72 all work without hand-tuned strides.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int, act=nn.SiLU) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), act()]
        d = hidden
    mods += [nn.Linear(d, out_dim)]
    return nn.Sequential(*mods)


class ImageEncoder(nn.Module):
    def __init__(self, image_hwc: tuple[int, int, int], depth: int, out_dim: int):
        super().__init__()
        h, w, c = image_hwc
        chans = [c, depth, 2 * depth, 4 * depth, 8 * depth]
        self.convs = nn.ModuleList()
        for i in range(4):
            self.convs.append(nn.Conv2d(chans[i], chans[i + 1], 4, stride=2, padding=1))
        self.act = nn.SiLU()
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            feat = self._conv_forward(dummy)
            self._enc_hw = (feat.shape[2], feat.shape[3])
            self._flat = feat.numel()
        self.fc = nn.Linear(self._flat, out_dim)
        self.out_dim = out_dim

    def _conv_forward(self, x: Tensor) -> Tensor:
        for conv in self.convs:
            x = self.act(conv(x))
        return x

    @property
    def enc_hw(self) -> tuple[int, int]:
        return self._enc_hw

    def forward(self, image_uint8: Tensor) -> Tensor:
        # image_uint8: (..., H, W, C) in [0,255]
        lead = image_uint8.shape[:-3]
        h, w, c = image_uint8.shape[-3:]
        x = image_uint8.reshape(-1, h, w, c).permute(0, 3, 1, 2).float() / 255.0 - 0.5
        x = self._conv_forward(x).reshape(x.shape[0], -1)
        x = self.fc(x)
        return x.reshape(*lead, self.out_dim)


class ImageDecoder(nn.Module):
    def __init__(self, feat_dim: int, image_hwc: tuple[int, int, int], depth: int,
                 enc_hw: tuple[int, int]):
        super().__init__()
        self.h, self.w, self.c = image_hwc
        self.depth = depth
        self.eh, self.ew = enc_hw
        self.fc = nn.Linear(feat_dim, 8 * depth * self.eh * self.ew)
        chans = [8 * depth, 4 * depth, 2 * depth, depth]
        self.deconvs = nn.ModuleList()
        for i in range(3):
            self.deconvs.append(nn.ConvTranspose2d(chans[i], chans[i + 1], 4, 2, 1))
        self.out = nn.ConvTranspose2d(depth, self.c, 4, 2, 1)
        self.act = nn.SiLU()

    def forward(self, feat: Tensor) -> Tensor:
        lead = feat.shape[:-1]
        x = self.fc(feat).reshape(-1, 8 * self.depth, self.eh, self.ew)
        for dc in self.deconvs:
            x = self.act(dc(x))
        x = self.out(x)  # (-1, C, eh*16, ew*16)
        x = F.interpolate(x, size=(self.h, self.w), mode="bilinear", align_corners=False)
        x = x.permute(0, 2, 3, 1)  # (-1, H, W, C)
        return x.reshape(*lead, self.h, self.w, self.c)


class VectorEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, layers: int):
        super().__init__()
        self.net = mlp(in_dim, hidden, out_dim, layers)
        self.out_dim = out_dim

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class VectorDecoder(nn.Module):
    """Predicts the vector obs mean; loss is computed in symlog space by the world model."""

    def __init__(self, feat_dim: int, hidden: int, out_dim: int, layers: int):
        super().__init__()
        self.net = mlp(feat_dim, hidden, out_dim, layers)

    def forward(self, feat: Tensor) -> Tensor:
        return self.net(feat)


class ScalarHead(nn.Module):
    """Two-hot symlog head: outputs logits over `bins` (decoded by TwoHotSymlog)."""

    def __init__(self, feat_dim: int, hidden: int, bins: int, layers: int):
        super().__init__()
        self.net = mlp(feat_dim, hidden, bins, layers)

    def forward(self, feat: Tensor) -> Tensor:
        return self.net(feat)


class BernoulliHead(nn.Module):
    """Continuation predictor: logits for P(episode continues)."""

    def __init__(self, feat_dim: int, hidden: int, layers: int):
        super().__init__()
        self.net = mlp(feat_dim, hidden, 1, layers)

    def forward(self, feat: Tensor) -> Tensor:
        return self.net(feat)
