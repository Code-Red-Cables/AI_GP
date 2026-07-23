"""Deployment-clean inference controller.

Contains ONLY: observation preprocessing, the Dreamer encoder, the recurrent latent
update (RSSM posterior), the actor, action scaling, and watchdog/safety logic.

Compliance (asserted by tests/test_no_privileged_input.py): this module does NOT import
`sim.privileged_state`, `env.reward`, `env.termination`, or `env.curriculum`, and it
builds NO critic / decoder / reward head. It runs fully with every privileged service
disabled.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn

from ..config import Config
from ..dreamer.agent import Actor  # Actor only; pulls no privileged module
from ..dreamer.networks import ImageEncoder, VectorEncoder
from ..dreamer.rssm import RSSM
from ..env.observation_builder import build_obs
from ..env.spaces import ACTION_DIM, VECTOR_DIM, neutral_action
from ..sim.action_sender import ActionSender
from ..sim.camera_io import CameraIO
from ..sim.mavlink_io import MavlinkIO


class DeployPolicy(nn.Module):
    """Encoder + RSSM + Actor — the entire deployed network. Nothing else."""

    def __init__(self, cfg: Config):
        super().__init__()
        m = cfg.model
        img_hwc = (cfg.obs.image_h, cfg.obs.image_w, 1 if cfg.obs.grayscale else 3)
        self.img_enc = ImageEncoder(img_hwc, m.cnn_depth, m.hidden)
        self.vec_enc = VectorEncoder(VECTOR_DIM, m.hidden, m.hidden // 2, m.vector_layers)
        embed_dim = self.img_enc.out_dim + self.vec_enc.out_dim
        self.rssm = RSSM(ACTION_DIM, embed_dim, m.deter_dim, m.stoch_dim,
                         m.stoch_classes, m.hidden)
        self.actor = Actor(self.rssm.feature_dim, ACTION_DIM, m.hidden)

    def load_export(self, path: str, map_location="cpu") -> None:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        self.img_enc.load_state_dict(ckpt["img_enc"])
        self.vec_enc.load_state_dict(ckpt["vec_enc"])
        self.rssm.load_state_dict(ckpt["rssm"])
        self.actor.load_state_dict(ckpt["actor"])

    @torch.no_grad()
    def initial_state(self, device) -> dict[str, Tensor]:
        return self.rssm.initial(1, device)

    @torch.no_grad()
    def act(self, image: Tensor, vector: Tensor, state: dict, prev_action: Tensor,
            sample: bool = False):
        """Deterministic (mode) for deployment; `sample=True` draws a stochastic action for
        exploration during training collection."""
        embed = torch.cat([self.img_enc(image), self.vec_enc(vector)], dim=-1)
        post, _ = self.rssm.obs_step(state, prev_action, embed, sample=sample)
        feat = self.rssm.feature(post)
        action = self.actor.sample(feat)[0] if sample else self.actor.mode(feat)
        return action, post

    @torch.no_grad()
    def sync_from(self, agent) -> None:
        """Copy the live learner weights (encoder+RSSM+actor) into this inference policy.
        Cross-device copy is handled by load_state_dict; called by the learner between
        updates, so the source params are stable."""
        self.img_enc.load_state_dict(agent.wm.img_enc.state_dict())
        self.vec_enc.load_state_dict(agent.wm.vec_enc.state_dict())
        self.rssm.load_state_dict(agent.wm.rssm.state_dict())
        self.actor.load_state_dict(agent.actor.state_dict())


class DeployController:
    """Real-time deployment loop: newest causal frame in, action out, drop-stale, hold
    on missing data. Reset the recurrent state between race attempts."""

    def __init__(self, cfg: Config, checkpoint: str, device: Optional[str] = None):
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy = DeployPolicy(cfg).to(self.device).eval()
        self.policy.load_export(checkpoint, map_location=self.device)
        self.boot_ms = int(time.time() * 1000)

        self.io = MavlinkIO(cfg.sim.mavlink_host, cfg.sim.mavlink_port, self.boot_ms,
                            cfg.sim.reset_cmd_id)
        self.cam: Optional[CameraIO] = None
        self.sender: Optional[ActionSender] = None
        self.state: Optional[dict] = None
        self._prev_action = neutral_action()
        self._last_frame_id = -1
        self._last_sim_ns = 0
        self.latencies: list[float] = []

    def connect(self) -> bool:
        if not self.io.wait_heartbeat(self.cfg.sim.heartbeat_timeout_s):
            return False
        self.io.start()
        self.cam = CameraIO(self.cfg.sim.camera_bind, self.cfg.sim.camera_port).start()
        self.sender = ActionSender(self.io, self.cfg.action)
        self.reset_recurrent()
        return True

    def reset_recurrent(self) -> None:
        self.state = self.policy.initial_state(self.device)
        self._prev_action = neutral_action()
        self._last_frame_id = -1
        if self.sender:
            self.sender.reset()

    def _to_tensor(self, obs: dict) -> tuple[Tensor, Tensor]:
        image = torch.from_numpy(obs["image"]).unsqueeze(0).to(self.device)
        vector = torch.from_numpy(obs["vector"]).unsqueeze(0).to(self.device)
        return image, vector

    def step_once(self) -> Optional[np.ndarray]:
        """Process the newest causal frame if fresh; return the action sent, or None if
        the frame was stale (held/neutral applied)."""
        assert self.cam and self.sender
        f = self.cam.get_latest()
        if f is None or f.frame_id == self._last_frame_id:
            # stale: hold last action decaying to neutral (safety)
            self.sender.send_neutral()
            return None
        dt = (f.sim_time_ns - self._last_sim_ns) / 1e9 if self._last_sim_ns else 0.0
        self._last_frame_id = f.frame_id
        self._last_sim_ns = f.sim_time_ns

        t0 = time.perf_counter()
        obs = build_obs(
            frame_bgr=f.image_bgr,
            imu=self.io.get("highres_imu"),
            prev_action_norm=self._prev_action, dt=dt if dt > 0 else 1.0 / self.cfg.action.control_hz,
            cfg=self.cfg.obs,
        )
        image, vector = self._to_tensor(obs)
        prev_a = torch.from_numpy(self._prev_action).unsqueeze(0).to(self.device)
        action, self.state = self.policy.act(image, vector, self.state, prev_a)
        self.latencies.append((time.perf_counter() - t0) * 1000.0)
        a = action.squeeze(0).cpu().numpy().astype(np.float32)
        applied = self.sender.send(a)
        self._prev_action = applied
        return applied

    def run(self, arm: bool = True, max_seconds: float = 480.0) -> None:
        if not self.connect():
            raise RuntimeError("no heartbeat; is the sim running?")
        if arm:
            self.io.arm()
        t_end = time.time() + max_seconds
        period = 1.0 / self.cfg.action.control_hz
        while time.time() < t_end:
            t0 = time.time()
            self.step_once()
            rem = period - (time.time() - t0)
            if rem > 0:
                time.sleep(rem)

    def close(self) -> None:
        if self.sender:
            self.sender.send_neutral()
        if self.cam:
            self.cam.stop()
        self.io.stop()
