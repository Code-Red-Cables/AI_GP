"""Sequence-friendly replay buffer.

Stores whole episodes as contiguous numpy arrays (images kept uint8 for compactness) and
samples fixed-length (B, T) windows for the world model. Deployment observations only —
privileged reward is already baked into the stored `reward`/`cont`; no privileged field
is stored in a way the actor could read (the actor only ever sees image+vector).
"""
from __future__ import annotations

import glob
import os
import random
import threading
from collections import deque
from typing import Optional

import numpy as np
import torch

_EP_KEYS = ("image", "vector", "action", "reward", "cont", "aux")


class EpisodeAccumulator:
    """Collects per-step transitions for one episode, then hands off a dict of arrays."""

    def __init__(self):
        self._buf: dict[str, list] = {}

    def add(self, image, vector, action, reward, cont, aux=None) -> None:
        self._buf.setdefault("image", []).append(np.asarray(image, dtype=np.uint8))
        self._buf.setdefault("vector", []).append(np.asarray(vector, dtype=np.float32))
        self._buf.setdefault("action", []).append(np.asarray(action, dtype=np.float32))
        self._buf.setdefault("reward", []).append(np.float32(reward))
        self._buf.setdefault("cont", []).append(np.float32(cont))
        if aux is not None:
            self._buf.setdefault("aux", []).append(np.asarray(aux, dtype=np.float32))

    def __len__(self) -> int:
        return len(self._buf.get("image", []))

    def flush(self) -> Optional[dict]:
        if len(self) == 0:
            return None
        ep = {
            "image": np.stack(self._buf["image"]),
            "vector": np.stack(self._buf["vector"]),
            "action": np.stack(self._buf["action"]),
            "reward": np.asarray(self._buf["reward"], dtype=np.float32)[:, None],
            "cont": np.asarray(self._buf["cont"], dtype=np.float32)[:, None],
        }
        if "aux" in self._buf:
            ep["aux"] = np.stack(self._buf["aux"])
        self._buf = {}
        return ep


class SequenceReplay:
    """Thread-safe: a background collector calls add_episode while the learner samples."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._episodes: deque[dict] = deque()
        self._transitions = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return self._transitions

    def add_episode(self, episode: dict) -> None:
        n = episode["image"].shape[0]
        if n < 2:
            return
        with self._lock:
            self._episodes.append(episode)
            self._transitions += n
            while self._transitions > self.capacity and len(self._episodes) > 1:
                old = self._episodes.popleft()
                self._transitions -= old["image"].shape[0]

    def can_sample(self, seq_len: int) -> bool:
        with self._lock:
            return any(ep["image"].shape[0] >= seq_len for ep in self._episodes)

    def load_episode_dir(self, directory: str, trim_crash_tail: int = 0) -> int:
        """Preload demonstration episodes (episode_*.npz) for replay seeding (Phase 5).
        Returns the number of transitions added.

        `trim_crash_tail`: drop the last N steps of episodes that END IN A CRASH
        (terminal cont=0 with a strongly negative final reward). Used for the
        behavior-cloning buffer so the actor imitates the gate-passing approach but
        not the final seconds of flying into an obstacle. Episodes ending in finish
        or truncation are kept whole.
        """
        total = 0
        for f in sorted(glob.glob(os.path.join(directory, "episode_*.npz"))):
            d = np.load(f)
            ep = {k: d[k] for k in _EP_KEYS if k in d.files}
            if "image" not in ep or ep["image"].shape[0] < 2:
                continue
            if trim_crash_tail > 0:
                cont = ep["cont"].reshape(-1)
                rew = ep["reward"].reshape(-1)
                crash_end = cont[-1] < 0.5 and rew[-1] < -1.0
                if crash_end and ep["image"].shape[0] > trim_crash_tail + 2:
                    ep = {k: v[:-trim_crash_tail] for k, v in ep.items()}
            self.add_episode(ep)
            total += ep["image"].shape[0]
        return total

    def sample(self, batch: int, seq_len: int, device="cpu") -> dict[str, torch.Tensor]:
        with self._lock:
            eligible = [ep for ep in self._episodes if ep["image"].shape[0] >= seq_len]
            if not eligible:
                raise RuntimeError(f"no episode >= seq_len {seq_len}")
            out: dict[str, list] = {}
            for _ in range(batch):
                ep = random.choice(eligible)
                n = ep["image"].shape[0]
                s = random.randint(0, n - seq_len)
                for k, v in ep.items():
                    out.setdefault(k, []).append(v[s:s + seq_len])
        return {k: torch.as_tensor(np.stack(v)).to(device) for k, v in out.items()}
