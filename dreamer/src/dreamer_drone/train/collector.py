"""Background collection thread for the decoupled trainer.

The sim is real-time-locked at ~30 Hz, so collection runs in its own thread doing CPU
inference on a *copy* of the policy (encoder+RSSM+actor). The learner trains the live
networks on the GPU and periodically syncs weights into this copy. The two threads share
only the thread-safe `SequenceReplay` and the policy copy (guarded by a lock during
forward/sync). This keeps the GPU learner from ever blocking on the 30 Hz sim, and lets
the learner keep training through a sim crash while the collector relaunches.
"""
from __future__ import annotations

import errno
import threading
import time

import numpy as np
import torch

from ..config import Config
from ..deploy.controller import DeployPolicy
from ..dreamer.replay import EpisodeAccumulator, SequenceReplay
from ..env.drone_racing_env import DroneRacingEnv
from ..env.spaces import ACTION_DIM, neutral_action
from ..sim.process_manager import SimUnavailable


class Collector:
    def __init__(self, env: DroneRacingEnv, replay: SequenceReplay, cfg: Config,
                 device: str = "cpu"):
        self.env = env
        self.replay = replay
        self.cfg = cfg
        self.device = torch.device(device)
        self.policy = DeployPolicy(cfg).to(self.device).eval()
        self._lock = threading.Lock()      # guards self.policy (forward vs weight-sync)
        self.mode = "random"               # "random" during prefill, then "policy"
        self.running = False
        self.thread: threading.Thread | None = None
        # telemetry (GIL-atomic ints / small dicts). last_info carries ALL keys from
        # the start so the trainer's CSV header is stable even before episode 1 ends.
        self.env_steps = 0
        self.episodes = 0
        self.last_info: dict = {
            "ep_reward": 0.0, "max_gate": 0, "gates_passed": 0, "len": 0,
            "gate_visible_frac": 0.0, "progress_sum": 0.0, "reason": "", "stage": "",
        }
        self.relaunches = 0
        # episode dump for behavior debugging: when the trainer sets dump_dir, every
        # dump_every-th collected episode is written there as npz
        self.dump_dir: str | None = None
        self.dump_every = 25

    # ---- lifecycle ---------------------------------------------------------
    def start(self) -> "Collector":
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)

    def sync(self, agent) -> None:
        """Called by the learner between updates to push fresh weights into the copy."""
        with self._lock:
            self.policy.sync_from(agent)

    # ---- policies ----------------------------------------------------------
    def _random_action(self) -> np.ndarray:
        a = (np.random.uniform(-1, 1, ACTION_DIM) * 0.3).astype(np.float32)
        a[0] = np.random.uniform(-0.3, 0.3)   # gentle thrust during exploration
        return a

    def _policy_action(self, obs: dict, state: dict, prev: np.ndarray):
        img = torch.from_numpy(obs["image"]).unsqueeze(0).to(self.device)
        vec = torch.from_numpy(obs["vector"]).unsqueeze(0).to(self.device)
        pv = torch.from_numpy(prev).unsqueeze(0).to(self.device)
        with self._lock:
            action, state = self.policy.act(img, vec, state, pv, sample=True)
        return action.squeeze(0).cpu().numpy().astype(np.float32), state

    # ---- loop --------------------------------------------------------------
    def _loop(self) -> None:
        while self.running:
            try:
                self._run_episode()
            except SimUnavailable as e:
                print(f"[collect] sim unavailable ({e}); relaunching ...", flush=True)
                self.relaunches += 1
                try:
                    self.env.close()
                except Exception:
                    pass
                self.env.pm.relaunch()
                time.sleep(3.0)
            except OSError as e:
                if e.errno == errno.EADDRINUSE:
                    # Another trainer already owns the camera port for this sim. Two
                    # collectors would send conflicting actions/resets to one drone,
                    # so this is fatal — do NOT retry (2026-07-24: two concurrent runs
                    # silently corrupted each other for 100 minutes).
                    print("[collect] FATAL: camera port already in use — another "
                          "trainer is attached to this sim. Stop the other "
                          "train_dreamer process and restart. Collection halted.",
                          flush=True)
                    self.running = False
                    return
                print(f"[collect] unexpected error: {e!r}; retrying in 2s", flush=True)
                time.sleep(2.0)
            except Exception as e:  # never let the collector thread die silently
                print(f"[collect] unexpected error: {e!r}; retrying in 2s", flush=True)
                time.sleep(2.0)

    def _run_episode(self) -> None:
        obs, _ = self.env.reset()
        acc = EpisodeAccumulator()
        state = self.policy.initial_state(self.device)
        prev = neutral_action()
        ep_reward, max_gate, steps = 0.0, 0, 0
        vis_steps, progress_sum = 0, 0.0
        info: dict = {}
        term = trunc = False
        for _ in range(self.cfg.train.max_episode_steps):
            if not self.running:
                break
            if self.mode == "random":
                action = self._random_action()
            else:
                action, state = self._policy_action(obs, state, prev)
            nobs, reward, term, trunc, info = self.env.step(action)
            acc.add(obs["image"], obs["vector"], action, reward, 0.0 if term else 1.0)
            obs, prev = nobs, action
            ep_reward += reward
            steps += 1
            self.env_steps += 1
            if info.get("active_gate") is not None:
                max_gate = max(max_gate, int(info["active_gate"]))
            # vision-proxy health: how often the detector fires + what the dense
            # progress term actually paid out (catches detector/reward exploits)
            if info.get("gate_visible"):
                vis_steps += 1
            progress_sum += info.get("reward_components", {}).get("progress", 0.0)
            if term or trunc:
                break
        ep = acc.flush()
        if ep is not None:
            self.replay.add_episode(ep)
            if (self.dump_dir is not None and self.mode == "policy"
                    and self.episodes % self.dump_every == 0):
                try:
                    import os
                    os.makedirs(self.dump_dir, exist_ok=True)
                    np.savez(os.path.join(self.dump_dir, f"ep_{self.episodes:05d}.npz"), **ep)
                except Exception as e:
                    print(f"[collect] episode dump failed: {e!r}", flush=True)
        self.episodes += 1
        # an empty reason with a full-length episode means the hard step cap hit,
        # i.e. env-side termination never fired — make that visible in the logs
        reason = info.get("term_reason", "")
        if not reason and not (term or trunc) and steps > 0:
            reason = "step_cap"
        self.last_info = {
            "ep_reward": round(ep_reward, 2), "max_gate": max_gate,
            "gates_passed": info.get("gates_passed", max_gate), "len": steps,
            "gate_visible_frac": round(vis_steps / max(1, steps), 3),
            "progress_sum": round(progress_sum, 3),
            "reason": reason, "stage": info.get("stage", ""),
        }
