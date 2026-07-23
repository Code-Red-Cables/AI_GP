"""Decoupled DreamerV3 trainer: a background `Collector` fills replay at the sim's real-
time ~30 Hz while this learner trains the world model + actor-critic on the GPU as fast as
the configured train-ratio allows. They interact only through the thread-safe replay and
periodic weight syncs, so the GPU is never blocked on the sim and training continues
across sim crashes.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Optional

from ..config import Config
from ..dreamer.agent import DreamerAgent
from ..dreamer.replay import SequenceReplay
from ..env.drone_racing_env import DroneRacingEnv
from .collector import Collector


class Trainer:
    def __init__(self, cfg: Config, resume: Optional[str] = None):
        self.cfg = cfg
        self.agent = DreamerAgent(cfg)
        if resume:
            self.agent.load(resume)
            print(f"[train] resumed from {resume}", flush=True)
        self.replay = SequenceReplay(cfg.train.replay_capacity)
        self.env = DroneRacingEnv(cfg)
        self.collector = Collector(self.env, self.replay, cfg, device=cfg.train.collector_device)

        self.run_dir = Path(cfg.train.log_dir) / f"{cfg.name}_{int(time.time())}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        cfg.save(self.run_dir / "config.yaml")
        self._csv_path = self.run_dir / "metrics.csv"
        self._csv_started = False
        print(f"[train] run dir: {self.run_dir}", flush=True)
        print(f"[train] learner device={self.agent.device} collector={cfg.train.collector_device}",
              flush=True)

    # ---- helpers -----------------------------------------------------------
    def _log(self, row: dict) -> None:
        with open(self._csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._csv_started:
                w.writeheader()
                self._csv_started = True
            w.writerow(row)

    def _checkpoint(self, env_steps: int, tag: str = "") -> None:
        name = tag or f"ckpt_{env_steps}"
        self.agent.save(self.run_dir / f"{name}.pt")
        self.agent.export_deploy(self.run_dir / "deploy_latest.pt")
        print(f"[train] checkpoint -> {self.run_dir / (name + '.pt')}", flush=True)

    # ---- main loop ---------------------------------------------------------
    def run(self) -> None:
        t = self.cfg.train
        bt = t.batch_size * t.seq_len
        self.collector.start()

        # prefill with random exploration
        print(f"[train] prefill to {t.prefill} transitions (random policy) ...", flush=True)
        while len(self.replay) < t.prefill:
            time.sleep(1.0)
            print(f"[train] prefill {len(self.replay)}/{t.prefill} "
                  f"(collector steps={self.collector.env_steps}, eps={self.collector.episodes})",
                  flush=True)
        self.collector.sync(self.agent)
        self.collector.mode = "policy"
        print("[train] prefill done -> learner + policy collection running", flush=True)

        updates = 0
        last_ckpt = 0
        t_start = time.time()
        last_report = t_start
        while self.collector.env_steps < t.total_env_steps:
            env_steps = self.collector.env_steps
            # train-ratio pacing: keep replayed steps ~= train_ratio * env steps
            target_updates = t.train_ratio * max(1, env_steps) / bt
            if updates < target_updates and self.replay.can_sample(t.seq_len):
                batch = self.replay.sample(t.batch_size, t.seq_len, self.agent.device)
                metrics = self.agent.train_step(batch)
                updates += 1

                if updates % t.sync_every == 0:
                    self.collector.sync(self.agent)
                if updates % t.log_every == 0:
                    now = time.time()
                    row = {
                        "wall_s": round(now - t_start, 1),
                        "env_steps": env_steps,
                        "updates": updates,
                        "steps_per_s": round(env_steps / (now - t_start + 1e-9), 1),
                        "updates_per_s": round(updates / (now - t_start + 1e-9), 1),
                        "replay": len(self.replay),
                        "episodes": self.collector.episodes,
                        "relaunches": self.collector.relaunches,
                        **{k: round(v, 4) for k, v in metrics.items()},
                        **{f"ep_{k}": v for k, v in self.collector.last_info.items()},
                    }
                    self._log(row)
                    print(f"[train] steps={env_steps} upd={updates} "
                          f"sps={row['steps_per_s']} ups={row['updates_per_s']} "
                          f"wm={metrics.get('wm/loss', float('nan')):.2f} "
                          f"actor={metrics.get('ac/actor_loss', float('nan')):.2f} "
                          f"ep_r={self.collector.last_info.get('ep_reward')} "
                          f"gate={self.collector.last_info.get('max_gate')} "
                          f"stage={self.collector.last_info.get('stage')}", flush=True)
                if env_steps - last_ckpt >= t.checkpoint_every:
                    last_ckpt = env_steps
                    self._checkpoint(env_steps)
            else:
                # ahead of ratio or waiting for data: yield to the collector
                time.sleep(0.005)

        self._checkpoint(self.collector.env_steps, tag="ckpt_final")
        self.agent.export_deploy(self.run_dir / "deploy_final.pt")
        self.collector.stop()
        self.env.close()
        print(f"[train] done. steps={self.collector.env_steps} updates={updates} "
              f"dir={self.run_dir}", flush=True)
