"""Evaluate a checkpoint on full racing runs — exploration off, no training updates,
no privileged actor inputs. Reports finish/gate/collision rates, lap time, inference
latency, and appends to a leaderboard CSV.

    python dreamer/scripts/evaluate.py --checkpoint <run>/ckpt_final.pt --episodes 10
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from dreamer_drone.config import load_config  # noqa: E402
from dreamer_drone.dreamer.agent import DreamerAgent  # noqa: E402
from dreamer_drone.env.drone_racing_env import DroneRacingEnv  # noqa: E402
from dreamer_drone.env.spaces import neutral_action  # noqa: E402

import torch  # noqa: E402


def run_episode(env, agent) -> dict:
    obs, info = env.reset()
    state = agent.initial_state(1)
    prev_a = neutral_action()
    t0 = time.time()
    latencies, actions = [], []
    max_gate, collisions, steps = 0, 0, 0
    finished = False
    while True:
        image = torch.from_numpy(obs["image"]).unsqueeze(0).to(agent.device)
        vector = torch.from_numpy(obs["vector"]).unsqueeze(0).to(agent.device)
        prev = torch.from_numpy(prev_a).unsqueeze(0).to(agent.device)
        ti = time.perf_counter()
        action, state = agent.act({"image": image, "vector": vector}, state, prev,
                                  training=False)
        latencies.append((time.perf_counter() - ti) * 1000.0)
        a = action.squeeze(0).cpu().numpy().astype(np.float32)
        obs, reward, term, trunc, info = env.step(a)
        actions.append(a)
        prev_a = a
        steps += 1
        if info.get("active_gate") is not None:
            max_gate = max(max_gate, int(info["active_gate"]))
        if info.get("term_reason") == "collision":
            collisions += 1
        if info.get("term_reason") == "finish":
            finished = True
        if term or trunc:
            break
    acts = np.array(actions) if actions else np.zeros((1, 4))
    smoothness = float(np.mean(np.abs(np.diff(acts, axis=0)))) if len(acts) > 1 else 0.0
    return {
        "finished": finished, "max_gate": max_gate, "collisions": collisions,
        "steps": steps, "wall_time_s": time.time() - t0,
        "mean_latency_ms": float(np.mean(latencies)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "control_smoothness": smoothness,
        "term_reason": info.get("term_reason", ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--leaderboard", default="artifacts/leaderboard.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    env = DroneRacingEnv(cfg)
    agent = DreamerAgent(cfg)
    agent.load(args.checkpoint, actor_only=True)

    results = [run_episode(env, agent) for _ in range(args.episodes)]
    env.close()

    finish_rate = np.mean([r["finished"] for r in results])
    agg = {
        "checkpoint": args.checkpoint,
        "episodes": args.episodes,
        "finish_rate": round(float(finish_rate), 3),
        "mean_max_gate": round(float(np.mean([r["max_gate"] for r in results])), 2),
        "collision_rate": round(float(np.mean([r["collisions"] > 0 for r in results])), 3),
        "mean_latency_ms": round(float(np.mean([r["mean_latency_ms"] for r in results])), 2),
        "worst_latency_ms": round(float(np.max([r["p99_latency_ms"] for r in results])), 2),
        "mean_smoothness": round(float(np.mean([r["control_smoothness"] for r in results])), 4),
    }
    print("[eval]", agg)

    lb = Path(args.leaderboard)
    lb.parent.mkdir(parents=True, exist_ok=True)
    new = not lb.exists()
    with open(lb, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg.keys()))
        if new:
            w.writeheader()
        w.writerow(agg)
    print(f"[eval] appended to {lb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
