"""Train DreamerV3 against the live DCL FlightSim (decoupled collector + learner).

    python dreamer/scripts/train_dreamer.py --config dreamer/configs/dreamer_small.yaml

A background thread collects at the sim's real-time ~30 Hz while the learner trains on the
GPU at the configured train-ratio; they share a thread-safe replay and periodic weight
syncs. Requires a running simulator (see docs/simulator_audit.md). Handles prefill,
curriculum, checkpoints/exports, and sim-crash recovery. Ctrl+C saves and exits.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from dreamer_drone.config import load_config  # noqa: E402
from dreamer_drone.train.trainer import Trainer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--resume", default=None, help="checkpoint (.pt) to resume from")
    ap.add_argument("--demos", default=None,
                    help="dir of episode_*.npz to seed the replay (e.g. artifacts/demos)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    trainer = Trainer(cfg, resume=args.resume, demos=args.demos)
    try:
        trainer.run()
    except KeyboardInterrupt:
        print("\n[train] interrupted — saving final checkpoint ...", flush=True)
        trainer._checkpoint(trainer.collector.env_steps, tag="ckpt_interrupt")
        trainer.agent.export_deploy(trainer.run_dir / "deploy_interrupt.pt")
        trainer.collector.stop()
        trainer.env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
