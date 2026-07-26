"""Run the deployment-clean controller against the sim (real-time, no privileged input).

    python dreamer/scripts/deploy.py --checkpoint <run>/deploy_final.pt

Use the exported `deploy_*.pt` (encoder+RSSM+actor only). A full training checkpoint is
NOT accepted here — the deployment path intentionally cannot load the critic/decoder.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from dreamer_drone.config import load_config  # noqa: E402
from dreamer_drone.deploy.controller import DeployController  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="exported deploy_*.pt")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seconds", type=float, default=480.0)
    ap.add_argument("--no-arm", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    controller = DeployController(cfg, args.checkpoint)
    print("[deploy] connecting to sim ...")
    try:
        controller.run(arm=not args.no_arm, max_seconds=args.seconds)
    except KeyboardInterrupt:
        pass
    finally:
        controller.close()
        if controller.latencies:
            import numpy as np
            print(f"[deploy] inference latency mean={np.mean(controller.latencies):.2f}ms "
                  f"p99={np.percentile(controller.latencies, 99):.2f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
