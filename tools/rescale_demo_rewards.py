"""One-off: rescale demo gate rewards to the current reward config (2026-07-25).

The demos in artifacts/demos were recorded with w_gate=10; training now uses w_gate=40.
The world model learns reward from these stored values, so demo gate passes were being
valued at a quarter of the live scale. Adds +30 to exactly the gate-pass steps,
identified from the per-step CSV logs (r_gate column), after verifying the CSV rows
align with the npz steps. Originals are backed up to artifacts/demos_orig once.
"""
import csv
import glob
import os
import shutil
import sys

import numpy as np

DEMOS = "artifacts/demos"
BACKUP = "artifacts/demos_orig"
BUMP = 30.0  # w_gate 10 -> 40

if not os.path.isdir(BACKUP):
    shutil.copytree(DEMOS, BACKUP)
    print(f"backed up {DEMOS} -> {BACKUP}")
else:
    print(f"backup {BACKUP} already exists; refusing to double-apply.")
    print("delete it (or restore from it) first if you really want to re-run.")
    sys.exit(1)

total_bumped = 0
for npz_path in sorted(glob.glob(os.path.join(DEMOS, "episode_*.npz"))):
    csv_path = npz_path.replace(".npz", "_log.csv")
    rows = list(csv.DictReader(open(csv_path)))
    d = dict(np.load(npz_path))
    reward = d["reward"]
    assert reward.shape[0] == len(rows), (npz_path, reward.shape, len(rows))

    gate_idx = [i for i, r in enumerate(rows) if float(r["r_gate"]) > 0]
    for i in gate_idx:
        # alignment check: stored reward must match the logged reward at this step
        logged = float(rows[i]["reward"])
        assert abs(float(reward[i, 0]) - logged) < 0.51, (npz_path, i, reward[i, 0], logged)
        reward[i, 0] += BUMP
    d["reward"] = reward
    np.savez(npz_path, **d)
    total_bumped += len(gate_idx)
    print(f"{os.path.basename(npz_path)}: bumped {len(gate_idx)} gate step(s) "
          f"at {gate_idx} -> max reward {reward.max():.1f}")

print(f"done: {total_bumped} gate steps bumped by +{BUMP}")
