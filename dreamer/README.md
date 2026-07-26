# DreamerV3 for AI Grand Prix (DCL FlightSim)

An end-to-end DreamerV3 training + deployment system for the AI Grand Prix drone racer.
Self-contained under `dreamer/`; the existing flight stack in the repo root is left intact
as the fallback controller.

The sim is a closed Unreal Engine 4 Windows binary controlled over MAVLink + UDP. Training
runs in WSL2 (GPU via `/dev/dxg`); the sim renders on the Windows host; localhost UDP
bridges them. See `docs/` for the full audit and design.

## Layout
```
configs/     dreamer_small.yaml (RTX 4070), dreamer_tiny.yaml (fast smoke)
src/dreamer_drone/
  sim/       transport + process control (MAY read privileged state)
  env/       Gymnasium-style env, reward, termination, curriculum, baselines
  dreamer/   world model (RSSM) + actor-critic + replay  (PyTorch)
  deploy/    deployment-clean inference controller
tools/       probe_simulator.py  (Milestone 1)
scripts/     train / evaluate / deploy / collect_demos
tests/       reward geometry, leakage audit, model shapes  (run without a sim)
docs/        simulator_audit, system_architecture, interface_inventory,
             assumptions_and_open_questions, deployment_compliance
```

## Quick start
```bash
# 0) deps (CUDA torch recommended in WSL)
pip install -e dreamer            # or: pip install -r requirements + torch

# 1) tests that need NO simulator
python dreamer/tests/test_reward_geometry.py
python dreamer/tests/test_no_privileged_input.py
python dreamer/tests/test_dreamer_shapes.py
# or: pytest dreamer

# 2) Milestone 1 — measure the real sim timing (start the sim first)
python dreamer/tools/probe_simulator.py --seconds 30 --send-neutral

# 3) Milestone 2/3 — validate the env with a baseline
python dreamer/scripts/collect_demos.py --policy scripted --episodes 5

# 4) train / evaluate / deploy
python dreamer/scripts/train_dreamer.py --config dreamer/configs/dreamer_small.yaml
python dreamer/scripts/evaluate.py --checkpoint <run>/ckpt_final.pt --episodes 10
python dreamer/scripts/deploy.py --checkpoint <run>/deploy_final.pt
```

## Compliance (VQ2)
The deployed policy reads **only** competition-legal runtime signals: camera RGB,
`ATTITUDE` (orientation + body rates), `HIGHRES_IMU`, previous action, and measured `dt`.
Privileged state (race gate index, collision, position/gate poses) is used **only** for
reward, termination, curriculum, and evaluation — never as a policy input. The
deployment controller imports no privileged module; `tests/test_no_privileged_input.py`
enforces this. Details in `docs/deployment_compliance.md`.

## Status (milestones)
- **M1 (probe):** tool built; live numbers require running the sim (all timing is
  measured, never assumed — spec's "30 Hz" is unverified until you run the probe).
- **M2/M3 (env + baselines):** env, reward, termination, curriculum built and unit-tested;
  needs a live sim to validate resets/action-mapping end-to-end.
- **M4 (DreamerV3):** world model + actor-critic implemented and shape/gradient-verified
  on CPU (`test_dreamer_shapes.py`). Learning quality pending live data.
- **M5/M6:** evaluation harness + deployment-clean export built; benchmarks pending a sim.

## Key open questions (resolve with the probe — `docs/assumptions_and_open_questions.md`)
1. Does the sim advance in real time regardless of client inference (or pause)?
2. Is position telemetry re-enablable for training (dense progress) or truly absent?
3. `cmd 31000` reset semantics / ACK.
4. Does raw ACRO rate control work immediately post-arm?
