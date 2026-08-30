# Autonomous lap via HG-DAgger

Goal: a full autonomous lap with zero human input at run time.

Method: human-gated DAgger. A vision-only student flies; the human takes the
stick the instant it goes wrong; those corrections become training data.
Interactive imitation rather than offline cloning is the deliberate choice
(Xing et al., CoRL 2024: offline BC failed every track they measured; DAgger
on the same data did not).

The timed client is `FLIGHT_MODE=policy`. `tools/run_policy.py` (and
`tools/tune_flight.py policy`) set that mode and never take a gamepad.
Spec §7: human interaction during a timed run is a disqualification.

## Observation

Shared by train and flight in `race_obs.py`.

- **29 base features** — 8 gate keypoints as normalised pixels (unseen = `-1`),
  per-keypoint visibility, roll, pitch, body rates.
- **No** position, PnP, odometry, or absolute yaw. Camera + IMU only.
- **`--context` (live flyer)** — those 29 plus an 18-gate one-hot and
  fractional lap progress from `race_status` (48 features). Course-specific
  on purpose: the question becomes “what do I do at gate 7 of this track.”

Attitude for the observation comes from the controller AHRS
(`att_raw_*` → `ahrs_*`). `race_obs.attitude_is_trusted` rejects EKF-only
rows. Integrated gyro walks; gravity-referenced AHRS does not.

### Visual snap

On a new lock, `GATE_PASSED` / one-hot change, a centre jump ≥ 0.20
normalised pixels, or a reacquire, rewrite **keypoints + visibility +
context** across the history buffer. Keep IMU (roll / pitch / gyro). Clear
the chunk-plan queue. Losing the gate does **not** snap — that would paint
sentinels over the last good view.

Helpers: `visual_target_changed`, `apply_visual_snap` (`VISUAL_END` /
`STATE_END`). Applied in `policy_planner.py` and in
`tools/train_policy.py` `build_windows` (online, same-attempt only).

## Policy

Causal 3-layer TCN (temporal embedding 128) → 2-layer MLP. Live architecture
is stored in the checkpoint and read by coach / `main.py`:

| Knob | Live flyer |
|---|---|
| History `H` | 64 |
| Action chunk | 5 (0.5 s at 10 Hz) |
| Action head | 21 categorical bins / channel |
| Context | on |

A single-frame policy has to guess through every keypoint dropout. Xing et
al. Table 5 is why there is a history at all.

**Bins.** Human roll is multi-modal: hold, then commit. A regressor averages
those modes into a permanent mild input. Classification lets argmax land on
a mode. The planner averages overlapping chunk *probabilities*, then
argmax — averaging decoded values would undo the head.

**Chunks.** Predicting `K` future commands commits a short plan. The planner
re-plans every step (temporal ensembling). Keep `K` short: a chunk is briefly
open-loop, and this sim is not environmentally deterministic (wind, density,
spawn vary). `--chunk 5` at 10 Hz is the working trade.

## What the simulator provides

Spec §4.3 lists `HEARTBEAT`, `ATTITUDE`, `HIGHRES_IMU`, `TIMESYNC`. On the
current VQ1/VQ2 builds:

| Source | Status |
|---|---|
| Camera 640×360 | Present — only exteroceptive input |
| `HIGHRES_IMU` | Present — body rates |
| `race_status` (`active_gate`) | Present — scoring and context one-hot |
| `ATTITUDE` | Absent — use controller AHRS |
| `ODOMETRY` / `TRACK_DATA` | Absent — no map, no ground truth |

Nothing in the policy may depend on odometry or a course map. That is also
why the same observation transfers to VQ2.

`tools/eval_observation.py` is the ground-truth-free gate (rigidity from
gyro + image motion, identity at `GATE_PASSED`, centring). It must print
`PASS` on a clean human lap before you train. Coupling to the stick is
reported but cannot be the only test: a clean pilot drives bearing error
toward zero and leaves correlation nothing to measure.

## Timing

The policy history advances once per control-loop iteration. Training
windows are built from telemetry rows. **Those rates have to agree.**
`coach` logs at the loop rate. `--target-dt` (default 0.1 s) strides any
faster log so every window covers the same span.

Open-loop attitude-tape replay is not a substitute for a policy. Spec §3.5
claims determinism; the sim does not deliver it. `--replay-attitude` is for
rehearsing a supervised prefix only.

## Operator setup

```powershell
cd D:\Code\Competitions\AI_GP
python -m venv winvenv
.\winvenv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.\winvenv\Scripts\python.exe -m pip install -r requirements.txt
.\winvenv\Scripts\python.exe tools\probe_vq1.py --seconds 20
```

Done when the probe reports `HIGHRES_IMU` present. Missing `ATTITUDE` /
`ODOMETRY` / `TRACK_DATA` is expected.

## Detector

Eight-keypoint YOLO-pose. Dataset:
`datasets/AIGP_8keypoints.v5i.yolov8/`. Install notes and the Roboflow yaml
fixes: [`models/README.md`](../models/README.md).

```powershell
.\winvenv\Scripts\python.exe tools\train_gate_pose.py --no-augment --name gate_pose_v5 --output models\gate_pose_v5.pt
```

HSV confirm / fallback is **off**. The policy needs identified keypoints,
not a colour blob. Do not feed HSV-filtered images into the student.

Default live weights: `models/gate_pose_v5.pt`. Do not switch
`YOLO_POSE_MODEL_PATH` to a newer train until that file has been copied
in.

## Seed policy

Fly clean **finished** human laps (gates 0–17). File them in `logs/seed/`
only. Early quits stay out — the trainer globs every matching file.

`acro` records vision observe-only on the rate plant. Do not use
`--start-human` when you later coach: launch is part of the task.

```powershell
.\winvenv\Scripts\python.exe tools\tune_flight.py acro
.\winvenv\Scripts\python.exe tools\train_policy.py --glob "logs/seed/telem_*.csv" --history 64 --chunk 5 --bins 21 --context --balance-gates --epochs 150 --out models\policy_seed_17.pt
```

Judge the seed in the air, not by a validation number. A discretised head
has a quantisation floor; `val_mae` is not a flight score.

## HG-DAgger rounds

Policy flies. **H** takes the stick. **T** gives it back. **K** marks the
current stretch `exclude=1`. After a missed gate, go back through **that**
gate — the scorekeeper does not advance, and skipping ahead mismatches the
one-hot against the YOLO lock.

File **finished** mixed laps in `logs/coach/` only.

```powershell
.\winvenv\Scripts\python.exe tools\tune_flight.py coach --weights models\policy_seed_17.pt --panel
```

Retrain on the aggregate. Warm-start from the flyer you actually like so
the launch is not re-rolled from scratch. `--init` requires matching
`H` / chunk / bins / `n_in` or the trainer exits.

```powershell
.\winvenv\Scripts\python.exe tools\train_policy.py `
  --glob "logs/seed/telem_*.csv" `
  --glob "logs/coach/telem_*.csv" `
  --history 64 --chunk 5 --bins 21 --context --balance-gates `
  --lead-s 0.5 --tail-s 0.5 --epochs 150 --lr 3e-4 `
  --init models\policy_seed_17.pt `
  --out models\policy_rN.pt
```

`--lead-s` back-dates each intervention (the states leading into a failure
are where the correction should have begun). `--tail-s` keeps a recovery
tail. Policy-flown frames are dropped by default so the student does not
imitate itself.

## Timed run

Zero human input. Do not edit `config.py`.

```powershell
.\winvenv\Scripts\python.exe tools\run_policy.py
.\winvenv\Scripts\python.exe tools\tune_flight.py policy --panel
```

`tools/score_policy.py` summarises logged attempts; it is not a substitute
for watching the flight.

## Layout

| Piece | Role |
|---|---|
| `race_obs.py` | Observation / labels / visual snap |
| `policy_net.py` | Causal TCN → MLP, optional bin head |
| `policy_planner.py` | Online student (`FLIGHT_MODE=policy`) |
| `tools/train_policy.py` | BC seed + DAgger aggregation + `--init` |
| `tools/run_policy.py` | Autonomous timed flyer (no human input) |
| `tools/tune_flight.py coach` | Intervention harness |
| `tools/eval_observation.py` | Hard observation gate |
| `tools/score_policy.py` | Autonomous attempt report |
| `logs/seed/`, `logs/coach/` | Filed keepers only |
