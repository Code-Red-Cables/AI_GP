# VQ1 — vision-only drone race client

Python client for the AI Grand Prix / VQ1 simulator. A timed lap is a
**vision-only policy**: a TCN reads 8-keypoint gate pose plus IMU and writes
roll / pitch / yaw-rate / throttle. No odometry, no GPS, no map, no human
on the sticks. Spec §7: a hand on the sticks during a timed run is a DQ.

The work is imitation, not RL. Clean human laps become a seed policy.
The seed flies; a human grabs only when it is wrong; those recoveries are
aggregated (HG-DAgger) and the student is warm-started so the launch is
not re-rolled from scratch. The air is the test. Validation MAE is not.

## Fly a lap (autonomous)

This is the timed path. No gamepad. No keyboard flying. Esc or Ctrl+C
disarms.

```powershell
.\winvenv\Scripts\python.exe tools\run_policy.py
```

Equivalent: `.\winvenv\Scripts\python.exe tools\tune_flight.py policy`.
Add `--panel` to see the camera beside the exact observation the net sees.

**In the FlightSim window before it arms:** log in, start a race, stay on
the pad facing gate 1. The client pre-warms YOLO, waits for a gate box,
then arms.

| Flag | Default | What it is |
|---|---|---|
| `--weights` | `models/policy_seed_17.pt` | Live policy (`H=64`, chunk=5, 21 bins, `--context`) |
| `--yolo` | `models/ROBOFLOW_RETRAIN.pt` | 8-keypoint pose detector |
| `--hz` | 20 | How often the policy history advances |

`main.py` is the same loop once `FLIGHT_MODE=policy` is set. Prefer
`tools/run_policy.py` so weights, loop rate, and the no-human contract are
explicit.

## Requirements

- Windows host. The **Windows** venv drives the GPU (`.\winvenv\`).
- Python 3.12, CUDA PyTorch, Ultralytics, OpenCV, pygame, pymavlink.

```powershell
.\winvenv\Scripts\python.exe -c "import torch, ultralytics, cv2, pygame; print(torch.cuda.is_available())"
```

The sim publishes camera, `HIGHRES_IMU`, and `race_status`. `ATTITUDE`,
`ODOMETRY`, and `TRACK_DATA` are absent. The policy must not need them.

## The process

End to end, this is how a flyer is made and how it is supposed to get
better. Later DAgger rounds that forgot the launch are why several steps
are strict.

### 1. See the gate

The only exteroceptive input is a 640×360 camera. A YOLO-pose model
returns **8 keypoints** (outer square 0–3, flyable opening 4–7, clockwise
from top-left). Class `gate`. A dummy Roboflow class `0` is ignored.

The policy uses those corners as normalised pixels (`unseen = -1`). It
does **not** consume PnP or a map. HSV confirm / fallback is off — a
colour blob has no corner identity.

Dataset: `datasets/AIGP_8keypoints.v5i.yolov8/`. A Roboflow export is
usually wrong (`train` path, class name, `flip_idx`). The trainer refuses
a broken yaml. When the export already contains augs, train with no extra
ones, and do not stretch frames to 640×640 (letterbox):

```powershell
.\winvenv\Scripts\python.exe tools\train_gate_pose.py --no-augment --name gate_pose_v5 --output models\gate_pose_v5.pt
```

Default live pose: `models/ROBOFLOW_RETRAIN.pt`. Details:
[`models/README.md`](models/README.md).

### 2. Fly expert laps (the seed set)

A human flies the rate plant with vision recording. Every finished lap
(gates 0–17) is a demonstration of launch, line, and commit.

```powershell
.\winvenv\Scripts\python.exe tools\tune_flight.py acro
```

`--start-human` on coach is the same idea if you want the H/T HUD, but
the file still has to be **all human** to count as seed.

File **only** finishing attempts in `logs/seed/`. Trim the post-finish
sit. Early quits, pad sits, and DNFs stay in `logs/` — the trainer globs
every file in the seed folder.

`logs/best/` holds the human PB (14.036 s). Fast clean laps are the
heaters you want cloned.

### 3. Train a seed policy (behavior cloning)

From-scratch BC on `logs/seed/` only. Live architecture:

| Knob | Value | Why |
|---|---|---|
| History `H` | 64 | A single frame cannot see through a dropout |
| `--target-dt` | 0.1 s | Windows cover the same time whether the log is 20 or 50 Hz |
| Action chunk | 5 | Short plan; planner averages overlapping *probabilities* |
| Head | 21 bins / channel | Human roll is hold-or-slam; a regressor averages those modes into a mild nowhere |
| `--context` | on | 29 visual+IMU features + 18-gate one-hot + lap progress (48). Course-specific on purpose |
| Epochs | **300** | Seed `policy_seed_17.pt` was 300 from scratch. Val locked ~epoch 255 |

```powershell
.\winvenv\Scripts\python.exe tools\train_policy.py `
  --glob "logs/seed/telem_*.csv" `
  --history 64 --chunk 5 --bins 21 --context --balance-gates `
  --epochs 300 --out models\policy_seed_17.pt
```

`--balance-gates` stops a long loiter in front of one gate from owning
the loss. Policy-flown frames are not in this set.

**Judge the seed in the air.** A discretised head has a quantisation
floor; `val_mae` is not a flight score. The seed that actually leaves
the pad is `models/policy_seed_17.pt`.

### 4. Fly the seed (no human)

```powershell
.\winvenv\Scripts\python.exe tools\run_policy.py --weights models\policy_seed_17.pt --panel
```

It will crash. That is expected (Xing et al.: offline BC failed every
track they measured). The job of the seed is to produce failures worth
correcting, without forgetting how to leave the pad.

### 5. Coach (HG-DAgger)

Policy flies first. The human takes the stick the instant it looks
wrong, then gives it back after the save. Interactive imitation, not
another stack of expert laps.

```powershell
.\winvenv\Scripts\python.exe tools\tune_flight.py coach --weights models\policy_seed_17.pt --panel
```

| Key | Meaning |
|---|---|
| **H** | Human — take the sticks now |
| **T** | Policy — return control after the recovery |
| **K** | Mark this stretch `exclude=1` (do not imitate) |
| Esc / X | Disarm and quit |

Rules that matter:

- Do **not** pass `--start-human`. Launch is part of the task. If you
  always take off for it, the student never has to.
- After a missed gate, go back through **that** gate. The scorekeeper
  does not advance. Skipping to N+1 paints the wrong one-hot on the
  YOLO lock and poisons context.
- Grab early. Curve through with leftover speed. Do not stop and rebuild
  a pretty approach.

The trainer keeps the 0.5 s of policy states before each grab
(`--lead-s`), every human frame, and 0.5 s after T (`--tail-s`). Those
windows are weighted ×3. Pure policy frames are dropped so the student
does not clone itself.

### 6. File keepers only

| Kind | Folder | Rule |
|---|---|---|
| Clean human laps | `logs/seed/` | Finished 0–17, no DNF |
| Mixed coach laps | `logs/coach/` | Finished 0–17 only |

Extract the **finishing attempt**, trim the sit after `RACE_FINISH`.
Leave pad-sits, Y-resets, and DNFs in `logs/`. The trainer globs
**every** file in the folder.

A mid-lap `active_gate` jump to 0 (then to N+1) is the skip/poison
case. Those rows must be `exclude=1` or the file stays out.

### 7. Retrain (warm-start, do not start over)

Aggregate seed + new coach. Load the flyer you still like (`--init`) so
the launch weights are not re-sampled.

```powershell
.\winvenv\Scripts\python.exe tools\train_policy.py `
  --glob "logs/seed/telem_*.csv" `
  --telem logs\coach\telem_<keeper1>.csv logs\coach\telem_<keeper2>.csv `
  --history 64 --chunk 5 --bins 21 --context --balance-gates `
  --lead-s 0.5 --tail-s 0.5 --epochs 150 --lr 3e-4 `
  --init models\policy_seed_17.pt `
  --out models\policy_rN.pt
```

`--init` requires matching `H`, chunk, bins, and `n_in` or the trainer
exits. Prefer an explicit `--telem` list over `--glob "logs/coach/*"` —
old or rejected files in that folder will come along.

Then fly the new weights **autonomously**. If the craft idles on the pad
(pitch ≈ 0, first motion is your grab), park that checkpoint. Do not
coach it further and do not mix those recoveries into the next train.
That is how `policy_r1.pt` / `policy_r2.pt` / `policy_r4.pt` lost the
takeoff: recoveries are ~60–70% human at 3× weight, and “don’t commit”
overwrites the seed launch. Val MAE looked fine each time.

If a DAgger round fails in the air, go back to `policy_seed_17.pt` and
collect more **seed** laps (all-human finishes) rather than more panicked
saves.

### 8. Repeat or submit

A better flyer is more clean seed + a few finished H/T laps that still
leave the pad, then a warm-start that still leaves the pad. The
submission command is step 4 with the weights you trust:

```powershell
.\winvenv\Scripts\python.exe tools\run_policy.py --weights models\policy_seed_17.pt
```

## What the policy sees

29 visual + IMU features: 8 corners as normalised pixels, visibility,
roll, pitch, body rates. Attitude is controller AHRS (`att_raw_*` →
`ahrs_*`), not the EKF. With `--context` that is 48 features.

**Visual snap.** On a new lock, `GATE_PASSED`, a centre jump ≥ 0.20, or
a reacquire, rewrite keypoints / visibility / context across the history
buffer. IMU stays. Losing the gate does not snap (that would paint
sentinels over the last good view). The chunk-plan queue is cleared so
a stale “coast” cannot keep voting.

The checkpoint stores `history`, `chunk`, `bins`, and `use_context`.
Flight reads those flags. Do not override them at run time.

More: [`docs/HG_DAGGER.md`](docs/HG_DAGGER.md).

## Weights

| File | Use |
|---|---|
| `models/policy_seed_17.pt` | **Live policy.** 17-lap seed (now 18 filed), 300 epochs. Use this. |
| `models/ROBOFLOW_RETRAIN.pt` | Default live pose (YOLO11s, unstretched, 8 kpts) |
| `models/gate_pose_v5.pt` | Newer local pose train — pass `--yolo` to try it |
| `models/policy.pt` | Trainer default output name, not the flyer |
| `models/policy_r1.pt`, `policy_r2.pt`, `policy_r4.pt` | Do not fly. Warm-starts that lost the launch. |
| `models/ROBOFLOW_gatepose.pt` | Stretched-nano pose. Do not use. |

## Human teleop (not the timed path)

```powershell
.\winvenv\Scripts\python.exe tools\tune_flight.py acro
.\winvenv\Scripts\python.exe tools\tune_flight.py pilot
```

Stick map and rate signs: [`MANUAL_INSTRUCTIONS.md`](MANUAL_INSTRUCTIONS.md).
Hover trim and Kalman knobs: [`TUNING.md`](TUNING.md).

`FLIGHT_MODE=assist` / `kalman` / `spline` / `race` are classical
experiment paths. They are not the timed submission.

## Layout

| Path | Role |
|---|---|
| `tools/run_policy.py` | Autonomous timed flyer (no human input) |
| `tools/tune_flight.py policy` | Same flyer, via the harness |
| `main.py` | Control loop (`FLIGHT_MODE=policy`) |
| `controller.py` | Rate plant, crash / finish |
| `vision/yolo_gate_detector.py` | YOLO pose, class aliases, no HSV |
| `policy_net.py`, `policy_planner.py`, `race_obs.py` | TCN, bins, visual snap |
| `tools/train_policy.py` | BC seed + DAgger + `--init` |
| `tools/train_gate_pose.py` | Ultralytics pose train |
| `tools/tune_flight.py coach` | H/T intervention harness |
| `tools/tune_flight.py acro` | Human seed laps (vision observe-only) |
| `models/` | Pose + policy checkpoints |
| `logs/seed/`, `logs/coach/` | Filed training laps only |
| `logs/best/` | Human PB archive |
| `test_*.py` | `python -m unittest discover -p "test_*.py"` |

## What this client does not use

- A gamepad or keyboard on a timed run
- HSV-first detection, or HSV images as policy input
- Privileged state, PPO, or a gym
- `gatenet_handoff/` (4-corner CNN; unused)
- `FLIGHT_MODE=race` / `spline` on the timed clock
