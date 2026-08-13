# Autonomous drone racing via HG-DAgger

Goal: a full autonomous lap with zero human input at run time.

Method: human-gated DAgger. A vision-only student policy flies, the human takes
over the instant it goes wrong, and those corrections become training data.
Interactive imitation rather than offline cloning is the deliberate choice —
Xing et al. (CoRL 2024) measure offline behaviour cloning at **0% success on all
three tracks** and DAgger at 52–64% on identical data.

An earlier behaviour-cloning attempt in this repo failed for two identified,
fixable reasons, both addressed here: the observation was not coupled to the gate
being flown (`corr(gate_u, cmd_roll_rate)` = **+0.03**, because the detector
often tracked a different gate), and it used a single-frame observation where the
paper's ablation shows `H=4` → 0% and `H=32` → 100% success.

Architecture:

- **Observation** — 8 gate keypoints as normalised pixels (unseen = `-1`) plus
  per-keypoint visibility, roll, pitch, and body rates. No position, no PnP, no
  absolute yaw: only what a camera and IMU can produce, which is what makes it
  portable across maps.
- **Policy** — causal 3-layer TCN (temporal embedding 128) into a 2-layer MLP,
  over `H=32` frames, emitting collective thrust and three body rates.
- **Training** — BC seed on clean human laps, then DAgger rounds on the
  aggregate `D = D_seed ∪ D_round1 ∪ …`, weighting intervention segments,
  back-dating each intervention ~0.5 s (the states leading *into* a failure are
  where the correction should have begun) and keeping a recovery tail.
- **Compliance** — spec §7 makes human interaction during a timed run grounds
  for disqualification, so interventions are training-only and `main.py` never
  imports a gamepad path.

## What the simulator actually provides

Spec §4.3 lists `HEARTBEAT`, `ATTITUDE`, `HIGHRES_IMU` and `TIMESYNC`. Measured
on current builds, over a full 17-gate lap (5,854 logged rows):

| Source | Status | Consequence |
|---|---|---|
| Camera, 640×360 | Present, 30 Hz sim | The policy's only exteroceptive input |
| `HIGHRES_IMU` | Present, 99.9% of rows | Body rates; all three axes confirmed live |
| `race_status` (`active_gate`) | Present | Gate scoring and the identity check |
| `ATTITUDE` | **Absent** (0 / 5,854) | Attitude comes from the controller AHRS |
| `ODOMETRY` | **Absent** (was on the retired 3391 build) | No ground truth, ever |
| `TRACK_DATA` | Absent / nulled | No gate map |

So there is no ground truth and no sim attitude. Neither is coming back, and
nothing may depend on either.

**What did not change.** The policy was deliberately built vision-only — eight
gate keypoints, roll and pitch, body rates (`race_obs.py`). It never read
position, so Phases 4–7 stand as written. This is also why it transfers to VQ2.

**What changed.**

| Was | Now |
|---|---|
| Attitude from `shared_data['attitude']` (EKF) | `att_raw_*` → `ahrs_*` → EKF, and rows that reach the EKF are dropped from training |
| Phase 3b gate = detector bearing vs ODOMETRY bearing | Three ground-truth-free checks (below) |
| `course_map.json` required before training | Only needed for the optional ODOMETRY check |

The attitude change is not cosmetic. `shared_data['attitude']` is integrated
gyro, measured in this repo walking +4° → −23° over 50 s; a policy trained on it
learns an attitude channel that is a slow random walk. The controller AHRS is
gravity-referenced, so `race_obs.attitude_is_trusted` accepts `att_raw`/`ahrs`
and `tools/train_policy.py` excludes anything else. On the reference lap that is
96.5% of rows kept, 206 dropped — all pre-arm, where there is no useful action
label anyway.

### The replacement hard gate

`tools/eval_observation.py` needs no position:

1. **Rigidity** — under camera rotation a world-fixed point's image motion is
   depth-independent and predictable from `HIGHRES_IMU`. The gyro coefficients
   are *fitted*, not assumed, so an inverted axis shows up as a negative
   coefficient instead of a false failure. Two subtleties cost real debugging:
   consecutive log rows often carry the *same* camera frame (1,351 of them on the
   reference lap), which pairs zero measured motion against nonzero prediction;
   and at 10 Hz logging with rates reaching 10 rad/s a single step spans ~30°,
   where the first-order flow model is meaningless. Both are now excluded.
   *Diagnostic only:* translation flow, which the model cannot represent,
   dominates at racing speed.
2. **Identity** — at each `active_gate` increment the sim says a gate was just
   flown through, so the tracked gate should be large and near frame centre then.
   Noisy per-pass, because the detector may legitimately have switched to the
   *next* gate by that instant, so it needs ≥5 sampled passes to fail a run.
3. **Centring, with coupling as fallback** — a detector locked onto the wrong
   gate cannot keep its centroid at the image centre across a series of
   successful passes, so the distribution of |u − cx| is the primary test.
   Correlation between bearing and stick is reported but cannot be the gate on
   its own: **a pilot who flies well drives the bearing error to zero and leaves
   correlation nothing to measure.** Measured — a clean 17-gate lap scored 0.20
   where a sloppy 6-gate lap scored 0.87. A run fails only when bearing is large
   *and* uncorrelated, which is the original +0.03 disaster.

Reference lap (17/17 gates, 3,203 training-usable rows): tracked gate sits a
median **2.2 px** from frame centre, median bearing **0.4°**, identity jumps
>80 px on 5.2% of frames. The observation is sound; the earlier failure did not
recur.

## Course-specific mode: when one track is the whole goal

The observation above is deliberately portable — corners, attitude, body rates
and nothing else — so the same policy would work on another course. If the only
requirement is flying **this** course autonomously, two options trade that
portability for a much easier learning problem.

**Course context** (`--context`). Adds the sim's `active_gate` as a one-hot plus
fractional lap progress, both ordinary run-time telemetry from `race_status`.
This changes the question from "given these corners, how do I follow a gate" to
"given these corners *and the fact that I am at gate 7 of this track*, what do I
do", which is a far smaller function to fit from a handful of laps. The policy
becomes specific to this course, which is the point.

**Discretised actions** (`--bins 21`). Measured across every seed lap, roll is
within 0.1 rad/s of zero in **71%** of frames and beyond 2.5 rad/s in **18%**,
with little between: the pilot holds, then commits. That is a multi-modal action
distribution, and a single-output regressor cannot represent it — a mean-seeking
loss (Huber/L2) lands *between* the modes and rolls mildly forever, a
median-seeking loss (L1) collapses onto zero and never commits. Both were
observed in flight, in that order. Predicting *which bin* the action falls in,
as a classification, makes the argmax land on a mode instead of averaging across
them. The planner combines overlapping chunk predictions as probabilities and
then takes the argmax, because averaging the decoded values would reintroduce
exactly the averaging this head exists to remove.

**Action chunking** (`--chunk K`). The network predicts the next `K` commands
instead of one. Committing to a short plan rather than re-deciding every frame
is the standard remedy for single-step behaviour cloning's jitter-then-diverge
failure, and `policy_planner` averages the overlapping predictions for the
current instant (temporal ensembling), which smooths the command stream without
adding lag.

```powershell
.\winvenv\Scripts\python.exe tools\train_policy.py --glob "logs/telem_*.csv" `
  --context --chunk 5 --bins 21 --balance-gates --drop-collision-s 1.0 `
  --epochs 150 --out models\policy_ctx.pt
```

Note that `val_mae` is not comparable between a regression and a discretised run
— the latter carries a quantisation floor of half a bin — so judge these by
flight, using seconds of policy control per intervention.

Both settings are recorded in the checkpoint, so `coach` and `main.py` configure
themselves; older checkpoints keep loading unchanged.

### Timing must match, or nothing else matters

The policy's history advances once per control-loop iteration, and training
builds its windows from telemetry rows. **Those two rates have to agree.** A
coach session logged at 48 Hz against a 10 Hz control loop produced intervention
windows spanning 0.66 s where the seed laps spanned 3.2 s, and the model was
asked to learn both as though they were the same thing. `coach` now logs at
exactly the loop rate, and `--target-dt` (default 0.1 s) strides any faster log
so every window covers the same span.

### Open-loop replay is not an option

Spec §3.5 claims environmental determinism, but the simulator does not deliver
it: wind, air density and the spawn position all vary slightly between runs
(operator observation, and it overrides the document). So the attitude tapes in
`practice/history/` cannot be replayed open-loop to fly a lap — the same stick
sequence lands somewhere different every time, and error accumulates with no
feedback to correct it. `--replay-attitude` remains useful for rehearsing a
deterministic *prefix* under supervision, and nothing more.

This also sets an upper bound on action chunking. A chunk is a short committed
plan, i.e. briefly open loop, so it must stay short enough that drift cannot
build inside it. `--chunk 5` at 10 Hz commits 0.5 s, and the planner re-plans
every step and averages overlapping predictions, so it is not truly open loop.
Do not raise the chunk length far beyond that.

The upside is that this variation is free domain randomisation: every lap you
fly is a different sample of wind and spawn, so a policy trained across enough
laps is forced to be reactive rather than to memorise one trajectory. The cost
is that covering that variation needs *more* laps than a deterministic course
would — which is the strongest argument for continuing to add clean laps.

## Operator setup (Phase 1)

```powershell
cd D:\Code\Competitions\AI_GP
python -m venv winvenv
.\winvenv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.\winvenv\Scripts\python.exe -m pip install -r requirements.txt
```

Extract the simulator **outside** the repo, launch it, start a race, then:

```powershell
.\winvenv\Scripts\python.exe tools\probe_vq1.py --seconds 20
```

Done when the probe reports **HIGHRES_IMU present**. `ATTITUDE`, `ODOMETRY` and
`TRACK_DATA` are all absent on current builds — expected, and handled. If
`ODOMETRY`/`TRACK_DATA` ever reappear, run
`tools\build_course_map.py --probe artifacts\vq1_probe.json` to enable the extra
ground-truth check.

## Detector (Phase 3a)

An eight-keypoint YOLO-pose model is trained by `tools/train_gate_pose.py` from
`datasets/AIGP_8keypoints.v1i.yolov8` and installed to `models/gate_pose.pt`
(150 epochs: pose mAP50 0.834, mAP50-95 0.751). The Roboflow export ships three
bugs that are corrected in the committed `data.yaml` — dataset-relative `train:`
paths, a class literally named `gate`, and `flip_idx: [1,0,3,2,5,4,7,6]` so
`fliplr` augmentation does not teach mirrored corner identities.

To A/B against the classical HSV detector:

```powershell
.\winvenv\Scripts\python.exe tools\eval_detectors.py --frames frames\run1
.\winvenv\Scripts\python.exe tools\eval_detectors.py --data datasets\AIGP_8keypoints.v1i.yolov8 --split test
```

HSV may win raw detection rate, but the policy needs eight identified keypoints,
so YOLO-pose is the deployed detector.

## Hard observation gate (Phase 3b) — stop if this fails

Fly one lap so telem has `kp*_u/v`, `ahrs_*` and `active_gate`, then:

```powershell
.\winvenv\Scripts\python.exe tools\tune_flight.py acro --slow-mo --slow-mo-scale 0.2
.\winvenv\Scripts\python.exe tools\eval_observation.py --telem logs\telem_....csv
```

Must print `PASS`. Failure means the detector is tracking a gate you are not
flying to — fix vision before training, because DAgger cannot repair a broken
observation. `WARN` lines are expected: rigidity is usually inconclusive, and
coupling is inconclusive whenever you fly cleanly.

## Seed policy (Phase 5)

Fly 10–15 clean laps. `acro` is the flying mode (rate sticks, vision recorded
observe-only); `manual` is the same logging on the angle-mode plant.

```powershell
.\winvenv\Scripts\python.exe tools\tune_flight.py acro --slow-mo --slow-mo-scale 0.2
.\winvenv\Scripts\python.exe tools\train_policy.py --glob "logs/telem_*.csv" --out models/policy_seed.pt
```

Match Cheat Engine / DxWnd to the same slow-mo factor. Training reads
`logs\telem_*.csv`, never `logs\tuning\`.

Expect the seed to fly badly. That is intended.

## HG-DAgger rounds (Phase 6)

```powershell
.\winvenv\Scripts\python.exe tools\tune_flight.py coach --weights models/policy_seed.pt
# Intervene with H the instant it looks wrong; T returns policy.
# Aggregate every round — never drop earlier data:
.\winvenv\Scripts\python.exe tools\train_policy.py --glob "logs/telem_*.csv" --out models/policy_r1.pt --lead-s 0.5 --tail-s 0.5
```

Repeat until gates-cleared plateaus.

## Autonomous eval (Phase 7)

Timed runs: **zero human input**. Spec §7 — `main.py` never imports the gamepad.

```powershell
$env:FLIGHT_MODE="policy"
$env:POLICY_WEIGHTS="models/policy_rN.pt"
.\winvenv\Scripts\python.exe main.py
.\winvenv\Scripts\python.exe tools\score_policy.py --glob "logs/telem_*.csv"
```

## Layout

| Piece | Role |
|---|---|
| `race_obs.py` | Shared observation / label definition (train + flight) |
| `gate_bearing.py` | Bearing, rotation-flow and correlation helpers |
| `logger.py` | `logs/telem_*.csv`: keypoints, IMU, commands, authority |
| `policy_net.py` | Causal 3-layer TCN → MLP, H=32 |
| `policy_planner.py` | Online student (`FLIGHT_MODE=policy`) |
| `tools/train_policy.py` | BC seed + DAgger aggregation |
| `tools/tune_flight.py coach` | Intervention harness + authority logging |
| `tools/eval_observation.py` | Hard observation gate |
| `tools/score_policy.py` | Autonomous success / failure-gate report |
