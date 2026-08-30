# Verifying the client against the simulator

How to confirm the VQ1 client talks to the sim and that the **policy**
timed path can arm. The sim is a Windows GUI app that needs an interactive
login, so flight checks have manual steps (flagged **[MANUAL]**).

The timed submission is `FLIGHT_MODE=policy`. HSV-first detection, dry-run
velocity planners, and the old example venv are not this client.

## Paths

- **Simulator:** whatever install you use. Extract it **outside** this repo.
  Log in, select Virtual Qualifier Round 1, start a race. The menu alone
  publishes nothing.
- **Python:** the Windows venv in this repo — `.\winvenv\Scripts\python.exe`.
- **Networking:** MAVLink `udpin 127.0.0.1:14550`, vision `udp 0.0.0.0:5600`.

```powershell
.\winvenv\Scripts\python.exe -c "import torch, ultralytics, cv2, pygame; print(torch.cuda.is_available())"
```

## Step 0 — offline checks (no sim)

From the repo root:

```powershell
.\winvenv\Scripts\python.exe -m unittest discover -p "test_*.py"
```

This covers observation, policy bins / snap / warm-start, pose-train
preflight, detector class aliases, and the classical solvers. It does not
fly the craft.

## Step 1 — launch the sim **[MANUAL]**

Start FlightSim, log in, start a race so telemetry and video stream.

```powershell
.\winvenv\Scripts\python.exe tools\probe_vq1.py --seconds 20
```

Expect `HIGHRES_IMU` present. Missing `ATTITUDE` / `ODOMETRY` /
`TRACK_DATA` is normal on current builds.

## Step 2 — perception

Default detector is YOLO pose (`GATE_DETECTOR_BACKEND=yolo_pose`) with
`YOLO_POSE_MODEL_PATH=models/gate_pose_v5.pt`. HSV confirm and the
global HSV fallback are **off**. Do not calibrate HSV as a prerequisite.

Optional: dump frames for later pose eval (`tools/capture_frames.py`) or
A/B detectors (`tools/eval_detectors.py`). The policy still consumes
keypoints + IMU, not HSV images.

## Step 3 — observation gate **[MANUAL]**

Fly one finished human lap (acro or coach-human) so telem has `kp*_u/v`,
`ahrs_*`, and `active_gate`:

```powershell
.\winvenv\Scripts\python.exe tools\tune_flight.py acro
.\winvenv\Scripts\python.exe tools\eval_observation.py --telem logs\seed\telem_....csv
```

Must print `PASS`. Failure means the detector is tracking a gate you are
not flying to — fix vision before training. `WARN` lines are expected
(rigidity is often inconclusive; coupling is inconclusive when you fly
cleanly).

## Step 4 — policy timed path **[MANUAL]**

Zero human input. Spec §7.

```powershell
.\winvenv\Scripts\python.exe tools\run_policy.py
```

Do not edit `config.py`. Confirm: heartbeat, camera frames, and that the
planner is `policy`. `--panel` is optional debug (camera + observation);
leave it off for a compliant timed run.

Coach / intervene only in `tools/tune_flight.py coach`, never on this path.

## Failure signals

- Hangs at `Waiting for heartbeat...` → sim not in a race, wrong port, or
  firewall.
- VisionRX raises at startup → pose weights missing or wrong path.
- Policy refuses to load → checkpoint `H` / chunk / bins / `n_in` does not
  match what the planner expects; use the checkpoint’s own flags (coach and
  `main.py` read them).
- Craft never leaves the pad in policy → usually a weights / observation
  problem, not hover-thrust. Hover trim (`HOVER_THRUST`) belongs to the
  classical / kalman / assist plants; see [`TUNING.md`](../TUNING.md).
- `src=known` / HSV-only language in old notes → stale. This client is
  YOLO-pose first.
