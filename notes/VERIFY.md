# Verifying the client against the simulator

How to run the autonomous pilot against the DCL flight simulator and confirm each
pipeline stage works. The sim is a Windows GUI app that requires an interactive
login, so full verification has manual steps (flagged **[MANUAL]**).

## Paths
- **Simulator launcher:** `C:\Users\rocky\docs\AI_GP\Simulator\FlightSim.exe`
  (wraps the UE5 binary `Simulator\FlightSim\Binaries\Win64\DCGame-Win64-Shipping.exe`).
- **Python interpreter (has pymavlink + numpy + cv2):**
  `C:\Users\rocky\docs\AI_GP\PyAIPilotExample\myenv\Scripts\python.exe`
- **Client repo:** `C:\Users\rocky\docs\AI_GP\AI_GP`
- **Networking (hardcoded, no config file):** MAVLink on `udpin 127.0.0.1:14550`,
  vision on `udp 0.0.0.0:5600`.

## Step 0 — offline checks (no sim needed)
From the repo dir, run the unit + smoke tests with the bundled interpreter:
```
& "C:\Users\rocky\docs\AI_GP\PyAIPilotExample\myenv\Scripts\python.exe" test_camera_model.py
& "C:\Users\rocky\docs\AI_GP\PyAIPilotExample\myenv\Scripts\python.exe" test_pipeline_smoke.py
```
Both must print `ALL ... PASSED`. This validates geometry, detection→estimation→
planning→control-send wiring without flying.

## Step 1 — launch the sim **[MANUAL]**
```
Start-Process "C:\Users\rocky\docs\AI_GP\Simulator\FlightSim.exe"
```
Then **log in** with your AI-GP simulator account, **select Virtual Qualifier
Round 1**, and **start the race** so the drone is in the world and telemetry +
video are streaming. (None of this can be automated.)

## Step 2 — capture frames & calibrate HSV (one-time, **[MANUAL]** for flying the view)
With video streaming, dump frames, then tune thresholds:
```
& "...\myenv\Scripts\python.exe" tools/capture_frames.py 50
& "...\myenv\Scripts\python.exe" tools/hsv_tuner.py notes/frames/frame_0000.png
```
Paste the printed `LOWER_HSV` / `UPPER_HSV` into `vision/gate_detector.py`. Sanity-
check detection on a saved frame:
```
& "...\myenv\Scripts\python.exe" vision/gate_detector.py notes/frames/frame_0000.png
```
(writes `_detect_debug.png`).

## Step 3 — perception-only run (DRY_RUN, the safe default)
`main.py` ships with `DRY_RUN = True` and `DEBUG_VISION` available. Run the client
while the sim is in a race:
```
cd C:\Users\rocky\docs\AI_GP\AI_GP
& "C:\Users\rocky\docs\AI_GP\PyAIPilotExample\myenv\Scripts\python.exe" main.py
```
**Expected console output:**
```
Waiting for heartbeat...
Connected to system: <N>
Setting up MAVLink rx...
Setting up Timesync loop...
Listening for camera frames...
Setting up planner...
Logging run to logs/run_<ts>.jsonl
Arming drone...
Starting control loop... (DRY_RUN=True)
[DRY] src=...  range=...m vel_ned=(...) yaw=...
```
Because `DRY_RUN=True`, the drone does **not** move — the `[DRY]` lines show what the
planner *would* command. Confirm: heartbeat connects, the gate list is received
(check `logs/...jsonl` for a `gates` array), `src=vision` appears when a gate is in
view (else `src=known`), and detection overlays look right (set `DEBUG_VISION=True`).

## Step 4 — enable flight
Once the `[DRY]` commands look sane, set `DRY_RUN = False` in `main.py` and re-run.
The drone should steer toward the active gate. Tune the gains in `planner.py`
(`MAX_SPEED`, `KP_POS`, `PASS_THROUGH_DIST`) against the deterministic course using
the JSONL logs. Keep `DEBUG_VISION=False` for any compliant timed run (no human
interaction allowed — disqualification per spec §7).

## Failure signals
- Hangs at `Waiting for heartbeat...` → sim not in a race, wrong port, or firewall.
- `src=known` only, never `vision` → HSV thresholds need calibration (Step 2).
- Watchdog `src=watchdog_hover` → telemetry stopped arriving (the planner safely holds).
