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
& "...\myenv\Scripts\python.exe" tools/hsv_tuner.py reference/frames/frame_0000.png
```
Paste the printed `LOWER_HSV` / `UPPER_HSV` into `vision/gate_detector.py`. Sanity-
check detection on a saved frame:
```
& "...\myenv\Scripts\python.exe" vision/gate_detector.py reference/frames/frame_0000.png
```
(writes `_detect_debug.png`).

## Step 3 — perception-only run (DRY_RUN=True for safe validation)
**CAUTION:** `main.py` currently ships with `DRY_RUN = False` (the drone will attempt to fly).
Set `DRY_RUN = True` to validate perception safely before flying. Run the client while the sim is in a race:
```
cd C:\Users\rocky\docs\AI_GP\AI_GP
& "C:\Users\rocky\docs\AI_GP\PyAIPilotExample\myenv\Scripts\python.exe" main.py
```
**Expected console output (with `DRY_RUN=True`):**
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
[DRY] src=...  range=...m vel_ned=(+...,+...,-...) att=(r...,p...,y...) thr=...
```
Because `DRY_RUN=True`, the drone does **not** move — the `[DRY]` lines show what the
planner and controller would command (desired velocity, then the attitude+thrust mapping).
The "Priming" and "Requested control mode" lines are **skipped** in `DRY_RUN=True`.
Confirm: heartbeat connects, the gate list is received (check `logs/...jsonl` for a `gates` array),
`src=vision` appears when a gate is in view (else `src=known`), and detection overlays look right
(set `DEBUG_VISION=True`).

## Step 4 — enable flight & tune gains

Once the `[DRY]` commands look sane, set `DRY_RUN = False` in `main.py` and re-run.
The drone should steer toward the active gate. On startup you will see:
```
Priming attitude-hold stream...
Sim mode map: [list of mode names or 'unknown (no mode_mapping)']
Requested control mode: <ANGLE|STABILIZE|...>
Arming drone...
Starting control loop... (DRY_RUN=False)
[FLY] src=...  range=...m vel_ned=(+...,+...,-...) att=(r...,p...,y...) thr=...
```

**Watch for controlled startup & attitude mode confirmation:** After the mode-switch message,
the sim's HUD (top-right) should show **"FLIGHT MODE: ANGLE"** (self-levelling attitude mode).
The drone should hold steady or take off smoothly once armed. If it climbs uncontrollably
(24+ m/s vertical as in logs/run_1780516557.jsonl), something is wrong with the attitude
control setup — **check the console output first**: if `Sim mode map` is empty or shows unexpected
mode names, the mode switch may have failed; if mode-switch succeeded but the drone still climbs,
re-verify `HOVER_THRUST` is tuned correctly (see below).

**Tuning checklist (in order):**
1. **Attitude gains (MUST DO FIRST)** — `controller.py` constants:
   - Confirm **`HOVER_THRUST` holds altitude level** when hovering with zero lean. Watch
     the `att=(r,p,y)` line — when hovering, roll/pitch should be ~0.0, and altitude should
     be stable. If it climbs, lower `HOVER_THRUST`; if it sinks, raise it.
   - Once altitude is solid, raise `KP_LEAN` (start ~0.15) and test forward/lateral flight.
     Watch lean angles in the `att=` line; they should scale reasonably with commanded velocity.
   - If the drone oscillates, lower `KP_LEAN` and/or check `HOVER_THRUST` again.
2. **Guidance gains** (after attitude is stable) — `planner.py` constants:
   - `MAX_SPEED`, `MAX_VSPEED`, `KP_POS`, `PASS_THROUGH_DIST` per `docs/CALIBRATION.md` §5b.

Tune against the deterministic course using the JSONL logs. Keep `DEBUG_VISION=False` for
any compliant timed run (no human interaction allowed — disqualification per spec §7).

**Monitor the altitude envelope:** the planner enforces a 15 m ceiling above the arm point;
if the drone climbs past it, `target['source']` switches to `'alt_guard'` and the drone descends.
If this happens early in a run, the root cause is usually under-tuned `HOVER_THRUST` (drone's
collective thrust is too high for level flight) — re-calibrate it per the checklist above.
Review the JSONL log with `Get-Content logs\run_*.jsonl | ConvertFrom-Json | Select-Object -Last 20` to inspect
altitude traces and `source` field.

## Failure signals
- Hangs at `Waiting for heartbeat...` → sim not in a race, wrong port, or firewall.
- **Drone arms but never takes off / doesn't move (in DRY_RUN=True mode)** → expected behavior.
  The planner computes setpoints (the `[DRY]` lines) but `controller.update()` does **not** transmit them.
  Set `DRY_RUN=False` in `main.py` (Step 4) to actually fly.
- **Drone arms but doesn't move (in DRY_RUN=False)** → check that the mode switch succeeded.
  Look for "Requested control mode: ANGLE" in the console; if it's absent or shows a different mode,
  the mode switch may have failed. Check the sim's HUD — it should show "FLIGHT MODE: ANGLE".
  If the mode is wrong, try manually investigating the sim's mode names and editing
  `controller.py` `request_offboard_mode()` to try them.
- **Drone climbs uncontrollably (24+ m/s vertical or ~5.5 m/s steady climb)** → usually under-tuned `HOVER_THRUST` in
  `controller.py`. The attitude control can track desired vertical velocity, but if the
  collective thrust baseline is wrong, the drone climbs (or sinks). First live test (logs/run_1780521287.jsonl) 
  showed steady climb at ~5.5 m/s with `HOVER_THRUST=0.5` because that was above the racing quad's true hover 
  throttle and `KP_THRUST=0.05` was too weak to overcome it. Current starting values: `HOVER_THRUST=0.35`, 
  `KP_THRUST=0.15`. **Re-calibrate `HOVER_THRUST`:** arm the drone, steer toward a gate slowly, and watch the 
  `[FLY]` console line altitude and the `att=` angles. If hovering level causes climb, lower `HOVER_THRUST` further. 
  Once level flight is stable, raise `KP_LEAN` for responsiveness. See Step 4 tuning checklist.
- `src=known` only, never `vision` → HSV thresholds need calibration (Step 2).
- Watchdog `src=watchdog_hover` → telemetry stopped arriving (the planner safely holds).
