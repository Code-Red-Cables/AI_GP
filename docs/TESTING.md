# Testing guide

How to test the pipeline. Most of it runs **offline with no simulator** — that's
deliberate, so logic bugs are caught before flying. End-to-end on-sim verification
(which needs a human login) is in [`../reference/VERIFY.md`](../reference/VERIFY.md).

## The interpreter
All Python must run on the bundled venv (it has `numpy`, `cv2`, `pymavlink`); the
system `py` does **not** have these.

```
$PY = "C:\Users\rocky\docs\AI_GP\PyAIPilotExample\myenv\Scripts\python.exe"
```
Run everything from the repo root (`C:\Users\rocky\docs\AI_GP\AI_GP`).

## 1. Offline test suites (run these first, every time)

```powershell
& $PY test_camera_model.py        # geometry sign-checks
& $PY test_pipeline_smoke.py      # detector -> estimator -> planner -> control-send
& $PY test_waypoint_mission.py    # preplanning: planner flies the square (stop-at-each)
& $PY test_spline_mission.py      # spline-path: SplinePlanner flies through continuously
```
Each must end with `ALL ... PASSED`. Combined they take < 2 s. (The vision tests are no-ops
of behavior on the vision-off branches but still validate the geometry/contract; the
mission tests are the relevant ones for the autonomous-flight branches.)

### `test_camera_model.py` — geometry (8 checks)
Validates the frame math in isolation (numpy only). Each asserts a *physical*
expectation, so a flipped sign fails loudly:

| test | asserts |
|------|---------|
| project/deproject round-trip | `project(deproject(u,v,Z)) == (u,v)` |
| straight-ahead level | gate at body `[10,0,0]` → `u≈320`, `v>180` (lower half: camera looks up) |
| straight-ahead 20° up | direction `[cos20,0,−sin20]` → image center `(320,180)` |
| ahead & right | body `[10,2,0]` → `u>320` |
| ahead & below | body `[10,0,2]` → `v` greater than the level case |
| range-from-size | `range_from_size(96) ≈ 5.0` |
| body→ned | identity at zero attitude; yaw 90° turns +x (fwd) into +y (East); round-trips |
| cam↔body | `cam_to_body(body_to_cam(p)) ≈ p` |

If you change `R_CB`, the tilt, or the rotation conventions, these are the guardrail
— **fix the code to match the physics, never weaken the asserts.**

### `test_pipeline_smoke.py` — integration (6 checks)
Runs the whole chain on a synthetic gate frame, no sim:

1. **detect + estimate** — synthetic gate is found; with the 20° tilt a gate at
   image center is forward (`x>0`), centered (`y≈0`), **up** (`z<0`), elevation ≈
   +20°, range > 0. Confirms detector + camera model + PnP agree.
2. **planner (vision)** — a fresh vision estimate yields `source∈{vision,vision_level}`
   and a forward (North+) velocity within `MAX_SPEED`.
3. **planner (known geometry)** — with no vision but a known gate at `(10,0,−2)` and
   the drone at origin, `source=known` and it heads straight North.
4. **planner (watchdog)** — empty `shared_data` → zero velocity, `source=*hover*`.
5. **controller send path** — `send_attitude_target()` builds a valid 9-arg
   `set_attitude_target_send` call with quaternion + thrust (captured via a fake connection; no socket).
6. **velocity→attitude sign checks** — `velocity_to_attitude()` correctly maps: forward velocity →
   nose-down pitch, rightward velocity → right roll, climb command → thrust > HOVER_THRUST, and caps
   lean angles at `MAX_LEAN_RAD`.

### `test_waypoint_mission.py` — preplanning planner (5 checks)
Runs the real `Planner` against a kinematic drone (integrate the commanded velocity; slew yaw at the controller's cap). Asserts the square mission is built correctly (CCW corners, all `yaw=None`), flown end-to-end it visits every waypoint in order on-position+heading, the commanded velocity never exceeds the speed caps, the clockwise mirror works, and a mission survives a JSON save/load round-trip.

### `test_spline_mission.py` — spline-path planner (4 checks)
Runs the real `SplinePlanner` against the same kinematic drone:

1. **spline interpolates** — `build_spline_path()` passes through every control point; arc length is monotone.
2. **flight visits waypoints** — flown end-to-end the drone passes within `LOOKAHEAD_M` of every intermediate waypoint and settles on the last; the mission completes.
3. **continuous cruise** — across the cruise region (excluding the intended final-approach taper) the commanded horizontal speed stays ≥ 0.9·`CRUISE_SPEED` — i.e. it flies *through* the waypoints instead of stalling at each one (the distinguishing property vs. `Planner`).
4. **speed caps** — commanded velocity never exceeds `MAX_SPEED` / `MAX_VSPEED`.

## 2. Import / compile check (catches wiring breakage)

```powershell
& $PY -m py_compile camera_model.py gate_estimator.py planner.py controller.py `
    logger.py mavlink_rx.py vision_rx.py setup.py main.py timesync.py `
    vision/gate_detector.py tools/capture_frames.py tools/hsv_tuner.py
& $PY -c "import setup; from vision.gate_detector import detect_gate; print('import graph OK')"
```
Importing `setup` loads the entire component graph **without** connecting to the
sim — a fast way to catch interface mismatches after edits.

## 3. Detector self-test on a frame

```powershell
& $PY vision/gate_detector.py                       # synthetic gate -> _detect_debug.png
& $PY vision/gate_detector.py reference/frames/frame_0000.png   # a real captured frame
```
Prints the `GateDetection` and writes an annotated `_detect_debug.png`. Use this
while calibrating HSV (see [`CALIBRATION.md`](CALIBRATION.md)).

## 4. On-sim verification (manual)
Requires launching the sim and logging in — full runbook in
[`../reference/VERIFY.md`](../reference/VERIFY.md). Summary: launch sim → log in / start a
race → `& $PY main.py` (ships in `DRY_RUN`, so it logs `[DRY]` guidance without
flying) → confirm heartbeat connects, gates are received, and `[DRY]` commands look
sane → set `DRY_RUN=False` to fly.

## Adding tests
Keep new logic offline-testable: pure functions (detector, estimator, camera model)
take inputs and return values — assert on those. For planner/controller, build a
fake `shared_data` dict (with a `threading.RLock()` under `'lock'`) and, for the
send path, a fake connection object that captures the MAVLink call args — see
`test_pipeline_smoke.py:test_controller_send_path` for the pattern. Both test files
exit non-zero on failure, so they drop straight into any CI later.
