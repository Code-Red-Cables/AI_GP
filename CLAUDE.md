# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Python client for the **AI Grand Prix (AI-GP)** autonomous drone-racing competition. The client connects to the DCL flight simulator over **MAVLink 2 / UDP**, ingests telemetry and a first-person camera stream, and must fly a racing drone through a sequence of gates with **zero human input** (any human input during a timed run is disqualification).

The committed code on `main` is the organizer-provided **example client**: it connects, arms, and spins a control loop, but perception/planning/control are stubs. The work is filling in the pipeline:

```
Vision + Telemetry → Perception → Planning → Control → Pilot Commands → Sim
```

The vision-first pipeline (PLAN.md §8) is **implemented** on the `modified-starter` branch (perception → planning → attitude+thrust control), verified offline. **HSV is calibrated** (2026-06-03): the Round-1 gate is **red/orange (glowing)** using a two-piece hue mask → `LOWER_HSV=(0,0,80)`, `UPPER_HSV=(15,255,255)`, `LOWER_HSV2=(170,0,80)`, `UPPER_HSV2=(180,255,255)` in `vision/gate_detector.py` to handle the OpenCV hue wrap-around at 0/180 (see `docs/CALIBRATION.md` §2 for details and re-verification steps). The **control strategy was reworked** after discovery that the DCL simulator is a Betaflight-style FPV racer (boots in ACRO, runs in ANGLE self-levelling mode) with **no velocity or position loop** — velocity setpoints via `SET_POSITION_TARGET_LOCAL_NED` are silently ignored (evidence: logs/run_1780516557.jsonl shows 24 m/s uncontrolled climb while commanding descent). Flight control now uses **attitude+thrust** via `SET_ATTITUDE_TARGET` (roll/pitch/yaw + collective thrust), with the planner's velocity command mapped to small lean angles and a thrust that tracks desired vertical velocity. **IMPLEMENTED, PENDING LIVE-SIM TUNING** (gains especially HOVER_THRUST must be tuned). **CAUTION:** `main.py` currently ships with `DRY_RUN=False`, meaning the drone will attempt to fly on startup — set `DRY_RUN=True` for safe perception-only validation (computes commands but sends nothing).

> **`preplanning` branch (current): vision is OFF — the drone flies a fixed waypoint mission.** Instead of reacting to the camera, the planner (`planner.py`) follows an ordered list of position+yaw waypoints from `shared_data['mission']` (defined in `mission.py`): for each waypoint it commands a velocity toward the target NED position (P control, speed-capped) and the waypoint's yaw, advancing once it is within the position radius and heading tolerance (after a short dwell). A waypoint with `yaw=None` **holds the current heading** (no commanded turn, arrival not gated on heading). The default mission is a **clockwise square** (`square_mission(counter_clockwise=False)`, selected by `main.py`; set env `SQUARE_CCW=1` to mirror it counter-clockwise): take off **straight up holding the boot heading** (the takeoff waypoint is `yaw=None`, so there is no spin during the climb), then a 90° turn at each of the four corners, facing along each leg. Drop a `mission.json` next to `main.py` (schema = `mission.save_mission`; `yaw_deg` may be `null` = hold heading) to fly a custom path, or set `SQUARE_SIDE_M` / `SQUARE_ALT_M` / `SQUARE_CCW`. `USE_VISION=False` in `main.py`, so `VisionRX` (and the camera socket / detector) is never started; the body-rate+thrust **controller is unchanged in structure**. The same `compute_target()` → `{'mode':'velocity', 'vel_ned', 'yaw', ...}` contract feeds the controller. Yaw convention (mission/planner): 0 = North, +90 = East, −90 = West, 180 = South. **Note:** the sim's reported yaw is *left-handed* vs NED, so `controller.velocity_to_attitude` rotates the velocity error by **−yaw** when turning it into lean angles — see the Protocol details. Positions are true NED, so the square's *shape* is flown correctly; only the nose direction may be mirrored from the mission's intent (irrelevant with vision off).

> **`manual-control` branch (built off `preplanning`): fly the drone by keyboard to map the course.** Vision and the autonomous planner are OFF (`config.USE_TELEOP=True`, `config.USE_VISION=False`). `setup.py` swaps in `teleop.TeleopPlanner` (same `compute_target()` → `{'mode':'velocity','vel_ned','yaw',...}` contract — the controller is untouched) and starts a `teleop.KeyboardTeleop` background thread (the usual `create_*`/`get_thread_for_join` lifecycle). **Controls:** `W/S` forward/back, `A/D` strafe left/right, `Space`/`C` climb/descend, `Q/E` yaw left/right, **`B` captures the drone's current NED pose** as a waypoint and rewrites `captured_waypoints.json` (`config.CAPTURE_PATH`) in the exact `mission.save_mission` schema — so a captured file can be dropped onto the `preplanning` branch as a `mission.json` and replayed. WASD is mapped along the **physical nose** `(cos yaw, −sin yaw)` (the inverse of `velocity_to_attitude`'s body loop) so motion matches the FPV camera; `Q/E` integrate a heading setpoint the controller turns toward (flip `config.TELEOP_YAW_SIGN` if reversed). The keyboard thread reads true key STATE via Win32 `GetAsyncKeyState` (movement continues while keys are held); a POSIX raw-tty fallback exists for offline use. **All run settings live in `config.py`** (this branch reads NO OS env vars): `USE_TELEOP` (manual vs autonomous), teleop speeds, mission paths, `DRY_RUN`, `MAX_WP_DIST_M`, etc. — edit that file to configure a run. Offline test: `python test_teleop.py`. Set `config.USE_TELEOP=False` to fly the autonomous square/mission instead. **CAUTION:** `DRY_RUN=False` here too — the drone arms and flies on startup (it hovers until you press a key).

### Documentation map
- `PLAN.md` — engineering plan, requirements, original bug list, intended design (source of truth before changing behavior). §8 is the vision-slice plan.
- `docs/IMPLEMENTATION.md` — what's built, module-by-module, the geometry math, and the `shared_data` schema/contract.
- `docs/TESTING.md` — how to run the offline test suites and what each asserts.
- `docs/CALIBRATION.md` — HSV calibration, detector-filter and guidance-gain tuning, reading run logs.
- `reference/VERIFY.md` — launching the sim and the end-to-end verification runbook.
- `reference/AI Grand Prix Tech Specs.pdf` — authoritative spec (extracted text in `reference/_specs_text.txt`).

## Running

The simulator (`FlightSim.exe`, shipped separately, Windows-only) must be **launched first** — `main.py` blocks on `wait_heartbeat()` until the sim is up.

```bash
pip install -r requirements.txt
python main.py
```

- Target runtime is **Python 3.14.2 on Windows 11** (the sim does not run on Linux).
- **Use the bundled interpreter** `C:\Users\rocky\docs\AI_GP\PyAIPilotExample\myenv\Scripts\python.exe` — it has `numpy`/`cv2`/`pymavlink`; the system `py` does not.
- **Offline tests (no sim needed):** `python test_camera_model.py` (geometry sign-checks), `python test_pipeline_smoke.py` (detector→estimator→planner→control-send), and `python test_waypoint_mission.py` (preplanning branch: runs the real planner against a kinematic drone and asserts it flies the square mission — visits every waypoint in order, on-position+heading, within the speed caps). All print `ALL ... PASSED`. There is no linter/build step.
- **Safety flags in `main.py`:** `DRY_RUN` (default **False** — CAUTION: the drone will attempt to fly on startup; set to True to compute & log guidance only without sending flight setpoints), `DEBUG_VISION` (write detection overlays; keep off for timed runs), `LOGGING` (JSONL run logs under `logs/`).
- End-to-end verification against the sim (manual login required): see `reference/VERIFY.md`.
- Connection defaults live at the top of `main.py` (MAVLink `udpin:127.0.0.1:14550`) and `vision_rx.py` (camera `udp:0.0.0.0:5600`). Edit these to talk to a remote sim.

**Startup sequence (when `DRY_RUN=False`):** The DCL simulator is a Betaflight-style FPV racer that boots in ACRO (raw rate) mode, which ignores our commands when armed. Before arming, `main.py` establishes a setpoint (attitude-hold) stream, switches the sim to ANGLE (self-levelling attitude) mode via `request_offboard_mode()` (which prints the available sim mode names and the one requested), then keeps the stream alive across the mode switch, and finally arms. This handshake is required because (1) ACRO mode ignores all commands, and (2) the mode switch is rejected unless a setpoint stream is already flowing. In `DRY_RUN=True` mode, this entire block is skipped — no setpoints are sent, and the drone remains grounded.

## Architecture

`main.py` calls `setup.py:setup_components()`, which opens the MAVLink connection, waits for a heartbeat, and constructs the components below that all share a single `shared_data` dict. `main.py` then arms the drone and runs `while True: controller.update()`.

Concurrency model — **one main-thread control loop + background daemon threads**. The full pipeline is `Vision/Telemetry → Perception → Planning → Control`:

| Component | File | Thread? | Role |
|-----------|------|---------|------|
| `Controller` | `controller.py` | main loop | **Outbound**: each tick calls `Planner.compute_target()`, maps the desired velocity → lean angles via `velocity_to_attitude` (body frame, rotated by **−yaw**), then closes the attitude loop client-side and sends **body rates + thrust** (`SET_ATTITUDE_TARGET` with attitude **ignored**, ≤`CONTROL_HZ`=60); gated by `shared_data['dry_run']`. Owns the offboard-entry sequence (setpoint-stream prime → mode switch → arm), plus `arm()` and sim-reset. |
| `Planner` | `planner.py` | (called in loop) | **Preplanning branch:** follows the fixed waypoint mission (`shared_data['mission']`) — steers toward the current waypoint's NED position + yaw (or holds heading when the waypoint's `yaw=None`), advances on arrival → writes `shared_data['target']`; telemetry watchdog → hover; altitude-envelope guard (climb runaway) **and horizontal-envelope guard** (`dist > MAX_WP_DIST_M=25 m` from the active waypoint → hover so the velocity loop brakes back). No vision. |
| `MAVLinkRX` | `mavlink_rx.py` | background | **Inbound telemetry**: parses + **stores** HEARTBEAT/ATTITUDE/ODOMETRY/IMU/race/gates/collision into `shared_data` under the lock |
| `VisionRX` | `vision_rx.py` | background | **Inbound video**: reassembles chunked JPEG → `cv2` image → `detect_gate` → `estimate_gate` → `shared_data['vision']` |
| `TimeSync` | `timesync.py` | background | Sends MAVLink TIMESYNC requests at 10 Hz |
| `Logger` | `logger.py` | background | Snapshots `shared_data` to `logs/run_*.jsonl` (~10 Hz) for offline tuning |

Pure helper modules (no threads): `camera_model.py` (intrinsics + frame transforms, numpy-only), `gate_detector.py` (HSV→square), `gate_estimator.py` (detection+camera model → gate body/NED pose via size or `solvePnP`).

**`shared_data` is the cross-thread blackboard**, guarded by `shared_data['lock']` (an `RLock`, created in `main.py`). Producers (RX threads) write the latest state; consumers (`Planner`/`Controller`) read it. Never reach into another thread's internals — `shared_data` is the only contract. Its schema is documented in `PLAN.md` §8.6.

**Component lifecycle convention**: stateful threaded components are built via a `create_*` classmethod (e.g. `MAVLinkRX.create_mavlink_rx`) that constructs the object, starts its thread, and sets `is_running = True`. Shutdown is two-step: `get_thread_for_join()` flips `is_running = False` and returns the thread so `main.py` can `join()` it. Follow this pattern for new background components.

## Protocol details that aren't obvious from a single file

- **Flight control via body-rate + thrust (not velocity, not held-attitude):** The DCL simulator is a Betaflight-style FPV racer. It has **no velocity or position loop** — `SET_POSITION_TARGET_LOCAL_NED` velocity setpoints are silently ignored (logs/run_1780516557.jsonl: 24 m/s uncontrolled climb while commanding descent) — and it does **not** hold a commanded attitude either (commanding level left it pinned at ~75° pitch), so the FC only honours **body RATES**. `controller.py` therefore sends `SET_ATTITUDE_TARGET` with the attitude quaternion **ignored** (`send_rate_target`: roll/pitch/yaw body rates + collective thrust) and closes the attitude loop itself: `velocity_to_attitude` maps the `Planner`'s desired NED velocity → desired lean angles (roll/pitch) + thrust (`thrust = HOVER_THRUST + KP_THRUST*(vz_now - vd)`), then an outer P loop turns the lean-angle error into commanded body rates (per-axis `RATE_SIGN_*`).
- **The sim's reported yaw is LEFT-HANDED vs NED** (physical forward = `(cos y, −sin y)`): `velocity_to_attitude` rotates the world velocity error by **−yaw** before splitting it into forward (pitch) and lateral (roll) lean. Sign constants: `LEAN_SIGN_FWD=+1` (`pitch+ → body-forward`, verified clean at yaw 0 = North and yaw ±180 = South); `LEAN_SIGN_LAT=+1` (textbook "roll right to go right", **pending live confirmation** — never exercised on a clean fore/aft leg). History: the original `+yaw` rotation flew the drone **East while commanded West** on the yaw=−90 leg (180 m off a 5 m square, log run_1781506925); a no-rotation "world-referenced" attempt then flew **straight off in the boot heading** (run_1781509561). The `−yaw` rotation flies the square in offline closed-loop sim against a left-handed-yaw + drag model. Velocity send path (`send_velocity_ned`) is retained for reference/offline tests but is **not used in flight**.
- **Mode entry handshake (required):** The sim boots in ACRO mode and must be switched to ANGLE before armed commands are accepted. The mode switch is rejected unless a setpoint stream is already flowing. `controller.py` provides `hold()` (send a single attitude-hold setpoint), `prime_setpoint_stream(seconds, hz)` (stream holds for a duration, stays under 100 Hz spec cap), and `request_offboard_mode()` (resolve and request ANGLE, STABILIZE, STABILIZED, ALTCTL, GUIDED by name from the autopilot's mode map, or fall back to PX4 custom mode id 6). `main.py` uses these before arming (skipped in `DRY_RUN=True` where no setpoints are sent).
- **Spec note:** The spec lists both `SET_POSITION_TARGET_LOCAL_NED` and `SET_ATTITUDE_TARGET` as supported; this sim ignores the former. Not actuator control — the original example's motor control was a bug.
- **No GPS / global position** — everything is **local NED**, origin at the arm point. Z is negative-up.
- **Custom payloads ride inside `ENCAPSULATED_DATA`**, demuxed by a leading byte: `1` = race status, `2` = track info (see `ENCAPSULATED_*_MSG_ID` in `mavlink_rx.py`). These are hand-unpacked with `struct`.
- **`DATA_TRANSMISSION_HANDSHAKE` is repurposed** as the header for chunked **track/gate geometry**; `mavlink_rx.py` reassembles the numbered chunks (`track_chunks` / `expected_num_track_chunks`) before decoding gate positions/orientations.
- **Vision is a separate UDP socket** (not MAVLink): JPEG frames arrive as numbered chunks keyed by `frame_id`, reassembled in `VisionRX._vision_loop` and decoded with `cv2.imdecode`. **Only one process can bind UDP 5600 at a time** — a running `main.py` (its `VisionRX`) holds it, so `tools/capture_frames.py` will fail with `WinError 10048` until you stop the client. Don't run two clients at once either.

## Hard constraints from the spec (enforce when writing control code)

- **Outbound command rate must stay < 100 Hz.** `controller.py` sets `CONTROL_HZ = 60` (safe margin).
- Heartbeat must be kept alive at **≥ 2 Hz**; physics runs at 120 Hz, camera at 30 Hz / 640×360; max run length **8 minutes**.
- Course geometry and physics are deterministic and identical across teams.

## Repo conventions

- Branching: `main` holds the pristine organizer starter; team changes live on feature branches (e.g. `modified-starter`) so diffs against the baseline stay clean.
- Virtual environments (`venv/`, `myenv/`) and `__pycache__/` are git-ignored — never commit them.
