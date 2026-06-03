# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Python client for the **AI Grand Prix (AI-GP)** autonomous drone-racing competition. The client connects to the DCL flight simulator over **MAVLink 2 / UDP**, ingests telemetry and a first-person camera stream, and must fly a racing drone through a sequence of gates with **zero human input** (any human input during a timed run is disqualification).

The committed code on `main` is the organizer-provided **example client**: it connects, arms, and spins a control loop, but perception/planning/control are stubs. The work is filling in the pipeline:

```
Vision + Telemetry → Perception → Planning → Control → Pilot Commands → Sim
```

`PLAN.md` is the engineering plan and the **source of truth for requirements, the current bug list, and intended design** — read it before changing behavior. The authoritative spec is `notes/AI Grand Prix Tech Specs.pdf` (extracted text in `notes/_specs_text.txt`).

## Running

The simulator (`FlightSim.exe`, shipped separately, Windows-only) must be **launched first** — `main.py` blocks on `wait_heartbeat()` until the sim is up.

```bash
pip install -r requirements.txt
python main.py
```

- Target runtime is **Python 3.14.2 on Windows 11** (the sim does not run on Linux).
- **Use the bundled interpreter** `C:\Users\rocky\docs\AI_GP\PyAIPilotExample\myenv\Scripts\python.exe` — it has `numpy`/`cv2`/`pymavlink`; the system `py` does not.
- **Offline tests (no sim needed):** `python test_camera_model.py` (geometry sign-checks) and `python test_pipeline_smoke.py` (detector→estimator→planner→control-send). Both print `ALL ... PASSED`. There is no linter/build step.
- **Safety flags in `main.py`:** `DRY_RUN` (default **True** — computes & logs guidance but sends no flight setpoints; flip to False to actually fly), `DEBUG_VISION` (write detection overlays; keep off for timed runs), `LOGGING` (JSONL run logs under `logs/`).
- End-to-end verification against the sim (manual login required): see `notes/VERIFY.md`.
- Connection defaults live at the top of `main.py` (MAVLink `udpin:127.0.0.1:14550`) and `vision_rx.py` (camera `udp:0.0.0.0:5600`). Edit these to talk to a remote sim.

## Architecture

`main.py` calls `setup.py:setup_components()`, which opens the MAVLink connection, waits for a heartbeat, and constructs the components below that all share a single `shared_data` dict. `main.py` then arms the drone and runs `while True: controller.update()`.

Concurrency model — **one main-thread control loop + background daemon threads**. The full pipeline is `Vision/Telemetry → Perception → Planning → Control`:

| Component | File | Thread? | Role |
|-----------|------|---------|------|
| `Controller` | `controller.py` | main loop | **Outbound**: each tick calls `Planner.compute_target()` then sends NED-velocity `SET_POSITION_TARGET_LOCAL_NED` (≤`CONTROL_HZ`=60); gated by `shared_data['dry_run']`. Owns `arm()` / sim-reset |
| `Planner` | `planner.py` | (called in loop) | Picks the active gate (fresh vision else known geometry by `active_gate_index`) → writes `shared_data['target']`; telemetry watchdog → hover |
| `MAVLinkRX` | `mavlink_rx.py` | background | **Inbound telemetry**: parses + **stores** HEARTBEAT/ATTITUDE/ODOMETRY/IMU/race/gates/collision into `shared_data` under the lock |
| `VisionRX` | `vision_rx.py` | background | **Inbound video**: reassembles chunked JPEG → `cv2` image → `detect_gate` → `estimate_gate` → `shared_data['vision']` |
| `TimeSync` | `timesync.py` | background | Sends MAVLink TIMESYNC requests at 10 Hz |
| `Logger` | `logger.py` | background | Snapshots `shared_data` to `logs/run_*.jsonl` (~10 Hz) for offline tuning |

Pure helper modules (no threads): `camera_model.py` (intrinsics + frame transforms, numpy-only), `gate_detector.py` (HSV→square), `gate_estimator.py` (detection+camera model → gate body/NED pose via size or `solvePnP`).

**`shared_data` is the cross-thread blackboard**, guarded by `shared_data['lock']` (an `RLock`, created in `main.py`). Producers (RX threads) write the latest state; consumers (`Planner`/`Controller`) read it. Never reach into another thread's internals — `shared_data` is the only contract. Its schema is documented in `PLAN.md` §8.6.

**Component lifecycle convention**: stateful threaded components are built via a `create_*` classmethod (e.g. `MAVLinkRX.create_mavlink_rx`) that constructs the object, starts its thread, and sets `is_running = True`. Shutdown is two-step: `get_thread_for_join()` flips `is_running = False` and returns the thread so `main.py` can `join()` it. Follow this pattern for new background components.

## Protocol details that aren't obvious from a single file

- **Control commands**: the spec only supports `SET_POSITION_TARGET_LOCAL_NED` and `SET_ATTITUDE_TARGET` from Client→Sim (NOT actuator control — the original example's motor control was a bug). `controller.py` uses NED **velocity** via `set_position_target_local_ned_send` with a type-mask that ignores position/accel/yaw-rate.
- **No GPS / global position** — everything is **local NED**, origin at the arm point. Z is negative-up.
- **Custom payloads ride inside `ENCAPSULATED_DATA`**, demuxed by a leading byte: `1` = race status, `2` = track info (see `ENCAPSULATED_*_MSG_ID` in `mavlink_rx.py`). These are hand-unpacked with `struct`.
- **`DATA_TRANSMISSION_HANDSHAKE` is repurposed** as the header for chunked **track/gate geometry**; `mavlink_rx.py` reassembles the numbered chunks (`track_chunks` / `expected_num_track_chunks`) before decoding gate positions/orientations.
- **Vision is a separate UDP socket** (not MAVLink): JPEG frames arrive as numbered chunks keyed by `frame_id`, reassembled in `VisionRX._vision_loop` and decoded with `cv2.imdecode`.

## Hard constraints from the spec (enforce when writing control code)

- **Outbound command rate must stay < 100 Hz.** Note `controller.py` sets `CONTROL_HZ = 250`, which violates this — see `PLAN.md` for the full bug list before relying on loop timing.
- Heartbeat must be kept alive at **≥ 2 Hz**; physics runs at 120 Hz, camera at 30 Hz / 640×360; max run length **8 minutes**.
- Course geometry and physics are deterministic and identical across teams.

## Repo conventions

- Branching: `main` holds the pristine organizer starter; team changes live on feature branches (e.g. `modified-starter`) so diffs against the baseline stay clean.
- Virtual environments (`venv/`, `myenv/`) and `__pycache__/` are git-ignored — never commit them.
