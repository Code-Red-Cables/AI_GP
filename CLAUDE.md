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
- There is **no test suite, linter, or build step** — it's a runnable script. "Run" means launching the sim and then `main.py`, and watching console output / observed flight.
- Connection defaults live at the top of `main.py` (MAVLink `udpin:127.0.0.1:14550`) and `vision_rx.py` (camera `udp:0.0.0.0:5600`). Edit these to talk to a remote sim.

## Architecture

`main.py` calls `setup.py:setup_components()`, which opens the MAVLink connection, waits for a heartbeat, and constructs four components that all share a single `shared_data` dict. `main.py` then arms the drone and runs `while True: controller.update()`.

Concurrency model — **one main-thread control loop + three daemon RX threads**:

| Component | File | Thread? | Role |
|-----------|------|---------|------|
| `Controller` | `controller.py` | main loop | **Outbound**: sends motor/attitude/position setpoints to the sim each tick; owns `arm()` and sim-reset |
| `MAVLinkRX` | `mavlink_rx.py` | background | **Inbound telemetry**: parses HEARTBEAT, ATTITUDE, ODOMETRY, IMU, race status, gate/track data |
| `VisionRX` | `vision_rx.py` | background | **Inbound video**: reassembles chunked JPEG UDP packets → `cv2` image in `process_frame()` |
| `TimeSync` | `timesync.py` | background | Sends MAVLink TIMESYNC requests at 10 Hz to keep clocks aligned |

**`shared_data` is the cross-thread blackboard.** It is threaded into every component and is the intended integration point: RX threads should write parsed telemetry/vision into it, and `Controller.update()` should read it to decide commands. In the example code the RX handlers parse messages but **store nothing** — wiring producers and consumers through `shared_data` is the core integration task.

**Component lifecycle convention**: stateful components are built via a `create_*` classmethod (e.g. `MAVLinkRX.create_mavlink_rx`) that constructs the object, starts its thread, and sets `is_running = True`. Shutdown is two-step: `get_thread_for_join()` flips `is_running = False` and returns the thread so `main.py` can `join()` it. Follow this pattern for new background components.

## Protocol details that aren't obvious from a single file

- **Control commands** use `set_actuator_control_target` (direct motor), `set_attitude_target` (rate/thrust), or `set_position_target_local_ned` (velocity). The example sends motor control only; the others are commented out in `Controller.update()`.
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
