# AI Grand Prix — Autonomous Pilot: Implementation Plan

> Planning document for engineering agents. Read this before touching code.
> Source of truth for requirements: `notes/AI Grand Prix Tech Specs.pdf`
> (doc `VADR-TS-001`, issue 00.02). Extracted text: `notes/_specs_text.txt`.

---

## 1. What this project is

We are building an **autonomous drone-racing pilot** for the AI Grand Prix Virtual
Qualifier (Round One). Our Python client connects to the **DCL simulator** over
**MAVLink 2 / UDP**, receives telemetry and a first-person camera stream, and must
fly the drone through a sequence of gates — **start gate → intermediate gates →
finish gate** — with **no human interaction** (human input during a timed run =
disqualification).

The repo today is the organizer-provided **example client**: it connects, arms,
and spins a control loop, but every perception/planning/control stage is a stub.
Our job is to fill in the pipeline:

```
Vision + Telemetry → Perception → Planning → Control → Pilot Commands → Sim
```

### Hard constraints (from spec)
- **Command rate: < 100 Hz** (Client → Sim). Do not exceed.
- **Physics runs at 120 Hz**; camera stream at **30 Hz**, 640×360.
- **Minimum heartbeat: 2 Hz** — client must keep MAVLink alive.
- **Max run duration: 8 minutes.**
- **No GPS / global position.** Local NED only. Origin = arm point on the ground.
- Runtime: Python 3.14.2 on **Windows 11** (Linux unsupported by the sim). 8GB VRAM GPU.
- Course geometry & physics are **deterministic and identical** for all teams.

---

## 2. Current code map

| File | Role | State |
|------|------|-------|
| `main.py` | Entry point: setup → `arm()` → `while: controller.update()` → join threads | Works, but exit path is broken (see bugs) |
| `setup.py` | Wires up MAVLink connection + all components, returns dict | **Bug: TimeSync never started** |
| `controller.py` | Outbound control: motor / attitude / position senders + `Controller` loop | **Stub commands; spec-violating** |
| `mavlink_rx.py` | Inbound telemetry RX thread; parses HEARTBEAT, ATTITUDE, ODOMETRY, IMU, race status, track/gate data | **Parses everything, stores nothing** |
| `vision_rx.py` | Inbound FPV camera RX thread; reassembles chunked JPEG → `cv2` image | Reassembly works; **`process_frame()` is a no-op** |
| `timesync.py` | TIMESYNC request loop @ 10 Hz | Logic fine; **not instantiated correctly** |
| `requirements.txt` | `pymavlink`, `opencv-python`, `numpy`, `matplotlib`, `keyboard` | OK |
| `myenv/` | Pre-built virtualenv (Python 3.14, cv2, numpy, etc.) | Use this interpreter |

**`shared_data` dict** is created in `main.py` and threaded through every component
as the intended cross-thread blackboard — **but nothing currently writes to it.**
This is the central integration point the whole pipeline must use.

---

## 3. Known bugs / spec violations — fix these first

These are pre-existing defects in the example. Triage before building features.

1. **TimeSync never runs.** `setup.py:27` calls `TimeSync(sim_conn, shared_data)`
   (the constructor), not `TimeSync.create_timesync(...)`. The constructor leaves
   `thread = None` and `is_running = False`, so no TIMESYNC is ever sent.
   → Change to `TimeSync.create_timesync(sim_conn, shared_data)`.

2. **Exit path crashes.** `main.py:34` calls
   `ts_loop.get_thread_for_join().join(...)`. With bug #1, `thread` is `None` →
   `AttributeError` on `None.join`. Fixing #1 resolves this; also guard against
   `None` threads in the join sequence.

3. **Command rate exceeds spec.** `controller.py:127` sets `CONTROL_HZ = 250`, but
   spec caps Client→Sim commands at **< 100 Hz**. Set to e.g. **50–90 Hz**.

4. **Control method likely unsupported.** `Controller.update()` calls
   `update_motor_control()` → `set_actuator_control_target_send`. The spec's
   **Supported Messages** table lists only `SET_POSITION_TARGET_LOCAL_NED` and
   `SET_ATTITUDE_TARGET` as Client→Sim control inputs — **not** actuator control.
   → Pilot via **attitude-rate + thrust** (`update_attitude_flight_control`) or
   **velocity/position** (`update_position_flight_control`); treat actuator control
   as out of scope unless verified working against the sim.

5. **Motor constants are placeholder/wrong.** `controller.py:13-16` define
   `MOTOR_FRONT_LEFT=0, MOTOR_FRONT_RIGHT=1, MOTOR_BACK_LEFT=0, MOTOR_BACK_RIGHT=0`
   (note duplicated `0` value name pattern). Irrelevant once we drop actuator
   control (#4), but do not ship as-is.

6. **All RX handlers discard data.** Every `on_*` in `mavlink_rx.py` unpacks into
   local variables and returns. Likewise `on_track_data()` parses gate geometry and
   throws it away. Nothing reaches `shared_data`. This must be wired up before
   planning/control can use telemetry.

7. **Vision is a no-op.** `vision_rx.py:process_frame()` is `pass`. Decoded frames
   are dropped.

---

## 4. Target architecture & work breakdown

Build in this dependency order. Each block is a self-contained task suitable for one agent.

### Task A — Telemetry state store (`mavlink_rx.py` + `shared_data`)
**Goal:** make every inbound message land in `shared_data` as the latest known state,
thread-safely.
- Define a clear schema in `shared_data`, e.g. `shared_data['attitude']`,
  `['position_ned']`, `['velocity_ned']`, `['imu']`, `['armed']`, `['race']`,
  `['gates']`, `['last_collision']`, plus monotonic timestamps for each.
- Populate from `on_attitude`, `on_local_position_ned`, `on_odometry`,
  `on_highres_imu`, `on_heartbeat`, `on_race_status`, `on_collision`.
- **Store parsed gates** from `on_track_data()` (gate_id, NED position, NED
  orientation quaternion `[w,x,y,z]`, width, height) into `shared_data['gates']`.
- Guard shared state with a `threading.Lock` (3 writer threads + 1 reader).
- Prefer ODOMETRY for fused pose/velocity; ATTITUDE + LOCAL_POSITION_NED as fallback.

### Task B — Vision perception (`vision_rx.py`)
**Goal:** turn decoded FPV frames into gate detections in the camera frame.
- Implement `process_frame()`: detect the next gate(s) in the 640×360 BGR image.
- Use the camera model below to back-project detections to bearings/positions.
- Publish detections to `shared_data['vision']` (e.g. gate center pixel, estimated
  range, confidence, frame timestamp).
- Account for the **20° upward camera tilt** and the NED→camera convention
  difference when relating pixels to body/world directions.

### Task C — Planning (new module, e.g. `planner.py`)
**Goal:** decide the current target — which gate, and a waypoint/heading to reach it.
- Consume `shared_data['gates']` (known track geometry) + current pose + the race
  status `active_gate_index` to pick the active gate.
- Fuse vision detections (Task B) for closed-loop correction when a gate is in view.
- Output a target setpoint (NED position or velocity vector + desired heading) to
  `shared_data['target']`.
- Handle gate-passed transitions (advance `active_gate_index`) and the finish gate.

### Task D — Control (`controller.py`)
**Goal:** drive the drone toward `shared_data['target']` within the command budget.
- In `Controller.update()`, read the target and emit either
  `SET_POSITION_TARGET_LOCAL_NED` (velocity/position) or `SET_ATTITUDE_TARGET`
  (rate + thrust). Start with **velocity control** — simplest stable option.
- Implement a guidance law (e.g. pure-pursuit / proportional nav toward the gate
  center) plus altitude and yaw hold.
- Respect `CONTROL_HZ < 100`. Keep the loop's `time.sleep` accurate.
- Keep `arm()` and `send_sim_reset_command()` (useful for iteration/testing).

### Task E — Integration, safety & telemetry logging
- Clean shutdown: fix join sequence, handle Ctrl-C, set `is_running=False` on all threads.
- Add a watchdog: if no telemetry for N ms, hold/hover rather than fly blind.
- Optional debug HUD with `matplotlib`/`cv2` (the env already has both) — **but**
  ensure debug tooling is disabled for compliant timed runs (no human interaction).
- Log telemetry + decisions for offline tuning (deterministic sim → reproducible runs).

---

## 5. Technical reference (pulled from spec — keep handy)

### Connection / transport
- MAVLink 2 over UDP. Example client listens on `udpin:127.0.0.1:14550`
  (`main.py` `SIM_SERVER_UDP_IP/PORT`). Vision on UDP **port 5600**.
- Supported messages:
  - **Sim → Client:** `HEARTBEAT`, `ATTITUDE`, `HIGHRES_IMU`, `TIMESYNC`
    (plus the example also handles `ODOMETRY`, `LOCAL_POSITION_NED`,
    `ACTUATOR_OUTPUT_STATUS`, `COLLISION`, and `ENCAPSULATED_DATA` /
    `DATA_TRANSMISSION_HANDSHAKE` for race status + track data).
  - **Client → Sim:** `SET_POSITION_TARGET_LOCAL_NED`, `SET_ATTITUDE_TARGET`.

### Coordinate frames (NED)
- `MAV_FRAME_LOCAL_NED`: origin (0,0,0) fixed on the ground at the arm point.
- `MAV_FRAME_BODY_NED`: origin at vehicle. **X forward, Y right, Z down.**
- **Altitude is negative** in NED (down is +Z).
- Body→IMU = identity. Body→Camera = same origin, **camera tilted 20° upward**.
- All coords are NED — you may need to rotate the camera frame into your image
  library's camera convention.

### Camera intrinsics (pinhole, no distortion)
- Resolution **640 × 360**, 30 Hz.
- Principal point `[cx, cy] = [320, 180]`.
- Focal `[fx, fy] = [320, 320]`. VFoV = 90°.

### Vision packet (UDP 5600, little-endian, 24-byte header `<IHHIIQ`)
| Field | Type | Bytes | Meaning |
|-------|------|-------|---------|
| `frame_id` | uint32 | 4 | frame sequence ID |
| `chunk_id` | uint16 | 2 | packet index within frame (0..total-1) |
| `total_chunks` | uint16 | 2 | packets needed to complete frame |
| `jpeg_size` | uint32 | 4 | full reconstructed JPEG size |
| `payload_size` | uint32 | 4 | JPEG slice size in this packet |
| `sim_time_ns` | uint64 | 8 | sim epoch timestamp (ns) |

Reassembly already implemented in `vision_rx.py`; frames decode via `cv2.imdecode`.

### Encapsulated telemetry (over `ENCAPSULATED_DATA`)
- First payload byte = message type: `1` = race status, `2` = track info chunk.
- **Race status** struct: `"<BQqqIq"` =
  `data_type, sim_boot_time_ms, race_start_boot_time_ms, race_finish_time_ns,
  active_gate_index, last_gate_race_time`. Negative start/finish = not started / ongoing.
- **Track info** arrives chunked via `DATA_TRANSMISSION_HANDSHAKE` (`width` =
  transfer_id, `packets` = chunk count) + `ENCAPSULATED_DATA` chunks (`<BH` header:
  type, transfer_id; then `seqnr`-ordered body). Reassembled payload =
  `<H` num_gates, then per gate `<Hfffffffff` =
  `gate_id, pos_ned(x,y,z), orient_ned(w,x,y,z), width, height`. (38 bytes/gate.)

### Physical dimensions
- Drone chassis: 280 × 280 × 160 mm.
- Gate outer boundary: 2700 × 2700 mm, depth 260 mm.
- Gate **inner (flyable) square: 1500 × 1500 mm**, depth 260 mm.

### Collision IDs (from `COLLISION` msg)
- `1001` = gate, `1002` = environment. `threat_level` 1–2 (2 = harder hit).
  `horizontal_minimum_delta` is actually impulse magnitude (kg·m/s), not a delta.

---

## 6. Conventions for agents

- **Use the bundled interpreter:** `./myenv/Scripts/python.exe` (Windows paths).
- This repo is **not** a git repository — no branch/PR workflow; coordinate via this file.
- Don't break `main.py`'s component contract: `setup_components()` must keep
  returning the `controller / ts_loop / mavlink_rx / vision_rx / sim_conn` keys.
- Treat `shared_data` as the single source of cross-thread truth; never reach into
  another thread's internals directly.
- Keep all human-in-the-loop / debug input **out of the timed-run path** (compliance).
- Match the existing code style (module-level constants, `on_*` handler methods,
  `create_*` thread factories).

## 7. Suggested first milestone

A minimal end-to-end loop that flies the known track open-loop, then closes the loop:
1. Fix bugs #1–#3 (timesync, exit, command rate). Verify connect + arm + heartbeat.
2. Task A: persist telemetry + gates to `shared_data`. Confirm gate list received.
3. Task D (velocity control) + Task C (simple "fly to next gate center" planner)
   using **known track geometry only** — should complete the course open-loop.
4. Task B: add vision-based gate detection and fuse it for closed-loop correction.
5. Task E: hardening, logging, tuning against the deterministic course.
