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
- This repo **is** a git repository (`Code-Red-Cables/AI_GP`). `main` holds the pristine
  organizer starter; do team work on feature branches and keep diffs against the baseline
  clean. `venv/`, `myenv/`, and `__pycache__/` are git-ignored — never commit them.
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

---

## 8. Vision-first vertical slice — detailed step-by-step plan (current focus)

> **Goal of this slice:** a classic-CV perception pipeline that finds the next gate in the
> FPV frame, recovers the gate's position relative to the drone (pinhole reverse-projection),
> and drives the drone through it. Built as **small, independently testable modules** so each
> stage can be debugged in isolation (offline on a saved frame) before flying.
>
> **Strategy:** start with the simplest geometry that works (centroid bearing + size-based
> range), then upgrade the *same interface* to 4-corner `solvePnP` for full 3D pose. "For
> sake of speed," classic CV only — no learned models in this slice.

### 8.0 New modules (keep each one small and pure where possible)

| Module | Responsibility | Pure / testable offline? |
|--------|----------------|--------------------------|
| `camera_model.py` | Camera intrinsics + all frame transforms (pixel↔ray, camera↔body↔NED, 20° tilt, range-from-size). **numpy only, no cv2.** | **Yes** — unit-test with asserts |
| `vision/gate_detector.py` | `detect_gate(bgr) -> GateDetection \| None`. HSV threshold → contour → square. Returns center px, corners, bbox, area, confidence. | **Yes** — run on a saved PNG |
| `vision/gate_estimator.py` | Combine `GateDetection` + `camera_model` + current attitude → gate position in **body** and **NED**, range, bearing. | **Yes** — feed a fake detection |
| `tools/capture_frames.py` | Dump live FPV frames to disk for tuning (debug-only, never in timed run). | n/a |
| `tools/hsv_tuner.py` | Trackbar UI to calibrate HSV thresholds against captured frames. | n/a |
| `planner.py` | Pick the active gate + emit a target setpoint to `shared_data['target']`. | partly |
| `controller.py` (modify) | Velocity guidance toward `shared_data['target']`; `<100 Hz`; dry-run flag. | n/a |

`vision/` is a package (add `__init__.py`). Do **not** put cv2/HSV logic in `camera_model.py`
— geometry must stay dependency-light so its unit tests run without a frame or the sim.

### 8.1 The camera geometry (get this exactly right — everything depends on it)

Constants from spec §3.7/§3.8 — put in `camera_model.py`:
```
WIDTH, HEIGHT = 640, 360
CX, CY        = 320.0, 180.0
FX, FY        = 320.0, 320.0          # trust these; the doc's "VFoV=90°" is mislabeled
K             = [[FX,0,CX],[0,FY,CY],[0,0,1]]
CAMERA_TILT_UP_DEG = 20.0             # camera pitched UP 20° relative to body
GATE_INNER_M  = 1.5                   # flyable inner square side (1500 mm)
```

**Frames.** World = NED (X=North, Y=East, Z=Down), origin at arm point. Body = X-fwd,
Y-right, Z-down. OpenCV camera-optical = x-right, y-down, z-forward (optical axis).

**Body → camera-optical** is a fixed rotation `R_cb` = (axis permutation) ∘ (20° pitch-up
about body-Y). Build it once; verify with these **sign-check asserts** (this is where bugs
hide, so test empirically):
- A point straight ahead and level (body `[X>0, 0, 0]`) projects to the **lower** half
  (`v > CY`) — because the camera looks up, the horizon sits below the optical axis.
- A point straight ahead but **20° above horizontal** projects to vertical center (`v ≈ CY`).
- A point straight ahead projects to horizontal center (`u ≈ CX`); a point to the body-right
  projects to `u > CX`.

**Functions to implement (all small, all unit-tested):**
- `pixel_to_ray(u, v) -> unit vector in camera-optical frame` = normalize(`[(u-CX)/FX, (v-CY)/FY, 1]`).
- `project(point_cam) -> (u, v)` (inverse of above; for tests / overlay).
- `range_from_size(pixel_size, real_size=GATE_INNER_M, f=FX) -> Z`: `Z = f*real_size/pixel_size`
  (fronto-parallel approximation; bias shrinks as we line up head-on). Use width with FX,
  height with FY; average when both available.
- `deproject(u, v, Z) -> point in camera-optical frame` = `Z * [(u-CX)/FX, (v-CY)/FY, 1]`.
- `cam_to_body(p) / body_to_cam(p)` using `R_cb`.
- `body_to_ned(p, roll, pitch, yaw) / ned_to_body(...)` using `R_wb` from ATTITUDE
  (or quaternion from ODOMETRY — preferred when available).

### 8.2 Gate detection — `vision/gate_detector.py` (HSV → contour → square)

Round One is a *high-contrast, desaturated* environment, so the gate is the salient
saturated/bright object. Detector pipeline:
1. `cv2.cvtColor(bgr, COLOR_BGR2HSV)`.
2. `cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)` — **thresholds are TBD and must be calibrated**
   from real frames (see 8.5). Keep them in a module-level constant / small config so tuning
   is a one-line change. Support a wrapped hue range (two `inRange` ORed) in case the gate hue
   straddles 0/180.
3. Morphology: `MORPH_OPEN` then `MORPH_CLOSE` (small kernel) to despeckle and close gaps.
4. `findContours` (`RETR_CCOMP` so we get the gate's **inner hole**); keep contours above an
   area floor; score by squareness (bbox aspect ≈ 1) and extent.
5. **MVP output:** the best contour's centroid + bounding box + pixel area.
   **Upgrade path:** `approxPolyDP` the inner-hole contour to a 4-point quad → ordered corners
   (TL,TR,BR,BL) for PnP.
6. Return a small dataclass/dict `GateDetection{ center_px:(u,v), corners_px:[4]|None,
   bbox_px, area_px, confidence }`, or `None` if nothing passes the filters.

**Debuggability:** when a `DEBUG` flag is on, draw the mask + contour + center on a copy and
`cv2.imwrite` it (rate-limited). The function itself stays pure (input frame → detection); the
saving is a thin wrapper so offline tests don't write files.

### 8.3 Gate geometry estimate — `vision/gate_estimator.py`

Turn a `GateDetection` into a position estimate:
- **MVP (centroid + size):** `pixel_size` = mean of bbox width/height → `Z = range_from_size(...)`
  → `gate_cam = deproject(u, v, Z)` → `cam_to_body` → `body_to_ned(...)` using latest attitude.
  Also emit bearing `(azimuth, elevation)` from `pixel_to_ray` (useful for a pure yaw/pitch
  guidance law that needs no range).
- **Upgrade (PnP, the real "reverse solver"):** object points = 1.5 m square centered at
  origin in the gate plane; `cv2.solvePnP(obj, corners_px, K, None)` → `rvec, tvec`. `tvec` is
  the gate-center position in camera-optical frame (metric, no size assumption); `rvec` gives
  the gate **normal** so the planner can approach perpendicular and aim at the true center.
- Publish to `shared_data['vision']` (lock-guarded, see 8.6) with a frame timestamp and
  `confidence`. Stale detections (timestamp too old) must be ignored downstream.

### 8.4 Planner + control — fly through the gate

- **Planner (`planner.py`):** if a fresh vision detection exists, target = its gate position;
  else fall back to **known track geometry** (`shared_data['gates']` indexed by the race
  status `active_gate_index` — see Task A) so we never fly blind. Emit
  `shared_data['target'] = { mode, vel_body|pos_ned, yaw, ts }`. Use `active_gate_index`
  increments as the ground-truth **gate-passed** signal to advance to the next gate.
- **Control (`controller.py`, modify):** start with **velocity control** via
  `SET_POSITION_TARGET_LOCAL_NED` (PLAN §4-D): a proportional guidance law — yaw to center the
  gate (drive bearing azimuth → 0), hold/adjust altitude (elevation), and command forward
  speed scaled down as range shrinks. Respect `CONTROL_HZ < 100` (set 50–90). Keep a
  **`DRY_RUN` flag**: when set, compute and log/print commands but **don't send** flight
  setpoints — lets us validate perception + planning safely before the drone moves.

### 8.5 Calibration & offline test workflow (do this before trusting live flight)

1. `tools/capture_frames.py`: run sim, fly/hover manually *outside a timed run*, dump ~50
   frames containing gates to `notes/frames/`.
2. `tools/hsv_tuner.py`: load a frame, expose HSV trackbars, read off `LOWER/UPPER_HSV`, paste
   into `gate_detector.py`.
3. Offline harness: load a saved frame → `detect_gate` → `gate_estimator` → print
   center/range/body/NED estimate and write an annotated overlay. Iterate until the center and
   range look sane on several frames. **No sim or flying needed for this loop.**
4. `camera_model` unit tests (the 8.1 sign-checks) run standalone and must pass first.

### 8.6 `shared_data` additions for this slice (extends Task A schema)

```python
shared_data['vision'] = {
    'ts': sim_time_ns, 'detected': bool, 'confidence': float,
    'center_px': (u, v), 'corners_px': [...]|None, 'area_px': float,
    'range_m': float, 'bearing': (az_rad, el_rad),
    'gate_body': (x, y, z), 'gate_ned': (n, e, d)|None,
}
shared_data['target'] = { 'mode': 'velocity'|'position',
    'vel_body'|'pos_ned': (..,..,..), 'yaw': float, 'ts': sim_time_ns }
```
All cross-thread access goes through the Task-A `threading.Lock`. Never mutate another
thread's internals — `shared_data` is the only contract.

### 8.7 Suggested one-week sequencing (each day ends on something runnable)

- **Day 1 — unblock + telemetry.** Fix bugs #1–#3 (timesync, exit-path, command rate). Do
  Task A: persist attitude/position/odometry/race-status/gates to `shared_data` under a lock.
  Verify connect + arm + heartbeat, and that the gate list is received. Run
  `capture_frames.py` to collect tuning frames.
- **Day 2 — geometry foundation.** Implement `camera_model.py` + its unit tests (8.1
  sign-checks must pass). Calibrate HSV with `hsv_tuner.py`.
- **Day 3 — perception.** `gate_detector.py` + `gate_estimator.py`; validate on saved frames
  offline; wire `vision_rx.process_frame()` → `shared_data['vision']`; do a live
  **perception-only** run (DRY_RUN) with debug overlay — confirm stable detections.
- **Day 4 — fly through one gate.** `planner.py` + velocity guidance in `controller.py`;
  DRY_RUN first, then enable; tune gains until the drone passes the first gate.
- **Day 5 — close the loop, multi-gate.** Advance on `active_gate_index`; known-geometry
  fallback when no detection; upgrade estimator to `solvePnP` (corners → pose + normal) for
  accuracy and a perpendicular approach.
- **Days 6–7 — harden & tune (Task E).** Telemetry/decision logging, no-telemetry watchdog
  (hover, don't fly blind), full deterministic-course run to the finish gate, gain tuning.

### 8.8 Guardrails for any agent picking this up

- **Verify the camera-tilt sign empirically** (8.1) — a flipped sign sends the drone the wrong
  way and is the most likely silent bug in this slice.
- Trust the intrinsics matrix, **not** the doc's "VFoV=90°" label (it's inconsistent).
- Keep `DRY_RUN` on until perception is validated; keep all debug/tuning UIs **out of the
  timed-run path** (human interaction = DQ, spec §7).
- Preserve `setup_components()`'s returned keys and the `create_*`/`on_*` style (PLAN §6).
- The sim is deterministic — log enough to reproduce and diff runs while tuning.
