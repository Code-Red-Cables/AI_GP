# Implementation reference — autonomous pilot pipeline

What has been built, how it fits together, and the math behind it. This is the
"what exists and why" companion to `PLAN.md` (the forward-looking plan).
For running against the sim see [`../notes/VERIFY.md`](../notes/VERIFY.md);
for tuning see [`CALIBRATION.md`](CALIBRATION.md); for tests see [`TESTING.md`](TESTING.md).

Status: **Days 1–7 of PLAN.md §8 implemented and verified offline.** The only
work that requires the live sim (and a human) is HSV calibration and on-sim flight
tuning — see Calibration. `main.py` ships with `DRY_RUN=True` (no flight commands
sent) until perception is validated.

---

## 1. Pipeline at a glance

```
            UDP 5600                    UDP 14550 (MAVLink2)
               │                              │
        ┌──────▼───────┐              ┌───────▼────────┐
        │  VisionRX    │              │   MAVLinkRX    │   TimeSync (10 Hz)
        │ (thread)     │              │  (thread)      │   Logger (10 Hz)
        └──────┬───────┘              └───────┬────────┘
   detect_gate │                              │ on_* handlers store telemetry
 estimate_gate │                              │
        ┌──────▼──────────────────────────────▼───────┐
        │           shared_data  (RLock-guarded)       │   ← cross-thread blackboard
        │  vision · attitude · odometry · gates · race │
        └──────┬───────────────────────────────────────┘
       reads   │
        ┌──────▼───────┐  compute_target()   ┌────────────────┐
        │   Planner    │ ───────────────────▶│  shared_data    │
        │              │   vel_ned + yaw      │   ['target']    │
        └──────────────┘                      └───────┬────────┘
                                                reads  │
                                            ┌──────────▼─────────┐
                                            │    Controller      │ main loop, 60 Hz
                                            │ send_velocity_ned()│ ─▶ SET_POSITION_TARGET_LOCAL_NED
                                            └────────────────────┘     (gated by DRY_RUN)
```

Data flows one way through `shared_data`; no component calls another's internals.

---

## 2. Module reference

### `camera_model.py` — geometry foundation (numpy-only, no cv2)
The single source of truth for all frame math. Constants from spec §3.7/§3.8:
`WIDTH,HEIGHT=640,360`, `CX,CY=320,180`, `FX,FY=320,320`, `CAMERA_TILT_UP_DEG=20`,
`GATE_INNER_M=1.5`, plus `K` and the body→camera rotation `R_CB`.

Functions: `pixel_to_ray`, `project`, `deproject`, `range_from_size`,
`cam_to_body`/`body_to_cam`, `rot_world_body`, `body_to_ned`/`ned_to_body`.
See §3 for the math. Has its own unit tests (`test_camera_model.py`).

### `vision/gate_detector.py` — HSV → square (pure, frame in → detection out)
`detect_gate(bgr, cfg=None) -> GateDetection | None`. Pipeline: BGR→HSV →
`inRange` (+ optional wrapped-hue range) → open/close morphology →
`findContours(RETR_CCOMP)` → filter by area/squareness/extent → pick best →
inner-hole centroid + `approxPolyDP` corners (TL,TR,BR,BL) + confidence.
`GateDetection` = `{center_px, corners_px|None, bbox_px, area_px, confidence}`.
Thresholds (`LOWER_HSV`/`UPPER_HSV`, `DEFAULT_CFG`) are **placeholders pending
calibration**. `draw_detection()` annotates a copy; `__main__` self-tests on a
synthetic frame or an image path argument.

### `gate_estimator.py` — pixels → 3D pose (uses camera_model, cv2 for PnP)
`estimate_gate(det, attitude=None, position_ned=None, use_pnp=True, ts=None) -> dict`.
Two methods, same output schema:
- **size** (always available): apparent opening side in px → `range_from_size`
  → `deproject(center, Z)` → `cam_to_body`. Fronto-parallel assumption.
- **PnP** (when 4 corners exist): `cv2.solvePnP` on the known 1.5 m square →
  metric `gate_cam` + the gate's surface **normal**. Independent of apparent size.
Returns `{ts, detected, confidence, center_px, corners_px, area_px, range_m,
bearing:(az,el), gate_body, gate_ned, normal_body, method}`. `gate_ned` is absolute
only when both attitude and `position_ned` are supplied (relative-in-world axes if
only attitude; `None` if no attitude).

### `mavlink_rx.py` — inbound telemetry → `shared_data` (Task A)
Each `on_*` handler now **stores** its parsed message under `shared_data['lock']`
via the `_store()` helper, with a `time.time_ns()` `ts`. Populates `attitude`,
`position_ned`, `odometry`, `imu`, `armed`/`heartbeat_ts`, `race`, `gates`,
`last_collision`. Chunked track geometry is reassembled then stored as a
gate-id-sorted list. ODOMETRY is the rich pose source; ATTITUDE/LOCAL_POSITION_NED
are fallbacks.

### `vision_rx.py` — inbound video → perception
Reassembles chunked JPEG (unchanged) then `process_frame()` runs `detect_gate` →
`estimate_gate` (using the latest attitude/position snapshot) → publishes
`shared_data['vision']`. `_quat_to_rpy()` converts the ODOMETRY quaternion to
roll/pitch/yaw when ATTITUDE isn't present. Writes `_vision_NN.png` overlays when
`shared_data['debug_vision']` is set.

### `planner.py` — choose the target gate (Task C)
`Planner.compute_target()` snapshots `shared_data`, then:
1. **Vision** if fresh (`< VISION_TIMEOUT_NS`) and `confidence ≥ CONF_MIN`: rotate
   `gate_body` into NED with current attitude.
2. else **known geometry**: `gates[active_gate_index].pos_ned − drone_pos`. The sim
   increments `active_gate_index`, so it doubles as the gate-passed signal.
3. else / stale telemetry → **hover** (watchdog).
Produces a NED velocity (`speed = min(MAX_SPEED, KP_POS·dist)`, full speed within
`PASS_THROUGH_DIST` to commit through the gate) and a yaw that points the nose
(camera) at the gate. Writes `shared_data['target']`. Tunables at the top of the file.

### `controller.py` — outbound control (Task D)
`Controller.update()` (called every main-loop tick): `planner.compute_target()` →
`send_velocity_ned()` via `SET_POSITION_TARGET_LOCAL_NED` with a type-mask that
commands velocity + yaw and ignores position/accel/yaw-rate. `CONTROL_HZ=60`
(spec < 100 Hz). Gated by `shared_data['dry_run']`; throttled status log at ~2 Hz.
Keeps `arm()` and `send_sim_reset_command()`.

### `logger.py` — run logging (Task E)
`Logger.create_logger()` spawns a thread that snapshots `shared_data` to
`logs/run_<ts>.jsonl` at ~10 Hz. Deterministic sim ⇒ diffable/replayable runs for
gain tuning. Follows the `create_*`/`get_thread_for_join()` convention.

### `main.py` / `setup.py` — wiring
`main.py` creates `shared_data` with the `RLock` and the `DRY_RUN`/`DEBUG_VISION`/
`LOGGING` flags, runs the control loop under `try/except KeyboardInterrupt`, and
joins all threads on exit (None-guarded — fixes the original crash). `setup.py`
builds the MAVLink connection and every component (now incl. `Planner` and
`Logger`) and returns them.

### `tools/` — calibration utilities (never in a timed run)
`capture_frames.py` saves reassembled FPV frames; `hsv_tuner.py` is a trackbar UI
to pick HSV thresholds. See [`CALIBRATION.md`](CALIBRATION.md).

---

## 3. The geometry (why the signs are what they are)

**Frames.** Body (NED): x-fwd, y-right, z-down. Camera-optical (OpenCV): x-right,
y-down, z-forward. World: NED, origin at arm point.

**Body → camera.** The optical axis is body-forward pitched **up 20°**. With
`t = 20°`:
```
R_CB = [[0,      1,     0   ],
        [sin t,  0,     cos t],
        [cos t,  0,    -sin t]]      v_cam = R_CB · v_body ,  cam→body = R_CBᵀ
```
Consequence to remember: because the camera looks up, a gate **straight ahead and
level** appears in the **lower** half of the image, and the image center
corresponds to a direction **20° above** body-forward. (This is exactly what the
smoke test sees: a gate at pixel (320,180) → `gate_body ≈ (range·cos20, 0,
−range·sin20)`, i.e. elevation +20°.)

**Pixel → ray.** `pixel_to_ray(u,v) = normalize([(u−cx)/fx, (v−cy)/fy, 1])`.

**Range from size.** Fronto-parallel pinhole: `Z = f · real_size / pixel_size`
(`f=320`, `real_size=1.5 m`). A 1.5 m gate that is 96 px across is 5 m away.

**Body → world.** `R_wb = Rz(yaw)·Ry(pitch)·Rx(roll)` (standard NED aerospace ZYX).

**PnP.** Object points are the 1.5 m square in the canonical y-up order
`SOLVEPNP_IPPE_SQUARE` expects — TL,TR,BR,BL = `(−s,+s),(+s,+s),(+s,−s),(−s,−s)`,
matching the detector's corner order. (Getting this order wrong silently produces a
garbage near-zero range — it was a real bug, now covered by the smoke test.)

---

## 4. `shared_data` schema (the contract)

Guarded by `shared_data['lock']` (an `RLock` created in `main.py`). Flags:
`dry_run`, `debug_vision`, `logging`. State keys (each value carries a `ts` in ns):

| key | shape |
|-----|-------|
| `attitude` | `{roll,pitch,yaw,rollspeed,pitchspeed,yawspeed,time_boot_ms,ts}` |
| `position_ned` | `{x,y,z,vx,vy,vz,time_boot_ms,ts}` |
| `odometry` | `{pos:(x,y,z), q:(w,x,y,z), vel:(vx,vy,vz), rates:(r,p,y), time_usec, reset_counter, ts}` |
| `imu` | `{acc:(x,y,z), gyro:(x,y,z), time_usec, ts}` |
| `armed` / `heartbeat_ts` | `bool` / `ns` |
| `race` | `{sim_boot_time_ms, race_start_boot_time_ms, race_finish_time_ns, active_gate_index, last_gate_race_time, ts}` |
| `gates` | `list[{gate_id, pos_ned:(x,y,z), orient_ned:(w,x,y,z), width, height}]` (sorted by id) |
| `last_collision` | `{collision_id, threat_level, impulse, ts}` |
| `vision` | `{ts, detected, confidence, center_px, corners_px, area_px, range_m, bearing:(az,el), gate_body, gate_ned, normal_body, method, frame_id, sim_time_ns}` |
| `target` | `{mode:'velocity', vel_ned:(n,e,d), yaw, range_m, source, ts}` |

`target['source']` ∈ `{vision, vision_level, known, hover, watchdog_hover}` — a quick
read on what the planner is doing.

---

## 5. Key design decisions

- **Trust the intrinsics matrix, not the doc's "VFoV=90°"** — that label is
  inconsistent; `fx=fy=320` give a ~90° *horizontal* / ~58.7° vertical FoV.
- **Velocity control, not actuator control** — the spec only supports
  `SET_POSITION_TARGET_LOCAL_NED` and `SET_ATTITUDE_TARGET`; the example's motor
  control was a bug. Velocity is the simplest stable option to start.
- **`active_gate_index` as the gate-passed signal** — the sim is authoritative, so
  we don't guess when a gate is cleared.
- **Vision-fresh-else-known-geometry** — we always have a target (known track
  layout) even before/without a detection; vision refines it when confident.
- **DRY_RUN default True + telemetry watchdog** — never fly blind or by accident.
- **Everything offline-testable** — geometry and the full perception→control chain
  run without the sim, so logic bugs are caught before flying.

---

## 6. What is NOT done yet (next steps)

- **HSV thresholds are placeholders** — must be calibrated against real frames
  (`CALIBRATION.md`). Until then `target['source']` will be `known`, not `vision`.
- **Guidance gains are conservative starting values** — tune `planner.py`
  (`MAX_SPEED`, `KP_POS`, `PASS_THROUGH_DIST`) on the deterministic course using the
  JSONL logs.
- **No obstacle avoidance / no SET_ATTITUDE_TARGET aggressive mode** — velocity
  guidance only. Higher-speed attitude-rate control is a later optimization.
- **PnP validated on synthetic gates only** — re-confirm on real frames once HSV is
  calibrated (corner detection quality drives PnP accuracy).
