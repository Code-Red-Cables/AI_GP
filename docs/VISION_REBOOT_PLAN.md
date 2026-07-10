# Vision Autonomy Reboot Plan

**Goal:** replace the accreted heuristic planner with a clean, pure-vision architecture:
gates are perceived by the camera, fused into a persistent world map, and flown through
by a spline path planner. No hand-captured waypoint missions, no broadcast track
geometry in the flight loop.

**Audience:** implementation agents. Every phase below has explicit file targets,
interfaces, and acceptance criteria. Read the whole document before starting any phase.
The sections "What we keep", "Hard-won facts", and "Interfaces" are binding contracts —
do not re-derive or contradict them.

---

## 1. Why a reboot

The repo evolved through five strategies (vision chaser → waypoint mission → teleop
capture → spline replay → VQ2 hover), and the current working tree on
`manual-control-vq2` is the worst of it:

- `planner.py` (506 lines, uncommitted) is the reactive vision gate-chaser with ~10
  layers of band-aid state: post-gate coast, yaw sweep, gate-1 look-down ramp,
  confidence relaxation windows, range-jump rejection, world-jump rejection, separate
  vertical-trust logic, yaw-freeze zones. Each patch fixed one logged failure and
  created coupling with the others. It is not tunable or reasoned about anymore.
- `planner_modified.py` (502 lines, untracked) is a near-duplicate of `planner.py`
  (35-line diff). Delete it.
- `drone_gui.py` (untracked) drives flight by synthesizing Win32 keyboard events into
  the teleop thread. It must never be near a timed run (human input = DQ). Move it to
  `tools/` or delete.
- The winning race performance ("sub 14 seconds", branch `spline-path`) comes from
  **teleop-captured waypoints replayed through a spline** — zero vision. Fast, but
  brittle (breaks on any new course) and philosophically not what we're building.

The root architectural defect of the old vision planner: **it steered off single
frames.** Every heuristic above exists to compensate for per-frame noise, FOV loss, or
a bad detection. The fix is structural, not more patches: put a **state estimator (gate
mapper)** between perception and planning so the planner only ever sees smooth,
persistent, world-frame gate estimates — and steer along a **path**, not at a point.

---

## 2. What we keep (verified assets — do not rewrite)

| Asset | Source branch/state | Why it's kept |
|---|---|---|
| `controller.py` | `spline-path` version (26 KB) | The only part proven live at speed. Body-rate+thrust control, split lean caps (pitch 60° / roll 52°), per-axis rate signs, LPF, mode-entry handshake. Calibrated 2026-06-15. **Do not touch in phases 0–3.** |
| `camera_model.py` + `test_camera_model.py` | any (identical) | Unit-tested intrinsics/frame math. All geometry goes through it. |
| `vision/gate_detector.py` | working tree | HSV calibrated 2026-06-03 (two-piece red/orange mask). Upgraded in Phase 1, not rewritten. |
| `gate_estimator.py` | working tree | Size + PnP pose estimation, correct schema. Moves to `vision/`, minor upgrades Phase 1. |
| `mavlink_rx.py`, `vision_rx.py`, `timesync.py`, `logger.py` | `spline-path` | Working I/O layer. |
| `config.py` single-source pattern | `spline-path` | All knobs in one file, no env vars. |
| Spline + pure-pursuit machinery | `spline_planner.py` on `spline-path` | Catmull-Rom fit, speed-scaled carrot, curvature-aware speed cap (`sqrt(A_LAT_MAX/κ)`), longitudinal brake-ahead, `KP_VERT_PATH` altitude pull. Extracted into `guidance/path.py` in Phase 3. |
| `teleop.py`, `mission.py` | `manual-control` | Kept as **dev tools only** (course scouting, controller calibration). Never the default mode. |
| Offline closed-loop test pattern | `test_waypoint_mission.py` / `test_spline_mission.py` | Kinematic drone model driven by the real planner+controller math. Reused for the vision race sim (Phase 4 test). |

**What we discard:** `planner.py` (both branches' versions), `planner_modified.py`,
`drone_gui.py` (or park in `tools/`), the `mission.json`-replay flight mode as a race
strategy, all broadcast-track-geometry fallbacks in the flight loop.

---

## 3. Hard-won facts (paid for in crashed runs — preserve verbatim)

Any agent writing flight code must honor these. Citations are run logs / docs already
in the repo.

**Simulator control model**
1. The DCL sim is a Betaflight-style FPV racer. It has **no velocity or position
   loop** — `SET_POSITION_TARGET_LOCAL_NED` velocity setpoints are silently ignored
   (log `run_1780516557`: 24 m/s uncontrolled climb while commanding descent). It also
   does **not hold a commanded attitude**. The only honored inputs are **body rates +
   collective thrust** via `SET_ATTITUDE_TARGET` with the attitude quaternion ignored.
2. **Mode-entry handshake is mandatory:** sim boots in ACRO (ignores commands when
   armed). Sequence: prime a setpoint stream (~50 Hz for ~1 s) → request ANGLE mode →
   keep the stream alive across the switch → arm. Implemented in
   `controller.hold/prime_setpoint_stream/request_offboard_mode` + `main.py`.
3. **The sim's reported yaw is left-handed vs NED**: physical forward =
   `(cos yaw, −sin yaw)`. `velocity_to_attitude` rotates the world velocity error by
   **−yaw** (history: `+yaw` flew East when commanded West, log `run_1781506925`).
   Mission/planner yaw convention: 0 = North, +90 = East.
4. Rate limits: outbound commands **< 100 Hz** (we run `CONTROL_HZ = 60`), heartbeat
   ≥ 2 Hz, physics 120 Hz, max run 8 min. Any human input during a timed run = DQ.

**Camera & geometry**
5. 640×360 @ 30 Hz over UDP 5600 (chunked JPEG, 24-byte `<IHHIIQ` header; only one
   process can bind 5600). Pinhole, no distortion: `fx = fy = 320`,
   `(cx, cy) = (320, 180)`. The spec's "VFoV = 90°" is mislabeled — trust the
   intrinsics (VFoV ≈ 58.7°, HFoV = 90°).
6. Camera is tilted **+20° UP** from body-forward. A gate at image center is 20° above
   the nose. Consequences: at long range a centered gate back-projects **8–14 m too
   high** (the old planner's "vertical trust window" hack); the next, lower gate needs
   descent or nose-down speed to enter the FOV at all (log 2026-06-06: 0 detections in
   4 s of level coast).
7. Gate inner opening 1.5×1.5 m, outer 2.7×2.7 m, depth 0.26 m. Range from apparent
   size: `Z = 320 · 1.5 / pixel_size` (96 px → 5.0 m). PnP on 4 corners is metric and
   strictly better when corners are available.
8. HSV (Round 1, calibrated): `(0,0,80)–(15,255,255)` OR `(170,0,80)–(180,255,255)`
   (red hue wraps at 0/180). Re-verify per `docs/CALIBRATION.md` §2 on any new course.

**Perception failure modes observed (the mapper must absorb these)**
9. Single-frame bearing is garbage during maneuvers: yaw_tgt swung −179°→−82° with the
   drone nearly stationary and the gate steady at 21 m (log 2026-06-06).
10. Garbage detection bursts happen: range jumped 13→38→64 m with an 80° bearing swing
    in 0.4 s and hijacked guidance for the rest of the run (log 2026-06-05).
11. At point-blank range the gate fills the frame and detected bearing whipsaws
    (gb_az −10→+43→−47→+37° in 0.5 s while passing, log 2026-06-06) — never steer from
    vision inside the final ~4 m; commit and fly through.
12. The narrow FOV loses the gate ~1 s into an approach. Without world-frame memory the
    drone parks 2–3 m short forever (logs 2026-06-05/06).

**Telemetry**
13. No GPS. Local NED, origin at arm point, Z negative-up. `ATTITUDE`,
    `LOCAL_POSITION_NED`/`ODOMETRY`, `HIGHRES_IMU` all stream ~30 Hz on the standard
    platform. (Qualifier-2 platform blocks attitude/position — out of scope here; see
    §10.)
14. Race status (`ENCAPSULATED_DATA` type 1, ~10 Hz) carries `active_gate_index`,
    which the sim increments as gates are passed — a free, reliable **gate-passed
    signal**. This is race telemetry, not track geometry; the pure-vision mandate
    allows it.
15. Ground-truth gate poses ARE broadcast (`DATA_TRANSMISSION_HANDSHAKE` +
    `ENCAPSULATED_DATA` type 2) and using them is legal per the spec — but they are
    **excluded from the flight loop by this plan's mandate**. They MAY be used in
    offline tests and log analysis as ground truth for grading the mapper (§8), which
    is exactly what they're good for.

---

## 4. Target architecture

```
                 UDP 5600                    UDP 14550
                    │                            │
              ┌─────▼─────┐               ┌──────▼──────┐
              │ VisionRX  │               │  MAVLinkRX  │
              │ (thread)  │               │  (thread)   │
              └─────┬─────┘               └──────┬──────┘
        frame → detect_gates() → estimate()      │ attitude, position_ned,
                    │                            │ race (active_gate_index)
                    │  list[GateObservation]     │
              ┌─────▼────────────────────────────▼─────┐
              │        shared_data (RLock blackboard)  │
              └─────┬────────────────────────────┬─────┘
                    │ observations               │ pose, race
              ┌─────▼─────────┐                  │
              │  GateMapper   │  fuse into       │
              │ (in planner   │  world frame     │
              │  tick, pure)  │                  │
              └─────┬─────────┘                  │
                    │ CourseMap: list[GateEstimate]
              ┌─────▼──────────────────────────────────┐
              │  RacePlanner (FSM + path)               │
              │  SCAN / APPROACH / PASS / ADVANCE ...   │
              │  builds spline through gate estimates,  │
              │  pure-pursuit carrot → velocity + yaw   │
              └─────┬──────────────────────────────────┘
                    │ target = {'mode':'velocity', 'vel_ned', 'yaw', ...}
              ┌─────▼─────────┐
              │  Controller   │  UNCHANGED: velocity → lean angles → body
              │ (main loop)   │  rates + thrust, SET_ATTITUDE_TARGET @60Hz
              └───────────────┘
```

Three principles, each fixing a documented failure class:

1. **Perception never steers directly.** Frames produce `GateObservation`s;
   only the `GateMapper`'s filtered `GateEstimate`s are flown to. This one change
   subsumes the old range-jump / world-jump / bearing-whipsaw / vertical-trust hacks
   into a single gating+filtering step (facts 9–12).
2. **Fly a path, not a point.** The planner maintains a spline through the estimated
   gate sequence (however much of it is known) and pure-pursues a carrot on it. Gate
   approach direction comes from the estimated gate normal, giving a through-the-gate
   segment for free. Reuses the proven `spline-path` machinery.
3. **Behavior is one explicit FSM,** not distributed booleans. Every mode the old
   planner encoded implicitly (takeoff, search sweep, approach, commit, post-pass
   reacquire) becomes a named state with entry/exit conditions in one file.

The `Planner.compute_target()` → `shared_data['target']` → `Controller` contract is
**unchanged**, so the controller and its calibration carry over untouched, and the
kinematic offline test harness keeps working.

---

## 5. Interfaces (binding contracts)

All new code is plain Python 3.14, numpy allowed, cv2 only inside `vision/`. Angles in
radians, distances in meters, NED world frame, timestamps in ns (`time.monotonic_ns()`
for wall, `sim_time_ns` from the vision header for frames) — matching existing code.

### 5.1 `vision/gate_detector.py` (Phase 1 upgrade)

```python
@dataclass
class GateDetection:
    center_px: tuple[float, float]        # (u, v)
    corners_px: list[tuple[float, float]] | None   # TL, TR, BR, BL or None
    bbox_px: tuple[float, float, float, float]     # x, y, w, h
    area_px: float
    confidence: float                     # 0..1, see confidence model below

def detect_gates(bgr: np.ndarray, cfg: dict | None = None) -> list[GateDetection]:
    """All plausible gates in the frame, best first. [] when none.
    Replaces single-detection detect_gate(); a thin detect_gate() wrapper
    returning the first element is kept for the old tests."""
```

Confidence model (replaces the current ad-hoc score): product of factors in [0,1] —
squareness of bbox, mask fill ratio in the ring region, corner count (1.0 with 4
corners, 0.6 without), area fraction sanity. Document each factor inline.

### 5.2 `vision/gate_estimator.py` (moved from repo root, Phase 1)

```python
@dataclass
class GateObservation:
    ts_ns: int                    # sim_time_ns of the source frame
    frame_id: int
    gate_body: np.ndarray         # (3,) gate center, body NED frame
    normal_body: np.ndarray       # (3,) unit normal, body frame (sign: toward drone)
    gate_ned: np.ndarray | None   # (3,) world NED (needs pose at frame time)
    normal_ned: np.ndarray | None
    range_m: float
    method: str                   # 'pnp' | 'size'
    confidence: float             # carried from detection
    center_px: tuple[float, float]

def estimate_gates(dets: list[GateDetection],
                   attitude: dict | None,
                   position_ned: dict | None) -> list[GateObservation]:
```

Prefer PnP whenever 4 corners exist (`cv2.SOLVEPNP_IPPE_SQUARE`, 1.5 m inner square).
The size method stays as fallback; its observation gets `confidence *= 0.7` and a
larger measurement noise in the mapper (see 5.3).

`VisionRX` publishes `shared_data['vision'] = {'ts': ..., 'observations':
[GateObservation, ...], 'frame_id': ...}`. Keep the old single-gate keys populated from
`observations[0]` during migration so `logger.py` and old tests don't break; remove in
Phase 5.

### 5.3 `mapping/gate_mapper.py` (new, Phase 2 — pure module, no threads, no cv2)

```python
@dataclass
class GateEstimate:
    gate_id: int                     # our ordinal, 0-based in pass order
    pos_ned: np.ndarray              # (3,) filtered center
    normal_ned: np.ndarray           # (3,) filtered unit normal (toward approach side)
    pos_var: np.ndarray              # (3,) per-axis variance, m^2
    n_obs: int
    last_seen_ns: int
    passed: bool

class GateMapper:
    def __init__(self, cfg: MapperConfig): ...
    def update(self, obs: list[GateObservation], now_ns: int,
               active_gate_index: int | None) -> None: ...
    def course(self) -> list[GateEstimate]:          # pass-order, stable ids
    def gate(self, gate_id: int) -> GateEstimate | None: ...
    def next_unpassed(self) -> GateEstimate | None: ...
    def to_json(self) -> dict / from_json(cls, d) -> GateMapper   # persistence
```

Filter spec (keep it this simple — no full EKF):
- **Per-gate 3D position filter**: scalar-gain Kalman per axis. Measurement variance
  `R = (σ0 + k·range_m)²` with `σ0 = 0.3 m`, `k = 0.05` for PnP; multiply by 4 for
  size-method observations; multiply the **D (vertical) axis** variance by
  `max(1, (range_m / 12)²)` — this encodes fact 6 (far elevation is untrustworthy)
  as noise, not as a special case.
- **Gating**: Mahalanobis distance of the observation vs. each existing estimate;
  associate to the best gate under `GATE_ASSOC_CHI2 = 9.0` (3σ), else spawn a new
  candidate. A candidate needs `MIN_OBS_CONFIRM = 3` associated observations within
  2 s to become a real gate (kills the garbage-burst failure, fact 10).
- **Normal filter**: running mean of unit normals, re-normalized; flip incoming
  normals to the hemisphere facing the drone's side at first observation.
- **Ordering / ids**: gates are ordered by pass sequence. While
  `active_gate_index` from race status is available, it labels the gate we're
  currently being scored against — use it to anchor ordering and to set `passed`.
  Fallback ordering when it's missing: order of first confirmation.
- **Persistence**: `course_map.json` (schema = `to_json`). The course is deterministic
  across runs, so a map learned in a scouting run is loaded at startup of a timed run
  and refined online. This is vision-derived data — within the pure-vision mandate.

Mapper state is published each tick to `shared_data['course_map']` (plain dicts, for
the logger and offline analysis).

### 5.4 `guidance/path.py` (Phase 3 — extraction, not invention)

Extract from `spline_planner.py` (branch `spline-path`) into pure functions/classes,
unchanged math:

```python
class SplinePath:
    @classmethod
    def through(cls, points_ned: list[np.ndarray]) -> "SplinePath"   # centripetal Catmull-Rom
    def project(self, p: np.ndarray) -> float                        # arc-length s of closest point
    def point(self, s: float) -> np.ndarray
    def tangent(self, s: float) -> np.ndarray
    def curvature(self, s: float) -> float
    def length(self) -> float

def carrot_velocity(path, pos_ned, vel_ned, cfg) -> tuple[np.ndarray, float]:
    """Pure-pursuit: speed-scaled lookahead carrot, curvature speed cap
    sqrt(A_LAT_MAX/κ) with brake-ahead at A_LON_MAX, KP_VERT_PATH vertical pull.
    Returns (vel_ned_cmd, yaw_cmd). Constants from config.py."""
```

`test_spline_mission.py` must pass against the extracted module before anything else
uses it.

### 5.5 `guidance/race_planner.py` (Phase 3 — the only stateful planner)

```python
class RacePlanner:
    """compute_target() contract identical to every previous planner:
    returns {'mode': 'velocity', 'vel_ned': (n,e,d), 'yaw': float,
             'source': str, 'state': str, 'ts': int} and writes it to
    shared_data['target']. 'source'/'state' are diagnostics for the logger."""
    def __init__(self, shared_data, mapper: GateMapper, cfg): ...
    def compute_target(self) -> dict: ...
```

FSM (all states in one file, one transition function, logged every change):

| State | Behavior | Exit |
|---|---|---|
| `TAKEOFF` | climb straight up to `TAKEOFF_ALT_M`, hold boot heading | at altitude → `SCAN` (or `APPROACH` if map already has an unpassed gate) |
| `SCAN` | slow yaw sweep ±`SCAN_YAW_MAX` around a base heading; altitude schedule descends slowly toward `SCAN_ALT_MIN` (fact 6: tilt-up camera must look down to find low gates); creep forward at `SCAN_SPEED` | mapper confirms an unpassed gate → `APPROACH`; timeout → widen sweep |
| `APPROACH` | build/refresh spline: current position → standoff point (`gate.pos − APPROACH_STANDOFF · normal`) → gate center → exit point (`gate.pos + EXIT_OVERSHOOT · normal`) → (next gate's standoff if mapped); pure-pursue it; yaw follows path tangent | distance to gate < `COMMIT_DIST` → `PASS` |
| `PASS` | **vision ignored** (fact 11): freeze yaw, fly the frozen spline segment through the plane at commit speed | plane crossed OR `active_gate_index` incremented → `ADVANCE` |
| `ADVANCE` | mark gate passed; if next gate mapped → `APPROACH`; else short coast on exit heading with gentle descent (facts 6, 12) → `SCAN` biased toward the exit normal direction | — |
| `FINISH` | after last gate: maintain speed across the line, then brake to hover | — |
| `HOLD` | safety hover | entered from any state by the guards below |

Safety guards (checked before the FSM, same as today): telemetry watchdog
(>500 ms stale pose → `HOLD`), altitude envelope (`MAX_ALT_M`), horizontal envelope
(distance to active path > `MAX_PATH_DIST_M` → `HOLD`). Guards are the ONLY thing
allowed to bypass the FSM.

**Racing mode:** when the loaded map covers the whole course
(`all gate estimates confirmed`, config `RACE_MODE = True`), skip SCAN entirely:
`TAKEOFF` → build one spline through every gate (standoff/center/exit triplets) →
fly it at `CRUISE_SPEED` with curvature braking, while the mapper keeps refining gate
positions from whatever vision sees and the path is re-fit when an estimate moves by
more than `REFIT_THRESHOLD_M`. This is the sub-14-second flight, but driven by a
vision-built map instead of hand-captured waypoints.

### 5.6 `config.py` additions

One block per module, constants only, no env vars (existing convention):

```python
# --- vision autonomy ---
USE_VISION = True          # the only supported flight mode on this branch
RACE_MODE = False          # False: scout & map. True: fly the loaded map fast.
COURSE_MAP_PATH = 'course_map.json'
# mapper
GATE_ASSOC_CHI2 = 9.0; MIN_OBS_CONFIRM = 3; PNP_SIGMA0_M = 0.3; PNP_SIGMA_K = 0.05
SIZE_METHOD_VAR_MULT = 4.0; FAR_VERT_RANGE_M = 12.0
# fsm
TAKEOFF_ALT_M = 3.0; SCAN_SPEED = 1.0; SCAN_YAW_MAX_DEG = 45.0; SCAN_ALT_MIN_M = 1.0
APPROACH_STANDOFF_M = 3.0; EXIT_OVERSHOOT_M = 2.5; COMMIT_DIST_M = 4.0
MAX_PATH_DIST_M = 60.0
# path (values proven on spline-path; re-tune only in Phase 4)
CRUISE_SPEED = ...; LOOKAHEAD_M = ...; LOOKAHEAD_TIME = ...; LOOKAHEAD_MAX = ...
A_LAT_MAX = ...; A_LON_MAX = ...; KP_VERT_PATH = ...; FINISH_SPEED = ...
```

(`...` = copy the current values from `spline-path`'s `config.py`.)

### 5.7 `shared_data` schema deltas

| Key | Producer | Change |
|---|---|---|
| `vision` | VisionRX | `observations: list[dict]` added; legacy single-gate keys kept until Phase 5 |
| `course_map` | RacePlanner tick | new: `{'gates': [GateEstimate dicts], 'ts': ns}` |
| `target` | RacePlanner | unchanged shape; `source` ∈ {`fsm:<STATE>`, `watchdog_hover`, `alt_guard`, `dist_guard`}, plus `state` |

Everything else (attitude, position_ned, odometry, race, gates, lock discipline)
unchanged. `shared_data['gates']` (broadcast ground truth) keeps being *recorded* by
`mavlink_rx.py` — for logs and test grading only; grep must show no read of it under
`guidance/` or `mapping/`.

---

## 6. Repo reset (Phase 0)

1. Create branch `vision-autonomy` **from `spline-path`** (best controller + config +
   spline code). Do not branch from the dirty working tree.
2. Port from the current working tree / `modified-starter`: `vision/gate_detector.py`
   (calibrated HSV), `gate_estimator.py`, `test_camera_model.py`,
   `test_pipeline_smoke.py`.
3. Delete on the new branch: `planner.py`'s vision-chaser content (the file will be
   replaced by `guidance/race_planner.py` + a thin `planner.py` shim if `setup.py`
   needs the name), `planner_modified.py`, `drone_gui.py` (or `git mv` to `tools/`),
   `mission.json` as a flight input. Keep `teleop.py` + `mission.py` importable for
   dev tooling.
4. New package dirs: `vision/` (exists), `mapping/`, `guidance/`, each with
   `__init__.py`.
5. `setup.py`: select `RacePlanner` when `USE_VISION`, `TeleopPlanner` when
   `USE_TELEOP` (dev only). Delete other selection paths.
6. Set `DRY_RUN = True` as the committed default. Flying is an explicit local edit.
7. On `manual-control-vq2`: commit the current mess to a `wip-archive/manual-vq2`
   branch for forensics, then abandon it. Nothing merges from it except the two vision
   files in step 2.

**Acceptance:** `python test_camera_model.py` and `python test_spline_mission.py` pass
on the new branch with the bundled interpreter
(`C:\Users\rocky\docs\AI_GP\PyAIPilotExample\myenv\Scripts\python.exe`); `git grep`
shows no imports of deleted modules; `main.py` starts in DRY_RUN against no sim and
exits cleanly on Ctrl-C.

---

## 7. Implementation phases

Each phase is a self-contained agent task with offline acceptance tests. Do not start
a phase before the previous one's acceptance passes. No live sim needed before
Phase 4 (DRY_RUN) / Phase 5 (flight).

### Phase 1 — Perception hardening (`vision/`)
- `detect_gates()` multi-detection + factored confidence (5.1); keep `detect_gate()`
  wrapper.
- Move estimator to `vision/gate_estimator.py`; emit `GateObservation` lists (5.2);
  PnP-first; attach `normal_ned`.
- Build `tools/eval_detector.py`: run the detector over every frame in `frames/` and
  `_vision_debug/`, write an HTML/PNG contact sheet of detections + a stats line
  (frames, detection rate, 4-corner rate, mean confidence).
- **Acceptance:** `test_pipeline_smoke.py` (updated for the list API) passes;
  eval tool reports ≥ 95 % detection and ≥ 80 % 4-corner rate on the existing captured
  frames that contain a gate, zero detections on gateless frames.

### Phase 2 — Gate mapper (`mapping/gate_mapper.py`)
- Implement 5.3 exactly. Pure module: inputs are observation lists + timestamps;
  fully deterministic; no `shared_data` access inside the class.
- `test_gate_mapper.py` (new, synthetic): (a) noisy observations of 3 static gates
  (σ matching the R model, plus 5 % garbage-burst outliers per fact 10) → all 3
  confirmed, position RMSE < 0.5 m, zero phantom gates; (b) far-range high-elevation
  bias injected on D axis (fact 6) → converged height within 0.4 m of truth;
  (c) `to_json`/`from_json` round-trip; (d) association stays stable while the
  observer moves (bearing sweep).
- **Acceptance:** `python test_gate_mapper.py` → `ALL ... PASSED`.

### Phase 3 — Guidance (`guidance/`)
- Extract `guidance/path.py` from `spline_planner.py`; `test_spline_mission.py` green
  against it before proceeding.
- Implement `RacePlanner` FSM (5.5) + config block (5.6).
- `test_race_fsm.py` (new): drive the FSM with scripted mapper/pose states, assert
  the transition table (incl. guards preempting every state, PASS ignoring vision,
  ADVANCE→SCAN when next gate unknown).
- **Acceptance:** FSM test + spline test green; `compute_target()` output validates
  against the target schema for every state.

### Phase 4 — Closed-loop vision race sim (the keystone test)
- `test_vision_race.py`: kinematic drone model (reuse the one in
  `test_spline_mission.py`, incl. left-handed yaw) + **synthetic camera**: project the
  true gate corners through `camera_model` each tick, add pixel noise, drop frames when
  outside FOV, occasionally inject a garbage detection → feed
  `estimate_gates → GateMapper → RacePlanner → velocity_to_attitude` (full real stack,
  no sockets).
- Course = the real captured course geometry (use `captured_waypoints.json` /
  broadcast gate list from an old log as ground truth — allowed offline, fact 15).
- Assert: all gates passed in order **through the 1.5 m opening** (plane-crossing
  point within the inner square), no guard trips, total time under a budget; second
  run with the persisted map + `RACE_MODE=True` is ≥ 40 % faster.
- **Acceptance:** `python test_vision_race.py` → `ALL ... PASSED`, both scouting and
  racing scenarios.

### Phase 5 — Live integration (requires sim, human launches it — see `reference/VERIFY.md`)
1. `DRY_RUN=True` perception run: fly nothing; verify `observations` populate, mapper
   confirms gate 1 from the pad, `course_map` logs look sane. Re-verify HSV
   (CALIBRATION.md §2) — course may have changed.
2. `DRY_RUN=False`, scouting mode, speeds at scan defaults: complete one full lap on
   vision, map persisted. Compare `course_map.json` vs the broadcast `gates` in the
   log (offline): position error < 1 m per gate.
3. `RACE_MODE=True`: fly the learned map at speed; tune only `config.py` path
   constants (the controller is already calibrated — touch it last, if at all).
4. Remove the legacy single-gate `vision` keys (5.2), update `docs/IMPLEMENTATION.md`,
   `docs/TESTING.md`, `CLAUDE.md` via the docs-sync agent.
- **Acceptance:** full course, zero collisions (COLLISION msg count in the log),
  `target.source` never `known`/mission-derived, timed run reproducible twice.

---

## 8. Testing & instrumentation rules

- Every module ships with an offline test runnable as `python test_<x>.py` printing
  `ALL ... PASSED` (repo convention; no pytest dependency).
- The logger already snapshots `shared_data` at 10 Hz. New keys (`course_map`,
  `target.state`) ride along for free — check `logger.py` serializes numpy arrays
  (convert in the producers: publish plain lists, never ndarrays, into
  `shared_data`).
- Ground-truth grading: offline scripts may read the broadcast `gates` from logs to
  score the mapper. Flight code may not (grep-enforced, §5.7).
- When a live failure happens: pull the JSONL log, reproduce in the Phase-4 synthetic
  sim first, fix there with a regression case, then re-fly. **No fix goes in without
  an offline reproduction** — this rule is the entire lesson of the old planner.

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| PnP corner rate too low on real frames → size-method noise dominates | Phase 1 eval tool measures it before any flight; corner refinement (sub-pixel, `cv2.cornerSubPix` on the mask edges) is the first lever |
| Mapper mis-association on gates < 3 m apart | chi-square gate is distance-scaled; if the real course has adjacent gates, add pass-order prior (next expected gate gets association priority) |
| SCAN never finds a gate (bad HSV / gate behind drone) | SCAN widens sweep then rotates the base heading 360° in 45° steps; watchdog altitude floor prevents ground contact; 8-min budget is huge vs. a lap |
| Left/lateral lean sign (`LEAN_SIGN_LAT`) still unconfirmed live | Phase 5 step 2 starts at scan speeds where a flipped sign is a slow drift, not a crash; verify against log before RACE_MODE |
| Vision-frame timestamps vs. pose timestamps misaligned during fast yaw → world-projection smear | Mapper's R already inflates with range; if smear shows up in Phase-4 sim, interpolate pose to `sim_time_ns` in `estimate_gates` (both streams carry sim time) |

## 9b. ADDENDUM (2026-07-09): VQ2 platform — this plan now runs on branch `Qualifier2`

The new spec revision (`260624_Technical_Spec_0003.pdf`, VADR-TS-003 00.03,
extracted to `reference/_specs_vq2_text.txt`) changes the ground rules for
Round-2 qualification runs. Spec §9.3 **blocks** these simulator interfaces:
`ATTITUDE`, `LOCAL_POSITION_NED`, `ODOMETRY`, **and `GATE_INFO`**. So on a
qualification run there is **no pose telemetry and no broadcast gate geometry at
all** — pure vision stops being a design mandate and becomes a hard requirement,
and the "VIO-lite" horizontal estimator that §10 deferred is now in scope.

What changes relative to the main plan:

- **Pose source is abstracted, not assumed.** The mapper and planner read
  `shared_data['attitude']` / `shared_data['position_ned']` as before, but on VQ2
  those are produced by the on-board estimator (`ahrs.py` complementary filter +
  `state_estimator.py` baro-z observer, both already implemented and calibrated
  on this branch), not by `mavlink_rx.py`. Training-mode flights (spec §9.2) can
  still use real telemetry to validate the pipeline before switching.
- **New Phase 2b — vision-anchored horizontal position.** IMU+baro cannot
  observe x/y (the estimator publishes `x = y = None` today). Fix: the origin is
  exact at arm; the first gate observations are mapped from the origin-anchored
  pose; thereafter, every observation that associates to a confirmed mapped gate
  yields a drone-position fix `pos = gate.pos_ned − R_wb·gate_body`, published as
  `shared_data['pos_fix'] = {'x','y','var','ts'}`. `state_estimator.py` gains a
  horizontal observer: predict with AHRS-rotated world-frame acceleration
  (damped), correct with `pos_fix`. Drift is bounded whenever any mapped gate is
  in view; between gates it dead-reckons for the few seconds of the gap.
  Circularity is benign: the map and the pose are self-consistent (a shared
  offset cancels in the control error).
- **Fact 15 is void on VQ2** (no GATE_INFO): offline tests must take ground
  truth from training-mode logs or synthetic scenes instead.
- **§14 caveat**: race status availability under VQ2 is unconfirmed — do not
  hard-depend on `active_gate_index`; the plane-crossing pass check (§5.5 PASS
  state) must work standalone.
- Speeds: config on this branch is already dialed to a VQ2 safe baseline
  (`CRUISE_SPEED=3`); keep it there until the estimator chain is validated in
  training mode.

## 10. Out of scope (explicitly)

- **Qualifier-2 platform** (ATTITUDE/POSITION blocked): needs the `origin/Qualifier2`
  AHRS + a vision-aided horizontal position estimator (VIO-lite anchored on mapped
  gates). The `GateMapper` here is a prerequisite for that, but the estimator itself
  is a separate plan.
- Obstacle/environment perception beyond gates; multi-lap strategy optimization;
  learning-based detection (HSV + geometry is sufficient for the glowing gates and
  keeps everything offline-testable).
