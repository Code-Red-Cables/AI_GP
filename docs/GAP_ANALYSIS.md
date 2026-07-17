# Gap Analysis — Vision + IMU Autonomy (branch `Qualifier2_testing`)

**Date:** 2026-07-17 · **HEAD:** `3551a23` ("smooth 1st gate") + uncommitted tuning edits
**Goal being audited:** fully autonomous flight through the orange gates using **vision + IMU only**
(VQ2 rules: no ATTITUDE / LOCAL_POSITION_NED / ODOMETRY / GATE_INFO — spec §9.3).
**Audience:** implementation agents. Read `docs/VISION_REBOOT_PLAN.md` first (it is the binding
design), then this document. Every claim below was verified on 2026-07-17 with the commands in
Appendix A — re-verify before acting if the tree has moved.

---

## 1. Where the project ACTUALLY is (read this before anything else)

**The vision pipeline is not connected to flight. At all.**

- `config.py:35` has `USE_VISION = False`, so `setup.py:47` never starts `VisionRX` — the camera
  socket is never even opened in a live run.
- `setup.py:59-72` selects only three planners: teleop / `SplinePlanner` / `Planner`. **There is no
  code path anywhere that constructs `RacePlanner` or `GateMapper` for flight.** They exist only
  inside `tools/test_vision_race.py` and `tools/test_race_fsm.py`.
- Nothing in the flight stack ever calls `GateMapper.update()` — in the offline test the test loop
  feeds the mapper by hand (`tools/test_vision_race.py:157`).

What the recent live runs ("pass 1st gate", "lands on ground") actually were: `SplinePlanner`
replaying the hand-captured `mission.json` on **pure IMU dead-reckoning**. Evidence from
`logs/latest.jsonl` (380 lines, run of 2026-07-15):

```
t=1.6   pos=( 0.05, 0.00, -0.07)  src=spline:wp0  vel_cmd=(-0.50, -0.01, -0.50)
t=6.4   pos=( 3.25, 0.14, -1.93)  src=spline:wp0  vel_cmd=(-0.50, -0.01, +0.36)  col=[1001,2,4.15]
t=12.8  pos=( 5.31, 0.34, -1.94)  src=spline:wp0  vel_cmd=(-0.50, -0.01, +0.38)  gate_idx=1
```

Read that carefully: the planner commands **−0.5 m/s North** the whole run, while the estimated
position marches **+5.3 m North**. The estimator and the controller disagree about which way the
drone is pointing/moving (see G-13), the sim's `active_gate_index` still incremented to 1 (the
drone physically drifted through gate 1), **two COLLISION messages were logged**, and the drone
never got anywhere near `wp0` at n=−19.7. That is the "passes gate 1 then lands" symptom: it is
**estimator/sign divergence on a blind waypoint replay**, not a vision look-ahead problem —
vision was off.

The user-visible complaints map to real gaps like this:

| Symptom | Actual cause | Gaps |
|---|---|---|
| "passes 1st gate then lands / doesn't see other gates" | vision isn't wired into flight; live runs are dead-reckoned waypoint replay that diverges after ~10 s | G-1..G-3, G-13, G-14 |
| "no look-ahead to future gates" (true of RacePlanner too, offline) | APPROACH spline stops at the exit point, ADVANCE hovers, SCAN has no altitude floor and extrapolates underground, no FINISH state | G-6..G-10 |
| "want drone to fly steady" | horizontal velocity feedback is dead-reckoned garbage; frame/sign conventions patched against each other; config self-contradictory | G-13..G-16 |

---

## 2. Verified baseline — what actually works today

| Component | Status | Evidence (2026-07-17, bundled interpreter) |
|---|---|---|
| `vision/gate_detector.py` | **Good.** 97.5 % detection, 80.6 % 4-corner rate over 471 captured frames — meets the Phase-1 acceptance bar (≥95 % / ≥80 %) | `tools/eval_detector.py` output; contact sheet in `eval_results.html` |
| `vision/gate_estimator.py` | Works (PnP-first, size fallback); has correctness gaps G-19/G-20 | `test_pipeline_smoke.py` passes |
| `mapping/gate_mapper.py` | Unit test passes but it is a **stub vs. plan §5.3** (G-4); its own test prints "Total mapped gates (including inactive noise): 11" for a ~2-gate scene | `PYTHONPATH=. python tools/test_mapper.py` |
| `guidance/path.py` + `race_planner.py` FSM | Unit tests pass | `PYTHONPATH=. python tools/test_race_fsm.py`, `test_spline_mission.py` |
| `tools/test_vision_race.py` (Phase-4 keystone) | **FAILS.** Race run (perfect preloaded map): 18/18 gates in 243.5 s, but ends in SCAN at `pos=(−155, −4, z=+24.6)` — i.e. 24 m **underground**, 100 m past the course. Scout run (discovery): **4/18 gates, then stalls until the 3000 s budget expires.** | ran it; assertion fails at `tools/test_vision_race.py:242` |
| `ahrs.py` / `state_estimator.py` (attitude + baro-z) | Unit tests pass; calibrated signs 2026-06-27. Horizontal x/y is raw double-integration (drifts in seconds) — G-13/G-14 | `test_ahrs.py`, `test_state_estimator.py` |
| `controller.py` body-rate+thrust core | Proven live at speed on `spline-path`; currently strangled by uncommitted 4° lean caps (G-16) | git diff vs HEAD |
| Race status under VQ2 | **`active_gate_index` IS being received** on the current platform (log above shows gate_idx 0→1 with the estimator on). The spec caveat (§9b: "unconfirmed under VQ2") is resolved *for the training sim*; re-confirm once on a qualification run, but build on it now. | `logs/latest.jsonl`, `mavlink_rx.py:220` |

The offline scout failure **reproduces the user's live symptom in the vision stack too**: pass the
first few gates, lose the course, wander/descend forever. So even after wiring (G-1), the FSM gaps
below would produce the same behavior. Fix both layers.

---

## 3. Gap register

Ordered by priority. Each gap: **where → what's wrong → evidence → what to do → how to verify.**

### P0 — Integration (nothing else matters until these are done)

**G-1. RacePlanner/GateMapper are never instantiated for flight.**
`setup.py:59-72`. Add a fourth selection branch (highest priority when `USE_VISION`): build a
`GateMapper`, optionally load `course_map.json`, construct `RacePlanner(shared_data, mapper, config)`.
Plan §6 step 5 specified exactly this and it was never done.
*Verify:* `main.py` in DRY_RUN with `USE_VISION=True` prints `src=fsm:TAKEOFF` targets.

**G-2. No component feeds observations to the mapper.**
`VisionRX.process_frame` (`vision_rx.py:217`) publishes `est['observations']` and stops. Nothing in
the flight loop consumes it. Per plan §4 the mapper runs **inside the planner tick**: at the top of
`RacePlanner.compute_target()`, pop unconsumed observations from `shared_data['vision']` (track
`frame_id` to avoid double-integrating the same frame) and call `self.mapper.update(...)`. Also
publish `shared_data['course_map']` (plain lists, not ndarrays) each tick for the logger — specified
in plan §5.3, missing.
*Verify:* DRY_RUN against the sim from the start pad → logged `course_map` contains gate 1 with a
sane NED position.

**G-3. `config.py` is missing the entire vision-autonomy block.**
Plan §5.6 lists `RACE_MODE`, `COURSE_MAP_PATH`, mapper constants, and all FSM constants
(`TAKEOFF_ALT_M`, `SCAN_*`, `APPROACH_STANDOFF_M`, `EXIT_OVERSHOOT_M`, `COMMIT_DIST_M`,
`MAX_PATH_DIST_M`, `MAX_ALT_M`). None exist in `config.py`. `race_planner.py` papers over this with
`getattr(cfg, 'X', default)` everywhere — so a typo'd or missing constant silently becomes a
default and is untunable. Add the block; then make missing constants a hard error (drop the
defaults from the `getattr` calls).

### P1 — Correctness bugs that will corrupt any flight (fix before first vision flight)

**G-4. `GateMapper` is a stub compared to the binding spec (plan §5.3).**
`mapping/gate_mapper.py` (64 lines) has: fixed-radius (2 m) nearest-neighbor association, fixed
EMA α=0.2, hit-count activation. It is missing, in rough order of importance:
- **Observation quality is completely ignored** — `obs.confidence` and `obs.method` are never read.
  A conf-0.05 size-method spike updates the map exactly as hard as a conf-0.9 PnP fix. The spec's
  range-scaled measurement noise (`R = (σ0 + k·range)²`, ×4 for size method, D-axis inflated by
  `max(1,(range/12)²)` to encode the +20° camera's far-elevation error) is the mechanism that was
  designed to absorb documented failure modes 9–12; none of it exists.
- **No time-windowed confirmation** (spec: 3 obs within 2 s) and **no pruning** — a garbage burst
  leaves phantom candidates forever (`test_mapper.py` output: 11 gates mapped in a ~2-gate scene).
  Phantoms are fatal combined with G-8 (insertion-order gate identity).
- **No persistence** (`to_json`/`from_json`, `course_map.json`) — RACE_MODE (the fast lap on a
  scouted map) is impossible without it.
- **No `passed` flag / `next_unpassed()` / ordering anchor.** Spec says anchor ordering and
  `passed` on `active_gate_index` — which we now know is available (see §2).
*Verify:* extend `tools/test_mapper.py` per plan Phase 2 acceptance: 3 static gates + 5 % garbage
bursts → all 3 confirmed, RMSE < 0.5 m, **zero phantom gates**, json round-trip.

**G-5. Gate normal sign is never disambiguated — approach/pass logic can be 180° wrong.**
PnP returns a normal that can point toward *or* away from the camera. `estimate_gates`
(`vision/gate_estimator.py:94-97`) explicitly punts ("here we just attach the normal") and the
mapper never flips it either — `gate_mapper.py:34` feeds `atan2(normal_ned)` straight into the yaw
average. Consequences: `APPROACH` builds its standoff on the wrong side (`race_planner.py:163`,
`pos − standoff·normal`), and `PASS`'s plane-crossing check `dot(pos−gate, normal) > 0`
(`race_planner.py:190-192`) fires **immediately on entry** for a flipped normal — instant
false "passed". Fix in the mapper (per spec): on first association, flip the incoming normal into
the hemisphere facing the drone (`dot(normal, drone_pos − gate_pos) > 0`), and flip every
subsequent observation to match the stored hemisphere before averaging.
**Sub-bug:** size-method observations carry `normal_body = zeros(3)` (`gate_estimator.py:131`) →
`atan2(0,0) = 0` → every corner-less detection drags the gate's yaw toward due-North. The mapper
must skip yaw updates for observations without a real normal (`method == 'size'`).
**Sub-bug:** `mapper.update()` crashes on `obs.gate_ned is None` (published when attitude isn't
available yet — first frames of a run): `gate_mapper.py:40` does `g.pos - None`. Guard and drop.

**G-6. No look-ahead past the current gate — the literal complaint.**
`race_planner.py:166`: the APPROACH path is `[pos, standoff, gate, exit]` and **stops at the exit
point**. Plan §5.5 says: append "(next gate's standoff if mapped)". Because of the +20° up-tilted
camera (hard-won fact 6), the next, lower gate only enters the FOV if the drone keeps moving and/or
descends after the pass — a path that dead-ends 2.5 m past the gate guarantees the "never sees the
next gate" outcome. Fixes, all per plan:
1. APPROACH: when `mapper` has a gate after the current one, extend the spline
   `... exit → next_standoff → next_gate → next_exit`.
2. ADVANCE (`race_planner.py:215-227`): currently returns a **hover** tick — kills all momentum at
   every gate. When the next gate is unmapped it must **coast on the exit heading with gentle
   descent** (plan: "short coast on exit heading with gentle descent → SCAN biased toward the exit
   normal"), not stop.
*Verify:* extended `test_vision_race.py` scout run passes all 18 gates (it currently dies at 4/18
precisely here — after an early pass, the next gate never becomes mapped and SCAN flails).

**G-7. SCAN can command the drone underground; no altitude floor anywhere.**
`race_planner.py:97-101`: with `idx ≥ 2` SCAN sets
`seek_alt = g_last.z + (g_last.z − g_prev.z)` — a linear extrapolation with no clamp, chased at
`KP_VERT_PATH`. Observed result (race run): terminal state SCAN at **z = +24.6 (24 m below the
origin)**. The alt guard (`race_planner.py:63-65`) only catches flying too HIGH. Add: (a) clamp
`seek_alt` to `[SCAN_ALT_MIN_M ceiling…floor]` band; (b) a global MIN-altitude guard symmetric to
`MAX_ALT_M` (the plan's SCAN spec already includes `SCAN_ALT_MIN` — it was never implemented).

**G-8. Gate identity = mapper insertion order — breaks whenever ≥2 gates are visible.**
`race_planner.py:40-43` (`_find_target_gate`) indexes `mapper.get_active_gates()[active_gate_idx]`,
and the comment's claim ("mapper maintains insertion order matching observation order i.e. closest
gates first") is wishful: with 2–3 gates in one frame (which the scout test shows happens
immediately — `mapper=3` at step 0), insertion order is detection-sort order of the first frame,
not pass order. Combined with phantom gates (G-4) a single garbage candidate shifts every
subsequent index. Fix: order gates by course progression (anchor on `active_gate_index` when
available; else by pass sequence — chain each gate's exit to the nearest-unpassed candidate in the
exit hemisphere), and target `mapper.next_unpassed()`, per plan §5.3/5.5.

**G-9. No end-of-course handling — FINISH is unreachable.**
Nothing ever sets `state = 'FINISH'`. After the last gate, ADVANCE increments the index past the
map and drops into SCAN forever (race-run evidence: `g_idx=18`, SCAN, 100 m past the course,
underground). When `active_gate_idx ≥` course length (race status `race_finish_time_ns` ≥ 0, or
all mapped gates passed): FINISH = maintain `FINISH_SPEED` across the line then brake to hover.

**G-10. PASS commands raw `CRUISE_SPEED`, bypassing every cap; and has no timeout.**
`race_planner.py:145-154`: `frozen_vel = dir * CRUISE_SPEED` — with the current uncommitted config
that is **10 m/s**, while APPROACH (through `carrot_velocity`, `guidance/path.py:205`) is capped at
`MAX_SPEED = 0.5 m/s`. A 20× step command at the commit point, at 4° lean authority (G-16), means
the drone crosses the plane ballistic and blind. Cap the commit speed (`min(CRUISE_SPEED,
MAX_SPEED)` or a dedicated `COMMIT_SPEED`). Separately: if the plane-crossing never triggers (bad
normal, bad map), PASS holds `frozen_vel` **forever** — add a distance/time budget
(~`2·(COMMIT_DIST+EXIT_OVERSHOOT)` of travel) that falls back to ADVANCE-without-mark or SCAN.

**G-11. APPROACH rebuilds the Catmull-Rom path from scratch every 60 Hz tick.**
`race_planner.py:169` constructs `Path(...)` (spline sampling at 40/seg + O(n²) curvature stencil)
every `compute_target()` call, because `pos` is a path waypoint. Besides CPU, this defeats
pure-pursuit (`_last_t` projection memory is meaningless across rebuilds) and makes the "path" an
ever-moving point-chase. Rebuild only on: state entry, target-gate estimate moving > `REFIT_THRESHOLD_M`
(plan §5.5 racing mode), or next-gate appearing in the map.

**G-12. `active_gate_index` (available!) is unused by planner and mapper.**
See §2 — the sim broadcasts it under the current VQ2 training platform and `mavlink_rx.py:220`
already parses it. It is the designed gate-passed signal (fact 14) and the ordering anchor (G-8),
and doubles as a cross-check for PASS (plan: "plane crossed OR active_gate_index incremented").
Wire it through `RacePlanner.compute_target()` (read `shared_data['race']`), but keep the
plane-crossing check working standalone (spec §9b caveat).

### P1b — State estimation (the IMU half of "vision + IMU")

**G-13. Frame/sign conventions are inconsistent and patched against each other — this is the
"backward drift" and the top steadiness killer.**
The facts in tension:
- The sim's *reported* yaw (training telemetry) is **left-handed** vs NED; `velocity_to_attitude`
  rotates the velocity error by `−yaw` to compensate (`controller.py:224-230`).
- The VQ2 AHRS yaw (`ahrs.py`) is integrated from the gyro in a **right-handed NED** convention,
  and `state_estimator._horizontal` (`state_estimator.py:119-134`) uses the right-handed rotation.
- `GYRO_SIGN[2]` is explicitly **unverified** (`ahrs.py:44`), `LEAN_SIGN_LAT` was flipped to −1.0
  "to fix left veer" (`controller.py:132`), and `StateEstimator.__init__` seeds yaw with
  `atan2(e, n)` of mission wp0 (`state_estimator.py:52-57`) — a hack that (a) assumes the drone
  boots facing wp0, (b) does nothing when there is no mission (i.e. in RacePlanner mode), and
  (c) mixes the two yaw conventions.
If the estimator's yaw convention doesn't match what the controller assumes, the body-frame
rotation mirrors the lateral axis — commanded West, flies East — exactly the class of failure in
the log (§1) and in run_1781506925 history. **Do not add more sign flips.** Run one dedicated
calibration session (Appendix B) that pins down, in a single frame-convention document:
sim-physical-forward vs AHRS yaw, `GYRO_SIGN[2]`, `LEAN_SIGN_LAT`, and the `_horizontal` rotation
handedness — each from a scripted single-axis live test in training mode with ground truth
enabled. Then delete the wp0 yaw-seed hack: define world frame = boot heading (yaw₀ ≡ 0). The map
and controller only need *self-consistency*, not true North (plan §9b: "a shared offset cancels").

**G-14. Phase 2b (vision-anchored horizontal position) is not implemented — the keystone VQ2 gap.**
`state_estimator.py` dead-reckons x/y by double-integrating accel; drift is unbounded (visible
within 10 s in the live log). Plan §9b specifies the fix and it exists nowhere: every observation
that associates to a **confirmed mapped gate** yields a drone-position fix
`pos = gate.pos_ned − R_wb·gate_body`, published as `shared_data['pos_fix'] = {x, y, var, ts}`;
`state_estimator` gains a horizontal observer (predict with rotated accel, damped; correct with
`pos_fix`). Producer: the mapper (it owns the association). Without this, *no* horizontal plan —
spline replay or RacePlanner — can work under VQ2 for more than a few seconds.
Note the docstring lie while you're in there: `state_estimator.py:10-15` still claims "x/y are
None, vx/vy published as 0.0" but the code publishes drifting `_x/_vx` (lines 187-191). The
comment describes the design that was *safe*; the code silently switched to the design that
diverges. Reconcile.

**G-15. Horizontal velocity feedback into the controller is dead-reckoned noise.**
`controller._telemetry` (`controller.py:302-305`) feeds `position_ned.vx/vy` into the lean loop.
Under VQ2 those are the integrated-drift values from G-14 — so the "velocity error" the drone
leans on is fiction, which both destabilizes (wobble as the loop chases drift) and mis-steers.
Until G-14 lands: gate horizontal-velocity feedback on `position_ned.estimated` + a variance/health
flag — fall back to leaning on desired velocity only (open-loop, as the estimator originally
published 0.0 for). After G-14: use the corrected velocity.

### P2 — Configuration & controller coherence ("fly steady")

**G-16. The uncommitted config/controller state is self-contradictory. Commit an intentional set.**
Current working-tree values (all uncommitted diffs vs `3551a23`):
- `CRUISE_SPEED = 10.0` but `MAX_SPEED = 0.5` — `carrot_velocity` caps at `MAX_SPEED`
  (`guidance/path.py:194,205`), so the real cruise is 0.5 m/s and every comment saying otherwise
  lies; meanwhile PASS ignores the cap entirely (G-10).
- `A_LAT_MAX = A_LON_MAX = 3.0` but `MAX_PITCH_RAD = MAX_LEAN_RAD = 4°` → the airframe can produce
  at most `g·tan 4° ≈ 0.69 m/s²`. The config's own "CRITICAL COUPLING" comment
  (`config.py:117-122`) forbids exactly this: the planner will command corner speeds the drone
  physically cannot fly, so it drifts wide off every curve. Keep `A_LAT_MAX ≤ g·tan(MAX_LEAN_RAD)`.
- `FINISH_SPEED = 8.0` — unreachable at a 0.5 m/s cap; `KP_LEAN = 0.03` with a 4° cap gives a
  proportional range of ±2.3 m/s (fine), but `HOVER_THRUST/KP_THRUST` also moved (0.30→0.32,
  0.30→0.15) in the same uncommitted blob.
Rule for agents: pick ONE coherent slow-flight tier, e.g. `MAX_SPEED=CRUISE_SPEED=1.0`,
`A_LAT=A_LON=0.5`, `FINISH_SPEED=0.5`, lean caps 6–8°, and **commit it with a message saying what
tier it is**. Speed ladder later (§4, WP-7). Also note `DRY_RUN = False` is committed on this
branch — plan §6 step 6 mandates `DRY_RUN = True` as the committed default; restore it.

**G-17. Uncommitted `guidance/path.py` change: yaw now sampled at the carrot (`s_carrot`) instead
of the projection (`s_proj`)** (`guidance/path.py:221`). This is plausibly right (turn the nose
toward where you're going — helps keep the next gate in the FOV) but it is **untested and
unlogged**: no offline test pins the choice and REBOOT_LOG has no entry. Add a regression case to
`test_spline_mission.py` (yaw begins turning before a corner by ~LOOKAHEAD) and commit with
rationale, or revert.

### P3 — Vision quality & pipeline hygiene

**G-18. Detector is good; its *confidence* is meaningless in practice (mean 0.296) and nothing
consumes it anyway.** The factored model (`gate_detector.py:208-230`) multiplies four [0,1] terms,
so real gates score ~0.3 and there is no usable "garbage" threshold. Since the mapper's R-model
(G-4) is the designed consumer, calibrate then: pick the confidence→σ mapping from the eval set
(`eval_results.html` has per-frame values). Until the mapper reads confidence, tuning it is wasted
work — sequence after G-4.

**G-19. `estimate_gates` uses the pose at *processing* time, not *frame* time.**
`vision_rx.py:160-176` snapshots attitude/position when the frame is decoded, then stamps the
observation with the frame's `sim_time_ns`. During a yaw sweep at 70 °/s a 100 ms skew smears the
world-projection by ~7° of bearing — the exact "association smear" risk the plan flags (§9 last
row). Both streams carry sim time; keep a short ring buffer of recent (ts, attitude, position) in
`shared_data` (producer: state estimator / mavlink_rx) and interpolate to `sim_time_ns` in
`estimate_gates`. Do this when the mapper starts mis-associating during turns in the Phase-4 sim —
the plan calls it the FIRST lever for that symptom.

**G-20. Assorted vision bugs (small, real):**
- `vision_rx.py:225` — `draw_detection(img, det)`: **`det` is undefined** in `process_frame`
  (renamed to `dets`/`obs_list`); with `DEBUG_VISION=True` this NameErrors every frame and the
  bare `except` silently eats it, so debug overlays just never appear. Fix to
  `dets[0] if dets else None`.
- `vision_rx.py:217` publishes raw `GateObservation` dataclasses (numpy arrays inside) into
  `shared_data` — plan §8: publish plain lists/dicts only (logger/json safety).
- The legacy single-gate keys get outlier-rejection + EMA (`_filter_estimate`) but the
  `observations` list the mapper will consume is **unfiltered** — fine *if* the mapper implements
  its gating (G-4); until then the mapper eats raw spikes. Keep this in mind about ordering: G-4
  before G-2 goes live, or gate observations crudely (range < 40 m, confidence floor) in the
  interim.
- `tools/eval_detector.py:14` (uncommitted): the `'.'` glob sweeps repo-root PNGs including the 20
  `_vision_*.png` **annotated overlay outputs** — self-grading on its own drawings. Evaluate only
  `frames/` + `_vision_debug/`, and write the stats line INTO `eval_results.html` (today it goes
  to stdout only and is lost).

**G-21. The Phase-4 "keystone" test is too idealized to protect you.**
`tools/test_vision_race.py` bypasses the detector (synthesizes perfect corners ±0.5 px), flies a
kinematic particle (commanded velocity = instant velocity; no controller, no lean dynamics, no
estimator), always projects with roll=pitch=0, never drops frames, never injects garbage bursts —
all things the plan's Phase-4 spec requires. It therefore cannot catch G-5/G-13/G-15 classes at
all. Extend it per plan: reuse `test_spline_mission.py`'s kinematic drone WITH
`velocity_to_attitude` + the left-handed-yaw model in the loop, run the state estimator on
synthesized IMU when `USE_STATE_ESTIMATOR`, add FOV-loss from lean, 5 % garbage detections, frame
drops. This test is the gate for every flight change (repo rule: no fix without offline repro).

### P4 — Process / hygiene

**G-22.** `tools/test_mapper.py` and `tools/test_race_fsm.py` fail as documented
(`ModuleNotFoundError: vision`) — they lack the `sys.path` shim `test_vision_race.py` has. Add it
so `python tools/test_x.py` works from repo root (repo convention).
**G-23.** Docs are stale/misleading for anyone landing here: `CLAUDE.md` top section describes
`modified-starter`/`preplanning`/`spline-path` and never mentions Qualifier2/VQ2/RacePlanner;
`docs/REBOOT_LOG.md` stops at 2026-07-09 Phase 0, yet Phases 1–4 code exists (commits `5f67e01`,
`de8f828`, and untracked Phase-3/4 work). Backfill the log (what was built, what deviated from
plan — notably the mapper simplification), and run the docs-sync agent after each WP below.
**G-24.** Repo clutter: 20 `_vision_*.png` at root, 470 files in `eval_out/` untracked,
`test_vision.py` (trivial wrapper) untracked, `eval_results.html` modified. Gitignore
`eval_out/`, `_vision_*.png`; decide on the wrapper.

---

## 4. Recommended execution order (work packages for agents)

Do them in order; each has an offline acceptance gate. **Controller core (`velocity_to_attitude`,
rate loop, signs) is frozen except via WP-4's calibration protocol.**

| WP | Scope (gaps) | Acceptance |
|---|---|---|
| **WP-0** | Commit-or-revert the working tree into a coherent slow tier; restore `DRY_RUN=True`; fix test shims & clutter (G-16, G-17, G-22, G-24) | clean `git status`; all existing suites runnable per docs |
| **WP-1** | Mapper to spec + normal-sign flip + crash guards (G-4, G-5) | extended `test_mapper.py`: 3 gates + bursts → 0 phantoms, RMSE < 0.5 m, json round-trip, flipped-normal case |
| **WP-2** | Wire vision flight: setup selection, mapper feed in planner tick, config block (G-1, G-2, G-3) | DRY_RUN vs live sim: `course_map` logs gate 1 from the pad; `src=fsm:*` targets logged |
| **WP-3** | RacePlanner overhaul: next-gate path extension, ADVANCE coast, SCAN floor+bias, FINISH, PASS cap+timeout, gate ordering via `active_gate_index`, path-rebuild throttle (G-6…G-12) | hardened `test_vision_race.py` (G-21): scout 18/18, race 18/18, **z never below ground**, no state stuck > 20 s |
| **WP-4** | One calibration session for all sign/frame questions; delete wp0-yaw-seed hack; write `docs/FRAMES.md` as the single convention source (G-13) | scripted single-axis training-mode runs match prediction; offline closed-loop sim uses the SAME convention constants |
| **WP-5** | Phase 2b estimator: gate-anchored `pos_fix`, horizontal observer, velocity-feedback gating (G-14, G-15) | `test_state_estimator.py` extension: DR-only drift vs fix-corrected < 1 m over a synthetic lap; Phase-4 sim passes with estimator in the loop |
| **WP-6** | Vision polish as needed: pose interpolation, confidence→σ calibration, debug-overlay fix (G-18, G-19, G-20) | eval tool ≥ same detection stats; Phase-4 sim association stable during yaw sweeps |
| **WP-7** | Speed ladder on live sim: 1 → 3 → 6 m/s, raising lean caps + A_LAT/A_LON together per the coupling rule; RACE_MODE with persisted map | full course on vision, zero COLLISION msgs, reproducible twice (plan Phase-5 acceptance) |

Rules carried over (binding): every fix lands with an offline reproduction first; grep must show no
flight-loop reads of broadcast `gates`; command rate < 100 Hz; no human input in timed runs.

---

## Appendix A — How every claim above was verified (repro commands)

Bundled interpreter: `C:\Users\rocky\docs\AI_GP\PyAIPilotExample\myenv\Scripts\python.exe` (= `$PY`).

```bash
git log --oneline -8 && git diff --stat HEAD          # branch state + uncommitted soup
$PY tools/eval_detector.py                             # → frames: 471, det 97.5%, 4-corner 80.6%, conf 0.296
$PY tools/test_vision_race.py                          # → race 18/18 but ends SCAN underground; scout 4/18 FAIL
PYTHONPATH=. $PY tools/test_mapper.py                  # → passes; "11 mapped gates incl. noise"
PYTHONPATH=. $PY tools/test_race_fsm.py                # → passes
# latest live run (spline replay on dead reckoning):
#   parse logs/latest.jsonl: src=spline:wp0 throughout, vel_cmd −0.5 N vs pos +5.3 N, col=[1001,...]x2
```

## Appendix B — WP-4 calibration protocol sketch (training mode, ground truth ON)

Goal: one session, four unknowns, no compensating flips afterward.
1. **Gyro-Z sign:** hover, command pure `yaw_rate` +0.3 rad/s for 2 s; compare AHRS yaw delta sign
   vs ground-truth ATTITUDE yaw delta (mind the left-handed truth convention when comparing).
2. **AHRS-yaw ↔ physical-forward:** face boot heading, command body-forward 1 m/s (pitch only);
   record ground-truth NED velocity direction; compute the rotation the estimator must use so that
   `_horizontal` reproduces it. This fixes the `_horizontal` handedness AND the
   `velocity_to_attitude` rotation for estimated yaw in one measurement.
3. **LEAN_SIGN_LAT:** from the same hover, command body-right 1 m/s; sign of ground-truth lateral
   motion settles it (never confirmed on a clean leg — controller.py:128-131).
4. Write results into `docs/FRAMES.md` with the raw log filenames; update `ahrs.py` /
   `controller.py` / `state_estimator.py` constants in ONE commit.
