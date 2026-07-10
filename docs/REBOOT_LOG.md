# Vision Autonomy Reboot — Progress Log

Running log of the implementation of `docs/VISION_REBOOT_PLAN.md` on branch
`Qualifier2`. Newest entries at the bottom. Each entry: what was done, what was
found, what's next. Written for other agents picking up the work — read this
top-to-bottom, then the plan, before touching code.

---

## 2026-07-09 — Session start: branch state survey (Phase 0 prep)

**Done:**
- Plan (`docs/VISION_REBOOT_PLAN.md`) was committed by the user onto local
  `Qualifier2` (dd13410). Working there per user direction (overrides the plan's
  §6 "branch `vision-autonomy` from `spline-path`" — Qualifier2 is a superset of
  spline-path, so nothing is lost).
- The dirty `manual-control-vq2` working tree (vision-chaser graft,
  `planner_modified.py`, `drone_gui.py`) was committed by the user as c3f3580 on
  `manual-control-vq2` — that branch **is** the WIP archive (plan §6 step 7
  satisfied; no separate archive branch created).

**Branch contents (Qualifier2 = spline-path + VQ2 additions):**
- From spline-path: calibrated `controller.py` (body-rate+thrust, split lean
  caps), `spline_planner.py` (Catmull-Rom + pure-pursuit + curvature braking),
  `config.py` single-source pattern, `planner.py` (waypoint stopper),
  `teleop.py`, all offline tests.
- VQ2 additions (955d2ab "v2 commit", 2026-06-28): `ahrs.py` (complementary
  filter, calibrated sensor signs), `state_estimator.py` (attitude + baro-z
  observer; **x/y position = None**), `hover_planner.py` (Phase-0 hover
  milestone), `260624_Technical_Spec_0003.pdf` (NEW spec revision — reading
  next), mavlink_rx HIGHRES_IMU parsing, conditional planner selection in
  config/setup.
- `vision/gate_detector.py` present with calibrated HSV; `gate_estimator.py`
  at repo root (plan moves it to `vision/`).
- No slop files on this branch (`planner_modified.py` / `drone_gui.py` exist
  only on `manual-control-vq2`).

**Next:** read the new spec PDF (what telemetry does VQ2 block? — determines
whether the GateMapper can use MAVLink pose or must run on estimated state),
then baseline the existing offline tests, then Phase 0.

---

## 2026-07-09 — Spec finding + Phase 0 complete

**CRITICAL SPEC FINDING:** extracted the new spec PDF with pypdf (installed into
the bundled venv) → `reference/_specs_vq2_text.txt`. **VADR-TS-003 §9.3 blocks
`ATTITUDE`, `LOCAL_POSITION_NED`, `ODOMETRY` AND `GATE_INFO` on VQ2
qualification runs.** No pose telemetry, no broadcast gates. Consequences
written up as a new **§9b addendum in `docs/VISION_REBOOT_PLAN.md`** — read it.
Headline: a new Phase 2b (vision-anchored horizontal position estimator:
gate-relative fixes `pos = gate.pos − R_wb·gate_body` correcting an IMU
dead-reckoning observer in `state_estimator.py`) is now REQUIRED, and
`active_gate_index` must not be hard-depended on (race-status availability
under VQ2 unconfirmed — PASS detection needs the plane-crossing check to work
standalone). Spec §9.2: "Training" flights still exist for free validation;
§9.4: unlimited attempts, best time counts, code audits on suspicion.

**Baseline:** all 6 offline suites pass on the bundled interpreter
(camera_model, pipeline_smoke, spline_mission, ahrs, state_estimator,
waypoint_mission).

**Phase 0 done:**
- `git mv gate_estimator.py vision/gate_estimator.py`; imports fixed in
  `vision_rx.py` and `test_pipeline_smoke.py`.
- Created empty packages `mapping/` and `guidance/` (plan §5.3–5.5 will fill
  them).
- `DRY_RUN=True` already the committed default on this branch — kept.
- No slop to delete here (`planner_modified.py`/`drone_gui.py` never existed on
  this lineage). `planner.py` here is the clean waypoint-stopper, kept as a
  debug fallback.
- Tests re-run green after the move.

**Bug spotted for Phase 1:** `vision/gate_estimator.py::estimate_gate`
overwrites `normal_body` with the NED-rotated normal when attitude is given —
the returned `normal_body` is silently in the wrong frame. Fix = separate
`normal_ned` field (plan §5.2 already specifies this).

**Next:** Phase 1 (multi-gate `detect_gates()`, factored confidence,
`estimate_gates()` → GateObservation list, VisionRX publishes `observations`,
smoke-test update, detector eval tool).
