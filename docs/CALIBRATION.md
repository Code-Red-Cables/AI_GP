# Calibration & tuning guide

Three things need calibrating/tuning against the **real** simulator, in this order:

1. **HSV thresholds** — so the detector actually sees the gate (required; placeholders ship).
2. **Detector filters** — so it picks the gate cleanly and finds 4 corners.
3. **Guidance gains** — so the drone flies the course fast but stable.

The camera model needs **no** calibration — intrinsics and the 20° tilt are fixed
by the spec and already unit-tested. You only *verify* it (§4).

Interpreter for all commands (the bundled venv has numpy/cv2/pymavlink):
```
$PY = "C:\Users\rocky\docs\AI_GP\PyAIPilotExample\myenv\Scripts\python.exe"
```

---

## 1. Capture real frames
Launch the sim, log in, start a Round 1 race so video streams (see
[`../reference/VERIFY.md`](../reference/VERIFY.md)), then dump frames:
```powershell
& $PY tools/capture_frames.py 60 reference/frames
```
This binds UDP 5600, reassembles JPEGs, and saves `reference/frames/frame_0000.png …`.
Grab frames where a gate is clearly in view at a few distances/angles.
(`reference/frames/` is git-ignored.)

---

## 2. Calibrate HSV thresholds
The detector masks the gate by colour in HSV (OpenCV ranges: **H 0–179, S 0–255,
V 0–255**). Round 1 is a *high-contrast, desaturated* environment, so the gate is
the saturated/bright object — generally a **high-S, high-V** band.

```powershell
& $PY tools/hsv_tuner.py reference/frames/frame_0000.png
```
Drag the 6 trackbars (`Hlo/Hhi/Slo/Shi/Vlo/Vhi`) until the **mask shows only the
gate** (right pane = original masked by your range). Aim for the gate solid white
with little background leakage. Press `q`; it prints:
```
LOWER_HSV = (h, s, v)
UPPER_HSV = (h, s, v)
```
Paste those into `vision/gate_detector.py`:
```python
LOWER_HSV = (h, s, v)
UPPER_HSV = (h, s, v)
```
Repeat on a few frames (near/far, lit/shadowed) and pick a range that covers all.

**Hue wrap (red gates):** if the gate hue straddles the 0/179 boundary, one range
can't capture it. Set the second range too — it's ORed into the mask:
```python
LOWER_HSV  = (0,   120, 120);  UPPER_HSV  = (10,  255, 255)
LOWER_HSV2 = (170, 120, 120);  UPPER_HSV2 = (179, 255, 255)
```

**Verify** on saved frames:
```powershell
& $PY vision/gate_detector.py reference/frames/frame_0000.png   # writes _detect_debug.png
```
Open `_detect_debug.png`: the green bbox should hug the gate, the red dot sit at the
opening center, and TL/TR/BR/BL magenta corners land on the inner square. Good
detections report `confidence` ≳ 0.5 and (for PnP) `corners_px` not None.

**Calibrated values (Round 1, 2026-06-03).** The shipping placeholders were replaced
with the values now in `vision/gate_detector.py`:
```python
LOWER_HSV  = (0,   0, 80)   # red/orange gate; two-piece hue mask for wrap-around
UPPER_HSV  = (15,  255, 255)
LOWER_HSV2 = (170, 0, 80)   # second range to capture hue near 180 (OpenCV hue wraps at 0/180)
UPPER_HSV2 = (180, 255, 255)
```
The two-piece mask handles the **OpenCV hue wrap-around** at 0/180: red/orange hues are in the
0–15 and 170–180 ranges, so both `inRange` calls are ORed into a single mask. The `S` floor of
0 and `V` floor of 80 pick out the saturated, bright gate in a desaturated environment
(the V threshold of 80 ensures only well-lit regions are detected). **Detection performance needs re-verification against `reference/frames/`** after
deployment to ensure the gate is reliably detected across range, lighting, and viewing angles.
A prior offline validation on 60 saved frames found good detection with sensible geometry; run
this check again after any HSV edit or before a timed run.

---

## 3. Tune detector filters (`vision/gate_detector.py` → `DEFAULT_CFG`)
Only if detection is noisy after HSV is right:

| knob | default | raise it to… | lower it to… |
|------|---------|--------------|--------------|
| `min_area` | 400 px | reject small blobs / distant noise | detect gates from further away |
| `min_extent` | 0.30 | demand a more solid shape | allow thin/partial frames |
| `kernel_size` | 5 | close bigger gaps (blurrier mask) | preserve fine edges |
| `approx_eps_frac` | 0.04 | force simpler quads (helps get exactly 4 corners) | keep more polygon detail |

If `corners_px` is often `None` (so PnP can't run and it falls back to the size
method), nudge `approx_eps_frac` up to ~0.05–0.06 so `approxPolyDP` collapses the
opening to a clean 4-gon. You can pass a `cfg` dict to `detect_gate()` to A/B test
without editing the file.

---

## 4. Verify the camera model (no calibration, just a sanity check)
The geometry is fixed and unit-tested (`& $PY test_camera_model.py`). To sanity-
check against reality: place the drone a known distance in front of a gate, run the
pipeline, and confirm `vision['range_m']` matches the true distance and
`vision['bearing']` is ~0 az / ~+20° el when the gate is centered in the image
(the +20° is the camera tilt — expected, not an error). If range is consistently
off by a constant factor, suspect the detected opening size (HSV/filters), not the
model.

---

## 5. Tune control gains (attitude layer + guidance layer)

### 5a. Tune attitude gains (`controller.py` — MUST DO FIRST)

The DCL simulator is a Betaflight-style FPV racer (ANGLE mode) with **no velocity loop**.
Flight control uses attitude+thrust (`SET_ATTITUDE_TARGET`). **These gains must be tuned on
the live simulator, in this order**, before guidance gains are adjusted.

| constant            | default | meaning                                                        | symptom → change                                                             |
| ------------------- | ------- | -------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `HOVER_THRUST`      | 0.35    | collective thrust (0..1) that holds altitude level (TUNE FIRST)| **climbs while level/hovering** → lower; **sinks** → raise                   |
| `KP_THRUST`         | 0.15    | extra thrust per m/s of vertical-velocity error                | sluggish altitude correction → raise; altitude oscillates → lower            |
| `THRUST_MIN`        | 0.05    | minimum collective thrust (safety floor)                       | rarely changed unless drone goes too slow                                   |
| `THRUST_MAX`        | 0.9     | maximum collective thrust (safety cap)                         | rarely changed unless drone can't climb fast enough                         |
| `KP_LEAN`           | 0.15    | rad of lean per m/s of horizontal-velocity error               | **sluggish forward/lateral flight** → raise; oscillates/overshoots → lower  |
| `MAX_LEAN_RAD`      | 20.0°   | cap on commanded pitch/roll angle                              | rarely tuned; raise if pitch/roll authority seems limited                   |

**Workflow (attitude gains):**
1. Set `DRY_RUN=True` initially to validate perception safely.
2. Set `DRY_RUN=False`, arm, and confirm the drone **holds altitude level** with zero lean
   when hovering. If it climbs, lower `HOVER_THRUST` slightly; if it sinks, raise it.
   **Do not proceed until altitude is stable.**
3. Once altitude-hold is solid, raise `KP_LEAN` (start ~0.15) to test forward/lateral responsiveness.
   Observe the `[FLY]` console line: lean angles should be small (~0.1 rad = 6°) for moderate speeds.
4. If the drone oscillates in pitch/roll, lower `KP_LEAN` and/or raise `KP_THRUST`.
5. Once attitude is stable, proceed to guidance-gain tuning (below).

### 5b. Tune guidance gains (`planner.py`)
Start conservative, fly in `DRY_RUN` first (read the `[DRY]` log lines), then enable
flight (`DRY_RUN=False` in `main.py`) and tune against the deterministic course.

| constant            | default | meaning                                                          | symptom → change                                                         |
| ------------------- | ------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `MAX_SPEED`         | 2.0 m/s | velocity magnitude cap                                           | too slow to be competitive → raise gradually; overshooting gates → lower |
| `MAX_VSPEED`        | 1.0 m/s | **vertical component cap** (climb or descend)                    | climbing too aggressively → lower; sluggish vertical → raise (but < MAX_SPEED) |
| `KP_POS`            | 3.0     | `speed = KP_POS·distance` (before cap)                           | sluggish approach → raise; oscillates/overshoots → lower                 |
| `PASS_THROUGH_DIST` | 2.5 m   | within this range, command full speed to commit through the gate | stalling *in* the gate → raise; blowing past misaligned → lower          |
| `ARRIVE_DIST`       | 0.1 m   | "at waypoint" threshold                                          | rarely needs changing                                                    |
| `CONF_MIN`          | 0.40    | min vision confidence to trust a detection                       | chasing false gates → raise; ignoring real gates → lower                 |
| `VISION_TIMEOUT_NS` | 300 ms  | older vision is "stale" → fall back to known geometry            | jittery switching → raise; acting on stale frames → lower                |
| `TELEM_TIMEOUT_NS`  | 500 ms  | no pose for this long → hover (watchdog)                         | nuisance hovers → raise; flying blind on dropouts → lower                |
| `MAX_ALT_M`         | 15.0 m  | altitude ceiling above arm point (safety envelope)               | **rarely tuned** — if hitting it, investigate root cause (see note below) |

Workflow: change one constant, re-run, watch the `[DRY]`/`[FLY]` console line and the
JSONL log, compare against the previous run (the sim is deterministic, so runs are
directly comparable). Controller rate (`CONTROL_HZ=60` in `controller.py`) must stay
**< 100 Hz** per spec — don't raise it past ~90.

**Note on `MAX_VSPEED` and `MAX_ALT_M`:** The vertical component of the commanded
velocity is capped separately at `MAX_VSPEED` (1.0 m/s by default) because the up-tilted
FPV camera (20° tilt) biases gate elevation upward: a gate near image-centre is
estimated to sit well above the drone. Without a vertical speed limit, the planner can
command an aggressive climb. The controller's thrust loop (`thrust = HOVER_THRUST + KP_THRUST*(vz_now - vd)`)
tracks the desired vertical velocity. The first live-sim test revealed that `HOVER_THRUST` was the critical tuning knob: 
with `HOVER_THRUST=0.5`, the drone climbed steadily at ~5.5 m/s to 100+ m altitude even though `alt_guard` 
was commanding descent, because 0.5 was well above the racing quad's true hover throttle and `KP_THRUST=0.05` was 
too weak to pull it back (evidence: logs/run_1780521287.jsonl). Values have been adjusted: `HOVER_THRUST` → 0.35 
and `KP_THRUST` → 0.15 to enable decisive altitude control. These are current tuning values; continue bisecting 
`HOVER_THRUST` (lower if it still climbs, raise if it sinks) and `KP_THRUST` as needed for your aircraft.
If the drone climbs past `MAX_ALT_M` (15 m), the planner abandons the gate target and publishes
a controlled descent at `MAX_VSPEED` until back below the ceiling (`target['source'] = 'alt_guard'`,
visible in logs). This is a **client-side fail-safe** — if you hit it frequently, the root cause
is usually an under-tuned `HOVER_THRUST`. Re-calibrate `HOVER_THRUST`
first (see §5a) and ensure it holds altitude level when hovering.

---

## 6. Read the run logs (`logs/run_*.jsonl`)
With `LOGGING=True` (default), each run writes one JSON object per ~100 ms with
`attitude`, `position_ned`, `odometry`, `race`, `vision`, `target`, `last_collision`.
Quick inspections (PowerShell):
```powershell
# how often did we actually see a gate vs fall back to known geometry?
Get-Content logs\run_*.jsonl | ForEach-Object { ($_ | ConvertFrom-Json).target.source } |
    Group-Object | Select-Object Name, Count

# any collisions? (1001 = gate, 1002 = environment)
Get-Content logs\run_*.jsonl | ForEach-Object { ($_ | ConvertFrom-Json).last_collision } |
    Where-Object { $_ } | Select-Object -Last 5

# altitude trace (altitude = -z):
Get-Content logs\run_*.jsonl | ForEach-Object { $json = $_ | ConvertFrom-Json; @{ alt = -$json.position_ned.z; source = $json.target.source } } |
    Where-Object { $_.alt } | Select-Object -Last 20
```
What to look for: mostly `source=vision` near gates (else HSV needs work);
`active_gate_index` climbing steadily (gates being cleared); no repeated
`last_collision` at the same gate (then lower `MAX_SPEED` / `PASS_THROUGH_DIST`);
`watchdog_hover` only on genuine telemetry gaps; **`alt_guard` appearing in source when
altitude approaches or exceeds 15 m** (that's the safety envelope working as intended).
If `alt_guard` is *never* triggered, the vertical control is stable; if it triggers
often, investigate the underlying tracking issue (usually under-tuned `HOVER_THRUST` —
see §5a and `PLAN.md` §8.8).

---

## Quick checklist
- [x] `test_camera_model.py` + `test_pipeline_smoke.py` pass.
- [x] Frames captured to `reference/frames/`.
- [x] HSV calibrated; `_detect_debug.png` shows a clean gate with 4 corners.
- [x] Perception-only (`DRY_RUN=True`) run shows `source=vision` when a gate is in view.
- [ ] Gains tuned with `DRY_RUN=False`; course completed without repeated collisions.
- [ ] `DEBUG_VISION=False` for any compliant timed run (no human interaction allowed).
