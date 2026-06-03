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
[`../notes/VERIFY.md`](../notes/VERIFY.md)), then dump frames:
```powershell
& $PY tools/capture_frames.py 60 notes/frames
```
This binds UDP 5600, reassembles JPEGs, and saves `notes/frames/frame_0000.png …`.
Grab frames where a gate is clearly in view at a few distances/angles.
(`notes/frames/` is git-ignored.)

---

## 2. Calibrate HSV thresholds
The detector masks the gate by colour in HSV (OpenCV ranges: **H 0–179, S 0–255,
V 0–255**). Round 1 is a *high-contrast, desaturated* environment, so the gate is
the saturated/bright object — generally a **high-S, high-V** band.

```powershell
& $PY tools/hsv_tuner.py notes/frames/frame_0000.png
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
& $PY vision/gate_detector.py notes/frames/frame_0000.png   # writes _detect_debug.png
```
Open `_detect_debug.png`: the green bbox should hug the gate, the red dot sit at the
opening center, and TL/TR/BR/BL magenta corners land on the inner square. Good
detections report `confidence` ≳ 0.5 and (for PnP) `corners_px` not None.

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

## 5. Tune guidance gains (`planner.py`)
Start conservative, fly in `DRY_RUN` first (read the `[DRY]` log lines), then enable
flight (`DRY_RUN=False` in `main.py`) and tune against the deterministic course.

| constant | default | meaning | symptom → change |
|----------|---------|---------|------------------|
| `MAX_SPEED` | 4.0 m/s | velocity cap | too slow to be competitive → raise gradually; overshooting gates → lower |
| `KP_POS` | 0.6 | `speed = KP_POS·distance` (before cap) | sluggish approach → raise; oscillates/overshoots → lower |
| `PASS_THROUGH_DIST` | 2.5 m | within this range, command full speed to commit through the gate | stalling *in* the gate → raise; blowing past misaligned → lower |
| `ARRIVE_DIST` | 0.6 m | "at waypoint" threshold | rarely needs changing |
| `CONF_MIN` | 0.40 | min vision confidence to trust a detection | chasing false gates → raise; ignoring real gates → lower |
| `VISION_TIMEOUT_NS` | 300 ms | older vision is "stale" → fall back to known geometry | jittery switching → raise; acting on stale frames → lower |
| `TELEM_TIMEOUT_NS` | 500 ms | no pose for this long → hover (watchdog) | nuisance hovers → raise; flying blind on dropouts → lower |

Workflow: change one constant, re-run, watch the `[DRY]`/`[FLY]` console line and the
JSONL log, compare against the previous run (the sim is deterministic, so runs are
directly comparable). Controller rate (`CONTROL_HZ=60` in `controller.py`) must stay
**< 100 Hz** per spec — don't raise it past ~90.

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
```
What to look for: mostly `source=vision` near gates (else HSV needs work);
`active_gate_index` climbing steadily (gates being cleared); no repeated
`last_collision` at the same gate (then lower `MAX_SPEED` / `PASS_THROUGH_DIST`);
`watchdog_hover` only on genuine telemetry gaps.

---

## Quick checklist
- [ ] `test_camera_model.py` + `test_pipeline_smoke.py` pass.
- [ ] Frames captured to `notes/frames/`.
- [ ] HSV calibrated; `_detect_debug.png` shows a clean gate with 4 corners.
- [ ] Perception-only (`DRY_RUN=True`) run shows `source=vision` when a gate is in view.
- [ ] Gains tuned with `DRY_RUN=False`; course completed without repeated collisions.
- [ ] `DEBUG_VISION=False` for any compliant timed run (no human interaction allowed).
