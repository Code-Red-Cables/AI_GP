# OpenCV gate detection and navigation

## Existing interface and control flow

`main.py` creates a locked `shared_data` blackboard and calls
`setup.setup_components()`. The main thread runs `Controller.update()` at 60 Hz.
Background threads receive MAVLink telemetry, receive/reassemble camera JPEG
chunks, perform time synchronization, and log state.

The camera receiver binds UDP 5600, decodes complete JPEGs with
`cv2.imdecode(..., cv2.IMREAD_COLOR)`, and therefore supplies **BGR** images.
The simulator specification documents 640×360 at 30 Hz; physics runs at 120 Hz.
Old incomplete frames are pruned so perception always works toward the newest
causal frame.

The flight interface is `SET_ATTITUDE_TARGET` in body-rate mode plus collective
thrust. The simulator's empirically measured raw rate signs are centralized in
`controller.py`: roll `+1`, pitch `-1`, yaw `-1`. The planner/controller boundary
remains NED velocity (`+z` down) plus absolute yaw. The new image controller uses
body `+forward`, `+right`, `+down`, and `+yaw` toward image right; `Planner`
performs the one frame conversion.

No neural checkpoint or model loader is present on the checked-out `main`
branch. Dreamer source exists on the repository's `Q2_CV`/`Q2_new` branches but
no trained weights are committed. The AI mode is therefore a lazy provider
boundary that preserves learned-policy use without adding PyTorch to OpenCV-only
runs.

## Detection

`OrangeGateDetector.detect(frame)` performs:

1. Explicit input conversion (BGR by default; RGB/BGRA/RGBA supported).
2. Optional downscale, 3×3 Gaussian blur, and BGR→HSV.
3. Two configurable hue ranges for the red/orange wrap at HSV hue 0/179.
4. Configurable opening/closing morphology.
5. `RETR_TREE` contour extraction and external-contour candidate generation.
6. Child-opening contours, four-corner diagonal intersections, rotated
   rectangles, Hough-line reconstruction, and edge-clipped partial-gate
   reconstruction.
7. Geometric, photometric, and timestamped tracker-consistency scoring.

Confidence is 8% apparent size, 11% aspect quality, 8% rectangularity, 5%
convex-hull quality, 22% method/opening quality, 13% orange-border support, 10%
opening-center clearance, 8% corner quality, and 15% temporal consistency.
Single posts, filled objects, and unsupported centers receive explicit
penalties. Final selection is 68% confidence, 23% temporal consistency, 6%
image-center prior, and 3% opening-method quality, so the largest orange object
does not automatically win.

`GateDetection` reports pixel and normalized center, width/height, contour area,
rotated angle, confidence, ordered TL/TR/BR/BL corners, approximate distance, and
tracking metadata. Normalized x is -1 left / +1 right; normalized y is -1 top /
+1 bottom.

When four measured opening corners are reliable, `gate_estimator.py` tries
`SOLVEPNP_IPPE_SQUARE`, falls back to ITERATIVE for a degenerate IPPE result, and
accepts the pose only when it is in front of the camera, within 0.2–50 m, and has
at most 6 px RMS reprojection error. Otherwise it falls back to apparent size.

## Tracking and navigation

`GateTracker` uses timestamped alpha-beta center/velocity filtering plus smoothed
opening size and angle. It rejects configurable center and size jumps, supplies
a non-mutating predicted hint for next-frame candidate scoring, predicts through
at most five missing frames with decaying confidence, and then resets.

`GateNavigator` owns six states:

- `SEARCH`: slow forward creep and bounded yaw scan, biased toward the last side.
- `TRACK`: confirm geometry and motion for several measured frames at low speed.
- `ALIGN_AND_APPROACH`: remove yaw/vertical/lateral errors while alignment,
  confidence, edge distance, center speed, and worsening error scale forward speed.
- `COMMIT`: a short controlled forward burst with only small corrections.
- `PASS_THROUGH`: keep moving after the close gate disappears from view.
- `RECOVER`: stop forward motion and search locally before returning to `SEARCH`.

State dwell time, approach/commit tolerances, close distance/size, commit/pass
timers, deadbands, gains, command caps, slew rate, and low-pass alpha live in
`NavigationConfig`.

## Running modes

OpenCV only (the default):

```bash
GATE_NAVIGATION_MODE=opencv python main.py
```

Existing AI policy only:

```bash
GATE_NAVIGATION_MODE=existing_ai \
AI_POLICY_FACTORY=my_policy.adapter:create_policy \
python main.py
```

The factory receives `shared_data` and returns either a callable or an object
with `predict(frame_bgr, context)`. It must return physical values:

```python
{
    "thrust": 0.27,
    "roll_rate": 0.0,
    "pitch_rate": 0.0,
    "yaw_rate": 0.0,
}
```

The two modes are exclusive and their outputs are never blended. If
`existing_ai` is selected without a provider, the router chooses safe hover.
The controller checks action freshness, clips thrust/rates, and slew-limits the
AI result. `VISION_MODE` remains only as a launch-script compatibility fallback;
new configurations should use `GATE_NAVIGATION_MODE`.

For safe simulator validation, set `DRY_RUN=True` in `main.py` before running.
The repository currently ships with `DRY_RUN=False`.

## Offline viewer and HSV calibration

The offline viewer accepts an image, directory, or video:

```bash
python tools/offline_gate_viewer.py frames --save-dir debug_frames
python tools/offline_gate_viewer.py flight.mp4 --video-out annotated.mp4
python tools/offline_gate_viewer.py frames --show --step
python tools/offline_gate_viewer.py frame.jpg --tune-hsv --show-hsv
```

The four panels show original BGR, raw orange mask, cleaned mask, and annotated
output; `--show-hsv` adds H/S/V channel panels. Overlays include
accepted/rejected candidates, opening contour, corners, raw/tracked/predicted
centers, errors, confidence, method, opening ratio, state, commands, detector
time, total time, and effective FPS. `--step` uses space/n/right for next,
p/left for previous, r to reprocess, and q/Escape to quit. `--tune-hsv` exposes
both HSV ranges as trackbars and prints the final values. `--save-dir` writes
the overlay plus raw and cleaned masks; `--video-out` records the panels.

Important tuning values are:

- `hsv_ranges`, especially saturation/value floors for lighting changes;
- `min_contour_area`, `min_opening_area`, and minimum width/height for distant
  gates versus noise;
- morphology kernel sizes, keeping close small enough to preserve tiny openings;
- `min_confidence` and the tracker confidence/jump/missing-frame limits;
- navigation align/commit tolerances, gains, speed/rate caps, and commit timers;
- `GATE_INNER_M=1.5`, `FX=FY=320`, and the controller hover/rate signs.

## Verification status and limitations

Verified offline:

- camera-frame geometry tests;
- full detector→PnP/size→planner→MAVLink send smoke path;
- 31 focused synthetic tests for segmentation, opening-center coordinates,
  perspective/rotation/blur/lighting/clipping, broken sides, false-object
  rejection, temporal selection, PnP validation, timestamped tracking, jump
  rejection, signs/caps, all navigation transitions, dropout behavior, and
  exclusive mode selection, plus one repository-frame regression test;
- all Python files compile;
- a warmed 240-frame replay averaged 1.182 ms detector and 0.012 ms tracker
  (2.086 ms detector p95); a cold single pass over all 12 repository frames
  averaged 5.17 ms detector and 5.28 ms total;
- `f_13300.jpg` tiny outlined floor markers are rejected, while the real gate
  opening in `f_14400.jpg` is detected by `inner_contour` at confidence 0.708.

Not verified in this environment: a live simulator flight, AI inference (no
checkpoint/provider), AI comparison, and the final rate/thrust gains.
Remaining weaknesses are color sensitivity, ambiguous corner pose at severe
occlusion, size-range bias when the opening is fragmented, and lack of optical
flow. Recommended next steps are a labeled replay set with precision/recall
metrics, live dry-run overlay capture, camera/gate dimension confirmation, and
only then conservative simulator flight tuning.
