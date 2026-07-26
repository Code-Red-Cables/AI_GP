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
6. Rotated rectangles, child-hole search, polygon approximation, border-density
   measurement, and partial-edge handling.
7. Candidate confidence and nearest-gate selection.

Confidence is 20% apparent size, 20% aspect quality, 10% rectangularity, 5%
convex-hull quality, 25% visible inner opening, 10% plausible border density,
and 10% four-corner quality.
Filled orange rectangles receive a 0.45 penalty. Candidate selection adds a small
size multiplier so the nearest of several nested race gates wins.

`GateDetection` reports pixel and normalized center, width/height, contour area,
rotated angle, confidence, ordered TL/TR/BR/BL corners, approximate distance, and
tracking metadata. Normalized x is -1 left / +1 right; normalized y is -1 top /
+1 bottom.

When four measured opening corners are reliable, `gate_estimator.py` tries
`SOLVEPNP_IPPE_SQUARE`, falls back to ITERATIVE for a degenerate IPPE result, and
accepts the pose only when it is in front of the camera, within 0.2–50 m, and has
at most 6 px RMS reprojection error. Otherwise it falls back to apparent size.

## Tracking and navigation

`GateTracker` applies an EMA to center, size, angle, corners, and distance. It
rejects normalized center jumps over 0.48, predicts through at most five missing
frames with decaying confidence, then resets.

`GateNavigator` owns six states:

- `SEARCH`: slow forward creep and bounded yaw scan, biased toward the last side.
- `ALIGN`: restricted forward speed while yaw/vertical/lateral errors are removed.
- `APPROACH`: alignment/confidence/angle scale forward speed.
- `COMMIT`: a short controlled forward burst with only small corrections.
- `PASS_THROUGH`: keep moving after the close gate disappears from view.
- `RECOVER`: stop forward motion and search locally before returning to `SEARCH`.

State dwell time, approach/commit tolerances, close distance/size, commit/pass
timers, deadbands, gains, command caps, slew rate, and low-pass alpha live in
`NavigationConfig`.

## Running modes

OpenCV only (the default):

```bash
VISION_MODE=opencv python main.py
```

Hybrid with an existing AI policy:

```bash
VISION_MODE=hybrid \
AI_POLICY_FACTORY=my_policy.adapter:create_policy \
python main.py
```

AI only:

```bash
VISION_MODE=ai \
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

Hybrid switches to AI only after eight consecutive low-confidence OpenCV frames,
then uses recovery-frame and cooldown hysteresis before returning. If no provider
is configured, hybrid stays OpenCV; explicit AI mode uses safe hover. The
controller checks action freshness, clips thrust/rates, slew-limits the result,
and never sends OpenCV and AI commands on the same tick.

For safe simulator validation, set `DRY_RUN=True` in `main.py` before running.
The repository currently ships with `DRY_RUN=False`.

## Offline viewer and HSV calibration

The offline viewer accepts an image, directory, or video:

```bash
python tools/offline_gate_viewer.py frames --save-dir debug_frames
python tools/offline_gate_viewer.py flight.mp4 --video-out annotated.mp4
python tools/offline_gate_viewer.py frame.jpg --show
```

Its three panels show raw input, HSV mask, and the annotated output. Overlays
include accepted/rejected contours, bbox, corners, centers/errors, confidence,
distance, angle, state, command, detector time, and FPS. `--save-dir` writes
overlays/masks; `--video-out` records processed video.

For threshold tuning:

```bash
python tools/hsv_tuner.py frame.jpg
```

Then update `DetectorConfig.hsv_ranges`. Important tuning values are:

- `hsv_ranges`, especially saturation/value floors for lighting changes;
- `min_contour_area` and `min_side_px` for distant gates/noise;
- morphology kernel sizes, keeping close small enough to preserve tiny openings;
- `min_confidence` and the tracker confidence/jump/missing-frame limits;
- navigation align/commit tolerances, gains, speed/rate caps, and commit timers;
- `GATE_INNER_M=1.5`, `FX=FY=320`, and the controller hover/rate signs.

## Verification status and limitations

Verified offline:

- camera-frame geometry tests;
- full detector→PnP/size→planner→MAVLink send smoke path;
- 14 focused synthetic tests for segmentation, coordinates, shape rejection,
  partial/rotated/multiple gates, PnP validation, tracking, signs/caps, states,
  dropout behavior, and hybrid hysteresis;
- all Python files compile;
- 12 repository simulator frames process at roughly 1–3 ms mean detector time on
  the development machine; detections occur on the two frames containing visible
  red gate pixels.

Not verified in this environment: a live simulator flight, AI inference (no
checkpoint/provider), real hybrid handoff, and the final rate/thrust gains.
Remaining weaknesses are color sensitivity, ambiguous corner pose at severe
occlusion, size-range bias when the opening is fragmented, and lack of optical
flow. Recommended next steps are a labeled replay set with precision/recall
metrics, live dry-run overlay capture, camera/gate dimension confirmation, and
only then conservative simulator flight tuning.
