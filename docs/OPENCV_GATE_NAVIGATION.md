# Q2 OpenCV gate navigation

This branch is based directly on `Q2_new` and preserves its MAVLink receiver,
100 Hz body-rate control architecture, logging, simulator protocol, and
complete `dreamer/` implementation.

## Command ownership

`GATE_NAVIGATION_MODE` selects exactly one flight-command owner at process
startup:

- `opencv` (default): the UDP camera receiver runs the orange gate detector,
  timestamped tracker, and gate navigation state machine. `OpenCVGatePlanner`
  rotates its body-frame commands into Q2's existing
  `{vn, ve, vd, yaw_rate}` planner contract. The Q2 controller uses the
  demonstrated complementary-filter AHRS and sends rate-capped body rates and
  thrust to the simulator.
- `existing_ai`: `main.py` delegates to the unchanged Dreamer
  `DeployController`. The OpenCV receiver and Q2 controller are not started, so
  their outputs cannot be blended.

Run OpenCV mode:

```bash
python main.py
```

Detector selection defaults to `auto`. It prefers the learned four-corner
pose model at `models/gate_pose.pt`, then the older box/HSV hybrid at
`models/gate_detector.pt`, and finally prints a warning before using the
established HSV detector. Generic COCO weights are never used for live gate
inference.

Require the four-corner pose detector on Windows:

```powershell
$env:GATE_DETECTOR_BACKEND="yolo_pose"
$env:YOLO_POSE_MODEL_PATH="models/gate_pose.pt"
.\.venv\Scripts\python.exe main.py
```

Run the existing Q2 Dreamer policy:

```bash
GATE_NAVIGATION_MODE=existing_ai \
DREAMER_CHECKPOINT=/path/to/deploy_final.pt \
python main.py
```

`DREAMER_CONFIG` and `DREAMER_MAX_SECONDS` are optional environment variables.

## OpenCV data flow

1. `VisionRX` reassembles the newest complete UDP JPEG frame.
2. `YoloPoseGateDetector` runs one inference on that frame. Ultralytics is
   called with class-aware NMS (`agnostic_nms=False`) and a configurable
   default IoU threshold of `0.70`, preserving separately labeled overlapping
   gates when the trained model supports them. Acquisition selects the largest
   valid YOLO gate only when the calibrated orange HSV mask independently
   confirms it. Confirmation requires 12–72% orange coverage inside the YOLO
   box and orange support on at least three of its four border bands; this
   preserves angled/partly occluded gates while rejecting isolated reflections
   and solid orange patches. The surviving proposal must then remain spatially
   consistent for three consecutive inference frames. A one-frame or
   HSV-rejected false positive therefore cannot become the green steering
   target.
   While locked, overlap with the previous target is authoritative and
   center/size similarity breaks ties, preventing rapid switching between
   overlapping gates. Tracker predictions retain identity through a brief
   dropout but command no motion in `TRACK`; movement resumes only after a
   fresh YOLO-plus-HSV measurement.
3. Each pose instance supplies four annotated outer-gate keypoints:
   top-left, top-right, bottom-left, and bottom-right. Dataset inspection shows
   that these points span a median 97.3% of the YOLO box and share virtually
   the same labeled center, so they are not inner-opening corners. The selected
   YOLO box center is therefore authoritative for steering and target scale.
   High-confidence keypoints contribute only the gate's image-plane
   orientation angle and remain visible in the debug overlay. They are not
   published as `GateDetection.corners`, preventing physically invalid
   inner-square PnP. A short complete disappearance uses the previous box
   center; pass confirmation resets the pose instance lock and tracker before
   the next gate is acquired.
   The older `yolo_hybrid` backend remains available for detection-box models.
   It runs orange opening extraction only inside the selected YOLO crop.
4. With no custom model, `auto` mode uses `OrangeGateDetector`, which finds the
   flyable opening rather than the orange
   material centroid. Acquisition selects the largest valid opening in view.
   The deployed HSV mask is calibrated to the illuminated gate face
   (`H=3..17`, `S>=105`, `V>=180`), rejecting the dimmer `H=18..20`
   floor/wall glow visible in the Q2 course. Reconstructed openings must also
   be at least 18 pixels on both axes, excluding small reflection geometry.
   Once tracked, center-and-size hysteresis holds that gate until it disappears
   or becomes implausible, preventing a farther off-axis gate from stealing
   control during normal contour-area fluctuations.
   If overlapping gates form one connected orange component with multiple
   openings, the detector selects one plausible child opening and constructs
   local bounds for it. Every inner-opening detection publishes and draws its
   box from that opening—not from the parent orange contour—so the combined
   component is never reported or displayed as one gate.
   Hough-line fallback segments are spatially clustered before rectangle
   fitting. Within each cluster, opposing rail pairs form separate rectangle
   hypotheses, so even overlapping projected gates cannot become one union
   box. Each hypothesis must satisfy the same single-opening aspect limits;
   an implausible aspect is a hard rejection rather than merely a confidence
   penalty. Candidate interiors are also checked in HSV space: a mostly bright,
   neutral non-orange region is gate lettering, not an opening. When the
   tracked gate reaches a frame edge, a sudden apparent shrink is treated as
   clipping and the last full geometry is predicted instead of resizing onto
   a logo or rail fragment. Line reconstruction publishes its fitted
   four-corner rectangle rather than the irregular hull of every Hough
   endpoint, and the live overlay draws that quadrilateral directly instead
   of an axis-aligned enclosing box.
   Candidate association is identity-locked by normalized center and apparent
   opening size. A visible nested next gate cannot steal the target when its
   confidence fluctuates; an unmatched frame becomes a short tracker
   prediction, and only pass confirmation/reset authorizes acquisition of the
   next gate.
   When a farther gate is visible, its horizontal direction may be retained
   through the current gate pass, but it contributes no steering before the
   active gate has been cleared. The pass-through controller first clears the
   frame without reusing the old gate's final correction, then permits a
   bounded lateral/yaw handoff toward the retained next gate. The old tracker
   prediction is released at pass-through so it cannot block acquisition.
5. `GateTracker` rejects implausible jumps and predicts through at most five
   missed frames. When starting a new track, it also rejects tiny openings,
   extreme side targets, and objects at the bottom of the image. These guards
   prevent post-pass signs and gate-frame fragments from taking control.
6. `GateNavigator` moves through `SEARCH`, `TRACK`,
   `ALIGN_AND_APPROACH`, `COMMIT`, `PASS_THROUGH`, and `RECOVER`.
   Its attitude gains come from `collect_demos.py` and
   `StabilizedController`, but consecutive-gate flight uses a slower approach
   profile: reduced blind/track lean, progressive leveling from 0.8% opening
   area, a short bounded braking command near 2.5%, and slew-limited lateral
   and yaw commands. This prevents the constant demo lean from accelerating
   through gate one too quickly and prevents a reacquired side gate from
   producing a full-bank command step. The gate target is held at 62% image
   height with a 20%-frame
   vertical deadband. Altitude correction remains active even for distant
   gates, keeping the selected opening away from the top and bottom of the
   frame while retaining bottom-rail clearance. Horizontal gate capture favors
   lateral bank over camera yaw,
   so turning toward an off-axis gate cannot masquerade as moving onto its
   flight line.
   Blue-path detection and steering are disabled in the Q2 runtime. Only the
   accepted orange gate affects navigation before the current gate is clear.
   The Q2 close-gate threshold is calibrated below the measured first-gate
   peak, ensuring the controller commits through the opening before it drops
   out of view.
   A fully supported farther gate can supply a post-clearance direction hint,
   but it cannot add lateral or yaw commands during primary-gate alignment.
   The race timer's active-gate increment is used only to confirm a completed
   pass and release the old visual track immediately; it supplies no steering
   geometry.
   If the selected gate approaches any image edge, forward speed is reduced
   and then reversed while centering continues. With no measured gate, blind
   forward flight stops and the vehicle scans toward the last known direction.
   Gate commit requires three stable frames with horizontal error within
   `0.05` normalized image units (about 16 pixels at 640-wide input). Lateral
   centering remains live during commit; drift beyond `0.08` (about 26 pixels)
   aborts the pass attempt back to alignment rather than clipping a side.
7. `OpenCVGatePlanner` rejects commands older than 350 ms and maps body
   forward/right/down to Q2 NED velocity.
8. `Controller` reuses the demonstration AHRS from
   `dreamer/src/dreamer_drone/env/ahrs.py`, applies the demonstrated P+D
   attitude gains and inverted rate axes. Forward pitch keeps the demonstrated
   gain, while lateral requests use a stronger bank mapping for the Q2 gate
   spacing. Gate-navigation yaw is capped at 0.48 rad/s; body pitch and roll
   retain the demonstrated 1.05 rad/s safety cap. A stale-IMU watchdog
   commands neutral hover. Arming applies a one-second `0.31` takeoff boost,
   then returns to the calibrated `0.25` hover baseline.

The profile is grounded in the saved demonstration corpus: 105 inspected
episodes contain a gate pass, with the original demo set passing its first gate
around steps 75-83.

The camera tilt and 640x360 pinhole model are centralized in
`camera_model.py`. `gate_estimator.py` reports a size-based range and upgrades
to PnP when reliable opening corners are available.

## YOLO pose model and free local training

No trained racing-gate weights are committed to Git. The downloaded Roboflow
YOLOv8 keypoint export belongs at
`datasets/AI_GP.v1i.yolov8/`; datasets and weights remain ignored because they
are large/private. The current dataset has four outer-gate points in the order
top-left, top-right, bottom-left, bottom-right. Its `data.yaml` must contain:

```yaml
train: train/images
val: valid/images
kpt_shape: [4, 3]
flip_idx: [1, 0, 3, 2]
names: ['gate']
```

The corrected `flip_idx` is important: horizontal augmentation must swap the
left and right corners. Train locally without a Roboflow API:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe tools\train_gate_pose.py
```

The trainer validates all image/label pairs before starting, fine-tunes
`yolo26s-pose.pt`, and copies the resulting `best.pt` to
`models/gate_pose.pt`. The pretrained file is only the starting point; live
inference accepts only a one-class `gate` pose model with exactly four
keypoints.

Primary runtime tuning variables:

- `YOLO_POSE_MODEL_PATH` (default `models/gate_pose.pt`)
- `YOLO_MODEL_PATH` (default `models/gate_detector.pt`)
- `YOLO_CONFIDENCE_THRESHOLD` (default `0.50`)
- `YOLO_KEYPOINT_CONFIDENCE_THRESHOLD` (default `0.25`)
- `YOLO_NMS_IOU_THRESHOLD` (default `0.70`)
- `YOLO_TARGET_LOCK_SECONDS` (default `0.75`)
- `YOLO_ACQUISITION_CONFIRMATION_FRAMES` (default `3`)
- `YOLO_REQUIRE_HSV_CONFIRMATION` (default `true`)
- `YOLO_HSV_MIN_ORANGE_RATIO` / `YOLO_HSV_MAX_ORANGE_RATIO` (defaults
  `0.12` / `0.72`)
- `YOLO_HSV_SIDE_BAND_FRACTION` (default `0.28`)
- `YOLO_HSV_MIN_SIDE_DENSITY` (default `0.10`)
- `YOLO_HSV_MIN_SUPPORTED_SIDES` (default `3`)
- `YOLO_CROP_PADDING_PX` (default `14`)
- `YOLO_MIN_GATE_AREA_PX` (default `400`)
- `YOLO_PREVIOUS_CENTER_FRAMES` (default `5`)
- `YOLO_ESTIMATED_OPENING_SCALE` (default `0.72`)
- `GATE_HSV_LOWER` / `GATE_HSV_UPPER` (comma-separated HSV triples)
- `GATE_MIN_CONTOUR_AREA` (default `45`)

## Diagnostics and tests

Set `VISION_DEBUG=1` to save periodically annotated frames in
`_vision_debug/`. This is off by default to avoid disk I/O in the race loop.

Raw gate-dataset capture is enabled by default. Every frame with a real
detector measurement is saved unannotated under `frames/`; tracker-predicted
frames are excluded. Set `GATE_FRAME_CAPTURE=0` to disable it, change
`GATE_FRAME_CAPTURE_DIR` to select another directory, or set
`GATE_FRAME_CAPTURE_INTERVAL_S=0.2` to limit capture to five frames per
second. The default interval is zero, which saves every detector hit.

For a safe live Windows view without arming or sending commands:

```powershell
$env:PERCEPTION_ONLY="1"
$env:VISION_DISPLAY="1"
.\.venv\Scripts\python.exe main.py
```

The live window keeps the annotated camera full-size on the left and stacks
two diagnostics on the right: the orange color mask and an accepted-target
view containing only the orange gate geometry that may influence steering.
In pose mode the overlay shows every separate pose instance, confidence, all
four keypoints, the selected opening quadrilateral, calculated center, and
center source (`yolo_box_center` or `previous_frame_fallback`). The pose-mode
orange-mask panel is the same calibrated mask used as the hard YOLO
confirmation layer. Candidate labels show `hsv=<coverage> <sides>/4`: red
boxes failed HSV confirmation, cyan boxes passed HSV but are not selected,
and green is the selected confirmed target. Purple
keypoints show outer-gate orientation for diagnostics only; they do not move
the red steering center away from the green selected box. The accepted-target
panel uses green only for a measured, confirmed steering target; a short
tracker prediction is yellow, and unconfirmed candidates are not drawn in
that panel. In hybrid mode the
overlay shows the equivalent selected box, padded crop, and color-extracted
corners.
Orange pixels in the mask are color candidates, not necessarily
accepted detections. Press `q` or Escape to close only the window, or `Ctrl+C`
to stop the client. Reset the simulator before using perception-only mode so
the vehicle is not left armed from an earlier run.

Use the offline viewer without sending flight commands:

```bash
python tools/offline_gate_viewer.py frames/f_00070.png
```

Run the deterministic detector, tracker, navigation, camera-model,
demo-profile, and Q2 planner-contract tests:

```bash
python -m pytest -q
```
