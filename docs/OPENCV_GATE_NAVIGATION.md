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

Run the existing Q2 Dreamer policy:

```bash
GATE_NAVIGATION_MODE=existing_ai \
DREAMER_CHECKPOINT=/path/to/deploy_final.pt \
python main.py
```

`DREAMER_CONFIG` and `DREAMER_MAX_SECONDS` are optional environment variables.

## OpenCV data flow

1. `VisionRX` reassembles the newest complete UDP JPEG frame.
2. `OrangeGateDetector` finds the flyable opening rather than the orange
   material centroid.
3. `GateTracker` rejects implausible jumps and predicts through at most five
   missed frames. When starting a new track, it also rejects tiny openings,
   extreme side targets, and objects at the bottom of the image. These guards
   prevent post-pass signs and gate-frame fragments from taking control.
4. `BluePathDetector` uses only the lower 60% of the image and requires
   converging left/right cyan lane boundaries. It estimates a smoothed lane
   center and heading; isolated blue objects are not accepted as a path.
5. `GateNavigator` moves through `SEARCH`, `TRACK`,
   `ALIGN_AND_APPROACH`, `COMMIT`, `PASS_THROUGH`, and `RECOVER`.
   Its deployed gains come from `collect_demos.py` and
   `StabilizedController`: constant 1.0 m/s approach, gate target at 58% image
   height, a 30%-frame vertical deadband, and vertical correction only after
   the gate opening reaches the demonstrated close-range size.
   The blue path supplies bounded lateral/yaw assistance, at only 20% strength
   while a usable orange gate is visible, so the flyable gate stays primary.
   Path assistance is suspended during gate commit, then resumes at bounded
   strength 150 ms into pass-through so the drone can anticipate the turn
   toward the next gate without clipping the current gate frame.
   The Q2 close-gate threshold is calibrated below the measured first-gate
   peak, ensuring the controller commits through the opening before it drops
   out of view.
   When another fully supported gate is already visible beyond a centered,
   nearby active gate, a separately bounded look-ahead term begins the next
   turn early. It cannot override primary-gate alignment.
6. `OpenCVGatePlanner` rejects commands older than 350 ms and maps body
   forward/right/down to Q2 NED velocity.
7. `Controller` reuses the demonstration AHRS from
   `dreamer/src/dreamer_drone/env/ahrs.py`, applies the demonstrated P+D
   attitude gains and inverted rate axes, and caps all rates at 1.05 rad/s. A
   stale-IMU watchdog commands neutral hover.

The profile is grounded in the saved demonstration corpus: 105 inspected
episodes contain a gate pass, with the original demo set passing its first gate
around steps 75-83.

The camera tilt and 640x360 pinhole model are centralized in
`camera_model.py`. `gate_estimator.py` reports a size-based range and upgrades
to PnP when reliable opening corners are available.

## Diagnostics and tests

Set `VISION_DEBUG=1` to save periodically annotated frames in
`_vision_debug/`. This is off by default to avoid disk I/O in the race loop.

For a safe live Windows view without arming or sending commands:

```powershell
$env:PERCEPTION_ONLY="1"
$env:VISION_DISPLAY="1"
.\.venv\Scripts\python.exe main.py
```

The live window keeps the annotated camera full-size on the left and stacks
two diagnostics on the right: the orange color mask and an accepted-target
view containing only the gate and blue path geometry that may influence
steering. Orange pixels in the mask are color candidates, not necessarily
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
