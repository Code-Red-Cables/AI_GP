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
   local bounds for it; the combined component is never reported as one gate.
3. `GateTracker` rejects implausible jumps and predicts through at most five
   missed frames. When starting a new track, it also rejects tiny openings,
   extreme side targets, and objects at the bottom of the image. These guards
   prevent post-pass signs and gate-frame fragments from taking control.
4. `GateNavigator` moves through `SEARCH`, `TRACK`,
   `ALIGN_AND_APPROACH`, `COMMIT`, `PASS_THROUGH`, and `RECOVER`.
   Its attitude gains come from `collect_demos.py` and
   `StabilizedController`, but consecutive-gate flight uses a slower approach
   profile: reduced blind/track lean, progressive leveling from 0.8% opening
   area, and a short bounded braking command near 2.5%. This prevents the
   constant demo lean from accelerating through gate one too quickly to align
   gate two. The gate target is held at 62% image height with a 20%-frame
   vertical deadband. Altitude correction remains active even for distant
   gates, keeping the selected opening away from the top and bottom of the
   frame while retaining bottom-rail clearance. Horizontal gate capture favors
   lateral bank over camera yaw,
   so turning toward an off-axis gate cannot masquerade as moving onto its
   flight line.
   Blue-path detection and steering are disabled in the Q2 runtime. Only the
   accepted orange gate and orange next-gate look-ahead affect navigation.
   The Q2 close-gate threshold is calibrated below the measured first-gate
   peak, ensuring the controller commits through the opening before it drops
   out of view.
   When another fully supported gate is already visible beyond a centered,
   nearby active gate, a separately bounded look-ahead term begins lateral
   translation toward the next gate while applying only a small yaw term. It
   cannot override primary-gate alignment.
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
5. `OpenCVGatePlanner` rejects commands older than 350 ms and maps body
   forward/right/down to Q2 NED velocity.
6. `Controller` reuses the demonstration AHRS from
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
view containing only the orange gate geometry that may influence steering.
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
