# Q2 OpenCV gate navigation

This branch is based directly on `Q2_new` and preserves its MAVLink receiver,
100 Hz body-rate controller, logging, simulator protocol, and complete
`dreamer/` implementation.

## Command ownership

`GATE_NAVIGATION_MODE` selects exactly one flight-command owner at process
startup:

- `opencv` (default): the UDP camera receiver runs the orange gate detector,
  timestamped tracker, and gate navigation state machine. `OpenCVGatePlanner`
  rotates its body-frame commands into Q2's existing
  `{vn, ve, vd, yaw_rate}` planner contract. The unchanged Q2 controller sends
  body rates and thrust to the simulator.
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
   missed frames.
4. `GateNavigator` moves through `SEARCH`, `TRACK`,
   `ALIGN_AND_APPROACH`, `COMMIT`, `PASS_THROUGH`, and `RECOVER`.
   Its vertical setpoint is the body-forward ray, compensating for Q2's
   20-degree upward camera tilt.
5. `OpenCVGatePlanner` rejects commands older than 350 ms and maps body
   forward/right/down to Q2 NED velocity.
6. Q2's original `Controller` converts that velocity to roll/pitch rates,
   yaw rate, and thrust.

The camera tilt and 640x360 pinhole model are centralized in
`camera_model.py`. `gate_estimator.py` reports a size-based range and upgrades
to PnP when reliable opening corners are available.

## Diagnostics and tests

Set `VISION_DEBUG=1` to save periodically annotated frames in
`_vision_debug/`. This is off by default to avoid disk I/O in the race loop.

Use the offline viewer without sending flight commands:

```bash
python tools/offline_gate_viewer.py frames/f_00070.png
```

Run the deterministic detector, tracker, navigation, camera-model, and Q2
planner-contract tests:

```bash
python test_camera_model.py
python -m unittest -v test_opencv_gate_navigation
```
