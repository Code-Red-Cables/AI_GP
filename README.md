# Electric Fire AI Grand Prix pilot

The live client receives the simulator's MAVLink telemetry and 640×360 JPEG
camera stream, detects red/orange race gates with deterministic OpenCV, and sends
bounded body-rate/thrust commands.

See [docs/OPENCV_GATE_NAVIGATION.md](docs/OPENCV_GATE_NAVIGATION.md) for modes,
architecture, offline tuning, and safety notes. Run the offline checks before
connecting to the simulator:

```bash
python test_camera_model.py
python test_pipeline_smoke.py
python test_opencv_gate_navigation.py
```
