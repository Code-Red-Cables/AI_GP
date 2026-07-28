# Dual-gate PnP + EKF + Cascaded PID (Q2_kalman)

Default racing mode: `GATE_NAVIGATION_MODE=kalman`.

## Algorithm

1. **YOLO** finds pose-gate instances; `vision/dual_gate_pnp.py` keeps the two closest solved PnP centres in the **body frame** (drone = origin).
2. **EKF** (`ekf/drone_ekf.py`) predicts from IMU (~sample rate, thread paced at 500 Hz) and corrects from those PnP measurements (~30 Hz camera). Gate 2’s last NED fix is retained when it leaves the FOV.
3. **Planner** (`kalman_planner.py`) is **body-frame only** while a fresh PnP exists:
   - yaw nulls Gate-1 bearing (soft Gate-2 bias only when G1 is already centred)
   - lean toward a chase point on the Gate-1 ray (never a behind-camera waypoint)
   - thrust from Gate-1 body-z around hover
   - lost gate → hover (no EKF spin)
4. **Cascaded PID** (`control/cascaded_pid.py`) maps body position error → roll/pitch rates. The sim has no raw motor throttle API — rates + thrust go out via `SET_ATTITUDE_TARGET`.

The old IBVS navigator / velocity gate-chaser / bearing latch are **not** used in `GATE_NAVIGATION_MODE=kalman`.

## Dead-reckoning when Gate 2 leaves the FOV

While both gates are visible, the EKF refreshes `gate2_ned` each correction. When Gate 2 exits the camera FOV, the correction step updates only Gate 1 and sets `gate2_fresh=False`, but **keeps the last `gate2_ned`**. IMU prediction continues to move the drone state, so the look-at vector and exit heading remain continuous — the filter acts as a short-term memory buffer until Gate 2 reappears or Gate 1 is passed and a new nearest pair is selected.

## Run

```bash
# default on this branch
python main.py

# legacy IBVS
GATE_NAVIGATION_MODE=opencv python main.py
```

Offline smoke: `python test_kalman_dual_gate.py`.
