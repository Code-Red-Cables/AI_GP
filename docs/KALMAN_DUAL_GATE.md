# Dual-gate PnP + EKF + Cascaded PID (Q2_kalman)

`FLIGHT_MODE=kalman` — classical estimator + body-frame planner. **Not the
timed submission.** The timed path is `FLIGHT_MODE=policy`
([`HG_DAGGER.md`](HG_DAGGER.md)).

This document describes one navigation path: dual-gate PnP fixes fused with
the IMU in an EKF, steered by a body-frame planner. Teleop, assist, spline,
`FLIGHT_MODE=race`, and the HG-DAgger student live in other modes.

## Algorithm

1. **YOLO** finds pose-gate instances; `vision/dual_gate_pnp.py` keeps the two closest solved PnP centres in the **body frame** (drone = origin).
2. **EKF** (`ekf/drone_ekf.py`, driven by `ekf_estimator.py`) predicts from IMU (~sample rate, thread paced at 500 Hz) and corrects from those PnP measurements (~30 Hz camera). Gate 2’s last NED fix is retained when it leaves the FOV. It owns `shared_data['attitude']` and `shared_data['position_ned']`; the sim's own ATTITUDE message is published as `attitude_raw` for comparison only.
3. **Planner** (`kalman_planner.py`) is **body-frame only** while a fresh PnP exists:
   - yaw nulls Gate-1 bearing (soft Gate-2 bias only when G1 is already centred)
   - lean toward a chase point on the Gate-1 ray (never a behind-camera waypoint)
   - thrust from Gate-1 body-z around hover
   - lost gate → hover (no EKF spin)
4. **Inner attitude loop** (in `kalman_planner.py`) maps desired lean − measured lean → body rates via `KALMAN_KP_ATT` / `KALMAN_KD_ATT`. The sim has no raw motor throttle API — rates + thrust go out via `SET_ATTITUDE_TARGET`.

The vertical channel is **open loop**: `thrust = HOVER_THRUST - 0.030 * norm_y`,
clamped. There is no altitude PID, so `HOVER_THRUST` must be trimmed first.

`control/cascaded_pid.py` is **not** in the live path — it is an alternative
position→attitude cascade kept for reference and exercised only by
`test_kalman_dual_gate.py`. Likewise the `KP_ATT` / `KP_ROLL_ATT` /
`KP_THRUST_VEL` gains in `config.py` serve only the controller's velocity
fallback branch, which the kalman planner never takes.

## Tuning

See [`TUNING.md`](../TUNING.md) for the full runbook. Order matters:
`localize` (read-only, validate the estimator first) → `hover` (trim
`HOVER_THRUST`) → `step` (tune `KALMAN_KP_ATT` / `KALMAN_KD_ATT`) →
`KALMAN_KP_YAW` in a real run.

Attitude for the rate loop comes from the complementary-filter AHRS in
`ahrs.py` (gyro-dominant, gravity-corrected only near 1 g).

## Dead-reckoning when Gate 2 leaves the FOV

While both gates are visible, the EKF refreshes `gate2_ned` each correction. When Gate 2 exits the camera FOV, the correction step updates only Gate 1 and sets `gate2_fresh=False`, but **keeps the last `gate2_ned`**. IMU prediction continues to move the drone state, so the look-at vector and exit heading remain continuous — the filter acts as a short-term memory buffer until Gate 2 reappears or Gate 1 is passed and a new nearest pair is selected.

## Run

```powershell
$env:FLIGHT_MODE="kalman"
.\winvenv\Scripts\python.exe main.py

# bounded attempt (seconds) instead of running until Ctrl+C
$env:RUN_MAX_SECONDS="60"
.\winvenv\Scripts\python.exe main.py

# perception only: no arming, no flight commands
$env:PERCEPTION_ONLY="1"
.\winvenv\Scripts\python.exe main.py
```

Offline smoke: `.\winvenv\Scripts\python.exe test_kalman_dual_gate.py`.
