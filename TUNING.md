# Tuning the dual-gate PnP + IMU flight stack

Runbook for `tools/tune_flight.py`. Work the phases in order — each one
depends on the previous being correct.

## The two simulator builds

| | ATTITUDE | LOCAL_POSITION_NED | ODOMETRY |
|---|---|---|---|
| **OLD sim** (tuning build) | yes | yes | yes |
| **NEW sim** (VQ2 competition build) | no | no | no |

Physics are identical between the two, so **every gain found on the OLD sim
transfers unchanged to the NEW sim**. The OLD build is strictly better for
tuning: its extra messages are ground truth, which turns "the estimate looks
plausible" into a measured error.

You never tell the script which build you are on — it detects ground truth at
runtime and enables or skips the scoring sections. The only command that
*requires* the OLD sim is `--feedback truth`, and it fails cleanly before
arming if the messages are absent.

Ground truth is **scoring only**. `mavlink_rx` parks the sim's ATTITUDE in
`attitude_raw`, which nothing in the control path reads, so truth physically
cannot leak into the loop.

## Which gains actually matter

The gains you would reach for by name in `config.py` are not the ones that fly:

| Setting | Effect |
|---|---|
| `HOVER_THRUST` | **Live.** Trim value, not a loop. Tune first. |
| `KALMAN_KP_ATT`, `KALMAN_KD_ATT` | **Live.** Inner attitude loop in `kalman_planner.py`. |
| `KALMAN_KP_YAW` | **Live.** Image-space yaw. Needs real gates. |
| `KALMAN_MAX_LEAN_DEG`, `KALMAN_MAX_RATE_RAD_S` | **Live.** Authority limits. |
| `KP_ATT`, `KD_ATT`, `KP_ROLL_ATT`, `KP_THRUST_VEL` | **Dead.** Only feed the controller's velocity-fallback branch, which the kalman planner never takes. |

There is **no altitude PID**. The vertical channel is open loop:
`thrust = HOVER_THRUST - 0.030 * norm_y`, clamped. That is why `HOVER_THRUST`
comes first — nothing downstream can compensate for it.

`control/cascaded_pid.py` is not in the live path; only
`test_kalman_dual_gate.py` imports it.

## Before you start

- `models/gate_pose.pt` must be the trained pose model, or `VisionRX` raises at
  startup.
- Sim running, logged in, **and in a race** — the menu alone publishes nothing.
- A gate in view for anything that measures altitude.
- Every run writes a CSV to `logs/tuning/` with the gains in each row, so any
  two runs can be diffed directly.

Both arming modes disarm in a `finally`, abort at 3 m altitude deviation or 35°
lean, and force `AUTO_RESET_ON_CRASH=0` so a run cannot be yanked mid-measurement.

---

## Phase 1 — Validate the estimator · OLD sim

Read-only: never arms, never sends a flight command.

```bash
python tools/tune_flight.py localize --seconds 45
```

**Rotate the drone through roll and pitch while it runs.** Motion is required
or the gyro check reports "not enough motion to judge".

Gate on all three:

- PnP solve rate > 50%
- `EKF attitude err` within a few degrees
- gyro check reports `same sign` on both axes

If an axis reports `INVERTED`, flip the matching `gyro_sign_roll` /
`gyro_sign_pitch` in `ahrs.py`'s `AHRSConfig` and re-run. An attitude error
above 15° is the same class of bug.

**Do not continue past this phase until it is clean.** Every later measurement
is meaningless if the feedback signal has the wrong sign, and no gain can
compensate for it.

## Phase 2 — Trim HOVER_THRUST · OLD sim

The OLD sim gives exact altitude, so this verdict is measured, not inferred.

```bash
python tools/tune_flight.py hover --seconds 12
```

Expect it to report sinking. See [Why 0.24 is suspect](#why-024-is-suspect)
below — jump straight to the prior rather than crawling upward:

```bash
python tools/tune_flight.py hover --hover-thrust 0.28 --seconds 12
```

Follow the suggested 0.005 steps until it reports `holds altitude`. Record the
winner as `HT`.

If `HT` lands near 0.28, also widen the planner's clamp in `kalman_planner.py`:

```python
thrust = float(np.clip(thrust, hover - 0.035, hover + 0.015))
```

`+0.015` of climb authority against `-0.035` of descent is biased toward
sinking. Raise the upper bound and re-run this phase to confirm.

## Phase 3 — Controller ceiling with perfect sensing · OLD sim only

```bash
python tools/tune_flight.py step --axis pitch --amplitude-deg 8 \
    --feedback truth --hover-thrust HT
```

Iterate until rise is under ~0.6 s with under ~25% overshoot:

```bash
python tools/tune_flight.py step --axis pitch --amplitude-deg 8 \
    --feedback truth --hover-thrust HT --kp-att 2.6 --kd-att 0.12
```

This closes the loop on the sim's ATTITUDE, so it measures the controller with
sensing error removed. **These are not shippable gains** — the NEW sim has no
ATTITUDE. The script prints a `*** DIAGNOSTIC ***` banner and labels the CSV
rows `truth` so the two runs cannot be confused later.

Record the best rise time and overshoot you reach. That is the controller's
ceiling.

## Phase 4 — Real feedback signal · OLD sim

Same gains, but the loop now closes on the EKF estimate exactly as it will
race. Stay on the OLD sim so you keep the truth-comparison columns.

```bash
python tools/tune_flight.py step --axis pitch --amplitude-deg 8 \
    --feedback ekf --hover-thrust HT --kp-att X --kd-att Y

python tools/tune_flight.py step --axis roll --amplitude-deg 8 \
    --feedback ekf --hover-thrust HT --kp-att X --kd-att Y
```

Run roll separately — the two axes have never behaved symmetrically in this
sim, which is why `config.py` carries a distinct `KP_ROLL_ATT` history.

Compare against Phase 3:

- **Small gap** → the estimator is good; these gains ship.
- **Large gap**, or a flagged `estimator bias` / overshoot disagreement → the
  remaining error is observer work, not gain work. The report says which. Do
  not try to tune your way out of it: if the estimate is biased, the loop holds
  the *estimate* at the setpoint while the airframe leans by the bias.

## Phase 5 — Full-run rehearsal · OLD sim

Fly the real planner with your gains, forcing EKF-only crash detection so the
client behaves like the NEW sim (`main.py` otherwise prefers
`LOCAL_POSITION_NED` for crash logic, which the NEW sim will not provide):

```bash
CRASH_USE_SIM_ODOMETRY=0 HOVER_THRUST=HT KALMAN_KP_ATT=X KALMAN_KD_ATT=Y \
  RUN_MAX_SECONDS=90 python main.py
```

Watch the `[MODE]` banner and the phase transitions. `kalman_path['phase']`
takes one of `approach`, `commit`, `through`, `coast`, `seek`, `hover` — a
healthy lap cycles `approach` → `commit` → `through`, drops to `coast` for the
blind punch, and uses `seek` to re-acquire. Falling to `hover` means the
planner lost the gate entirely.

Tune `KALMAN_KP_YAW` here — it is an image-space gain, so it needs real gates
and cannot be measured by a step test. Start at the `0.9` default: raise it if
alignment is late, lower it if yaw oscillates.

## Phase 6 — Confirm on the competition build · NEW sim

```bash
python tools/tune_flight.py localize --seconds 45
```

Truth sections will be absent — expected. Check that PnP solve rate is still
above 50% and that `EKF blind drift` (worst-case position travel across a
no-PnP stretch) is small. Then race it:

```bash
HOVER_THRUST=HT KALMAN_KP_ATT=X KALMAN_KD_ATT=Y KALMAN_KP_YAW=Z python main.py
```

Once satisfied, bake `HT` / `X` / `Y` / `Z` into `config.py` as the new
defaults so competition runs do not depend on shell environment.

---

## Why 0.24 is suspect

`HOVER_THRUST` defaults to `0.24`. The deleted pose_debug path, on identical
physics, converged on `0.285`, and its notes record why:

- *"141553: hover 0.26 scraped (21k collisions, thr dipped to 0.245)"*
- *"0.25s×0.263 never got off the floor"*
- `POSE_DEBUG_THRUST_MIN = 0.270`
- in `controller.py`: *"0.24 drops the craft"*

Now consider what the kalman planner can command. It computes
`thrust = HOVER_THRUST - 0.030*ny`, then clamps to
`[HOVER_THRUST-0.035, HOVER_THRUST+0.015]`. At `0.24` that is
**[0.205, 0.255]**. The controller passes this straight through on the kalman
path — no tilt compensation, no re-clamp — with only a takeoff boost to
`TAKEOFF_THRUST=0.30` while `climbed < 0.55 m`.

So once the drone clears half a metre, its thrust ceiling is `0.255`, below the
~`0.28` hover point that the pose_debug work measured. It cannot hold
altitude — only sink. That is consistent with the extensive floor-crash and
auto-reset machinery in `main.py`.

This is inference from the code and those notes, not a hardware measurement.
Phase 2 settles it in twelve seconds.

## Caveats

- The pass/fail thresholds in the reports (0.6 s rise, 25% overshoot,
  0.05 m/s trim) are engineering defaults, not values measured on this
  airframe. Treat the first runs of Phases 3 and 4 as calibrating the
  thresholds as much as the gains.
- `--feedback truth` is a diagnostic. Never ship gains found with it.
- The gyro sign check needs deliberate motion; a stationary run tells you
  nothing about sign conventions.
