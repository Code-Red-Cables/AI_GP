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
| `KALMAN_KP_YAW` | **Live.** Image-space yaw. Tune in Phase 4.7. |
| `RATE_SIGN_YAW` | **Live.** Yaw command polarity. Wrong sign drives the gate off-frame — Phase 4.7 catches it. |
| `LEAN_THRUST_BOOST` | **Live.** Extra collective while leaned (`kalman_planner` + crawl). Tune in Phase 4.6. |
| `TAKEOFF_THRUST` | **Live.** Pad climb pulse. Tune in Phase 2.5. |
| `KALMAN_MAX_LEAN_DEG`, `KALMAN_MAX_RATE_RAD_S` | **Live.** Authority limits. Tune in Phase 4.8. |
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

## Default race client — image assist (`main.py`)

`FLIGHT_MODE=assist` (default) flies the **image-chase** planner: YOLO/`nx`/`ny`
→ yaw + forward lean + hover thrust on the same attitude plant as `manual`.
It does **not** steer on EKF position.

```powershell
# Full run (logs → logs/telem_*.csv + logs/events_*.txt)
.\winvenv\Scripts\python.exe main.py

# Short tune harness
.\winvenv\Scripts\python.exe tools\tune_flight.py assist --seconds 30

# Offline
.\winvenv\Scripts\python.exe test_assist_planner.py

# Old dual-gate body-path planner
$env:FLIGHT_MODE="kalman"; .\winvenv\Scripts\python.exe main.py
```

Tune: `ASSIST_LEAN_DEG`, `ASSIST_FWD_FRAC`, `ASSIST_NY_AIM`,
`ASSIST_NY_THRUST_GAIN`, plus existing `HOVER_THRUST` / `KALMAN_KP_ATT` /
`KALMAN_KP_YAW` / `LEAN_THRUST_BOOST`. Watch `[ASSIST]` lines and CSV columns
`path_phase`, `path_nx`, `path_ny`, `path_thrust`.

---

## Phase 1 — Validate the estimator · OLD sim

Default is read-only (never arms). For the gyro sign check you need roll/pitch
motion — use built-in keyboard teleop (arms after the usual 3.5 s early-start
hold):

```bash
# Focus the console after arm — same hold-to-fly keys as `manual`:
python tools/tune_flight.py localize --teleop --seconds 60

# Keys: W/S pitch, A/D roll, Q/E yaw, R/F climb/sink (hold-to-fly),
# Space = level, Esc/X = quit. Release = hover.
```

Without `--teleop` the run stays grounded and the gyro check will say
"not enough motion to judge" unless you tip the craft some other way.

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

## Phase 2.5 — Takeoff thrust pulse · OLD sim

Level craft, pulse `TAKEOFF_THRUST` for a few seconds, then return to `HT`.
Confirms the climb out of the pad is neither a float nor a rocket:

```bash
python tools/tune_flight.py climb --seconds 8 --pulse-s 2.5 \
    --takeoff-thrust 0.30 --hover-thrust HT
```

Pass = climb rate roughly **0.2–1.8 m/s** during the pulse. Bake the winner
as `TAKEOFF_THRUST`.

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

## Phase 4.5 — Lean-hover (tilt-compensated thrust) · OLD sim

Phase 2 only proved level hover. Flight needs the same altitude hold while
tilted. Default is an **8° roll** lean (not forward pitch): pitching off the
pad on the OLD sim triggers an early-start DQ, which resets/teleports the
craft and looks like an altitude crash even though the loop was fine.

```bash
# Uses the already-running sim (default). Pass -Restart only for a fresh
# FlightSim + race-start window during YOLO pre-warm.
powershell -File tools/_run_tune.ps1 -ArgsLine "lean-hover --amplitude-deg 8 --seconds 14 --hover-thrust HT --kp-att 2.2 --kd-att 0.10" -OutName phase45.txt

# or direct (sim already in a race):
python tools/tune_flight.py lean-hover --amplitude-deg 8 --seconds 14 \
    --hover-thrust HT --kp-att X --kd-att Y
# optional: --axis pitch  (only after a clean race start; DQ-prone)
# lean-hover defaults --lean-boost 0 so HT/cos(tilt) is not masked by the
# flight LEAN_THRUST_BOOST=0.008 (that made HT=0.255 still send ~0.2655).
```

Have an active race running (menu alone publishes nothing). Each armed tune
sends sim reset (31000) to the pad before arming unless `--no-sim-reset`.
You still start the race in the FlightSim UI — the client cannot.

Pass if post-settle altitude rate stays within ~±0.05 m/s (truth preferred)
and `race_finish_ns` stays 0. Climb → reduce HT / tilt boost. Sink → raise HT
or check CSV `thrust` (> HT while leaned). If the log shows
`early-start DQ`, ignore the altitude verdict and re-run with `--axis roll`.

## Phase 4.6 — Image crawl · OLD sim

Steer gently on YOLO / dual-gate image centre (`nx`, `ny`) with a small fixed
forward lean — no punch, no takeoff boost. Confirms vision→lean wiring before
the full planner:

```bash
python tools/tune_flight.py crawl --seconds 12 --lean-deg 4 \
    --hover-thrust HT --kp-att X --kd-att Y
```

Pass if a gate is visible for most of the run, commanded pitch is nonzero
whenever a detection exists, and **altitude rate stays within ±0.05 m/s**
(truth preferred; ±0.08 is the hard fail band). Bracket `--lean-boost`
(exports `LEAN_THRUST_BOOST` into the live planner too):

```bash
# prior: 0.000 sank ~0.4 m/s, 0.005 climbed ~0.4 → start near 0.002
python tools/tune_flight.py crawl --seconds 12 --lean-deg 4 \
    --hover-thrust HT --lean-boost 0.002 --kp-att X --kd-att Y
```

Failures with no gate / no pitch are perception; altitude fails are boost trim.

## Phase 4.9 — Drive toward gate · OLD sim

The full planner can sit in `hover` / `missing_gate1_pnp` and look like
**nothing is happening**. This phase bypasses the planner and proves the
airframe can close on a visible gate: hard forward lean + yaw-on-`nx` whenever
YOLO/PnP has a centre.

```bash
python tools/tune_flight.py drive --seconds 16 --lean-deg 10 \
    --hover-thrust HT --kp-att X --kd-att Y
```

Pass if gate range closes ≥1.5 m or bbox area grows ≥400 px. **Ignore
local-N**. If the craft **backs up**, flip `FORWARD_PITCH_SIGN` in
`config.py` (`+1` = positive `des_pitch` is forward; drive_e needed `+1`
after negative pitch tracked but moved away from the gate). Do **not**
start Phase 5 until this passes.

## Phase 4.7 — Yaw align · OLD sim

Hold level and close **only** image `nx` with the same yaw PID the planner
uses (`KALMAN_KP_YAW`). Needs the gate off-centre at arm:

```bash
python tools/tune_flight.py yaw-align --seconds 12 \
    --hover-thrust HT --kp-yaw 0.9
```

The harness yaws open-loop briefly to create an offset, then closes the loop.
Pass if `|nx|` halves within ~4 s or finishes below ~0.15. If nx grows after
close, **flip `RATE_SIGN_YAW`** (Phase 4.7 found `+1.0` recentres; `-1.0`
drove the gate off-frame). Raise `kp-yaw` if recovery is slow; lower it if
yaw oscillates. Phase 5 still re-checks under forward lean.

## Phase 4.8 — Max-lean authority · OLD sim

Step to `KALMAN_MAX_LEAN_DEG` (default 14°) and confirm the rate ceiling can
actually get there:

```bash
python tools/tune_flight.py authority --axis roll --seconds 14 \
    --hover-thrust HT --kp-att X --kd-att Y
```

Pass if steady lean ≥70% of the command and 10–90% rise ≤1.2 s. Fail → raise
`KALMAN_MAX_RATE_RAD_S` (or `KALMAN_KP_ATT`). A high saturation fraction with
a clean rise is fine — it means the rate limit is doing its job.

## Phase 5.0 — Gate acquire · OLD sim

Short armed run of the **real** planner. Within ~2 s of arm the craft must
publish `DUAL_PNP` or planner `source=yolo_fallback` **and** a nonzero
`desired_pitch`. This catches the “stuck in hover with `desired_pitch=0`”
failure before a full lap:

```bash
python tools/tune_flight.py acquire --seconds 8 \
    --hover-thrust HT --kp-att X --kd-att Y
```

Pass = acquire deadline met. Fail = fix vision / planner gate selection first;
do not start Phase 5.

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
