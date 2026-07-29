# Manual (teleop) flying

`tools/tune_flight.py manual` is auto-stabilised teleop: **you** command lean,
yaw and climb rate; the client holds attitude and hover trim. It flies the same
plant as the race client, so it is the right place to get a feel for the craft
and to capture waypoints.

Two modes share the identical control code:

| Command | What it adds |
|---|---|
| `manual` | pure flying, plus optional waypoint capture |
| `localize --teleop` | the same stick feel, while recording the full PnP → EKF localization report |

Use `localize --teleop` when you want the gyro-sign check and EKF-vs-truth
scoring — it needs deliberate rotation, and flying is the easiest way to
produce it.

## Run it

```powershell
# Windows (the sim host)
.\winvenv\Scripts\python.exe tools\tune_flight.py manual

# fly while collecting the localization report
.\winvenv\Scripts\python.exe tools\tune_flight.py localize --teleop --seconds 60
```

The sim must be running, logged in, **and in a race** — the menu alone
publishes nothing. `manual` defaults to `--seconds 0`, i.e. runs until you quit.

## Controls

Hold-to-fly. Release an axis and it returns to level / hover.

| Key | Action |
|---|---|
| `W` `S` or `↑` `↓` | pitch forward / back |
| `A` `D` or `←` `→` | roll left / right |
| `Q` `E` | yaw left / right |
| `R` `F` | climb / sink |
| `Space` | force level and zero climb now |
| `M` | mark a waypoint (with `--capture`) |
| `Esc` or `X` | disarm and quit |

**Focus the console window**, not the sim window — keys are read from the
terminal.

### R/F is a climb *rate*, not a thrust offset

Holding `R` commands a fixed climb rate (default 0.6 m/s) and releasing
commands zero, which the loop actively brakes toward. This matters: a raw
thrust offset is an *acceleration* command, so it never settles at a rate and
releasing only stops accelerating — you keep coasting and have to counter-hold
`F`. Measured on a simulated vertical plant, releasing after a 2 s climb:

| | coast after release | time to stop |
|---|---|---|
| thrust offset (old) | 3.97 m | never — still +0.285 m/s at t=8 s |
| rate hold (now) | 0.07 m | 0.26 s |

If 0.6 m/s feels sluggish, raise it: `--climb-rate 1.0`. To compare against the
old behaviour: `--open-loop-thrust --thrust-step 0.022`.

Vertical velocity comes from the EKF, so this works on both sim builds. On the
competition build it degrades gracefully: EKF `vz` picks up a bias when no gate
is in view, and the loop then holds the *estimate* at zero, so the true rate
settles at minus that bias — a slow residual drift rather than a hard stop.
Still better than open loop unless the bias exceeds ~0.17 m/s.

## Constants that matter

**Feel — CLI flags, not persisted:**

| Flag | Default | Effect |
|---|---|---|
| `--lean-deg` | 14° | lean while a key is held. The biggest feel knob. |
| `--yaw-rate-deg` | 40°/s | yaw rate on `Q`/`E` |
| `--climb-rate` | 0.6 m/s | climb/sink rate on `R`/`F` |
| `--climb-auth` | 0.08 | thrust the rate loop may add |
| `--climb-kp` / `--climb-ki` | `KP_THRUST_VEL` / `KI_THRUST_VEL` | vertical loop gains |

**Plant — shared with the race client, so tuning here transfers:**

| Constant | Current | Symptom if wrong |
|---|---|---|
| `HOVER_THRUST` | 0.264 | drifts up or down with no keys held |
| `KALMAN_KP_ATT` | 2.2 | sluggish response to a lean command |
| `KALMAN_KD_ATT` | 0.10 | oscillates / overshoots the lean |
| `KALMAN_MAX_RATE_RAD_S` | 0.9 | leans feel rate-limited |
| `LEAN_THRUST_BOOST` | 0.0 | sinks while leaning (extra thrust once lean > 0.5°) |

Override per-run with `--hover-thrust`, `--kp-att`, `--kd-att`, `--max-rate`,
`--lean-boost`.

**Signs — get these wrong and the controls are inverted:**

- `FORWARD_PITCH_SIGN` (1.0) — whether `W` pitches forward or backward
- `RATE_SIGN_YAW` (1.0) — whether `Q` yaws left

Both were settled against VQ1 ground truth; don't change them casually.

## Pre-flight sequence

Each run, in order:

1. **Pad reset** — MAVLink command 31000, then waits `SIM_RESET_SETTLE_S`.
   Skip with `--no-sim-reset` to leave the craft where it is.
2. **Optional vision gate** — `--wait-pad` blocks (45 s max) until dual-gate
   PnP sees gate 1, so you start facing the course.
3. **Early-start hold** — 3.5 s before arming, so the old sim does not flag an
   early start. Change with `--early-start-hold-s`.
4. **Arm**, then the control loop starts.

On exit it commands level-and-hover for 8 cycles *before* disarming, so the
craft does not drop while still armed.

## Safety

> `manual` has **no automatic aborts.** Unlike `hover` and `step`, which disarm
> at 3 m altitude deviation or 35° lean, manual assumes the pilot is the safety
> system. `Space` levels and zeroes climb immediately; `Esc` disarms.

Thrust is clamped to `[0.18, 0.36]` — deliberately wider than the race planner
so manual has real authority.

## Waypoint capture

Used by the spline branch. Fly the course and press `M` at each point you want
on the path:

```powershell
$env:EKF_USE_PNP="0"
.\winvenv\Scripts\python.exe tools\tune_flight.py manual --capture
```

Marks print as you make them; waypoints are written on exit to
`SPLINE_CAPTURE_PATH` (default `captured_waypoints.json`). Two is the minimum,
and a mark is rejected if the EKF has no pose yet.

Mark generously through corners — the path is a centripetal Catmull-Rom spline
through your marks, so corner shape follows from how densely you mark.

The capture file records the `EKF_USE_PNP` setting, because **capture and
replay must match**. See `SPLINE_BRANCH.md` for why and for the replay
procedure.

## Reading the HUD

```
    t   climb  vz_up  roll  pitch   yaw_r   thr   des_r  des_p   g1_rng  n
```

- `climb` metres above the arm point, `vz_up` measured climb rate (m/s)
- `roll` / `pitch` actual attitude, `des_r` / `des_p` what you commanded
- `thr` collective actually sent
- `g1_rng` PnP range to gate 1, `n` gates solved this frame

Every run writes a CSV to `logs/tuning/` including `climb_meas_mps` and
`climb_src`, so the vertical loop is reviewable offline.

## Known quirks

- **Windows only.** Keys are read via `msvcrt`. On macOS or Linux the mode arms
  and flies but ignores every key — it becomes a plain hover.
- **Hold is inferred from autorepeat.** `msvcrt` has no key-up event, so an
  axis is treated as released after 180 ms without a repeat. If your keyboard
  repeat rate is slower than that, held inputs will stutter back to level.
