# Manual / pilot teleop

Human flying and coaching. This is **not** the timed submission — that is
`FLIGHT_MODE=policy` with no gamepad ([`README.md`](README.md),
[`docs/HG_DAGGER.md`](docs/HG_DAGGER.md)).

Focus the **PowerShell console** for keys, not the FlightSim window.

## Fly mode (pure stick — no vision / assist)

```powershell
.\winvenv\Scripts\python.exe tools\tune_flight.py fly
# same thing:
.\winvenv\Scripts\python.exe tools\tune_flight.py pilot --pure
```

Stick-only ANGLE mode: no YOLO/pose, no assist, no AHRS blend, no EKF level
realign, no gate attitude aids. Stick commands a lean; release self-levels.
Arm on stick, **Y** reset.

**Zero attitude** (declare “I am level now”): pad **X**, or keys **Z** / **H**.
Does not change yaw. Hold roughly level, then press. **A** / Space only zeros
stick lean, not the EKF. Each press is stored on the practice attitude tape
as `type: zero_attitude` and is re-applied on
`tools/replay_attitude.py` / `--replay-attitude`.

## Acro mode (rate mode — no angle limits)

```powershell
.\winvenv\Scripts\python.exe tools\tune_flight.py acro
```

Sticks command **body rates** (default ±400°/s roll/pitch/yaw). Center stick
stops rotation but does **not** self-level. Triggers / **R**·**F** are
collective thrust levels, not climb/sink rates: full RT/R = `+ACRO_CLIMB_AUTH`
(default +0.55), full LT/F = `−ACRO_SINK_AUTH` (default −0.55), clamped to
`ACRO_THRUST_MAX` 0.70. Space/A = zero rates only. X/Z = optional emergency
EKF level.

YOLO + dual-gate PnP run **observe-only** in acro by default. They do not
feed the EKF, planner, or body-rate commands. The OpenCV display is off to
save render time; detection and raw capture stay on. `--no-vision` recovers
the old no-detector mode if inference load bothers the stick.

Override rates: `--roll-rate-deg` / `--pitch-rate-deg` / `--yaw-rate-deg`.
Override thrust: `--climb-auth` / `ACRO_SINK_AUTH` / `ACRO_THRUST_MAX`.

Seed laps for the policy come from finished acro (or coach-human) flights
filed into `logs/seed/`. Early quits stay out of that folder.

## Coach (HG-DAgger)

Policy flies; you take over only when it is wrong.

```powershell
.\winvenv\Scripts\python.exe tools\tune_flight.py coach --weights models\policy_seed_17.pt --panel
```

| Key | Action |
|---|---|
| **H** | HUMAN — take the sticks now |
| **T** | POLICY — give it back after recovery |
| **K** | toggle `exclude=1` on the current stretch (do not train on it) |
| **Y** | RESET — new attempt without restarting the client |
| Esc / X | quit |

Do **not** pass `--start-human`. Launch is part of the task.

After a missed gate, go back through **that** gate. The scorekeeper does not
advance until you thread it. Do not skip to N+1 (one-hot vs YOLO mismatch)
and do not stop-and-rebuild the approach from a hover if you can curve
through with leftover speed.

File **finished** 0–17 laps only:

| Kind | Folder |
|---|---|
| Clean human | `logs/seed/` |
| Mixed coach | `logs/coach/` |

The trainer globs every matching file. DNF / early-quit sessions poison the
set if they sit in those folders.

## Pilot mode (manual ↔ assist)

```powershell
.\winvenv\Scripts\python.exe tools\tune_flight.py pilot
```

Plug in a PlayStation / Xbox pad before starting. Console should print
`[PAD] connected: …`. Keyboard still works; stick deflection overrides that
axis. Unbind the controller inside FlightSim so sticks reach the XInput
reader.

Manual pilot stays disarmed until you move a stick or flight key — then it
arms. No early hover / early start.

### Gamepad (Xbox / Mode 2)

| Stick / button | Action |
|---|---|
| **Left** stick | roll / pitch (up=forward) |
| **Right** stick X | yaw |
| **RT** | climb (analog) |
| **LT** | sink (analog) |
| **RB** | extra thrust (collective bump) |
| **A** | level |
| **B** | quit |
| **X** | HUMAN (or zero-attitude in pure fly) |
| **Y** | RESET run (sim pad; arms on next stick/key) |
| **LB** (or D-pad ↑) | AUTO on LOCK |
| **D-pad ↓** | toggle client slow-mo |
| **Start** | KEEP |

Stick feel: `PILOT_PAD_SMOOTH` (default 0.55), `PILOT_PAD_EXPO` (default
0.40), `PILOT_MAX_RATE_DEG` (default 100).

### Keyboard

| Key | Action |
|---|---|
| `W` `S` / `↑` `↓` | pitch forward / back |
| `A` `D` / `←` `→` | roll left / right |
| `Q` `E` | yaw left / right |
| `R` `F` | climb / sink |
| `Space` | level now |
| **`T`** | AUTO on LOCK |
| **`H`** | HUMAN sticks again |
| **`Y`** | RESET run |
| **`O`** | toggle client slow-mo |
| **`K`** | KEEP remembered keys (with `--capture`) |
| `Esc` / `X` | quit |

`W` is pitch only. Yaw is `Q`/`E` (or right-stick X). Live auto
yaw-after-GATE-1 is off (`PILOT_LIVE_POST_G1_YAW=0`).

Human teleop boosts collective while leaned (`PILOT_TILT_COMPENSATE=1`) so
hard forward does not drop altitude. Set `PILOT_TILT_COMPENSATE=0` if you
want to own altitude with LT/RT only.

## Slow-mo (practice only)

There is no MAVLink time-scale API. Slow the **sim** with Cheat Engine
Speedhack or DxWnd, and toggle the **client** to the same factor so
attitude / key tapes stay aligned.

| Control | Effect |
|---|---|
| **`O`** / D-pad ↓ | toggle client slow-mo |
| `--slow-mo` | start with it ON |
| `PILOT_SLOW_MO_SCALE` | factor when ON (default 0.77) |
| `PILOT_SLOW_MO=1` | start ON via env |

When ON, the HUD shows the scale. **Match CE/DxWnd to that same number.**
Estimator reports stay wall-referenced — HUD speed reads low by the
slow-mo factor. Do not re-time IMU arrivals into sim seconds; scaling `dt`
halves reported lean and the craft will somersault.

**Timed / policy runs: leave slow-mo OFF.**

## Why “level” walks, and what holds it

Integrated gyro has no absolute reference, so the EKF’s idea of neutral
slides. The accelerometer cannot fix this on a quadrotor: specific force
equals gravity only in equilibrium. `EKF_ACCEL_TILT_GAIN` therefore ships
at **0**.

Gates hang vertical. PnP on an upright gate recovers gravity. Two slow,
fail-safe aids run off that:

| Knob | Default | What it does |
|---|---|---|
| `EKF_GATE_HORIZON_GAIN` | 0.10 | Proportional pull of roll/pitch onto the gate vertical |
| `EKF_GATE_HORIZON_BIAS_GAIN` | 0.30 | Integral into gyro bias — the term that matters |
| `EKF_GATE_YAW_GAIN` / `_BIAS_GAIN` | 0.06 / 0.20 | Same pair for yaw against a per-gate heading |
| `EKF_GATE_*_MAX_STEP_DEG` | 1.0 | Clamp per correction |
| `EKF_GATE_ATT_MAX_RANGE_M` | 30 | Rotation goes bad with range before the centre does |
| `EKF_GATE_ATT_MAX_REPROJ_PX` | 6 | Screen outliers on solve quality |

Outliers are screened on solve geometry and reprojection, **never** on how
far the measurement sits from the filter’s own belief. The HUD shows
`gh<count> b<x>/<y>` — applied horizon fixes and learned gyro bias. If
`gh` stays at 0, check `EKF_USE_PNP` was not left at `0` from a spline
capture.

On the pilot side the AHRS blend trusts raw IMU when it can vouch for
AHRS (near-1 g, not spinning): `PILOT_LEVEL_AHRS_DRIFT_DEG`,
`_DRIFT_WIDEN`, `_MAX_DISAGREE_DEG`.

## Practice from a gate

The sim cannot teleport mid-course. Practice replays a saved **pad
attitude** tape through gate N, then gives you the sticks for N+1.

While you fly, practice auto-saves under `practice/`:

| File | Meaning |
|---|---|
| `practice/through_gate_N.json` | Best attitude tape through GATE N |
| `practice/index.json` | Attempt index |
| `practice/runs/partial/` | Unfinished attempts (Y-reset or quit) |
| `practice/runs/complete/` | Full-course finishes (cleared gate 17) |

```powershell
.\winvenv\Scripts\python.exe tools\tune_flight.py practice
.\winvenv\Scripts\python.exe tools\tune_flight.py pilot --practice-from-gate 3
```

- **Y** / pad **Y** archives the attempt, then restarts.
- **K** / Start force-saves through-gate checkpoints from the current run.
- Disable auto-save: `--no-practice-save` or `PRACTICE_AUTO_SAVE=0`.

Open-loop tape replay is **not** a reliable full-lap path. Wind, density,
and spawn vary. Use tapes as a supervised prefix, then fly or hand to
assist / policy.

Replay GO is `sim_boot_ms >= race_start_ms`. Wait for a 31000 reset to land
(`race_start_ms` back to −1) before trusting that test, or you arm on the
previous race’s clock. `tools/probe_race_clock.py` and
`tools/replay_attitude.py` implement that handshake.

Tapes that omit `sim_speed` no longer default to 1.0 — `_infer_sim_speed`
recovers it from finish markers and snaps to the nearest speed-hack
setting. Keep CE at the same factor the tape was recorded with.

## Remember-path (exact key presses)

Records which keys you held and for how long, then presses them again.

```powershell
$env:EKF_USE_PNP="1"
.\winvenv\Scripts\python.exe tools\tune_flight.py pilot --capture
```

Fly with **held** keys (not taps). **K** writes `captured_controls.json`
(`type: key_timeline`).

Replay through a gate, then you fly:

```powershell
.\winvenv\Scripts\python.exe tools\tune_flight.py pilot --replay captured_controls.json --keep-until-gate 2
```

Hybrid: keys through GATE 1, then closed-loop ASSIST:

```powershell
.\winvenv\Scripts\python.exe tools\tune_flight.py pilot --replay captured_controls.json --keep-until-gate 1 --assist-after-gate 1
```

`--human-after-gate` defaults to 2. Same via env:
`PILOT_ASSIST_AFTER_GATE`, `PILOT_HUMAN_AFTER_GATE`,
`PILOT_ASSIST_AFTER_GATE_DELAY_S`.

Old `control_timeline` / waypoint JSON files will not load — re-capture.

Spline waypoint capture (EKF position, not keys) is
[`SPLINE_BRANCH.md`](SPLINE_BRANCH.md). That is also not the timed path.

## Assist / classical tune

```powershell
.\winvenv\Scripts\python.exe tools\tune_flight.py manual
.\winvenv\Scripts\python.exe tools\tune_flight.py assist --seconds 30
```

Gains and hover trim: [`TUNING.md`](TUNING.md). Kalman dual-gate:
[`docs/KALMAN_DUAL_GATE.md`](docs/KALMAN_DUAL_GATE.md).
