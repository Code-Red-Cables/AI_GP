# Manual / pilot teleop

Focus the **PowerShell console** for keys — not the FlightSim window.

## Fly mode (pure stick — no vision / assist)

```powershell
.\winvenv\Scripts\python.exe tools\tune_flight.py fly
# same thing:
.\winvenv\Scripts\python.exe tools\tune_flight.py pilot --pure
```

Stick-only ANGLE mode: no YOLO/pose, no assist (T), no AHRS blend, no
EKF level realign, no gate attitude aids. Stick commands a lean; release
self-levels. Arm on stick, Y reset, practice run archives.

**Zero attitude** (clears EKF roll/pitch drift — declare “I am level now”):
pad **X**, or keys **Z** / **H**. Does not change yaw. Hold roughly level,
then press. (**A** / Space only zeros stick lean, not the EKF.)
Each press is stored on the practice attitude tape as an `events` entry
(`type: zero_attitude`) and is re-applied on
`tools/replay_attitude.py` / `--replay-attitude`.

## Acro mode (rate mode — no angle limits)

```powershell
.\winvenv\Scripts\python.exe tools\tune_flight.py acro
```

Like a normal FPV acro drone: sticks command **body rates** (default
±400°/s roll/pitch/yaw). Center stick stops rotation but does **not**
self-level — attitude stays wherever you left it. No lean/angle caps.
Triggers / R·F are **collective thrust levels** (not climb/sink rates):
full RT/R = `+ACRO_CLIMB_AUTH` (default +0.55), full LT/F =
`−ACRO_SINK_AUTH` (default −0.55), clamped to `ACRO_THRUST_MAX` 0.70.
Same engage-on-stick and Y reset as fly. Space/A = zero rates only
(craft keeps its attitude). X/Z = optional emergency EKF level if you
want to declare “I am upright now.”

YOLO + dual-gate PnP are now **on by default in acro as observe-only
recorders**. They do not feed the EKF, planner, attitude loop, or body-rate
commands. The OpenCV display is disabled in acro to avoid rendering overhead;
detection and raw capture remain active. Every attitude-tape command sample
carries synchronized gate
center/corners/range, PnP body pose, race clock, attitude, gyro/acceleration,
post-controller wire commands, and an epoch timestamp linking the camera and
50 Hz telemetry logs. Use `--no-vision` only to recover
the old no-detector mode if inference load affects the manual flight.
Race-status receive timestamps are also stored. Promotion can therefore
extrapolate the 4 Hz simulator clock to the first command's `perf_counter`
timestamp instead of estimating launch phase from the delayed G17 marker.
Acro also saves sparse raw JPEGs at the existing three-per-second cap for the
whole run, including frames where the detector missed. The tape records the
absolute capture directory and frame IDs so a better detector can reprocess
the same visual flight later without another manual run.

Override rates: `--roll-rate-deg` / `--pitch-rate-deg` / `--yaw-rate-deg`.
Override thrust: `--climb-auth` / `ACRO_SINK_AUTH` / `ACRO_THRUST_MAX`.

### Replay your last acro run (hardcoded)

Plays `practice/runs/HARDCODED_REPLAY_CLEAN.json`
(`complete_20260803_014712_g17_94.641s` — G17, **18.979 s, zero
collisions**) as exact body rates + thrust. The promoted tape preserves the
manual pilot's inferred **50.4 ms post-GO engage phase**.

> The two older reference runs are archived read-only and checksummed under
> `practice/best/`, each with a README. Verify either with
> `sha256sum -c SHA256SUMS.txt` in its directory.
>
> | Archive | Race time | Collisions | Role |
> |---|---|---|---|
> | `practice/runs/complete/complete_20260803_014712_g17_94.641s.json` | **18.979 s** | **0** | **replay default** |
> | `best_20260803_001438_22.60s_clean/` | 22.596 s | **0** | clean fallback |
> | `best_20260802_193228_20.18s/` | 20.180 s | 2 | floor-contact history |
>
> The faster run takes a ~10 g floor strike between gates 2 and 3, and gates 3
> through 17 all inherit the state it leaves that 40 ms of contact with.
> Contact amplifies upstream error instead of damping it, so it never replayed
> past two gates. It is retained only for comparison.
> Use it explicitly with
> `--tape practice/runs/HARDCODED_REPLAY.json`.

Tapes that omit `sim_speed` no longer default to 1.0 and run the playhead 5×
slow — `_infer_sim_speed` recovers it from `race_finish_ns` divided by the last
gate's tape time and snaps to the nearest round speed-hack setting.

The current reference was recorded with **Cheat Engine at 0.2×**, so tape time
is ~5× race time (G17 @ 94.641 s tape = 18.979 s race, after accounting for
the 50.4 ms engage phase). Replay measures the live simulator rate during the
countdown and advances the tape appropriately. Keep CE at 0.2× to reproduce
the same command cadence and physics conditions as the manual run.

```powershell
# CE at 0.2x, matching the recording
.\winvenv\Scripts\python.exe tools\replay_attitude.py
```

#### Early-start DQ

The tape's first sample is already a full forward tip, so arming one tick
before GO is an instant disqualification. Three DQs came from inferring GO
from a wall-clock hold while the countdown ran on sim time.

`tools/probe_race_clock.py` settled it. The sim publishes the answer:

| Field | Behaviour |
|---|---|
| `sim_boot_ms` | restarts at 1 on a 31000 reset, advances in **sim** time |
| `race_start_ms` | `-1` while pending, then the **future** `sim_boot_ms` of GO |

So GO is exactly `sim_boot_ms >= race_start_ms`. On this course
`race_start_ms` is 3298, i.e. a 3.3 sim-second countdown — 16.5 s of wall time
at 0.2×.

That test cannot be applied naively. The 31000 reset takes ~2.6 s of wall time
to land, and until it does the sim still reports the *previous* race, where
`sim_boot_ms` is long past `race_start_ms` — so the test passes instantly and
arms with no countdown. Replay walks three phases: wait for the reset to land
(`race_start_ms` back to -1), wait for a countdown to be scheduled and still
*pending* (`sim_boot_ms < race_start_ms`), then wait for GO.

Two things follow, and both remove a knob:

* CE speed is **measured** from the slope of `sim_boot_ms` instead of assumed,
  so `--match-record-speed` is no longer needed. `--ce-speed` only overrides.
* The tape playhead is anchored to the sim race clock at arm, so replay is
  indifferent to CE speed, frame rate and UDP jitter.

### The playhead is hybrid, and it has to be

Driving the playhead *directly* off the race clock looks obviously right and is
badly wrong: **`race_status` only arrives at 4 Hz.** Between messages the sim
time is a constant, so the playhead froze for 250 ms at a time — 1.25 seconds of
tape at 0.2× — and then jumped. Commands went out at **4.1 Hz against a 48.7 Hz
tape**, roughly one of every twelve samples, and the drone flew into the terrain
at tape 6.8 s every time. Every run between the sim-clock change and this note
is invalid for that reason alone.

So the playhead free-runs on the wall clock at the measured CE speed, and each
4 Hz race_status only *trims the rate* (±15%, gain 0.25). Output is smooth and
monotonic at 48.8 Hz while staying locked to sim time within ~150 ms.

Replay now prints an achieved command rate at exit and shouts `*** STARVED ***`
below 60% of the tape rate. If you see that, the flight means nothing — fix the
clock before reading anything into the trajectory.

### Measure CE speed during the countdown, never at startup

The wall rate is `ce_speed / record_sim_speed`, so CE speed has to be right or
the tape plays at the wrong speed outright. Sampling it for 2 s at startup is
not good enough: on a session that raced at **0.200x** it read **0.85x**,
because the measurement happens before the reset while the sim is not yet in
race time. The playhead then ran **4.25 tape-s per wall-s** and dumped the whole
62 s tape in 14.6 s — the drone just climbed and never flew the course.

So the speed is estimated *continuously* by `_SimRate`, a least-squares slope
over a 3 s sliding window of the race clock. The countdown gives ~16 wall
seconds of settling before arm, and the estimate recovers 0.2000x from a 0.85
seed in about 1 second. Guards, all of which earned their place:

* the window clears on a backward jump, so a reset can't corrupt the slope;
* estimates outside 0.02–4.0x are discarded as glitches;
* the live rate is clamped to 0.5–1.5× the countdown baseline, so even a bad
  window can only nudge the playhead rather than rip through the tape.

Replay prints `playing at CE Nx ... → N tape-s per wall-s` at arm. **Check that
line.** For a 0.2x tape at 0.2x it must read `1.00`.

Running a 0.2x tape at full speed needs 5× the send rate — 243 Hz, above the
poll ceiling — so replay refuses quietly no longer: it prints
`*** CANNOT KEEP UP ***` and tells you to set the sim back to 0.2x.

### On Windows, `time.monotonic()` is a 15.6 ms clock

`time.perf_counter()`, not `time.monotonic()`, is the clock for anything paced
finer than a frame. Measured on this machine's Python 3.12:

| Clock | Implementation | Resolution |
|---|---|---|
| `time.monotonic()` | `GetTickCount64()` | **15.625 ms** |
| `time.perf_counter()` | `QueryPerformanceCounter()` | 0.0001 ms |

The replay loop asked for 8.2 ms sleeps and got a flat **31.0 ms** — two clock
ticks — so it could not place a sample nearer than 31 ms to its recorded time.
Note the sleep was never the problem: with `perf_counter` the same loop hits
122 Hz with no other change. `timeBeginPeriod(1)` is set anyway to keep sleeps honest under
load, but the clock was the bug. `tools/replay_attitude.py` is on
`perf_counter` throughout and paces to an absolute deadline, since sleeping a
fixed period *after* the work adds the work time to every cycle.

The general pilot loop still uses `time.monotonic()` for UI/control state, but
acro attitude-tape samples, gate markers, and zero-attitude events now use an
isolated `perf_counter()` time base. That removes the 15.6 ms timestamp
quantization without changing planner/component clock domains.

### The pad tape is not decimated, and 16 Hz is the real cadence

The current reference contains **1,551 pad samples over 95.041 wall seconds**,
with a median 62 ms interval — four 15.625 ms clock ticks — so the recording
loop itself was quantised to ~16 Hz. The current pad tape carries 1,272 command
changes; dense telemetry merely observes each held value several times:

| Tape | Samples | Rate | Distinct commands |
|---|---|---|---|
| `HARDCODED_REPLAY_CLEAN.json` (default) | 1551 | 16.3 Hz | 1272 |
| telemetry observations | ~4625 | ~48.7 Hz | same command staircase |

The 48.7 Hz tape is not higher fidelity — it is the telemetry logger sampling a
16 Hz staircase, holding each command for exactly 3 samples (938 of its runs
are 3 long). Switching the default to it made replay send **three times the
messages the pilot sent**, and the run dropped from two gates to one. Don't
mistake a higher sample count for more information; check how often the value
actually *changes*.

Comparing tapes with linear interpolation makes the low-rate one look lossy
(0.44 rad/s RMS "missing"). That is an artifact of interpolating a staircase.
Compare with zero-order hold.

**Zero-order hold is the correct semantics at any tape rate** — an acro stick
holds its value until the pilot moves it. Exact mode used to be gated behind
`clock.hz() > 30`, which excluded the pad tape and interpolated it instead,
inventing intermediate stick positions that were never commanded. That gate is
gone; any acro tape now plays verbatim.

### The rate trim was a mistake — leave it at 1.0

`--rate-scale-roll` / `--rate-scale-pitch` exist and work, but **do not use
them.** Keep them at the 1.0 default. This is recorded so nobody repeats it.

The reasoning that produced them looked sound. With the clock and cadence
fixed, replay tracks the reference rotation at correlation **+0.98** with only
4 ms of lag, and regressing achieved-replay against achieved-reference over
tape 0.3–12 gave a gain below 1 on two consecutive runs — 0.960/0.936 on roll,
0.924/0.917 on pitch. Two runs agreeing within 2% reads as a systematic plant
deficit, so 1.05/1.09 should null it.

Flying it produced the worst run of the session: **zero gates**, against two and
one for the untrimmed runs. Early pitch rate went from below reference to above
it (at tape 0.5, reference 9.79 rad/s, untrimmed 9.34, trimmed 11.93).

The error was the measurement, not the arithmetic. **A ratio between two
trajectories that have already diverged is not a plant gain.** It mixes "the
plant under-responds to this command" with "the drone is somewhere else, doing
something else". Boosting the command to correct the second kind of error just
diverges faster. A real plant gain has to be measured *within* a single run, as
achieved against commanded — and done that way the two agree far more closely
(reference −2.52/−2.57, replay −2.47/−2.34), leaving much less to correct than
the cross-trajectory ratio implied.

Also worth knowing: a replay CSV that is still being written will give
misleading numbers. Check that the run has finished before analysing it.

### Run-to-run variance comes from the anchor, quantised at 4 Hz

The two runs started the tape **4 ms and 61 ms** after GO respectively. That
spread is one `race_status` tick (the clock publishes at 4 Hz = 50 ms of sim
time at 0.2×), and 57 ms of sim time is 0.29 s of tape. It is the largest
uncontrolled difference between otherwise identical runs.

Curiously the *later* anchor tracked gate 1 better (1.288 s vs the reference
1.291 s, against 1.258 s for the early one), because the reference itself
started ~60 ms after GO. But the later-anchored run died sooner, so anchor
alignment is not what decides survival — lateral error at the gate is.

### Read the collision stream

`logs/events_*.txt` carries `COLLISION <Environment|Gate> impulse=N`, which is
the fastest way to score a replay against the reference:

| Run | First contact | Events | Total impulse |
|---|---|---|---|
| Winning manual run | tape 14.2 s (Gate) | 2 | 11.2 |
| Starved replay | tape 6.8 s (Environment) | 125 | 31.2 |

To watch a countdown without any DQ risk (never arms):

```powershell
.\winvenv\Scripts\python.exe tools\probe_race_clock.py
```

### Rate-tracking replay (closed loop on the gyro)

Open-loop replay diverges by gate 2: the tape holds rate *commands*, and rate
error integrates twice into position error. `build_tracking_tape.py` rebuilds
the tape straight from the telemetry log — 48.7 Hz instead of 16 Hz, the exact
wire commands, plus the body rates the winning run actually *achieved*.

```powershell
# Build once (writes practice/runs/HARDCODED_TRACKING.json)
.\winvenv\Scripts\python.exe tools\build_tracking_tape.py

# Fly it: feedforward commands + feedback onto the recorded gyro
.\winvenv\Scripts\python.exe tools\replay_attitude.py `
  practice\runs\HARDCODED_TRACKING.json --match-record-speed
```

The plant fit is stored in **pre-rate-sign** command units, matching the
tape's `des_*` and the value replay corrects. It is fitted on `cmd_*`, which
are post-sign wire values, so the slope is divided by `RATE_SIGN_*`. Skipping
that inverts every correction on `RATE_SIGN_PITCH = -1`: measured on
`replay_att_20260802_205533`, 97% of pitch corrections pushed the wrong way
while roll (sign +1) was 96% correct.

**Tracking is off by default (`--track-kp 0`).** Head-to-head on identical
tapes with a correct launch:

| | roll corr | roll RMSE | gate reached |
|---|---|---|---|
| `--track-kp 0.5` | +0.20 | 6.63 rad/s | 0 |
| `--track-kp 0` (open loop) | +0.79 | 1.73 rad/s | **1** |

Closing a proportional loop through the 123 ms actuation lag rings instead of
correcting, even with the sign bug fixed. Raise `--track-kp` only with a log
to prove it helps. Yaw tracking is off as well (`--track-yaw`) since yaw only
correlates −0.33.

Do **not** track the recorded *attitude* — the EKF attitude on that run
implies angular speeds 5× the gyro and passes through gimbal lock at 89°
pitch. The gyro is the trustworthy reference.

Dense tapes replay **verbatim**: one send per recorded sample, at its recorded
sim time, with no interpolation (`--interpolate` restores lerping). Acro
sticks step by up to a full 3.14 rad/s between adjacent samples, so lerping
synthesised rates that were never commanded — 0.147 rad/s RMS against a
0.26 rad/s signal. Verbatim mode also reproduces the original's ~48.7 Hz
cadence instead of resampling it to the loop rate.

The replay CSV logs `ref_gr/ref_gp` against `gx/gy`, which is the error signal
an iterative-learning pass would fold back into the tape.

### What actually limits the replay

Three identical open-loop runs, measured:

* Runs are **deterministic** through tape ~6.5s — achieved roll rate differs
  by 0.00–0.12 rad/s between runs.
* All three then hit a **1214–2839 m/s² acceleration spike at tape 7.08–7.18s**
  — a collision, at the same place every time, just before gate 1 (tape 7.578).
* After the collision the runs fork completely.

So the sim is repeatable and the launch and command paths are solved; the
limit is accumulated position error with no position reference to correct it
(`LOCAL_POSITION_NED` is disabled, and EKF `pos_d` reads 2000 m). Determinism
is what makes iterative learning viable — errors are reproducible, so folding
them back into the tape converges.

## Pilot mode (manual ↔ auto)

```powershell
.\winvenv\Scripts\python.exe -m pip install pygame
.\winvenv\Scripts\python.exe tools\tune_flight.py pilot
```

Plug in a **PlayStation / Xbox** pad before starting. Console should print
`[PAD] connected: …`. Keyboard still works; stick deflection overrides that axis.

Manual pilot stays **disarmed with no setpoints** until you move a stick or
flight key — then it arms and flies your input. No early hover / early start.

Stick reverse feel: `PILOT_PAD_SMOOTH` (default 0.55, higher=snappier),
`PILOT_PAD_EXPO` (default 0.40), `PILOT_MAX_RATE_DEG` (default 100 — how fast
attitude can flip left↔right).

Neutral sticks blend gravity-aided AHRS (with disagreement fallback to EKF)
so level corrects mid-race, not only after a long hover. Brief wings-level
also re-snaps the EKF (`PILOT_LEVEL_REALIGN_S`).

### Replay a saved fast prefix

Practice auto-saves best-through-gate attitude tapes under `practice/`.
Fast G6 (16.187s) freeze: `practice/through_gate_6.fast_20260801_040349.json`

```powershell
# Replay best tape through gate 6, then you fly
.\winvenv\Scripts\python.exe tools\tune_flight.py pilot --practice-from-gate 6

# Or through gate 5, then push for a faster G6 yourself
.\winvenv\Scripts\python.exe tools\tune_flight.py pilot --practice-from-gate 5
```

### Gamepad (Xbox / Mode 2)

Unbind the controller inside FlightSim so sticks reach our XInput reader.
You want `[PAD] connected: … via XInput` in the console.

| Stick / button | Action |
|---|---|
| **Left** stick | roll / pitch (up=forward) |
| **Right** stick X | yaw |
| **RT** | climb (analog) |
| **LT** | sink (analog) |
| **RB** | extra thrust (collective bump) |
| **A** | level |
| **B** | quit |
| **X** | HUMAN |
| **Y** | **RESET** run (sim pad; arms on next stick/key) |
| **LB** (or D-pad ↑) | AUTO on LOCK |
| **D-pad ↓** | toggle **slow-mo** (client time scale) |
| **Start** | KEEP |

### Keyboard

| Key | Action |
|---|---|
| `W` `S` / `↑` `↓` | pitch forward / back |
| `A` `D` / `←` `→` | roll left / right |
| `Q` `E` | yaw left / right |
| `R` `F` | climb / sink |
| `Space` | level now |
| **`T`** | **AUTO** on LOCK |
| **`H`** | **HUMAN** sticks again |
| **`Y`** | **RESET** run (sim pad; arms on next stick/key) |
| **`O`** | toggle **slow-mo** (client time scale) |
| **`K`** | **KEEP** remembered keys (with `--capture`) |
| `Esc` / `X` | quit |

### Slow-mo (practice with Cheat Engine / DxWnd)

There is no MAVLink time-scale API. Slow the **sim** with CE Speedhack or
DxWnd, and toggle the **client** to the same factor so attitude/key tapes
stay aligned:

| Control | Effect |
|---|---|
| **`O`** / **D-pad ↓** | toggle client slow-mo |
| `--slow-mo` | start with it ON |
| `PILOT_SLOW_MO_SCALE=0.77` | factor when ON (default 0.77) |
| `PILOT_SLOW_MO=1` | start ON via env |

When ON, the pilot sleeps longer and advances tape/key playheads slower
(HUD shows `x0.77`). **Always match CE/DxWnd to that same number** or
replay / attitude drift. Timed runs / PBs: leave slow-mo **OFF**.

Everything the estimator reports stays **wall-referenced**, so under CE the
HUD / pilot CSV speed (`km/h`, `speed_kmh`) reads low by the slow-mo factor
— 141532 showed 21 km/h at x0.50 where a 1x run reads 40–50. That is a
readout artefact of the slowdown; multiply by `1/scale` when comparing runs.

Do **not** re-time IMU arrivals into sim seconds to "fix" it. Under CE the
IMU stream is already wall-referenced, so scaling `dt` halves reported lean:
`ekf/des` fell to 0.49 with pitch rate pinned at 100°/s and the craft
somersaulted off the pad (143405). Attitude is only trustworthy on the raw
wall clock.

## Why "level" used to walk off, and what now holds it

Integrated gyro has no absolute reference: bias, timing slop and the fact
that rotations do not commute all accumulate, so the EKF's idea of neutral
slides. 141532 went +4° → −23° over 50 s — by the end, holding the stick
centred meant holding a real 23° bank.

The accelerometer cannot fix this on a quadrotor. Specific force only equals
gravity in equilibrium; under thrust it lies along body −z and reads *level*
while you are genuinely leaned. "Leaned + accelerating" and "level + wrong
estimate" produce the identical signal (both `|acc_ned| = g·sin θ`), which is
why an ungated accel blend dragged a held 30° step flat
(`step_pitch_20260728_174353`). `EKF_ACCEL_TILT_GAIN` therefore ships at **0**.

The gate is the way out. Gates hang vertical, so an upright gate's own Y axis
*is* gravity, and PnP recovers it exactly — immune to acceleration. Two aids
run off it, both slow and both fail-safe (a rejected frame simply means no
correction):

| Knob | Default | What it does |
|---|---|---|
| `EKF_GATE_HORIZON_GAIN` | 0.10 | Proportional pull of roll/pitch onto the gate's vertical axis |
| `EKF_GATE_HORIZON_BIAS_GAIN` | 0.30 | **Integral into gyro bias — the one that matters** |
| `EKF_GATE_YAW_GAIN` / `_BIAS_GAIN` | 0.06 / 0.20 | Same pair for yaw, against a per-gate heading anchor |
| `EKF_GATE_*_MAX_STEP_DEG` | 1.0 | Clamp per correction, so one bad pose nudges but never yanks |
| `EKF_GATE_ATT_MAX_RANGE_M` | 30 | Gate *rotation* goes bad with range well before its centre does |
| `EKF_GATE_ATT_MAX_REPROJ_PX` | 6 | Screen outliers on solve quality |

It is a Mahony filter: proportional on attitude, integral into gyro bias. The
bias term is what actually fixes things — 152912 had a near gate only **64%**
of the time with gaps up to 2.5 s, and pulling attitude only while a gate is
in view lets it drift straight back during the gaps. Learning the bias removes
the cause, so attitude holds through the gaps. Flying that measured duty
cycle at 0.5°/s bias, the worst excursion goes **1.29° → 0.07°**, and the
filter recovers the bias to within 0.001°/s.

Outliers are screened on solve geometry and reprojection error, **never** on
how far the measurement sits from the filter's own belief. The first version
rejected anything more than 30° from the current estimate, which meant a
drifted filter threw away the evidence that it had drifted — 23% of 152912
sat beyond that threshold.

Tilted gates still bias the horizon by their own lean while you look at them,
but they do *not* corrupt the learned bias: once attitude settles on the tilt
the innovation goes to zero, so the error stays a transient offset.

The HUD shows `gh<count> b<x>/<y>` — applied horizon fixes and the learned
gyro bias in deg/s. If `gh` stays at 0, the aid is not running at all (check
`EKF_USE_PNP` is not left at `0` from a spline capture); the bias should
settle to a small non-zero constant. Same fields land in the telemetry CSV as
`gh_fixes`, `gh_rejects`, `gh_skips`, `bias_gx/gy/gz`.

On the pilot side the AHRS blend gate was inverted: disagreement *grows* as
the EKF drifts, so "they disagree, keep EKF" muted the correction exactly
when it was needed (eligible on only 11.6% of 141532). It now checks whether
the raw IMU can vouch for AHRS (near-1g, not spinning) and trusts it when it
can — `PILOT_LEVEL_AHRS_DRIFT_DEG`, `_DRIFT_WIDEN`, `_MAX_DISAGREE_DEG`.

Human teleop boosts collective while leaned (`PILOT_TILT_COMPENSATE=1`) so
hard forward does not drop altitude. Set `PILOT_TILT_COMPENSATE=0` for flat
hover if you prefer to own altitude with LT/RT only.

`W` is pitch only. Yaw is `Q`/`E` (or left stick X) — live auto
yaw-after-GATE-1 is **off** (`PILOT_LIVE_POST_G1_YAW=0`).

## Best lap (coach notes)

Tracked PB: **35.963s** full course (`20260731_001506`, `logs/best/`).
(Same session also had a slower 37.473s finish — scorer keeps the fastest.)
Open-loop attitude speed-ups diverge on this sim — fly these yourself:

| Split | Time | Tip |
|---|---|---|
| **g16→g17** | **2.80s** | Still a big chunk — commit earlier into last gate |
| **g7→g8** | **2.22s** | Carry more pitch through the turn |
| g5→g6 | 3.12s | Biggest mid-course loss — less hover |
| g4→g5 | 2.25s | |
| g12→g13 | 2.62s | |

Faithful tape: `logs/best/attitude_pb.json` (replay drifts; use as reference only).

```powershell
.\winvenv\Scripts\python.exe tools\tune_flight.py pilot `
  --replay-attitude logs\best\attitude_pb.json --seconds 55
```

## Practice from a gate (optimize splits)

The sim **cannot teleport** mid-course. Practice works by replaying your best
**pad attitude commands** (lean / yaw-rate / thrust — full analog precision)
from the start pad **through gate N**, then giving you the sticks for **N+1**.

While you fly normally, practice auto-saves under `practice/`:

| File | Meaning |
|---|---|
| `practice/through_gate_3.json` | Best attitude tape through GATE 3 |
| `practice/index.json` | Times / splits summary |
| `practice/runs/partial/` | Every unfinished attempt (Y-reset or quit) |
| `practice/runs/complete/` | Full-course finishes (cleared gate 17) |
| `practice/runs/index.json` | Recent attempt list |

```powershell
# See what you've banked
.\winvenv\Scripts\python.exe tools\tune_flight.py practice

# Replay your saved run through gate 3, then YOU fly gate 4+
.\winvenv\Scripts\python.exe tools\tune_flight.py pilot --practice-from-gate 3
```

- **Y** / pad **Y** archives the attempt (partial/complete), then restarts.
- Quit also archives the current attempt.
- **K** / Start force-saves every through-gate *best* checkpoint from the current run.
- Disable auto-save: `--no-practice-save` or `PRACTICE_AUTO_SAVE=0`.

## Remember-path (exact key presses)

Records **which keys you held and for how long**, then presses them again on
replay — same mapping as manual.

### Capture

```powershell
$env:EKF_USE_PNP="1"
.\winvenv\Scripts\python.exe tools\tune_flight.py pilot --capture
```

Fly with **held** keys (not taps). Console prints `KEY W DOWN` / `KEY W UP`.
Wait for `GATE 1 @ t=...`, then **K**. Writes `captured_controls.json`
(`type: key_timeline`).

### Replay through a gate, then append your keys

Memorizes **keys** (not position). YOLO does not steer during REPLAY.

Pure keys through GATE 2 (open-loop; can miss G2 under sim variance):

```powershell
.\winvenv\Scripts\python.exe tools\tune_flight.py pilot `
  --replay captured_controls_g2_locked.json --keep-until-gate 2
```

Sticks stay on the key timeline until **GATE 2 actually clears** (not the early
tag), then hand to **you**. New key holds append — fly the next segment, then
**K** to keep.

### Hybrid: keys through GATE 1 → ASSIST for GATE 2

Open-loop keys cannot guarantee G2 every time. Prefer keys for the solid
takeoff/G1, then closed-loop ASSIST tip-through for G2, then HUMAN for G3+:

```powershell
.\winvenv\Scripts\python.exe tools\tune_flight.py pilot `
  --replay captured_controls_g2_locked.json `
  --keep-until-gate 1 --assist-after-gate 1
```

(`--human-after-gate` defaults to 2. After real GATE 2 clears, sticks return to
you — fly/append, then **K**.)

Same via config / env: `PILOT_ASSIST_AFTER_GATE=1`, `PILOT_HUMAN_AFTER_GATE=2`.
Short settle before ASSIST: `PILOT_ASSIST_AFTER_GATE_DELAY_S` (default 0.35 s).

### Hybrid: clean acro tape through GATE 1 → ASSIST through the finish

Pure open-loop acro replay is timing-faithful but accumulates lateral state
error and clips gate 2. For a completion attempt, keep Cheat Engine at **0.2×**
and hand the clean, deterministic G1 prefix to vision assist for G2–G17:

```powershell
.\winvenv\Scripts\python.exe tools\tune_flight.py pilot `
  --replay-attitude practice\runs\HARDCODED_REPLAY_CLEAN.json `
  --assist-after-gate 1 --human-after-gate 17
```

This path preserves the source tape as exact/ZOH `acro_rates` commands until
the real G1 event. It keeps playing the tape until vision has a live next-gate
lock, seeds assist from that lock, then stays closed-loop through G17. Watch
for `REPLAY→ASSIST after real GATE 1`; if it instead says it is waiting for
`LOCK`, do not change CE speed—the missing vision lock is the issue.

F is mild through GATE 1 (`PILOT_SINK_RATE`=0.6 m/s). After GATE 1, F uses
`PILOT_G2_SINK_RATE` (default 1.0 m/s). Pure-key HUMAN handoff delay:
`PILOT_HANDOFF_AFTER_GATE_S`.

Old `control_timeline` / waypoint JSON files will not load — re-capture once.

## Manual / assist

```powershell
.\winvenv\Scripts\python.exe tools\tune_flight.py manual
.\winvenv\Scripts\python.exe tools\tune_flight.py assist --seconds 30
```
