# Manual / pilot teleop

Focus the **PowerShell console** for keys — not the FlightSim window.

## Pilot mode (manual ↔ auto)

```powershell
.\winvenv\Scripts\python.exe -m pip install pygame
.\winvenv\Scripts\python.exe tools\tune_flight.py pilot
```

Plug in a **PlayStation / Xbox** pad before starting. Console should print
`[PAD] connected: …`. Keyboard still works; stick deflection overrides that axis.

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
| **Y** | **RESET** run (sim pad + re-arm) |
| **LB** (or D-pad ↑) | AUTO on LOCK |
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
| **`Y`** | **RESET** run (sim pad + re-arm) |
| **`K`** | **KEEP** remembered keys (with `--capture`) |
| `Esc` / `X` | quit |

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

F is mild through GATE 1 (`PILOT_SINK_RATE`=0.6 m/s). After GATE 1, F uses
`PILOT_G2_SINK_RATE` (default 1.0 m/s). Pure-key HUMAN handoff delay:
`PILOT_HANDOFF_AFTER_GATE_S`.

Old `control_timeline` / waypoint JSON files will not load — re-capture once.

## Manual / assist

```powershell
.\winvenv\Scripts\python.exe tools\tune_flight.py manual
.\winvenv\Scripts\python.exe tools\tune_flight.py assist --seconds 30
```
