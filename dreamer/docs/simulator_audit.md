# Simulator Audit — AI Grand Prix DCL FlightSim

_Phase 1 deliverable. Derived from static inspection of `AIGP_3385/`, the existing
WSL flight client (`../*.py`), `reference/AI Grand Prix Tech Specs.pdf`, and
`docs/VISION_REBOOT_PLAN.md`. Fields marked **(MEASURE)** must be confirmed with a
live probe run (`tools/probe_simulator.py`) — they are inferred, not yet measured._

## 1. Identity & engine

| Property | Value | Source |
|---|---|---|
| Product | "DCGame" / DCL FlightSim (Drone Champions League FPV racer) | `Manifest_NonUFSFiles_Win64.txt` → `DCGame-Win64-Shipping.exe` |
| Engine | Unreal Engine 4 (PhysX3, FMOD, Steamworks, `UE4PrereqSetup`) | Manifest DLL set |
| Launcher | `AIGP_3385/FlightSim.exe` (167 KB shim) → `FlightSim/Binaries/Win64/DCGame-Win64-Shipping.exe` | `file(1)` = PE32+ GUI x86-64 |
| OS | **Windows 11 only.** Spec §5.1: "Currently we do not support Linux OS." | Spec |
| Source/project files | **None present.** Only shipped `.pak` content + binaries. No `.uproject`, no Blueprints, no plugin SDK. | `find AIGP_3385` |
| Config | `FlightSim/Binaries/pgos_res/pgos_config.ini` (PGOS online-services, not sim behavior) | manifest |

**Consequence:** the sim is a **closed black box**. No engine console, no RPC, no
plugin interface, no editable project. The *only* control surface is the documented
MAVLink + UDP SITL bridge. We integrate as an external client — never by modifying
the binary (which also satisfies the prompt's "preserve the installation" rule).

## 2. How it is launched / reset

- **Launch:** run `FlightSim.exe` on Windows. From WSL2 this is reachable via interop
  (`/mnt/d/.../FlightSim.exe`) but it opens a GUI window on the Windows desktop and
  renders there. Headless / offscreen: **(MEASURE)** — UE4 supports `-nullrhi` /
  `-RenderOffscreen`, but unknown if this shipping build accepts engine args. Assume
  **not headless** until proven.
- **Reset:** the existing client issues `command_long(command=31000)` — a
  vendor-specific `MAVLINK_CMD_SIM_RESET` (`controller.py:8,89`). **MEASURED
  (`probe_control`):** a fresh camera frame arrives **~26 ms** after the reset (fast, one
  frame). Whether it fully rewinds gate progress is **still unconfirmed** — the probe ran
  from `active_gate=0`, so re-run it mid-race (after passing a gate) to verify `active_gate`
  returns to 0. Adequate for a hover / single-gate curriculum now; confirm before full-course.

## 3. Transport & interfaces

| Channel | Transport | Direction | Notes |
|---|---|---|---|
| MAVLink 2 | UDP `:14550` (`main.py:6`) | both | `udpin` — client binds, sim connects in. Heartbeat-gated. |
| Camera | UDP `:5600` | Sim→Client | Custom 24-byte header + chunked JPEG. |
| Control | MAVLink `SET_ATTITUDE_TARGET` | Client→Sim | Body-rates + thrust (see §5). |
| Reset/arm | MAVLink `COMMAND_LONG` | Client→Sim | `ARM_DISARM`, `31000` reset. |

WSL2↔Windows: the current pipeline already bridges localhost UDP across the boundary,
so networking is a solved problem we inherit. Host/port are made configurable in the
new stack (`configs/env.yaml`) so we can point at the Windows host IP if mirrored-mode
networking is not in use.

## 4. Telemetry pipeline (Sim→Client, MAVLink)

Parsed today in `../mavlink_rx.py`:

| Message | VQ2 status | Use |
|---|---|---|
| `HEARTBEAT` | ✅ | connection liveness |
| `ATTITUDE` | ✅ arrives (spec lists it "disabled" but it is sent) | **orientation + body rates** — primary legal telemetry |
| `HIGHRES_IMU` | ✅ | accel + gyro (mag = 0, no magnetometer) |
| `TIMESYNC` | ✅ | clock alignment (client requests @10 Hz, `timesync.py`) |
| `LOCAL_POSITION_NED` | ❌ nulled in VQ2 | privileged position (VQ1 only) |
| `ODOMETRY` | ❌ nulled in VQ2 | privileged pose/vel (VQ1 only) |
| `ENCAPSULATED_DATA` type=1 (race status) | ✅ | `active_gate_idx`, `race_start_ms`, `race_finish_ns`, `last_gate_time` |
| `ENCAPSULATED_DATA` type=2 (track info) | ⚠️ gate x/y/z/quat **nulled** in VQ2 | gate count + sizes only |
| `ACTUATOR_OUTPUT_STATUS` | ✅ | motor outputs (debug) |
| `COLLISION` | ✅ | `id` 1001=Gate / 1002=Environment, threat, impulse |

Struct layouts (verbatim from working parser):
- Race status: `<BQqqIq>` = `type, sim_boot_ms, race_start_ms, race_finish_ns, active_gate_idx(u32), last_gate_time`.
- Track gate record: `<Hfffffffff>` = `id, x, y, z, qw,qx,qy,qz, w, h` (x..qz = 0 in VQ2).
- Collision: `msg.id`, `msg.threat_level`, `msg.horizontal_minimum_delta` (impulse kg·m/s).

**(MEASURE):** confirm `LOCAL_POSITION_NED`/`ODOMETRY` are truly absent (not just
unused) in the VQ2 build, and whether they can be re-enabled for **training reward**.
This is the single most impactful open question — see §7.

## 5. Control pipeline (Client→Sim) — **binding**

From `../controller.py` + `VISION_REBOOT_PLAN.md` §3 (paid for in crashed runs):

1. The sim is a **Betaflight-style ACRO racer**. It has **no velocity/position loop** —
   `SET_POSITION_TARGET_LOCAL_NED` velocity setpoints are *silently ignored*. It does
   **not** hold a commanded attitude.
2. The **only honored control** is `SET_ATTITUDE_TARGET` with
   `ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE` set → `[body_roll_rate, body_pitch_rate,
   body_yaw_rate, thrust]`. This is exactly the DreamerV3 action vector.
3. Axis conventions (empirical, `config.py`): **pitch-rate axis is inverted**
   (`RATE_SIGN_PITCH=-1`), roll normal (`RATE_SIGN_ROLL=+1`). Thrust normalized `0..1`,
   open-loop hover ≈ `0.27`, working clamp `[0.05, 0.90]`.
4. Reported **yaw is left-handed vs NED**: physical forward = `(cosψ, −sinψ)`. (Only
   matters if we convert world↔body; the RL policy works in body/rate space and can
   avoid this.)
5. `arm()` = `COMMAND_LONG(MAV_CMD_COMPONENT_ARM_DISARM,1)`. Command rate spec: **< 100 Hz**.
6. **MEASURED (`probe_control` 2026-07-23):** raw ACRO rate control **responds immediately
   post-arm** (no ANGLE handshake needed). All three rate axes are **inverted** vs the gyro
   (`rate_sign_roll/pitch/yaw = -1`), and the sim applies a **~2.5× gain** on roll/pitch
   (cmd 1.5 → ~4 rad/s), ~0.5× on yaw — so `max_rate_rad_s` set to 3.0. Thrust responds
   strongly (Δacc_z ≈ −16 for +0.3). Accel-z reads ≈ −9.8 at rest (gravity-negative
   convention → tilt uses `-az`).

## 6. Timing — **MEASURED** (probe run 2026-07-23, 30 s; `artifacts/probe/`)

| Quantity | Spec | **Measured** | Notes |
|---|---|---|---|
| Physics tick | 120 Hz | — | HIGHRES_IMU ≈ physics clock |
| Camera | 30 Hz, 640×360 | **29.997 Hz**, dt 33.3 ms ±3.0 ms, p95 36.3 ms, min 25/max 39 ms | 0 incomplete dropped |
| HIGHRES_IMU | (n/a) | **114.2 Hz**, dt 8.75 ms ±0.73 ms | primary telemetry (see below) |
| **ATTITUDE** | listed | **0 msgs in 30 s — ABSENT** | ⇒ no direct orientation/yaw; derive from IMU |
| Command rate | < 100 Hz | sent 48.5 Hz OK | 30 Hz control is safely under |
| Sim-time vs wall-clock | 1:1 | **≈1:1** (900 frames / 30 s wall = 30 Hz) | no observed slow-down; RL is wall-clock bound (~30 steps/s/instance) |
| Frame re-transmission | — | **~38× per frame** (33 106 dup completions / 900 frames) | `camera_io` now skips re-decoding duplicates |
| Position telemetry | — | **absent** | confirmed; no privileged dense progress |
| Track gate info | — | **0 gates sent** | gate count/poses unavailable in VQ2 |
| race_status | — | **streams** (active_gate, race_start, finish) | privileged reward source works |

**Key consequence:** ATTITUDE does not arrive, so orientation is not directly observable.
The observation vector is built from **HIGHRES_IMU** instead: body rates (gyro), linear
accel, and accelerometer gravity tilt (roll/pitch, driftless). Yaw is unobservable (no
magnetometer) and omitted — the camera carries heading, the RSSM integrates orientation.
This matches the project's VQ2 pivot (the deleted `ahrs.py`/`state_estimator.py`).
All telemetry-liveness checks key on HIGHRES_IMU, not ATTITUDE.

## 7. Privileged-state sources & the VQ2 reward problem

Reward normally wants dense progress = Δ(distance to next gate), which needs **drone
position** and **gate world-pose**. In VQ2 **both are nulled**. Available privileged
signals reduce to:

- `active_gate_idx` increments → **sparse gate-pass** event.
- `race_finish_ns > 0` → **finish** event.
- `COLLISION` → **collision penalty**.
- `sim_time` → **time penalty**.
- IMU-integrated pseudo-velocity (drift-prone) → weak shaping only.

→ Design decision (see `system_architecture.md`): build the reward on the **guaranteed
sparse signals**, and gate the *dense* progress term behind a `privileged_position`
capability flag that is **auto-detected at runtime** and **off by default**. If the
probe shows position/gate-pose *can* be enabled for training, dense shaping switches on
automatically; if not, we fall back to sparse gate-pass + a **vision-based** proxy
(detected-gate area growth / centering) which is *deployment-legal* and needs no
privileged state.

## 8. Integration strategy (recommended)

- **Least invasive:** external MAVLink+UDP client, zero binary changes. ✅ matches rules.
- Reuse the proven I/O layer (`mavlink_rx`, `vision_rx`, `timesync`, `controller`) as the
  transport substrate; wrap it in a Gymnasium-style env.
- One sim instance per env (UE4 GUI ⇒ parallel instances need separate UDP ports +
  Windows processes — **(MEASURE)** feasibility; assume single instance first).
- Real-time-locked: if the sim does **not** slow during inference (§6), training is
  bounded by 30 Hz camera ⇒ ~30 env-steps/s/instance. Plan throughput around that.

## 9. Compliance risks (flagged, not adjudicated)

- `active_gate_idx` / `race_status`: almost certainly **legal runtime feedback** (the
  race system tells you your progress). Used for reward; **not** fed to the deployed policy.
- `COLLISION` message: ambiguous — a real drone doesn't get a collision oracle. Treated
  as **privileged (training-only)**; never a policy input.
- Gate world-poses / position: **privileged**, nulled in VQ2 anyway.
- The deployed policy consumes **only** camera + `ATTITUDE` + `HIGHRES_IMU` + prev-action
  + dt. See `deployment_compliance.md`.
