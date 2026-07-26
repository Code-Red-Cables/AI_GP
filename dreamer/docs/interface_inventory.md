# Interface Inventory

Exhaustive list of every value crossing the sim↔client boundary, its transport, its
VQ2 availability, and its **classification**: `LEGAL` (usable by the deployed policy),
`PRIV` (training/reward/eval only — must never enter the actor), or `ABSENT` (nulled in
VQ2). This table is the source of truth for `tests/test_no_privileged_input.py`.

## Inbound: Telemetry (MAVLink UDP :14550)

| Field | Msg | Transport | VQ2 | Class | Notes |
|---|---|---|---|---|---|
| roll, pitch, yaw | ATTITUDE | UDP | ❌ **ABSENT (measured)** | — | ATTITUDE not sent in VQ2 (0 msgs/30 s probe); no direct orientation |
| rollspeed, pitchspeed, yawspeed | ATTITUDE | UDP | ❌ **ABSENT** | — | use HIGHRES_IMU gyro instead |
| xacc, yacc, zacc | HIGHRES_IMU | UDP | ✅ | **LEGAL** | body linear accel (m/s²) — orientation tilt derived from this |
| xgyro, ygyro, zgyro | HIGHRES_IMU | UDP | ✅ | **LEGAL** | body angular rates (rad/s) — the control-feedback signal |
| xmag, ymag, zmag | HIGHRES_IMU | UDP | ⚠️ 0 | — | no magnetometer; always ~0 |
| abs_pressure, pressure_alt | HIGHRES_IMU | UDP | ⚠️ | LEGAL? | baro — **(MEASURE)** if meaningful in VQ2 (currently treated as absent) |
| time_usec | HIGHRES_IMU | UDP | ✅ | LEGAL | IMU timestamp |
| x, y, z (position) | LOCAL_POSITION_NED / ODOMETRY | UDP | ❌ | ABSENT | nulled VQ2; PRIV if re-enabled for training |
| vx, vy, vz (velocity) | LOCAL_POSITION_NED / ODOMETRY | UDP | ❌ | ABSENT | nulled VQ2; PRIV if re-enabled |
| active_gate_idx | ENCAPSULATED_DATA(1) | UDP | ✅ | **PRIV** | race progress; reward + eval only |
| race_start_ms, race_finish_ns | ENCAPSULATED_DATA(1) | UDP | ✅ | **PRIV** | episode timing; reward + eval only |
| last_gate_time | ENCAPSULATED_DATA(1) | UDP | ✅ | **PRIV** | split time; eval only |
| gate id, x,y,z,quat | ENCAPSULATED_DATA(2) | UDP | ❌ | ABSENT | pose nulled VQ2 |
| gate w, h | ENCAPSULATED_DATA(2) | UDP | ✅ | PRIV | gate aperture size; reward geometry / eval |
| collision id, threat, impulse | COLLISION | UDP | ✅ | **PRIV** | reward penalty only; never a policy input |
| actuator[0..3] | ACTUATOR_OUTPUT_STATUS | UDP | ✅ | PRIV | motor debug |

## Inbound: Camera (UDP :5600)

24-byte little-endian header `<IHHIIQ>` then JPEG slice:

| Field | Type | Class | Notes |
|---|---|---|---|
| frame_id | u32 | LEGAL | sequence id (dup/drop detection) |
| chunk_id / total_chunks | u16 | LEGAL | reassembly |
| jpeg_size / payload_size | u32 | LEGAL | reassembly |
| sim_time_ns | u64 | LEGAL (timing) | frame stamp; **causal** — used only for age/dt, never future frames |
| JPEG payload → RGB 640×360 | bytes | **LEGAL** | the primary observation |

## Outbound: Control (MAVLink UDP :14550)

| Command | Msg | Fields | Notes |
|---|---|---|---|
| body-rate + thrust | SET_ATTITUDE_TARGET | mask=ATTITUDE_IGNORE, roll_rate, pitch_rate, yaw_rate, thrust | **the only honored control** |
| arm | COMMAND_LONG | MAV_CMD_COMPONENT_ARM_DISARM=1 | |
| reset | COMMAND_LONG | cmd=31000 (vendor) | episode reset — **(MEASURE)** semantics/ACK |

## Derived / client-side (LEGAL, computed from LEGAL inputs only)

| Field | Source | Class |
|---|---|---|
| gate_detection (center_px, corners, area, confidence) | `vision/gate_detector.py` on camera RGB | **LEGAL** (pure vision) |
| dt (since last processed frame) | camera `sim_time_ns` deltas | LEGAL |
| prev_action | client memory | LEGAL |

## The deployed-policy observation contract (LEGAL only)

```
obs = {
  "image":   uint8[H,W,3]        # downsampled camera RGB
  "vector":  float32[13]         # [gyro(3), accel(3), tilt_roll, tilt_pitch,
                                 #  prev_action(4), dt]  — all LEGAL, all IMU-derived
}
```
(ATTITUDE-based orientation/yaw removed after the probe showed ATTITUDE is absent; tilt
comes from the accelerometer, yaw is carried by the camera + integrated by the RSSM.)
Nothing classified `PRIV`/`ABSENT` may appear in `obs`. Enforced by
`env/observation_builder.py` (it is only handed the legal subset) and asserted by
`tests/test_no_privileged_input.py`.
