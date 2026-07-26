# System Architecture — DreamerV3 Drone Racing

## Package layout (`dreamer/`)

```
configs/                    version-controlled YAML (env, model, reward, curriculum)
src/dreamer_drone/
  config.py                 dataclass config + YAML loader; every run saves its full config
  sim/                      SIM INTERFACE (may touch privileged state)
    mavlink_io.py           pymavlink RX/TX thread (adapted from ../mavlink_rx.py)
    camera_io.py            UDP JPEG receiver thread (adapted from ../vision_rx.py)
    action_sender.py        normalized action -> SET_ATTITUDE_TARGET, scaling/slew/watchdog
    timing.py               rate/jitter/latency meters
    privileged_state.py     PRIV signal reader (race status, collision) — TRAINING ONLY
    process_manager.py      launch/attach/kill the Windows sim from WSL
    reset_manager.py        cmd 31000 reset + settle + stream health
  env/                      RL ENVIRONMENT (Gymnasium-style; may use privileged reward)
    spaces.py               obs/action space defs, action scaling constants
    observation_builder.py  LEGAL-only obs assembly (image + vector); no leakage
    reward.py               componentized reward (all terms logged), privileged-gated
    termination.py          terminated/truncated logic + tests hooks
    curriculum.py           success-rate-driven reset-distribution scheduler
    drone_racing_env.py     reset()/step() orchestration, watchdogs, crash recovery
  dreamer/                  WORLD MODEL + ACTOR-CRITIC (PyTorch)
    distributions.py        symlog, two-hot, OneHotCategorical straight-through
    networks.py             CNN image encoder/decoder, MLP encoders/heads
    rssm.py                 recurrent state-space model (deterministic GRU + stochastic categorical)
    agent.py                WorldModel + Actor + Critic; imagination rollout; losses
  deploy/
    controller.py           DEPLOYMENT-CLEAN inference: encoder+RSSM+actor+scaling+watchdog
tools/probe_simulator.py    Milestone-1 standalone timing/interface probe
scripts/                    train_dreamer.py, evaluate.py, deploy.py, collect_demos.py
tests/                      reward geometry, no-privileged-input, model shapes
docs/                       this audit set + throughput + compliance reports
artifacts/                  probe output, checkpoints, replays, eval videos, leaderboard
```

## Two-interface separation (compliance backbone)

```
                 ┌─────────────── TRAINING PROCESS ───────────────┐
   sim ──MAVLink──▶ mavlink_io ─┬─▶ observation_builder ─(LEGAL)─▶ obs ─▶ agent
        ──UDP img──▶ camera_io ─┘                                          │
                    privileged_state ─(PRIV)─▶ reward / termination / eval │
                                                                           ▼
   sim ◀─SET_ATTITUDE_TARGET── action_sender ◀──────────── action ◀──── actor

                 ┌────────────── DEPLOYMENT PROCESS ──────────────┐
   sim ──▶ mavlink_io + camera_io ─▶ observation_builder ─(LEGAL)─▶ deploy/controller
   sim ◀── action_sender ◀── actor            (NO privileged_state import at all)
```

`deploy/controller.py` imports **only** `sim/{mavlink_io,camera_io,action_sender}`,
`env/observation_builder`, `env/spaces`, and the frozen `dreamer` encoder+RSSM+actor.
It **does not import** `sim/privileged_state`, `env/reward`, `env/curriculum`, or the
critic/decoder. `tests/test_no_privileged_input.py` asserts this by import inspection
and by checking the deployed obs dict keys against the LEGAL contract.

## Observation design (VQ2, LEGAL only)

- `image`: latest **causal** camera frame, JPEG-decoded → configurable resize
  (benchmark 64×64 / 96×64 / 128×72; pick smallest with reliable gate perception),
  optional grayscale, `uint8` for replay compactness (normalized in the encoder).
- `vector` (`float32[13]`, **IMU-derived — ATTITUDE is absent in VQ2, measured**):
  `gyro_xyz, ax, ay, az, tilt_roll, tilt_pitch, prev_action[4], dt`. Body rates come from
  HIGHRES_IMU gyro; `tilt_roll/pitch` are the accelerometer gravity tilt (driftless).
  **Yaw is omitted** — it is unobservable without a magnetometer, so the camera carries
  heading and the RSSM integrates orientation over the sequence. `dt` is the measured
  inter-frame interval (~33 ms ±3 ms), letting the RSSM condition on irregular timing.
- Fabricated/duplicate frames are **marked** with a `valid` mask, never silently repeated.

## Action design

- Policy outputs `a ∈ [−1,1]^4` = `[thrust, roll_rate, pitch_rate, yaw_rate]`.
- `action_sender` scales: `thrust = HOVER + a0·THR_SPAN` clamped `[0.05,0.90]`;
  `rate_i = a_i·RATE_MAX`, applies `RATE_SIGN_PITCH=-1`, optional slew-rate limit + LPF,
  emergency-neutral on watchdog, configurable action-hold if control loop > camera rate.
- Identical scaling code path is shared by training and deployment (no train/deploy skew).

## Reward (componentized, all terms logged raw + weighted)

```
r = w_progress·Δprogress            # dense IF privileged position available, else vision-proxy
  + w_gate·gate_passed              # active_gate_idx increment  (PRIV, always available)
  + w_finish·finished               # race_finish_ns>0           (PRIV)
  - w_time·dt_sim                    # sim-time penalty
  - w_collision·collision            # COLLISION msg              (PRIV)
  - w_control·control_cost           # |Δaction| smoothness + rate magnitude
  - w_offcourse·gate_lost            # vision: no gate visible (LEGAL proxy)
```

`Δprogress` source is auto-selected: privileged position→gate-plane distance if the probe
proves position is available for training; otherwise the **deployment-legal** vision proxy
= growth in detected-gate area + centering (bounded, exploit-guarded). Gate crossing is a
`PRIV` state-machine on `active_gate_idx`, not on geometry, so it is robust in VQ2.

## World model (DreamerV3)

`image→CNN`, `vector→MLP`, `prev_action→embed` → concat → **RSSM** (GRU deterministic
`h` + categorical stochastic `z`, symlog+two-hot reward/critic, KL-balanced free-bits).
Heads off latent: image decoder, vector decoder, reward, continue, **and training-only
auxiliary heads** (relative next-gate bearing/size, collision prob, progress) that shape
the latent but are **never** concatenated into the actor input. Actor/critic trained in
imagination with λ-returns. Variable `dt` enters as a vector-obs feature and (optionally)
as an RSSM step scaler.

## Process lifecycle

Each background thread follows the repo's `create_* / get_thread_for_join()` pattern.
Env `reset()` = `reset_manager` (cmd 31000 → settle → stream-health gate) → prime rate
stream → arm → wait first fresh frame. Watchdogs abort on stale image/telemetry; env
raises a typed `SimUnavailable` that the training loop catches → relaunch via
`process_manager` → resume from checkpoint.
