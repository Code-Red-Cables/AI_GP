# Claude Task: Build a DreamerV3 Training and Deployment System for Autonomous Drone Racing

<!-- ============================================================================ -->
<!-- LIVING PROGRESS TRACKER — updated every work session. Keep this in sync with reality. -->
<!-- All code lives under dreamer/. Spec (unchanged) begins below the tracker. -->
<!-- ============================================================================ -->

> **📋 PROGRESS TRACKER — last updated: 2026-07-23 (M4 live: DreamerV3 trains; reward inversion fixed; needs GPU)**

### 🔬 Live-run findings (M2) — **ENVIRONMENT VALIDATED**
The purpose of M2 was to verify the sim plumbing/constants (resets, action mapping,
telemetry, rewards, loop) — NOT to build a good pilot. **That is done and passing.**
- **Run 1:** constant-pitch-rate baseline → continuous tumble. Diagnosed from per-step data.
- **Run 2 (lean-hold baseline):** no longer flips up, but limit-cycles (gyro_y ≈ +1.9 rad/s)
  and descends. Root cause: **accelerometer-only tilt is unusable during dynamic motion**, so
  a hand-coded attitude loop chases a corrupted angle. A stable scripted pilot would need a
  gyro+accel **AHRS** — the exact state-estimator we chose NOT to build (the RSSM learns it).
- **Verified correct:** action→gyro mapping is consistent (`gyro_y ≈ a_pitch × ~7.5`,
  matches probe), reset works, telemetry + reward components populate, loop runs/terminates.
- **Decision:** do NOT rabbit-hole a scripted AHRS pilot. Env is validated → proceed to the
  DreamerV3 training run (the agent learns the controller + state estimation). Optional: a
  complementary-filter baseline later, only if we want cleaner replay-seeding demos.

### Status at a glance
| Milestone | State | Notes |
|---|---|---|
| M1 — Probe: timing + control | ✅ **done** | camera 30 Hz, IMU 114 Hz, ATTITUDE absent, all signs measured |
| M2 — Env + reset + reward tests | 🟡 built, **needs live validation** | unit-tested; run `collect_demos` against sim |
| M3 — Dataset + baseline | 🟡 built | scripted/random baselines + replay + accumulator |
| M4 — DreamerV3 trains | 🟢 model + **decoupled trainer built & offline-verified** · 🟡 needs live run | collector+learner threads, weight-sync, crash-recovery |
| M5 — Multi-gate/full course + eval | ⬜ pending live data | eval harness built |
| M6 — Deploy-clean export + leakage | ✅ export + leakage test pass · 🟡 live latency pending | |

### Measured ground truth (from probe runs — binding, do not re-assume)
- **Camera** 29.997 Hz, dt 33±3 ms (p95 36), 0 drops. Sim re-sends each frame ~38× → `camera_io` dedupes.
- **HIGHRES_IMU** 114 Hz. **ATTITUDE / LOCAL_POSITION_NED / ODOMETRY ABSENT** (VQ2, confirmed). No position, no gate poses, 0 track gates. `race_status` streams (active_gate/finish).
- **Control:** ACRO rates respond immediately post-arm; **all 3 rate axes inverted (sign −1)**; **~2.5× gain** → `max_rate_rad_s=3.0`; thrust strong; accel-z gravity-negative → tilt uses `-az`. Reset (cmd 31000) → fresh frame in ~26 ms (**gate-progress rewind unconfirmed**).
- **Sim runs ≈real-time 1:1** → ~30 env-steps/s/instance (wall-clock bound).

### Observation / action (LOCKED — RSSM-only state estimation, per user)
- `obs.image` = camera RGB 64×64. `obs.vector[13]` = gyro(3)+accel(3)+tilt_roll/pitch(2)+prev_action(4)+dt. IMU-only; yaw omitted (camera + RSSM carry heading). RSSM latent = the learned state estimator.
- `action ∈ [−1,1]^4` = `[thrust, roll_rate, pitch_rate, yaw_rate]`, scaled in `env/spaces.py`.

## ✅ Completed so far (built + verified offline)
All code lives under `dreamer/`. Everything below is implemented and passing its offline
tests; nothing here has been run against the live sim yet except the two probes.
- [x] **Audit & docs** — `docs/`: simulator_audit, system_architecture, interface_inventory, assumptions_and_open_questions, deployment_compliance, training_throughput
- [x] **Package + config** — `config.py` (typed dataclasses + YAML), `configs/dreamer_small.yaml` & `dreamer_tiny.yaml`, `pyproject.toml`, `README.md`
- [x] **Sim interface** (`sim/`) — mavlink_io, camera_io (frame-dedupe), action_sender (scaling/slew/watchdog), timing, privileged_state, process_manager, reset_manager
- [x] **RL env** (`env/`) — Gym-style drone_racing_env, observation_builder (13-dim IMU obs), reward (componentized, privileged-gated), termination, curriculum, baselines, spaces
- [x] **DreamerV3** (`dreamer/`) — distributions (symlog/two-hot/categorical-ST/tanh-normal), networks (CNN±MLP), RSSM, agent (world model + actor-critic + imagination), replay (thread-safe)
- [x] **Deploy-clean controller** (`deploy/controller.py`) + minimal `export_deploy` (encoder+RSSM+actor only)
- [x] **Probes** — `tools/probe_simulator.py` (timing) + `tools/probe_control.py` (signs/reset); **both run live; config calibrated from the results**
- [x] **Decoupled trainer** (`train/`) — collector thread (CPU inference → thread-safe replay, sim-crash recovery) + learner (GPU, train-ratio pacing, weight-sync, checkpoints) → `scripts/train_dreamer.py`; also `evaluate.py`, `deploy.py`, `collect_demos.py`
- [x] **Tests** — reward-geometry 10/10 · leakage-audit 3/3 · dreamer-shapes 8/8 · decoupled-trainer · offline integration (config→replay→train→export→deploy)

## 🔲 What still needs to be done (ordered)
1. [x] **Live env validation** — DONE: resets/action-mapping/telemetry/reward/loop all verified live (see Live-run findings). Scripted pilot intentionally not perfected.
2. [x] **First live training run** — DONE: DreamerV3 trains live end-to-end, no errors, WM loss 4.1→2.0, checkpoints save (M4 ✅ live). Found two blockers (both addressed): CPU-only torch (learner never used GPU; ups~1.3, sps~6) and a **reward inversion** (hover −39 < crash −20 → agent learned to crash; fixed `w_offcourse→0` so hover −3 > crash −20, race +30, finish +150).
2b. [ ] **Install CUDA torch** ← NEXT · `pip uninstall -y torch && pip install torch --index-url https://download.pytorch.org/whl/cu124` — learner onto the RTX 4070 (~50–100× updates/s; collector reclaims CPU → ~30 sps)
2c. [ ] **Re-run training on GPU** with `dreamer_small` for a longer stint; watch ep_r trend upward, episodes lengthen, first gate passed. Fill `docs/training_throughput.md`.
3. [ ] Fill in `docs/training_throughput.md` from that run (GPU util, updates/s, VRAM)
4. [ ] Re-run `probe_control` **mid-race** to confirm cmd-31000 rewinds gate progress (only tested from gate 0 so far)
5. [ ] Scale throughput: parallel sim instances on separate UDP ports
6. [ ] Full training campaign: curriculum live · multi-gate/full-course · reward-exploit checks
7. [ ] Evaluation harness live (finish/gate/collision rates, latency) · leaderboard
8. [ ] Deploy latency benchmark (CPU vs GPU) · deployment-clean full-course attempt

## 🧹 Workspace (as of 2026-07-23 cleanup)
- Removed all generated `__pycache__/`; added a root `.gitignore` (ignores venv, caches, `*.pt`, training runs, sim binary, old debug dirs; **keeps** `artifacts/probe/`).
- **Kept:** root flight stack (`controller.py`, `mavlink_rx.py`, `vision/`, … = fallback controller per spec), `artifacts/probe/` (measured data), `AIGP_3385/` (sim).
- **Recommend you delete** (stale June vision captures, ~98 MB, not used by the DreamerV3 system): `frames/` (33 MB) and `_vision_debug/` (65 MB) — left in place pending your OK since they're your captured images.

<!-- ============================================================================ -->
<!-- ORIGINAL TASK SPEC (unchanged) BELOW -->
<!-- ============================================================================ -->

## Role

Act as a senior reinforcement-learning, robotics, simulation, and systems engineer.

Your task is to inspect the available drone-racing simulator and build a complete, reproducible DreamerV3-based training and deployment pipeline.

I have full local access to the simulator, including the executable, binaries, engine files, configuration files, logs, and any project/source assets present in the repository or installation directory.

Do not assume that the simulator behaves exactly as external documentation claims. Measure the actual behavior of the installed simulator.

The final system must include a real training environment, not only a neural-network implementation.

---

# Primary Goal

Build an end-to-end system that:

1. Launches and resets the simulator automatically.
2. Receives the drone's observations.
3. Measures the actual camera and telemetry timing.
4. Converts the simulator into a reinforcement-learning environment.
5. Trains a DreamerV3 agent.
6. Evaluates the trained agent on complete racing runs.
7. Exports a lightweight inference-only controller.
8. Preserves a deployment path that does not depend on privileged simulator information.

The target controller should race using only observations that will be available during the real competition run.

Privileged simulator state may be used for reward calculation, curriculum generation, debugging, evaluation, and auxiliary training targets, but not as an input to the final deployed policy unless I explicitly confirm that the competition permits it.

---

# Important Working Rules

## Do not make undocumented assumptions

Before implementing the full training system, inspect the simulator and determine:

- How it is launched.
- How it is reset.
- Whether it exposes an API, socket, shared memory region, log stream, plugin interface, engine console, RPC endpoint, MAVLink stream, or other control channel.
- How images are transmitted.
- How telemetry is transmitted.
- How actions are submitted.
- Whether simulator time can be accelerated.
- Whether rendering can be disabled or reduced.
- Whether multiple simulator instances can run in parallel.
- Whether deterministic seeding is supported.
- Whether gate positions, drone pose, collisions, and race progress can be accessed.
- Whether the simulator can be instrumented without modifying race behavior.
- Whether source or engine project files are available.
- Whether there is an existing baseline controller or interface implementation.

Do not assume the camera is 30 Hz.

Measure:

- Mean camera frame rate.
- Frame-rate distribution.
- Inter-frame jitter.
- Duplicate frames.
- Dropped frames.
- End-to-end image latency if measurable.
- Telemetry rate.
- Action-command rate.
- Simulator physics tick rate.
- Relationship between wall-clock time and simulation time.
- Whether image and telemetry timestamps share the same clock.
- Whether the simulator pauses, slows, or drops frames during model inference.

Record these measurements in a generated report.

## Preserve the simulator installation

- Do not overwrite original simulator files.
- Do not patch binaries in place without explicit approval.
- Make backups before changing any configuration or engine project file.
- Prefer wrappers, plugins, launch arguments, copied environments, or reversible instrumentation.
- Document every simulator-side change.

## Maintain two distinct interfaces

Implement:

### Training interface

May use privileged simulator state for:

- Dense rewards.
- Gate-crossing detection.
- Collision detection.
- Curriculum initialization.
- Automated resets.
- Evaluation metrics.
- Auxiliary world-model targets.

### Deployment interface

Must operate only on competition-legal runtime observations and the allowed action interface.

The deployed actor must not silently depend on:

- Global position.
- Exact gate coordinates.
- Unreal/engine object references.
- Simulator-only collision state.
- Hidden race-progress values.
- Ground-truth depth.
- Segmentation masks.
- Future frames.
- Noncausal state.

Keep these interfaces structurally separate so accidental privileged-state leakage is easy to detect.

---

# Phase 1: Repository and Simulator Audit

Inspect the full workspace before changing code.

Create:

- `docs/simulator_audit.md`
- `docs/system_architecture.md`
- `docs/interface_inventory.md`
- `docs/assumptions_and_open_questions.md`

The simulator audit should include:

- Relevant files and directories.
- Executable names and launch commands.
- Engine type and version if discoverable.
- Networking interfaces.
- Existing APIs.
- Existing sample code.
- Configuration files.
- Log locations.
- Camera pipeline.
- Telemetry pipeline.
- Control pipeline.
- Reset mechanisms.
- Race-state mechanisms.
- Potential privileged-state sources.
- Potential compliance risks.
- Performance bottlenecks.
- Recommended integration strategy.

Do not begin a large rewrite until the audit is complete.

---

# Phase 2: Timing and Interface Probe

Build a small standalone probe before integrating DreamerV3.

Create a tool such as:

```text
tools/probe_simulator.py
```

It should:

- Launch or attach to the simulator.
- Receive camera frames.
- Receive telemetry.
- Send safe neutral or low-risk commands.
- Timestamp every received and transmitted item.
- Detect duplicates and out-of-order packets.
- Measure frame dimensions and encoding.
- Measure actual update rates.
- Measure latency where timestamps permit.
- Save a short sample dataset.
- Produce machine-readable statistics.
- Produce a human-readable report.

Expected outputs:

```text
artifacts/probe/
  camera_timestamps.csv
  telemetry_timestamps.csv
  action_timestamps.csv
  rate_summary.json
  probe_report.md
  sample_frames/
```

The probe must work independently of the RL stack.

---

# Phase 3: Build the Reinforcement-Learning Environment

Yes, this task explicitly includes building the environment used to train DreamerV3.

Create an environment with a clean API similar to Gymnasium or DeepMind `dm_env`.

Example:

```python
obs, info = env.reset(seed=seed)
obs, reward, terminated, truncated, info = env.step(action)
```

The environment must support:

- Simulator launch.
- Simulator shutdown.
- Clean reset.
- Episode timeout.
- Crash recovery.
- Observation synchronization.
- Action scaling.
- Reward calculation.
- Gate-progress tracking.
- Collision detection.
- Finish detection.
- Logging.
- Deterministic seeds when available.
- Reconnection after simulator or socket failure.
- Watchdogs for stale image or telemetry streams.

Suggested structure:

```text
src/
  sim/
    process_manager.py
    transport.py
    camera_receiver.py
    telemetry_receiver.py
    action_sender.py
    timing.py
    privileged_state.py
    reset_manager.py
  env/
    drone_racing_env.py
    observation_builder.py
    reward.py
    termination.py
    curriculum.py
    wrappers.py
```

## Observation design

Start with a multimodal observation containing:

```text
camera:
  latest causal RGB frame

telemetry:
  orientation representation
  angular velocity
  linear velocity, if available at deployment
  other explicitly legal runtime fields

history:
  previous action
  optional time delta since previous observation
  optional valid-data masks
```

Do not hardcode the camera update rate.

The environment should expose the measured `dt` or otherwise account for irregular observation intervals.

Do not fabricate repeated observations without marking them. If the action loop runs faster than the camera, hold the last action or reuse the latest observation in a deliberate, documented way.

Implement configurable preprocessing:

- Original image capture.
- Resize.
- Center crop or letterbox.
- Normalization.
- Optional grayscale.
- Optional frame stacking for baselines.
- No future-frame leakage.
- No use of image timestamps unavailable at deployment unless represented only through causal timing metadata.

Initial image sizes to benchmark:

- 64×64
- 96×64
- 128×72

Select the smallest resolution that preserves reliable gate perception.

## Action design

Prefer a continuous action space matching the simulator's supported high-level control interface.

Candidate action:

```text
[
  collective_thrust,
  roll_rate_target,
  pitch_rate_target,
  yaw_rate_target
]
```

Determine the real allowed ranges empirically and from the simulator interface.

Implement:

- Normalized policy output in `[-1, 1]`.
- Safe scaling to physical command limits.
- Optional slew-rate limiting.
- Optional low-pass filtering.
- Emergency neutral action.
- Watchdog behavior.
- Configurable action repeat or control hold.

Do not place DreamerV3 directly in the motor commutation loop. Use the simulator or flight stack's stable lower-level controller where available.

## Reward design

Use privileged state only inside the training environment.

Implement reward terms separately and log each component:

```text
progress_reward
gate_pass_reward
finish_reward
time_penalty
collision_penalty
wrong_gate_penalty
backtracking_penalty
control_smoothness_penalty
excessive_rate_penalty
off_course_penalty
```

Start with:

```text
reward =
    w_progress * change_in_progress_to_next_gate
  + w_gate * gate_crossed
  + w_finish * race_finished
  - w_time * elapsed_simulation_time
  - w_collision * collision
  - w_control * control_cost
```

Requirements:

- Dense progress must not reward flying toward the wrong side of a gate.
- Gate crossing should be based on crossing the gate plane through the valid aperture.
- Reward should be based on simulation time rather than training-machine wall-clock time.
- Log raw and weighted reward components.
- Make all weights configurable.
- Detect reward exploits.
- Add unit tests for reward geometry.

## Reset and curriculum support

Support reset distributions for:

1. Stable hover or basic forward flight.
2. Single-gate approach.
3. Random pose near a gate.
4. Two-gate sequence.
5. Short track segment.
6. Full course.
7. High-speed full-course racing.
8. Recovery from imperfect orientation or velocity.

Curriculum advancement should depend on measured success rates, not only a fixed number of steps.

---

# Phase 4: Baselines Before DreamerV3

Before debugging DreamerV3, verify the environment with simpler agents.

Implement or integrate:

1. A scripted safety controller.
2. A random-action smoke test with strict command limits.
3. A privileged-state oracle controller for environment validation only.
4. A simple behavior-cloning baseline if demonstration data exists.
5. A state-based PPO or SAC baseline using privileged observations.
6. An image-based simple policy baseline if practical.

The privileged baseline is not for deployment. It is used to prove that:

- Resets work.
- Rewards point in the correct direction.
- Gate detection works.
- Actions are correctly mapped.
- The course is solvable through the wrapper.

Do not blame DreamerV3 until the environment passes these tests.

---

# Phase 5: Dataset and Replay Collection

Build a replay collection pipeline.

Sources may include:

- Existing controller runs.
- Human or scripted demonstrations.
- Privileged oracle runs.
- Randomized low-speed exploration.
- DreamerV3 rollouts.
- Failure and recovery trajectories.

Store:

```text
image
telemetry
previous_action
action
reward
reward_components
terminated
truncated
episode_id
step_index
camera_timestamp
telemetry_timestamp
action_timestamp
simulation_time
gate_index
collision_flag
privileged_training_labels
```

Privileged fields must be clearly marked and excluded from deployment observations.

Use an efficient sequence-friendly storage format such as:

- Reverb
- Zarr
- HDF5
- Parquet plus compressed image storage
- DreamerV3's expected replay format

Choose based on the selected DreamerV3 implementation.

Create dataset inspection tools that can:

- Replay an episode.
- Plot rewards.
- Plot action values.
- Plot observation timing.
- Show gate-crossing events.
- Detect missing or corrupted frames.
- Confirm causal ordering.
- Verify that deployment observations contain no privileged fields.

---

# Phase 6: DreamerV3 Integration

Prefer starting from a maintained DreamerV3 implementation rather than reimplementing the entire algorithm from memory.

Before choosing an implementation, evaluate:

- Framework compatibility.
- GPU support on the available machine.
- Windows compatibility.
- WSL compatibility.
- Replay format.
- Continuous-action support.
- Dictionary or multimodal observation support.
- Recurrent-state export.
- Checkpointing.
- Inference latency.
- License.
- Code quality.
- Community usage.
- Ease of modifying encoders.

Document the choice.

## Model architecture

Use:

```text
RGB frame -> CNN encoder
telemetry -> MLP encoder
previous action -> action embedding
combined features -> recurrent state-space model
latent state -> actor, critic, reward predictor, continuation predictor
```

The world model should learn from temporally ordered sequences.

The actor should output a continuous distribution over normalized controls.

Account for variable or measured time intervals. At minimum:

- Include observed `dt` as an input, or
- Resample deliberately to a fixed decision rate, or
- Advance the recurrent model only on new camera observations while holding actions between frames.

Do not silently assume uniform 30 Hz transitions.

## Auxiliary privileged prediction

Optionally add training-only prediction heads from the latent state for:

- Relative next-gate position.
- Relative next-gate orientation.
- Distance to gate plane.
- Gate image coordinates.
- Drone velocity.
- Collision probability.
- Track progress.

These targets can encourage useful latent representations.

Rules:

- Auxiliary labels must never be concatenated into the actor's deployment input.
- The actor must still run when privileged-state code is disabled.
- Add an automated privileged-information leakage test.

## Replay seeding

If a working controller exists:

1. Collect successful and unsuccessful runs.
2. Pretrain the world model on those sequences.
3. Train the actor and critic in imagination.
4. Begin cautious online collection.
5. Mix demonstration, failure, and current-policy data.

Do not fill the entire replay buffer with only ideal runs. Include recoveries and near-failures.

## Configuration

Create version-controlled configuration files for:

```text
environment
observation preprocessing
action limits
reward weights
curriculum
Dreamer model size
sequence length
batch size
replay size
train ratio
imagination horizon
learning rates
exploration
checkpoint frequency
evaluation frequency
seed
```

Every run must save its complete configuration.

---

# Phase 7: Training Throughput

Measure actual simulator throughput.

Investigate:

- Headless mode.
- Lower rendering quality.
- Fixed or uncapped simulation speed.
- Disabling unnecessary visual effects.
- Smaller camera resolution at the source.
- Multiple independent simulator instances.
- Parallel replay collection.
- Decoupled learner and actor processes.
- GPU inference.
- Batched environment inference.
- Remote learner support.
- Checkpoint resume.

Do not change simulator physics merely to gain speed unless the change is understood, documented, and validated.

If simulation acceleration changes camera timing or controller behavior, treat accelerated training and real-time evaluation as separate modes.

Create:

```text
docs/training_throughput.md
```

Include:

- Environment steps per second.
- Camera frames per second.
- Learner updates per second.
- GPU utilization.
- CPU utilization.
- Replay write throughput.
- Estimated time for 1M, 5M, and 10M transitions.
- Main bottlenecks.
- Recommended scaling strategy.

---

# Phase 8: Evaluation

Create a separate evaluation harness.

It must run with:

- Exploration disabled.
- Fixed checkpoints.
- Fixed seed sets where supported.
- No training updates.
- No privileged actor inputs.
- Real-time timing constraints.
- Competition-like startup and reset behavior.

Report:

- Finish rate.
- Gate completion rate.
- Lap or course time.
- Collision rate.
- Wrong-gate rate.
- Mean and worst inference latency.
- Camera-frame age at action time.
- Control smoothness.
- Performance by starting condition.
- Performance across repeated runs.
- Performance with privileged-state access physically disabled from the policy process.

Save videos with overlays showing:

- Current action.
- Inference latency.
- Frame age.
- Gate index for evaluation only.
- Reward components for debugging only.
- Whether the current run is deployment-clean.

Keep a leaderboard CSV for checkpoints.

---

# Phase 9: Deployment Controller

Export an inference-only controller containing only:

```text
observation preprocessing
Dreamer encoder
recurrent latent-state update
actor
action scaling
watchdog and safety logic
```

Do not include:

- Replay buffer.
- World-model decoder if unnecessary.
- Critic if unnecessary.
- Training optimizer.
- Privileged-state readers.
- Reward code.
- Curriculum code.
- Evaluation-only overlays.

The deployment process should:

1. Start cleanly.
2. Connect to camera and telemetry.
3. Measure initial stream health.
4. Initialize recurrent state.
5. Process the newest causal observation.
6. Emit actions within the allowed interface.
7. Drop stale frames rather than queueing them.
8. Hold or safely decay actions if observations are missing.
9. Reset recurrent state between race attempts.
10. Log enough timing information for debugging without blocking control.

Benchmark CPU and GPU inference.

Target inference latency should be comfortably below the observed inter-frame interval, not an assumed 33 ms.

---

# Phase 10: Compliance and Leakage Audit

Create:

```text
docs/deployment_compliance.md
tests/test_no_privileged_input.py
```

The audit should answer:

- What exact values enter the deployed policy?
- Where does each value come from?
- Is each value available during a competition run?
- Are any simulator memory reads used?
- Are any hidden engine values used?
- Are gate coordinates embedded in model inputs?
- Is the course memorized through an explicit lookup table?
- Does the final runtime require modified simulator files?
- Can the deployment process run with privileged-state services disabled?
- Are training-only modules excluded from the package?

Do not make a rules judgment when rules are unclear. Flag the uncertainty and preserve a conservative, clean deployment mode.

---

# Engineering Requirements

Use:

- Typed Python where practical.
- Clear module boundaries.
- Structured logging.
- Configuration files rather than scattered constants.
- Unit tests.
- Integration tests.
- Reproducible seeds.
- Checkpointing.
- Resume support.
- Graceful process cleanup.
- Timeouts on all external communication.
- No infinite blocking socket reads.
- No hidden global mutable state.
- No silent exception swallowing.

Add:

```text
README.md
pyproject.toml or requirements files
scripts/setup.*
scripts/train.*
scripts/evaluate.*
scripts/deploy.*
```

Commands should be simple, for example:

```bash
python -m tools.probe_simulator
python -m scripts.collect_demonstrations
python -m scripts.train_dreamer --config configs/dreamer_small.yaml
python -m scripts.evaluate --checkpoint checkpoints/best
python -m scripts.deploy --checkpoint checkpoints/best
```

Use platform-appropriate equivalents where needed.

---

# Testing Milestones

Do not try to finish everything in one unverified pass.

## Milestone 1

- Simulator launches.
- Camera and telemetry are received.
- Safe actions are transmitted.
- Actual rates are measured.
- No RL code required yet.

## Milestone 2

- Automated reset works.
- Gym-style environment works.
- Reward and termination tests pass.
- Privileged oracle can pass at least one gate.

## Milestone 3

- Dataset collection and replay inspection work.
- A simple baseline learns or behavior cloning reproduces demonstrations.
- Timing and action mapping are validated.

## Milestone 4

- DreamerV3 trains without shape, replay, or synchronization errors.
- World-model predictions improve on held-out sequences.
- One-gate curriculum succeeds.

## Milestone 5

- Multi-gate and full-course learning.
- Evaluation is repeatable.
- Reward exploits are addressed.
- Throughput is characterized.

## Milestone 6

- Inference-only export.
- Real-time performance benchmark.
- Privileged-state leakage audit passes.
- Deployment-clean full-course attempts.

After each milestone:

1. Run tests.
2. Summarize what works.
3. List failures and uncertainties.
4. Commit or checkpoint the code.
5. Update the architecture documentation.

---

# Decision-Making Guidance

When multiple approaches are possible:

1. Prefer the least invasive simulator integration.
2. Prefer measured behavior over documentation assumptions.
3. Prefer a working environment and baseline over premature model complexity.
4. Prefer legal runtime observations over privileged state.
5. Prefer body-rate plus thrust control over direct motor output unless the simulator interface clearly requires otherwise.
6. Prefer a smaller Dreamer model that runs reliably in real time before scaling up.
7. Prefer automated tests for reward geometry and synchronization.
8. Prefer collecting useful demonstrations before random pixel-based exploration.
9. Preserve a fallback controller throughout development.
10. Keep training, evaluation, and deployment code paths separate.

---

# Questions You Should Resolve From the Local Files

Answer these through inspection rather than asking me immediately when possible:

- What engine is the simulator built on?
- Are engine project or source files present?
- Is there a built-in Python, C++, Blueprint, plugin, or RPC interface?
- Can the simulator run headlessly?
- Can simulation time run faster than real time?
- Can multiple instances bind to separate ports?
- How is race state represented internally?
- How are gate crossings detected?
- Can drone pose and gate transforms be accessed for training rewards?
- What is the actual camera refresh rate?
- Is the camera rate fixed or load-dependent?
- What is the telemetry rate?
- What is the physics rate?
- What control commands are accepted?
- What are the command limits?
- What reset mechanisms exist?
- Can deterministic seeds be configured?
- What existing controller code can seed the replay buffer?
- What operating system and GPU constraints apply?

Only ask me for information that cannot be determined safely from the available files or runtime behavior.

---

# Initial Deliverable

Start by producing the audit and a proposed implementation plan.

Do not immediately create a massive codebase.

Your first response after inspecting the workspace should contain:

1. A concise summary of the simulator architecture.
2. The measured or discoverable observation and action interfaces.
3. The proposed training-environment design.
4. The proposed reward source.
5. The proposed DreamerV3 implementation choice.
6. The major technical risks.
7. A phased implementation plan.
8. The exact files you plan to create or modify.
9. Any rule or compliance uncertainty.
10. The smallest next implementation step that validates the approach.

Then implement Milestone 1.

---

# Definition of Done

The task is complete only when:

- The simulator can be controlled programmatically.
- The actual camera and telemetry timing are measured.
- A reliable automated training environment exists.
- Rewards and resets are tested.
- Demonstrations can be collected.
- DreamerV3 can train from simulator sequences.
- The agent can progress through the curriculum.
- Evaluation is automated.
- Inference runs under real timing constraints.
- The deployment controller uses no privileged input.
- The full system is documented and reproducible.
