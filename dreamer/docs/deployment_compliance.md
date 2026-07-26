# Deployment Compliance & Leakage Audit

Answers the Phase-10 questions. Enforced by `tests/test_no_privileged_input.py`.

### What exact values enter the deployed policy?
Only the LEGAL observation contract (`interface_inventory.md` §"deployed-policy obs"):
- `image`: downsampled camera RGB (from UDP :5600 JPEG).
- `vector`: `[roll, pitch, sinψ, cosψ, roll_rate, pitch_rate, yaw_rate, ax, ay, az,
  prev_action(4), dt]` — all from `ATTITUDE` + `HIGHRES_IMU` + client memory.
Plus the actor's own recurrent latent `(h, z)` carried across steps.

### Where does each value come from?
Camera → `sim/camera_io.py`. Attitude/IMU → `sim/mavlink_io.py`. prev_action/dt →
client state. **No value originates in `sim/privileged_state.py`.**

### Is each value available during a competition run?
Yes — camera, ATTITUDE, HIGHRES_IMU are the runtime streams. `dt` is measured locally.
No position, no gate pose, no gate index, no collision oracle in the policy input.

### Are any simulator memory reads / hidden engine values used?
No. The client is a pure external MAVLink+UDP consumer; the binary is never patched or
memory-read (`simulator_audit.md` §1).

### Are gate coordinates embedded in model inputs? Is the course memorized via a lookup table?
No. Gate world-poses are nulled in VQ2 and never referenced. There is no waypoint/gate
coordinate table in the deployment path (unlike the legacy `spline_planner` mission
replay, which is **excluded** from deployment).

### Does the runtime require modified simulator files?
No.

### Can deployment run with privileged-state services disabled?
Yes — that is the design. `deploy/controller.py` does **not import**
`sim/privileged_state`, `env/reward`, `env/termination`, `env/curriculum`, the critic, or
the world-model decoder. Killing every privileged service leaves the actor fully
functional. The leakage test imports `deploy.controller` and asserts none of those modules
are in `sys.modules` as a result.

### Are training-only modules excluded from the exported package?
Yes. `scripts/deploy.py` loads a checkpoint and instantiates only
`encoder + RSSM + actor + action_sender`. Replay buffer, optimizer, decoders, critic,
reward, curriculum, and eval overlays are not part of the export.

## Leakage guardrails (automated)
1. `observation_builder.build_obs()` is only ever handed the LEGAL field subset; PRIV
   fields are read exclusively inside `reward.py` / `termination.py` / eval.
2. `tests/test_no_privileged_input.py`:
   - asserts the built obs dict keys ⊆ LEGAL contract;
   - asserts `import deploy.controller` does not pull in any PRIV module;
   - asserts the actor forward pass runs with `privileged_state` monkey-patched to raise.
3. Auxiliary privileged prediction heads (next-gate bearing, collision prob, progress) live
   on the world model for representation shaping and are **structurally unreachable** from
   the actor's input tensor.

## Uncertainty handling
Where a rule is unclear (`assumptions_and_open_questions.md` §12–14) we **do not
adjudicate**; we keep the conservative clean mode as the default deployment and flag the
item. The clean policy needs none of the ambiguous signals, so it is safe under the strict
reading regardless of how the ambiguity resolves.
