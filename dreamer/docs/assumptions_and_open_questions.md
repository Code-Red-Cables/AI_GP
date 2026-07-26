# Assumptions & Open Questions

Ranked by impact on the design. **(MEASURE)** = resolvable by `tools/probe_simulator.py`
against a live sim. **(ASK)** = needs a competition-rules / human answer.

## Blocking — status after probe (2026-07-23)

1. **RESOLVED — sim runs ≈real-time 1:1.** 900 camera frames over 30 s wall = 30.00 Hz;
   no slow-down observed while streaming commands. ⇒ training is **wall-clock bound at
   ~30 env-steps/s/instance**. Acceleration/headless still unconfirmed (see #7).
2. **RESOLVED — position telemetry is truly ABSENT** (user-confirmed + probe
   `position_available=false`). No privileged dense-progress reward is possible.
   `reward.use_privileged_progress` stays **false**; dense shaping = the deployment-legal
   **vision proxy** (detected-gate area growth) plus sparse gate-pass/finish. Also
   **track gate info is not sent** (0 gates), so `num_gates` is unknown → finish detection
   relies solely on `race_finish_ns > 0`.
2b. **NEW FINDING — `ATTITUDE` is ABSENT** (0 msgs/30 s). No direct orientation. The
   observation vector was reworked to be **IMU-derived** (gyro rates + accel + accel tilt);
   yaw omitted (no mag), carried by the camera. All liveness checks re-keyed to
   HIGHRES_IMU. This was a latent bug (resets would have hung on a dead ATTITUDE stream).
3. **PARTIAL — `cmd 31000` reset** produces a fresh frame in ~26 ms (fast, acknowledged),
   but whether it rewinds gate progress is **unconfirmed** (probe ran from `active_gate=0`).
   Re-run `probe_control` mid-race to confirm `active_gate→0` before relying on it for
   full-course episodes. Fine for hover/single-gate now.
4. **RESOLVED — ACRO control responds immediately post-arm** (no ANGLE handshake). Measured:
   all three rate axes **inverted** (`rate_sign_*=-1`), ~2.5× gain on roll/pitch (~0.5× yaw)
   → `max_rate_rad_s=3.0`; thrust responds strongly; accel-z is gravity-negative (tilt uses
   `-az`). Config updated. NB: the rate signs matter for the scripted baseline and
   interpretability; the RL policy learns the mapping either way.

## High — resolve early

5. **RESOLVED — camera 29.997 Hz**, dt 33.3 ms ±3.0 ms (p95 36.3, min 25/max 39), 0
   dropped. Decision rate 30 Hz confirmed; watchdogs must tolerate ~40 ms gaps.
5b. **NEW — sim re-transmits each frame ~38×** (33 106 dup completions). `camera_io` now
   skips re-decoding already-completed frame ids (big latency/CPU win).
6. **(MEASURE, still open) Do camera `sim_time_ns` and IMU `time_usec` share an epoch?**
   The env currently derives `dt` from camera `sim_time_ns` deltas and sim-time from IMU
   `time_usec`; confirm the two clocks are consistent.
7. **(MEASURE, still open) Headless / reduced-render / accelerated-sim support.** Since the
   sim is wall-clock bound, this is the main throughput lever. Governs `training_throughput.md`.
8. **(MEASURE, still open) Multiple sim instances on separate UDP ports** for parallel
   collection. Given real-time binding, parallel instances are the primary way to scale steps/s.

## Medium

9. Smallest camera resolution preserving reliable gate perception (benchmark 64×64 /
   96×64 / 128×72). Affects encoder + throughput.
10. Body-rate command limits (`RATE_MAX`) and thrust span that keep control stable —
    refine from `config.py` values (hover 0.27, lean/rate signs) via probe + oracle runs.
11. Reward exploit surface: does the vision-proxy progress term admit a
    "hover-and-stare-at-gate" exploit? Guarded by pairing area-growth with gate-index
    advancement; monitor in eval.

## Compliance / rules **(ASK if unresolved by rules doc)**

12. Is `active_gate_idx` / `race_status` legal as **runtime feedback** to the client
    (not to the policy net)? Assumed yes (it is the race system's own progress readout).
    It is used only for reward/eval, never as a policy input, so the deployed policy is
    clean either way.
13. Is the `COLLISION` message legal at runtime? Conservatively treated as **privileged**;
    policy never sees it. No action needed unless we later want it as a deploy input.
14. Confirm the competition forbids **only** human-in-the-loop control (spec §7 says human
    interaction ⇒ DQ) and permits a fully autonomous learned policy. Assumed yes.

## Environment assumptions (this workstation)

- WSL2 + `/dev/dxg` GPU passthrough (RTX 4070, 8 GB). CUDA torch expected to work in WSL;
  CPU torch used here only to verify model shapes. **8 GB VRAM caps Dreamer model size** —
  default to `dreamer_small` (see config); benchmark VRAM before scaling.
- Sim runs on the Windows host; client + trainer run in WSL; localhost UDP already bridges
  (inherited from the working pipeline). If mirrored networking is off, set the Windows
  host IP in `configs/env.yaml`.
- Sim cannot be launched/probed from this non-interactive session (it opens a GUI on the
  Windows desktop). All **(MEASURE)** items require the user to run the probe live; the
  numbers currently in `docs/simulator_audit.md` §6 are spec-claimed, not measured.
