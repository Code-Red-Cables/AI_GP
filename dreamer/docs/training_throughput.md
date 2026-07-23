# Training Throughput

_Partly measured (probe 2026-07-23), partly pending live training._

## The binding constraint: the sim is wall-clock, real-time, single-stream

The probe measured **30.00 Hz camera over 30 s of wall clock with no slow-down** while
commands streamed. So one sim instance yields **~30 environment steps per second** and
does **not** free-run or accelerate on its own. This is the dominant throughput fact:
collection is real-time-bound, not compute-bound.

| Quantity | Value | Source |
|---|---|---|
| Env steps/s per instance | ~30 | camera 29.997 Hz, decision rate = camera rate |
| Camera frames/s | 29.997 | probe |
| IMU telemetry/s | 114.2 | probe |
| Frame re-transmit factor | ~38× (now discarded pre-decode) | probe + `camera_io` fix |
| Learner updates/s | GPU-bound (pending) | RTX 4070, measure live |
| Wall-clock for 1 M / 5 M / 10 M steps @30 Hz, 1 instance | ~9.3 h / ~46 h / ~93 h | 1e6/30/3600 |

At 30 steps/s a single instance needs ~9 hours per **million** transitions — too slow for
a from-scratch DreamerV3 run (which typically wants millions). The two levers:

## Lever 1 — decouple the learner from the actor
The learner (GPU) is far faster than 30 Hz. Run collection and learning **concurrently**:
a collector thread/process fills the replay at 30 Hz while the learner trains at the
configured `train_ratio` off the replay. The current `train_dreamer.py` interleaves them
in one loop; the next step is to split them (a background collector + a learner consuming
the shared `SequenceReplay`) so GPU time is never blocked on the 30 Hz sim.

## Lever 2 — parallel sim instances (the real scaler)
Because each instance is real-time-locked, **N instances ≈ N×30 steps/s**. Needs:
- separate UDP ports per instance (`SimConfig.mavlink_port` / `camera_port` are configurable);
- separate Windows sim processes (`process_manager` launch args) — **(MEASURE)** whether the
  shipping build accepts a port override and can run headless/reduced-render to fit VRAM/CPU.
4 instances → ~120 steps/s → ~2.3 h/M. This is the recommended scaling path.

## Data-efficiency levers (reduce steps needed)
- **Replay seeding** with scripted/vision-baseline runs (`collect_demos.py`) before RL, so
  the world model starts from real gate-approach sequences (prompt Phase 5).
- **High `train_ratio`** (DreamerV3 is very sample-efficient): squeeze many gradient steps
  per env step from the GPU while the sim plods at 30 Hz.
- Small image (64×64) + `dreamer_small` model → fits the 8 GB 4070 and keeps GPU updates
  fast; benchmark updates/s and VRAM live and record here.

## Measured — first live run (2026-07-23, `dreamer_tiny`, **CPU learner**)
The learner ran on **CPU** because the installed torch is CPU-only (`2.5.1+cpu`,
`cuda_available=False`) — the RTX 4070 was never used. Result:

| metric | value | note |
|---|---|---|
| learner updates/s | **~1.3** | CPU train_step on the tiny model ≈ 0.7 s each |
| collector steps/s | **~6** (started ~20, fell as learner ramped) | starved of CPU by the learner (both threads on CPU + GIL) |
| episodes | all end in collision, mean **0.5 s** | drone crashes instantly (untrained + reward bug, since fixed) |
| projection @6 sps | 1 M steps ≈ **46 h** | unusable |

Two findings from this run: (1) **CPU is the bottleneck** — the learner starves the
collector, so *both* are slow; (2) a **reward inversion** made crashing optimal (fixed:
`w_offcourse→0`, so hover −3 > crash −20).

## Next: get on the GPU (the #1 lever)
Install a CUDA build of torch so the learner uses the RTX 4070 (WSL `/dev/dxg`):
```
pip uninstall -y torch && pip install torch --index-url https://download.pytorch.org/whl/cu124
```
Expected effect: learner updates/s ~50–100× (GPU), and the collector reclaims the CPU →
approaches the sim-bound ~30 sps. Single instance then does 1 M steps in ~9–11 h; parallel
instances scale from there. **To fill in after the GPU run:** updates/s, GPU util, VRAM at
`dreamer_small`, realized sps, and N-instance scaling.
