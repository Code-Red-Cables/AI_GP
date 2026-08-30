# Classical vision racing (Li & de Croon)

`FLIGHT_MODE=race` — an alternate planner, **not** the timed submission.
The timed path is `FLIGHT_MODE=policy` ([`HG_DAGGER.md`](HG_DAGGER.md)).

## What it is

Bearing-vector least-squares gate pose, a 7-state drag EKF, and a PD-align /
feed-forward-arc planner:

- `vision/gate_ls_pose.py` — attitude known, 3 unknowns
- `ekf/drag_ekf.py` — bias integrates once, not twice
- `race_planner.py` — PD when the gate is visible, arc when blind
- `tools/build_course_map.py`, `tools/eval_gate_pose.py`, `tools/identify_drag.py`

Wired through `config.py` / `setup.py` / `main.py`. Pose weights and the
Windows venv live in this checkout (`models/gate_pose_v5.pt`, `winvenv/`).

## Run (experiment only)

```powershell
$env:FLIGHT_MODE='race'
.\winvenv\Scripts\python.exe main.py
```

Score with `tools/score_runs.py` / `tools/approach_metrics.py` if you are
comparing this stack to itself. Do not use those numbers as the submission
clock.

Offline solver / EKF tests:

```powershell
.\winvenv\Scripts\python.exe -m unittest test_gate_ls_pose test_drag_ekf
```

## If you revive this path

1. Simulator running, logged in, race started.
2. YOLO pose weights under `models/` (same detector as the rest of the client).
3. Optional: `tools/eval_gate_pose.py` on saved frames before trusting range
   at speed.
4. One flown lap → `tools/identify_drag.py --telem logs/telem_....csv` →
   `DRAG_KX` / `DRAG_KY`.
5. `tools/build_course_map.py --telem ...` if you want a map file. The
   policy path does not need one.
