# Classical vision racing (Li & de Croon)

Branch: `classical-vision-racing`.

## What landed

- `vision/gate_ls_pose.py` — bearing-vector least-squares gate pose (attitude known, 3 unknowns)
- `ekf/drag_ekf.py` — 7-state drag EKF (bias integrates once, not twice)
- `race_planner.py` — PD align when gate visible, feed-forward arc when blind
- `tools/build_course_map.py`, `tools/eval_gate_pose.py`, `tools/identify_drag.py`
- `FLIGHT_MODE=race` wired through `config.py` / `setup.py` / `main.py`

## Run (when the sim is back)

```powershell
$env:FLIGHT_MODE='race'
.\winvenv\Scripts\python.exe main.py
```

Score with `tools/score_runs.py` / `tools/approach_metrics.py`.

## Blocked on this checkout

The working tree no longer contains:

- `AIGP_3385/` — the simulator
- `frames/` — 36k images for Phase 1b GO/NO-GO
- `logs/telem_*.csv` — drag identification / GATE_PASSED anchors
- `models/*.pt` — YOLO weights
- `venv/` / `winvenv/` — Python environments

So Phase 1b (`tools/eval_gate_pose.py`), drag fitting (`tools/identify_drag.py`), and live flight cannot run here yet. Synthetic unit tests for the solver and EKF do run without those assets:

```bash
python -m pytest test_gate_ls_pose.py test_drag_ekf.py -q
```

## Restore checklist

1. Put the simulator back under `AIGP_3385/` (or update paths).
2. Restore `winvenv/` (or recreate) and YOLO weights under `models/`.
3. Optionally restore `frames/` and run `python tools/eval_gate_pose.py` before trusting range at speed.
4. Fly one lap, then `python tools/identify_drag.py --telem logs/telem_....csv` and set `DRAG_KX` / `DRAG_KY`.
5. `python tools/build_course_map.py --telem ...` to replace the stub `course_map.json`.
