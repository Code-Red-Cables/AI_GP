# Spline waypoint following on derived position (`Q2_spline`)

Experimental branch off `Q2_kalman`. Adds a **vision-free** flight mode that
follows a pre-captured path using the EKF's own position estimate.

`FLIGHT_MODE=assist` is still the default and is untouched. Everything here is
additive: `FLIGHT_MODE=spline` opts in.

## The idea, and why it can work despite drift

The derived position drifts, so captured waypoints are wrong in absolute
terms. But if the *same estimator* produced them, the error is largely **common
mode** — the follower is wrong in the same direction the capture was, and the
two substantially cancel. The quantity that decides whether this flies is
therefore run-to-run **repeatability**, not absolute accuracy.

That also means the acceptance test needs no ground truth. `RACE_STATUS.active_gate`
only advances when a gate is actually passed, so "did it fly the course" is
directly observable on VQ2.

> **Capture and replay must use the same `EKF_USE_PNP` setting.**
> PnP corrections pull the frame toward a landmark whose world pose was itself
> initialised from the drone's own belief, so their timing and magnitude vary
> run to run. That is precisely the non-repeatable component. Pure dead
> reckoning (`EKF_USE_PNP=0`) is absolutely worse but more deterministic, which
> is usually the better trade here. Mismatch the two and nothing cancels.
>
> The capture file records which setting was used, in `ekf_use_pnp`.

## Capture

Fly the course by hand in the stabilised teleop and press **M** at each point
you want on the path:

```powershell
# with PnP corrections off, for a deterministic frame
$env:EKF_USE_PNP="0"
.\winvenv\Scripts\python.exe tools\tune_flight.py manual --capture
```

Keys are the usual hold-to-fly set — `W/S` pitch, `A/D` roll, `Q/E` yaw,
`R/F` climb/sink (a rate command, so release brakes to zero), `Space` level,
`M` mark a waypoint, `Esc`/`X` quit. Waypoints are written on exit to
`SPLINE_CAPTURE_PATH` (default `captured_waypoints.json`).

Mark generously through corners — the spline is centripetal Catmull-Rom
through the marks, so corner shape follows from how densely you mark them.
Two waypoints is the minimum.

## Replay

```powershell
$env:EKF_USE_PNP="0"          # must match the capture
$env:FLIGHT_MODE="spline"
.\winvenv\Scripts\python.exe main.py
```

## Repeatability test — the one that matters

Replay the same file twice and compare. If the two runs agree to well under a
metre at each gate and `active_gate` reaches the finish both times, the
approach works. If it stalls at the same gate every time, that is your drift
limit and its location. Run it with `EKF_USE_PNP` both `0` and `1` and keep
whichever is more repeatable — that is an empirical question, not an
architectural one.

## Tuning

| Knob | Default | Effect |
|---|---|---|
| `SPLINE_CRUISE_MPS` | 2.0 | speed cap; the profile slows for corners below this |
| `SPLINE_A_LAT` | 4.0 | lateral accel budget — lower slows corners more |
| `SPLINE_A_LON` | 2.5 | braking budget ahead of corners and the finish |
| `SPLINE_KP_VEL_LEAN` | 0.09 | velocity error (m/s) → lean (rad). Raise to hold speed, but it is P-only so expect steady-state error |
| `SPLINE_LOOKAHEAD_M` / `_TIME_S` / `_MAX_M` | 1.5 / 0.6 / 4.0 | carrot distance. Longer = smoother, cuts corners more |
| `SPLINE_KP_YAW` | 1.2 | heading onto the path tangent |
| `SPLINE_MAX_LEAN_DEG` | 12 | lean authority |
| `SPLINE_VERT_AUTH` | 0.08 | thrust the vertical rate loop may add |
| `SPLINE_MAX_XTE_M` | 6.0 | cross-track fail-safe → hover |
| `SPLINE_MAX_ALT_M` | 8.0 | altitude fail-safe → hover |

Attitude and hover trim are inherited from `Q2_kalman` and are *not* re-tuned
here: `KALMAN_KP_ATT`, `KALMAN_KD_ATT`, `HOVER_THRUST`.

## Offline test

```
python test_spline_mission.py
```

18 tests, no simulator: path geometry, capture→JSON→load round trip, and the
planner flown closed-loop against a kinematic plant including every fail-safe.

## Design notes

`spline_planner.py` emits the **rate + thrust** contract (`kalman: True`), the
same one assist and kalman use, so it runs on the plant that has actually been
tuned. It deliberately avoids the controller's velocity-fallback branch, whose
`KP_ATT` / `KP_ROLL_ATT` gains are untuned on this branch.

`compute_target(shared_data, dt=None)` takes an optional `dt`. Without it the
loop uses wall-clock, which makes it impossible to test against a simulated
clock — the derivative term sees a real-time interval while the plant advances
by a fixed step, saturating the rate clamp. Tests pass `dt` explicitly.

`planning/spline_path.py` is the geometry from `origin/spline-path`, ported
unchanged in substance with the `config` imports stripped so it stays pure and
unit-testable. `mission.py` came across as-is.

## Known limitations

- The course has no loops, so drift is monotonic and worst at the finish. The
  last gates are where this fails first.
- Captured waypoints inherit the drift at the moment of capture. Cancellation
  is the only thing making that acceptable.
- If a replay diverges — a clipped gate, a saturated command — the symmetry
  breaks and the drift is no longer the captured drift, with no loop closure to
  recover. Worth considering a PnP-corrected position running in parallel
  purely as a divergence monitor, without letting it perturb the flown frame.
- `SPLINE_KP_VEL_LEAN` is P-only, so cruise speed settles below the commanded
  value. Fine for a first test; add an integrator if the speed error matters.
