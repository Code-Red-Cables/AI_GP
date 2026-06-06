---
name: planner-vertical-safety-fixes
description: Documentation update for MAX_VSPEED and MAX_ALT_M constants added to planner.py
metadata:
  type: project
---

## What changed in planner.py (2026-06-03)
Three key changes to combat vertical-loop runaway observed in run_1780515186.jsonl (586 m climb):

1. **MAX_VSPEED = 1.0 m/s** (NEW) — separate cap on the vertical component (after magnitude cap)
   - Rationale: up-tilted camera (20°) biases gate elevation upward; size-based range is noisy
   - Without this, aggressive climbs trigger vehicle's vertical-tracking overshoot
   
2. **MAX_ALT_M = 15.0 m** (NEW) — altitude ceiling above arm point
   - If drone exceeds this, planner abandons gate and descends at MAX_VSPEED
   - Client-side fail-safe that publishes `source='alt_guard'` in target
   
3. **Vertical-clamp logic** — line 175: `vd = max(-MAX_VSPEED, min(MAX_VSPEED, vel_ned[2]))`
   - Applied after magnitude clamp, only affects z-component
   - Keep `MAX_VSPEED < MAX_SPEED` per comment

## Documentation updates made
- **docs/CALIBRATION.md**: added MAX_VSPEED and MAX_ALT_M to §5 tuning table; detailed note on rationale + observed runaway
- **docs/IMPLEMENTATION.md**: updated Planner module description; added alt_guard to target['source'] list; mention vertical clamp in velocity production
- **docs/TESTING.md**: no changes (test doesn't exercise alt_guard, which requires high altitude)
- **CLAUDE.md**: updated Planner row in architecture table to mention altitude-envelope guard
- **PLAN.md §8 STATUS**: added "Recently fixed" and "Open / needs verification" sections; updated §8.8 guardrails with OFFBOARD flight-mode open item
- **reference/VERIFY.md §4**: added altitude envelope check to Step 4 verification runbook

## Important note: KP_POS constant mismatch
- Code has `KP_POS = 3.0` (set by prior commit)
- Docs had stale `KP_POS = 0.6` in CALIBRATION.md §5 table
- **Fixed** to match current code: `KP_POS = 3.0`

## Key cross-references in docs
- `shared_data['target']['source']` now has 6 values: `vision`, `vision_level`, `known`, `hover`, `watchdog_hover`, `alt_guard`
- Altitude envelope is a **fail-safe, not a fix** — root cause (vehicle vertical overshoot / missing OFFBOARD handshake) still open
- All changes auditable: `planner.py` lines 28–48, 146–154, 172–175
