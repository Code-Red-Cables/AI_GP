---
name: attitude-gains-tuned-2026-06-03
description: Controller attitude/thrust gains were tuned after first live sim test; all doc references updated from stubs to real values
metadata:
  type: project
---

**Completed 2026-06-03.** After first live flight test, attitude control gains in `controller.py` were adjusted based on observed behavior:

**Changes made:**
- `HOVER_THRUST`: 0.5 → 0.35 (was above true hover throttle, causing steady ~5.5 m/s climb; logs/run_1780521287.jsonl)
- `KP_THRUST`: 0.05 → 0.15 (needed stronger vertical authority to overcome climb and drive descent)
- `THRUST_MIN`: 0.1 → 0.05 (allow near-zero thrust so alt_guard descent command has range to work)
- `KP_LEAN`, `MAX_LEAN_RAD`, `THRUST_MAX`: unchanged (0.15, 20°, 0.9)

**Docs updated:**
1. `docs/CALIBRATION.md`: Updated attitude-gain tuning table (§5a) with new defaults; expanded Note on MAX_VSPEED/MAX_ALT_M (§5b) to include historical context and evidence.
2. `docs/IMPLEMENTATION.md`: Line 124 tunable-gains list updated with new values (0.35, 0.15, 0.05).
3. `PLAN.md`: Three locations updated:
   - §8 blockquote (line 251–252): new constants
   - §8.0 table (line 290): tunable-gains specs now include values
   - §8.4 (line 379–386): thrust formula and gains section expanded with rationale
   - §8.8 (line 454–461): bug-fix narrative updated with live-test observation
4. `reference/VERIFY.md`: Failure signals section (line 125–130) expanded with first-live-test context and current starting values.

**Notes:**
- Old values (0.5, 0.05 KP_THRUST) appear *only* in historical/narrative sections explaining what was changed and why—never in current prescriptive statements.
- All "current values," "defaults," "starting values," and "TUNE FIRST" directives now reference 0.35, 0.15, 0.05.
- The sym run log `logs/run_1780521287.jsonl` is cited as evidence of the climb problem; this is a verifiable artifact.
- `CLAUDE.md` and `main.py` already correctly describe the problem (mention "HOVER_THRUST must be tuned") but don't commit to specific old values, so no changes needed there.

**Next tuning cycle:** if another flight test adjusts these further, update all five files again using same strategy (preserve narrative, update prescriptive statements).
