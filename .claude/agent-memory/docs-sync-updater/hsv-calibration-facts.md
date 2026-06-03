---
name: hsv-calibration-facts
description: HSV gate-detection calibration values and documentation locations (Round 1, 2026-06-03)
metadata:
  type: reference
---

## Ground truth (source of authority)

**File:** `vision/gate_detector.py` (lines 25-28)

```python
LOWER_HSV  = (0, 150, 150)
UPPER_HSV  = (15, 255, 255)
LOWER_HSV2 = (170, 150, 150)
UPPER_HSV2 = (180, 255, 255)
```

**Gate color:** Red/orange (glowing), **NOT magenta/pink**

**Why two ranges:** OpenCV HSV hue is 0–179; red/orange wraps at 0/180 boundary. The two-piece mask (0–15 and 170–180) captures the full red range via bitwise OR in `_build_mask()`.

## Documentation locations where these values appear or are referenced

1. **CLAUDE.md (line 15)** — Executive summary of HSV calibration status
   - Mentions gate color, both HSV ranges, file location, and re-verification requirement
   
2. **docs/CALIBRATION.md (lines 71-82)** — Detailed calibration section §2
   - Shows all four constants with comments
   - Explains two-piece mask rationale
   - Notes detection-performance re-verification requirement (no longer asserts 90% — that claim is now unverifiable)
   
3. **docs/IMPLEMENTATION.md (lines 63-64)** — Module reference for gate_detector.py
   - Notes that thresholds are calibrated (not placeholders)
   - References CALIBRATION.md §2 for details

4. **reference/VERIFY.md (lines 39-42)** — Generic HSV tuning instructions
   - Says "paste the printed LOWER_HSV / UPPER_HSV" (still correct for either single or dual ranges)
   - No update needed unless the tuning workflow changes

5. **tools/hsv_tuner.py (lines 58-61)** — Prints `LOWER_HSV_1`, `UPPER_HSV_1`, `LOWER_HSV_2`, `UPPER_HSV_2`
   - This tool output format already supports dual ranges; no code change needed

## Stale values to search for and remove in future

- `(136, 0, 183)` and `(179, 255, 255)` — old magenta/pink range (now deleted from docs)
- `magenta` or `pink` in gate-color context (keep `magenta corners` which refers to visualization color)
- Claims of `90%` detection rate without context (was verified offline on 60 frames; need live re-verification)

## Detection re-verification status

**Prior claim:** 54/60 frames detected (90%), confidence 0.56–0.61, tight HSV mask (~0.5% of frame).

**Current status:** This claim is no longer verifiable by the docs-sync process (no access to reference/frames/ or test harness). The CALIBRATION.md §2 now says "Detection performance needs re-verification against reference/frames/ after deployment" rather than asserting the old number. This is intentional — calibration values can change, and a static "90%" claim becomes stale fast.
