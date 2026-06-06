"""Quick digest of a run log produced by logger.py (the tuning debug loop).

Usage:
    python tools/analyze_log.py                 # analyses logs/latest.jsonl
    python tools/analyze_log.py logs/run_X.jsonl

Prints the active gains, a coarse timeline, and the metrics that matter for
diagnosing wobble / runaway: body-rate RMS (the wobble signature), vertical
tracking, attitude tracking error, lean/thrust saturation, gate progression
and any collisions. No third-party deps (stdlib only) so it runs anywhere.
"""

import json
import math
import os
import sys


def _load(path):
    header, rows = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("_meta"):
                header = rec
            else:
                rows.append(rec)
    return header, rows


def _nums(rows, key, idx=None):
    out = []
    for r in rows:
        v = r.get(key)
        if idx is not None:
            v = v[idx] if isinstance(v, list) and len(v) > idx else None
        if isinstance(v, (int, float)):
            out.append(v)
    return out


def _rms(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else 0.0


def _stat(xs):
    return (min(xs), sum(xs) / len(xs), max(xs)) if xs else (None, None, None)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("logs", "latest.jsonl")
    if not os.path.exists(path):
        print(f"No log at {path}")
        return 1
    header, rows = _load(path)
    print(f"=== {path} ===")
    if header:
        print(f"run_start={header.get('run_start')}  hz={header.get('hz')}  samples={len(rows)}")
        g = header.get("gains", {})
        print("gains: " + "  ".join(f"{k}={v}" for k, v in g.items()))
    if not rows:
        print("(no data rows)")
        return 0

    dur = rows[-1].get("t", 0) - rows[0].get("t", 0)
    armed = [r for r in rows if r.get("armed")]
    print(f"duration={dur:.1f}s   armed_samples={len(armed)}/{len(rows)}")

    # Restrict the interesting metrics to the armed (actually-flying) window.
    win = armed or rows

    # --- Wobble signature: body-rate RMS (rad/s) ---
    rr = _rms(_nums(win, "rate", 0))
    pr = _rms(_nums(win, "rate", 1))
    yr = _rms(_nums(win, "rate", 2))
    print(f"\nWOBBLE (body-rate RMS, armed): roll={rr:.2f}  pitch={pr:.2f}  yaw={yr:.2f} rad/s"
          "   (lower=calmer; yaw>>1 => yaw thrash)")

    # --- Attitude tracking error (cmd - measured) ---
    for i, name in enumerate(("roll", "pitch", "yaw")):
        lo, mean, hi = _stat([abs(x) for x in _nums(win, "att_err", i)])
        if mean is not None:
            print(f"  |att_err {name}|: mean={mean:.2f} max={hi:.2f} rad")

    # --- Altitude / vertical ---
    alt = _nums(win, "alt")
    vz = _nums(win, "vel", 2)
    lo, mean, hi = _stat(alt)
    if lo is not None:
        print(f"\nALT (m above arm): min={lo:.1f} mean={mean:.1f} max={hi:.1f}"
              f"   ceiling={header.get('gains',{}).get('MAX_ALT_M') if header else '?'}")
    lo, mean, hi = _stat(vz)
    if lo is not None:
        print(f"vz (m/s, +down): min={lo:.1f} mean={mean:.1f} max={hi:.1f}"
              "   (large magnitudes => vertical loop over/undershoot)")

    # --- Command saturation ---
    thr = _nums(win, "cmd_thr")
    lo, mean, hi = _stat(thr)
    if lo is not None:
        print(f"\ncmd_thr: min={lo:.2f} mean={mean:.2f} max={hi:.2f}")
    max_lean = math.radians((header or {}).get("gains", {}).get("MAX_LEAN_DEG", 20))
    leans = [abs(x) for x in _nums(win, "cmd_att", 0) + _nums(win, "cmd_att", 1)]
    if leans:
        sat = 100.0 * sum(1 for x in leans if x >= max_lean - 1e-3) / len(leans)
        print(f"lean cmd at cap: {sat:.0f}% of samples (cap={math.degrees(max_lean):.0f}deg)"
              "   (high => planner constantly demanding max lean)")

    # --- Source breakdown ---
    srcs = {}
    for r in win:
        srcs[r.get("src")] = srcs.get(r.get("src"), 0) + 1
    print("\nsource mix (armed): " + ", ".join(f"{k}={v}" for k, v in
                                                sorted(srcs.items(), key=lambda kv: -kv[1])))

    # --- Gate progression ---
    gis = [r.get("gate_idx") for r in rows if isinstance(r.get("gate_idx"), int)]
    if gis:
        print(f"gate_idx: start={gis[0]} end={gis[-1]} max={max(gis)}")

    # --- Vision health ---
    det = [r for r in rows if r.get("vis") and r["vis"][0]]
    print(f"vision detections: {len(det)}/{len(rows)} frames-with-target")

    # --- Collisions ---
    cols = [(r.get("t"), r.get("col")) for r in rows if r.get("col")]
    if cols:
        print(f"\nCOLLISIONS ({len(cols)}): first @ t={cols[0][0]}s -> {cols[0][1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
