"""Track and persist the best / fastest pilot run from events logs."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
BEST_DIR = LOGS / "best"
BEST_META = BEST_DIR / "best_run.json"


# Round-1 course ends at gate 17. Stale RACE_FINISH lines from a prior
# finished race used to mark short DNFs as "finished" and steal the best slot.
COURSE_LAST_GATE = 17


def score_events_text(text: str, *, stem: str = "") -> dict[str, Any]:
    gates: list[tuple[float, int]] = []
    # gate-17 crossings with sim race clock: (wall_t, last_gate_time)
    gate17: list[tuple[float, float]] = []
    finish_candidates: list[tuple[float, float]] = []  # (wall_t, finish_time)
    for line in text.splitlines():
        m = re.search(r"^\s*([\d.]+)\s+GATE_PASSED\s+gate=(\d+)", line)
        if m:
            wall_t, gate = float(m.group(1)), int(m.group(2))
            gates.append((wall_t, gate))
            if gate == COURSE_LAST_GATE:
                m_lgt = re.search(r"last_gate_time=([\d.]+)", line)
                if m_lgt:
                    gate17.append((wall_t, float(m_lgt.group(1))))
        m = re.search(
            r"^\s*([\d.]+)\s+RACE_FINISH\s+finish_time=([\d.]+)", line
        )
        if m:
            finish_candidates.append((float(m.group(1)), float(m.group(2))))
    max_gate = max((g for _, g in gates), default=-1)

    # A multi-attempt log can contain several full finishes. Prefer the
    # *fastest* validated sim finish_time (must match a gate-17 last_gate_time).
    validated_finishes: list[float] = []
    for wall17, lgt in gate17:
        matched = None
        for wall_f, ft in finish_candidates:
            if wall_f + 1e-3 >= wall17 and abs(float(ft) - float(lgt)) <= 0.05:
                matched = float(ft)
                break
        if matched is None:
            # Accept the gate-17 race clock alone when finish spam is missing.
            matched = float(lgt)
        validated_finishes.append(matched)

    finish_t: Optional[float] = (
        min(validated_finishes) if validated_finishes else None
    )

    # Prefer sim race clock on the best finish; else last sighting of max_gate.
    time_to_max: Optional[float] = finish_t
    if time_to_max is None and max_gate >= 0:
        for line in text.splitlines():
            m = re.search(
                rf"GATE_PASSED\s+gate={max_gate}\s+last_gate_time=([\d.]+)",
                line,
            )
            if m:
                time_to_max = float(m.group(1))
        if time_to_max is None and gates:
            time_to_max = next(
                (t for t, g in gates if g == max_gate), gates[-1][0]
            )

    return {
        "stem": stem,
        "max_gate": max_gate,
        "gates_passed": sorted({g for _, g in gates}),
        "n_gates": len({g for _, g in gates}),
        "finished": finish_t is not None,
        "finish_time_s": finish_t,
        "time_to_max_gate_s": time_to_max,
        "all_finish_times_s": validated_finishes,
        "gate_events": [{"t": t, "gate": g} for t, g in gates],
    }


def score_events_file(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    stem = path.stem.replace("events_", "")
    return score_events_text(
        path.read_text(encoding="utf-8", errors="replace"), stem=stem
    )


def _really_finished(score: dict[str, Any]) -> bool:
    """Full-course finish only (ignores stale short-run RACE_FINISH metas)."""
    return bool(score.get("finished")) and int(
        score.get("max_gate", -1)
    ) >= COURSE_LAST_GATE


def is_better(new: dict[str, Any], old: Optional[dict[str, Any]]) -> bool:
    """Prefer finished + faster; else farther gate; else faster to that gate."""
    if old is None:
        return new.get("max_gate", -1) >= 0
    nf, of_ = _really_finished(new), _really_finished(old)
    if nf and not of_:
        return True
    if of_ and not nf:
        return False
    if nf and of_:
        nt = new.get("finish_time_s")
        ot = old.get("finish_time_s")
        if nt is not None and ot is not None:
            return float(nt) < float(ot)
        return False
    ng, og = int(new.get("max_gate", -1)), int(old.get("max_gate", -1))
    if ng > og:
        return True
    if ng < og:
        return False
    nt = new.get("time_to_max_gate_s")
    ot = old.get("time_to_max_gate_s")
    if nt is not None and ot is not None:
        return float(nt) < float(ot)
    return False


def load_best() -> Optional[dict[str, Any]]:
    if not BEST_META.exists():
        return None
    try:
        return json.loads(BEST_META.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def maybe_save_best(
    *,
    events_stem: str,
    pilot_csv: Optional[Path] = None,
    label: str = "pilot",
) -> Optional[dict[str, Any]]:
    """If this run beats the stored best, copy artifacts under logs/best/."""
    events = LOGS / f"events_{events_stem}.txt"
    score = score_events_file(events)
    if score is None:
        return None
    score["label"] = label
    old = load_best()
    if not is_better(score, old):
        print(
            f"[BEST] not a record - max_gate={score['max_gate']} "
            f"finish={score.get('finish_time_s')} "
            f"(best max={old.get('max_gate') if old else None} "
            f"finish={old.get('finish_time_s') if old else None})",
            flush=True,
        )
        return None

    BEST_DIR.mkdir(parents=True, exist_ok=True)
    # Clear previous copies of these names, then copy.
    for pattern in ("events_best.*", "pilot_best.*", "telem_best.*"):
        for p in BEST_DIR.glob(pattern):
            try:
                p.unlink()
            except OSError:
                pass
    shutil.copy2(events, BEST_DIR / "events_best.txt")
    telem = LOGS / f"telem_{events_stem}.csv"
    if telem.exists():
        shutil.copy2(telem, BEST_DIR / "telem_best.csv")
    if pilot_csv is not None and Path(pilot_csv).exists():
        shutil.copy2(pilot_csv, BEST_DIR / "pilot_best.csv")
    else:
        # Heuristic: matching tuning CSV.
        cand = LOGS / "tuning" / f"pilot_{events_stem}.csv"
        if cand.exists():
            shutil.copy2(cand, BEST_DIR / "pilot_best.csv")
            score["pilot_csv"] = str(cand)
    score["events"] = str(events)
    score["saved_at"] = events_stem
    BEST_META.write_text(
        json.dumps(score, indent=2) + "\n", encoding="utf-8"
    )
    fin = score.get("finish_time_s")
    if fin is not None:
        summary = f"FINISHED {float(fin):.3f}s"
    else:
        summary = f"DNF t_max={score.get('time_to_max_gate_s')}s"
    print(
        f"[BEST] NEW RECORD saved -> logs/best/  "
        f"gates=0..{score['max_gate']} (n={score['n_gates']}) {summary}",
        flush=True,
    )
    return score


def latest_events_stem() -> Optional[str]:
    best = None
    bm = -1.0
    for p in LOGS.glob("events_*.txt"):
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if m > bm:
            bm = m
            best = p.stem.replace("events_", "")
    return best
