"""Persistent course map for learn-then-replay preplanning (see PLAN.md / preplanning).

The sim does NOT broadcast the track geometry (spec VADR-TS-001 4.3 lists no track-data
message), so we cannot preplan from a sim-provided map. But the course is DETERMINISTIC
(spec 3.5: "course geometry is identical for all participants ... deterministic"), so a
map LEARNED on one run is valid on every later run.

This module stores the world (NED) position of each gate keyed by its gate index. The
planner records a gate's position the moment it is passed (using the vision-confirmed
world belief) and, on subsequent runs, flies toward the KNOWN next-gate position when it
is not yet in the narrow camera FOV -- instead of coasting blind and parking (the gate-2
hover failure).

Storage format (JSON), running-averaged across runs so repeated laps refine the estimate::

    {
      "gates": {
        "0": {"pos": [n, e, d], "n": 3},
        "1": {"pos": [n, e, d], "n": 2}
      }
    }

The map is thread-safe (a single RLock) because the planner records from the control
thread while main may save on shutdown.
"""

import json
import os
import threading
import time


class CourseMap:
    """A persistent, running-averaged map of gate index -> world NED position."""

    def __init__(self, path="course_map.json"):
        self.path = path
        self._lock = threading.RLock()
        # idx(int) -> {"pos": [n, e, d], "n": sample_count}
        self._gates = {}
        self._dirty = False

    # ------------------------------------------------------------------ load/save
    def load(self):
        """Load the map from disk if present. Missing/corrupt file -> empty map.

        Returns self so callers can write ``CourseMap(path).load()``.
        """
        with self._lock:
            if not os.path.exists(self.path):
                return self
            try:
                with open(self.path, "r") as f:
                    raw = json.load(f)
            except (OSError, ValueError):
                # Corrupt or unreadable: start empty rather than crash the client.
                return self
            gates = raw.get("gates", {}) if isinstance(raw, dict) else {}
            for k, v in gates.items():
                try:
                    if isinstance(v, dict):
                        pos, n = v.get("pos"), int(v.get("n", 1))
                    else:  # tolerate a bare [n,e,d] list
                        pos, n = v, 1
                    if pos and len(pos) == 3:
                        self._gates[int(k)] = {
                            "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                            "n": max(1, n),
                        }
                except (TypeError, ValueError):
                    continue
            return self

    def save(self):
        """Atomically persist the map (write temp + os.replace). No-op if unchanged."""
        with self._lock:
            if not self._dirty:
                return
            data = {
                "gates": {
                    str(k): {"pos": v["pos"], "n": v["n"]}
                    for k, v in self._gates.items()
                }
            }
            tmp = self.path + ".tmp"
            try:
                with open(tmp, "w") as f:
                    json.dump(data, f, indent=2, sort_keys=True)
                os.replace(tmp, self.path)
                self._dirty = False
            except OSError:
                # Best-effort persistence; never let a disk error take down the flight.
                pass

    # ------------------------------------------------------------------ access
    def record(self, idx, pos_ned):
        """Fold a gate's observed world NED position into the running average."""
        idx = int(idx)
        p = [float(pos_ned[0]), float(pos_ned[1]), float(pos_ned[2])]
        with self._lock:
            cur = self._gates.get(idx)
            if cur is None:
                print(f"LEARN: Gate {idx} recorded at {p}", flush=True)
                self._gates[idx] = {"pos": p, "n": 1}
            else:
                n = cur["n"]
                cur["pos"] = [(cur["pos"][i] * n + p[i]) / (n + 1) for i in range(3)]
                cur["n"] = n + 1
            self._dirty = True
            
        # Throttled auto-save (every 5 seconds max or on first discovery)
        now = time.time()
        if not hasattr(self, '_last_save'): self._last_save = 0
        if cur is None or (now - self._last_save) > 5.0:
            self.save()
            self._last_save = now

    def get(self, idx):
        """Return the averaged (n, e, d) tuple for a gate index, or None."""
        with self._lock:
            g = self._gates.get(int(idx))
            return tuple(g["pos"]) if g else None

    def indices(self):
        """Sorted list of known gate indices."""
        with self._lock:
            return sorted(self._gates)

    def has_any(self):
        with self._lock:
            return bool(self._gates)
