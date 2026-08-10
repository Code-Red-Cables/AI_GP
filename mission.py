"""Preplanned waypoint mission for the deterministic AI-GP course.

This branch flies a FIXED, vision-free path: an ordered list of waypoints, each a target
position in world NED (metres; +x = North, +y = East, +z = Down, origin at the arm point,
so z is negative-up and 2 m altitude is d = -2) together with the yaw to hold there. The
planner drives the drone to each waypoint's position while turning to its yaw, then
advances to the next. This gives deterministic point-to-point flight with full control
over the orientation held at every ordered point -- no perception required.

Yaw convention (matches the sim's ATTITUDE yaw): ``0 = facing North (+x)``, increasing
toward East (+y), so ``+90 = East``, ``-90 = West``, ``180 = South``. A LEFT turn
(counter-clockwise viewed from above) DECREASES yaw by the shortest path.
"""

import json
import math
import os

# Default per-waypoint tolerances. A Waypoint may override any of these; leave the field
# None to fall back to the value here. The planner reads these defaults.
DEFAULT_ARRIVE_RADIUS_M = 0.35   # within this many metres of the target position = "there"
DEFAULT_YAW_TOL_RAD = math.radians(6.0)   # within this heading error = "facing the right way"
DEFAULT_DWELL_S = 0.4            # hold position+heading this long before advancing (settle/turn)


class Waypoint:
    """One ordered target: a world-NED position plus the heading to hold there.

    ``n, e, d``  -- target position (m) in world NED (x=North, y=East, z=Down).
    ``yaw_deg``  -- heading to face here (deg): 0=North, +90=East, -90=West, 180=South.
                    The drone turns toward this while flying to the position. Pass
                    ``None`` to HOLD the current heading (no commanded turn, and arrival
                    is not gated on heading) -- used for a straight-up takeoff.

    The optional ``arrive_radius`` (m), ``yaw_tol_deg`` (deg) and ``dwell_s`` (s) override
    the module defaults for this waypoint; leave them None to use the defaults.
    """

    def __init__(self, n, e, d, yaw_deg=0.0, *, arrive_radius=None,
                 yaw_tol_deg=None, dwell_s=None, name="",
                 active_gate=None, event=None, t=None):
        self.pos = (float(n), float(e), float(d))
        self.yaw = math.radians(float(yaw_deg)) if yaw_deg is not None else None
        self.arrive_radius = arrive_radius
        self.yaw_tol = math.radians(yaw_tol_deg) if yaw_tol_deg is not None else None
        self.dwell_s = dwell_s
        self.name = name or ""
        # Optional remember-path tags (from capture / GATE_PASSED).
        self.active_gate = (
            int(active_gate) if active_gate is not None else None
        )
        self.event = event
        self.t = float(t) if t is not None else None

    @property
    def yaw_deg(self):
        return math.degrees(self.yaw) if self.yaw is not None else None

    def __repr__(self):
        n, e, d = self.pos
        yaw_s = f"{self.yaw_deg:.0f}deg" if self.yaw is not None else "hold"
        return (f"Waypoint({self.name or '?'}: n={n:.2f} e={e:.2f} d={d:.2f} "
                f"yaw={yaw_s})")


class Mission:
    """An ordered list of Waypoints the planner flies in sequence.

    ``loop=False`` (default): hold a hover at the final waypoint when the mission
    completes. ``loop=True``: wrap back to the first waypoint and fly it again.
    """

    def __init__(self, waypoints, loop=False, name="mission", *,
                 ekf_use_pnp=None, keep_until_gate=None):
        self.waypoints = list(waypoints)
        self.loop = bool(loop)
        self.name = name
        self.ekf_use_pnp = ekf_use_pnp
        self.keep_until_gate = (
            int(keep_until_gate) if keep_until_gate is not None else None
        )

    def __len__(self):
        return len(self.waypoints)

    def __repr__(self):
        return f"Mission({self.name!r}, {len(self.waypoints)} wps, loop={self.loop})"


def square_mission(side_m=5.0, altitude_m=2.0, *, counter_clockwise=True,
                   start_n=0.0, start_e=0.0, loop=False, land=False):
    """Build a square course flown FACING FORWARD: take off, then trace a square while
    holding the boot heading the whole time (no turns).

    The drone arms at the origin, climbs straight up to ``altitude_m`` HOLDING its boot
    heading, then strafes around a ``side_m``-per-side square. EVERY waypoint has
    ``yaw=None``, so the nose stays pointed "forward" (the boot heading) for the entire
    flight: the drone translates -- flying sideways and backward along the legs -- instead
    of turning to face each one. There are no turn-in-place waypoints, just the four
    corners. Counter-clockwise corners (NED North/East):

        A=(0,0) -> B=(side,0) -> C=(side,-side) -> D=(0,-side) -> back to A

    Legs travelled: North, West, South, East -- but the heading held is constant (forward)
    throughout. ``counter_clockwise=False`` mirrors the East axis into a clockwise loop
    (same fixed heading; only the shape mirrors). ``land=True`` appends a slow descent to
    the ground at the end instead of holding a hover.
    """
    d = -abs(altitude_m)
    s = float(side_m)
    sgn = 1.0 if counter_clockwise else -1.0   # mirror the East axis for clockwise

    def wp(n, e, name, **kw):
        # yaw=None at every corner: hold the boot heading (face forward), never turn.
        return Waypoint(start_n + n, start_e + sgn * e, d, None, name=name, **kw)

    wps = [
        Waypoint(start_n, start_e, d, None, name="takeoff", arrive_radius=0.4),
        wp(s,    0.0, "cornerB"),    # strafe to corner B (travel North)
        wp(s,   -s,   "cornerC"),    # travel West
        wp(0.0, -s,   "cornerD"),    # travel South
        wp(0.0,  0.0, "cornerA"),    # travel East, back to the start corner
    ]
    if land:
        wps.append(Waypoint(start_n, start_e, -0.15, None, name="land",
                            arrive_radius=0.3, dwell_s=2.0))
    return Mission(wps, loop=loop,
                   name="square_ccw" if counter_clockwise else "square_cw")


def load_mission(path):
    """Load a mission from a JSON file, or return None if it is missing / unreadable.

    Schema (see :func:`save_mission`)::

        {"name": "...", "loop": false, "ekf_use_pnp"?, "keep_until_gate"?,
         "waypoints": [{"n":, "e":, "d":, "yaw_deg":,
                        "arrive_radius"?, "yaw_tol_deg"?, "dwell_s"?, "name"?,
                        "active_gate"?, "event"?, "t"?}, ...]}

    ``yaw_deg`` may be ``null`` to HOLD the current heading at that waypoint (no turn);
    omitting it defaults to 0 (face North).
    """
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return None
    wps = []
    for w in raw.get("waypoints", []):
        try:
            wps.append(Waypoint(
                w["n"], w["e"], w["d"], w.get("yaw_deg", 0.0),
                arrive_radius=w.get("arrive_radius"),
                yaw_tol_deg=w.get("yaw_tol_deg"),
                dwell_s=w.get("dwell_s"),
                name=w.get("name", ""),
                active_gate=w.get("active_gate"),
                event=w.get("event"),
                t=w.get("t"),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    if not wps:
        return None
    return Mission(
        wps,
        loop=bool(raw.get("loop", False)),
        name=raw.get("name", "mission"),
        ekf_use_pnp=raw.get("ekf_use_pnp"),
        keep_until_gate=raw.get("keep_until_gate"),
    )


def save_mission(mission, path):
    """Persist a mission to JSON in the schema :func:`load_mission` reads."""
    data = {
        "name": mission.name,
        "loop": mission.loop,
        "waypoints": [],
    }
    if getattr(mission, "ekf_use_pnp", None) is not None:
        data["ekf_use_pnp"] = int(mission.ekf_use_pnp)
    if getattr(mission, "keep_until_gate", None) is not None:
        data["keep_until_gate"] = int(mission.keep_until_gate)
    for w in mission.waypoints:
        row = {
            "n": w.pos[0],
            "e": w.pos[1],
            "d": w.pos[2],
            "yaw_deg": round(w.yaw_deg, 3) if w.yaw is not None else None,
            "name": w.name,
        }
        if w.active_gate is not None:
            row["active_gate"] = int(w.active_gate)
        if w.event:
            row["event"] = w.event
        if w.t is not None:
            row["t"] = round(float(w.t), 3)
        data["waypoints"].append(row)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _gate_index_from_name(name: str):
    """Parse gate number from names like gate1 / gate_1 / g1."""
    if not name:
        return None
    import re
    m = re.search(r"(?:gate|g)_?(\d+)$", str(name).strip(), re.I)
    if not m:
        return None
    return int(m.group(1))


def keep_until_gate(mission, gate: int) -> Mission:
    """Return a prefix mission through the named / tagged gate.

    Keeps waypoints up to and including the first match for ``gate``:
      1. ``event==gate_pass`` and ``active_gate==gate``, else
      2. ``active_gate==gate``, else
      3. name like ``gate{N}`` / ``gate_N``.

    Raises ``ValueError`` if that gate cannot be found (need a richer capture).
    """
    g = int(gate)
    if g < 1:
        raise ValueError("keep_until_gate requires gate >= 1")
    if mission is None or len(mission) < 2:
        raise ValueError("mission needs at least 2 waypoints")

    cut = None
    for i, w in enumerate(mission.waypoints):
        if (
            str(getattr(w, "event", "") or "") == "gate_pass"
            and w.active_gate == g
        ):
            cut = i
            break
    if cut is None:
        for i, w in enumerate(mission.waypoints):
            if w.active_gate == g:
                cut = i
                break
    if cut is None:
        for i, w in enumerate(mission.waypoints):
            if _gate_index_from_name(w.name) == g:
                cut = i
                break
    if cut is None:
        raise ValueError(
            f"no gate {g} marker in mission {mission.name!r} "
            f"({len(mission)} wps). Re-capture with GATE_PASSED tags "
            f"or name waypoints gate1, gate2, ..."
        )

    kept = list(mission.waypoints[: cut + 1])
    if len(kept) < 2:
        # Sparse gate-pass-only captures: synthesize a pad start so the
        # spline has an approach segment before the first gate mark.
        w0 = kept[0]
        start = Waypoint(
            0.0, 0.0, w0.pos[2],
            w0.yaw_deg if w0.yaw is not None else 0.0,
            name="start",
        )
        kept = [start, w0]
    return Mission(
        kept,
        loop=False,
        name=f"{mission.name}_until_g{g}",
        ekf_use_pnp=getattr(mission, "ekf_use_pnp", None),
        keep_until_gate=g,
    )
