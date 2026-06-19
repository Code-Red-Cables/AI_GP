"""Telemetry + decision logging for offline tuning (see PLAN.md 8.7 / Task E).

The sim is deterministic, so a per-run log of telemetry, vision estimates, the
planner's targets AND the controller's actual attitude/thrust commands makes runs
reproducible and diffable while tuning gains. Runs as its own daemon-style thread
following the repo's create_*/get_thread_for_join convention. Logging is data-only
(no human interaction), so it is safe in timed runs.

Two files are written every run:

* ``logs/latest.jsonl`` — TRUNCATED at the start of every run (the "restarts every
  run" debug log). This is the file to read when debugging the most recent flight.
* ``logs/run_<unix>.jsonl`` — a timestamped archive kept for comparing tuning runs
  (the course is deterministic, so same gains => same trajectory).

Both share an identical format:

* Line 1 is a ``{"_meta": true, ...}`` header recording the run start time and the
  *active tuning constants* (gains) that produced the run — so a log is always
  self-describing and you never have to guess which gains were live.
* Every subsequent line is one flattened, human-readable sample (see ``_LEGEND``).
  All angles are radians, positions/velocities metres & m/s in world NED
  (z is negative-up), thrust is 0..1. ``cmd_*`` are what we commanded the FC;
  ``att``/``rate`` are what the FC actually did — the gap is the wobble.
"""

import json
import math
import os
import threading
import time

# Field legend, embedded in every log header so the file self-documents.
_LEGEND = {
    "t": "seconds since run start",
    "armed": "drone armed flag (from HEARTBEAT)",
    "dry_run": "True => commands computed+logged but NOT sent",
    "src": "planner target source: waypoint name (e.g. takeoff, cornerB)|alt_guard|watchdog_hover|no_mission",
    "gate_idx": "race.active_gate_index (sim-reported; unused by the preplanned planner)",
    "range_m": "planner's distance to the active gate (m)",
    "alt": "height above arm point (m) = -pos_z",
    "pos": "measured position NED [x,y,z] (m)",
    "vel": "measured velocity NED [vx,vy,vz] (m/s)",
    "att": "measured attitude [roll,pitch,yaw] (rad)",
    "rate": "measured body rates [rollspeed,pitchspeed,yawspeed] (rad/s) -- wobble metric",
    "cmd_att": "commanded attitude [roll,pitch,yaw] (rad) sent to the FC",
    "cmd_thr": "commanded collective thrust (0..1)",
    "cmd_rate": "commanded body rates [roll,pitch,yaw] (rad/s); compare to measured rate[]",
    "cmd_yawrate": "commanded body yaw rate (rad/s); compare to rate[2]",
    "mode": "FC-reported flight mode name (must be a rate/ACRO or self-level mode we can fly)",
    "att_err": "cmd_att - att [roll,pitch,yaw] (rad) -- tracking error",
    "vel_cmd": "planner's commanded velocity NED [vn,ve,vd] (m/s)",
    "yaw_tgt": "planner's raw desired yaw before slew-limiting (rad)",
    "vis": "[detected(0/1), confidence, range_m, method] from perception",
    "col": "[collision_id, threat_level, impulse] if a collision is fresh (<1s) else null",
}


def _gains_snapshot():
    """Best-effort capture of the live tuning constants (controller + planner)."""
    g = {}
    try:
        import controller as c
        g.update({
            "CONTROL_HZ": c.CONTROL_HZ,
            "HOVER_THRUST": c.HOVER_THRUST,
            "KP_THRUST": c.KP_THRUST,
            "THRUST_MIN": c.THRUST_MIN,
            "THRUST_MAX": c.THRUST_MAX,
            "KP_LEAN": c.KP_LEAN,
            "MAX_LEAN_DEG": round(math.degrees(c.MAX_LEAN_RAD), 2),
        })
        g["KP_YAW"] = c.KP_YAW
        g["YAW_RATE_MAX_DEG"] = round(math.degrees(c.YAW_RATE_MAX), 1)
        for name in ("LEAN_LPF_ALPHA",):
            if hasattr(c, name):
                g[name] = getattr(c, name)
    except Exception as e:  # pragma: no cover - keep logging alive regardless
        g["controller_import_error"] = str(e)
    try:
        import planner as p
        g.update({
            "MAX_SPEED": p.MAX_SPEED,
            "MAX_VSPEED": p.MAX_VSPEED,
            "KP_POS": p.KP_POS,
            "MAX_ALT_M": p.MAX_ALT_M,
        })
        import mission as m
        g.update({
            "ARRIVE_RADIUS_M": m.DEFAULT_ARRIVE_RADIUS_M,
            "YAW_TOL_DEG": round(math.degrees(m.DEFAULT_YAW_TOL_RAD), 1),
            "DWELL_S": m.DEFAULT_DWELL_S,
        })
    except Exception as e:  # pragma: no cover
        g["planner_import_error"] = str(e)
    try:
        import config as cfg
        if getattr(cfg, "USE_SPLINE", False):
            import spline_planner as sp
            g.update({
                "CRUISE_SPEED": cfg.CRUISE_SPEED,
                "LOOKAHEAD_M": cfg.LOOKAHEAD_M,
                "KP_POS_END": sp.KP_POS_END,
                "SAMPLES_PER_SEG": sp.SAMPLES_PER_SEG,
            })
    except Exception as e:  # pragma: no cover
        g["spline_import_error"] = str(e)
    return g


def _r(x, n=4):
    """Round a float for compact logs; pass through None/non-numbers unchanged."""
    try:
        return round(float(x), n)
    except (TypeError, ValueError):
        return None


class Logger:

    LOG_KEYS = ('armed', 'flight_mode', 'attitude', 'position_ned', 'odometry', 'race',
                'vision', 'target', 'control', 'last_collision')

    LATEST_PATH = os.path.join('logs', 'latest.jsonl')

    def __init__(self, data, path=None, hz=30):
        self.data = data
        if 'lock' not in self.data:
            self.data['lock'] = threading.RLock()
        self.hz = hz
        self.start_ns = time.time_ns()
        # Timestamped archive (kept across runs) + fixed "latest" file (truncated each run).
        self.path = path or os.path.join('logs', f'run_{int(time.time())}.jsonl')
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
        self.thread = None
        self.is_running = False

    @classmethod
    def create_logger(cls, data, **kwargs):
        lg = cls(data, **kwargs)
        lg.thread = threading.Thread(target=lg._log_loop, daemon=False)
        lg.is_running = True
        lg.thread.start()
        print(f"Logging run to {lg.path} (and {cls.LATEST_PATH}, truncated each run)", flush=True)
        return lg

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    def _header(self):
        return {
            "_meta": True,
            "run_start": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_ns / 1e9)),
            "start_ns": self.start_ns,
            "hz": self.hz,
            "gains": _gains_snapshot(),
            "legend": _LEGEND,
            "units": "angles=rad, pos=m NED (z neg-up), vel=m/s, thrust=0..1",
        }

    def _snapshot(self):
        with self.data['lock']:
            return {k: self.data.get(k) for k in self.LOG_KEYS}

    def _record(self, snap):
        """Flatten one shared_data snapshot into a compact, descriptive sample."""
        now = time.time_ns()
        att = snap.get('attitude') or {}
        pos = snap.get('position_ned') or {}
        odo = snap.get('odometry') or {}
        race = snap.get('race') or {}
        vis = snap.get('vision') or {}
        ctrl = snap.get('control') or {}
        cmd = ctrl.get('cmd') or {}
        col = snap.get('last_collision') or {}

        # Position/velocity: prefer LOCAL_POSITION_NED, fall back to ODOMETRY.
        if pos:
            px, py, pz = pos.get('x'), pos.get('y'), pos.get('z')
            vx, vy, vz = pos.get('vx'), pos.get('vy'), pos.get('vz')
        elif odo:
            px, py, pz = odo.get('pos', (None, None, None))
            vx, vy, vz = odo.get('vel', (None, None, None))
        else:
            px = py = pz = vx = vy = vz = None
        alt = (-pz) if pz is not None else None

        # Measured attitude (fall back to odometry quaternion only if no ATTITUDE msg).
        roll, pitch, yaw = att.get('roll'), att.get('pitch'), att.get('yaw')

        # Commanded attitude/thrust from the controller (may be absent in DRY_RUN startup).
        c_roll, c_pitch, c_yaw = cmd.get('roll'), cmd.get('pitch'), cmd.get('yaw')

        def _err(c, m):
            if c is None or m is None:
                return None
            # shortest signed angular difference (handles +/-pi wrap on yaw)
            return _r(math.atan2(math.sin(c - m), math.cos(c - m)))

        vel_cmd = ctrl.get('vel_cmd_ned')

        collision = None
        if col and col.get('ts') and (now - col['ts']) < 1_000_000_000:
            collision = [col.get('collision_id'), col.get('threat_level'), _r(col.get('impulse'))]

        return {
            "t": round((now - self.start_ns) / 1e9, 3),
            "armed": snap.get('armed'),
            "mode": snap.get('flight_mode'),
            "dry_run": ctrl.get('dry_run'),
            "src": ctrl.get('source'),
            "gate_idx": race.get('active_gate_index'),
            "range_m": _r(ctrl.get('range_m'), 2),
            "alt": _r(alt, 2),
            "pos": [_r(px, 2), _r(py, 2), _r(pz, 2)],
            "vel": [_r(vx, 2), _r(vy, 2), _r(vz, 2)],
            "att": [_r(roll), _r(pitch), _r(yaw)],
            "rate": [_r(att.get('rollspeed')), _r(att.get('pitchspeed')), _r(att.get('yawspeed'))],
            "cmd_att": [_r(c_roll), _r(c_pitch), _r(c_yaw)],
            "cmd_thr": _r(cmd.get('thrust'), 3),
            "cmd_rate": [_r(cmd.get('roll_rate')), _r(cmd.get('pitch_rate')), _r(cmd.get('yaw_rate'))],
            "cmd_yawrate": _r(cmd.get('yaw_rate')),
            "att_err": [_err(c_roll, roll), _err(c_pitch, pitch), _err(c_yaw, yaw)],
            "vel_cmd": [_r(v, 3) for v in vel_cmd] if vel_cmd else None,
            "yaw_tgt": _r(ctrl.get('yaw_target')),
            "vis": [int(bool(vis.get('detected'))), _r(vis.get('confidence'), 2),
                    _r(vis.get('range_m'), 2), vis.get('method')] if vis else None,
            # Raw body-frame gate vector (x-fwd,y-right,z-down) + its azimuth-off-nose
            # (deg). az should be ~0 for a centred gate REGARDLESS of lean; a swinging az
            # while the gate is centred pinpoints bearing corruption at the source.
            "gb": [_r(c, 2) for c in vis.get('gate_body')] if vis and vis.get('gate_body') else None,
            "gb_az": _r(math.degrees(math.atan2(vis['gate_body'][1], vis['gate_body'][0])), 1)
                     if vis and vis.get('gate_body') else None,
            "col": collision,
        }

    def _log_loop(self):
        dt = 1.0 / self.hz
        header_line = json.dumps(self._header()) + '\n'
        # 'w' truncates latest.jsonl every run; the archive is a fresh file each run.
        with open(self.path, 'w', buffering=1) as archive, \
                open(self.LATEST_PATH, 'w', buffering=1) as latest:
            archive.write(header_line)
            latest.write(header_line)
            while self.is_running:
                try:
                    line = json.dumps(self._record(self._snapshot())) + '\n'
                    archive.write(line)
                    latest.write(line)
                except Exception:
                    pass
                time.sleep(dt)
