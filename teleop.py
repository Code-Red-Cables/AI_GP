import json
import math
import os
import select
import sys
import termios
import threading
import time
import tty

import config

DECAY_S    = 0.6    # key stays "held" for this long after last key-down event.
                    # Must be > key-repeat initial delay (~500ms on most systems).
WAYPOINT_FILE = 'captured_waypoints.json'


class _PosixKeyReader:
    """Raw-tty keyboard reader for Linux / WSL.

    Uses select() so the read loop never blocks permanently, and a time-decay
    model to convert discrete key-down events into a held-key state.
    """

    def __init__(self):
        self._seen  = {}   # char -> last-seen wall time
        self._lock  = threading.Lock()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r:
                    ch = sys.stdin.read(1)
                    if ch in ('\x03', '\x04'):   # Ctrl+C / Ctrl+D
                        import os as _os, signal
                        _os.kill(_os.getpid(), signal.SIGINT)
                        break
                    with self._lock:
                        self._seen[ch] = time.time()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def held(self, key: str) -> bool:
        with self._lock:
            return (time.time() - self._seen.get(key, 0.0)) < DECAY_S

    def axis(self, pos: str, neg: str) -> int:
        """Return +1 if pos held, -1 if neg held, 0 otherwise (pos wins ties)."""
        p = self.held(pos)
        n = self.held(neg)
        if p:
            return 1
        if n:
            return -1
        return 0

    def consume(self, key: str) -> bool:
        """Return True once for a key-down, clearing it immediately (one-shot)."""
        with self._lock:
            t = self._seen.get(key, 0.0)
            if (time.time() - t) < DECAY_S:
                self._seen[key] = 0.0
                return True
        return False


class TeleopPlanner:
    """Keyboard teleop planner.

    Controls:
        W / S       forward / back
        A / D       strafe left / right
        Space / C   climb / descend
        Q / E       yaw left / right
        B           capture waypoint (logs position + attitude to JSON)
        X           emergency hover (clear all axes)
    """
    name = 'teleop'

    def __init__(self, shared_data):
        self._kb = _PosixKeyReader()
        print('[TELEOP] W/S=fwd/back  A/D=strafe  Space/C=up/down  Q/E=yaw', flush=True)
        print('[TELEOP] B=capture waypoint  Ctrl+C=quit', flush=True)
        self._waypoints = []

    def compute_target(self, shared_data):
        shared_data['planner_mode'] = self.name
        kb = self._kb

        fwd   = kb.axis('w', 's')
        right = kb.axis('d', 'a')
        up    = kb.axis(' ', 'c')
        turn  = kb.axis('e', 'q')

        shared_data['teleop_cmd'] = {
            'fwd': fwd, 'right': right, 'up': up, 'yaw': turn,
        }

        if kb.consume('b'):
            self._capture_waypoint(shared_data)

        att = shared_data.get('attitude') or {}
        yaw = att.get('yaw', 0.0)

        yaw_rate = math.radians(config.TELEOP_YAW_RATE_DPS) * turn

        v_fwd   = fwd   * config.TELEOP_SPEED
        v_right = right * config.TELEOP_SPEED
        vd      = -up   * config.TELEOP_VSPEED   # Space (up=+1) → climb → vd negative

        # Body → NED
        vn =  math.cos(yaw) * v_fwd - math.sin(yaw) * v_right
        ve =  math.sin(yaw) * v_fwd + math.cos(yaw) * v_right

        return {'vn': vn, 've': ve, 'vd': vd, 'yaw_rate': yaw_rate}

    def _capture_waypoint(self, shared_data):
        att = shared_data.get('attitude') or {}
        wp = {
            'roll':  att.get('roll',  0.0),
            'pitch': att.get('pitch', 0.0),
            'yaw':   att.get('yaw',   0.0),
            'gate':  (shared_data.get('race_status') or {}).get('active_gate'),
            't':     time.time(),
        }
        self._waypoints.append(wp)
        with open(WAYPOINT_FILE, 'w') as f:
            json.dump(self._waypoints, f, indent=2)
        fn = shared_data.get('log_event')
        if fn:
            fn('WAYPOINT', f'#{len(self._waypoints)}  yaw={wp["yaw"]:.3f}rad')
        print(f'[TELEOP] waypoint #{len(self._waypoints)} captured', flush=True)
