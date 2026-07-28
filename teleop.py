"""Keyboard teleop planner (Windows + POSIX).

Controls:
    W / S       forward / back
    A / D       strafe left / right
    Space / C   climb / descend
    Q / E       yaw left / right
    X           emergency hover (clear axes)
"""
from __future__ import annotations

import math
import select
import sys
import threading
import time

import config

DECAY_S = 0.6


class _PosixKeyReader:
    """Raw-tty keyboard reader for Linux / macOS / WSL."""

    def __init__(self):
        import termios
        import tty

        self._termios = termios
        self._tty = tty
        self._seen = {}
        self._lock = threading.Lock()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        fd = sys.stdin.fileno()
        old = self._termios.tcgetattr(fd)
        try:
            self._tty.setraw(fd)
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if ready:
                    ch = sys.stdin.read(1)
                    if ch in ('\x03', '\x04'):
                        import os as _os
                        import signal
                        _os.kill(_os.getpid(), signal.SIGINT)
                        break
                    with self._lock:
                        self._seen[ch] = time.time()
        finally:
            self._termios.tcsetattr(fd, self._termios.TCSADRAIN, old)

    def held(self, key: str) -> bool:
        with self._lock:
            return (time.time() - self._seen.get(key, 0.0)) < DECAY_S

    def axis(self, pos: str, neg: str) -> int:
        if self.held(pos):
            return 1
        if self.held(neg):
            return -1
        return 0


class _WindowsKeyReader:
    """Console keyboard reader for native Windows (msvcrt)."""

    def __init__(self):
        import msvcrt

        self._msvcrt = msvcrt
        self._seen = {}
        self._lock = threading.Lock()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while True:
            if self._msvcrt.kbhit():
                raw = self._msvcrt.getwch()
                if raw in ('\x00', '\xe0'):
                    # Swallow arrow-key prefix + code
                    if self._msvcrt.kbhit():
                        self._msvcrt.getwch()
                    continue
                if raw in ('\x03',):
                    import os as _os
                    import signal
                    _os.kill(_os.getpid(), signal.SIGINT)
                    break
                ch = ' ' if raw == '\r' else raw.lower()
                with self._lock:
                    self._seen[ch] = time.time()
            else:
                time.sleep(0.01)

    def held(self, key: str) -> bool:
        with self._lock:
            return (time.time() - self._seen.get(key, 0.0)) < DECAY_S

    def axis(self, pos: str, neg: str) -> int:
        if self.held(pos):
            return 1
        if self.held(neg):
            return -1
        return 0


def _make_key_reader():
    if sys.platform.startswith('win'):
        return _WindowsKeyReader()
    return _PosixKeyReader()


class TeleopPlanner:
    name = 'teleop'

    def __init__(self, shared_data):
        self._kb = _make_key_reader()
        print(
            '[TELEOP] W/S=fwd/back  A/D=strafe  Space/C=up/down  Q/E=yaw  X=hover',
            flush=True,
        )
        print('[TELEOP] Click this terminal so it has focus, then fly. Ctrl+C=quit', flush=True)

    def compute_target(self, shared_data):
        shared_data['planner_mode'] = self.name
        kb = self._kb

        if kb.held('x'):
            shared_data['teleop_cmd'] = {
                'fwd': 0, 'right': 0, 'up': 0, 'yaw': 0,
            }
            return {'vn': 0.0, 've': 0.0, 'vd': 0.0, 'yaw_rate': 0.0}

        fwd = kb.axis('w', 's')
        right = kb.axis('d', 'a')
        up = kb.axis(' ', 'c')
        turn = kb.axis('e', 'q')

        shared_data['teleop_cmd'] = {
            'fwd': fwd, 'right': right, 'up': up, 'yaw': turn,
        }

        att = shared_data.get('attitude') or {}
        yaw = float(att.get('yaw', 0.0) or 0.0)
        if not math.isfinite(yaw):
            yaw = 0.0

        yaw_rate = math.radians(config.TELEOP_YAW_RATE_DPS) * turn
        v_fwd = fwd * config.TELEOP_SPEED
        v_right = right * config.TELEOP_SPEED
        vd = -up * config.TELEOP_VSPEED  # Space → climb → vd negative

        vn = math.cos(yaw) * v_fwd - math.sin(yaw) * v_right
        ve = math.sin(yaw) * v_fwd + math.cos(yaw) * v_right
        return {'vn': vn, 've': ve, 'vd': vd, 'yaw_rate': yaw_rate}
