"""Manual keyboard teleoperation (this branch) -- fly the drone by hand to map the course.

This branch swaps the autonomous waypoint :class:`~planner.Planner` for a
:class:`TeleopPlanner` driven by LIVE keyboard input, plus a one-key "capture this spot as
a waypoint" feature. You fly the course manually and press B at each gate to record its
position+heading in EXACTLY the JSON schema the preplanning branch flies
(:func:`mission.save_mission`) -- so a captured file can be dropped in as a ``mission.json``
on that branch to replay the path.

Controls (held continuously; horizontal motion is relative to where the NOSE points, i.e.
the camera view):

    W / S   forward / backward
    A / D   strafe left / right
    Space   climb (increase altitude)
    C       descend (decrease altitude)
    Q / E   yaw left / right (turn the nose)
    B       capture the drone's current NED pose as a waypoint and append it to the
            capture file (default ``captured_waypoints.json``), rewriting the whole list.

The planner emits the SAME ``{'mode':'velocity','vel_ned':(vn,ve,vd),'yaw':rad,...}``
contract the controller already consumes, so ``controller.py`` is untouched.

Input backends (auto-selected): on Windows the input thread polls true key STATE via
``GetAsyncKeyState`` (user32), so keys are read while HELD. On POSIX (offline testing only)
it falls back to non-blocking raw-tty char reads with a short hold-timeout.

NOTE on yaw handedness: the sim reports a LEFT-HANDED yaw, so physical nose-forward is
``(cos yaw, -sin yaw)`` in NED (see CLAUDE.md / controller.velocity_to_attitude). WASD is
mapped along that PHYSICAL forward/right so motion matches the camera. If Q/E turn the
opposite way to what you expect, flip ``TELEOP_YAW_SIGN``.
"""

import math
import sys
import threading
import time

from mission import Mission, Waypoint, save_mission
from config import TELEOP_SPEED, TELEOP_VSPEED, TELEOP_YAWRATE_DPS, TELEOP_YAW_SIGN

# --------------------------------------------------------------------------------------
# Teleop tuning lives in config.py: TELEOP_SPEED / TELEOP_VSPEED (full-stick horizontal /
# vertical speeds), TELEOP_YAWRATE_DPS (deg/s turn rate), TELEOP_YAW_SIGN (flip if Q/E
# turn the wrong way). Convert the yaw rate to rad/s for the controller here.
# --------------------------------------------------------------------------------------
TELEOP_YAW_RATE = math.radians(TELEOP_YAWRATE_DPS)   # rad/s turn rate while Q/E held

# Safety: no fresh pose for this long -> hover (we'd be flying blind). The altitude guard
# from the autonomous planner is intentionally dropped here -- the pilot has direct
# vertical control (Space/C) and an auto-descend would fight them.
TELEM_TIMEOUT_NS = 500_000_000


def _quat_yaw(q):
    """Yaw (rad, NED) from a quaternion ``(w, x, y, z)`` -- odometry-only fallback."""
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# --------------------------------------------------------------------------------------
# Keyboard backends
# --------------------------------------------------------------------------------------
class _WinKeyState:
    """True key-state polling via Win32 ``GetAsyncKeyState`` (Windows only).

    Reads whether each key is currently down regardless of console focus, so movement is
    continuous while a key is HELD.
    """

    VK = {'w': 0x57, 'a': 0x41, 's': 0x53, 'd': 0x44, 'q': 0x51, 'e': 0x45,
          'b': 0x42, 'space': 0x20, 'down': 0x43}   # 'down' = descend, the C key (0x43)

    def __init__(self):
        import ctypes
        self._get = ctypes.windll.user32.GetAsyncKeyState

    def poll(self):
        pass  # state is read on demand in down()

    def down(self, name):
        return bool(self._get(self.VK[name]) & 0x8000)

    def close(self):
        pass


class _PosixKeyReader:
    """Non-blocking raw-tty char reads with a hold-timeout (POSIX, offline testing).

    There is no key-up event on a terminal, so a key counts as "down" while its character
    keeps arriving (auto-repeat) and for ``HOLD_TIMEOUT`` after the last one.
    """

    HOLD_TIMEOUT = 0.25
    KEYMAP = {'w': 'w', 'a': 'a', 's': 's', 'd': 'd', 'q': 'q', 'e': 'e',
              'b': 'b', ' ': 'space', 'c': 'down'}

    def __init__(self):
        import termios
        import tty
        self._termios = termios
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)   # cbreak (not raw) keeps Ctrl-C delivering SIGINT
        self._last = {}

    def poll(self):
        import select
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1).lower()
            name = self.KEYMAP.get(ch)
            if name:
                self._last[name] = time.time()

    def down(self, name):
        t = self._last.get(name)
        return t is not None and (time.time() - t) <= self.HOLD_TIMEOUT

    def close(self):
        self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, self._old)


def make_key_backend():
    """Return the platform key backend (Win32 state poll, else POSIX raw-tty)."""
    if sys.platform.startswith('win'):
        return _WinKeyState()
    return _PosixKeyReader()


# --------------------------------------------------------------------------------------
# Background input thread: reads keys -> writes intent to shared_data['teleop'],
# and captures a waypoint on each B press (rising edge).
# --------------------------------------------------------------------------------------
class KeyboardTeleop:
    """Daemon thread that turns the keyboard into a normalized intent + B-capture.

    Writes ``shared_data['teleop'] = {'fwd','right','up','turn','ts'}`` where each axis is
    in [-1, 1] (held = full stick). Follows the repo's ``create_*`` / ``get_thread_for_join``
    lifecycle.
    """

    POLL_HZ = 60

    def __init__(self, data, capture_path='captured_waypoints.json'):
        self.data = data
        self.capture_path = capture_path
        self.thread = None
        self.is_running = False
        self._backend = None
        self._b_was_down = False
        self._captured = []   # list[Waypoint], rewritten to the file on every capture
        if 'lock' not in self.data:
            self.data['lock'] = threading.RLock()
        self._publish(0.0, 0.0, 0.0, 0.0)

    @classmethod
    def create_keyboard_teleop(cls, data, capture_path='captured_waypoints.json'):
        t = cls(data, capture_path)
        t.thread = threading.Thread(target=t._loop, daemon=True)
        t.is_running = True
        t.thread.start()
        return t

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    def _publish(self, fwd, right, up, turn):
        with self.data['lock']:
            self.data['teleop'] = {'fwd': fwd, 'right': right, 'up': up,
                                   'turn': turn, 'ts': time.time_ns()}

    def _loop(self):
        dt = 1.0 / self.POLL_HZ
        self._backend = make_key_backend()
        try:
            while self.is_running:
                self._backend.poll()
                d = self._backend.down
                fwd = (1.0 if d('w') else 0.0) - (1.0 if d('s') else 0.0)
                right = (1.0 if d('d') else 0.0) - (1.0 if d('a') else 0.0)
                up = (1.0 if d('space') else 0.0) - (1.0 if d('down') else 0.0)
                turn = (1.0 if d('e') else 0.0) - (1.0 if d('q') else 0.0)
                self._publish(fwd, right, up, turn)

                b = d('b')
                if b and not self._b_was_down:   # rising edge -> one capture per press
                    self._capture_waypoint()
                self._b_was_down = b

                time.sleep(dt)
        finally:
            try:
                self._backend.close()
            except Exception:
                pass

    def _capture_waypoint(self):
        """Append the drone's current NED pose as a Waypoint and rewrite the capture file."""
        with self.data['lock']:
            pos = self.data.get('position_ned')
            odo = self.data.get('odometry')
            att = self.data.get('attitude')
        if pos is not None:
            n, e, d = pos['x'], pos['y'], pos['z']
        elif odo is not None:
            n, e, d = odo['pos']
        else:
            print("[TELEOP] B ignored -- no position telemetry yet", flush=True)
            return
        if att is not None:
            yaw_deg = math.degrees(att['yaw'])
        elif odo is not None:
            yaw_deg = math.degrees(_quat_yaw(odo['q']))
        else:
            yaw_deg = 0.0

        idx = len(self._captured)
        self._captured.append(Waypoint(n, e, d, yaw_deg, name=f"wp{idx}"))
        try:
            save_mission(Mission(self._captured, loop=False, name="captured"),
                         self.capture_path)
            dest = self.capture_path
        except OSError as exc:
            dest = f"FAILED ({exc})"
        print(f"[TELEOP] captured #{idx + 1}: n={n:+.2f} e={e:+.2f} d={d:+.2f} "
              f"yaw={yaw_deg:+.0f}deg -> {dest}", flush=True)


# --------------------------------------------------------------------------------------
# Teleop planner: keyboard intent -> velocity setpoint (same contract as planner.Planner)
# --------------------------------------------------------------------------------------
class TeleopPlanner:
    """Maps live keyboard intent into a NED velocity + yaw setpoint each control tick.

    Reads ``shared_data['teleop']`` (written by :class:`KeyboardTeleop`) and the measured
    pose, then publishes ``shared_data['target']`` exactly as ``controller.py`` expects.
    WASD is mapped along the PHYSICAL nose direction ``(cos yaw, -sin yaw)`` so motion
    matches the camera; Q/E integrate a desired heading the controller turns toward.
    """

    def __init__(self, data):
        self.data = data
        if 'lock' not in self.data:
            self.data['lock'] = threading.RLock()
        self._desired_yaw = None     # heading setpoint, seeded to the measured yaw on tick 1
        self._last_t = None          # wall-clock of the previous tick (for yaw integration)

    @staticmethod
    def _wrap(a):
        return math.atan2(math.sin(a), math.cos(a))

    def _snapshot(self):
        with self.data['lock']:
            return {
                'attitude': self.data.get('attitude'),
                'odometry': self.data.get('odometry'),
                'position_ned': self.data.get('position_ned'),
                'teleop': self.data.get('teleop'),
            }

    @staticmethod
    def _pose(s):
        """Return ``(position tuple|None, yaw rad|None, telem_ts ns|None)``."""
        pos, odo, att = s['position_ned'], s['odometry'], s['attitude']
        position = ts = None
        if pos is not None:
            position = (pos['x'], pos['y'], pos['z'])
            ts = pos.get('ts')
        elif odo is not None:
            position = tuple(odo['pos'])
            ts = odo.get('ts')
        yaw = None
        if att is not None:
            yaw = att['yaw']
            ts = att.get('ts', ts)
        elif odo is not None:
            yaw = _quat_yaw(odo['q'])
        return position, yaw, ts

    def _publish(self, target):
        with self.data['lock']:
            self.data['target'] = target
        return target

    def _hover(self, yaw, source, now):
        return {'mode': 'velocity', 'vel_ned': (0.0, 0.0, 0.0),
                'yaw': float(yaw), 'source': source, 'ts': now}

    def compute_target(self):
        now = time.time_ns()
        s = self._snapshot()
        position, yaw, telem_ts = self._pose(s)
        yaw_now = yaw if yaw is not None else 0.0

        # Seed / advance the desired-heading integrator (turn intent * rate * dt).
        if self._desired_yaw is None:
            self._desired_yaw = yaw_now
        tele = s['teleop'] or {'fwd': 0.0, 'right': 0.0, 'up': 0.0, 'turn': 0.0}
        t_wall = time.time()
        dt = (t_wall - self._last_t) if self._last_t is not None else 0.0
        self._last_t = t_wall
        dt = max(0.0, min(dt, 0.1))   # clamp against scheduling hiccups
        self._desired_yaw = self._wrap(
            self._desired_yaw + TELEOP_YAW_SIGN * tele['turn'] * TELEOP_YAW_RATE * dt)

        # Watchdog: no recent pose -> hover (and don't drift the heading away).
        if position is None or telem_ts is None or (now - telem_ts) > TELEM_TIMEOUT_NS:
            self._desired_yaw = yaw_now
            return self._publish(self._hover(yaw_now, 'watchdog_hover', now))

        # Body-frame stick -> body velocity, magnitude-capped so a diagonal isn't faster.
        f_v = tele['fwd'] * TELEOP_SPEED
        r_v = tele['right'] * TELEOP_SPEED
        mag = math.hypot(f_v, r_v)
        if mag > TELEOP_SPEED:
            f_v *= TELEOP_SPEED / mag
            r_v *= TELEOP_SPEED / mag

        # Body (forward, right) -> world NED along the PHYSICAL nose (cos y, -sin y); this
        # is the inverse of the controller's body-frame velocity loop, so a "forward" stick
        # becomes +pitch toward where the camera looks. (Left-handed sim yaw -- see module
        # docstring.)
        c, snn = math.cos(yaw_now), math.sin(yaw_now)
        vn = c * f_v + snn * r_v
        ve = -snn * f_v + c * r_v
        vd = -tele['up'] * TELEOP_VSPEED   # +up stick climbs; NED down is +z

        return self._publish({'mode': 'velocity',
                              'vel_ned': (float(vn), float(ve), float(vd)),
                              'yaw': float(self._desired_yaw),
                              'source': 'teleop',
                              'ts': now})


def print_controls(capture_path):
    """Print the control scheme banner at startup."""
    print("\n" + "=" * 62, flush=True)
    print(" MANUAL CONTROL -- fly by keyboard, press B to map waypoints", flush=True)
    print("=" * 62, flush=True)
    print("   W / S : forward / backward      A / D : strafe left / right", flush=True)
    print("   Space : climb                   C     : descend", flush=True)
    print("   Q / E : yaw left / right         B     : capture waypoint", flush=True)
    print(f"   captured waypoints -> {capture_path}", flush=True)
    print("=" * 62 + "\n", flush=True)
