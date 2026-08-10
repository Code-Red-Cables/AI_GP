"""Analog gamepad teleop for pilot/manual (Xbox XInput + pygame fallback).

  Left stick     → roll (X) / pitch (Y, up = forward)
  Right stick X  → yaw
  Right trigger  → climb (analog)
  Left trigger   → sink (analog)
  Right bumper   → extra collective (thrust bump while held)

Face buttons:

  A       → level
  B       → quit
  X       → HUMAN
  Y       → RESET run (sim cmd 31000 + re-arm)
  LB      → AUTO (assist on LOCK)
  Start   → KEEP (remember commit)
  D-pad ↓ → toggle client slow-mo (pair with CE at same scale)

Xbox pads prefer **XInput** (ctypes) so sticks still work while FlightSim is
open. pygame alone often sees the pad but reads all-zero axes when the sim
owns the controller.
"""
from __future__ import annotations

import ctypes
import math
import os
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class PadAxes:
    """Normalized commands in [-1, 1] plus edge-triggered buttons."""

    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    thrust: float = 0.0  # +climb (RT) / -sink (LT)
    thrust_bump: bool = False  # RB — extra collective
    boost: bool = False  # unused; kept for callers
    level: bool = False
    quit: bool = False
    auto: bool = False
    human: bool = False
    keep: bool = False
    reset: bool = False  # Y — mid-run sim reset + re-arm
    slowmo: bool = False  # D-pad ↓ — toggle client time scale
    name: str = ''


_PAD = None  # GamepadReader | XInputReader
_INIT_TRIED = False
_ANNOUNCED = False

# XInput button bits
_XIN_DPAD_UP = 0x0001
_XIN_DPAD_DOWN = 0x0002
_XIN_A = 0x1000
_XIN_B = 0x2000
_XIN_X = 0x4000
_XIN_Y = 0x8000
_XIN_START = 0x0010
_XIN_BACK = 0x0020
_XIN_LB = 0x0100
_XIN_RB = 0x0200


def _expo(x: float, expo: float) -> float:
    ax = abs(float(x))
    e = max(0.0, min(0.95, float(expo)))
    curved = (1.0 - e) * ax + e * (ax ** 3)
    return math.copysign(curved, x)


def _dead(x: float, dz: float) -> float:
    ax = abs(float(x))
    if ax <= dz:
        return 0.0
    return math.copysign((ax - dz) / max(1e-6, 1.0 - dz), x)


def _smooth_into(filt: dict, key: str, raw: float, alpha: float) -> float:
    prev = filt.get(key, 0.0)
    # Snap hard to zero whenever the raw axis is idle. A soft decay used to
    # leave 0.01–0.05 of residual for several ticks after a trigger release,
    # which was enough for pilot to arm and spool hover thrust unprompted.
    if abs(raw) < 1e-4:
        filt[key] = 0.0
        return 0.0
    out = (1.0 - alpha) * prev + alpha * raw
    filt[key] = out
    return out


class _XINPUT_GAMEPAD(ctypes.Structure):
    _fields_ = [
        ('wButtons', ctypes.c_ushort),
        ('bLeftTrigger', ctypes.c_ubyte),
        ('bRightTrigger', ctypes.c_ubyte),
        ('sThumbLX', ctypes.c_short),
        ('sThumbLY', ctypes.c_short),
        ('sThumbRX', ctypes.c_short),
        ('sThumbRY', ctypes.c_short),
    ]


class _XINPUT_STATE(ctypes.Structure):
    _fields_ = [
        ('dwPacketNumber', ctypes.c_ulong),
        ('Gamepad', _XINPUT_GAMEPAD),
    ]


class XInputReader:
    """Windows XInput — works for Xbox pads even when the sim is focused."""

    def __init__(self, index: int = 0):
        self._index = int(index)
        self._xinput = None
        for dll in ('xinput1_4.dll', 'xinput1_3.dll', 'xinput9_1_0.dll'):
            try:
                self._xinput = ctypes.WinDLL(dll)
                break
            except OSError:
                continue
        if self._xinput is None:
            raise RuntimeError('XInput DLL not found')
        self._state = _XINPUT_STATE()
        if self._xinput.XInputGetState(self._index, ctypes.byref(self._state)):
            raise RuntimeError(f'no XInput pad at index {self._index}')
        self.name = f'XInput#{self._index}'
        self._prev_btns = 0
        self._filt = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'thrust': 0.0}

    def _stick(self, v: int) -> float:
        # Dead-band the XInput center notch (~±7849 is common; we use soft dz later).
        if abs(int(v)) < 4000:
            return 0.0
        return max(-1.0, min(1.0, float(v) / 32767.0))

    def read(self, *, deadzone: float, expo: float, smooth: float) -> PadAxes:
        err = self._xinput.XInputGetState(self._index, ctypes.byref(self._state))
        if err:
            return PadAxes(name=self.name)
        g = self._state.Gamepad
        # Left stick = roll/pitch; right stick X = yaw. Vertical = triggers.
        roll = _expo(_dead(self._stick(g.sThumbLX), deadzone), expo)
        pitch = _expo(_dead(self._stick(g.sThumbLY), deadzone), expo)
        yaw = _expo(_dead(self._stick(g.sThumbRX), deadzone), expo)
        # Triggers 0..255 → climb (RT) / sink (LT). Dominant trigger wins
        # (do NOT subtract): a light RT rest value used to cancel LT sink.
        rt = max(0.0, min(1.0, float(g.bRightTrigger) / 255.0))
        lt = max(0.0, min(1.0, float(g.bLeftTrigger) / 255.0))
        if rt < 0.03:
            rt = 0.0
        if lt < 0.03:
            lt = 0.0
        # Linear triggers (no stick expo) — light pulls must still sink.
        if lt >= rt and lt > 0.0:
            thrust = -lt
        elif rt > 0.0:
            thrust = rt
        else:
            thrust = 0.0

        a = max(0.05, min(1.0, float(smooth)))
        # Triggers need a snappy filter; stick smooth left RT→LT residual climb.
        a_thr = max(a, 0.65)
        yaw = _smooth_into(self._filt, 'yaw', yaw, a)
        thrust = _smooth_into(self._filt, 'thrust', thrust, a_thr)
        roll = _smooth_into(self._filt, 'roll', roll, a)
        pitch = _smooth_into(self._filt, 'pitch', pitch, a)

        btns = int(g.wButtons)
        edges = btns & ~self._prev_btns
        self._prev_btns = btns
        auto = bool(edges & (_XIN_LB | _XIN_DPAD_UP))
        return PadAxes(
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            thrust=thrust,
            thrust_bump=bool(btns & _XIN_RB),
            level=bool(edges & _XIN_A),
            quit=bool(edges & _XIN_B),
            human=bool(edges & _XIN_X),
            auto=auto,
            keep=bool(edges & (_XIN_START | _XIN_BACK)),
            reset=bool(edges & _XIN_Y),
            slowmo=bool(edges & _XIN_DPAD_DOWN),
            name=self.name,
        )


class GamepadReader:
    """pygame joystick fallback (DualShock / when XInput unavailable)."""

    def __init__(self, index: int = 0):
        import pygame

        os.environ.setdefault('SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS', '1')
        if os.environ.get('SDL_VIDEODRIVER') is None:
            os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
        pygame.init()
        pygame.joystick.init()
        n = pygame.joystick.get_count()
        if n <= 0:
            raise RuntimeError('no gamepad connected')
        idx = max(0, min(int(index), n - 1))
        self._pg = pygame
        self._js = pygame.joystick.Joystick(idx)
        self._js.init()
        self.name = self._js.get_name() or f'pad{idx}'
        self._prev_btns: set[int] = set()
        self._prev_hat_down = False
        self._filt = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'thrust': 0.0}
        # Left stick roll/pitch (0/1), right stick X yaw (2). Triggers separate.
        self.ax_roll = int(os.environ.get('PILOT_PAD_AXIS_ROLL', '0'))
        self.ax_pitch = int(os.environ.get('PILOT_PAD_AXIS_PITCH', '1'))
        self.ax_yaw = int(os.environ.get('PILOT_PAD_AXIS_YAW', '2'))
        # DualSense: if axis2 is idle trigger, yaw is on axis 3.
        is_xbox = 'xbox' in self.name.lower() or 'x-box' in self.name.lower()
        if (
            not is_xbox
            and self._js.get_numaxes() >= 5
            and self.ax_yaw == 2
            and 'PILOT_PAD_AXIS_YAW' not in os.environ
        ):
            try:
                pygame.event.pump()
                if float(self._js.get_axis(2)) < -0.5:
                    self.ax_yaw = 3
            except Exception:
                pass

    def _axis(self, i: int) -> float:
        if i < 0 or i >= self._js.get_numaxes():
            return 0.0
        try:
            return float(self._js.get_axis(i))
        except Exception:
            return 0.0

    def _trigger(self, i: int) -> float:
        if i < 0 or i >= self._js.get_numaxes():
            return 0.0
        v = self._axis(i)
        if v < -0.2:
            return max(0.0, min(1.0, (v + 1.0) * 0.5))
        return max(0.0, min(1.0, v))

    def _btn_held(self) -> set[int]:
        cur = set()
        for i in range(self._js.get_numbuttons()):
            try:
                if self._js.get_button(i):
                    cur.add(i)
            except Exception:
                pass
        return cur

    def _btn_edges(self) -> set[int]:
        cur = self._btn_held()
        edges = cur - self._prev_btns
        self._prev_btns = cur
        return edges

    def read(self, *, deadzone: float, expo: float, smooth: float) -> PadAxes:
        self._pg.event.pump()
        # pygame: +Y is usually down — invert pitch. Vertical = triggers.
        roll = _expo(_dead(self._axis(self.ax_roll), deadzone), expo)
        pitch = _expo(_dead(-self._axis(self.ax_pitch), deadzone), expo)
        yaw = _expo(_dead(self._axis(self.ax_yaw), deadzone), expo)
        # Xbox pygame: axes 4=LT, 5=RT (rest −1). DualSense similar on 2/5.
        nax = self._js.get_numaxes()
        lt_i = 4 if nax > 5 else 2
        rt_i = 5 if nax > 5 else (4 if nax > 4 else -1)
        lt = self._trigger(lt_i) if lt_i >= 0 else 0.0
        rt = self._trigger(rt_i) if rt_i >= 0 else 0.0
        if lt < 0.03:
            lt = 0.0
        if rt < 0.03:
            rt = 0.0
        if lt >= rt and lt > 0.0:
            thrust = -lt
        elif rt > 0.0:
            thrust = rt
        else:
            thrust = 0.0
        a = max(0.05, min(1.0, float(smooth)))
        a_thr = max(a, 0.65)
        yaw = _smooth_into(self._filt, 'yaw', yaw, a)
        thrust = _smooth_into(self._filt, 'thrust', thrust, a_thr)
        roll = _smooth_into(self._filt, 'roll', roll, a)
        pitch = _smooth_into(self._filt, 'pitch', pitch, a)
        held = self._btn_held()
        edges = self._btn_edges()
        # Xbox/SDL: 0=A 1=B 2=X 3=Y 4=LB 5=RB; 6/7 often Back/Start.
        auto = 4 in edges
        hat_down = False
        try:
            if self._js.get_numhats() > 0:
                _hx, hy = self._js.get_hat(0)
                hat_down = int(hy) < 0
        except Exception:
            hat_down = False
        slowmo = bool(hat_down and not self._prev_hat_down)
        self._prev_hat_down = hat_down
        return PadAxes(
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            thrust=thrust,
            thrust_bump=5 in held,
            level=0 in edges,
            quit=1 in edges,
            auto=auto,
            human=2 in edges,
            keep=bool(edges & {6, 7, 8, 9}),
            reset=3 in edges,
            slowmo=slowmo,
            name=self.name,
        )


def get_gamepad(index: int = 0):
    """Prefer XInput (Xbox), else pygame. None if nothing works."""
    global _PAD, _INIT_TRIED, _ANNOUNCED
    if _PAD is not None:
        return _PAD
    if _INIT_TRIED:
        return None
    _INIT_TRIED = True
    if not int(float(os.environ.get('PILOT_GAMEPAD', '1') or 1)):
        return None

    errors = []
    # XInput first — required for Xbox while FlightSim also sees the pad.
    try:
        _PAD = XInputReader(index=index)
    except Exception as exc:
        errors.append(f'XInput:{exc}')
        _PAD = None
        try:
            _PAD = GamepadReader(index=index)
        except Exception as exc2:
            errors.append(f'pygame:{exc2}')
            _PAD = None

    if _PAD is None:
        print(f'[PAD] unavailable ({"; ".join(errors)}) — keyboard only',
              flush=True)
        return None

    if not _ANNOUNCED:
        _ANNOUNCED = True
        backend = 'XInput' if isinstance(_PAD, XInputReader) else 'pygame'
        print(
            f'[PAD] connected: {_PAD.name} via {backend}\n'
            '      L stick: roll / pitch     R stick X: yaw\n'
            '      RT climb   LT sink   RB extra thrust\n'
            '      A level  B quit  X human  Y reset  LB auto  Start keep\n'
            '      D-pad ↓ slow-mo toggle (match Cheat Engine scale)\n'
            '      Unbind the pad inside FlightSim so sticks reach XInput.',
            flush=True,
        )
    return _PAD


def read_pad_axes(
    *,
    deadzone: float | None = None,
    expo: float | None = None,
    smooth: float | None = None,
) -> Optional[PadAxes]:
    pad = get_gamepad()
    if pad is None:
        return None
    dz = (
        float(deadzone)
        if deadzone is not None
        else float(os.environ.get('PILOT_PAD_DEADZONE', '0.16'))
    )
    ex = (
        float(expo)
        if expo is not None
        else float(os.environ.get('PILOT_PAD_EXPO', '0.55'))
    )
    sm = (
        float(smooth)
        if smooth is not None
        else float(os.environ.get('PILOT_PAD_SMOOTH', '0.28'))
    )
    try:
        return pad.read(deadzone=dz, expo=ex, smooth=sm)
    except Exception as exc:
        print(f'[PAD] read failed ({exc})', flush=True)
        return None
