"""Pose-debug: hover on the start pad with small righting moves.

Locks gate range + image (nx, ny) after takeoff, then holds that pose.
Keeps thrust above the floor-scrape floor (see run 125504).

  python main.py --pose-debug
"""

from __future__ import annotations

import math
import time

import config


class PoseDebugPlanner:
    name = 'pose_debug_standoff'

    def __init__(self):
        self._announced = False
        self._last_log_t = 0.0

    def reset_episode(self):
        self._announced = False
        self._last_log_t = 0.0

    def compute_target(self, shared_data):
        shared_data['planner_mode'] = self.name
        shared_data['post_pass_hunt'] = False

        if not self._announced:
            print(
                '[POSE_DEBUG] start-pad hover — level near lock, '
                f'blind bias {math.degrees(config.POSE_DEBUG_HOVER_BIAS_RAD):.1f}° '
                f'(rate≤{config.POSE_DEBUG_MAX_LEVEL_RATE})',
                flush=True,
            )
            self._announced = True

        now = time.monotonic()
        path = shared_data.get('kalman_path') or {}
        if now - self._last_log_t >= 1.0:
            phase = path.get('phase', '?')
            r = path.get('range_m')
            lock = path.get('locked_range_m')
            pitch = path.get('pitch_deg')
            thr = path.get('thrust')
            nx = path.get('nx')
            vf = path.get('v_fwd')
            msg = f'phase={phase}'
            if r is not None and lock is not None:
                msg += f' range={float(r):.2f}/{float(lock):.2f}m'
            elif r is not None:
                msg += f' range={float(r):.2f}m'
            if nx is not None:
                msg += f' nx={float(nx):+.2f}'
            if pitch is not None:
                msg += f' pitch={float(pitch):+.1f}°'
            if vf is not None:
                msg += f' vf={float(vf):+.2f}'
            if thr is not None:
                msg += f' thrust={float(thr):.3f}'
            print(f'[POSE_DEBUG] {msg}', flush=True)
            self._last_log_t = now

        return {
            'vn': 0.0,
            've': 0.0,
            'vd': 0.0,
            'yaw_rate': 0.0,
        }
