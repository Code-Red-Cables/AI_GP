"""Adapter from OpenCV body-frame navigation to Q2's planner contract."""

import math
import time

import config

# Soft altitude floor (NED down: more negative = higher).
# telem_025904: no floor → stuck at pos_d≈-0.15 into the ground.
# telem_030352: hard floor at -0.85 kept tgt_vd=-0.38 the whole approach and
# we hit the top rail at pos_d≈-0.76 before the floor ever released. Blend
# climb out near the deck, then hand vertical back to IBVS by cruise height.
_DECK_ALTITUDE_NED_M = -0.12
_CRUISE_ALTITUDE_NED_M = -0.48
_LOW_ALT_CLIMB_MPS = 0.30
_LOW_ALT_FORWARD_CAP_MPS = 0.10


class OpenCVGatePlanner:
    """Publish Q2 NED velocity and yaw-rate targets from the vision navigator."""

    name = 'opencv_gate'

    def __init__(self):
        self._last_state = None
        self._last_safety_reason = None

    @staticmethod
    def _hover():
        return {'vn': 0.0, 've': 0.0, 'vd': 0.0, 'yaw_rate': 0.0}

    @staticmethod
    def _ned_down(shared_data):
        for key in ('position_ned', 'local_position_ned'):
            pos = shared_data.get(key) or {}
            for axis in ('z', 'd', 'down'):
                value = pos.get(axis)
                if value is not None and math.isfinite(float(value)):
                    return float(value)
        return None

    def _safety_hover(self, shared_data, reason):
        shared_data['planner_mode'] = 'opencv_stale'
        if reason != self._last_safety_reason:
            log_event = shared_data.get('log_event')
            if log_event:
                log_event('PLANNER_SAFETY', reason)
            self._last_safety_reason = reason
        return self._hover()

    def compute_target(self, shared_data):
        navigation = shared_data.get('navigation') or {}
        timestamp_ns = navigation.get('ts')
        if timestamp_ns is None:
            return self._safety_hover(shared_data, 'missing_navigation')

        age_s = (time.time_ns() - timestamp_ns) / 1e9
        if age_s < -config.SENSOR_FUTURE_TOLERANCE_S:
            return self._safety_hover(
                shared_data, f'future_navigation age={age_s:.3f}s'
            )
        if age_s > config.VISION_COMMAND_TIMEOUT_S:
            return self._safety_hover(
                shared_data, f'stale_navigation age={age_s:.3f}s'
            )

        attitude = shared_data.get('attitude') or {}
        yaw = float(attitude.get('yaw', 0.0))
        forward = float(navigation.get('forward_mps', 0.0))
        right = float(navigation.get('right_mps', 0.0))
        down = float(navigation.get('down_mps', 0.0))
        yaw_rate = float(navigation.get('yaw_rate_rps', 0.0))
        if not all(
            math.isfinite(value)
            for value in (yaw, forward, right, down, yaw_rate)
        ):
            return self._safety_hover(shared_data, 'nonfinite_navigation')

        # Body (+forward, +right) -> world NED (+north, +east).
        vn = forward * math.cos(yaw) - right * math.sin(yaw)
        ve = forward * math.sin(yaw) + right * math.cos(yaw)

        # Soft altitude floor: full climb near the deck, fade to zero by cruise
        # height so IBVS can level/descend into the opening (030352).
        altitude_d = self._ned_down(shared_data)
        if (
            altitude_d is not None
            and altitude_d > _CRUISE_ALTITUDE_NED_M
        ):
            span = _CRUISE_ALTITUDE_NED_M - _DECK_ALTITUDE_NED_M
            if abs(span) < 1e-6:
                progress = 1.0
            else:
                progress = (altitude_d - _DECK_ALTITUDE_NED_M) / span
            progress = max(0.0, min(1.0, progress))
            floor_climb = -_LOW_ALT_CLIMB_MPS * (1.0 - progress)
            # Never override an IBVS descent once we have left the deck —
            # that override is exactly what drove the top-rail impact.
            if down < 0.0 or altitude_d > _DECK_ALTITUDE_NED_M:
                down = min(down, floor_climb)
            speed = math.hypot(vn, ve)
            forward_cap = _LOW_ALT_FORWARD_CAP_MPS * (0.35 + 0.65 * progress)
            if speed > forward_cap > 0.0:
                scale = forward_cap / speed
                vn *= scale
                ve *= scale

        target = {
            'vn': vn,
            've': ve,
            'vd': down,
            'yaw_rate': yaw_rate,
        }
        self._last_safety_reason = None

        state = navigation.get('state', 'UNKNOWN')
        shared_data['planner_mode'] = f'opencv_{state.lower()}'
        if state != self._last_state:
            log_event = shared_data.get('log_event')
            if log_event:
                log_event('OPENCV_STATE', str(state))
            self._last_state = state
        return target
