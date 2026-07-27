"""Adapter from OpenCV body-frame navigation to Q2's planner contract."""

import math
import time

import config


class OpenCVGatePlanner:
    """Publish Q2 NED velocity and yaw-rate targets from the vision navigator."""

    name = 'opencv_gate'

    def __init__(self):
        self._last_state = None
        self._last_safety_reason = None

    @staticmethod
    def _hover():
        return {'vn': 0.0, 've': 0.0, 'vd': 0.0, 'yaw_rate': 0.0}

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
