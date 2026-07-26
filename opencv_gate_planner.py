"""Adapter from OpenCV body-frame navigation to Q2's planner contract."""

import math
import time

import config


class OpenCVGatePlanner:
    """Publish Q2 NED velocity and yaw-rate targets from the vision navigator."""

    name = 'opencv_gate'

    def __init__(self):
        self._last_state = None

    @staticmethod
    def _hover():
        return {'vn': 0.0, 've': 0.0, 'vd': 0.0, 'yaw_rate': 0.0}

    def compute_target(self, shared_data):
        navigation = shared_data.get('navigation') or {}
        timestamp_ns = navigation.get('ts')
        if timestamp_ns is None:
            shared_data['planner_mode'] = 'opencv_stale'
            return self._hover()

        age_s = max(0.0, (time.time_ns() - timestamp_ns) / 1e9)
        if age_s > config.VISION_COMMAND_TIMEOUT_S:
            shared_data['planner_mode'] = 'opencv_stale'
            return self._hover()

        attitude = shared_data.get('attitude') or {}
        yaw = float(attitude.get('yaw', 0.0))
        forward = float(navigation.get('forward_mps', 0.0))
        right = float(navigation.get('right_mps', 0.0))

        # Body (+forward, +right) -> world NED (+north, +east).
        vn = forward * math.cos(yaw) - right * math.sin(yaw)
        ve = forward * math.sin(yaw) + right * math.cos(yaw)
        target = {
            'vn': vn,
            've': ve,
            'vd': float(navigation.get('down_mps', 0.0)),
            'yaw_rate': float(navigation.get('yaw_rate_rps', 0.0)),
        }

        state = navigation.get('state', 'UNKNOWN')
        shared_data['planner_mode'] = f'opencv_{state.lower()}'
        if state != self._last_state:
            log_event = shared_data.get('log_event')
            if log_event:
                log_event('OPENCV_STATE', str(state))
            self._last_state = state
        return target
