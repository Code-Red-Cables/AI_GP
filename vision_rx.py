"""Receive Q2 camera datagrams and publish tracked OpenCV navigation."""

import os
import socket
import struct
import threading
import time

import cv2
import numpy as np

import camera_model as cm
import config
from gate_estimator import estimate_gate
from vision.gate_detector import OrangeGateDetector, draw_detection
from vision.gate_tracker import GateTracker
from vision.navigation import GateNavigator, NavigationConfig


class VisionRX:

    def __init__(self, data):
        self.data = data
        self.detector = OrangeGateDetector()
        self.tracker = GateTracker()
        body_forward_v = cm.project(cm.body_to_cam((1.0, 0.0, 0.0)))[1]
        body_forward_normalized_y = 2.0 * body_forward_v / cm.HEIGHT - 1.0
        self.navigator = GateNavigator(
            NavigationConfig(
                vertical_setpoint_normalized=body_forward_normalized_y
            )
        )
        self._last_debug_t = 0.0
        self._last_state = None
        self._display_enabled = config.VISION_DISPLAY
        if config.VISION_DEBUG:
            os.makedirs(config.VISION_DEBUG_DIR, exist_ok=True)
        self.thread = threading.Thread(target=self._vision_loop, daemon=False)
        self.is_running = True
        self.thread.start()

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    def _log_state(self, state):
        if state == self._last_state:
            return
        log_event = self.data.get('log_event')
        if log_event:
            log_event('VISION_STATE', state)
        self._last_state = state

    # ------------------------------------------------------------------
    def _vision_loop(self):
        header_fmt = '<IHHIIQ'
        header_sz = struct.calcsize(header_fmt)
        frames = {}
        latest_complete_id = -1
        latest_complete_sim_time = -1

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((config.VISION_UDP_IP, config.VISION_UDP_PORT))
        sock.settimeout(0.2)
        print('[VISION] listening for camera frames...', flush=True)

        try:
            while self.is_running:
                try:
                    packet, _ = sock.recvfrom(65536)
                except socket.timeout:
                    continue
                if len(packet) < header_sz:
                    continue

                header = packet[:header_sz]
                payload = packet[header_sz:]
                (
                    frame_id,
                    chunk_id,
                    total_chunks,
                    jpeg_size,
                    payload_size,
                    sim_time_ns,
                ) = struct.unpack(header_fmt, header)
                if payload_size != len(payload) or chunk_id >= total_chunks:
                    continue
                if (
                    frame_id <= latest_complete_id
                    and sim_time_ns <= latest_complete_sim_time
                ):
                    continue

                entry = frames.setdefault(
                    frame_id,
                    {
                        'chunks': {},
                        'total': total_chunks,
                        'jpeg_size': jpeg_size,
                        'sim_time_ns': sim_time_ns,
                    },
                )
                entry['chunks'][chunk_id] = payload

                if len(entry['chunks']) == entry['total']:
                    try:
                        jpeg = b''.join(
                            entry['chunks'][index]
                            for index in range(entry['total'])
                        )
                    except KeyError:
                        jpeg = b''
                    if len(jpeg) == entry['jpeg_size']:
                        image = cv2.imdecode(
                            np.frombuffer(jpeg, dtype=np.uint8),
                            cv2.IMREAD_COLOR,
                        )
                        if image is not None:
                            self.process_frame(
                                frame_id,
                                image,
                                sim_time_ns=entry['sim_time_ns'],
                            )
                            latest_complete_id = frame_id
                            latest_complete_sim_time = sim_time_ns
                    del frames[frame_id]

                    # Process the newest complete frame, not queued old frames.
                    for old_id in [key for key in frames if key < frame_id]:
                        del frames[old_id]

                if len(frames) > 30:
                    del frames[min(frames)]
        finally:
            sock.close()
            if self._display_enabled:
                try:
                    cv2.destroyWindow('Q2 OpenCV vision')
                except cv2.error:
                    pass

    @staticmethod
    def build_display_frame(
        annotated: np.ndarray,
        cleaned_mask: np.ndarray,
    ) -> np.ndarray:
        """Compose camera detection and binary segmentation panels."""
        if cleaned_mask.shape[:2] != annotated.shape[:2]:
            cleaned_mask = cv2.resize(
                cleaned_mask,
                (annotated.shape[1], annotated.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask_panel = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            annotated,
            'CAMERA + DETECTION',
            (10, annotated.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            mask_panel,
            'CLEANED ORANGE BITMASK',
            (10, mask_panel.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return np.hstack((annotated, mask_panel))

    def _show_display(
        self,
        annotated: np.ndarray,
        cleaned_mask: np.ndarray,
    ) -> None:
        try:
            display = self.build_display_frame(annotated, cleaned_mask)
            cv2.imshow('Q2 OpenCV vision', display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                self._display_enabled = False
                cv2.destroyWindow('Q2 OpenCV vision')
        except cv2.error as exc:
            self._display_enabled = False
            print(f'[VISION] live display disabled: {exc}', flush=True)

    # ------------------------------------------------------------------
    def process_frame(
        self,
        frame_id: int,
        img: np.ndarray,
        sim_time_ns: int | None = None,
    ):
        started = time.perf_counter()
        monotonic_now = time.monotonic()
        timestamp_ns = time.time_ns()

        hint = self.tracker.hint(monotonic_now)
        measured = self.detector.detect(
            img,
            hint=hint,
            timestamp=monotonic_now,
        )
        tracked = self.tracker.update(
            measured if measured.found else None,
            timestamp=monotonic_now,
        )
        command = self.navigator.update(tracked, monotonic_now)
        state = command.state.value
        self._log_state(state)

        gate_detection = None
        if tracked is not None and tracked.found:
            gate_detection = {
                'center_px': tracked.center_px,
                'corners_px': tracked.corners_px,
                'bbox_px': tracked.bbox_px,
                'area_px': tracked.area_px,
                'confidence': tracked.confidence,
                'method': tracked.method,
                'predicted': tracked.predicted,
                'frame_id': frame_id,
                'ts': timestamp_ns,
            }

        attitude = self.data.get('attitude')
        position = self.data.get('local_position_ned')
        position_ned = None
        if position:
            position_ned = (
                position.get('x', 0.0),
                position.get('y', 0.0),
                position.get('z', 0.0),
            )
        gate_estimate = estimate_gate(
            tracked,
            attitude=attitude,
            position_ned=position_ned,
            ts=timestamp_ns,
        )

        navigation = {
            'ts': timestamp_ns,
            'sim_time_ns': sim_time_ns,
            'frame_id': frame_id,
            'forward_mps': command.forward_mps,
            'right_mps': command.right_mps,
            'down_mps': command.down_mps,
            'yaw_rate_rps': command.yaw_rate_rps,
            'state': state,
            'confidence': command.confidence,
            'predicted': command.predicted,
            'alignment_error': command.alignment_error,
        }
        total_ms = (time.perf_counter() - started) * 1000.0

        # Each value is replaced atomically so readers never see a partial dict.
        self.data['gate_detection'] = gate_detection
        self.data['vision'] = gate_estimate
        self.data['navigation'] = navigation
        self.data['vision_timings_ms'] = {
            **self.detector.last_debug.timings_ms,
            'tracker': self.tracker.last_update_ms,
            'total': total_ms,
        }

        should_save_debug = (
            config.VISION_DEBUG
            and time.time() - self._last_debug_t
            >= config.VISION_DEBUG_INTERVAL_S
        )
        if self._display_enabled or should_save_debug:
            annotated = draw_detection(
                img,
                tracked,
                debug=self.detector.last_debug,
                state=state,
                command=command,
                total_time_ms=total_ms,
            )
            if self._display_enabled:
                self._show_display(
                    annotated.copy(),
                    self.detector.last_debug.cleaned_mask,
                )
        if should_save_debug:
            self._last_debug_t = time.time()
            cv2.imwrite(
                os.path.join(
                    config.VISION_DEBUG_DIR,
                    f'frame_{frame_id:06d}.jpg',
                ),
                annotated,
            )
