"""Receive Q2 camera datagrams and publish tracked OpenCV navigation."""

import os
import socket
import struct
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import config
from gate_estimator import estimate_gate
from vision.gate_detector import (
    DetectorDebug,
    GateDetection,
    GateVisionConfig,
    OrangeGateDetector,
    draw_detection,
)
from vision.gate_tracker import GateTracker, q2_demo_tracker_config
from vision.navigation import GateNavigator, q2_demo_navigation_config


def save_gate_capture(
    output_dir: Path,
    session_id: str,
    frame_id: int,
    image: np.ndarray,
    jpeg_bytes: bytes | None = None,
) -> Path:
    """Save the unannotated camera frame without re-encoding when possible."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f'gate_{session_id}_{int(frame_id):010d}.jpg'
    )
    if jpeg_bytes:
        output_path.write_bytes(jpeg_bytes)
    elif not cv2.imwrite(str(output_path), image):
        raise OSError(f'OpenCV could not write gate frame: {output_path}')
    return output_path


def create_gate_detector():
    """Build the configured detector without changing downstream interfaces."""
    hsv_ranges = ((config.GATE_HSV_LOWER, config.GATE_HSV_UPPER),)
    legacy_config = GateVisionConfig(
        hsv_ranges=hsv_ranges,
        min_contour_area=config.GATE_MIN_CONTOUR_AREA,
    )
    backend = config.GATE_DETECTOR_BACKEND
    if backend == 'hsv':
        print('[VISION] detector=hsv (explicit legacy mode)', flush=True)
        return OrangeGateDetector(legacy_config)

    repository_root = Path(__file__).resolve().parent

    def resolved_model_path(configured_path):
        model_path = Path(configured_path)
        if not model_path.is_absolute():
            model_path = repository_root / model_path
        return model_path

    pose_model_path = resolved_model_path(config.YOLO_POSE_MODEL_PATH)
    box_model_path = resolved_model_path(config.YOLO_MODEL_PATH)
    use_pose = backend == 'yolo_pose' or (
        backend == 'auto' and pose_model_path.is_file()
    )
    if use_pose:
        if not pose_model_path.is_file():
            raise FileNotFoundError(
                '[VISION] custom YOLO gate pose weights are missing at '
                f'{pose_model_path}; run tools/train_gate_pose.py'
            )
        from vision.yolo_pose_gate_detector import (
            PoseGateConfig,
            YoloPoseGateDetector,
        )

        pose_config = PoseGateConfig(
            model_path=str(pose_model_path),
            gate_class_name=config.YOLO_GATE_CLASS_NAME,
            confidence_threshold=config.YOLO_CONFIDENCE_THRESHOLD,
            keypoint_confidence_threshold=(
                config.YOLO_KEYPOINT_CONFIDENCE_THRESHOLD
            ),
            nms_iou_threshold=config.YOLO_NMS_IOU_THRESHOLD,
            target_lock_seconds=config.YOLO_TARGET_LOCK_SECONDS,
            minimum_gate_area_px=config.YOLO_MIN_GATE_AREA_PX,
            maximum_outside_fraction=config.YOLO_MAX_OUTSIDE_FRACTION,
            previous_center_frames=config.YOLO_PREVIOUS_CENTER_FRAMES,
            inference_size=config.YOLO_INFERENCE_SIZE,
            device=config.YOLO_DEVICE,
            log_interval_s=config.YOLO_LOG_INTERVAL_S,
            minimum_opening_area_px=config.GATE_MIN_CONTOUR_AREA,
        )
        try:
            detector = YoloPoseGateDetector(pose_config)
        except (ImportError, RuntimeError) as exc:
            if backend == 'yolo_pose':
                raise
            print(
                f'[VISION] YOLO pose unavailable ({exc}); trying fallback',
                flush=True,
            )
        else:
            print(
                '[VISION] detector=yolo_pose '
                f'model={pose_model_path} '
                f'conf={config.YOLO_CONFIDENCE_THRESHOLD:.2f} '
                'corners=4',
                flush=True,
            )
            return detector

    if not box_model_path.is_file():
        message = (
            '[VISION] custom YOLO gate weights are missing at '
            f'{box_model_path}; place trained one-class weights there'
        )
        if backend == 'yolo_hybrid':
            raise FileNotFoundError(message)
        print(f'{message}; detector=HSV fallback', flush=True)
        return OrangeGateDetector(legacy_config)

    from vision.yolo_gate_detector import (
        HybridGateConfig,
        YoloHybridGateDetector,
    )

    hybrid_config = HybridGateConfig(
        model_path=str(box_model_path),
        gate_class_name=config.YOLO_GATE_CLASS_NAME,
        confidence_threshold=config.YOLO_CONFIDENCE_THRESHOLD,
        nms_iou_threshold=config.YOLO_NMS_IOU_THRESHOLD,
        target_lock_seconds=config.YOLO_TARGET_LOCK_SECONDS,
        crop_padding_px=config.YOLO_CROP_PADDING_PX,
        minimum_gate_area_px=config.YOLO_MIN_GATE_AREA_PX,
        maximum_outside_fraction=config.YOLO_MAX_OUTSIDE_FRACTION,
        previous_center_frames=config.YOLO_PREVIOUS_CENTER_FRAMES,
        inference_size=config.YOLO_INFERENCE_SIZE,
        device=config.YOLO_DEVICE,
        log_interval_s=config.YOLO_LOG_INTERVAL_S,
        minimum_opening_area_px=config.GATE_MIN_CONTOUR_AREA,
        hsv_ranges=hsv_ranges,
    )
    try:
        detector = YoloHybridGateDetector(hybrid_config)
    except (ImportError, RuntimeError) as exc:
        if backend == 'yolo_hybrid':
            raise
        print(
            f'[VISION] YOLO hybrid unavailable ({exc}); detector=HSV fallback',
            flush=True,
        )
        return OrangeGateDetector(legacy_config)
    print(
        '[VISION] detector=yolo_hybrid '
        f'model={box_model_path} conf={config.YOLO_CONFIDENCE_THRESHOLD:.2f} '
        f'iou={config.YOLO_NMS_IOU_THRESHOLD:.2f}',
        flush=True,
    )
    return detector


def course_lookahead_horizontal(
    primary: GateDetection,
    debug: DetectorDebug,
) -> float | None:
    """Find a smaller supported gate visible beyond the active gate."""
    if primary is None or not primary.found:
        return None
    height, width = debug.cleaned_mask.shape[:2]
    if width <= 0 or height <= 0:
        return None
    scale = max(float(debug.scale), 1e-9)
    primary_x = primary.center_x * scale
    primary_y = primary.center_y * scale
    candidates = []
    for item in debug.candidates:
        if (
            not item.accepted
            or item.confidence < 0.45
            or item.features.get('supported_sides', 0.0) < 3.0
        ):
            continue
        center_x, center_y = item.center
        separation = np.hypot(
            (center_x - primary_x) / width,
            (center_y - primary_y) / height,
        )
        if separation < 0.12:
            continue
        normalized_x = (center_x - width / 2.0) / (width / 2.0)
        normalized_y = (center_y - height / 2.0) / (height / 2.0)
        bbox_area_ratio = (
            item.bbox[2] * item.bbox[3] / float(width * height)
        )
        if (
            bbox_area_ratio < 0.0015
            or abs(normalized_x) > 0.85
            or normalized_y > 0.65
        ):
            continue
        candidates.append(
            (
                item.confidence + 0.10 * min(1.0, bbox_area_ratio / 0.02),
                float(normalized_x),
            )
        )
    return max(candidates)[1] if candidates else None


class VisionRX:

    def __init__(self, data):
        self.data = data
        self.detector = create_gate_detector()
        self.tracker = GateTracker(q2_demo_tracker_config())
        self.navigator = GateNavigator(q2_demo_navigation_config())
        self._last_active_gate = None
        self._last_debug_t = 0.0
        self._last_state = None
        self._display_enabled = config.VISION_DISPLAY
        self._overlay_enabled = True
        self._gate_capture_enabled = config.GATE_FRAME_CAPTURE
        self._gate_capture_dir = Path(config.GATE_FRAME_CAPTURE_DIR)
        if not self._gate_capture_dir.is_absolute():
            self._gate_capture_dir = (
                Path(__file__).resolve().parent / self._gate_capture_dir
            )
        self._gate_capture_session = str(time.time_ns())
        self._last_gate_capture_t = -float('inf')
        self._gate_capture_count = 0
        if self._gate_capture_enabled:
            self._gate_capture_dir.mkdir(parents=True, exist_ok=True)
            print(
                '[VISION] gate-frame capture enabled: '
                f'{self._gate_capture_dir}',
                flush=True,
            )
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
    def _consume_confirmed_gate_pass(self, now: float) -> None:
        race_status = self.data.get('race_status') or {}
        active_gate = race_status.get('active_gate')
        if active_gate is None:
            return
        active_gate = int(active_gate)
        if self._last_active_gate is None:
            self._last_active_gate = active_gate
            return
        if active_gate > self._last_active_gate:
            self.tracker.reset()
            reset_target_lock = getattr(
                self.detector, 'reset_target_lock', None
            )
            if reset_target_lock is not None:
                reset_target_lock()
            self.navigator.confirm_gate_pass(now)
        self._last_active_gate = max(self._last_active_gate, active_gate)

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
                                jpeg_bytes=jpeg,
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
        accepted_targets: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compose camera, orange segmentation, and accepted-target panels."""
        if cleaned_mask.shape[:2] != annotated.shape[:2]:
            cleaned_mask = cv2.resize(
                cleaned_mask,
                (annotated.shape[1], annotated.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask_panel = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)
        if accepted_targets is None:
            accepted_targets = np.zeros_like(annotated)
        elif accepted_targets.shape[:2] != annotated.shape[:2]:
            accepted_targets = cv2.resize(
                accepted_targets,
                (annotated.shape[1], annotated.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        if accepted_targets.ndim == 2:
            accepted_targets = cv2.cvtColor(
                accepted_targets, cv2.COLOR_GRAY2BGR
            )
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
        upper_height = annotated.shape[0] // 2
        lower_height = annotated.shape[0] - upper_height
        diagnostics = []
        for panel, label, panel_height in (
            (mask_panel, 'ORANGE COLOR MASK', upper_height),
            (
                accepted_targets,
                'ACCEPTED GATE TARGET',
                lower_height,
            ),
        ):
            panel = cv2.resize(
                panel,
                (annotated.shape[1], panel_height),
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.putText(
                panel,
                label,
                (10, panel.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            diagnostics.append(panel)
        return np.hstack((annotated, np.vstack(diagnostics)))

    @staticmethod
    def build_accepted_target_frame(
        shape: tuple[int, ...],
        gate_detection,
    ) -> np.ndarray:
        """Show only geometry accepted for steering, not rejected candidates."""
        target = np.zeros(shape, dtype=np.uint8)
        if gate_detection is not None and gate_detection.found:
            if gate_detection.corners is not None:
                corners = np.round(
                    gate_detection.corners
                ).astype(np.int32).reshape(-1, 2)
                if len(corners) >= 3:
                    cv2.polylines(
                        target, [corners], True, (0, 255, 0), 3, cv2.LINE_AA
                    )
            else:
                x, y, width, height = gate_detection.bbox
                cv2.rectangle(
                    target,
                    (x, y),
                    (x + width, y + height),
                    (0, 255, 0),
                    3,
                )
        return target

    def _show_display(
        self,
        annotated: np.ndarray,
        cleaned_mask: np.ndarray,
        accepted_targets: np.ndarray,
    ) -> None:
        try:
            display = self.build_display_frame(
                annotated, cleaned_mask, accepted_targets
            )
            cv2.imshow('Q2 OpenCV vision', display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                self._display_enabled = False
                cv2.destroyWindow('Q2 OpenCV vision')
        except cv2.error as exc:
            self._display_enabled = False
            print(f'[VISION] live display disabled: {exc}', flush=True)

    # ------------------------------------------------------------------
    def _capture_gate_frame(
        self,
        measured: GateDetection,
        frame_id: int,
        image: np.ndarray,
        jpeg_bytes: bytes | None,
        now: float,
    ) -> None:
        """Capture detector-confirmed gates, excluding tracker predictions."""
        if (
            not self._gate_capture_enabled
            or not measured.found
            or measured.predicted
            or now - self._last_gate_capture_t
            < config.GATE_FRAME_CAPTURE_INTERVAL_S
        ):
            return
        try:
            path = save_gate_capture(
                self._gate_capture_dir,
                self._gate_capture_session,
                frame_id,
                image,
                jpeg_bytes,
            )
        except (OSError, cv2.error) as exc:
            self._gate_capture_enabled = False
            print(
                f'[VISION] gate-frame capture disabled: {exc}',
                flush=True,
            )
            return
        self._last_gate_capture_t = now
        self._gate_capture_count += 1
        if self._gate_capture_count == 1 or self._gate_capture_count % 50 == 0:
            print(
                '[VISION] saved gate frame '
                f'#{self._gate_capture_count}: {path.name}',
                flush=True,
            )

    # ------------------------------------------------------------------
    def process_frame(
        self,
        frame_id: int,
        img: np.ndarray,
        sim_time_ns: int | None = None,
        jpeg_bytes: bytes | None = None,
    ):
        started = time.perf_counter()
        monotonic_now = time.monotonic()
        timestamp_ns = time.time_ns()

        self._consume_confirmed_gate_pass(monotonic_now)
        hint = self.tracker.hint(monotonic_now)
        measured = self.detector.detect(
            img,
            hint=hint,
            timestamp=monotonic_now,
        )
        self._capture_gate_frame(
            measured,
            frame_id,
            img,
            jpeg_bytes,
            monotonic_now,
        )
        tracked = self.tracker.update(
            measured
            if measured.found and not measured.predicted
            else None,
            timestamp=monotonic_now,
        )
        next_gate_horizontal = course_lookahead_horizontal(
            measured, self.detector.last_debug
        )
        command = self.navigator.update(
            tracked,
            monotonic_now,
            next_gate_horizontal=next_gate_horizontal,
        )
        if (
            command.state.value == 'PASS_THROUGH'
            and tracked is not None
            and tracked.predicted
        ):
            # COMMIT intentionally treats a missing measured gate as a pass.
            # Do not let that just-passed track reject the spatially separate
            # next gate for the tracker's remaining prediction frames.
            self.tracker.reset()
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
            'next_gate_horizontal': next_gate_horizontal,
            'active_gate': self._last_active_gate,
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
            annotated = img.copy()
            if self._overlay_enabled:
                try:
                    annotated = draw_detection(
                        img,
                        tracked,
                        debug=self.detector.last_debug,
                        state=state,
                        command=command,
                        total_time_ms=total_ms,
                        show_rejected_candidates=False,
                        show_accepted_candidates=False,
                        show_mask_insets=False,
                    )
                    draw_hybrid_overlay = getattr(
                        self.detector, 'draw_debug_overlay', None
                    )
                    if draw_hybrid_overlay is not None:
                        annotated = draw_hybrid_overlay(annotated)
                except Exception as exc:
                    self._overlay_enabled = False
                    print(
                        '[VISION] annotation disabled; detection continues: '
                        f'{type(exc).__name__}: {exc}',
                        flush=True,
                    )
            if not self._overlay_enabled:
                cv2.putText(
                    annotated,
                    'ANNOTATION DISABLED - DETECTION STILL RUNNING',
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
            if self._display_enabled:
                accepted_targets = self.build_accepted_target_frame(
                    annotated.shape,
                    tracked,
                )
                self._show_display(
                    annotated.copy(),
                    self.detector.last_debug.cleaned_mask,
                    accepted_targets,
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
