"""Receive Q2 camera datagrams and publish tracked OpenCV navigation."""

from __future__ import annotations

import math
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
from vision.gate_bearings import (
    GateBearingTable,
    clamp_contact_vertical,
    detection_from_observation,
    draw_bearing_overlay,
    observe_pose_candidates,
    post_pass_lock_allowed,
    select_visible_next_observation,
)
from vision.gate_tracker import GateTracker, q2_demo_tracker_config
from vision.navigation import GateNavigator, q2_demo_navigation_config
from vision.dual_gate_pnp import observe_two_closest_gates
from vision.yolo_pnp import draw_gate_frame_axes
from vision.yolo_pnp import solve_corners_pnp


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
    if not pose_model_path.is_file():
        # Local training export often lives at the repo root before install.
        alt_pose = repository_root / 'gate_yolo_pose_v1.pt'
        if alt_pose.is_file():
            pose_model_path = alt_pose
    box_model_path = resolved_model_path(config.YOLO_MODEL_PATH)
    use_pose = backend == 'yolo_pose' or (
        backend == 'auto' and pose_model_path.is_file()
    )
    if use_pose:
        if not pose_model_path.is_file():
            raise FileNotFoundError(
                '[VISION] custom YOLO gate pose weights are missing at '
                f'{pose_model_path} (also tried gate_yolo_pose_v1.pt); '
                'run tools/train_gate_pose.py or copy weights to '
                'models/gate_pose.pt'
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
            persistent_target_lock=config.YOLO_PERSISTENT_TARGET_LOCK,
            target_association_center_span=(
                config.YOLO_TARGET_ASSOCIATION_CENTER_SPAN
            ),
            target_association_min_area_ratio=(
                config.YOLO_TARGET_ASSOCIATION_MIN_AREA_RATIO
            ),
            target_association_max_area_ratio=(
                config.YOLO_TARGET_ASSOCIATION_MAX_AREA_RATIO
            ),
            acquisition_confirmation_frames=(
                config.YOLO_ACQUISITION_CONFIRMATION_FRAMES
            ),
            post_pass_rejection_seconds=(
                config.YOLO_POST_PASS_REJECTION_SECONDS
            ),
            post_pass_max_area_ratio=(
                config.YOLO_POST_PASS_MAX_AREA_RATIO
            ),
            require_hsv_confirmation=config.YOLO_REQUIRE_HSV_CONFIRMATION,
            hsv_ranges=hsv_ranges,
            hsv_min_orange_ratio=config.YOLO_HSV_MIN_ORANGE_RATIO,
            hsv_max_orange_ratio=config.YOLO_HSV_MAX_ORANGE_RATIO,
            hsv_side_band_fraction=config.YOLO_HSV_SIDE_BAND_FRACTION,
            hsv_min_side_density=config.YOLO_HSV_MIN_SIDE_DENSITY,
            hsv_min_supported_sides=config.YOLO_HSV_MIN_SUPPORTED_SIDES,
            minimum_gate_area_px=config.YOLO_MIN_GATE_AREA_PX,
            maximum_outside_fraction=config.YOLO_MAX_OUTSIDE_FRACTION,
            previous_center_frames=config.YOLO_PREVIOUS_CENTER_FRAMES,
            estimated_opening_scale=config.YOLO_ESTIMATED_OPENING_SCALE,
            inference_size=config.YOLO_INFERENCE_SIZE,
            device=config.YOLO_DEVICE,
            log_interval_s=config.YOLO_LOG_INTERVAL_S,
            minimum_opening_area_px=config.GATE_MIN_CONTOUR_AREA,
            score_confidence_weight=config.YOLO_SCORE_CONFIDENCE_WEIGHT,
            score_center_weight=config.YOLO_SCORE_CENTER_WEIGHT,
            score_area_weight=config.YOLO_SCORE_AREA_WEIGHT,
            score_reference_area_ratio=(
                config.YOLO_SCORE_REFERENCE_AREA_RATIO
            ),
            hsv_blur_kernel=config.YOLO_HSV_BLUR_KERNEL,
            hsv_opening_kernel=config.YOLO_HSV_OPENING_KERNEL,
            hsv_closing_kernel=config.YOLO_HSV_CLOSING_KERNEL,
            hsv_center_blend=config.YOLO_HSV_CENTER_BLEND,
            hsv_center_max_shift_fraction=(
                config.YOLO_HSV_CENTER_MAX_SHIFT_FRACTION
            ),
            global_hsv_fallback_enabled=(
                config.GLOBAL_HSV_FALLBACK_ENABLED
            ),
            global_hsv_fallback_confidence_scale=(
                config.GLOBAL_HSV_FALLBACK_CONFIDENCE_SCALE
            ),
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
            hsv_mode = (
                'hsv_confirm'
                if config.YOLO_REQUIRE_HSV_CONFIRMATION
                else 'yolo_only'
            )
            print(
                '[VISION] detector=yolo_pose '
                f'model={pose_model_path} '
                f'conf={config.YOLO_CONFIDENCE_THRESHOLD:.2f} '
                f'corners=4 mode={hsv_mode}',
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
        score_confidence_weight=config.YOLO_SCORE_CONFIDENCE_WEIGHT,
        score_center_weight=config.YOLO_SCORE_CENTER_WEIGHT,
        score_area_weight=config.YOLO_SCORE_AREA_WEIGHT,
        score_reference_area_ratio=config.YOLO_SCORE_REFERENCE_AREA_RATIO,
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
        # pose_debug reuses the dual-gate PnP publish / overlay path.
        _nav = str(
            data.get('gate_navigation_mode') or config.GATE_NAVIGATION_MODE
        ).lower()
        self._kalman_mode = _nav in {'kalman', 'pose_debug'}
        self.detector = create_gate_detector()
        # Legacy IBVS path kept only for GATE_NAVIGATION_MODE=opencv.
        self.tracker = None if self._kalman_mode else GateTracker(
            q2_demo_tracker_config()
        )
        self.navigator = None if self._kalman_mode else GateNavigator(
            q2_demo_navigation_config()
        )
        # Needed in kalman too for post-pass COURSE_BEARING (0725).
        self.bearing_table = GateBearingTable()
        # 0 (not None): None→first-seen=1 skipped pass handling entirely
        # (072414 had GATE_PASSED but no KALMAN_PASS / COURSE_BEARING).
        self._last_active_gate = 0
        # 0835: dual-gate contact often appears on approach, then vanishes
        # at the hole (FOV filled by gate 1). Keep the last good secondary.
        self._pre_pass_contact = None
        self._pre_pass_contact_t = -float('inf')
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
        if self.data.pop('vision_reset_episode', None):
            self._pre_pass_contact = None
            self._pre_pass_contact_t = -float('inf')
            self._last_active_gate = 0
            self.data['course_bearing'] = None
            set_prefer = getattr(self.detector, 'set_prefer_horizontal', None)
            if callable(set_prefer):
                set_prefer(None)
            reset_episode = getattr(self.detector, 'reset_episode', None)
            if callable(reset_episode):
                reset_episode()
            else:
                reset_target_lock = getattr(
                    self.detector, 'reset_target_lock', None
                )
                if callable(reset_target_lock):
                    reset_target_lock()
            if self.bearing_table is not None:
                unfreeze = getattr(self.bearing_table, 'unfreeze', None)
                if callable(unfreeze):
                    unfreeze()
                clear = getattr(self.bearing_table, 'clear', None)
                if callable(clear):
                    clear()
        race_status = self.data.get('race_status') or {}
        active_gate = race_status.get('active_gate')
        if active_gate is None:
            return
        active_gate = int(active_gate)
        if active_gate > self._last_active_gate:
            begin_next_gate = getattr(
                self.detector, 'begin_next_gate_acquisition', None
            )
            if begin_next_gate is not None:
                begin_next_gate(now)
            else:
                reset_target_lock = getattr(
                    self.detector, 'reset_target_lock', None
                )
                if reset_target_lock is not None:
                    reset_target_lock()
            # Latch the near→far course bearing before vision lock is wiped.
            course_bearing = None
            if self.bearing_table is not None:
                course_bearing = self.bearing_table.consume_pass(now)
                if course_bearing is None:
                    course_bearing = self.bearing_table.peek_next(now)
                    self.bearing_table.freeze()
                # 0729: queue latched a 24.5 m far gate (h=+0.58). Prefer the
                # live second-nearest when it is meaningfully closer.
                contact = self.bearing_table.contact_secondary_bearing()
                if contact is not None and (
                    course_bearing is None
                    or contact.range_m + 2.0 < course_bearing.range_m
                ):
                    course_bearing = contact
                # Prefer a fresh approach latch over a worse live/table hit.
                # 0847: kept attempt-1 h=+0.10 for every later pass (even when
                # live contact was h=-0.75) — clear after use and ignore stale.
                pre_age = now - self._pre_pass_contact_t
                # 0928: age=3.8s h=+0.35 yawed into a wall — pre-pass older
                # than ~2s is usually the wrong/stale next-gate guess.
                if (
                    self._pre_pass_contact is not None
                    and 0.0 <= pre_age <= 2.0
                ):
                    pre = self._pre_pass_contact
                    # 0930: left pre-pass (h=-0.45) was latched then force-
                    # bumped to +0.28 — wrong range, seek into the wall.
                    if float(pre.horizontal_normalized) < -0.05:
                        print(
                            '[VISION] ignoring left pre-pass contact '
                            f'h={pre.horizontal_normalized:+.2f} '
                            f'r={pre.range_m:.1f}m',
                            flush=True,
                        )
                        use_pre = False
                    else:
                        use_pre = course_bearing is None
                        if course_bearing is not None:
                            # Only override when pre is clearly better AND not
                            # an opposite-side leftover from a prior lap.
                            same_side = (
                                float(pre.horizontal_normalized)
                                * float(course_bearing.horizontal_normalized)
                                >= 0.0
                            ) or abs(
                                float(course_bearing.horizontal_normalized)
                            ) < 0.20
                            use_pre = bool(
                                same_side
                                and (
                                    abs(float(pre.horizontal_normalized))
                                    + 0.05
                                    < abs(
                                        float(
                                            course_bearing.horizontal_normalized
                                        )
                                    )
                                    or float(pre.range_m) + 1.0
                                    < float(course_bearing.range_m)
                                )
                            )
                    if use_pre:
                        course_bearing = pre
                        print(
                            '[VISION] using pre-pass contact bearing '
                            f'h={course_bearing.horizontal_normalized:+.2f} '
                            f'r={course_bearing.range_m:.1f}m '
                            f'age={pre_age:.1f}s',
                            flush=True,
                        )
                elif (
                    self._pre_pass_contact is not None
                    and pre_age > 2.0
                ):
                    print(
                        '[VISION] ignoring stale pre-pass contact '
                        f'h={self._pre_pass_contact.horizontal_normalized:+.2f} '
                        f'r={self._pre_pass_contact.range_m:.1f}m '
                        f'age={pre_age:.1f}s',
                        flush=True,
                    )
            # Drop left course bearings — gate-2 hunt is rightward; a left
            # latch (even if later bumped) still carries a wrong range.
            if (
                course_bearing is not None
                and float(course_bearing.horizontal_normalized) < -0.05
            ):
                print(
                    '[VISION] course bearing rejected '
                    f'h={course_bearing.horizontal_normalized:+.2f} '
                    f'r={course_bearing.range_m:.1f}m (left of course)',
                    flush=True,
                )
                course_bearing = None
            # Far contacts are useless; wide-but-near ones get clamped so we
            # still seek (0839: rejected h=+0.44 @ 3.5m → NONE → floor).
            # 0922: rejected h=+0.33 @ 6.9m → NONE while YOLO still saw the
            # next gate nearby. Allow mid-range contacts; far (>8 m) stay out.
            if course_bearing is not None and course_bearing.range_m > 7.5:
                print(
                    '[VISION] course bearing rejected '
                    f'h={course_bearing.horizontal_normalized:+.2f} '
                    f'r={course_bearing.range_m:.1f}m (too far)',
                    flush=True,
                )
                course_bearing = None
            elif course_bearing is not None and (
                abs(course_bearing.horizontal_normalized) >= 0.35
                or float(course_bearing.vertical_normalized) > 0.05
                or float(course_bearing.vertical_normalized) < -0.20
            ):
                from dataclasses import replace

                h_old = float(course_bearing.horizontal_normalized)
                v_old = float(course_bearing.vertical_normalized)
                # 0919: h=+0.40 skipped old >0.40 clamp; v=+0.37 made seek dive.
                # 0930: true next was h=+0.84 @ 5.2m; ±0.45 still undershot →
                # Environment scrape while YOLO sat on a small u≈413 vs u≈580.
                h_lim = 0.70 if float(course_bearing.range_m) <= 5.5 else 0.50
                h = float(np.clip(h_old, -h_lim, h_lim))
                v = float(np.clip(v_old, -0.20, 0.05))
                if abs(h - h_old) > 1e-3 or abs(v - v_old) > 1e-3:
                    print(
                        '[VISION] course bearing clamped '
                        f'h={h_old:+.2f}→{h:+.2f} '
                        f'v={v_old:+.2f}→{v:+.2f} '
                        f'r={course_bearing.range_m:.1f}m',
                        flush=True,
                    )
                course_bearing = replace(
                    course_bearing,
                    horizontal_normalized=h,
                    vertical_normalized=v,
                )
            bearing_published = False
            # 0811: NEVER saw 2+ YOLO gates → bearing NONE, then far dual_pnp
            # stole the seek. Synthesize from live gate2 body before wipe.
            dual_snap = self.data.get('dual_gate_pnp') or {}
            g2_body = dual_snap.get('gate2_body')
            if course_bearing is None and g2_body is not None:
                try:
                    g2 = np.asarray(g2_body, dtype=np.float64).reshape(3)
                    fwd = max(0.4, float(g2[0]))
                    nx = float(np.clip(float(g2[1]) / fwd, -0.70, 0.70))
                    ny = float(np.clip(float(g2[2]) / fwd, -0.35, 0.20))
                    range_m = float(np.linalg.norm(g2))
                    if range_m <= 8.0 and abs(nx) <= 0.50:
                        self.data['course_bearing'] = {
                            'nx': nx,
                            'ny': ny,
                            'range_m': range_m,
                            'source': 'gate2_body',
                            'ts': now,
                        }
                        detail = (
                            f'h={nx:+.2f} v={ny:+.2f} '
                            f'range={range_m:.1f}m src=gate2_body'
                        )
                        print(
                            f'[VISION] course bearing after pass: {detail}',
                            flush=True,
                        )
                        log_event = self.data.get('log_event')
                        if log_event:
                            log_event('COURSE_BEARING', detail)
                        bearing_published = True
                except (TypeError, ValueError):
                    pass
            if course_bearing is not None and not bearing_published:
                h_pub = float(course_bearing.horizontal_normalized)
                v_pub = float(
                    max(-0.20, min(0.05, course_bearing.vertical_normalized))
                )
                # Mild right bias for near-straight image bearings. Do not
                # flip a clear left bearing (already rejected above).
                if abs(h_pub) < 0.22:
                    h_pub = max(h_pub, 0.28)
                self.data['course_bearing'] = {
                    'nx': h_pub,
                    'ny': v_pub,
                    'range_m': float(course_bearing.range_m),
                    'source': str(course_bearing.source),
                    'ts': now,
                }
                detail = (
                    f'h={h_pub:+.2f} '
                    f'v={v_pub:+.2f} '
                    f'range={course_bearing.range_m:.1f}m '
                    f'src={course_bearing.source} '
                    f'remaining={len(self.bearing_table.upcoming)}'
                )
                print(
                    f'[VISION] course bearing after pass: {detail}',
                    flush=True,
                )
                log_event = self.data.get('log_event')
                if log_event:
                    log_event('COURSE_BEARING', detail)
                bearing_published = True
            if not bearing_published:
                # 0833: bearing table empty (single-gate approach) but YOLO
                # often still sees a second mid-sized instance — aim there.
                bearing_published = self._publish_secondary_pose_bearing(now)
            if not bearing_published:
                # Publish a synthetic right bias so the planner latches and
                # YOLO post-pass reject/left-filter actually run. Logging
                # NONE alone left _course_latched=False (0929: left u≈301
                # stole over right u≈518, then floored).
                self.data['course_bearing'] = {
                    'nx': 0.28,
                    'ny': -0.06,
                    'range_m': 8.0,
                    'source': 'default_right',
                    'ts': now,
                }
                print(
                    '[VISION] course bearing after pass: '
                    'h=+0.28 v=-0.06 range=8.0m src=default_right '
                    '(never saw 2+ YOLO gates at once)',
                    flush=True,
                )
                log_event = self.data.get('log_event')
                if log_event:
                    log_event(
                        'COURSE_BEARING',
                        'h=+0.28 v=-0.06 range=8.0m src=default_right',
                    )
                bearing_published = True
            set_prefer = getattr(self.detector, 'set_prefer_horizontal', None)
            if callable(set_prefer):
                cb = self.data.get('course_bearing') or {}
                # Gate-2 is consistently right of the exit. Near-straight
                # bearings (h≈+0.04) still need right prefer — 0929 locked
                # left u≈295 after seeing right u≈465, then wall-scraped.
                prefer = cb.get('nx')
                try:
                    prefer = float(prefer)
                except (TypeError, ValueError):
                    prefer = 0.28
                if prefer < 0.12:
                    prefer = max(prefer, 0.28)
                set_prefer(prefer)
            # One-shot: do not reuse this latch on the next gate/lap.
            self._pre_pass_contact = None
            self._pre_pass_contact_t = -float('inf')
            if self.bearing_table is not None:
                unfreeze = getattr(self.bearing_table, 'unfreeze', None)
                if callable(unfreeze):
                    unfreeze()

            if self._kalman_mode:
                # Dual-gate EKF path: drop sticky association so the next
                # nearest centered gate wins (065915 chased u≈600 edge junk).
                reset_target_lock = getattr(
                    self.detector, 'reset_target_lock', None
                )
                if reset_target_lock is not None:
                    reset_target_lock()
                begin_next_gate = getattr(
                    self.detector, 'begin_next_gate_acquisition', None
                )
                if begin_next_gate is not None:
                    begin_next_gate(now)
                self.data['dual_gate_pnp'] = None
                self.data['gate_detection'] = None
                print(
                    f'[VISION] kalman gate pass → active={active_gate}',
                    flush=True,
                )
                log_event = self.data.get('log_event')
                if log_event:
                    log_event('KALMAN_PASS', f'active={active_gate}')
            else:
                self.tracker.reset()
                if course_bearing is not None:
                    self.navigator.seed_next_gate_bearing(
                        course_bearing.horizontal_normalized,
                        course_bearing.vertical_normalized,
                        freeze_for_slew=True,
                    )
                self.navigator.confirm_gate_pass(now)
        elif active_gate < self._last_active_gate:
            # Race/sim reset rewound the index — allow future passes again.
            # Keep _pre_pass_contact: 0841 cleared the good h=+0.12/5.4m
            # latch on rewind, then the punch only saw wide junk → NONE.
            self._last_active_gate = active_gate
            self.data['course_bearing'] = None
            set_prefer = getattr(self.detector, 'set_prefer_horizontal', None)
            if callable(set_prefer):
                set_prefer(None)
        self._last_active_gate = active_gate

    def _publish_secondary_pose_bearing(self, now: float) -> bool:
        """Aim at the second mid-sized YOLO instance when the table is empty."""
        pose_debug = getattr(self.detector, 'last_pose_debug', None)
        candidates = getattr(pose_debug, 'candidates', None) or []
        if len(candidates) < 2:
            return False
        try:
            from camera_model import CX, CY, WIDTH, HEIGHT
        except Exception:
            return False
        frame_area = float(max(WIDTH * HEIGHT, 1))
        scored = []
        for candidate in candidates:
            box = getattr(candidate, 'box', None)
            if box is None:
                continue
            try:
                area = float(box.area)
                cx, cy = box.center
            except (TypeError, ValueError, AttributeError):
                continue
            # 0922: post-pass next gate often 800–2200 px; 2500 left NONE.
            # 0930: real next at ~667 px was skipped; edge-right 567 had
            # nx=+0.77 filtered by 0.70 — secondary fell through to image.
            if area < 600.0 or area > 0.35 * frame_area:
                continue
            nx = (float(cx) - CX) / (WIDTH * 0.5)
            ny = (float(cy) - CY) / (HEIGHT * 0.5)
            if abs(nx) > 0.85:
                continue
            scored.append((area, nx, ny))
        if len(scored) < 2:
            return False
        scored.sort(key=lambda item: -item[0])
        # Largest is usually the gate we just cleared; require a rightish next
        # box (0930: fallback scored[1]=h=-0.24 steered into the left wall).
        rightish = [s for s in scored[1:] if float(s[1]) >= -0.05]
        if not rightish:
            return False
        pick = rightish[0]
        _, nx, ny = pick
        ny = float(max(-0.20, min(0.05, ny)))
        # Rough range from box area (1.5 m gate ≈ area at a few meters).
        # 0930: ~2k px estimated 7.3m and was dropped by a 5.5m cap while
        # YOLO already had u≈537 — allow mid-far secondaries.
        range_m = float(
            np.clip(3.0 * math.sqrt(12000.0 / max(pick[0], 1.0)), 2.0, 8.0)
        )
        # Allow stronger right seek when the secondary is clearly off-center
        # (0930: image h=+0.84 needed ~0.70; 0.40 still scraped Environment).
        nx = float(np.clip(nx, 0.12, 0.65))
        self.data['course_bearing'] = {
            'nx': nx,
            'ny': ny,
            'range_m': range_m,
            'source': 'secondary_pose',
            'ts': now,
        }
        detail = (
            f'h={nx:+.2f} v={ny:+.2f} '
            f'range={range_m:.1f}m src=secondary_pose'
        )
        print(f'[VISION] course bearing after pass: {detail}', flush=True)
        log_event = self.data.get('log_event')
        if log_event:
            log_event('COURSE_BEARING', detail)
        return True

    def _update_bearing_table(self, image: np.ndarray, now: float) -> None:
        """Latch near→far bearings whenever two or more pose gates are visible."""
        pose_debug = getattr(self.detector, 'last_pose_debug', None)
        candidates = getattr(pose_debug, 'candidates', None)
        if not candidates:
            self.data['gate_bearings'] = self.bearing_table.as_dict(now)
            return
        height, width = image.shape[:2]
        # Do NOT require HSV here. Distant gates often fail orange confirmation
        # but YOLO still sees them — that early 2–3 gate view is exactly what
        # we need for post-pass look direction.
        observations = observe_pose_candidates(
            candidates,
            width,
            height,
            require_hsv=False,
            min_confidence=0.20,
        )
        previous_count = len(self.bearing_table.upcoming)
        had_pair = self.bearing_table.has_contact_pair
        self.bearing_table.update(observations, now)
        contact = self.bearing_table.contact_secondary_bearing()
        nxt = contact if contact is not None else self.bearing_table.peek_next(now)
        # Keep the best near secondary seen on approach for the pass moment
        # when the FOV is filled and the contact pair disappears (0835).
        # Clamp wide headings so a usable 3–5 m contact is never discarded.
        if contact is not None and 1.5 <= float(contact.range_m) <= 7.5:
            from dataclasses import replace

            h = float(
                np.clip(float(contact.horizontal_normalized), -0.45, 0.45)
            )
            v = float(
                np.clip(float(contact.vertical_normalized), -0.35, 0.20)
            )
            stored = replace(
                contact,
                horizontal_normalized=h,
                vertical_normalized=v,
            )
            prev = self._pre_pass_contact
            if (
                prev is None
                or float(stored.range_m) + 0.5 < float(prev.range_m)
                or abs(h) < abs(float(prev.horizontal_normalized))
            ):
                self._pre_pass_contact = stored
                self._pre_pass_contact_t = now
        # Pre-seed from the live second-nearest gate so approach keeps contact
        # and a late pass still has a direction (opencv navigator only).
        if (
            self.navigator is not None
            and nxt is not None
            and self.navigator._confirmed_gate_passes == 0
        ):
            self.navigator.seed_next_gate_bearing(
                nxt.horizontal_normalized,
                nxt.vertical_normalized,
            )
        if (
            self.bearing_table.has_contact_pair
            and (
                not had_pair
                or len(self.bearing_table.upcoming) != previous_count
            )
        ):
            primary = self.bearing_table.live_primary
            if primary is not None and contact is not None:
                detail = (
                    f'approach={primary.range_m:.1f}m '
                    f'contact_h={contact.horizontal_normalized:+.2f} '
                    f'contact_v={contact.vertical_normalized:+.2f} '
                    f'contact_r={contact.range_m:.1f}m '
                    f'src={contact.source}'
                )
                print(f'[VISION] nearest-two: {detail}', flush=True)
                log_event = self.data.get('log_event')
                if log_event:
                    log_event('NEAREST_TWO', detail)
        self.data['gate_bearings'] = self.bearing_table.as_dict(now)

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
            color = (
                (0, 255, 255)
                if gate_detection.predicted
                else (0, 255, 0)
            )
            if gate_detection.corners is not None:
                corners = np.round(
                    gate_detection.corners
                ).astype(np.int32).reshape(-1, 2)
                if len(corners) >= 3:
                    cv2.polylines(
                        target, [corners], True, color, 3, cv2.LINE_AA
                    )
            else:
                x, y, width, height = gate_detection.bbox
                cv2.rectangle(
                    target,
                    (x, y),
                    (x + width, y + height),
                    color,
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
    def _draw_dual_gate_overlay(
        self, annotated: np.ndarray, dual: dict
    ) -> None:
        """Label nearest PnP gates and project axes on the active guidance gate."""
        del dual  # ranges already drawn in the status line
        pose_debug = getattr(self.detector, 'last_pose_debug', None)
        candidates = list(getattr(pose_debug, 'candidates', None) or ())
        if not candidates:
            return
        preferred = getattr(pose_debug, 'selected', None)
        obs = observe_two_closest_gates(
            candidates,
            timestamp=0.0,
            min_confidence=0.45,
            preferred=preferred,
        )
        if obs is None:
            return
        # G1 is the gate we steer on (YOLO preferred / nearest solved).
        for label, gate, color, active in (
            ('G1', obs.gate1, (0, 255, 255), True),
            ('G2', obs.gate2, (255, 180, 0), False),
        ):
            if gate is None:
                continue
            corners = getattr(gate, 'corners_px', None)
            if corners is None:
                continue
            pts = np.round(np.asarray(corners)).astype(np.int32).reshape(-1, 2)
            if len(pts) >= 3:
                cv2.polylines(
                    annotated, [pts], True, color, 2, cv2.LINE_AA
                )
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                tag = f'{label} {gate.range_m:.1f}m'
                if active:
                    tag = f'BASE {tag}'
                cv2.putText(
                    annotated,
                    tag,
                    (cx - 28, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            # RGB = gate X (right) / Y (down) / Z (through).
            draw_gate_frame_axes(
                annotated,
                gate,
                axis_length_m=0.85 if active else 0.55,
                thickness=3 if active else 2,
                label='XYZ' if active else None,
            )
        cv2.putText(
            annotated,
            'axes: X=right Y=down Z=through',
            (8, annotated.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    def _publish_dual_gate_pnp(self, timestamp_ns: int) -> None:
        """YOLO → two closest gates → PnP body centres for the EKF planner."""
        pose_debug = getattr(self.detector, 'last_pose_debug', None)
        candidates = getattr(pose_debug, 'candidates', None) or ()
        preferred = getattr(pose_debug, 'selected', None)
        obs = observe_two_closest_gates(
            candidates,
            timestamp=time.monotonic(),
            min_confidence=0.45,
            preferred=preferred,
        )
        if obs is None:
            self.data['dual_gate_pnp'] = {
                'ts': timestamp_ns,
                'gate1_body': None,
                'gate2_body': None,
                'gate1_through_body': None,
                'gate1_range_m': None,
                'gate2_range_m': None,
                'gate1_norm_x': None,
                'gate1_norm_y': None,
                'n_solved': 0,
            }
            return
        # Image-normalized centre for altitude (immune to camera-tilt body-z bias).
        corners = obs.gate1.corners_px
        cx = float(np.mean(corners[:, 0]))
        cy = float(np.mean(corners[:, 1]))
        # Match GateDetection convention: ±1 at image half-width / half-height.
        from camera_model import CX, CY, WIDTH, HEIGHT

        norm_x = (cx - CX) / (WIDTH * 0.5)
        norm_y = (cy - CY) / (HEIGHT * 0.5)
        self.data['dual_gate_pnp'] = {
            'ts': timestamp_ns,
            'gate1_body': obs.gate1_body.tolist(),
            'gate2_body': None
            if obs.gate2_body is None
            else obs.gate2_body.tolist(),
            'gate1_through_body': None
            if obs.gate1_through_body is None
            else obs.gate1_through_body.tolist(),
            'gate1_range_m': float(obs.gate1.range_m),
            'gate2_range_m': None
            if obs.gate2 is None
            else float(obs.gate2.range_m),
            'gate1_norm_x': norm_x,
            'gate1_norm_y': norm_y,
            'n_solved': 1 if obs.gate2 is None else 2,
        }

    def _publish_pnp_fix(self, measured, timestamp_ns: int) -> None:
        """PnP-solve the pose detector's raw corners and feed the VIO.

        Reuses the keypoints the YoloPoseGateDetector already produced for
        this frame (no second inference). Only real detections qualify —
        tracker predictions have no fresh corners. The solve itself rejects
        degenerate quads and planar-ambiguity flips via reprojection error.
        """
        if not config.USE_VIO:
            return
        if measured is None or not measured.found or measured.predicted:
            return
        debug = getattr(self.detector, 'last_debug', None)
        selected = getattr(debug, 'selected', None)
        keypoints = getattr(selected, 'keypoints', None)
        if keypoints is None:
            return
        gate = solve_corners_pnp(keypoints, confidence=measured.confidence)
        if gate is None:
            return
        fix = {
            'ts': timestamp_ns,
            'R_cg': gate.R_cg.tolist(),
            't_cg': gate.t_cg.tolist(),
            'reproj_err_px': gate.reproj_err_px,
            'range_m': gate.range_m,
        }
        lock = self.data.get('lock')
        if lock is not None:
            with lock:
                self.data['pnp_fix'] = fix
        else:
            self.data['pnp_fix'] = fix

    # ------------------------------------------------------------------
    def _process_frame_kalman(
        self,
        frame_id: int,
        img: np.ndarray,
        sim_time_ns: int | None,
        jpeg_bytes: bytes | None,
        monotonic_now: float,
        timestamp_ns: int,
        started: float,
    ) -> None:
        """YOLO + dual-gate PnP only — no IBVS navigator / bearing chase."""
        # Keep YOLO prefer aligned with planner retargets (live_retarget).
        set_prefer = getattr(self.detector, 'set_prefer_horizontal', None)
        if callable(set_prefer):
            cb = self.data.get('course_bearing') or {}
            prefer = cb.get('nx') if isinstance(cb, dict) else None
            set_prefer(prefer)
        measured = self.detector.detect(
            img,
            hint=None,
            timestamp=monotonic_now,
        )
        self._update_bearing_table(img, monotonic_now)
        self._capture_gate_frame(
            measured,
            frame_id,
            img,
            jpeg_bytes,
            monotonic_now,
        )
        self._publish_dual_gate_pnp(timestamp_ns)
        dual = self.data.get('dual_gate_pnp') or {}
        n_solved = int(dual.get('n_solved') or 0)
        state = 'DUAL_PNP' if n_solved >= 1 else 'SEARCH'
        self._log_state(state)

        gate_detection = None
        if measured is not None and measured.found and not measured.predicted:
            gate_detection = {
                'center_px': measured.center_px,
                'corners_px': measured.corners_px,
                'bbox_px': measured.bbox_px,
                'area_px': measured.area_px,
                'confidence': measured.confidence,
                'method': measured.method,
                'predicted': measured.predicted,
                'frame_id': frame_id,
                'ts': timestamp_ns,
            }

        total_ms = (time.perf_counter() - started) * 1000.0
        self.data['gate_detection'] = gate_detection
        self.data['vision'] = None
        self.data['navigation'] = {
            'ts': timestamp_ns,
            'sim_time_ns': sim_time_ns,
            'frame_id': frame_id,
            'state': state,
            'n_solved': n_solved,
            'gate1_range_m': dual.get('gate1_range_m'),
            'gate2_range_m': dual.get('gate2_range_m'),
            'forward_mps': 0.0,
            'right_mps': 0.0,
            'down_mps': 0.0,
            'yaw_rate_rps': 0.0,
            'confidence': float(measured.confidence)
            if measured is not None and measured.found
            else 0.0,
            'predicted': False,
        }
        self.data['vision_timings_ms'] = {
            **getattr(self.detector.last_debug, 'timings_ms', {}),
            'total': total_ms,
        }
        self.data['control_source'] = 'kalman'

        should_save_debug = (
            config.VISION_DEBUG
            and time.time() - self._last_debug_t
            >= config.VISION_DEBUG_INTERVAL_S
        )
        if self._display_enabled or should_save_debug:
            annotated = img.copy()
            detection_for_overlay = (
                measured
                if measured is not None and measured.found
                else None
            )
            if self._overlay_enabled:
                try:
                    annotated = draw_detection(
                        img,
                        detection_for_overlay,
                        debug=self.detector.last_debug,
                        state=state,
                        command=None,
                        total_time_ms=total_ms,
                        show_rejected_candidates=False,
                        show_accepted_candidates=True,
                        show_mask_insets=False,
                    )
                    g1r = dual.get('gate1_range_m')
                    g2r = dual.get('gate2_range_m')
                    if g1r is not None:
                        label = f'PNP n={n_solved} g1={g1r:.1f}m'
                    else:
                        label = f'PNP n={n_solved}'
                    if g2r is not None:
                        label += f' g2={g2r:.1f}m'
                    cv2.putText(
                        annotated,
                        label,
                        (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    self._draw_dual_gate_overlay(annotated, dual)
                except Exception:
                    annotated = img
            if should_save_debug:
                path = os.path.join(
                    config.VISION_DEBUG_DIR,
                    f'kalman_{frame_id:06d}.jpg',
                )
                cv2.imwrite(path, annotated)
                self._last_debug_t = time.time()
            if self._display_enabled:
                debug = self.detector.last_debug
                cleaned = getattr(debug, 'cleaned_mask', None)
                if cleaned is None:
                    cleaned = np.zeros(annotated.shape[:2], dtype=np.uint8)
                accepted_targets = self.build_accepted_target_frame(
                    annotated.shape,
                    detection_for_overlay,
                )
                self._show_display(
                    annotated.copy(),
                    cleaned,
                    accepted_targets,
                )

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
        if self._kalman_mode:
            self._process_frame_kalman(
                frame_id, img, sim_time_ns, jpeg_bytes, monotonic_now,
                timestamp_ns, started,
            )
            return
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
        self._publish_pnp_fix(measured, timestamp_ns)
        self._update_bearing_table(img, monotonic_now)
        height, width = img.shape[:2]
        # Nearest-two policy before a pass: IBVS-approach live_primary while
        # feeding live_secondary as contact lookahead. After a pass, prefer a
        # live detection matching the remembered near next gate.
        visible_next = None
        if self.navigator._confirmed_gate_passes > 0:
            look_h = (
                self.navigator._post_pass_look_horizontal
                if self.navigator._post_pass_look_horizontal is not None
                else self.navigator._pending_next_gate_horizontal
            )
            # Only lock a live gate if it matches the remembered *near* next
            # target. A far end-of-course gate through the opening is ignored.
            visible_next = select_visible_next_observation(
                self.bearing_table.last_observations,
                look_horizontal=look_h,
                expected_range_m=self.bearing_table.expected_next_range_m,
            )
        steer_measured = measured
        primary = self.bearing_table.live_primary
        if (
            self.navigator._confirmed_gate_passes == 0
            and primary is not None
            and (
                measured is None
                or not measured.found
                or measured.predicted
                or (
                    measured.distance_m is not None
                    and math.isfinite(float(measured.distance_m))
                    and float(measured.distance_m) > 1.35 * primary.range_m
                )
            )
        ):
            # YOLO largest-lock can stick on a farther instance; force the
            # nearest near-course gate as the approach target.
            steer_measured = detection_from_observation(
                primary, width, height, monotonic_now, role='approach'
            )
        if visible_next is not None:
            steer_measured = detection_from_observation(
                visible_next, width, height, monotonic_now, role='visible_next'
            )
        # After a pass, never IBVS-chase a sky/far speck — that path took
        # telem_031946 to gv=18, RECOVER, then an Environment smash. Prefer
        # open-loop course bearing until a plausible next gate appears.
        if (
            self.navigator._confirmed_gate_passes > 0
            and not post_pass_lock_allowed(
                steer_measured,
                expected_range_m=self.bearing_table.expected_next_range_m,
            )
        ):
            steer_measured = None
        tracked = self.tracker.update(
            steer_measured
            if steer_measured is not None
            and steer_measured.found
            and not steer_measured.predicted
            else None,
            timestamp=monotonic_now,
        )
        next_gate_horizontal = course_lookahead_horizontal(
            measured, self.detector.last_debug
        )
        next_gate_vertical = None
        contact = self.bearing_table.contact_secondary_bearing()
        course_bearing = self.bearing_table.peek_next(monotonic_now)
        # During the post-pass slew, do not feed live/table bearings into the
        # navigator — that path overwrote a good +0.53 latch with -0.30.
        if self.navigator.in_post_pass_slew(monotonic_now):
            next_gate_horizontal = None
            next_gate_vertical = None
        elif (
            self.navigator._confirmed_gate_passes == 0
            and contact is not None
        ):
            # Maintain contact with the second-nearest gate while approaching
            # the first — prefer the live secondary over stale image lookahead.
            next_gate_horizontal = contact.horizontal_normalized
            next_gate_vertical = clamp_contact_vertical(
                contact.vertical_normalized
            )
        elif course_bearing is not None:
            if next_gate_horizontal is None:
                next_gate_horizontal = course_bearing.horizontal_normalized
            next_gate_vertical = clamp_contact_vertical(
                course_bearing.vertical_normalized
            )
        # Prefer post-pass visible-next, else nearest primary override, else
        # the normal tracker output. After a pass with no safe lock, clear the
        # target so SEARCH follows the frozen course bearing.
        nav_target = tracked
        if (
            visible_next is not None
            and steer_measured is not None
            and steer_measured.found
            and post_pass_lock_allowed(
                steer_measured,
                expected_range_m=self.bearing_table.expected_next_range_m,
            )
        ):
            nav_target = steer_measured
        elif (
            self.navigator._confirmed_gate_passes == 0
            and primary is not None
            and steer_measured is not None
            and steer_measured.found
            and str(getattr(steer_measured, 'method', '')).startswith(
                'approach_'
            )
        ):
            nav_target = steer_measured
        elif (
            self.navigator._confirmed_gate_passes > 0
            and not post_pass_lock_allowed(
                nav_target,
                expected_range_m=self.bearing_table.expected_next_range_m,
            )
        ):
            nav_target = None
        command = self.navigator.update(
            nav_target,
            monotonic_now,
            next_gate_horizontal=next_gate_horizontal,
            next_gate_vertical=next_gate_vertical,
        )
        # Once the slew window ends and we have a real lock, allow the table
        # to rebuild the remaining course from fresh multi-gate views.
        if (
            self.bearing_table._frozen
            and not self.navigator.in_post_pass_slew(monotonic_now)
            and tracked is not None
            and tracked.found
            and not tracked.predicted
            and command.state.value
            in ('TRACK', 'ALIGN_AND_APPROACH')
        ):
            self.bearing_table.unfreeze()
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
            'requested_forward_mps': command.requested_forward_mps,
            'framing_limited': command.framing_limited,
            'framing_edge': command.framing_edge,
            # Tracker velocity is normalized image width per second. It is the
            # observed angular motion of the target and therefore captures the
            # combined effect of lateral velocity and yaw rate.
            'gate_velocity_x': (
                tracked.velocity_x if tracked is not None else None
            ),
            'gate_velocity_y': (
                tracked.velocity_y if tracked is not None else None
            ),
            'horizontal_lead_error': (
                tracked.normalized_x
                + (
                    self.navigator.config.lateral_kd
                    / max(self.navigator.config.lateral_kp, 1e-6)
                )
                * tracked.velocity_x
                if tracked is not None
                else None
            ),
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
                    look_h = (
                        self.navigator._post_pass_look_horizontal
                        if self.navigator._post_pass_look_horizontal
                        is not None
                        else self.navigator._pending_next_gate_horizontal
                    )
                    look_v = (
                        self.navigator._post_pass_look_vertical
                        if self.navigator._post_pass_look_vertical
                        is not None
                        else self.navigator._pending_next_gate_vertical
                    )
                    annotated = draw_bearing_overlay(
                        annotated,
                        self.bearing_table,
                        pending_horizontal=look_h,
                        pending_vertical=look_v,
                    )
                    if self.navigator.in_post_pass_slew(monotonic_now):
                        cv2.putText(
                            annotated,
                            'POST-PASS SLEW (turning to NEXT)',
                            (10, annotated.shape[0] - 36),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
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
