"""Receive Q2 camera datagrams and publish dual-gate PnP observations."""

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
from vision.gate_detector import (
    GateVisionConfig,
    OrangeGateDetector,
    draw_detection,
)
from vision.gate_bearings import (
    GateBearingTable,
    observe_pose_candidates,
)
from vision.dual_gate_pnp import (
    observe_nearest_two_by_range,
    observe_two_closest_gates,
)
from vision.yolo_pnp import draw_gate_frame_axes
from vision.yolo_pose_gate_detector import draw_pose_debug_overlay


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


def _candidate_keypoints(pose_debug, center_px):
    """The eight keypoints of whichever pose candidate matches ``center_px``.

    Returns ``(keypoints, confidences)`` as plain lists, or ``(None, None)``
    when the active detector is not a pose model. Matching is by nearest box
    centre because the selected detection has already been through association
    and target-lock logic, so identity has to be recovered rather than assumed.
    """
    candidates = list(getattr(pose_debug, 'candidates', None) or ())
    if not candidates or center_px is None:
        return None, None
    try:
        cx, cy = float(center_px[0]), float(center_px[1])
    except (TypeError, ValueError, IndexError):
        return None, None

    best = None
    best_d2 = float('inf')
    for candidate in candidates:
        box = getattr(candidate, 'box', None)
        kps = getattr(candidate, 'keypoints', None)
        if box is None or kps is None:
            continue
        try:
            bx, by = box.center
        except (TypeError, ValueError, AttributeError):
            continue
        d2 = (float(bx) - cx) ** 2 + (float(by) - cy) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = candidate
    if best is None:
        return None, None

    try:
        points = np.asarray(best.keypoints, dtype=np.float64).reshape(-1, 2)
    except (TypeError, ValueError):
        return None, None
    confs = getattr(best, 'keypoint_confidences', None)
    try:
        conf_list = [
            float(c) for c in np.asarray(confs, dtype=np.float64).reshape(-1)
        ] if confs is not None else [1.0] * len(points)
    except (TypeError, ValueError):
        conf_list = [1.0] * len(points)
    return [[float(u), float(v)] for u, v in points], conf_list


class VisionRX:

    def __init__(self, data):
        self.data = data
        self.detector = create_gate_detector()
        # Post-pass COURSE_BEARING for the next gate (0725).
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
        self._init_gate_capture()
        if config.VISION_DEBUG:
            os.makedirs(config.VISION_DEBUG_DIR, exist_ok=True)
        # Daemon so a Ctrl+C or traceback cannot leave the console hung
        # waiting on this thread at interpreter exit.
        self.thread = threading.Thread(target=self._vision_loop, daemon=True)
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
        # Assist visual-commit / planner punch-through: drop sticky identity
        # so the next gate can be acquired (same path as race active_gate++).
        if self.data.pop('vision_begin_next_gate', None):
            begin_next_gate = getattr(
                self.detector, 'begin_next_gate_acquisition', None
            )
            if begin_next_gate is not None:
                begin_next_gate(now)
            else:
                reset_target_lock = getattr(
                    self.detector, 'reset_target_lock', None
                )
                if callable(reset_target_lock):
                    reset_target_lock()
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
        elif active_gate < self._last_active_gate:
            # Race/sim reset rewound the index — allow future passes again.
            # Keep _pre_pass_contact: 0841 cleared the good h=+0.12/5.4m
            # latch on rewind, then the punch only saw wide junk → NONE.
            self._last_active_gate = active_gate
            self.data['course_bearing'] = None
            set_prefer = getattr(self.detector, 'set_prefer_horizontal', None)
            if callable(set_prefer):
                set_prefer(None)
            # phase5 174620: after TRACK_DATA rewind, association stayed stuck
            # at y≈5 while a fresh pad gate sat at y≈177 — clear the lock.
            reset_target_lock = getattr(
                self.detector, 'reset_target_lock', None
            )
            if callable(reset_target_lock):
                reset_target_lock()
            reset_episode = getattr(self.detector, 'reset_episode', None)
            if callable(reset_episode):
                reset_episode()
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
        # A frame is several UDP datagrams, and nothing drains this socket
        # while process_frame runs. At the default buffer (~64 KB) that is
        # about one frame of slack, so every frame arriving mid-detection
        # loses chunks and is discarded incomplete -- the drone goes blind
        # for reasons that never show up as an error.
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        except OSError:
            pass
        sock.bind((config.VISION_UDP_IP, config.VISION_UDP_PORT))
        sock.settimeout(0.2)
        rcvbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        print(f'[VISION] listening for camera frames... '
              f'(rcvbuf {rcvbuf / 1024:.0f} KiB)', flush=True)

        stats_t = time.time()
        stats = {'emitted': 0, 'processed': 0, 'process_s': 0.0,
                 'detect_s': 0.0}
        newest_seen = -1

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
                            t_proc = time.perf_counter()
                            self.process_frame(
                                frame_id,
                                image,
                                sim_time_ns=entry['sim_time_ns'],
                                jpeg_bytes=jpeg,
                            )
                            stats['process_s'] += time.perf_counter() - t_proc
                            stats['processed'] += 1
                            try:
                                stats['detect_s'] += 0.001 * float(
                                    self.data['vision_timings_ms']['total']
                                )
                            except (KeyError, TypeError, ValueError):
                                pass
                            latest_complete_id = frame_id
                            latest_complete_sim_time = sim_time_ns
                    del frames[frame_id]

                    # Process the newest complete frame, not queued old frames.
                    for old_id in [key for key in frames if key < frame_id]:
                        del frames[old_id]

                if len(frames) > 30:
                    del frames[min(frames)]

                # frame_id is assigned by the simulator, so the high-water
                # mark counts what was sent and the gap to `processed` is
                # what we never looked at.
                if frame_id > newest_seen:
                    newest_seen = frame_id
                now_t = time.time()
                if now_t - stats_t >= config.VISION_STATS_INTERVAL_S:
                    span = now_t - stats_t
                    seen = newest_seen - stats['emitted'] if stats['emitted'] else 0
                    done = stats['processed']
                    busy = stats['process_s']
                    if seen > 0:
                        print(
                            f'[VISION] {done / span:5.1f} fps processed of '
                            f'{seen / span:5.1f} fps emitted '
                            f'({100.0 * (1.0 - done / max(seen, 1)):3.0f}% dropped), '
                            f'{1000.0 * busy / max(done, 1):5.1f} ms/frame '
                            f'({1000.0 * stats["detect_s"] / max(done, 1):5.1f} '
                            f'detect), {100.0 * busy / span:3.0f}% busy',
                            flush=True,
                        )
                    stats_t = now_t
                    stats['emitted'] = newest_seen
                    stats['processed'] = 0
                    stats['process_s'] = 0.0
                    stats['detect_s'] = 0.0
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
        *,
        pilot_lock: dict | None = None,
    ) -> np.ndarray:
        """Show only geometry accepted for steering, not rejected candidates."""
        target = np.zeros(shape, dtype=np.uint8)
        # Pilot: always paint LOCK status on this panel (manual + auto).
        # Without this, missing pilot_lock defaults look the same as LOCK green.
        if isinstance(pilot_lock, dict):
            locked = bool(pilot_lock.get('locked'))
            mode = str(pilot_lock.get('mode') or 'manual')
            if locked:
                color = (0, 255, 0)
                banner = (0, 140, 0)
                label = f'LOCK  press T  [{mode}]'
            else:
                color = (255, 0, 255)
                banner = (180, 0, 180)
                label = f'NO LOCK  [{mode}]'
            h, w = target.shape[:2]
            bar_h = max(28, h // 10)
            cv2.rectangle(target, (0, 0), (w, bar_h), banner, -1)
            cv2.putText(
                target,
                label,
                (8, bar_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            color = (0, 255, 255) if (
                gate_detection is not None
                and getattr(gate_detection, 'predicted', False)
            ) else (0, 255, 0)

        if gate_detection is not None and gate_detection.found:
            if isinstance(pilot_lock, dict) and not pilot_lock.get('locked'):
                color = (255, 0, 255)
            elif isinstance(pilot_lock, dict) and pilot_lock.get('locked'):
                color = (0, 255, 0)
            cx = int(round(float(gate_detection.center_x)))
            cy = int(round(float(gate_detection.center_y)))
            cv2.drawMarker(
                target, (cx, cy), color, cv2.MARKER_CROSS, 22, 2
            )
        return target

    def _apply_pilot_lock_overlay(
        self,
        annotated: np.ndarray,
        gate_detection,
    ) -> None:
        """Pilot LOCK banner: purple = tracking, green = LOCK ready."""
        pilot = self.data.get('pilot_lock')
        if not isinstance(pilot, dict):
            return
        locked = bool(pilot.get('locked'))
        mode = str(pilot.get('mode') or 'manual')
        # BGR: tracking = magenta/purple, LOCK = green.
        if locked:
            color = (0, 255, 0)
            banner = (0, 140, 0)
            label = f'LOCK  press T   [{mode}]'
        else:
            color = (255, 0, 255)
            banner = (180, 0, 180)
            label = f'NO LOCK   [{mode}]'
        h, w = annotated.shape[:2]
        bar_h = max(36, h // 12)
        cv2.rectangle(annotated, (0, 0), (w, bar_h), banner, -1)
        cv2.putText(
            annotated,
            label,
            (12, bar_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if gate_detection is not None and getattr(gate_detection, 'found', False):
            cx = int(round(float(gate_detection.center_x)))
            cy = int(round(float(gate_detection.center_y)))
            cv2.circle(annotated, (cx, cy), 6, color, -1)
            cv2.circle(annotated, (cx, cy), 10, (255, 255, 255), 2)

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
        """Capture sparse raw frames; acro reference mode also keeps misses."""
        capture_all = bool(self.data.get('vision_reference_capture_all'))
        confirmed = bool(
            measured is not None
            and measured.found
            and not measured.predicted
        )
        if (
            not self._gate_capture_enabled
            or (not capture_all and not confirmed)
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
    def _init_gate_capture(self) -> None:
        """Point frame capture at a fresh per-run subfolder."""
        self._gate_capture_enabled = config.GATE_FRAME_CAPTURE
        base = Path(config.GATE_FRAME_CAPTURE_DIR)
        if not base.is_absolute():
            base = Path(__file__).resolve().parent / base
        # Stamped like the telemetry logs, so a frame set can be matched to the
        # flight it came from. These all used to land in one directory, which
        # reached 282k files.
        self._gate_capture_session = time.strftime('%Y%m%d_%H%M%S')
        self._gate_capture_dir = base / f'run_{self._gate_capture_session}'
        self._last_gate_capture_t = -float('inf')
        self._gate_capture_count = 0
        self.data['gate_frame_capture'] = {
            'enabled': bool(self._gate_capture_enabled),
            'session': self._gate_capture_session,
            'directory': str(self._gate_capture_dir.resolve()),
            'interval_s': float(config.GATE_FRAME_CAPTURE_INTERVAL_S),
            'confirmed_detections_only': not bool(
                self.data.get('vision_reference_capture_all')
            ),
        }
        if self._gate_capture_enabled:
            self._gate_capture_dir.mkdir(parents=True, exist_ok=True)
            interval = config.GATE_FRAME_CAPTURE_INTERVAL_S
            rate = f'{1.0 / interval:.0f}/s' if interval > 0 else 'every hit'
            print(
                f'[VISION] gate-frame capture enabled ({rate}): '
                f'{self._gate_capture_dir}',
                flush=True,
            )

    # ------------------------------------------------------------------
    def _draw_dual_gate_overlay(
        self, annotated: np.ndarray, dual: dict
    ) -> None:
        """Draw range labels + RGB axes; keypoints come from the pose overlay."""
        del dual  # ranges already drawn in the status line
        pose_debug = getattr(self.detector, 'last_pose_debug', None)
        candidates = list(getattr(pose_debug, 'candidates', None) or ())
        if not candidates:
            return
        # Overlay = strict nearest-two by PnP range (not YOLO preferred),
        # so both pad gates get axes even when YOLO identity picks only one.
        obs = observe_nearest_two_by_range(
            candidates,
            timestamp=0.0,
            min_confidence=0.45,
        )
        if obs is None:
            return
        for label, gate, color in (
            ('G1', obs.gate1, (0, 255, 255)),
            ('G2', obs.gate2, (255, 180, 0)),
        ):
            if gate is None:
                continue
            # Prefer the measured keypoints (outer+inner); fall back to the
            # four projected outer corners only for the range label anchor.
            kpts = getattr(gate, 'keypoints_px', None)
            if kpts is not None and len(kpts) > 0:
                pts = np.asarray(kpts, dtype=np.float64).reshape(-1, 2)
            else:
                corners = getattr(gate, 'corners_px', None)
                if corners is None:
                    continue
                pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
            usable = pts[np.isfinite(pts).all(axis=1)]
            if len(usable) == 0:
                continue
            cx = int(np.mean(usable[:, 0]))
            cy = int(np.mean(usable[:, 1]))
            cv2.putText(
                annotated,
                f'{label} {gate.range_m:.1f}m',
                (cx - 28, cy - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
            # Same axis length on both — user needs to compare orientations.
            draw_gate_frame_axes(
                annotated,
                gate,
                axis_length_m=1.35,
                thickness=4,
                label=f'{label} XYZ',
            )
        n = 1 + int(obs.gate2 is not None)
        cv2.putText(
            annotated,
            f'dual PnP n={n}  axes: X=right Y=down Z=through',
            (8, annotated.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    def _publish_dual_gate_pnp(self, timestamp_ns: int) -> None:
        """YOLO-locked gate1 (+ nearest other) → PnP for planner range/body.

        Must prefer the detector's selected identity. Nearest-by-range alone
        flipped assist onto a far gate (18 m → 34 m) mid-approach.
        """
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
            # Hold last good dual while YOLO lock is still alive (preferred
            # PnP failed this frame — do not fall back to nearest-range).
            prev = self.data.get('dual_gate_pnp') or {}
            if (
                preferred is not None
                and prev.get('gate1_body') is not None
                and int(prev.get('n_solved') or 0) >= 1
            ):
                held = dict(prev)
                held['ts'] = timestamp_ns
                held['held'] = True
                self.data['dual_gate_pnp'] = held
                return
            self.data['dual_gate_pnp'] = {
                'ts': timestamp_ns,
                'gate1_body': None,
                'gate2_body': None,
                'gate1_through_body': None,
                'gate1_down_body': None,
                'gate1_normal_body': None,
                'gate1_reproj_px': None,
                'gate1_range_m': None,
                'gate2_range_m': None,
                'gate1_norm_x': None,
                'gate1_norm_y': None,
                'n_solved': 0,
                'held': False,
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
            'gate1_down_body': None
            if obs.gate1_down_body is None
            else obs.gate1_down_body.tolist(),
            'gate1_normal_body': None
            if obs.gate1_normal_body is None
            else obs.gate1_normal_body.tolist(),
            'gate1_reproj_px': float(obs.gate1.reproj_err_px),
            'gate1_range_m': float(obs.gate1.range_m),
            'gate2_range_m': None
            if obs.gate2 is None
            else float(obs.gate2.range_m),
            'gate1_norm_x': norm_x,
            'gate1_norm_y': norm_y,
            'n_solved': 1 if obs.gate2 is None else 2,
            'held': False,
            'preferred': preferred is not None,
        }

    # ------------------------------------------------------------------
    def process_frame(
        self,
        frame_id: int,
        img: np.ndarray,
        sim_time_ns: int | None = None,
        jpeg_bytes: bytes | None = None,
    ) -> None:
        """YOLO + dual-gate PnP only — feeds the EKF, never IBVS."""
        started = time.perf_counter()
        monotonic_now = time.monotonic()
        timestamp_ns = time.time_ns()

        self._consume_confirmed_gate_pass(monotonic_now)
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
        # Publish the raw pose candidates as observe-only geometry.  The
        # detector's persistent selected identity is useful for ordinary
        # course following, but during an acro flip it can reject the active
        # gate just as that gate becomes the largest object in the frame.
        # Replay can use this list for a short, explicitly bounded final
        # centering correction without changing detector selection globally.
        raw_candidates = []
        pose_debug = getattr(self.detector, 'last_pose_debug', None)
        for candidate in list(getattr(pose_debug, 'candidates', None) or ()):
            box = getattr(candidate, 'box', None)
            if box is None:
                continue
            try:
                cx, cy = box.center
                x1, y1, x2, y2 = box.bbox
                raw_candidates.append({
                    'center_px': (float(cx), float(cy)),
                    'bbox_px': (
                        float(x1), float(y1), float(x2), float(y2),
                    ),
                    'area_px': float(box.area),
                    'confidence': float(box.confidence),
                    'hsv_confirmed': bool(candidate.hsv_confirmed),
                })
            except (TypeError, ValueError, AttributeError):
                continue
        self.data['gate_candidates'] = {
            'frame_id': frame_id,
            'ts': timestamp_ns,
            'items': raw_candidates,
        }
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
            # The eight raw keypoints, taken from the pose candidate rather
            # than from dual_gate_pnp. The learned policy consumes corners
            # directly (paper1 uses no pose estimation), so it must not be
            # gated on a PnP solve succeeding — PnP was non-null in 1 of 39
            # laps on this detector.
            kps, kconf = _candidate_keypoints(pose_debug, measured.center_px)
            gate_detection['keypoints_px'] = kps
            gate_detection['keypoint_confidences'] = kconf

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
                    pose_debug = getattr(
                        self.detector, 'last_pose_debug', None
                    )
                    if pose_debug is not None:
                        # Pose path: outer+inner keypoints only — no box /
                        # contour outlines (those were drowning the rings).
                        annotated = draw_pose_debug_overlay(img, pose_debug)
                    else:
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
                # Lock banner must survive draw/overlay failures — needed in
                # pilot manual so LOCK/NO LOCK is visible before pressing T.
                try:
                    self._apply_pilot_lock_overlay(
                        annotated, detection_for_overlay
                    )
                except Exception:
                    pass
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
                    pilot_lock=self.data.get('pilot_lock'),
                )
                pose_debug = getattr(
                    self.detector, 'last_pose_debug', None
                )
                if pose_debug is not None:
                    # Same keypoint-only view on the lock panel.
                    accepted_targets = draw_pose_debug_overlay(
                        accepted_targets, pose_debug
                    )
                self._show_display(
                    annotated.copy(),
                    cleaned,
                    accepted_targets,
                )
