"""Live panel: the camera frame beside the exact vector fed to the network.

Debugging a vision policy by watching the drone is guesswork. This renders, side
by side and from one source of truth:

  left    the camera frame with the live detector's overlay only: YOLO
          keypoints when YOLO is running, GateNet inner corners when GateNet
          is running — never both at once
  right   every channel of the observation vector, the attitude source actually
          in use, the course context, and the command coming back out

The observation is rebuilt here with ``observation_from_shared``, the same
function the planner calls, so what you read is what the network receives. It is
not a copy that can drift out of step.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from race_obs import (
    FEATURE_DIM,
    KEYPOINT_COUNT,
    LABEL_NAMES,
    N_GATES,
    NOT_SEEN,
)

PANEL_W = 430
_FONT = 0.42
_LINE = 15

_WHITE = (235, 235, 235)
_DIM = (140, 140, 140)
_GOOD = (120, 235, 120)
_WARN = (90, 200, 250)
_BAD = (90, 90, 245)
_ACCENT = (250, 200, 120)
_SNAKE = (250, 120, 250)
_GATENET = (50, 220, 255)   # cyan: friend's 4 inner corners


def _fmt(value, digits: int = 2, width: int = 7) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 'nan'.rjust(width)
    if not math.isfinite(f):
        return 'nan'.rjust(width)
    return f'{f:+.{digits}f}'.rjust(width)


class ObservationPanel:
    """Composites the camera frame and the network's input into one window."""

    WINDOW = 'policy input'

    def __init__(self, *, with_context: bool = False, scale: float = 1.0):
        self.with_context = bool(with_context)
        self.scale = float(scale)
        self._enabled = True
        self._cv2 = None

    # ------------------------------------------------------------------
    def _cv(self):
        if self._cv2 is None:
            import cv2
            self._cv2 = cv2
        return self._cv2

    def close(self) -> None:
        if self._cv2 is not None and self._enabled:
            try:
                self._cv2.destroyWindow(self.WINDOW)
            except Exception:
                pass
        self._enabled = False

    # ------------------------------------------------------------------
    def _frame(self, shared_data: dict) -> np.ndarray:
        cv2 = self._cv()
        img = shared_data.get('debug_frame')
        if img is None:
            img = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(img, 'no camera frame', (170, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, _BAD, 2, cv2.LINE_AA)
            return img
        return np.ascontiguousarray(img.copy())

    def _draw_keypoints(self, img: np.ndarray, obs: list[float]) -> None:
        """Draw corners exactly as the observation encodes them."""
        cv2 = self._cv()
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        cv2.drawMarker(img, (cx, cy), _DIM, cv2.MARKER_CROSS, 18, 1)

        seen_pts = []
        for i in range(KEYPOINT_COUNT):
            u_n, v_n = obs[2 * i], obs[2 * i + 1]
            visible = obs[KEYPOINT_COUNT * 2 + i] > 0.5
            if not visible or u_n == NOT_SEEN:
                continue
            px, py = int(u_n * w), int(v_n * h)
            seen_pts.append((i, px, py))
            # Outer ring 0-3 vs inner ring 4-7: the distinction the observation
            # preserves by keeping identity order.
            colour = _ACCENT if i < 4 else _GOOD
            cv2.circle(img, (px, py), 5, colour, -1)
            cv2.putText(img, str(i), (px + 7, py - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)

        if seen_pts:
            mx = int(sum(p[1] for p in seen_pts) / len(seen_pts))
            my = int(sum(p[2] for p in seen_pts) / len(seen_pts))
            cv2.circle(img, (mx, my), 9, (255, 255, 255), 2)
            cv2.line(img, (cx, cy), (mx, my), (255, 255, 255), 1)
        n_seen = len(seen_pts)
        colour = _GOOD if n_seen == 8 else (_WARN if n_seen else _BAD)
        cv2.putText(img, f'{n_seen}/8 keypoints', (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)

    def _draw_candidates(self, img: np.ndarray, shared_data: dict) -> None:
        """Outline every raw YOLO box, including ones the policy never gets.

        A red box with no keypoints in it means the detector *did* find the gate
        and selection discarded it. An empty frame with no box at all means the
        detector genuinely saw nothing. Those need different fixes, and without
        this they look identical.
        """
        cv2 = self._cv()
        cands = (shared_data.get('gate_candidates') or {}).get('items') or ()
        det_frame = (shared_data.get('gate_detection') or {}).get('frame_id')
        cand_frame = (shared_data.get('gate_candidates') or {}).get('frame_id')
        delivered = det_frame is not None and det_frame == cand_frame
        for item in cands:
            bbox = item.get('bbox_px')
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = (int(v) for v in bbox)
            colour = _DIM if delivered else _BAD
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 1)
            conf = item.get('confidence')
            if conf is not None:
                cv2.putText(img, f'{float(conf):.2f}', (x1, max(10, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1,
                            cv2.LINE_AA)
        if cands and not delivered:
            cv2.putText(img, f'{len(cands)} box(es) REJECTED', (8, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _BAD, 2, cv2.LINE_AA)

    def _draw_snake(self, img: np.ndarray, shared_data: dict) -> None:
        """Snake gate detection's quadrilaterals, in magenta.

        Observe-only, so this is a second opinion drawn over the same frame the
        network sees. Where magenta appears with no YOLO keypoints, the colour
        method found a gate the pose model missed, and vice versa.
        """
        cv2 = self._cv()
        snake = shared_data.get('snake_gate') or {}
        items = snake.get('items') or ()
        for g in items:
            corners = g.get('corners_px')
            if not corners or len(corners) < 3:
                continue
            pts = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, _SNAKE, 2, cv2.LINE_AA)
            cf = g.get('color_fitness')
            if cf is not None:
                x, y = int(corners[0][0]), int(corners[0][1])
                cv2.putText(img, f'cf {float(cf):.2f}', (x, max(10, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, _SNAKE, 1,
                            cv2.LINE_AA)
        if snake:
            n = int(snake.get('n', 0) or 0)
            ms = snake.get('elapsed_ms')
            label = f'snake {n} gate(s)'
            if ms is not None:
                label += f'  {float(ms):.1f} ms'
            cv2.putText(img, label, (8, img.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        _SNAKE if n else _DIM, 1, cv2.LINE_AA)

    def _draw_gatenet(self, img: np.ndarray, shared_data: dict) -> None:
        """Friend's four inner-aperture corners (TL/TR/BR/BL).

        Drawn instead of the YOLO overlay whenever GateNet is the panel source
        — either as the live detector or as the ``--gatenet`` view.
        """
        cv2 = self._cv()
        gn = shared_data.get('gatenet') or {}
        corners = gn.get('corners_px') or ()
        scores = gn.get('scores') or ()
        names = gn.get('names') or ('TL', 'TR', 'BR', 'BL')
        thresh = float(gn.get('threshold') or 0.45)
        drawn = 0
        pts_seen = []
        for i, corner in enumerate(corners):
            if not corner or len(corner) < 2:
                continue
            score = float(scores[i]) if i < len(scores) else 0.0
            x, y = int(round(corner[0])), int(round(corner[1]))
            if score < thresh:
                cv2.drawMarker(img, (x, y), _DIM, cv2.MARKER_TILTED_CROSS, 10, 1)
                continue
            cv2.drawMarker(img, (x, y), _GATENET, cv2.MARKER_DIAMOND, 14, 2)
            cv2.putText(img, f'{names[i]} {score:.2f}', (x + 8, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, _GATENET, 1, cv2.LINE_AA)
            pts_seen.append((x, y))
            drawn += 1
        if len(pts_seen) >= 2:
            poly = np.asarray(pts_seen, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [poly], True, _GATENET, 1, cv2.LINE_AA)
        if gn:
            n = int(gn.get('n_seen', drawn) or 0)
            ms = gn.get('elapsed_ms')
            label = f'gatenet {n}/4 inner'
            if ms is not None:
                label += f'  {float(ms):.1f} ms'
            colour = _GATENET if n >= 2 else _DIM
            cv2.putText(img, label, (8, img.shape[0] - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    def _sidebar(self, shared_data: dict, obs: list[float],
                 height: int) -> np.ndarray:
        cv2 = self._cv()
        # Grow past the frame height rather than silently clipping: a HUD that
        # hides the command it is meant to show is worse than no HUD.
        panel = np.zeros((max(height, 620), PANEL_W, 3), dtype=np.uint8)
        limit = panel.shape[0]
        y = [16]

        def line(text: str, colour=_WHITE, indent: int = 8) -> None:
            if y[0] > limit - 6:
                return
            cv2.putText(panel, text, (indent, y[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, _FONT, colour, 1, cv2.LINE_AA)
            y[0] += _LINE

        def header(text: str) -> None:
            y[0] += 4
            line(text, _ACCENT)

        header('CORNERS  (normalised, -1 = not seen)')
        for i in range(KEYPOINT_COUNT):
            u_n, v_n = obs[2 * i], obs[2 * i + 1]
            vis = obs[KEYPOINT_COUNT * 2 + i] > 0.5
            ring = 'out' if i < 4 else 'in '
            colour = _WHITE if vis else _DIM
            line(f'kp{i} {ring}  u{_fmt(u_n)}  v{_fmt(v_n)}  '
                 f'{"seen" if vis else "----"}', colour)

        header('ATTITUDE + RATES')
        src, s_colour = self._attitude_source(shared_data)
        line(f'source     {src}', s_colour)
        base = KEYPOINT_COUNT * 3
        line(f'roll       {_fmt(obs[base + 0], 3)} rad '
             f'({math.degrees(obs[base + 0]):+6.1f} deg)')
        line(f'pitch      {_fmt(obs[base + 1], 3)} rad '
             f'({math.degrees(obs[base + 1]):+6.1f} deg)')
        line(f'gyro x/y/z {_fmt(obs[base + 2])} {_fmt(obs[base + 3])} '
             f'{_fmt(obs[base + 4])}')

        if len(obs) > FEATURE_DIM:
            header('COURSE CONTEXT')
            ctx = obs[FEATURE_DIM:]
            hot = [i for i in range(N_GATES) if ctx[i] > 0.5]
            line(f'active gate  {hot[0] if hot else "none"}'
                 f'   progress {_fmt(ctx[N_GATES])}')

        race = shared_data.get('race_status') or {}
        header('RACE')
        line(f'active_gate  {race.get("active_gate", "-")}')

        pose = shared_data.get('race_pose')
        if pose:
            header('LS GATE POSE (classical planner)')
            line(f'mode       {pose.get("mode", "-")}')
            line(f'range      {_fmt(pose.get("range_m"))} m')
            line(f'lateral    {_fmt(pose.get("lateral_m"))} m')
            line(f'vertical   {_fmt(pose.get("vertical_m"))} m')
            line(f'bearing    {_fmt(pose.get("bearing_rad"))} rad')
            resid = pose.get('residual_m')
            ok = isinstance(resid, (int, float)) and float(resid) < 0.6
            line(f'residual   {_fmt(resid)} m', _GOOD if ok else _BAD)

        header('COMMAND OUT')
        tgt = shared_data.get('planner_target') or {}
        line(f'planner    {shared_data.get("planner_mode", "-")}')
        line(f'authority  {shared_data.get("control_authority", "-")}',
             _WARN if shared_data.get('control_authority') == 'human' else _GOOD)
        for name, key in zip(
            LABEL_NAMES, ('thrust', 'roll_rate', 'pitch_rate', 'yaw_rate')
        ):
            line(f'{name:10s} {_fmt(tgt.get(key), 3)}')
        return panel

    @staticmethod
    def _attitude_source(shared_data: dict) -> tuple[str, tuple]:
        raw = shared_data.get('attitude_raw') or {}
        ctrl = shared_data.get('control_output') or {}
        if raw.get('roll') is not None:
            return 'sim ATTITUDE', _GOOD
        if ctrl.get('ahrs_roll') is not None:
            return 'controller AHRS', _GOOD
        return 'EKF (DRIFTS!)', _BAD

    # ------------------------------------------------------------------
    def render(self, shared_data: dict) -> Optional[np.ndarray]:
        from policy_planner import observation_from_shared

        obs = observation_from_shared(
            shared_data, with_context=self.with_context
        )
        img = self._frame(shared_data)
        # One detector on the camera: GateNet diamonds *or* YOLO keypoints,
        # never both. Presence of a gatenet payload is the switch — that is
        # published when the backend is gatenet or when --gatenet is on.
        if shared_data.get('gatenet'):
            self._draw_gatenet(img, shared_data)
        else:
            self._draw_candidates(img, shared_data)
            self._draw_snake(img, shared_data)
            self._draw_keypoints(img, obs)
        side = self._sidebar(shared_data, obs, img.shape[0])
        if side.shape[0] != img.shape[0]:
            pad = np.zeros(
                (side.shape[0] - img.shape[0], img.shape[1], 3), dtype=np.uint8
            )
            img = np.vstack([img, pad])
        return np.hstack([img, side])

    def show(self, shared_data: dict) -> bool:
        """Draw one frame. Returns False once the window has been closed."""
        if not self._enabled:
            return False
        try:
            cv2 = self._cv()
            canvas = self.render(shared_data)
            if canvas is None:
                return True
            if self.scale != 1.0:
                canvas = cv2.resize(
                    canvas, None, fx=self.scale, fy=self.scale,
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imshow(self.WINDOW, canvas)
            if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                self.close()
                return False
        except Exception as exc:  # noqa: BLE001 - a HUD must never kill a flight
            print(f'[PANEL] disabled: {exc}', flush=True)
            self._enabled = False
            return False
        return True
