"""Multi-gate bearing memory for post-pass look-ahead.

At the start of a Q2 lap the camera often sees the first several gates at once.
This module PnP-solves (or size-orders) every pose instance, sorts them
near→far, and remembers only the *closest* upcoming gates as the course
queue. After a pass we look toward that remembered near next gate — not the
distant end-of-course gate that often appears through the opening.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

import camera_model as cm
from vision.yolo_pnp import solve_corners_pnp

# Gates farther than this are treated as background / end-of-course and are
# never latched as the immediate next target.
MAX_NEAR_COURSE_RANGE_M = 18.0
# After a pass, ignore live detections much farther than the remembered next.
MAX_VISIBLE_NEXT_RANGE_FACTOR = 1.55
MAX_VISIBLE_NEXT_RANGE_SLACK_M = 10.0
# Contact / course bearings must not command a sky stare. telem_031946 latched
# COURSE_BEARING v=-0.73 then YOLO chased gv=18 off the top of the frame.
MAX_CONTACT_LOOK_UP = -0.35
MAX_CONTACT_LOOK_DOWN = 0.45
# Post-pass IBVS may only lock a live gate below this image line.
MAX_POST_PASS_LOCK_NORMALIZED_Y = -0.55


@dataclass(frozen=True)
class GateBearing:
    """Where to look for one upcoming gate, in camera/body cues."""

    horizontal_normalized: float  # +right in image / body yaw sense
    vertical_normalized: float  # +down in image; negative => look/climb up
    yaw_offset_rad: float  # body: +right of forward
    pitch_offset_rad: float  # body: +down from forward
    range_m: float
    confidence: float
    source: str = "pnp"


@dataclass
class GateObservation:
    range_m: float
    horizontal_normalized: float
    vertical_normalized: float
    yaw_offset_rad: float
    pitch_offset_rad: float
    confidence: float
    source: str
    center_xy: tuple[float, float]


def _image_bearing(
    center_x: float,
    center_y: float,
    frame_width: float,
    frame_height: float,
) -> tuple[float, float, float, float]:
    """Return (nx, ny, yaw_rad, pitch_rad) from a pixel center."""
    half_w = max(frame_width * 0.5, 1.0)
    half_h = max(frame_height * 0.5, 1.0)
    nx = float(np.clip((center_x - half_w) / half_w, -1.5, 1.5))
    ny = float(np.clip((center_y - half_h) / half_h, -1.5, 1.5))
    # Pinhole rays through the calibrated intrinsics.
    ray_cam = np.array(
        [
            (center_x - cm.CX) / cm.FX,
            (center_y - cm.CY) / cm.FY,
            1.0,
        ],
        dtype=np.float64,
    )
    ray_body = cm.cam_to_body(ray_cam)
    horiz = math.hypot(float(ray_body[0]), float(ray_body[1]))
    yaw = math.atan2(float(ray_body[1]), max(float(ray_body[0]), 1e-6))
    pitch = math.atan2(float(ray_body[2]), max(horiz, 1e-6))
    return nx, ny, yaw, pitch


def observe_pose_candidates(
    candidates: Sequence[object],
    frame_width: int,
    frame_height: int,
    *,
    require_hsv: bool = True,
    min_confidence: float = 0.25,
) -> list[GateObservation]:
    """Build near→far observations from YOLO pose instances."""
    observations: list[GateObservation] = []
    for candidate in candidates:
        box = getattr(candidate, "box", None)
        if box is None:
            continue
        if require_hsv and not bool(getattr(candidate, "hsv_confirmed", True)):
            continue
        confidence = float(getattr(box, "confidence", 0.0))
        if confidence < min_confidence:
            continue
        keypoints = getattr(candidate, "keypoints", None)
        center = getattr(box, "center", None)
        if center is None:
            continue
        center_x, center_y = float(center[0]), float(center[1])
        nx, ny, yaw, pitch = _image_bearing(
            center_x, center_y, frame_width, frame_height
        )
        range_m = None
        source = "image"
        if keypoints is not None:
            solved = solve_corners_pnp(
                keypoints,
                confidence=confidence,
                bbox=getattr(box, "bbox", None),
            )
            if solved is not None and solved.solved:
                body = solved.center_body()
                range_m = float(solved.range_m)
                yaw = math.atan2(float(body[1]), max(float(body[0]), 1e-6))
                horiz = math.hypot(float(body[0]), float(body[1]))
                pitch = math.atan2(float(body[2]), max(horiz, 1e-6))
                source = "pnp"
        if range_m is None:
            area = max(float(getattr(box, "area", 0.0)), 1.0)
            # Monotonic stand-in: larger box => nearer gate.
            range_m = 1000.0 / math.sqrt(area)
        observations.append(
            GateObservation(
                range_m=range_m,
                horizontal_normalized=nx,
                vertical_normalized=ny,
                yaw_offset_rad=yaw,
                pitch_offset_rad=pitch,
                confidence=confidence,
                source=source,
                center_xy=(center_x, center_y),
            )
        )
    observations.sort(key=lambda item: item.range_m)
    return observations


def clamp_contact_vertical(vertical_normalized: float) -> float:
    """Limit look-up/down so contact bearings cannot command a sky chase."""
    return float(
        np.clip(
            vertical_normalized,
            MAX_CONTACT_LOOK_UP,
            MAX_CONTACT_LOOK_DOWN,
        )
    )


def _bearing_from_observation(item: GateObservation) -> GateBearing:
    return GateBearing(
        horizontal_normalized=item.horizontal_normalized,
        vertical_normalized=clamp_contact_vertical(item.vertical_normalized),
        yaw_offset_rad=item.yaw_offset_rad,
        pitch_offset_rad=item.pitch_offset_rad,
        range_m=item.range_m,
        confidence=item.confidence,
        source=item.source,
    )


def post_pass_lock_allowed(
    detection: object,
    *,
    expected_range_m: Optional[float] = None,
) -> bool:
    """Reject post-pass IBVS locks on sky/far speck gates (031946 gv≈18)."""
    if detection is None or not bool(getattr(detection, "found", False)):
        return False
    if bool(getattr(detection, "predicted", False)):
        return False
    ny = getattr(detection, "normalized_y", None)
    if ny is not None and float(ny) <= MAX_POST_PASS_LOCK_NORMALIZED_Y:
        return False
    distance = getattr(detection, "distance_m", None)
    if (
        expected_range_m is not None
        and expected_range_m > 0.0
        and distance is not None
    ):
        try:
            range_m = float(distance)
        except (TypeError, ValueError):
            range_m = float("nan")
        if math.isfinite(range_m):
            max_range = max(
                expected_range_m + MAX_VISIBLE_NEXT_RANGE_SLACK_M,
                expected_range_m * MAX_VISIBLE_NEXT_RANGE_FACTOR,
            )
            if range_m > max_range:
                return False
    return True


def _plausible_next_gate(
    primary: GateObservation,
    candidate: GateObservation,
    *,
    max_range_m: float,
) -> bool:
    """Reject through-opening end gates when the primary is already close."""
    if candidate.range_m > max_range_m:
        return False
    gap = candidate.range_m - primary.range_m
    # telem_031605: at approach=2.3 m CONTACT became 26.5 m (end course
    # through the opening). A real next gate stays within ~12–18 m of the
    # current one when we are already close.
    if primary.range_m <= 8.0 and gap > 12.0:
        return False
    if primary.range_m <= 5.0 and candidate.range_m > 22.0:
        return False
    # A next gate pinned to the top of the image is not usable contact.
    if candidate.vertical_normalized <= MAX_POST_PASS_LOCK_NORMALIZED_Y:
        return False
    return True


def near_course_observations(
    observations: Sequence[GateObservation],
    *,
    max_range_m: float = MAX_NEAR_COURSE_RANGE_M,
    max_upcoming: int = 2,
) -> list[GateObservation]:
    """Keep only the closest current + upcoming gates; drop end-of-course dots."""
    if not observations:
        return []
    current = observations[0]
    near = [current]
    for item in observations[1:]:
        if not _plausible_next_gate(near[0], item, max_range_m=max_range_m):
            continue
        # Further look-ahead must not leap past the nearest next toward a
        # distant finish-line gate.
        if len(near) >= 2 and item.range_m > 1.6 * near[-1].range_m + 8.0:
            continue
        near.append(item)
        if len(near) >= 1 + max_upcoming:
            break
    return near


class GateBearingTable:
    """Track the nearest two near-course gates and latch post-pass bearings.

    Every frame refreshes ``live_primary`` (approach target) and
    ``live_secondary`` (keep-in-view contact). The ``upcoming`` queue is a
    frozen post-pass memory and may stop refreshing when the primary is close;
    live secondary contact continues whenever the second gate is visible.
    """

    def __init__(self, stale_after_s: float = 12.0):
        self.stale_after_s = stale_after_s
        self.upcoming: list[GateBearing] = []
        self.last_update: float = 0.0
        self.last_visible_count: int = 0
        self.last_source: str = "none"
        self.last_observations: list[GateObservation] = []
        self.last_near_observations: list[GateObservation] = []
        self.live_primary: Optional[GateObservation] = None
        self.live_secondary: Optional[GateObservation] = None
        self._frozen: bool = False
        # Locked-in next-gate range from the early close multi-gate view.
        self.expected_next_range_m: Optional[float] = None
        # Bearing we are looking for *now* after a pass (may differ from the
        # remaining upcoming[0], which is the gate after next).
        self._active_look: Optional[GateBearing] = None

    def freeze(self) -> None:
        """Stop live refreshes from replacing the latched course queue."""
        self._frozen = True

    def unfreeze(self) -> None:
        self._frozen = False
        self._active_look = None

    @property
    def has_contact_pair(self) -> bool:
        return self.live_primary is not None and self.live_secondary is not None

    def contact_secondary_bearing(self) -> Optional[GateBearing]:
        """Live second-nearest gate for keep-in-view while approaching first."""
        if self.live_secondary is None:
            return None
        return _bearing_from_observation(self.live_secondary)

    def update(
        self,
        observations: Sequence[GateObservation],
        now: float,
        *,
        freeze_when_near_m: float = 8.0,
    ) -> Optional[GateBearing]:
        """Refresh nearest-two contact and (when not frozen) the course queue.

        ``observations`` must already be sorted near→far. Index 0 is the
        closest / current gate; only the next one or two nearby gates enter
        the look-ahead queue. Distant end-of-course gates are ignored.
        """
        self.last_visible_count = len(observations)
        self.last_observations = list(observations)
        near = near_course_observations(observations)
        self.last_near_observations = near
        previous_secondary = self.live_secondary
        self.live_primary = near[0] if near else None
        new_secondary = near[1] if len(near) >= 2 else None
        # When the primary is close, keep the early CONTACT latch if the live
        # second gate vanished or was replaced by a farther through-opening
        # speck (031605: 22.9 m latch → 26.5 m end gate).
        primary_close = (
            self.live_primary is not None
            and self.live_primary.range_m <= freeze_when_near_m
        )
        if previous_secondary is not None and primary_close:
            if new_secondary is None:
                self.live_secondary = previous_secondary
            elif new_secondary.range_m > previous_secondary.range_m + 5.0:
                self.live_secondary = previous_secondary
            else:
                self.live_secondary = new_secondary
        else:
            self.live_secondary = new_secondary
        # Freeze / active post-pass look owns expected range; do not let a
        # live far detection rewrite it.
        if (
            self.live_secondary is not None
            and not self._frozen
            and self._active_look is None
        ):
            self.expected_next_range_m = self.live_secondary.range_m
        if self._frozen:
            return self.peek_next(now)
        if len(near) < 2:
            return self.peek_next(now)
        # Once the nearest gate is close, further *queue* refreshes are
        # dominated by partial/edge false positives and far gates seen
        # through the opening. Live secondary contact above still updates.
        if near[0].range_m <= freeze_when_near_m and self.upcoming:
            self._frozen = True
            return self.peek_next(now)
        upcoming = [_bearing_from_observation(item) for item in near[1:]]
        if self.upcoming:
            previous = self.upcoming[0]
            candidate = upcoming[0]
            # Reject a refresh that collapses a clear side bearing toward zero
            # or replaces a near next-gate with a much farther one.
            if (
                abs(previous.horizontal_normalized) >= 0.25
                and abs(candidate.horizontal_normalized)
                < 0.5 * abs(previous.horizontal_normalized)
            ):
                return self.peek_next(now)
            if candidate.range_m > max(
                previous.range_m + MAX_VISIBLE_NEXT_RANGE_SLACK_M,
                MAX_VISIBLE_NEXT_RANGE_FACTOR * previous.range_m,
            ):
                return self.peek_next(now)
        self.upcoming = upcoming
        self.expected_next_range_m = upcoming[0].range_m
        self.last_update = now
        self.last_source = upcoming[0].source if upcoming else "none"
        return upcoming[0]

    def peek_next(self, now: Optional[float] = None) -> Optional[GateBearing]:
        # Prefer the post-pass active look so we don't jump to gate-after-next
        # while still acquiring the nearest remembered next gate.
        if self._active_look is not None:
            return self._active_look
        if not self.upcoming:
            return None
        if (
            now is not None
            and self.last_update > 0.0
            and now - self.last_update > self.stale_after_s
        ):
            return None
        return self.upcoming[0]

    def consume_pass(self, now: Optional[float] = None) -> Optional[GateBearing]:
        """Advance after a scored pass; return the bearing to look at now."""
        if not self.upcoming:
            return None
        target = self.upcoming.pop(0)
        # Freeze so a noisy post-pass multi-detect cannot replace the rest of
        # the course queue (or flip the just-consumed bearing via peek).
        self._frozen = True
        self._active_look = target
        self.expected_next_range_m = target.range_m
        if now is not None:
            self.last_update = now
        return target

    def as_dict(self, now: Optional[float] = None) -> dict:
        nxt = self.peek_next(now)
        contact = self.contact_secondary_bearing()
        primary = self.live_primary
        return {
            "visible_count": self.last_visible_count,
            "near_count": len(self.last_near_observations),
            "upcoming_count": len(self.upcoming),
            "has_contact_pair": self.has_contact_pair,
            "expected_next_range_m": self.expected_next_range_m,
            "last_update": self.last_update,
            "last_source": self.last_source,
            "primary": None
            if primary is None
            else {
                "horizontal_normalized": primary.horizontal_normalized,
                "vertical_normalized": primary.vertical_normalized,
                "range_m": primary.range_m,
                "source": primary.source,
            },
            "contact": None
            if contact is None
            else {
                "horizontal_normalized": contact.horizontal_normalized,
                "vertical_normalized": contact.vertical_normalized,
                "range_m": contact.range_m,
                "source": contact.source,
            },
            "next": None
            if nxt is None
            else {
                "horizontal_normalized": nxt.horizontal_normalized,
                "vertical_normalized": nxt.vertical_normalized,
                "yaw_offset_rad": nxt.yaw_offset_rad,
                "pitch_offset_rad": nxt.pitch_offset_rad,
                "range_m": nxt.range_m,
                "confidence": nxt.confidence,
                "source": nxt.source,
            },
        }


def detection_from_observation(
    observation: GateObservation,
    frame_width: int,
    frame_height: int,
    timestamp: float,
    *,
    role: str = "visible_next",
) -> "GateDetection":
    """Build a steering detection from a ranged multi-gate observation."""
    from vision.gate_detector import GateDetection

    center_x, center_y = observation.center_xy
    side = float(
        np.clip(900.0 / max(observation.range_m, 1.0), 18.0, 160.0)
    )
    x1 = int(round(center_x - 0.5 * side))
    y1 = int(round(center_y - 0.5 * side))
    return GateDetection(
        found=True,
        center_x=float(center_x),
        center_y=float(center_y),
        normalized_x=float(observation.horizontal_normalized),
        normalized_y=float(observation.vertical_normalized),
        opening_width=side,
        opening_height=side,
        apparent_area=side * side,
        confidence=max(0.35, float(observation.confidence)),
        method=f"{role}_{observation.source}",
        distance_m=float(observation.range_m),
        frame_width=int(frame_width),
        frame_height=int(frame_height),
        bbox=(x1, y1, int(round(side)), int(round(side))),
        timestamp=timestamp,
        stable_frames=5,
        predicted=False,
    )


def select_visible_next_observation(
    observations: Sequence[GateObservation],
    *,
    look_horizontal: Optional[float] = None,
    expected_range_m: Optional[float] = None,
) -> Optional[GateObservation]:
    """Choose a *nearby* visible gate matching the remembered next target.

    Far end-of-course gates (often visible through the opening after a pass)
    are ignored so we keep looking toward the closest remembered next gate.
    """
    if not observations:
        return None
    pool = list(observations)
    # Reject gates already racing out the top of the frame — locking those
    # for IBVS after a pass lofted us into RECOVER then the ground (031605).
    pool = [item for item in pool if item.vertical_normalized > -0.70]
    if not pool:
        return None
    if expected_range_m is not None and expected_range_m > 0.0:
        max_range = max(
            expected_range_m + MAX_VISIBLE_NEXT_RANGE_SLACK_M,
            expected_range_m * MAX_VISIBLE_NEXT_RANGE_FACTOR,
        )
        nearby = [item for item in pool if item.range_m <= max_range]
        # If every live detection is the distant end gate, return None so the
        # navigator keeps open-loop aiming at the remembered near bearing.
        if not nearby:
            return None
        pool = nearby
    if look_horizontal is not None and abs(look_horizontal) >= 0.05:
        sided = [
            item
            for item in pool
            if item.horizontal_normalized * look_horizontal > 0.0
        ]
        if sided:
            pool = sided
        return min(
            pool,
            key=lambda item: (
                abs(item.horizontal_normalized - look_horizontal),
                item.range_m,
            ),
        )
    # No side cue: take the closest near-field gate.
    return min(pool, key=lambda item: item.range_m)


def draw_bearing_overlay(
    frame: np.ndarray,
    table: GateBearingTable,
    *,
    pending_horizontal: Optional[float] = None,
    pending_vertical: Optional[float] = None,
) -> np.ndarray:
    """Draw near→far gate indices, NEXT arrow, and latch HUD on the camera pane."""
    output = frame
    height, width = output.shape[:2]
    near = table.last_near_observations or table.last_observations[:3]
    near_ids = {id(item) for item in near}
    expected = table.expected_next_range_m
    for index, item in enumerate(table.last_observations):
        cx, cy = item.center_xy
        px = int(round(cx))
        py = int(round(cy))
        is_near = id(item) in near_ids or item in near
        is_next = (
            is_near
            and len(near) >= 2
            and item is near[1]
        )
        is_current = is_near and item is near[0]
        if is_next:
            color = (0, 255, 255)
            label = "CONTACT"
        elif is_current:
            color = (0, 220, 0)
            label = "APPROACH"
        elif item.range_m > MAX_NEAR_COURSE_RANGE_M:
            color = (80, 80, 200)
            label = "FAR"
        else:
            color = (255, 180, 80)
            label = f"G{index}"
        cv2.circle(output, (px, py), 7, color, 2, cv2.LINE_AA)
        cv2.putText(
            output,
            f"{label} {item.range_m:.1f}m",
            (px + 8, max(16, py - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    if len(near) >= 2:
        cv2.arrowedLine(
            output,
            (
                int(round(near[0].center_xy[0])),
                int(round(near[0].center_xy[1])),
            ),
            (
                int(round(near[1].center_xy[0])),
                int(round(near[1].center_xy[1])),
            ),
            (0, 255, 255),
            2,
            cv2.LINE_AA,
            tipLength=0.2,
        )

    contact = table.contact_secondary_bearing()
    nxt = contact if contact is not None else table.peek_next()
    hud_color = (0, 255, 255) if nxt is not None else (0, 0, 255)
    lines = [
        f"NEAREST-2 vis={table.last_visible_count} "
        f"pair={'yes' if table.has_contact_pair else 'no'} "
        f"queued={len(table.upcoming)} src={table.last_source}",
    ]
    if table.live_primary is not None:
        lines.append(
            f"APPROACH r={table.live_primary.range_m:.1f}m "
            f"h={table.live_primary.horizontal_normalized:+.2f}"
        )
    if expected is not None:
        lines.append(f"EXPECT contact ~{expected:.1f}m")
    if nxt is not None:
        lines.append(
            f"CONTACT h={nxt.horizontal_normalized:+.2f} "
            f"v={nxt.vertical_normalized:+.2f} "
            f"r={nxt.range_m:.1f}m"
        )
        # Aim mark where the latched next gate lives in the image.
        aim_x = int(round((nxt.horizontal_normalized + 1.0) * 0.5 * width))
        aim_y = int(round((nxt.vertical_normalized + 1.0) * 0.5 * height))
        cv2.drawMarker(
            output,
            (aim_x, aim_y),
            (0, 255, 255),
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=22,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
    else:
        lines.append("NEXT look NONE (need 2+ YOLO gates)")
    if pending_horizontal is not None:
        lines.append(
            f"NAV latch h={pending_horizontal:+.2f}"
            + (
                f" v={pending_vertical:+.2f}"
                if pending_vertical is not None
                else ""
            )
        )
    else:
        lines.append("NAV latch NONE")

    y = 42
    for line in lines:
        cv2.putText(
            output,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            hud_color,
            1,
            cv2.LINE_AA,
        )
        y += 18
    return output
