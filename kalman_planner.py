"""Dual-gate PnP + image IBVS hybrid (Q2_kalman).

Agent run 043043: forward lean was nearly zero while yaw slammed ±0.78 as
YOLO/PnP jumped between gates (gu 320→597). Now:
- EMA-smooth image errors
- stronger forward lean when framed
- reject huge yaw steps from lock jumps
"""

from __future__ import annotations

import math
import time

import numpy as np

import config
from control.pid import PIDConfig, PIDController


class KalmanDualGatePlanner:
    name = 'kalman_dual_gate'

    def __init__(self):
        max_yaw = min(config.YAW_RATE_MAX_RAD_S, math.radians(25.0))
        max_rate = 0.9
        self._max_lean = math.radians(
            getattr(config, 'KALMAN_MAX_LEAN_DEG', 12.0)
        )
        # kd=0: run 044123 saturated yaw at nx≈0.06 via D-term spikes.
        self._yaw_pid = PIDController(
            PIDConfig(
                kp=getattr(config, 'KALMAN_KP_YAW', 0.9),
                kd=0.0,
                output_min=-max_yaw,
                output_max=max_yaw,
            )
        )
        self._yaw_slew = math.radians(90.0)  # rad/s^2-ish limit on cmd change
        self._last_yaw_cmd = 0.0
        self._roll_pid = PIDController(
            PIDConfig(
                kp=2.2, kd=0.10, output_min=-max_rate, output_max=max_rate
            )
        )
        self._pitch_pid = PIDController(
            PIDConfig(
                kp=2.2, kd=0.10, output_min=-max_rate, output_max=max_rate
            )
        )
        self._last_t = None
        self._last_g1 = None
        self._last_g2 = None
        self._last_safety = None
        self._arm_z = None
        self._nx_f = 0.0
        self._ny_f = 0.0
        self._have_filt = False
        self._coast_until = 0.0
        self._seek_until = 0.0
        self._last_area_px = 0.0
        self._active_gate = None
        self._course_nx = None
        self._course_ny = None
        self._course_range = None
        self._course_latched = False
        self._pass_t = None

    def reset_episode(self):
        """Clear episode state after a sim reset / floor crash."""
        self._yaw_pid.reset()
        self._roll_pid.reset()
        self._pitch_pid.reset()
        self._last_yaw_cmd = 0.0
        self._last_t = None
        self._last_g1 = None
        self._last_g2 = None
        self._last_safety = None
        self._arm_z = None
        self._nx_f = 0.0
        self._ny_f = 0.0
        self._have_filt = False
        self._coast_until = 0.0
        self._seek_until = 0.0
        self._last_area_px = 0.0
        self._active_gate = None
        self._course_nx = None
        self._course_ny = None
        self._course_range = None
        self._course_latched = False
        self._pass_t = None

    def compute_target(self, shared_data):
        shared_data['planner_mode'] = self.name
        shared_data['post_pass_hunt'] = False
        now = time.monotonic()
        dt = 0.02 if self._last_t is None else max(1e-3, now - self._last_t)
        self._last_t = now

        # 070629: sim scored the pass while YOLO had already snapped to
        # u≈518–587. Force a forward coast and ignore edge junk.
        race = shared_data.get('race_status') or {}
        ag = race.get('active_gate')
        if ag is not None:
            try:
                ag_i = int(ag)
            except (TypeError, ValueError):
                ag_i = None
            if ag_i is not None:
                if self._active_gate is not None and ag_i > self._active_gate:
                    # Fresh post-pass coast/seek windows (replace, don't stack).
                    # 0928: longer coast/seek — straight bearings still need
                    # >12s when punch is soft / YOLO flickers.
                    self._coast_until = now + 3.2
                    self._seek_until = now + 16.0
                    self._pass_t = now
                    self._have_filt = False
                    self._last_area_px = 0.0
                    self._yaw_pid.reset()
                    self._last_yaw_cmd = 0.0
                    # Wait for vision course_bearing (may arrive one tick
                    # later). Level placeholder until then — not latched.
                    self._course_latched = False
                    self._course_nx = 0.0
                    self._course_ny = 0.0
                    self._course_range = None
                    g2_range = (
                        float(np.linalg.norm(self._last_g2))
                        if self._last_g2 is not None
                        else None
                    )
                    # Next gate was gate2 before the score — keep aiming there.
                    if self._last_g2 is not None and (
                        g2_range is not None and g2_range <= 8.0
                    ):
                        self._last_g1 = self._last_g2.copy()
                        self._last_g2 = None
                    else:
                        # Drop stale/far body fixes so seek flies level.
                        self._last_g1 = None
                        self._last_g2 = None
                self._active_gate = ag_i

        # Vision may publish course_bearing a tick after active_gate bumps
        # (0825: h=+0.07/3.1m logged but planner had already latched 0,0).
        if (
            not self._course_latched
            and now < self._seek_until
        ):
            bearing = shared_data.get('course_bearing') or {}
            try:
                self._course_nx = float(bearing['nx'])
                self._course_ny = float(
                    np.clip(float(bearing['ny']), -0.20, 0.05)
                )
                try:
                    self._course_range = float(bearing.get('range_m'))
                except (TypeError, ValueError):
                    self._course_range = None
                self._course_latched = True
            except (KeyError, TypeError, ValueError):
                aim = self._last_g1
                pass_age = (
                    (now - self._pass_t) if self._pass_t is not None else 0.0
                )
                if (
                    aim is not None
                    and pass_age >= 0.40
                    and float(np.linalg.norm(aim)) <= 8.0
                ):
                    fwd = max(0.4, float(aim[0]))
                    self._course_nx = float(
                        np.clip(float(aim[1]) / fwd, -0.70, 0.70)
                    )
                    self._course_ny = float(
                        np.clip(float(aim[2]) / fwd, -0.20, 0.05)
                    )
                    self._course_range = float(np.linalg.norm(aim))
                    self._course_latched = True
        # 0926: latched h=+0.12 @ 4.2m but live nearest was ~2.4–2.7m at
        # h=-0.40 — retarget when a nearer live primary/contact appears.
        post_pass = (
            self._pass_t is not None
            and now < self._seek_until
            and (now - self._pass_t) <= 14.0
        )
        shared_data['post_pass_hunt'] = bool(post_pass)
        if post_pass:
            self._maybe_retarget_live_bearing(shared_data, now)
            # Gate contact during seek ≈ almost through; punch, don't hover.
            col = shared_data.get('collision') or {}
            if col.get('id') == 1001:
                try:
                    impulse = float(col.get('impulse') or 0.0)
                    ts_ns = int(col.get('ts') or 0)
                    age_s = (
                        (time.time_ns() - ts_ns) * 1e-9 if ts_ns else 999.0
                    )
                except (TypeError, ValueError):
                    impulse, age_s = 0.0, 999.0
                if impulse >= 1.5 and 0.0 <= age_s <= 0.6:
                    self._coast_until = max(self._coast_until, now + 1.8)
                    self._seek_until = max(self._seek_until, now + 4.0)
        coasting = now < self._coast_until

        dual = shared_data.get('dual_gate_pnp') or {}
        ekf = shared_data.get('ekf_state') or {}
        attitude = shared_data.get('attitude') or {}
        position = shared_data.get('position_ned') or {}

        fresh = (
            dual.get('gate1_body') is not None
            and dual.get('gate1_norm_x') is not None
        )
        seeking = bool(now < self._seek_until)
        if dual.get('gate1_body') is not None:
            g1_cand = np.asarray(
                dual['gate1_body'], dtype=np.float64
            ).reshape(3)
            # 0819: during seek, far dual gate1 (~13 m) was overwriting aim
            # and LEARN showed range=13 while phase=body_seek.
            if not (seeking and float(np.linalg.norm(g1_cand)) > 6.5):
                self._last_g1 = g1_cand
            else:
                fresh = False
        if dual.get('gate2_body') is not None:
            g2_cand = np.asarray(
                dual['gate2_body'], dtype=np.float64
            ).reshape(3)
            if not (seeking and float(np.linalg.norm(g2_cand)) > 8.0):
                self._last_g2 = g2_cand
        # 0811: right after a score, dual_pnp re-locked a ~10 m far gate and
        # aborted body-seek toward the real next (~3 m) gate. Ignore far PnP
        # while the post-pass seek window is open.
        if (
            fresh
            and self._last_g1 is not None
            and seeking
            and float(np.linalg.norm(self._last_g1)) > 6.5
        ):
            fresh = False

        roll = float(attitude.get('roll', 0.0))
        pitch = float(attitude.get('pitch', 0.0))
        yaw = float(attitude.get('yaw', 0.0))
        if not math.isfinite(yaw):
            yaw = 0.0

        z_ned = position.get('z')
        if z_ned is not None and math.isfinite(float(z_ned)):
            z_ned = float(z_ned)
            if self._arm_z is None:
                self._arm_z = z_ned
        else:
            z_ned = None

        # YOLO bbox fallback when PnP drops near the gate (044425: lost
        # DUAL_PNP at ~20k px while the detector still had the target).
        det = shared_data.get('gate_detection') or {}
        det_nx = det_ny = None
        area_px = float(det.get('area_px') or 0.0)
        # 0812: dual-only lock at ~19 m with no YOLO dived into the floor.
        # Far PnP without an image lock is not trustworthy enough to chase.
        if (
            fresh
            and self._last_g1 is not None
            and area_px < 2500.0
            and float(np.linalg.norm(self._last_g1)) > 12.0
        ):
            fresh = False
        # 064800: coast rejected tiny specks but then last_area fell to ~5k
        # and the next speck was accepted. Only grow/hold last_area.
        speck = bool(
            self._last_area_px >= 20000.0
            and 0.0 < area_px < 0.30 * self._last_area_px
        )
        if coasting and speck:
            area_px = 0.0
            det = {}
        if det.get('center_px') is not None and area_px > 0.0:
            try:
                from camera_model import CX, CY, WIDTH, HEIGHT

                cx, cy = det['center_px']
                det_nx = (float(cx) - CX) / (WIDTH * 0.5)
                det_ny = (float(cy) - CY) / (HEIGHT * 0.5)
                # 0750: YOLO locked a flat bottom-bar at v≈332 mid-approach.
                # 080022: a near-gate 52k box was also "flat" (wide perspective)
                # and got rejected → hover → stale-PnP punch into the floor.
                # Only reject small flat bars / low edge junk, not FOV fillers.
                bbox = det.get('bbox_px')
                flat = False
                if bbox is not None and len(bbox) >= 4:
                    bw = max(1.0, float(bbox[2]) - float(bbox[0]))
                    bh = max(1.0, float(bbox[3]) - float(bbox[1]))
                    flat = bool(bw / bh > 2.8)
                # 0930: post-pass right gate grew at cy≈312–339 and was wiped
                # by 0.82*H (=295) before ny_lim — LEARN stayed body_seek
                # with area~1–2k while YOLO still had u≈501–556.
                bot_cut = (
                    0.94 * HEIGHT
                    if (coasting or seeking)
                    else 0.82 * HEIGHT
                )
                bottom = bool(cy > bot_cut)
                if (bottom and area_px < 90000.0) or (
                    flat and area_px < 25000.0
                ):
                    det_nx = det_ny = None
                    area_px = 0.0
                    det = {}
            except (TypeError, ValueError):
                det_nx = det_ny = None
        # Drop opposite-side locks during right-biased post-pass hunt
        # (0929: left u≈295 stole from right u≈465 after h≈+0.04).
        # Also apply during coast/seek right after a pass even if latch is
        # late — default hunt is right for gate 2.
        if (coasting or seeking) and det_nx is not None:
            course_side = float(self._course_nx or 0.0)
            if not self._course_latched or abs(course_side) < 0.12:
                course_side = 0.28
            if course_side > 0.10 and float(det_nx) < -0.05:
                det_nx = det_ny = None
                area_px = 0.0
                det = {}
                fresh = False
        # Drop low boxes while seeking; allow mid-low near-course.
        # 0930: right locks at cy≈279–315 (ny≈0.55–0.75) were wiped by
        # ny_lim=0.50 → never entered yolo_fallback, seek expired → hover.
        if (coasting or seeking) and det_ny is not None and det_nx is not None:
            prefer_nx = float(self._course_nx or 0.0)
            if abs(prefer_nx) < 0.12:
                prefer_nx = 0.28
            near_pref = bool(
                self._course_latched
                and abs(float(det_nx) - prefer_nx) <= 0.50
            )
            if near_pref and prefer_nx >= 0.18:
                ny_lim = 0.75
            elif near_pref:
                ny_lim = 0.60
            else:
                ny_lim = 0.35
            if float(det_ny) > ny_lim:
                det_nx = det_ny = None
                area_px = 0.0
                det = {}
                fresh = False
        # Post-pass edge snaps are not the next gate — unless they sit near
        # the latched course bearing (0843: next gate often still off-center).
        # 0929: gate at nx≈+0.58 with prefer=+0.32 was wiped by >0.45.
        if (coasting or seeking) and det_nx is not None and abs(det_nx) > 0.70:
            near_latched = bool(
                self._course_latched
                and self._course_nx is not None
                and abs(float(det_nx) - self._course_nx) <= 0.55
            )
            if not near_latched:
                det_nx = det_ny = None
                area_px = 0.0
                det = {}
                fresh = False
        elif (
            (coasting or seeking)
            and det_nx is not None
            and abs(det_nx) > 0.45
            and not (
                self._course_latched
                and self._course_nx is not None
                and abs(float(det_nx) - float(self._course_nx)) <= 0.55
            )
        ):
            det_nx = det_ny = None
            area_px = 0.0
            det = {}
            fresh = False
        # 0917: floor of 2000 wiped real next-gate boxes at ~750–1800 px
        # after near_course was lowered to 500.
        if (coasting or seeking) and 0.0 < area_px < 500.0:
            det_nx = det_ny = None
            area_px = 0.0
            det = {}
            fresh = False
        # 0826: after a good 3.1 m bearing, YOLO locked a 110k full-frame
        # remnant of the gate behind us and punched into freefall. The real
        # next gate grows from smaller; reject FOV-filling boxes in seek.
        if seeking and area_px >= 80000.0:
            det_nx = det_ny = None
            area_px = 0.0
            det = {}
            fresh = False
        if area_px >= max(8000.0, 0.5 * self._last_area_px):
            self._last_area_px = area_px
        if coasting and speck:
            fresh = False
            det_nx = det_ny = None
            area_px = 0.0
        # 0917: after pass, dual_pnp stayed "fresh" and stole aim from a
        # good ~2–3k YOLO lock near COURSE_BEARING → body_seek → floor.
        if (
            fresh
            and (seeking or coasting)
            and self._course_latched
            and det_nx is not None
            and area_px >= 500.0
            and abs(float(det_nx) - float(self._course_nx)) <= 0.55
        ):
            fresh = False

        if not fresh:
            # 0843: had COURSE_BEARING h=+0.23/2.8m but ignored YOLO at
            # area~2–4.5k near that bearing → coast blind → hover → floor.
            near_course = False
            if (
                det_nx is not None
                and self._course_latched
                and self._course_nx is not None
                and (seeking or coasting)
            ):
                near_course = bool(
                    # 0906: next gate at ~620 px near bearing was ignored.
                    area_px >= 500.0
                    and abs(float(det_nx) - self._course_nx) <= 0.55
                    # 0930: growing right gate sat at cy≈279–315 after pass.
                    and float(det_ny or 0.0) <= 0.75
                    and float(det_ny or 0.0) >= -0.55
                )
                # Same-side centered grower even if not sitting on the bearing.
                # 0929: with h≈+0.04, left u≈295 stole lock from right u≈465.
                # Treat near-straight bearings as right-biased for gate-2.
                course_side = float(self._course_nx)
                if abs(course_side) < 0.12:
                    course_side = 0.28
                same_side = bool(
                    area_px >= 2000.0
                    and abs(float(det_nx)) <= 0.40
                    and float(det_ny or 0.0) <= 0.40
                    and float(det_ny or 0.0) >= -0.55
                    and float(det_nx) * course_side >= 0.0
                )
            else:
                same_side = False
            # 0910: wrong-side locks scraped; 0912: bearing-only missed the
            # real centered next gate. Accept near_course OR same-side grower.
            accept_fallback = (
                det_nx is not None
                and det.get('predicted') is not True
                and (
                    near_course
                    or same_side
                    or (
                        not coasting
                        and not seeking
                        and area_px >= 3000.0
                        and abs(float(det_nx)) <= 0.55
                    )
                    or (
                        (coasting or seeking)
                        and not self._course_latched
                        and area_px >= 800.0
                        and abs(float(det_nx)) <= 0.55
                        and float(det_ny or 0.0) <= 0.40
                        and float(det_ny or 0.0) >= -0.55
                    )
                )
            )
            if accept_fallback:
                # 0849: keep hunting while the next gate is in view.
                # 0904: ending coast here left last_g1=None → hover as soon as
                # the YOLO flicker dropped. Keep coast; only extend seek.
                if seeking or coasting:
                    self._seek_until = max(self._seek_until, now + 4.0)
                nx_raw = float(np.clip(det_nx, -1.2, 1.2))
                ny_raw = float(np.clip(det_ny or 0.0, -1.2, 1.2))
                # 0928: default 6.0 blocked commit; prefer latched course range.
                if self._course_range is not None:
                    range_m = float(self._course_range)
                elif self._last_g1 is not None:
                    range_m = float(np.linalg.norm(self._last_g1))
                else:
                    range_m = 4.0
                return self._track_image(
                    shared_data,
                    dt,
                    nx_raw,
                    ny_raw,
                    range_m,
                    roll,
                    pitch,
                    yaw,
                    z_ned,
                    ekf,
                    source='yolo_fallback',
                    area_px=area_px,
                )
            # 064333: full-frame centred gate then SEARCH→hover→fall.
            # Keep punching for a short coast after a close sighting.
            if coasting:
                self._latch_course_bearing(shared_data)
                return self._coast_through(
                    shared_data, dt, roll, pitch, yaw, z_ned
                )
            # 080022: lost YOLO right after a large framed sighting — do not
            # hover or trust stale PnP; commit the hole for a short coast.
            # Latch once: re-extending every frame made coast immortal (0813).
            if self._last_area_px >= 40000.0 and not coasting:
                self._coast_until = now + 2.6
                self._last_area_px = 0.0
                self._latch_course_bearing(shared_data)
                return self._coast_through(
                    shared_data, dt, roll, pitch, yaw, z_ned
                )
            # 0904: after a pass last_g1 is often None; still seek on the
            # latched COURSE_BEARING instead of hovering into the floor.
            # 0928: seek expired with bearing still latched → hover → floor.
            # Keep punching, but stop before a long blind climb (attempt-2
            # peak_climb=4.0 after an empty secondary_pose hunt).
            if self._course_latched and self._pass_t is not None:
                pass_age = now - self._pass_t
                if pass_age <= 12.0:
                    self._seek_until = max(self._seek_until, now + 2.0)
            if now < self._seek_until and (
                self._last_g1 is not None or self._course_latched
            ):
                self._latch_course_bearing(shared_data)
                return self._seek_body_gate(
                    shared_data, dt, roll, pitch, yaw, z_ned, ekf
                )
            self._have_filt = False
            return self._hover(
                shared_data, yaw, z_ned, roll, pitch, 'missing_gate1_pnp'
            )

        self._last_safety = None
        g1 = self._last_g1
        range_m = float(np.linalg.norm(g1)) if g1 is not None else 8.0
        # YOLO box centre owns yaw/alt — PnP corner means spiked yaw at
        # gu≈320 in 064131 (req=-0.13 while the box was centred).
        if det_nx is not None:
            nx_raw = float(np.clip(det_nx, -1.2, 1.2))
            ny_raw = float(np.clip(det_ny or 0.0, -1.2, 1.2))
        else:
            nx_raw = float(np.clip(float(dual['gate1_norm_x']), -1.2, 1.2))
            ny_raw = float(
                np.clip(float(dual.get('gate1_norm_y') or 0.0), -1.2, 1.2)
            )

        return self._track_image(
            shared_data,
            dt,
            nx_raw,
            ny_raw,
            range_m,
            roll,
            pitch,
            yaw,
            z_ned,
            ekf,
            source='dual_pnp',
            area_px=area_px,
        )

    def _track_image(
        self,
        shared_data,
        dt,
        nx_raw,
        ny_raw,
        range_m,
        roll,
        pitch,
        yaw,
        z_ned,
        ekf,
        *,
        source: str,
        area_px: float = 0.0,
    ):
        self._last_safety = None

        # EMA — kill one-frame lock jumps that saturated yaw in 043043.
        alpha = 0.35
        if not self._have_filt:
            self._nx_f, self._ny_f = nx_raw, ny_raw
            self._have_filt = True
        else:
            # If the measurement jumps >0.55 of the frame, treat as new lock
            # and soft-reset instead of commanding a 180° slew.
            if abs(nx_raw - self._nx_f) > 0.55:
                self._nx_f = 0.6 * self._nx_f + 0.4 * nx_raw
                self._yaw_pid.reset()
            else:
                self._nx_f = (1 - alpha) * self._nx_f + alpha * nx_raw
            # Reject sudden vertical lock jumps (bottom-bar steal).
            if abs(ny_raw - self._ny_f) > 0.55:
                ny_raw = self._ny_f
            self._ny_f = (1 - alpha) * self._ny_f + alpha * ny_raw
        nx, ny = self._nx_f, self._ny_f

        # Close-range commit (044610): area ~110k / gu≈297 — nearly through.
        # Stop chasing yaw (perspective walk) and punch forward.
        commit = bool(area_px >= 8000.0 or range_m <= 3.5)
        filling = bool(area_px >= 80000.0)  # FOV filled — commit the hole

        # Non-inverted: gate-left → negative yaw_rate; controller applies
        # RATE_SIGN_YAW. 064131: cmd>0 turned further left off a left gate.
        yaw_rate = float(self._yaw_pid.update(nx, dt))
        if commit:
            # Keep a little yaw while committing so a right-offset hole
            # (nx≈+0.20–0.30) still gets centered instead of scraping left.
            if abs(nx) < 0.08:
                yaw_rate = 0.0
            else:
                yaw_rate *= 0.70
            self._yaw_pid.reset()
        # Slew-limit yaw so a single noisy frame cannot slam the stick.
        max_step = self._yaw_slew * dt
        yaw_rate = float(
            np.clip(
                yaw_rate,
                self._last_yaw_cmd - max_step,
                self._last_yaw_cmd + max_step,
            )
        )
        self._last_yaw_cmd = yaw_rate

        align = float(np.clip(1.0 - abs(nx) / 0.45, 0.0, 1.0))
        vert_align = float(np.clip(1.0 - abs(ny) / 0.65, 0.0, 1.0))

        climbed = 0.0
        if z_ned is not None and self._arm_z is not None:
            climbed = self._arm_z - z_ned

        # 064453: at the hole climbed fell below 0.3 m → fwd=0 while the
        # gate filled the frame → sank into thousands of Gate collisions.
        fwd = 0.75 + 0.25 * float(np.clip((12.0 - range_m) / 10.0, 0.0, 1.0))
        close = bool(area_px >= 20000.0 or range_m <= 2.5)
        post_pass_hunt = bool(
            self._course_latched
            and self._pass_t is not None
            and (time.monotonic() - self._pass_t) <= 14.0
        )
        # 0928: small next-gate YOLO (~0.7–2k) after pass hit climbed<0.3
        # → fwd=0 while locked on u≈385 — never closed the 3.1 m gap.
        # Only exempt post-pass yolo_fallback; first-gate yolo_fallback with
        # fwd always on scraped the rim (0 GATE_PASSED this run).
        if (
            climbed < 0.30
            and not close
            and not commit
            and not (post_pass_hunt and source == 'yolo_fallback')
        ):
            fwd = 0.0
        elif climbed > 2.0 and not close and not post_pass_hunt:
            fwd *= 0.40
        # Keep forward lean; only aim pitch at the gate once close (0748: early
        # ny-bias at v≈304 commanded max dive and missed the gate).
        if commit or close or (post_pass_hunt and source == 'yolo_fallback'):
            fwd = max(fwd, 0.90 if abs(nx) < 0.35 else 0.75)
            align = max(align, 0.70)
        des_pitch = -self._max_lean * fwd * max(align, 0.40) * max(vert_align, 0.50)
        # Only punch when the hole is roughly framed. ny>0.25 ⇒ still too high
        # on the top bar (0752 scrapes).
        framed = bool(abs(nx) < 0.28 and abs(ny) < 0.28)
        # 080022: after YOLO dropped, dual_pnp still looked "framed" and
        # punched the top bar. Require a live/large image lock to commit.
        yolo_alive = bool(area_px >= 6000.0 or filling)
        # 0849: post-pass next gate was framed at ~5–8k then we hovered.
        # 0928: next gate starts ~700–2500 px — 5000 never fired.
        now = time.monotonic()
        post_pass_commit = bool(
            now < self._seek_until
            and self._course_latched
            and area_px >= 800.0
            and abs(nx) < 0.32
            # Keep punch level; planner still tracks mid-low boxes via fallback.
            and -0.35 <= ny <= 0.35
        )
        # 0928: filling=True with cy≈290 (ny≈0.55) coast-scraped the rim
        # for 1.5s without a score — require a level frame to punch/through.
        filling_level = bool(filling and abs(float(ny)) <= 0.22)
        # Close post-pass box biased onto the lower bar: level + punch, don't dive.
        low_fill = bool(
            post_pass_hunt
            and area_px >= 25000.0
            and abs(float(nx)) <= 0.35
            and float(ny) > 0.20
        )
        punch_ready = bool(
            filling_level
            or low_fill
            or post_pass_commit
            or (yolo_alive and (commit or close) and framed)
        )
        if punch_ready:
            des_pitch = -self._max_lean * max(fwd, 0.95)
            des_pitch = float(np.clip(des_pitch, -self._max_lean, -0.05))
            if low_fill:
                # Hold altitude; box center is on the bottom bar, hole is above.
                ny = 0.0
        elif (commit or close) and ny > 0.25 and not filling:
            # Drop into the opening before committing speed.
            fwd = min(fwd, 0.65)
            des_pitch = -self._max_lean * fwd * max(align, 0.40)
        # Roll toward image error (same sign convention as yaw).
        des_roll = float(
            np.clip(-0.50 * nx * self._max_lean, -self._max_lean, self._max_lean)
        )
        if commit and abs(nx) < 0.20:
            des_roll *= 0.30
        if filling_level or low_fill:
            des_roll = 0.0
            yaw_rate = 0.0
            self._last_yaw_cmd = 0.0
        punch = punch_ready

        # Gate contact while framed large → commit the hole (0928 scrape).
        col = shared_data.get('collision') or {}
        gate_contact = False
        if col.get('id') == 1001 and post_pass_hunt and area_px >= 15000.0:
            try:
                impulse = float(col.get('impulse') or 0.0)
                ts_ns = int(col.get('ts') or 0)
                age_s = (time.time_ns() - ts_ns) * 1e-9 if ts_ns else 999.0
                gate_contact = bool(impulse >= 0.5 and 0.0 <= age_s <= 0.8)
            except (TypeError, ValueError):
                gate_contact = False
        if gate_contact:
            punch_ready = True
            punch = True
            des_pitch = -self._max_lean
            des_roll = 0.0
            yaw_rate = 0.0
            ny = 0.0
            self._coast_until = max(self._coast_until, now + 2.0)

        roll_rate = float(self._roll_pid.update(des_roll - roll, dt))
        pitch_rate = float(self._pitch_pid.update(des_pitch - pitch, dt))
        thrust, z_err, vert_src = self._thrust_for_gate(ny, climbed)
        if punch_ready:
            tilt = max(0.88, math.cos(abs(des_pitch)))
            thrust = max(thrust, config.HOVER_THRUST / tilt)
            if climbed < 0.50:
                thrust = max(thrust, config.HOVER_THRUST + 0.02)
            # 0829: filling punch climbed 0.3→1.9 m and cleared the top bar
            # without a score. Hold altitude once the hole fills the FOV.
            if (filling_level or low_fill or gate_contact) and climbed > 0.90:
                thrust = min(thrust, config.HOVER_THRUST - 0.010)
            if (filling_level or low_fill or gate_contact) and climbed > 1.40:
                thrust = min(thrust, config.HOVER_THRUST - 0.025)
            if low_fill or gate_contact:
                # Don't dump thrust into the bottom bar.
                thrust = max(thrust, config.HOVER_THRUST)
            thrust = float(np.clip(thrust, 0.210, 0.31))
            vert_src += '+punch_hold'
        elif (commit or close) and ny > 0.25 and not filling:
            thrust = min(thrust, config.HOVER_THRUST - 0.015 - 0.02 * ny)
            thrust = float(np.clip(thrust, 0.210, config.HOVER_THRUST))
            vert_src += '+drop_to_hole'
        elif commit and climbed >= 1.8 and ny > 0.05:
            thrust = min(thrust, config.HOVER_THRUST - 0.015)
            vert_src += '+commit_descend'

        phase = 'approach'
        if commit:
            phase = 'commit'
        # FOV-filled or centered large gate → coast through the hole.
        # 0928: bare filling with ny≫0 scraped; require level or gate contact.
        centered_through = bool(
            filling_level
            or low_fill
            or gate_contact
            or (
                area_px >= 22000.0
                and abs(nx) < 0.28
                and abs(ny) < 0.28
            )
        )
        if centered_through or (punch and range_m < 1.5):
            phase = 'through'
            # Latch once per sighting. If we are still FOV-filled after the
            # coast expires without a score, allow one short re-punch.
            if time.monotonic() >= self._coast_until:
                self._coast_until = time.monotonic() + (
                    3.2 if (filling_level or low_fill or gate_contact) else 2.8
                )

        shared_data['kalman_path'] = {
            'phase': phase,
            'range_m': range_m,
            'norm_x': nx,
            'norm_y': ny,
            'norm_x_raw': nx_raw,
            'align': align,
            'climbed': climbed,
            'thrust': thrust,
            'des_pitch': des_pitch,
            'gate2_fresh': bool(ekf.get('gate2_fresh', False)),
            'vert_src': vert_src,
            'z_err': z_err,
            'source': source,
        }

        return {
            'vn': float(0.35 * max(align, 0.2) * fwd),
            've': float(-0.12 * nx),
            'vd': float(np.clip(z_err * 0.18, -0.20, 0.20)),
            'yaw_rate': yaw_rate,
            'kalman': True,
            'roll_rate': roll_rate,
            'pitch_rate': pitch_rate,
            'thrust': thrust,
            'desired_roll': des_roll,
            'desired_pitch': des_pitch,
            'desired_yaw': yaw + nx * 0.5,
        }

    def _latch_course_bearing(self, shared_data):
        if self._course_latched:
            return
        bearing = shared_data.get('course_bearing') or {}
        try:
            self._course_nx = float(bearing['nx'])
            # 0753: v=+0.41 after a good pass → sank. Cap look-down.
            self._course_ny = float(np.clip(float(bearing['ny']), -0.20, 0.05))
            try:
                self._course_range = float(bearing.get('range_m'))
            except (TypeError, ValueError):
                self._course_range = None
            self._course_latched = True
        except (KeyError, TypeError, ValueError):
            return

    def _maybe_retarget_live_bearing(self, shared_data, now: float) -> None:
        """Prefer a nearer live primary/contact over a stale post-pass latch."""
        bearings = shared_data.get('gate_bearings') or {}
        candidates = []
        for key in ('primary', 'contact'):
            item = bearings.get(key) or {}
            try:
                nx = float(item['horizontal_normalized'])
                ny = float(item.get('vertical_normalized', 0.0))
                range_m = float(item['range_m'])
            except (KeyError, TypeError, ValueError):
                continue
            if 1.2 <= range_m <= 4.0 and abs(nx) <= 0.70:
                candidates.append((range_m, nx, ny))
        bearing = shared_data.get('course_bearing') or {}
        try:
            candidates.append(
                (
                    float(bearing['range_m']),
                    float(bearing['nx']),
                    float(bearing.get('ny', 0.0)),
                )
            )
        except (KeyError, TypeError, ValueError):
            pass
        if not candidates:
            return
        candidates.sort(key=lambda item: item[0])
        range_m, nx, ny = candidates[0]
        latched_r = (
            float(self._course_range)
            if self._course_range is not None
            else 99.0
        )
        latched_nx = (
            float(self._course_nx) if self._course_nx is not None else 0.0
        )
        nearer = range_m + 0.8 < latched_r
        # 0928: latched h=+0.10 @ 3.1m with YOLO on u≈385–398, then
        # contact_h=-0.62 flipped aim and scraped — never opposite-side
        # retarget while the latched bearing is nearly straight / same-range.
        same_side = (nx * latched_nx >= 0.0) or abs(latched_nx) < 0.12
        side_flip = (
            same_side
            and range_m <= 3.5
            and abs(nx - latched_nx) > 0.25
            and range_m <= latched_r + 0.5
        )
        # Hold a good near-straight latch; only retarget if clearly nearer
        # on the same side.
        if self._course_latched and abs(latched_nx) < 0.18:
            if not (nearer and same_side and range_m + 1.2 < latched_r):
                return
        if not (nearer or side_flip or not self._course_latched):
            return
        self._course_nx = float(np.clip(nx, -0.45, 0.45))
        self._course_ny = float(np.clip(ny, -0.20, 0.05))
        self._course_range = float(range_m)
        self._course_latched = True
        # Refresh shared bearing so YOLO prefer_nx matches the retarget.
        shared_data['course_bearing'] = {
            'nx': self._course_nx,
            'ny': self._course_ny,
            'range_m': self._course_range,
            'source': 'live_retarget',
            'ts': now,
        }

    def _body_aim_nx_ny(self):
        # Prefer latched multi-gate course bearing (seen before the hole).
        # 0904: pass sets course_nx=0 placeholder — only use when latched.
        if self._course_latched:
            aim = self._last_g2 if self._last_g2 is not None else self._last_g1
            if aim is not None and float(np.linalg.norm(aim)) > 8.0:
                aim = None
            # 0919: look-down bearings (v=+0.37) sank seek into the floor;
            # keep post-pass body aim nearly level.
            return (
                float(np.clip(self._course_nx, -0.85, 0.85)),
                float(np.clip(self._course_ny, -0.15, 0.05)),
                aim,
            )
        aim = self._last_g2 if self._last_g2 is not None else self._last_g1
        if aim is None:
            return 0.0, 0.0, None
        if float(np.linalg.norm(aim)) > 8.0:
            # Far body fix is not the next gate — fly level until YOLO.
            return 0.0, 0.0, None
        fwd = max(0.4, float(aim[0]))
        nx = float(np.clip(float(aim[1]) / fwd, -0.70, 0.70))
        ny = float(np.clip(float(aim[2]) / fwd, -0.70, 0.70))
        return nx, ny, aim

    def _coast_through(self, shared_data, dt, roll, pitch, yaw, z_ned):
        """Punch forward after losing the gate; ease yaw toward last gate2."""
        # 0928: h=+0.06 @ 4.8m still died after 12s seek — commit full lean
        # on near-straight bearings.
        if self._course_latched and abs(float(self._course_nx)) < 0.12:
            fwd = 1.0
        elif self._course_latched and abs(float(self._course_nx)) < 0.22:
            fwd = 0.95
        elif self._course_latched:
            fwd = 0.75
        else:
            fwd = 0.95
        return self._body_flight(
            shared_data,
            dt,
            roll,
            pitch,
            yaw,
            z_ned,
            phase='coast',
            source='coast',
            fwd_frac=fwd,
        )

    def _seek_body_gate(
        self, shared_data, dt, roll, pitch, yaw, z_ned, ekf
    ):
        """Cruise toward last body fix while YOLO reacquires after a pass."""
        if self._course_latched and abs(float(self._course_nx)) < 0.12:
            fwd = 1.0
        elif self._course_latched and abs(float(self._course_nx)) < 0.22:
            fwd = 0.90
        elif self._course_latched:
            # Forward pressure, but 0.85 + yaw into +0.28 hit Environment
            # while a small right YOLO was still visible (0930 attempt-2).
            fwd = 0.75
        else:
            fwd = 0.50
        return self._body_flight(
            shared_data,
            dt,
            roll,
            pitch,
            yaw,
            z_ned,
            phase='seek',
            source='body_seek',
            fwd_frac=fwd,
        )

    def _body_flight(
        self,
        shared_data,
        dt,
        roll,
        pitch,
        yaw,
        z_ned,
        *,
        phase: str,
        source: str,
        fwd_frac: float,
    ):
        nx, ny, aim = self._body_aim_nx_ny()
        # 0833: with COURSE_BEARING NONE, level seek never found gate 2.
        # Gentle right/up sweep matches the usual next-gate offset.
        if (
            phase == 'seek'
            and not self._course_latched
            and abs(float(nx)) < 1e-3
            and abs(float(ny)) < 1e-3
        ):
            # Default hunt: bias right for typical gate-2 offset.
            nx, ny = 0.28, -0.06
        des_pitch = -self._max_lean * fwd_frac
        des_roll = float(
            np.clip(-0.55 * nx * self._max_lean, -self._max_lean, self._max_lean)
        )
        yaw_rate = float(self._yaw_pid.update(nx, dt))
        # 0919/0924: soften yaw; nearly-straight bearings should barely turn.
        if phase in ('seek', 'coast'):
            # Ease yaw only while clearing the hole; then track hard — true
            # gate-2 bearing was +0.77 and YOLO sat at u≈504.
            if phase == 'coast':
                yaw_rate *= 0.45
            elif self._course_latched and abs(float(self._course_nx)) >= 0.35:
                # Hard yaw at ±0.5+ scraped Environment (0930 secondary +0.54).
                yaw_rate *= 0.70
            elif self._course_latched and abs(float(self._course_nx)) >= 0.25:
                yaw_rate *= 0.85
            else:
                yaw_rate *= 0.80
        # More right/left lean when the next gate is clearly off-center.
        if phase in ('seek', 'coast') and abs(float(nx)) >= 0.20:
            lean_scale = 0.55 if abs(float(nx)) >= 0.35 else 0.70
            des_roll = float(
                np.clip(
                    -lean_scale * nx * self._max_lean,
                    -self._max_lean,
                    self._max_lean,
                )
            )
        max_step = self._yaw_slew * dt
        yaw_rate = float(
            np.clip(
                yaw_rate,
                self._last_yaw_cmd - max_step,
                self._last_yaw_cmd + max_step,
            )
        )
        self._last_yaw_cmd = yaw_rate
        roll_rate = float(self._roll_pid.update(des_roll - roll, dt))
        pitch_rate = float(self._pitch_pid.update(des_pitch - pitch, dt))
        tilt = max(0.88, math.cos(abs(des_pitch)))
        climbed = (
            (self._arm_z - z_ned)
            if z_ned is not None and self._arm_z is not None
            else 0.0
        )
        thrust = float(
            np.clip(
                (config.HOVER_THRUST + 0.01 - 0.012 * ny) / tilt,
                config.HOVER_THRUST - 0.01,
                0.31,
            )
        )
        if phase in ('seek', 'coast'):
            # Prefer altitude hold over speed while hunting the next gate.
            thrust = max(thrust, config.HOVER_THRUST / max(tilt, 0.90))
            # 0906: seek deaths at peak_climb≈1.0 — hold harder while low.
            if climbed < 0.80:
                thrust = max(thrust, config.HOVER_THRUST + 0.025)
            if climbed < 0.40:
                thrust = max(thrust, config.HOVER_THRUST + 0.035)
        # 0812/0859: hard thrust dumps after climb→2 m caused freefall. Soft
        # brake only; keep a little forward lean so we don't hover-stall.
        if climbed > 1.1:
            thrust = min(thrust, config.HOVER_THRUST)
            des_pitch = max(des_pitch, -self._max_lean * 0.70)
        if climbed > 1.6:
            thrust = min(thrust, config.HOVER_THRUST - 0.010)
            des_pitch = max(des_pitch, -self._max_lean * 0.55)
        if climbed > 2.2:
            thrust = min(thrust, config.HOVER_THRUST - 0.020)
            des_pitch = max(des_pitch, -self._max_lean * 0.40)
        # 0928: seek runaway to peak_climb=4–15 m after losing the next gate.
        if phase in ('seek', 'coast') and climbed > 2.5:
            thrust = min(thrust, config.HOVER_THRUST - 0.035)
            des_pitch = max(des_pitch, -self._max_lean * 0.25)
            yaw_rate *= 0.3
        if phase in ('seek', 'coast') and climbed > 3.2:
            thrust = min(thrust, config.HOVER_THRUST - 0.050)
            des_pitch = -0.02
            yaw_rate = 0.0
        shared_data['kalman_path'] = {
            'phase': phase,
            'range_m': (
                float(np.linalg.norm(aim)) if aim is not None else 0.5
            ),
            'norm_x': nx,
            'norm_y': ny,
            'align': 1.0,
            'climbed': climbed,
            'thrust': thrust,
            'des_pitch': des_pitch,
            'source': source,
        }
        return {
            'vn': 0.35 if phase == 'seek' else 0.4,
            've': float(-0.10 * nx),
            'vd': float(np.clip(ny * 0.12, -0.15, 0.15)),
            'yaw_rate': yaw_rate,
            'kalman': True,
            'roll_rate': roll_rate,
            'pitch_rate': pitch_rate,
            'thrust': thrust,
            'desired_roll': des_roll,
            'desired_pitch': des_pitch,
            'desired_yaw': yaw + nx * 0.4,
        }

    def _thrust_for_gate(self, ny: float, climbed: float):
        hover = config.HOVER_THRUST
        z_err = ny
        # Image: ny>0 means gate below centre → descend a touch; ny<0 climb.
        # 065915 approached with v≈240–260 (gate low in frame) — bias a bit
        # stronger so we punch the hole, not the top bar.
        thrust = hover - 0.030 * ny
        thrust = float(np.clip(thrust, hover - 0.035, hover + 0.015))
        vert_src = 'image_ny'
        # Hold a low cruise altitude; don't dump thrust to freefall (063921).
        if climbed > 0.8:
            thrust = min(thrust, hover - 0.010)
            vert_src += '+alt_brake'
        if climbed > 1.5:
            thrust = min(thrust, hover - 0.025)
            vert_src += '+alt_hard'
        if climbed > 2.5:
            thrust = min(thrust, hover - 0.045)
            vert_src += '+alt_max'
        if climbed > 4.0:
            thrust = min(thrust, hover - 0.070)
            vert_src += '+alt_emergency'
        return thrust, z_err, vert_src

    def _hover(self, shared_data, yaw: float, z_ned, roll: float, pitch: float, reason: str):
        if reason != self._last_safety:
            log = shared_data.get('log_event')
            if log:
                log('PLANNER_SAFETY', reason)
            self._last_safety = reason
        self._yaw_pid.reset()
        self._last_yaw_cmd = 0.0
        # Rate commands of zero HOLD the last attitude in this sim — actively
        # regulate back to level (044425 froze at pitch=-0.26 in SEARCH).
        dt = 0.02
        des_roll = 0.0
        des_pitch = 0.0
        roll_rate = float(self._roll_pid.update(des_roll - roll, dt))
        pitch_rate = float(self._pitch_pid.update(des_pitch - pitch, dt))
        hover = config.HOVER_THRUST
        thrust = hover
        climbed = 0.0
        if z_ned is not None and self._arm_z is not None:
            climbed = self._arm_z - z_ned
            if climbed > 0.6:
                thrust = hover - 0.015
            if climbed > 1.5:
                thrust = hover - 0.028
        # 0825: stale phase=coast after hover suppressed crash resets forever.
        shared_data['kalman_path'] = {
            'phase': 'hover',
            'range_m': None,
            'source': 'hover',
            'climbed': climbed,
            'thrust': float(thrust),
            'reason': reason,
        }
        return {
            'vn': 0.0,
            've': 0.0,
            'vd': 0.0,
            'yaw_rate': 0.0,
            'kalman': True,
            'roll_rate': roll_rate,
            'pitch_rate': pitch_rate,
            'thrust': float(thrust),
            'desired_roll': des_roll,
            'desired_pitch': des_pitch,
            'desired_yaw': yaw,
        }


GateChaserPlanner = KalmanDualGatePlanner
