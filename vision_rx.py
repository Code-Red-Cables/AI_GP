import math
import socket
import struct
import threading
import time

import cv2
import numpy as np

from vision.gate_detector import OrangeGateDetector, draw_detection
from vision.gate_tracker import GateTracker
from vision.navigation import GateNavigator
from vision.mode_router import (
    GateNavigationMode,
    ModeRouterConfig,
    VisionModeRouter,
)
from vision.ai_adapter import validate_ai_action
from gate_estimator import estimate_gate
import camera_model as cm


def _quat_to_rpy(q):
    """Quaternion (w, x, y, z) -> (roll, pitch, yaw) dict in radians (NED ZYX)."""
    w, x, y, z = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return {"roll": roll, "pitch": pitch, "yaw": yaw}

# Modify these properties if you want to run the server remotely for example
SIM_SERVER_UDP_IP = "0.0.0.0"
SIM_SERVER_UDP_PORT = 5600

# --------------------------------------------------------------------------- #
# Vision estimate filtering. The range estimate is bimodal in quality (logs
# 2026-06-05): the PnP path (clean 4-corner quad) is tight (~19-22m on the same
# gate) while the size-method fallback (Z = f*1.5/sqrt(area)) is garbage (11-64m)
# because the inner-opening area breaks up frame-to-frame. Unfiltered, every bad
# frame reached the planner and yanked guidance/yaw around. We reject teleporting
# detections and low-pass the accepted gate_body so range AND bearing are stable.
# --------------------------------------------------------------------------- #
VIS_MAX_RANGE_M = 40.0             # gates sit within ~25m; beyond is a bad estimate
VIS_JUMP_MAX_M = 8.0               # reject a range that jumps this far from the belief
VIS_BELIEF_TIMEOUT_NS = 500_000_000  # belief older than this is stale -> reseed
VIS_EMA_ALPHA = 0.5                # gate_body low-pass (1.0 = no smoothing)

# PnP target continuity (2026-07-27 run: the published "best" gate alternated
# between the 8.6 m gate ahead and 22-39 m background gates frame to frame; the
# reactive planner re-aimed at each in turn and yawed 70 deg off the course in
# 1 s). Once we are flying at a gate, only detections that MATCH it (similar
# range + azimuth) may steer; mismatches are dropped. If the target vanishes
# for PNP_TARGET_MEMORY_S (passed it / left the frame) we adopt the best fresh
# detection again.
PNP_TARGET_MEMORY_S = 1.5
PNP_TARGET_MATCH_COST = 0.55       # ~30% range diff + ~0.25 rad azimuth change


class VisionRX:

    def __init__(self, data):
        self.data = data
        if 'lock' not in self.data:
            self.data['lock'] = threading.RLock()
        # Temporal filter belief: last accepted (smoothed) gate_body + its timestamp.
        self._gb_filt = None
        self._gb_ts = 0
        self.detector = OrangeGateDetector()
        self.tracker = GateTracker()
        self.navigator = GateNavigator()
        requested_mode = str(
            self.data.get('gate_navigation_mode', 'opencv')
        ).lower()
        # 'pnp' mode: YOLO corner detection + solvePnP is the perception source
        # (feeds the VIO state estimator AND the planner's world-waypoint path).
        # It bypasses the HSV detector and the opencv/ai mode router entirely.
        self.pnp_mode = requested_mode == 'pnp'
        self.pnp_detector = None
        self._pnp_last = None      # {'ts','range','az'} of the gate we are flying at
        self._pnp_stats = [0, 0, 0]   # frames, frames-with-yolo-boxes, published
        self._pnp_stats_t = time.monotonic()
        if self.pnp_mode:
            from vision.yolo_pnp import YoloGatePnP
            print("Loading YOLO gate model...", flush=True)
            self.pnp_detector = YoloGatePnP()
            print("YOLO gate model ready.", flush=True)
        self.mode_router = VisionModeRouter(
            ModeRouterConfig(mode=GateNavigationMode(
                'opencv' if self.pnp_mode else requested_mode))
        )
        self.thread = threading.Thread(
            target=self._vision_loop,
            daemon=False
        )
        self.is_running = True
        self.thread.start()

    def _filter_estimate(self, est, now):
        """Reject teleporting range outliers and EMA-smooth gate_body in place.

        Drops a frame (marks detected=False) whose range is absurd or jumps more
        than VIS_JUMP_MAX_M from the recent belief, so the noisy size-method spikes
        never reach the planner. Accepted frames are low-passed for stability.
        """
        if not est.get('detected') or est.get('gate_body') is None:
            return est
        gb = np.asarray(est['gate_body'], float)
        rng = float(np.linalg.norm(gb))
        if not (0.0 < rng <= VIS_MAX_RANGE_M):
            est['detected'] = False
            est['reject'] = 'range'
            return est
        fresh = self._gb_filt is not None and (now - self._gb_ts) <= VIS_BELIEF_TIMEOUT_NS
        if fresh and abs(rng - float(np.linalg.norm(self._gb_filt))) > VIS_JUMP_MAX_M:
            est['detected'] = False
            est['reject'] = 'jump'
            return est
        gb_f = (VIS_EMA_ALPHA * gb + (1.0 - VIS_EMA_ALPHA) * self._gb_filt
                if fresh else gb)
        self._gb_filt = gb_f
        self._gb_ts = now
        # Republish the smoothed body vector and its derived range/bearing.
        x, y, z = float(gb_f[0]), float(gb_f[1]), float(gb_f[2])
        est['gate_body'] = (x, y, z)
        est['range_m'] = float(np.linalg.norm(gb_f))
        est['bearing'] = (math.atan2(y, x), math.atan2(-z, math.hypot(x, y)))
        return est

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    def _vision_loop(self):
        header_format = "<IHHIIQ"
        header_sz = struct.calcsize(header_format)
        frames = {}  # frame_id -> received associated frame data

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT))
        sock.settimeout(0.2)
        print("Listening for camera frames...")

        while self.is_running:
            try:
                packet, addr = sock.recvfrom(65536)  # max UDP size
            except socket.timeout:
                continue

            header = packet[:header_sz]
            payload = packet[header_sz:]

            # frame_id - identifier for this vision frame
            # chunk_id - identifier for this chunk packet of data of this frame
            # total_chunks - total number of chunk packets that make up this frame
            # jpeg_size - full size of jpeg data
            # payload_size - size of this packet
            # sim_time_ns - frame's epoch timestamp in ns on the server
            frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns = struct.unpack(header_format, header)

            if frame_id not in frames:
                frames[frame_id] = {
                    "chunks": {},
                    "total": total_chunks,
                    "size": jpeg_size,
                    "time": sim_time_ns
                }

            frames[frame_id]["chunks"][chunk_id] = payload

            # Check if frame is complete
            if len(frames[frame_id]["chunks"]) == total_chunks:
                jpeg_bytes = bytearray()

                frame_complete = True
                for i in range(total_chunks):
                    if i not in frames[frame_id]["chunks"]:
                        print('Missing packet %s in frame %s' % (i, frame_id,))
                        frame_complete = False
                        continue
                    jpeg_bytes.extend(frames[frame_id]["chunks"][i])

                if not frame_complete:
                    del frames[frame_id]
                    continue

                img_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if image is not None:
                    self.process_frame(frame_id, image, frames[frame_id]["time"])
                else:
                    print(f"Failed to decode frame: {frame_id}")

                del frames[frame_id]
                # The controller needs the newest causal image, not an unbounded
                # queue of stale partial frames.
                for stale_id in [key for key in frames if key < frame_id - 2]:
                    del frames[stale_id]
        sock.close()

    def _pick_pnp_gate(self, gates, now):
        """Continuity-aware target choice: stick with the gate we're flying at.

        Ranks solved candidates against the last published target by range +
        azimuth similarity. A frame whose best match is still too different is
        DROPPED (returns None) — the planner coasts on its latched heading —
        until the memory window lapses and we adopt the best fresh detection.
        """
        solved = [g for g in gates if g.solved]
        if not solved:
            return None
        last = self._pnp_last
        if last is not None and (now - last['ts']) < PNP_TARGET_MEMORY_S * 1e9:
            def cost(g):
                gb = g.center_body()
                az = math.atan2(float(gb[1]), float(gb[0]))
                d_rng = abs(g.range_m - last['range']) / max(last['range'], 2.0)
                d_az = abs(math.atan2(math.sin(az - last['az']),
                                      math.cos(az - last['az'])))
                return d_rng + d_az
            cand = min(solved, key=cost)
            return cand if cost(cand) < PNP_TARGET_MATCH_COST else None
        return solved[0]

    def _process_frame_pnp(self, frame_id, img, sim_time_ns=None):
        """YOLO corners -> PnP pose. Publishes shared_data['vision'] (for the
        planner's world-waypoint path) and shared_data['pnp_fix'] (for the VIO
        state estimator). Perception only — no flight commands here."""
        t0 = time.perf_counter()
        gates = self.pnp_detector.detect(img)
        now = time.time_ns()
        best = self._pick_pnp_gate(gates, now)
        self._pnp_stats[0] += 1
        self._pnp_stats[1] += bool(gates)
        self._pnp_stats[2] += best is not None
        if time.monotonic() - self._pnp_stats_t >= 5.0:
            n, raw, pub = self._pnp_stats
            print(f"[vis] frames={n} yolo_hits={raw} published={pub}"
                  f" (last 5s)", flush=True)
            self._pnp_stats = [0, 0, 0]
            self._pnp_stats_t = time.monotonic()
        est = {'ts': now, 'detected': False, 'confidence': 0.0,
               'frame_id': frame_id, 'sim_time_ns': sim_time_ns,
               'n_gates': len(gates)}
        if best is not None:
            gb = best.center_body()
            x, y, z = (float(v) for v in gb)
            self._pnp_last = {'ts': now, 'range': best.range_m,
                              'az': math.atan2(y, x)}
            est.update({
                'detected': True,
                'confidence': best.confidence,
                'gate_body': (x, y, z),
                'range_m': best.range_m,
                'bearing': (math.atan2(y, x), math.atan2(-z, math.hypot(x, y))),
                'corners_px': best.corners_px.tolist(),
                'reproj_err_px': best.reproj_err_px,
            })
        with self.data['lock']:
            att = self.data.get('attitude')
            pos = self.data.get('position_ned')
            debug = self.data.get('debug_vision', False)
        if best is not None and att is not None:
            offset_ned = cm.body_to_ned(
                np.asarray(est['gate_body'], float),
                att['roll'], att['pitch'], att['yaw'])
            p = (pos['x'], pos['y'], pos['z']) if pos is not None else (0.0, 0.0, 0.0)
            est['gate_ned'] = tuple(float(a + b) for a, b in zip(p, offset_ned))
        est['vision_total_time_ms'] = (time.perf_counter() - t0) * 1000.0
        with self.data['lock']:
            self.data['vision'] = est
            self.data['control_source'] = 'pnp'
            if best is not None:
                self.data['pnp_fix'] = {
                    'ts': now,
                    'R_cg': best.R_cg.tolist(),
                    't_cg': best.t_cg.tolist(),
                    'reproj_err_px': best.reproj_err_px,
                    'range_m': best.range_m,
                }
        if debug:
            overlay = img.copy()
            for g in gates:
                color = (0, 255, 0) if g.solved else (0, 128, 255)
                for (u, v) in g.corners_px.astype(int):
                    cv2.circle(overlay, (u, v), 3, color, -1)
                if g.solved:
                    x1, y1 = int(g.bbox[0]), int(g.bbox[1])
                    cv2.putText(overlay, f"{g.range_m:.1f}m", (x1, max(12, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.imwrite(f"_vision_{frame_id % 20:02d}.png", overlay)

    def process_frame(self, frame_id, img, sim_time_ns=None):
        """Detect the gate, estimate its pose, and publish to shared_data['vision'].

        Perception only — no flight commands here. Detection runs every frame; the
        gate pose is back-projected with the latest telemetry attitude/position so
        the planner can fuse it. Returns None on no detection (still publishes a
        'detected': False record so downstream code can detect staleness).
        """
        if self.pnp_mode:
            return self._process_frame_pnp(frame_id, img, sim_time_ns)
        frame_time = time.monotonic()
        frame_started = time.perf_counter()
        hint = self.tracker.hint(frame_time)
        raw_detection = self.detector.detect(
            img, hint=hint, timestamp=frame_time
        )
        det = self.tracker.update(raw_detection, timestamp=frame_time)
        command = self.navigator.update(det, frame_time)
        navigation_done = time.perf_counter()

        # Snapshot the latest pose for the camera->NED transform.
        with self.data['lock']:
            att = self.data.get('attitude')
            odo = self.data.get('odometry')
            pos = self.data.get('position_ned')
            debug = self.data.get('debug_vision', False)
            ai_policy = self.data.get('ai_policy')
            context = {
                'attitude': att,
                'odometry': odo,
                'position_ned': pos,
                'imu': self.data.get('imu'),
            }

        source = self.mode_router.update(
            det.confidence if det is not None and det.found else 0.0,
            ai_available=ai_policy is not None,
        )
        ai_action = None
        ai_error = None
        if source == 'ai' and ai_policy is not None:
            try:
                predictor = getattr(ai_policy, 'predict', ai_policy)
                ai_action = validate_ai_action(predictor(img, context))
            except Exception as exc:
                ai_error = f"{type(exc).__name__}: {exc}"
                source = 'safe'

        attitude = None
        if att is not None:
            attitude = {k: att[k] for k in ('roll', 'pitch', 'yaw')}
        elif odo is not None:
            attitude = _quat_to_rpy(odo['q'])

        position = None
        if odo is not None:
            position = odo['pos']
        elif pos is not None:
            position = (pos['x'], pos['y'], pos['z'])

        now = time.time_ns()
        est = estimate_gate(det, attitude=attitude, position_ned=position, ts=now)
        # Temporal outlier-reject + smoothing on gate_body (kills size-method spikes).
        est = self._filter_estimate(est, now)
        # Recompute the absolute NED gate position from the SMOOTHED body vector so
        # gate_ned stays consistent with the filtered estimate the planner uses.
        if est.get('detected') and attitude is not None and est.get('gate_body'):
            import camera_model as cm
            offset_ned = cm.body_to_ned(
                np.asarray(est['gate_body'], float),
                attitude['roll'], attitude['pitch'], attitude['yaw'])
            if position is not None:
                est['gate_ned'] = tuple(float(a + b) for a, b in zip(position, offset_ned))
            else:
                est['gate_ned'] = tuple(float(c) for c in offset_ned)
        est['frame_id'] = frame_id
        est['sim_time_ns'] = sim_time_ns
        est['raw_confidence'] = raw_detection.confidence
        est['predicted'] = bool(det.predicted) if det is not None else False
        est['detector_timings_ms'] = (
            dict(self.detector.last_debug.timings_ms)
            if self.detector.last_debug else {}
        )
        est['tracker_time_ms'] = self.tracker.last_update_ms
        est['detection_tracking_navigation_time_ms'] = (
            navigation_done - frame_started
        ) * 1000.0
        est['vision_total_time_ms'] = (
            time.perf_counter() - frame_started
        ) * 1000.0

        with self.data['lock']:
            self.data['vision'] = est
            self.data['navigation'] = {
                'ts': now,
                'forward_mps': command.forward_mps,
                'right_mps': command.right_mps,
                'down_mps': command.down_mps,
                'yaw_rate_rps': command.yaw_rate_rps,
                'state': command.state.value,
                'confidence': command.confidence,
                'predicted': command.predicted,
            }
            self.data['control_source'] = source
            self.data['ai_action'] = (
                dict(ai_action, ts=now) if ai_action is not None else None
            )
            self.data['ai_error'] = ai_error

        if debug:
            overlay = draw_detection(
                img,
                det,
                self.detector.last_debug,
                state=command.state.value,
                command=command,
                raw_detection=raw_detection,
                total_time_ms=est['vision_total_time_ms'],
            )
            cv2.imwrite(f"_vision_{frame_id % 20:02d}.png", overlay)
