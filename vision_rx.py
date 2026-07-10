import math
import socket
import struct
import threading
import time

import cv2
import numpy as np

from vision.gate_detector import detect_gate, draw_detection
from vision.gate_estimator import estimate_gate


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


class VisionRX:

    def __init__(self, data):
        self.data = data
        if 'lock' not in self.data:
            self.data['lock'] = threading.RLock()
        # Temporal filter belief: last accepted (smoothed) gate_body + its timestamp.
        self._gb_filt = None
        self._gb_ts = 0
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
        print("Listening for camera frames...")

        while self.is_running:
            packet, addr = sock.recvfrom(65536)  # max UDP size

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

    def process_frame(self, frame_id, img, sim_time_ns=None):
        """Detect the gate, estimate its pose, and publish to shared_data['vision'].

        Perception only — no flight commands here. Detection runs every frame; the
        gate pose is back-projected with the latest telemetry attitude/position so
        the planner can fuse it. Returns None on no detection (still publishes a
        'detected': False record so downstream code can detect staleness).
        """
        det = detect_gate(img)

        # Snapshot the latest pose for the camera->NED transform.
        with self.data['lock']:
            att = self.data.get('attitude')
            odo = self.data.get('odometry')
            pos = self.data.get('position_ned')
            debug = self.data.get('debug_vision', False)

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

        with self.data['lock']:
            self.data['vision'] = est

        if debug:
            try:
                cv2.imwrite(f"_vision_{frame_id % 20:02d}.png",
                            draw_detection(img, det))
            except Exception:
                pass