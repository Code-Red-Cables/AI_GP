import os
import socket
import struct
import threading
import time

import cv2
import numpy as np

from vision.gate_detector import detect_gate, draw_detection

SIM_SERVER_UDP_IP   = '0.0.0.0'
SIM_SERVER_UDP_PORT = 5600

DEBUG_DIR      = '_vision_debug'
DEBUG_INTERVAL = 5.0   # save an annotated debug frame at most once per N seconds


class VisionRX:

    def __init__(self, data):
        self.data = data
        self._last_debug_t = 0.0
        os.makedirs(DEBUG_DIR, exist_ok=True)
        self.thread = threading.Thread(target=self._vision_loop, daemon=False)
        self.is_running = True
        self.thread.start()

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    # ------------------------------------------------------------------
    def _vision_loop(self):
        header_fmt = '<IHHIIQ'
        header_sz  = struct.calcsize(header_fmt)
        frames     = {}

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT))
        sock.settimeout(1.0)
        print('[VISION] listening for camera frames...', flush=True)

        while self.is_running:
            try:
                packet, _ = sock.recvfrom(65536)
            except socket.timeout:
                continue

            header  = packet[:header_sz]
            payload = packet[header_sz:]
            frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns = \
                struct.unpack(header_fmt, header)

            if frame_id not in frames:
                frames[frame_id] = {'chunks': {}, 'total': total_chunks}
            frames[frame_id]['chunks'][chunk_id] = payload

            if len(frames[frame_id]['chunks']) == total_chunks:
                jpeg = bytearray()
                ok   = True
                for i in range(total_chunks):
                    if i not in frames[frame_id]['chunks']:
                        ok = False
                        break
                    jpeg.extend(frames[frame_id]['chunks'][i])

                if ok:
                    arr   = np.frombuffer(jpeg, dtype=np.uint8)
                    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if image is not None:
                        self.process_frame(frame_id, image)

                del frames[frame_id]

            # Prune stale incomplete frames
            if len(frames) > 30:
                del frames[min(frames)]

    # ------------------------------------------------------------------
    def process_frame(self, frame_id: int, img: np.ndarray):
        det = detect_gate(img)

        if det is not None:
            self.data['gate_detection'] = {
                'center_px':  det.center_px,
                'corners_px': det.corners_px,
                'bbox_px':    det.bbox_px,
                'area_px':    det.area_px,
                'confidence': det.confidence,
                'frame_id':   frame_id,
                'ts':         time.time_ns(),
            }
        else:
            self.data['gate_detection'] = None

        # Periodic debug image
        now = time.time()
        if (now - self._last_debug_t) >= DEBUG_INTERVAL:
            self._last_debug_t = now
            annotated = draw_detection(img, det)
            cv2.imwrite(os.path.join(DEBUG_DIR, f'frame_{frame_id:06d}.jpg'), annotated)
